//
// Created by haoRan on 2026/4/3.
//
#ifndef ROCKSDB_V4_LITE_RIDGE_ADMISSION_H
#define ROCKSDB_V4_LITE_RIDGE_ADMISSION_H
#pragma once

#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

struct V4LiteRidgeFeatures {
  double block_size = 4080.0;
  double num_keys_in_block = 24.0;
  double recent_get_key_freq_hist = 3.0;
  double last_get_key_gap = 18.0;  // cold start -> median
  double recent_sst_freq_hist = 636.0;
  double recent_sst_miss_freq_hist = 109.0;
  double same_get_num_blocks = 4.0;
  double same_get_num_data_blocks = 2.0;
  double is_l0 = 0.0;
  double offset_bucket = 0.0;
  double keys_density = 0.0058823529411764;
};

inline double V4LiteRidgePredict(V4LiteRidgeFeatures f) {
  if (!std::isfinite(f.num_keys_in_block)) {
    f.num_keys_in_block = 24.0;
  }
  if (!std::isfinite(f.last_get_key_gap) || f.last_get_key_gap < 0.0) {
    f.last_get_key_gap = 18.0;
  }
  if (!std::isfinite(f.keys_density)) {
    f.keys_density = 0.0058823529411764;
  }

  // Flattened from ridge_final_model.joblib
  return 3.455541675149011
       + (-0.00048591834334812636) * f.block_size
       + ( 0.01901845525246172   ) * f.num_keys_in_block
       + (-0.20683552907219735   ) * f.recent_get_key_freq_hist
       + (-8.337345116200667e-05 ) * f.last_get_key_gap
       + ( 0.0008269698778882133 ) * f.recent_sst_freq_hist
       + ( 0.00019569651762663565) * f.recent_sst_miss_freq_hist
       + ( 0.23646211224877692   ) * f.same_get_num_blocks
       + (-0.0512844175616774    ) * f.same_get_num_data_blocks
       + ( 0.11765387164529403   ) * f.is_l0
       + ( 0.0                   ) * f.offset_bucket
       + (-72.78407721164811     ) * f.keys_density;
}

inline bool V4LiteRidgeShouldAdmit(const V4LiteRidgeFeatures& f,
                                   double threshold) {
  return V4LiteRidgePredict(f) >= threshold;
}

struct V4LitePerGetState {
  uint32_t num_blocks_seen = 0;
  uint32_t num_data_blocks_seen = 0;
  uint64_t last_seen_seq = 0;
};

class V4LiteRidgeAdmissionRuntime {
 public:
  V4LiteRidgeAdmissionRuntime(uint32_t history_window, double threshold)
      : history_window_(history_window), threshold_(threshold) {}

  bool ShouldAdmit(uint64_t sst_id,
                   int level,
                   uint64_t block_offset,
                   uint64_t block_size,
                   const Slice& user_key,
                   uint64_t tracing_get_id,
                   V4LiteRidgeFeatures* out_features = nullptr) {
    std::lock_guard<std::mutex> lg(mu_);
    ++seq_no_;

    if ((seq_no_ & 4095ULL) == 0) {
      CleanupStale();
    }

    V4LiteRidgeFeatures f;
    f.block_size = static_cast<double>(block_size);
    f.is_l0 = (level == 0) ? 1.0 : 0.0;
    f.offset_bucket = static_cast<double>(block_offset >> 12);

    // 第一版先用离线训练的中位数兜底
    f.num_keys_in_block = 24.0;
    f.keys_density = 0.0058823529411764;

    const uint64_t key_hash = HashKey(user_key);

    auto& key_q = key_hist_[key_hash];
    auto& sst_q = sst_hist_[sst_id];
    auto& sst_miss_q = sst_miss_hist_[sst_id];

    PruneQueue(key_q);
    PruneQueue(sst_q);
    PruneQueue(sst_miss_q);

    f.recent_get_key_freq_hist = static_cast<double>(key_q.size());
    if (!key_q.empty()) {
      f.last_get_key_gap = static_cast<double>(seq_no_ - key_q.back());
    } else {
      f.last_get_key_gap = 18.0;
    }

    f.recent_sst_freq_hist = static_cast<double>(sst_q.size());
    f.recent_sst_miss_freq_hist = static_cast<double>(sst_miss_q.size());

    auto& pg = per_get_[tracing_get_id];
    f.same_get_num_blocks = static_cast<double>(pg.num_blocks_seen + 1);
    f.same_get_num_data_blocks =
        static_cast<double>(pg.num_data_blocks_seen + 1);

    const bool admit = V4LiteRidgeShouldAdmit(f, threshold_);

    // 当前样本不算进自己的 history，先打分，再更新
    key_q.push_back(seq_no_);
    sst_q.push_back(seq_no_);
    sst_miss_q.push_back(seq_no_);

    pg.num_blocks_seen += 1;
    pg.num_data_blocks_seen += 1;
    pg.last_seen_seq = seq_no_;

    if (out_features != nullptr) {
      *out_features = f;
    }
    return admit;
  }

 private:
  static uint64_t HashKey(const Slice& s) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < s.size(); ++i) {
      h ^= static_cast<unsigned char>(s.data()[i]);
      h *= 1099511628211ULL;
    }
    return h;
  }

  void PruneQueue(std::deque<uint64_t>& q) {
    const uint64_t floor =
        (seq_no_ > history_window_) ? (seq_no_ - history_window_) : 0;
    while (!q.empty() && q.front() <= floor) {
      q.pop_front();
    }
  }

  void CleanupStale() {
    for (auto it = key_hist_.begin(); it != key_hist_.end();) {
      PruneQueue(it->second);
      if (it->second.empty()) {
        it = key_hist_.erase(it);
      } else {
        ++it;
      }
    }

    for (auto it = sst_hist_.begin(); it != sst_hist_.end();) {
      PruneQueue(it->second);
      if (it->second.empty()) {
        it = sst_hist_.erase(it);
      } else {
        ++it;
      }
    }

    for (auto it = sst_miss_hist_.begin(); it != sst_miss_hist_.end();) {
      PruneQueue(it->second);
      if (it->second.empty()) {
        it = sst_miss_hist_.erase(it);
      } else {
        ++it;
      }
    }

    const uint64_t stale_before =
        (seq_no_ > static_cast<uint64_t>(history_window_) * 4ULL)
            ? (seq_no_ - static_cast<uint64_t>(history_window_) * 4ULL)
            : 0;

    for (auto it = per_get_.begin(); it != per_get_.end();) {
      if (it->second.last_seen_seq <= stale_before) {
        it = per_get_.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::mutex mu_;
  uint64_t seq_no_ = 0;
  uint32_t history_window_;
  double threshold_;

  std::unordered_map<uint64_t, std::deque<uint64_t>> key_hist_;
  std::unordered_map<uint64_t, std::deque<uint64_t>> sst_hist_;
  std::unordered_map<uint64_t, std::deque<uint64_t>> sst_miss_hist_;
  std::unordered_map<uint64_t, V4LitePerGetState> per_get_;
};

}  // namespace ROCKSDB_NAMESPACE
#endif //ROCKSDB_V4_LITE_RIDGE_ADMISSION_H