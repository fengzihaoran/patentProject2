#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

struct V4LiteRidgeFeatures {
  double block_size = 4080.0;
  double num_keys_in_block = 24.0;
  double recent_get_key_freq_hist = 3.0;
  double last_get_key_gap = 18.0;
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
    (void)sst_id;
    (void)tracing_get_id;

    const uint64_t cur_seq = seq_.fetch_add(1, std::memory_order_relaxed) + 1;

    V4LiteRidgeFeatures f;
    f.block_size = static_cast<double>(block_size);
    f.num_keys_in_block = 24.0;

    // 动态 key 特征：低开销近似
    KeyStats ks = LookupAndUpdateKey(user_key, cur_seq);
    f.recent_get_key_freq_hist = ks.recent_count;
    f.last_get_key_gap = ks.last_gap;

    // 其余先保持轻量常量，避免把重 runtime 带回来
    f.recent_sst_freq_hist = 636.0;
    f.recent_sst_miss_freq_hist = 109.0;
    f.same_get_num_blocks = 4.0;
    f.same_get_num_data_blocks = 2.0;

    f.is_l0 = (level == 0) ? 1.0 : 0.0;
    f.offset_bucket = static_cast<double>(block_offset >> 12);
    f.keys_density = 0.0058823529411764;

    const double score = V4LiteRidgePredict(f);
    const bool admit = (score >= threshold_);

    ObserveScore(score);

    const uint64_t total =
        total_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (admit) {
      admit_.fetch_add(1, std::memory_order_relaxed);
    } else {
      reject_.fetch_add(1, std::memory_order_relaxed);
    }

    MaybePrint(total, score, admit, f);

    if (out_features != nullptr) {
      *out_features = f;
    }
    return admit;
  }

  uint64_t total() const {
    return total_.load(std::memory_order_relaxed);
  }

  uint64_t admit() const {
    return admit_.load(std::memory_order_relaxed);
  }

  uint64_t reject() const {
    return reject_.load(std::memory_order_relaxed);
  }

  double threshold() const { return threshold_; }
  uint32_t history_window() const { return history_window_; }

 private:
  struct KeyStats {
    double recent_count = 0.0;
    double last_gap = 18.0;
  };

  static constexpr uint32_t kNumShards = 64;
  static constexpr uint32_t kAssoc = 4;
  static constexpr uint32_t kNumSetsPerShard = 1024;
  static constexpr uint32_t kSlotsPerShard = kAssoc * kNumSetsPerShard;

  struct KeyEntry {
    uint64_t tag = 0;       // 0 means empty
    uint64_t last_seq = 0;
    uint16_t recent_count = 0;
  };

  struct KeyShard {
    std::mutex mu;
    std::array<KeyEntry, kSlotsPerShard> entries{};
  };

  static uint64_t HashKey(const Slice& s) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < s.size(); ++i) {
      h ^= static_cast<unsigned char>(s.data()[i]);
      h *= 1099511628211ULL;
    }
    if (h == 0) {
      h = 1;
    }
    return h;
  }

  static uint16_t SaturatingAdd(uint16_t x, uint16_t y) {
    const uint32_t sum = static_cast<uint32_t>(x) + static_cast<uint32_t>(y);
    return static_cast<uint16_t>(sum > 65535U ? 65535U : sum);
  }

  uint16_t DecayAndBump(uint16_t old_count, uint64_t gap) const {
    if (gap > history_window_) {
      return 1;
    }

    uint16_t decayed = old_count;
    if (gap > history_window_ / 2) {
      decayed = static_cast<uint16_t>(decayed >> 2);
    } else if (gap > history_window_ / 4) {
      decayed = static_cast<uint16_t>(decayed >> 1);
    } else if (gap > history_window_ / 8) {
      decayed = static_cast<uint16_t>((decayed * 3) >> 2);
    }

    if (decayed == 0) {
      decayed = 1;
    }
    return SaturatingAdd(decayed, 1);
  }

  KeyStats LookupAndUpdateKey(const Slice& user_key, uint64_t cur_seq) {
    const uint64_t h = HashKey(user_key);
    const uint32_t shard_idx = static_cast<uint32_t>(h & (kNumShards - 1));
    const uint32_t set_idx =
        static_cast<uint32_t>((h >> 6) & (kNumSetsPerShard - 1));
    const uint32_t base = set_idx * kAssoc;

    KeyShard& shard = key_shards_[shard_idx];
    std::lock_guard<std::mutex> lg(shard.mu);

    KeyEntry* victim = &shard.entries[base];
    KeyStats out;

    for (uint32_t i = 0; i < kAssoc; ++i) {
      KeyEntry& e = shard.entries[base + i];
      if (e.tag == h) {

        const uint64_t gap = (e.last_seq == 0) ? history_window_ + 1
                                               : (cur_seq - e.last_seq);
        if (gap > history_window_) {
          out.recent_count = 0.0;
          out.last_gap = 18.0;
        } else {
          out.recent_count = static_cast<double>(e.recent_count);
          out.last_gap = static_cast<double>(gap);
        }

        e.recent_count = DecayAndBump(e.recent_count, gap);
        e.last_seq = cur_seq;
        return out;
      }

      if (e.tag == 0) {
        victim = &e;
      } else if (victim->tag != 0 && e.last_seq < victim->last_seq) {
        victim = &e;
      }
    }

    // cold start
    out.recent_count = 0.0;
    out.last_gap = 18.0;

    victim->tag = h;
    victim->last_seq = cur_seq;
    victim->recent_count = 1;
    return out;
  }

  void ObserveScore(double score) {
    const int64_t scaled = static_cast<int64_t>(std::llround(score * 1000000.0));
    sum_score_scaled_.fetch_add(scaled, std::memory_order_relaxed);
    UpdateMin(min_score_scaled_, scaled);
    UpdateMax(max_score_scaled_, scaled);
  }

  static void UpdateMin(std::atomic<int64_t>& dst, int64_t value) {
    int64_t cur = dst.load(std::memory_order_relaxed);
    while (value < cur &&
           !dst.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
  }

  static void UpdateMax(std::atomic<int64_t>& dst, int64_t value) {
    int64_t cur = dst.load(std::memory_order_relaxed);
    while (value > cur &&
           !dst.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
  }

  void MaybePrint(uint64_t total,
                  double score,
                  bool admit,
                  const V4LiteRidgeFeatures& f) {
    if (total % 1000000ULL != 0) {
      return;
    }

    const uint64_t admit_cnt = admit_.load(std::memory_order_relaxed);
    const uint64_t reject_cnt = reject_.load(std::memory_order_relaxed);
    const double reject_ratio =
        total > 0 ? static_cast<double>(reject_cnt) / static_cast<double>(total)
                  : 0.0;

    const int64_t sum_scaled =
        sum_score_scaled_.load(std::memory_order_relaxed);
    const int64_t min_scaled =
        min_score_scaled_.load(std::memory_order_relaxed);
    const int64_t max_scaled =
        max_score_scaled_.load(std::memory_order_relaxed);

    const double avg_score =
        total > 0 ? (static_cast<double>(sum_scaled) / 1000000.0) /
                        static_cast<double>(total)
                  : 0.0;
    const double min_score = static_cast<double>(min_scaled) / 1000000.0;
    const double max_score = static_cast<double>(max_scaled) / 1000000.0;

    std::fprintf(
        stderr,
        "[v4-lite-ridge] total=%llu admit=%llu reject=%llu reject_ratio=%.6f "
        "avg_score=%.6f min_score=%.6f max_score=%.6f "
        "last_score=%.6f last_decision=%d threshold=%.6f history_window=%u "
        "key_freq=%.3f key_gap=%.3f is_l0=%.0f block_size=%.0f\n",
        static_cast<unsigned long long>(total),
        static_cast<unsigned long long>(admit_cnt),
        static_cast<unsigned long long>(reject_cnt),
        reject_ratio,
        avg_score,
        min_score,
        max_score,
        score,
        admit ? 1 : 0,
        threshold_,
        history_window_,
        f.recent_get_key_freq_hist,
        f.last_get_key_gap,
        f.is_l0,
        f.block_size);
  }

 private:
  uint32_t history_window_;
  double threshold_;

  std::atomic<uint64_t> seq_{0};

  std::array<KeyShard, kNumShards> key_shards_{};

  std::atomic<uint64_t> total_{0};
  std::atomic<uint64_t> admit_{0};
  std::atomic<uint64_t> reject_{0};

  std::atomic<int64_t> sum_score_scaled_{0};
  std::atomic<int64_t> min_score_scaled_{
      std::numeric_limits<int64_t>::max()};
  std::atomic<int64_t> max_score_scaled_{
      std::numeric_limits<int64_t>::min()};
};

}  // namespace ROCKSDB_NAMESPACE