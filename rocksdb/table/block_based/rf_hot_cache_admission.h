#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <unordered_map>

#include "table/block_based/ml_cache_admission.h"

namespace ROCKSDB_NAMESPACE {

struct RFHotCacheAdmissionFeatures {
  double block_size = 0.0;
  double level = 0.0;
  double recent_block_hits_10s = 0.0;
  double recent_block_hits_60s = 0.0;
  double recent_sst_hits_10s = 0.0;
  double recent_sst_hits_60s = 0.0;
  double recent_cf_hits_10s = 0.0;
  double recent_cf_hits_60s = 0.0;
  double l0_files = 0.0;
  double l1_files = 0.0;
  double estimate_pending_compaction_bytes = 0.0;
  double num_running_compactions = 0.0;
  double num_running_flushes = 0.0;
  double cur_size_all_mem_tables = 0.0;
  double block_cache_usage = 0.0;
  double block_cache_pinned_usage = 0.0;
};

struct RFHotBlockKey {
  uint64_t cf_id = 0;
  uint64_t sst_fd_number = 0;
  uint64_t block_offset = 0;

  bool operator==(const RFHotBlockKey& other) const {
    return cf_id == other.cf_id &&
           sst_fd_number == other.sst_fd_number &&
           block_offset == other.block_offset;
  }
};

struct RFHotSstKey {
  uint64_t cf_id = 0;
  uint64_t sst_fd_number = 0;

  bool operator==(const RFHotSstKey& other) const {
    return cf_id == other.cf_id && sst_fd_number == other.sst_fd_number;
  }
};

struct RFHotBlockKeyHash {
  size_t operator()(const RFHotBlockKey& key) const {
    return static_cast<size_t>(
        key.cf_id * 1315423911ULL ^
        key.sst_fd_number * 2654435761ULL ^
        key.block_offset);
  }
};

struct RFHotSstKeyHash {
  size_t operator()(const RFHotSstKey& key) const {
    return static_cast<size_t>(
        key.cf_id * 1315423911ULL ^ key.sst_fd_number * 2654435761ULL);
  }
};

template <typename Key, typename Hash>
class RFHotRecentCounter {
 public:
  struct Counts {
    uint32_t short_hits = 0;
    uint32_t long_hits = 0;
  };

  Counts GetCounts(const Key& key, uint64_t now_us) {
    std::lock_guard<std::mutex> lg(mu_);
    auto& q = events_[key];
    PruneOld(q, now_us);
    return Counts{CountRecent(q, now_us - kShortWindowUs),
                  static_cast<uint32_t>(q.size())};
  }

  void Touch(const Key& key, uint64_t now_us) {
    std::lock_guard<std::mutex> lg(mu_);
    auto& q = events_[key];
    PruneOld(q, now_us);
    q.push_back(now_us);
  }

 private:
  static constexpr uint64_t kShortWindowUs = 10ULL * 1000ULL * 1000ULL;
  static constexpr uint64_t kLongWindowUs = 60ULL * 1000ULL * 1000ULL;

  static void PruneOld(std::deque<uint64_t>& q, uint64_t now_us) {
    const uint64_t min_ts_us =
        now_us > kLongWindowUs ? now_us - kLongWindowUs : 0;
    while (!q.empty() && q.front() < min_ts_us) {
      q.pop_front();
    }
  }

  static uint32_t CountRecent(const std::deque<uint64_t>& q, uint64_t min_ts_us) {
    uint32_t count = 0;
    for (auto it = q.rbegin(); it != q.rend(); ++it) {
      if (*it < min_ts_us) {
        break;
      }
      ++count;
    }
    return count;
  }

  std::mutex mu_;
  std::unordered_map<Key, std::deque<uint64_t>, Hash> events_;
};

struct RFHotCacheAdmissionAccess {
  uint64_t ts_us = 0;
  RFHotBlockKey block_key;
  RFHotSstKey sst_key;
  uint64_t cf_id = 0;
  RFHotCacheAdmissionFeatures features;
  bool valid = false;
};

struct RFHotTreeNode {
  int32_t feature_index;
  int32_t left;
  int32_t right;
  double threshold;
  double value0;
  double value1;
  int32_t is_leaf;
};

struct RFHotTreeMeta {
  uint32_t root;
};

inline double RFHotFeatureAt(const RFHotCacheAdmissionFeatures& f, int32_t index) {
  switch (index) {
    case 0:
      return f.block_size;
    case 1:
      return f.level;
    case 2:
      return f.recent_block_hits_10s;
    case 3:
      return f.recent_block_hits_60s;
    case 4:
      return f.recent_sst_hits_10s;
    case 5:
      return f.recent_sst_hits_60s;
    case 6:
      return f.recent_cf_hits_10s;
    case 7:
      return f.recent_cf_hits_60s;
    case 8:
      return f.l0_files;
    case 9:
      return f.l1_files;
    case 10:
      return f.estimate_pending_compaction_bytes;
    case 11:
      return f.num_running_compactions;
    case 12:
      return f.num_running_flushes;
    case 13:
      return f.cur_size_all_mem_tables;
    case 14:
      return f.block_cache_usage;
    case 15:
      return f.block_cache_pinned_usage;
    default:
      return 0.0;
  }
}

inline double RFHotPredictProbability(const RFHotCacheAdmissionFeatures& f) {
  #include "table/block_based/rf_hot_cache_admission_params.inc"

  double sum = 0.0;
  for (size_t i = 0; i < kRFHotNumTrees; ++i) {
    int32_t node_idx = static_cast<int32_t>(kRFHotTrees[i].root);
    while (true) {
      const auto& node = kRFHotNodes[node_idx];
      if (node.is_leaf != 0) {
        const double denom = node.value0 + node.value1;
        const double prob = denom > 0.0 ? (node.value1 / denom) : 0.5;
        sum += prob;
        break;
      }
      double value = RFHotFeatureAt(f, node.feature_index);
      if (!std::isfinite(value)) {
        value = 0.0;
      }
      node_idx = (value <= node.threshold) ? node.left : node.right;
    }
  }
  return kRFHotNumTrees > 0 ? (sum / static_cast<double>(kRFHotNumTrees)) : 0.5;
}

class RFHotCacheAdmissionRuntime {
 public:
  RFHotCacheAdmissionAccess PrepareAccess(uint64_t block_size, int level,
                                         uint64_t cf_id,
                                         uint64_t sst_fd_number,
                                         uint64_t block_offset) {
    RFHotCacheAdmissionAccess access;
    access.ts_us = NowMicros();
    access.cf_id = cf_id;
    access.block_key = RFHotBlockKey{cf_id, sst_fd_number, block_offset};
    access.sst_key = RFHotSstKey{cf_id, sst_fd_number};

    const auto block_counts =
        block_tracker_.GetCounts(access.block_key, access.ts_us);
    const auto sst_counts = sst_tracker_.GetCounts(access.sst_key, access.ts_us);
    const auto cf_counts = cf_tracker_.GetCounts(cf_id, access.ts_us);
    const auto snapshot = ReadOnlineCacheAdmissionSnapshot();

    access.features.block_size = static_cast<double>(block_size);
    access.features.level = static_cast<double>(level);
    access.features.recent_block_hits_10s =
        static_cast<double>(block_counts.short_hits);
    access.features.recent_block_hits_60s =
        static_cast<double>(block_counts.long_hits);
    access.features.recent_sst_hits_10s =
        static_cast<double>(sst_counts.short_hits);
    access.features.recent_sst_hits_60s =
        static_cast<double>(sst_counts.long_hits);
    access.features.recent_cf_hits_10s =
        static_cast<double>(cf_counts.short_hits);
    access.features.recent_cf_hits_60s =
        static_cast<double>(cf_counts.long_hits);
    access.features.l0_files = static_cast<double>(snapshot.l0_files);
    access.features.l1_files = static_cast<double>(snapshot.l1_files);
    access.features.estimate_pending_compaction_bytes =
        static_cast<double>(snapshot.estimate_pending_compaction_bytes);
    access.features.num_running_compactions =
        static_cast<double>(snapshot.num_running_compactions);
    access.features.num_running_flushes =
        static_cast<double>(snapshot.num_running_flushes);
    access.features.cur_size_all_mem_tables =
        static_cast<double>(snapshot.cur_size_all_mem_tables);
    access.features.block_cache_usage =
        static_cast<double>(snapshot.block_cache_usage);
    access.features.block_cache_pinned_usage =
        static_cast<double>(snapshot.block_cache_pinned_usage);
    access.valid = true;
    return access;
  }

  void ObserveAccess(const RFHotCacheAdmissionAccess& access) {
    if (!access.valid) {
      return;
    }
    block_tracker_.Touch(access.block_key, access.ts_us);
    sst_tracker_.Touch(access.sst_key, access.ts_us);
    cf_tracker_.Touch(access.cf_id, access.ts_us);
  }

  bool ShouldAdmit(const RFHotCacheAdmissionAccess& access, double threshold,
                   double* out_probability = nullptr) {
    const double probability = RFHotPredictProbability(access.features);
    const bool admit = probability >= threshold;
    const uint64_t total =
        total_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (admit) {
      admit_.fetch_add(1, std::memory_order_relaxed);
    } else {
      reject_.fetch_add(1, std::memory_order_relaxed);
    }

    if (out_probability != nullptr) {
      *out_probability = probability;
    }
    MaybePrint(total, probability, admit, threshold, access.features);
    return admit;
  }

 private:
  static uint64_t NowMicros() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<microseconds>(steady_clock::now().time_since_epoch())
            .count());
  }

  void MaybePrint(uint64_t total, double probability, bool admit,
                  double threshold,
                  const RFHotCacheAdmissionFeatures& f) const {
    if (total % 1000000ULL != 0) {
      return;
    }
    const uint64_t admit_cnt = admit_.load(std::memory_order_relaxed);
    const uint64_t reject_cnt = reject_.load(std::memory_order_relaxed);
    std::fprintf(
        stderr,
        "[rf-hot-cache-admission] total=%llu admit=%llu reject=%llu "
        "reject_ratio=%.6f last_prob=%.6f last_decision=%d threshold=%.6f "
        "block10=%.0f block60=%.0f sst10=%.0f sst60=%.0f cf10=%.0f cf60=%.0f "
        "l0=%.0f l1=%.0f pending_compaction=%.0f cache_usage=%.0f "
        "cache_pinned=%.0f block_size=%.0f level=%.0f\n",
        static_cast<unsigned long long>(total),
        static_cast<unsigned long long>(admit_cnt),
        static_cast<unsigned long long>(reject_cnt),
        total > 0 ? static_cast<double>(reject_cnt) / static_cast<double>(total)
                  : 0.0,
        probability,
        admit ? 1 : 0,
        threshold,
        f.recent_block_hits_10s,
        f.recent_block_hits_60s,
        f.recent_sst_hits_10s,
        f.recent_sst_hits_60s,
        f.recent_cf_hits_10s,
        f.recent_cf_hits_60s,
        f.l0_files,
        f.l1_files,
        f.estimate_pending_compaction_bytes,
        f.block_cache_usage,
        f.block_cache_pinned_usage,
        f.block_size,
        f.level);
  }

  RFHotRecentCounter<RFHotBlockKey, RFHotBlockKeyHash> block_tracker_;
  RFHotRecentCounter<RFHotSstKey, RFHotSstKeyHash> sst_tracker_;
  RFHotRecentCounter<uint64_t, std::hash<uint64_t>> cf_tracker_;
  std::atomic<uint64_t> total_{0};
  std::atomic<uint64_t> admit_{0};
  std::atomic<uint64_t> reject_{0};
};

}  // namespace ROCKSDB_NAMESPACE
