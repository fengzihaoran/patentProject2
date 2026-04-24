#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace ROCKSDB_NAMESPACE {

#include "table/block_based/dynamic_threshold_lookup.inc"

inline double MLCacheAdmissionResolveThreshold(const std::string& workload,
                                               const std::string& cache_label,
                                               double fallback_threshold) {
  if (!workload.empty() && !cache_label.empty()) {
    for (const auto& entry : kDynamicThresholdTable) {
      if (std::strcmp(entry.workload, workload.c_str()) == 0 &&
          std::strcmp(entry.cache_label, cache_label.c_str()) == 0) {
        return entry.threshold;
      }
    }
  }
  return fallback_threshold;
}

struct OnlineCacheAdmissionSnapshot {
  uint64_t l0_files = 0;
  uint64_t l1_files = 0;
  uint64_t estimate_pending_compaction_bytes = 0;
  uint64_t num_running_compactions = 0;
  uint64_t num_running_flushes = 0;
  uint64_t cur_size_all_mem_tables = 0;
  uint64_t block_cache_usage = 0;
  uint64_t block_cache_pinned_usage = 0;
};

class OnlineCacheAdmissionSnapshotState {
 public:
  static OnlineCacheAdmissionSnapshotState& Instance() {
    static OnlineCacheAdmissionSnapshotState state;
    return state;
  }

  void Update(uint64_t l0_files, uint64_t l1_files,
              uint64_t pending_compaction_bytes,
              uint64_t num_running_compactions,
              uint64_t num_running_flushes,
              uint64_t cur_size_all_mem_tables,
              uint64_t block_cache_usage,
              uint64_t block_cache_pinned_usage) {
    l0_files_.store(l0_files, std::memory_order_relaxed);
    l1_files_.store(l1_files, std::memory_order_relaxed);
    pending_compaction_bytes_.store(pending_compaction_bytes,
                                    std::memory_order_relaxed);
    num_running_compactions_.store(num_running_compactions,
                                   std::memory_order_relaxed);
    num_running_flushes_.store(num_running_flushes, std::memory_order_relaxed);
    cur_size_all_mem_tables_.store(cur_size_all_mem_tables,
                                   std::memory_order_relaxed);
    block_cache_usage_.store(block_cache_usage, std::memory_order_relaxed);
    block_cache_pinned_usage_.store(block_cache_pinned_usage,
                                    std::memory_order_relaxed);
  }

  OnlineCacheAdmissionSnapshot Read() const {
    OnlineCacheAdmissionSnapshot snapshot;
    snapshot.l0_files = l0_files_.load(std::memory_order_relaxed);
    snapshot.l1_files = l1_files_.load(std::memory_order_relaxed);
    snapshot.estimate_pending_compaction_bytes =
        pending_compaction_bytes_.load(std::memory_order_relaxed);
    snapshot.num_running_compactions =
        num_running_compactions_.load(std::memory_order_relaxed);
    snapshot.num_running_flushes =
        num_running_flushes_.load(std::memory_order_relaxed);
    snapshot.cur_size_all_mem_tables =
        cur_size_all_mem_tables_.load(std::memory_order_relaxed);
    snapshot.block_cache_usage =
        block_cache_usage_.load(std::memory_order_relaxed);
    snapshot.block_cache_pinned_usage =
        block_cache_pinned_usage_.load(std::memory_order_relaxed);
    return snapshot;
  }

 private:
  std::atomic<uint64_t> l0_files_{0};
  std::atomic<uint64_t> l1_files_{0};
  std::atomic<uint64_t> pending_compaction_bytes_{0};
  std::atomic<uint64_t> num_running_compactions_{0};
  std::atomic<uint64_t> num_running_flushes_{0};
  std::atomic<uint64_t> cur_size_all_mem_tables_{0};
  std::atomic<uint64_t> block_cache_usage_{0};
  std::atomic<uint64_t> block_cache_pinned_usage_{0};
};

inline void UpdateOnlineCacheAdmissionSnapshot(
    uint64_t l0_files, uint64_t l1_files, uint64_t pending_compaction_bytes,
    uint64_t num_running_compactions, uint64_t num_running_flushes,
    uint64_t cur_size_all_mem_tables, uint64_t block_cache_usage,
    uint64_t block_cache_pinned_usage) {
  OnlineCacheAdmissionSnapshotState::Instance().Update(
      l0_files, l1_files, pending_compaction_bytes, num_running_compactions,
      num_running_flushes, cur_size_all_mem_tables, block_cache_usage,
      block_cache_pinned_usage);
}

inline OnlineCacheAdmissionSnapshot ReadOnlineCacheAdmissionSnapshot() {
  return OnlineCacheAdmissionSnapshotState::Instance().Read();
}

struct MLCacheAdmissionFeatures {
  double block_size = 0.0;
  double level = 0.0;
  double l0_files = 0.0;
  double l1_files = 0.0;
  double estimate_pending_compaction_bytes = 0.0;
  double num_running_compactions = 0.0;
  double num_running_flushes = 0.0;
  double cur_size_all_mem_tables = 0.0;
  double block_cache_usage = 0.0;
  double block_cache_pinned_usage = 0.0;
};

struct MLCacheAdmissionAdaptiveConfig {
  bool enabled = false;
  uint64_t window = 2000000;
  double step = 0.01;
  double return_step = 0.005;
  double reject_low = 0.70;
  double reject_high = 0.95;
  double min_threshold = 0.35;
  double max_threshold = 0.75;
  uint32_t warmup_windows = 2;
  uint32_t consecutive_windows = 2;
};

inline double MLCacheAdmissionPredictLogit(const MLCacheAdmissionFeatures& f) {
  #include "table/block_based/ml_cache_admission_params.inc"

  const double raw[] = {
      f.block_size,
      f.level,
      f.l0_files,
      f.l1_files,
      f.estimate_pending_compaction_bytes,
      f.num_running_compactions,
      f.num_running_flushes,
      f.cur_size_all_mem_tables,
      f.block_cache_usage,
      f.block_cache_pinned_usage,
  };

  double logit = kIntercept;
  for (size_t i = 0; i < sizeof(kWeights) / sizeof(kWeights[0]); ++i) {
    double value = raw[i];
    if (!std::isfinite(value)) {
      value = kMeans[i];
    }
    logit += ((value - kMeans[i]) / kScales[i]) * kWeights[i];
  }
  return logit;
}

inline double MLCacheAdmissionPredictProbability(
    const MLCacheAdmissionFeatures& f) {
  const double logit = MLCacheAdmissionPredictLogit(f);
  if (logit >= 30.0) {
    return 1.0;
  }
  if (logit <= -30.0) {
    return 0.0;
  }
  return 1.0 / (1.0 + std::exp(-logit));
}

class MLCacheAdmissionRuntime {
 public:
  bool ShouldAdmit(uint64_t block_size, int level, double base_threshold,
                   const MLCacheAdmissionAdaptiveConfig& adaptive_config,
                   MLCacheAdmissionFeatures* out_features = nullptr,
                   double* out_probability = nullptr) {
    const auto snapshot = ReadOnlineCacheAdmissionSnapshot();

    MLCacheAdmissionFeatures features;
    features.block_size = static_cast<double>(block_size);
    features.level = static_cast<double>(level);
    features.l0_files = static_cast<double>(snapshot.l0_files);
    features.l1_files = static_cast<double>(snapshot.l1_files);
    features.estimate_pending_compaction_bytes =
        static_cast<double>(snapshot.estimate_pending_compaction_bytes);
    features.num_running_compactions =
        static_cast<double>(snapshot.num_running_compactions);
    features.num_running_flushes =
        static_cast<double>(snapshot.num_running_flushes);
    features.cur_size_all_mem_tables =
        static_cast<double>(snapshot.cur_size_all_mem_tables);
    features.block_cache_usage =
        static_cast<double>(snapshot.block_cache_usage);
    features.block_cache_pinned_usage =
        static_cast<double>(snapshot.block_cache_pinned_usage);

    const double runtime_threshold =
        ResolveRuntimeThreshold(base_threshold, adaptive_config);
    const double probability = MLCacheAdmissionPredictProbability(features);
    const bool admit = probability >= runtime_threshold;

    const uint64_t total =
        total_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (admit) {
      admit_.fetch_add(1, std::memory_order_relaxed);
    } else {
      reject_.fetch_add(1, std::memory_order_relaxed);
    }

    MaybeAdjustAdaptiveThreshold(base_threshold, admit, adaptive_config);
    MaybePrint(total, probability, admit, base_threshold, runtime_threshold,
               features);

    if (out_features != nullptr) {
      *out_features = features;
    }
    if (out_probability != nullptr) {
      *out_probability = probability;
    }
    return admit;
  }

 private:
  static constexpr int32_t kThresholdDeltaScale = 1000;

  static double ClampThreshold(double threshold, double min_threshold,
                               double max_threshold) {
    if (min_threshold > max_threshold) {
      std::swap(min_threshold, max_threshold);
    }
    if (threshold < min_threshold) {
      return min_threshold;
    }
    if (threshold > max_threshold) {
      return max_threshold;
    }
    return threshold;
  }

  static int32_t EncodeThresholdDelta(double delta) {
    return static_cast<int32_t>(
        std::llround(delta * static_cast<double>(kThresholdDeltaScale)));
  }

  static double DecodeThresholdDelta(int32_t delta_units) {
    return static_cast<double>(delta_units) /
           static_cast<double>(kThresholdDeltaScale);
  }

  double ResolveRuntimeThreshold(
      double base_threshold,
      const MLCacheAdmissionAdaptiveConfig& adaptive_config) const {
    if (!adaptive_config.enabled) {
      return base_threshold;
    }
    const double adaptive_delta = DecodeThresholdDelta(
        adaptive_delta_units_.load(std::memory_order_relaxed));
    return ClampThreshold(base_threshold + adaptive_delta,
                          adaptive_config.min_threshold,
                          adaptive_config.max_threshold);
  }

  void MaybeAdjustAdaptiveThreshold(
      double base_threshold, bool admit,
      const MLCacheAdmissionAdaptiveConfig& adaptive_config) {
    if (!adaptive_config.enabled || adaptive_config.window == 0) {
      return;
    }

    const uint64_t window_total =
        adaptive_window_total_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!admit) {
      adaptive_window_reject_.fetch_add(1, std::memory_order_relaxed);
    }
    if (window_total < adaptive_config.window) {
      return;
    }
    if (adaptive_window_processing_.exchange(true,
                                             std::memory_order_relaxed)) {
      return;
    }

    const uint64_t total =
        adaptive_window_total_.exchange(0, std::memory_order_relaxed);
    const uint64_t reject =
        adaptive_window_reject_.exchange(0, std::memory_order_relaxed);
    if (total == 0) {
      adaptive_window_processing_.store(false, std::memory_order_relaxed);
      return;
    }

    const double reject_ratio =
        static_cast<double>(reject) / static_cast<double>(total);
    const uint64_t observed_windows =
        adaptive_observed_windows_.fetch_add(1, std::memory_order_relaxed) + 1;
    const int32_t step_units =
        std::max<int32_t>(1, EncodeThresholdDelta(adaptive_config.step));
    const double return_step = adaptive_config.return_step > 0.0
                                   ? adaptive_config.return_step
                                   : adaptive_config.step;
    const int32_t return_step_units =
        std::max<int32_t>(1, EncodeThresholdDelta(return_step));

    const int32_t old_delta_units =
        adaptive_delta_units_.load(std::memory_order_relaxed);
    int32_t new_delta_units = old_delta_units;
    const char* action = "hold";
    const uint32_t required_consecutive =
        std::max<uint32_t>(1, adaptive_config.consecutive_windows);
    uint32_t high_windows =
        adaptive_high_windows_.load(std::memory_order_relaxed);
    uint32_t low_windows =
        adaptive_low_windows_.load(std::memory_order_relaxed);

    if (observed_windows <= adaptive_config.warmup_windows) {
      adaptive_high_windows_.store(0, std::memory_order_relaxed);
      adaptive_low_windows_.store(0, std::memory_order_relaxed);
      high_windows = 0;
      low_windows = 0;
      action = "warmup";
    } else if (reject_ratio > adaptive_config.reject_high) {
      high_windows =
          adaptive_high_windows_.fetch_add(1, std::memory_order_relaxed) + 1;
      adaptive_low_windows_.store(0, std::memory_order_relaxed);
      low_windows = 0;
      if (high_windows >= required_consecutive) {
        new_delta_units -= step_units;
        adaptive_high_windows_.store(0, std::memory_order_relaxed);
        high_windows = 0;
        action = "lower";
      } else {
        action = "high_wait";
      }
    } else if (reject_ratio < adaptive_config.reject_low) {
      low_windows =
          adaptive_low_windows_.fetch_add(1, std::memory_order_relaxed) + 1;
      adaptive_high_windows_.store(0, std::memory_order_relaxed);
      high_windows = 0;
      if (low_windows >= required_consecutive) {
        new_delta_units += step_units;
        adaptive_low_windows_.store(0, std::memory_order_relaxed);
        low_windows = 0;
        action = "raise";
      } else {
        action = "low_wait";
      }
    } else if (old_delta_units > 0) {
      adaptive_high_windows_.store(0, std::memory_order_relaxed);
      adaptive_low_windows_.store(0, std::memory_order_relaxed);
      high_windows = 0;
      low_windows = 0;
      new_delta_units =
          std::max<int32_t>(0, old_delta_units - return_step_units);
      action = "return_down";
    } else if (old_delta_units < 0) {
      adaptive_high_windows_.store(0, std::memory_order_relaxed);
      adaptive_low_windows_.store(0, std::memory_order_relaxed);
      high_windows = 0;
      low_windows = 0;
      new_delta_units =
          std::min<int32_t>(0, old_delta_units + return_step_units);
      action = "return_up";
    } else {
      adaptive_high_windows_.store(0, std::memory_order_relaxed);
      adaptive_low_windows_.store(0, std::memory_order_relaxed);
      high_windows = 0;
      low_windows = 0;
    }

    const double runtime_threshold = ClampThreshold(
        base_threshold + DecodeThresholdDelta(new_delta_units),
        adaptive_config.min_threshold, adaptive_config.max_threshold);
    new_delta_units = EncodeThresholdDelta(runtime_threshold - base_threshold);
    adaptive_delta_units_.store(new_delta_units, std::memory_order_relaxed);

    std::fprintf(
        stderr,
        "[ml-cache-admission-adaptive] window_total=%llu window_reject=%llu "
        "reject_ratio=%.6f action=%s observed_windows=%llu "
        "high_windows=%u low_windows=%u base_threshold=%.6f "
        "runtime_threshold=%.6f adaptive_delta=%.6f\n",
        static_cast<unsigned long long>(total),
        static_cast<unsigned long long>(reject),
        reject_ratio,
        action,
        static_cast<unsigned long long>(observed_windows),
        high_windows,
        low_windows,
        base_threshold,
        runtime_threshold,
        runtime_threshold - base_threshold);
    adaptive_window_processing_.store(false, std::memory_order_relaxed);
  }

  void MaybePrint(uint64_t total, double probability, bool admit,
                  double base_threshold, double runtime_threshold,
                  const MLCacheAdmissionFeatures& f) const {
    if (total % 1000000ULL != 0) {
      return;
    }
    const uint64_t admit_cnt = admit_.load(std::memory_order_relaxed);
    const uint64_t reject_cnt = reject_.load(std::memory_order_relaxed);
    std::fprintf(
        stderr,
        "[ml-cache-admission-lite] total=%llu admit=%llu reject=%llu "
        "reject_ratio=%.6f last_prob=%.6f last_decision=%d threshold=%.6f "
        "base_threshold=%.6f adaptive_delta=%.6f "
        "l0_files=%.0f l1_files=%.0f pending_compaction=%.0f "
        "cache_usage=%.0f cache_pinned=%.0f block_size=%.0f level=%.0f\n",
        static_cast<unsigned long long>(total),
        static_cast<unsigned long long>(admit_cnt),
        static_cast<unsigned long long>(reject_cnt),
        total > 0 ? static_cast<double>(reject_cnt) / static_cast<double>(total)
                  : 0.0,
        probability,
        admit ? 1 : 0,
        runtime_threshold,
        base_threshold,
        runtime_threshold - base_threshold,
        f.l0_files,
        f.l1_files,
        f.estimate_pending_compaction_bytes,
        f.block_cache_usage,
        f.block_cache_pinned_usage,
        f.block_size,
        f.level);
  }

  std::atomic<uint64_t> total_{0};
  std::atomic<uint64_t> admit_{0};
  std::atomic<uint64_t> reject_{0};
  std::atomic<int32_t> adaptive_delta_units_{0};
  std::atomic<uint64_t> adaptive_window_total_{0};
  std::atomic<uint64_t> adaptive_window_reject_{0};
  std::atomic<uint64_t> adaptive_observed_windows_{0};
  std::atomic<uint32_t> adaptive_high_windows_{0};
  std::atomic<uint32_t> adaptive_low_windows_{0};
  std::atomic<bool> adaptive_window_processing_{false};
};

}  // namespace ROCKSDB_NAMESPACE
