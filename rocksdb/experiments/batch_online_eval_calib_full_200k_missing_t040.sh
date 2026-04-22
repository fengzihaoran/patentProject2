#!/usr/bin/env bash
set -euo pipefail

# Supplement the completed 200k calibration matrix with the missing 0.40
# threshold for:
#   10M/readrandom/{32MB,128MB}/seed_{101,202,303}
#   10M/readwhilewriting/{32MB,128MB}/seed_{101,202,303}
#
# Write to an independent output root so existing calibration results are not
# overwritten. Pass this script's compare_to_baseline.csv together with the
# other calibration compare files when building the dynamic threshold lookup.

DB_BENCH="${DB_BENCH:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench}"
BUILD_DIR="${BUILD_DIR:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release}"
SUMMARY_PY="${SUMMARY_PY:-/home/qhsf5/yuej/patentProject2/python/scripts/rebuild_online_eval_reports.py}"
RUNNER="${RUNNER:-/home/qhsf5/yuej/patentProject2/rocksdb/experiments/batch_online_eval_one_workload_debug.sh}"

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/online_eval_calib_full_directio_200k_missing_t040}"
DB10_PATH="${DB10_PATH:-/yuejData/rocksdb_exp/db_10m_pristine}"

SEEDS_CSV="${SEEDS_CSV:-101,202,303}"
LOW_THRESHOLDS_CSV="${LOW_THRESHOLDS_CSV:-0.40}"

THREADS="${THREADS:-16}"
READ_ONLY_DURATION="${READ_ONLY_DURATION:-180}"
MIXED_RW_DURATION="${MIXED_RW_DURATION:-300}"
KEY_SIZE="${KEY_SIZE:-20}"
VALUE_SIZE="${VALUE_SIZE:-100}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
CACHE_INDEX_AND_FILTER_BLOCKS="${CACHE_INDEX_AND_FILTER_BLOCKS:-false}"
USE_DIRECT_READS="${USE_DIRECT_READS:-true}"
USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION="${USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION:-true}"
REBUILD_BINARIES="${REBUILD_BINARIES:-0}"
PRESERVE_RUN_DB="${PRESERVE_RUN_DB:-0}"
COLLECT_ML_SNAPSHOT="${COLLECT_ML_SNAPSHOT:-0}"
INCLUDE_NODATACACHE="${INCLUDE_NODATACACHE:-0}"

echo "[MATRIX] out=$OUT_ROOT threshold=$LOW_THRESHOLDS_CSV"
DB_BENCH="$DB_BENCH" \
BUILD_DIR="$BUILD_DIR" \
SUMMARY_PY="$SUMMARY_PY" \
OUT_ROOT="$OUT_ROOT/10M_readrandom_readwhilewriting_cache32_128_t040" \
DB_PATH="$DB10_PATH" \
DB_LABEL="10M" \
NUM="10000000" \
WORKLOADS_CSV="readrandom,readwhilewriting" \
CACHE_SIZES_CSV="33554432,134217728" \
SEEDS_CSV="$SEEDS_CSV" \
LOW_THRESHOLDS_CSV="$LOW_THRESHOLDS_CSV" \
INCLUDE_NODATACACHE="$INCLUDE_NODATACACHE" \
THREADS="$THREADS" \
READ_ONLY_DURATION="$READ_ONLY_DURATION" \
MIXED_RW_DURATION="$MIXED_RW_DURATION" \
KEY_SIZE="$KEY_SIZE" \
VALUE_SIZE="$VALUE_SIZE" \
COMPRESSION_TYPE="$COMPRESSION_TYPE" \
CACHE_INDEX_AND_FILTER_BLOCKS="$CACHE_INDEX_AND_FILTER_BLOCKS" \
USE_DIRECT_READS="$USE_DIRECT_READS" \
USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION="$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION" \
REBUILD_BINARIES="$REBUILD_BINARIES" \
PRESERVE_RUN_DB="$PRESERVE_RUN_DB" \
COLLECT_ML_SNAPSHOT="$COLLECT_ML_SNAPSHOT" \
"$RUNNER"

echo "[DONE] missing 0.40 calibration run complete: $OUT_ROOT"
