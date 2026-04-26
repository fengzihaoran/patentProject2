#!/usr/bin/env bash
set -euo pipefail

# Supplement the already completed 200k calibration matrix without overwriting it.
#
# Existing assumed completed matrix:
#   /yuejData/rocksdb_exp/online_eval_calib_full_directio_200k
#     10M/readrandom/{32MB,128MB}
#     10M/readwhilewriting/{32MB,128MB}
#
# This script runs only:
#   1) 10M/readrandom,readwhilewriting/512MB
#   2) 10M/multireadrandom/32MB,128MB,512MB
#   3) 30M/readrandom,multireadrandom,readwhilewriting/32MB,128MB,512MB
#
# Results are written under OUT_ROOT subdirectories. Use all compare_to_baseline.csv
# files together when building the dynamic threshold lookup.

DB_BENCH="${DB_BENCH:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench}"
BUILD_DIR="${BUILD_DIR:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release}"
SUMMARY_PY="${SUMMARY_PY:-/home/qhsf5/yuej/patentProject2/python/scripts/rebuild_online_eval_reports.py}"
RUNNER="${RUNNER:-/home/qhsf5/yuej/patentProject2/rocksdb/experiments/batch_online_eval_one_workload_debug.sh}"

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/online_eval_calib_full_directio_200k_missing}"
DB10_PATH="${DB10_PATH:-/yuejData/rocksdb_exp/db_10m_pristine}"
DB30_PATH="${DB30_PATH:-/yuejData/rocksdb_exp/db_30m_pristine}"

SEEDS_CSV="${SEEDS_CSV:-101,202,303}"
LOW_THRESHOLDS_CSV="${LOW_THRESHOLDS_CSV:-0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80}"

THREADS="${THREADS:-16}"
READ_ONLY_DURATION="${READ_ONLY_DURATION:-180}"
MIXED_RW_DURATION="${MIXED_RW_DURATION:-300}"
KEY_SIZE="${KEY_SIZE:-20}"
VALUE_SIZE="${VALUE_SIZE:-100}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
CACHE_INDEX_AND_FILTER_BLOCKS="${CACHE_INDEX_AND_FILTER_BLOCKS:-false}"
USE_DIRECT_READS="${USE_DIRECT_READS:-true}"
USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION="${USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION:-true}"
REBUILD_BINARIES="${REBUILD_BINARIES:-1}"
PRESERVE_RUN_DB="${PRESERVE_RUN_DB:-0}"
COLLECT_ML_SNAPSHOT="${COLLECT_ML_SNAPSHOT:-0}"
INCLUDE_NODATACACHE="${INCLUDE_NODATACACHE:-0}"

run_matrix() {
  local out_dir="$1"
  local db_path="$2"
  local db_label="$3"
  local num="$4"
  local workloads="$5"
  local cache_sizes="$6"
  local rebuild="$7"

  echo "[MATRIX] out=$out_dir db=$db_label workloads=$workloads caches=$cache_sizes rebuild=$rebuild"
  DB_BENCH="$DB_BENCH" \
  BUILD_DIR="$BUILD_DIR" \
  SUMMARY_PY="$SUMMARY_PY" \
  OUT_ROOT="$out_dir" \
  DB_PATH="$db_path" \
  DB_LABEL="$db_label" \
  NUM="$num" \
  WORKLOADS_CSV="$workloads" \
  CACHE_SIZES_CSV="$cache_sizes" \
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
  REBUILD_BINARIES="$rebuild" \
  PRESERVE_RUN_DB="$PRESERVE_RUN_DB" \
  COLLECT_ML_SNAPSHOT="$COLLECT_ML_SNAPSHOT" \
  "$RUNNER"
}

mkdir -p "$OUT_ROOT"

# Rebuild only before the first matrix.
run_matrix \
  "$OUT_ROOT/10M_readrandom_readwhilewriting_cache512" \
  "$DB10_PATH" \
  "10M" \
  "10000000" \
  "readrandom,readwhilewriting" \
  "536870912" \
  "$REBUILD_BINARIES"

run_matrix \
  "$OUT_ROOT/10M_multireadrandom_all_caches" \
  "$DB10_PATH" \
  "10M" \
  "10000000" \
  "multireadrandom" \
  "33554432,134217728,536870912" \
  "0"

run_matrix \
  "$OUT_ROOT/30M_all" \
  "$DB30_PATH" \
  "30M" \
  "30000000" \
  "readrandom,multireadrandom,readwhilewriting" \
  "33554432,134217728,536870912" \
  "0"

echo "[DONE] missing calibration matrices complete: $OUT_ROOT"
