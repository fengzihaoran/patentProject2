#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for the RF-hot comparison path.
#
# Steps:
# 1) Train/export the RF-hot model from cache-admission datasets
# 2) Install exported rf_hot_cache_admission_params.inc into RocksDB source
# 3) Rebuild db_bench
# 4) Run baseline vs RF-hot online-evaluation matrix
#
# Everything is controlled by environment variables so you can quickly switch
# between a smoke run and a full paper-scale run without editing the script.

REPO_ROOT="${REPO_ROOT:-/home/qhsf5/yuej/patentProject2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/python/scripts/train_export_rf_hot.py}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$REPO_ROOT/python/output/train_export_rf_hot_final}"
TRAIN_THRESHOLDS_CSV="${TRAIN_THRESHOLDS_CSV:-0.50,0.55,0.60}"
N_ESTIMATORS="${N_ESTIMATORS:-300}"
MAX_DEPTH="${MAX_DEPTH:-8}"
MIN_SAMPLES_LEAF="${MIN_SAMPLES_LEAF:-10}"
RF_N_JOBS="${RF_N_JOBS:-8}"
RANDOM_STATE="${RANDOM_STATE:-42}"
INCLUDE_WORKLOADS="${INCLUDE_WORKLOADS:-}"
EXCLUDE_WORKLOADS="${EXCLUDE_WORKLOADS:-}"

RF_INC_TARGET="${RF_INC_TARGET:-$REPO_ROOT/rocksdb/table/block_based/rf_hot_cache_admission_params.inc}"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/rocksdb/cmake-build-release}"
DB_BENCH="${DB_BENCH:-$BUILD_DIR/db_bench}"
BUILD_JOBS="${BUILD_JOBS:-20}"

MATRIX_SCRIPT="${MATRIX_SCRIPT:-$REPO_ROOT/rocksdb/experiments/batch_online_eval_matrix.sh}"
OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/rf_hot_online_eval_$(date +%Y%m%d_%H%M%S)}"
RUN_DB_ROOT="${RUN_DB_ROOT:-$OUT_ROOT/_run_dbs}"

DB_PATHS_CSV="${DB_PATHS_CSV:-/yuejData/rocksdb_exp/db_10m_pristine,/yuejData/rocksdb_exp/db_30m_pristine}"
DB_LABELS_CSV="${DB_LABELS_CSV:-10M,30M}"
NUMS_CSV="${NUMS_CSV:-10000000,30000000}"
WORKLOADS_CSV="${WORKLOADS_CSV:-readrandom,multireadrandom,seekrandom,readwhilewriting,seekrandomwhilewriting}"
CACHE_SIZES_CSV="${CACHE_SIZES_CSV:-33554432,67108864,134217728,268435456,536870912}"
SEEDS_CSV="${SEEDS_CSV:-101,202,303}"
EVAL_THRESHOLDS_CSV="${EVAL_THRESHOLDS_CSV:-$TRAIN_THRESHOLDS_CSV}"
THREADS="${THREADS:-16}"
READ_ONLY_DURATION="${READ_ONLY_DURATION:-180}"
MIXED_RW_DURATION="${MIXED_RW_DURATION:-300}"
KEY_SIZE="${KEY_SIZE:-20}"
VALUE_SIZE="${VALUE_SIZE:-100}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
CACHE_INDEX_AND_FILTER_BLOCKS="${CACHE_INDEX_AND_FILTER_BLOCKS:-true}"
STATISTICS="${STATISTICS:-1}"
HISTOGRAM="${HISTOGRAM:-1}"
MULTIREAD_BATCH_SIZE="${MULTIREAD_BATCH_SIZE:-16}"
SEEK_NEXTS="${SEEK_NEXTS:-8}"
PRESERVE_RUN_DB="${PRESERVE_RUN_DB:-0}"
COLLECT_ADMISSION_SNAPSHOT="${COLLECT_ADMISSION_SNAPSHOT:-0}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1}"

RUN_TRAIN="${RUN_TRAIN:-1}"
INSTALL_PARAMS="${INSTALL_PARAMS:-1}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_MATRIX="${RUN_MATRIX:-1}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERR] missing file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[ERR] missing directory: $1" >&2
    exit 1
  fi
}

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] TRAIN_DATA_ROOT=$TRAIN_DATA_ROOT"
echo "[INFO] TRAIN_OUTPUT_DIR=$TRAIN_OUTPUT_DIR"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] RUN_TRAIN=$RUN_TRAIN INSTALL_PARAMS=$INSTALL_PARAMS RUN_BUILD=$RUN_BUILD RUN_MATRIX=$RUN_MATRIX"

if [[ "$RUN_TRAIN" == "1" ]]; then
  require_file "$TRAIN_SCRIPT"
  require_dir "$TRAIN_DATA_ROOT"

  train_args=(
    "$TRAIN_SCRIPT"
    --data-root "$TRAIN_DATA_ROOT"
    --output-dir "$TRAIN_OUTPUT_DIR"
    --thresholds "$TRAIN_THRESHOLDS_CSV"
    --n-estimators "$N_ESTIMATORS"
    --max-depth "$MAX_DEPTH"
    --min-samples-leaf "$MIN_SAMPLES_LEAF"
    --n-jobs "$RF_N_JOBS"
    --random-state "$RANDOM_STATE"
  )
  if [[ -n "$INCLUDE_WORKLOADS" ]]; then
    train_args+=(--include-workloads "$INCLUDE_WORKLOADS")
  fi
  if [[ -n "$EXCLUDE_WORKLOADS" ]]; then
    train_args+=(--exclude-workloads "$EXCLUDE_WORKLOADS")
  fi

  echo "[STEP] training RF-hot model"
  "$PYTHON_BIN" "${train_args[@]}"
fi

if [[ "$INSTALL_PARAMS" == "1" ]]; then
  require_file "$TRAIN_OUTPUT_DIR/rf_hot_cache_admission_params.inc"
  echo "[STEP] installing exported RF-hot params"
  cp "$TRAIN_OUTPUT_DIR/rf_hot_cache_admission_params.inc" "$RF_INC_TARGET"
fi

if [[ "$RUN_BUILD" == "1" ]]; then
  require_dir "$BUILD_DIR"
  echo "[STEP] rebuilding db_bench"
  (
    cd "$BUILD_DIR"
    make db_bench -j"$BUILD_JOBS"
  )
fi

if [[ "$RUN_MATRIX" == "1" ]]; then
  require_file "$MATRIX_SCRIPT"
  require_file "$DB_BENCH"

  echo "[STEP] running RF-hot online evaluation matrix"
  ADMISSION_MODE=rf_hot \
  ADMISSION_TAG=rf_hot \
  DB_BENCH="$DB_BENCH" \
  BUILD_DIR="$BUILD_DIR" \
  OUT_ROOT="$OUT_ROOT" \
  RUN_DB_ROOT="$RUN_DB_ROOT" \
  DB_PATHS_CSV="$DB_PATHS_CSV" \
  DB_LABELS_CSV="$DB_LABELS_CSV" \
  NUMS_CSV="$NUMS_CSV" \
  WORKLOADS_CSV="$WORKLOADS_CSV" \
  CACHE_SIZES_CSV="$CACHE_SIZES_CSV" \
  SEEDS_CSV="$SEEDS_CSV" \
  THRESHOLDS_CSV="$EVAL_THRESHOLDS_CSV" \
  THREADS="$THREADS" \
  READ_ONLY_DURATION="$READ_ONLY_DURATION" \
  MIXED_RW_DURATION="$MIXED_RW_DURATION" \
  KEY_SIZE="$KEY_SIZE" \
  VALUE_SIZE="$VALUE_SIZE" \
  COMPRESSION_TYPE="$COMPRESSION_TYPE" \
  CACHE_INDEX_AND_FILTER_BLOCKS="$CACHE_INDEX_AND_FILTER_BLOCKS" \
  STATISTICS="$STATISTICS" \
  HISTOGRAM="$HISTOGRAM" \
  MULTIREAD_BATCH_SIZE="$MULTIREAD_BATCH_SIZE" \
  SEEK_NEXTS="$SEEK_NEXTS" \
  PRESERVE_RUN_DB="$PRESERVE_RUN_DB" \
  COLLECT_ADMISSION_SNAPSHOT="$COLLECT_ADMISSION_SNAPSHOT" \
  SNAPSHOT_INTERVAL="$SNAPSHOT_INTERVAL" \
  "$MATRIX_SCRIPT"
fi

echo "[DONE] RF-hot pipeline complete"
echo "[DONE] train_output=$TRAIN_OUTPUT_DIR"
echo "[DONE] eval_output=$OUT_ROOT"
