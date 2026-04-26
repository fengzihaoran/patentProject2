#!/usr/bin/env bash
set -euo pipefail

# Rebuild cache_admission_dataset.csv files from already collected traces.
#
# This script does not rerun db_bench. It delegates process-level parallelism
# to python/scripts/build_all_datasets_parallel.sh. Keep the parameters here in
# sync with callers such as build_final_paper_matrix_datasets.sh.
#
# Input per run:
#   block_trace.txt, snapshot.csv, sst_trace.tsv
#
# Output per run:
#   cache_admission_dataset.csv, dataset_build.log
#
# Output per root:
#   rebuild_dataset_summary.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix_main_k24v400}"
DATASET_BUILDER="${DATASET_BUILDER:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset.py}"
PARALLEL_BUILDER="${PARALLEL_BUILDER:-$REPO_ROOT/python/scripts/build_all_datasets_parallel.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

JOBS="${JOBS:-6}"
TRACE_LOADER="${TRACE_LOADER:-polars}"
HORIZON_SECONDS="${HORIZON_SECONDS:-5}"
POSITIVE_REUSE_THRESHOLD="${POSITIVE_REUSE_THRESHOLD:-6}"
CANDIDATE_COOLDOWN_MS="${CANDIDATE_COOLDOWN_MS:-1000}"
MAX_FIRST_REUSE_SECONDS="${MAX_FIRST_REUSE_SECONDS:-3}"
MIN_BENEFIT_SCORE="${MIN_BENEFIT_SCORE:-0.05}"
FUTURE_REUSE_COUNT_MODE="${FUTURE_REUSE_COUNT_MODE:-unique_get}"
INCLUDE_NO_INSERT="${INCLUDE_NO_INSERT:-0}"
DATA_BLOCK_TYPES="${DATA_BLOCK_TYPES:-9}"
BACKUP_OLD_DATASET="${BACKUP_OLD_DATASET:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ALLOW_EMPTY_DATASET="${ALLOW_EMPTY_DATASET:-1}"
EXP_AWARE_FILTER="${EXP_AWARE_FILTER:-1}"
SUMMARY_CSV="${SUMMARY_CSV:-$OUT_ROOT/rebuild_dataset_summary.csv}"

if [[ ! -d "$OUT_ROOT" ]]; then
  echo "[ERR] missing OUT_ROOT: $OUT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$DATASET_BUILDER" ]]; then
  echo "[ERR] missing DATASET_BUILDER: $DATASET_BUILDER" >&2
  exit 1
fi
if [[ ! -f "$PARALLEL_BUILDER" ]]; then
  echo "[ERR] missing PARALLEL_BUILDER: $PARALLEL_BUILDER" >&2
  exit 1
fi

echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] DATASET_BUILDER=$DATASET_BUILDER"
echo "[INFO] PARALLEL_BUILDER=$PARALLEL_BUILDER"
echo "[INFO] JOBS=$JOBS TRACE_LOADER=$TRACE_LOADER SKIP_EXISTING=$SKIP_EXISTING EXP_AWARE_FILTER=$EXP_AWARE_FILTER"
echo "[INFO] builder_params: horizon=$HORIZON_SECONDS reuse=$POSITIVE_REUSE_THRESHOLD cooldown_ms=$CANDIDATE_COOLDOWN_MS max_first_reuse=$MAX_FIRST_REUSE_SECONDS min_benefit=$MIN_BENEFIT_SCORE future_mode=$FUTURE_REUSE_COUNT_MODE include_no_insert=$INCLUDE_NO_INSERT data_block_types=$DATA_BLOCK_TYPES"

JOBS="$JOBS" \
PYTHON_BIN="$PYTHON_BIN" \
SCRIPT="$DATASET_BUILDER" \
ROOT="$OUT_ROOT" \
TRACE_NAME=block_trace.txt \
SNAPSHOT_NAME=snapshot.csv \
SST_NAME=sst_trace.tsv \
OUT_NAME=cache_admission_dataset.csv \
LOG_NAME=dataset_build.log \
SUMMARY_CSV="$SUMMARY_CSV" \
TRACE_LOADER="$TRACE_LOADER" \
HORIZON_SECONDS="$HORIZON_SECONDS" \
POSITIVE_REUSE_THRESHOLD="$POSITIVE_REUSE_THRESHOLD" \
CANDIDATE_COOLDOWN_MS="$CANDIDATE_COOLDOWN_MS" \
MAX_FIRST_REUSE_SECONDS="$MAX_FIRST_REUSE_SECONDS" \
MIN_BENEFIT_SCORE="$MIN_BENEFIT_SCORE" \
FUTURE_REUSE_COUNT_MODE="$FUTURE_REUSE_COUNT_MODE" \
INCLUDE_NO_INSERT="$INCLUDE_NO_INSERT" \
DATA_BLOCK_TYPES="$DATA_BLOCK_TYPES" \
BACKUP_OLD_DATASET="$BACKUP_OLD_DATASET" \
SKIP_EXISTING="$SKIP_EXISTING" \
ALLOW_EMPTY_DATASET="$ALLOW_EMPTY_DATASET" \
EXP_AWARE_FILTER="$EXP_AWARE_FILTER" \
"$PARALLEL_BUILDER"
