#!/usr/bin/env bash
set -euo pipefail

# Build scale-robustness cache-admission datasets from collected traces.
#
# Run this after collect_final_paper_matrix_robust_traces.sh finishes.
# It does not run db_bench. It only converts each run's:
#   block_trace.txt + snapshot.csv + sst_trace.tsv
# into:
#   cache_admission_dataset.csv
#
# All parameters below are passed through to rebuild_cache_admission_datasets.sh
# and then to build_cache_admission_dataset.py. Keep these names aligned with
# the receiving scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix_robust_30m_k24v400}" \
DATASET_BUILDER="${DATASET_BUILDER:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset.py}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
JOBS="${JOBS:-6}" \
TRACE_LOADER="${TRACE_LOADER:-polars}" \
HORIZON_SECONDS="${HORIZON_SECONDS:-5}" \
POSITIVE_REUSE_THRESHOLD="${POSITIVE_REUSE_THRESHOLD:-6}" \
CANDIDATE_COOLDOWN_MS="${CANDIDATE_COOLDOWN_MS:-1000}" \
MAX_FIRST_REUSE_SECONDS="${MAX_FIRST_REUSE_SECONDS:-3}" \
MIN_BENEFIT_SCORE="${MIN_BENEFIT_SCORE:-0.05}" \
FUTURE_REUSE_COUNT_MODE="${FUTURE_REUSE_COUNT_MODE:-unique_get}" \
INCLUDE_NO_INSERT="${INCLUDE_NO_INSERT:-0}" \
DATA_BLOCK_TYPES="${DATA_BLOCK_TYPES:-9}" \
BACKUP_OLD_DATASET="${BACKUP_OLD_DATASET:-1}" \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
ALLOW_EMPTY_DATASET="${ALLOW_EMPTY_DATASET:-1}" \
EXP_AWARE_FILTER="${EXP_AWARE_FILTER:-1}" \
"$SCRIPT_DIR/rebuild_cache_admission_datasets.sh"
