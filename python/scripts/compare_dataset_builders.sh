#!/usr/bin/env bash
set -euo pipefail

# Compare old and new dataset builders on one small run.
#
# Use this before trusting PR1 performance changes:
#   1) Save the pre-PR1 script as build_cache_admission_dataset_old.py.
#   2) Point RUN_DIR at one small collected run.
#   3) This script builds old/new CSVs with the same label parameters and diffs.
#
# Validation target:
#   diff prints no output.

RUN_DIR="${1:?Usage: $0 RUN_DIR}"
SCRIPT_OLD="${SCRIPT_OLD:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset_old.py}"
SCRIPT_NEW="${SCRIPT_NEW:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TRACE="${TRACE:-$RUN_DIR/block_trace.txt}"
SNAPSHOT="${SNAPSHOT:-$RUN_DIR/snapshot.csv}"
SST="${SST:-$RUN_DIR/sst_trace.tsv}"

OLD_OUT="${OLD_OUT:-$RUN_DIR/cache_admission_dataset.old.csv}"
NEW_OUT="${NEW_OUT:-$RUN_DIR/cache_admission_dataset.new.csv}"

HORIZON_SECONDS="${HORIZON_SECONDS:-5}"
POSITIVE_REUSE_THRESHOLD="${POSITIVE_REUSE_THRESHOLD:-6}"
CANDIDATE_COOLDOWN_MS="${CANDIDATE_COOLDOWN_MS:-1000}"
MAX_FIRST_REUSE_SECONDS="${MAX_FIRST_REUSE_SECONDS:-3}"
MIN_BENEFIT_SCORE="${MIN_BENEFIT_SCORE:-0.05}"
FUTURE_REUSE_COUNT_MODE="${FUTURE_REUSE_COUNT_MODE:-unique_get}"
INCLUDE_NO_INSERT="${INCLUDE_NO_INSERT:-0}"
DATA_BLOCK_TYPES="${DATA_BLOCK_TYPES:-9}"
TRACE_LOADER="${TRACE_LOADER:-polars}"

if [[ ! -f "$SCRIPT_OLD" ]]; then
  echo "[ERR] missing old builder: $SCRIPT_OLD" >&2
  echo "Copy the pre-PR1 builder to this path or set SCRIPT_OLD=/path/to/old.py" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_NEW" ]]; then
  echo "[ERR] missing new builder: $SCRIPT_NEW" >&2
  exit 1
fi
if [[ ! -f "$TRACE" ]]; then
  echo "[ERR] missing trace: $TRACE" >&2
  exit 1
fi

old_args=(
  --block-trace "$TRACE"
  --output "$OLD_OUT"
  --horizon-seconds "$HORIZON_SECONDS"
  --positive-reuse-threshold "$POSITIVE_REUSE_THRESHOLD"
  --candidate-cooldown-ms "$CANDIDATE_COOLDOWN_MS"
  --max-first-reuse-seconds "$MAX_FIRST_REUSE_SECONDS"
  --min-benefit-score "$MIN_BENEFIT_SCORE"
  --future-reuse-count-mode "$FUTURE_REUSE_COUNT_MODE"
  --data-block-types "$DATA_BLOCK_TYPES"
)

new_args=(
  --trace-loader "$TRACE_LOADER"
  --block-trace "$TRACE"
  --output "$NEW_OUT"
  --horizon-seconds "$HORIZON_SECONDS"
  --positive-reuse-threshold "$POSITIVE_REUSE_THRESHOLD"
  --candidate-cooldown-ms "$CANDIDATE_COOLDOWN_MS"
  --max-first-reuse-seconds "$MAX_FIRST_REUSE_SECONDS"
  --min-benefit-score "$MIN_BENEFIT_SCORE"
  --future-reuse-count-mode "$FUTURE_REUSE_COUNT_MODE"
  --data-block-types "$DATA_BLOCK_TYPES"
)

if [[ -f "$SNAPSHOT" ]]; then
  old_args+=(--snapshot-csv "$SNAPSHOT")
  new_args+=(--snapshot-csv "$SNAPSHOT")
fi
if [[ -f "$SST" ]]; then
  old_args+=(--sst-trace-tsv "$SST")
  new_args+=(--sst-trace-tsv "$SST")
fi
if [[ "$INCLUDE_NO_INSERT" == "1" ]]; then
  old_args+=(--include-no-insert)
  new_args+=(--include-no-insert)
fi

"$PYTHON_BIN" "$SCRIPT_OLD" "${old_args[@]}"
"$PYTHON_BIN" "$SCRIPT_NEW" "${new_args[@]}"

diff -u "$OLD_OUT" "$NEW_OUT"
