#!/usr/bin/env bash
set -euo pipefail

# Convert deferred block_trace.bin files to block_trace.txt in parallel.
#
# Use this when collect_final_paper_matrix.sh was run with DEFER_TRACE_ANALYSIS=1.
# It does not run db_bench and does not build cache_admission_dataset.csv.
# Existing block_trace.txt files are skipped.

TRACE_ANALYZER="${TRACE_ANALYZER:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/block_cache_trace_analyzer}"
OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix_main_k24v400}"
JOBS="${JOBS:-4}"
KEEP_BLOCK_TRACE_BIN="${KEEP_BLOCK_TRACE_BIN:-0}"

if [[ ! -d "$OUT_ROOT" ]]; then
  echo "[ERR] missing OUT_ROOT: $OUT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$TRACE_ANALYZER" ]]; then
  echo "[ERR] missing TRACE_ANALYZER: $TRACE_ANALYZER" >&2
  exit 1
fi

convert_one_trace() {
  local block_bin="$1"
  local run_dir
  run_dir="$(dirname "$block_bin")"
  local block_txt="$run_dir/block_trace.txt"

  if [[ -s "$block_txt" ]]; then
    echo "[SKIP] $block_txt"
    return 0
  fi

  echo "[TRACE] $block_bin"
  "$TRACE_ANALYZER" \
    --block_cache_trace_path="$block_bin" \
    --human_readable_trace_file_path="$block_txt"

  if [[ "$KEEP_BLOCK_TRACE_BIN" != "1" ]]; then
    rm -f "$block_bin"
  fi
}

export -f convert_one_trace
export TRACE_ANALYZER KEEP_BLOCK_TRACE_BIN

find "$OUT_ROOT" -path '*/_run_dbs/*' -prune -o -type f -name 'block_trace.bin' -print0 \
  | sort -z \
  | xargs -0 -r -P "$JOBS" -I{} bash -c 'convert_one_trace "$@"' _ {}

echo "[DONE] converted deferred traces under $OUT_ROOT"
