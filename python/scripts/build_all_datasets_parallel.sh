#!/usr/bin/env bash
set -euo pipefail

# Build all cache_admission_dataset.csv files under one experiment root.
#
# This script is intentionally process-parallel instead of adding threads inside
# build_cache_admission_dataset.py. Each worker consumes one run directory:
#   block_trace.txt + snapshot.csv + sst_trace.tsv -> cache_admission_dataset.csv
#
# It preserves the dataset builder's label semantics. Tuning knobs below are
# passed through to build_cache_admission_dataset.py and should match the wrapper
# scripts in rocksdb/experiments.

JOBS="${JOBS:-6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset.py}"
ROOT="${ROOT:-/yuejData/rocksdb_exp/final_paper_matrix_main_k24v400}"

TRACE_NAME="${TRACE_NAME:-block_trace.txt}"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-snapshot.csv}"
SST_NAME="${SST_NAME:-sst_trace.tsv}"
OUT_NAME="${OUT_NAME:-cache_admission_dataset.csv}"
LOG_NAME="${LOG_NAME:-dataset_build.log}"
SUMMARY_CSV="${SUMMARY_CSV:-$ROOT/rebuild_dataset_summary.csv}"

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
SKIP_EXISTING="${SKIP_EXISTING:-0}"
ALLOW_EMPTY_DATASET="${ALLOW_EMPTY_DATASET:-1}"
EXP_AWARE_FILTER="${EXP_AWARE_FILTER:-1}"

if [[ ! -d "$ROOT" ]]; then
  echo "[ERR] missing ROOT: $ROOT" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERR] missing SCRIPT: $SCRIPT" >&2
  exit 1
fi

echo "[INFO] ROOT=$ROOT"
echo "[INFO] SCRIPT=$SCRIPT"
echo "[INFO] JOBS=$JOBS TRACE_LOADER=$TRACE_LOADER"
echo "[INFO] SKIP_EXISTING=$SKIP_EXISTING EXP_AWARE_FILTER=$EXP_AWARE_FILTER"
echo "[INFO] builder_params: horizon=$HORIZON_SECONDS reuse=$POSITIVE_REUSE_THRESHOLD cooldown_ms=$CANDIDATE_COOLDOWN_MS max_first_reuse=$MAX_FIRST_REUSE_SECONDS min_benefit=$MIN_BENEFIT_SCORE future_mode=$FUTURE_REUSE_COUNT_MODE include_no_insert=$INCLUDE_NO_INSERT data_block_types=$DATA_BLOCK_TYPES"

workload_supports_read_random_exp_range() {
  case "$1" in
    readrandom|multireadrandom)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

should_skip_trace_layout() {
  local rel="$1"
  IFS='/' read -r -a parts <<< "$rel"
  local workload="${parts[1]:-unknown}"
  local exp_dir="${parts[2]:-}"
  if [[ "$EXP_AWARE_FILTER" != "1" || "$exp_dir" != exp_* ]]; then
    return 1
  fi
  local exp_label="${exp_dir#exp_}"
  if workload_supports_read_random_exp_range "$workload"; then
    [[ "$exp_label" == "na" ]]
    return $?
  fi
  [[ "$exp_label" != "na" ]]
}

build_one_dataset() {
  local trace="$1"
  local run_dir
  run_dir="$(dirname "$trace")"
  local rel="${run_dir#${ROOT%/}/}"
  if should_skip_trace_layout "$rel"; then
    echo "[SKIP] incompatible workload/exp layout: $rel"
    return 0
  fi

  # Expected layout:
  #   <db>/<workload>/exp_<range>/cache_<size>/seed_<seed>
  # Older non-skew layout is also tolerated:
  #   <db>/<workload>/cache_<size>/seed_<seed>
  IFS='/' read -r -a parts <<< "$rel"
  local workload="${parts[1]:-unknown}"

  local snapshot_csv="$run_dir/$SNAPSHOT_NAME"
  local sst_tsv="$run_dir/$SST_NAME"
  local output="$run_dir/$OUT_NAME"
  local log="$run_dir/$LOG_NAME"

  if [[ ! -f "$snapshot_csv" ]]; then
    echo "[WARN] skip missing $SNAPSHOT_NAME: $run_dir"
    return 0
  fi
  if [[ ! -f "$sst_tsv" ]]; then
    echo "[WARN] skip missing $SST_NAME: $run_dir"
    return 0
  fi
  if [[ "$SKIP_EXISTING" == "1" && -s "$output" ]]; then
    echo "[SKIP] existing dataset: $output"
    return 0
  fi
  if [[ "$BACKUP_OLD_DATASET" == "1" && -f "$output" ]]; then
    cp -f "$output" "$output.bak"
  fi

  local args=(
    --trace-loader "$TRACE_LOADER"
    --block-trace "$trace"
    --snapshot-csv "$snapshot_csv"
    --sst-trace-tsv "$sst_tsv"
    --output "$output"
    --horizon-seconds "$HORIZON_SECONDS"
    --positive-reuse-threshold "$POSITIVE_REUSE_THRESHOLD"
    --candidate-cooldown-ms "$CANDIDATE_COOLDOWN_MS"
    --max-first-reuse-seconds "$MAX_FIRST_REUSE_SECONDS"
    --min-benefit-score "$MIN_BENEFIT_SCORE"
    --future-reuse-count-mode "$FUTURE_REUSE_COUNT_MODE"
    --data-block-types "$DATA_BLOCK_TYPES"
  )

  if [[ "$INCLUDE_NO_INSERT" == "1" ]]; then
    args+=(--include-no-insert)
  fi
  case "$workload" in
    seekrandom|seekrandomwhilewriting)
      args+=(--allow-iterator)
      ;;
  esac

  echo "[RUN] $rel"
  if "$PYTHON_BIN" "$SCRIPT" "${args[@]}" >"$log" 2>&1; then
    echo "[DONE] $output"
    return 0
  fi

  if grep -q "No candidate samples were generated" "$log"; then
    echo "[WARN] empty dataset: $rel"
    if [[ "$ALLOW_EMPTY_DATASET" == "1" ]]; then
      return 0
    fi
  fi

  echo "[ERR] failed: $rel" >&2
  cat "$log" >&2
  return 1
}

export -f build_one_dataset
export -f workload_supports_read_random_exp_range should_skip_trace_layout
export ROOT SCRIPT PYTHON_BIN TRACE_LOADER HORIZON_SECONDS POSITIVE_REUSE_THRESHOLD
export CANDIDATE_COOLDOWN_MS MAX_FIRST_REUSE_SECONDS MIN_BENEFIT_SCORE
export FUTURE_REUSE_COUNT_MODE INCLUDE_NO_INSERT DATA_BLOCK_TYPES
export BACKUP_OLD_DATASET SKIP_EXISTING ALLOW_EMPTY_DATASET EXP_AWARE_FILTER
export TRACE_NAME SNAPSHOT_NAME SST_NAME OUT_NAME LOG_NAME

find "$ROOT" -path '*/_run_dbs/*' -prune -o -type f -name "$TRACE_NAME" -print0 \
  | sort -z \
  | xargs -0 -r -P "$JOBS" -I{} bash -c 'build_one_dataset "$@"' _ {}

echo "run,status,rows,label_pos,label_neg,pos_ratio,unique_blocks,dataset_csv" > "$SUMMARY_CSV"

while IFS= read -r -d '' trace; do
  run_dir="$(dirname "$trace")"
  rel="${run_dir#${ROOT%/}/}"
  if should_skip_trace_layout "$rel"; then
    continue
  fi
  dataset_csv="$run_dir/$OUT_NAME"
  log="$run_dir/$LOG_NAME"

  if [[ -s "$dataset_csv" ]]; then
    "$PYTHON_BIN" - "$dataset_csv" "$SUMMARY_CSV" "$rel" "ok" <<'PY'
import csv
import math
import sys

dataset_csv, summary_csv, rel, status = sys.argv[1:]
rows = 0
label_pos = 0
label_neg = 0
unique_blocks = set()
with open(dataset_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows += 1
        if row.get("label") == "1":
            label_pos += 1
        else:
            label_neg += 1
        sst = row.get("sst_fd_number", "")
        off = row.get("block_offset", "")
        if sst or off:
            unique_blocks.add((sst, off))
pos_ratio = label_pos / rows if rows else math.nan
with open(summary_csv, "a", newline="") as f:
    csv.writer(f).writerow(
        [rel, status, rows, label_pos, label_neg, f"{pos_ratio:.6f}", len(unique_blocks), dataset_csv]
    )
PY
  elif [[ -f "$log" ]] && grep -q "No candidate samples were generated" "$log"; then
    "$PYTHON_BIN" - "$SUMMARY_CSV" "$rel" "$dataset_csv" <<'PY'
import csv
import sys

summary_csv, rel, dataset_csv = sys.argv[1:]
with open(summary_csv, "a", newline="") as f:
    csv.writer(f).writerow([rel, "empty", 0, 0, 0, "nan", 0, dataset_csv])
PY
  else
    "$PYTHON_BIN" - "$SUMMARY_CSV" "$rel" "$dataset_csv" <<'PY'
import csv
import sys

summary_csv, rel, dataset_csv = sys.argv[1:]
with open(summary_csv, "a", newline="") as f:
    csv.writer(f).writerow([rel, "missing", 0, 0, 0, "nan", 0, dataset_csv])
PY
  fi
done < <(find "$ROOT" -path '*/_run_dbs/*' -prune -o -type f -name "$TRACE_NAME" -print0 | sort -z)

echo "[DONE] summary=$SUMMARY_CSV"
