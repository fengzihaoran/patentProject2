#!/usr/bin/env bash
set -euo pipefail

# Rebuild cache_admission_dataset.csv files for an existing workload directory.
#
# Default use case:
#   rebuild multireadrandom datasets after fixing builder-side parsing bugs.
# This is a legacy single-workload helper. The final paper path should use
# build_final_paper_matrix_datasets.sh, which supports JOBS-based parallelism.
#
# It expects the run directories to already contain:
#   snapshot.csv
#   sst_trace.tsv
# and either:
#   block_trace.txt
# or:
#   block_trace.bin (in which case TRACE_ANALYZER is used to regenerate txt)

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix}"
WORKLOAD="${WORKLOAD:-multireadrandom}"
DB_LABELS_CSV="${DB_LABELS_CSV:-10M,30M}"
CACHE_LABELS_CSV="${CACHE_LABELS_CSV:-32MB,64MB,128MB,256MB,512MB}"
SEEDS_CSV="${SEEDS_CSV:-101,202,303}"

DATASET_BUILDER="${DATASET_BUILDER:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset.py}"
TRACE_ANALYZER="${TRACE_ANALYZER:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/block_cache_trace_analyzer}"

HORIZON_SECONDS="${HORIZON_SECONDS:-5}"
POSITIVE_REUSE_THRESHOLD="${POSITIVE_REUSE_THRESHOLD:-6}"
CANDIDATE_COOLDOWN_MS="${CANDIDATE_COOLDOWN_MS:-1000}"
MAX_FIRST_REUSE_SECONDS="${MAX_FIRST_REUSE_SECONDS:-3}"
MIN_BENEFIT_SCORE="${MIN_BENEFIT_SCORE:-0.05}"
FUTURE_REUSE_COUNT_MODE="${FUTURE_REUSE_COUNT_MODE:-unique_get}"
INCLUDE_NO_INSERT="${INCLUDE_NO_INSERT:-0}"
DATA_BLOCK_TYPES="${DATA_BLOCK_TYPES:-9}"
TRACE_LOADER="${TRACE_LOADER:-polars}"

split_csv() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "$csv"
}

append_builder_args() {
  local workload="$1"
  local -n args_ref="$2"
  case "$workload" in
    seekrandom|seekrandomwhilewriting)
      args_ref+=(--allow-iterator)
      ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERR] missing file: $1" >&2
    exit 1
  fi
}

require_file "$DATASET_BUILDER"
require_file "$TRACE_ANALYZER"

split_csv "$DB_LABELS_CSV" DB_LABELS
split_csv "$CACHE_LABELS_CSV" CACHE_LABELS
split_csv "$SEEDS_CSV" SEEDS

total=0
rebuilt=0
skipped=0

for db_label in "${DB_LABELS[@]}"; do
  for cache_label in "${CACHE_LABELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_dir="$OUT_ROOT/$db_label/$WORKLOAD/cache_${cache_label}/seed_${seed}"
      block_txt="$run_dir/block_trace.txt"
      block_bin="$run_dir/block_trace.bin"
      snapshot_csv="$run_dir/snapshot.csv"
      sst_tsv="$run_dir/sst_trace.tsv"
      dataset_csv="$run_dir/cache_admission_dataset.csv"
      dataset_log="$run_dir/dataset_build.log"

      total=$((total + 1))

      if [[ ! -d "$run_dir" ]]; then
        echo "[SKIP] missing run dir: $run_dir"
        skipped=$((skipped + 1))
        continue
      fi

      if [[ ! -f "$block_txt" ]]; then
        if [[ -f "$block_bin" ]]; then
          echo "[TRACE] regenerate txt: $run_dir"
          "$TRACE_ANALYZER" \
            --block_cache_trace_path="$block_bin" \
            --human_readable_trace_file_path="$block_txt"
        else
          echo "[SKIP] missing block trace txt/bin: $run_dir"
          skipped=$((skipped + 1))
          continue
        fi
      fi

      if [[ ! -f "$snapshot_csv" || ! -f "$sst_tsv" ]]; then
        echo "[SKIP] missing snapshot or sst trace: $run_dir"
        skipped=$((skipped + 1))
        continue
      fi

      builder_args=(
        --trace-loader "$TRACE_LOADER"
        --block-trace "$block_txt"
        --snapshot-csv "$snapshot_csv"
        --sst-trace-tsv "$sst_tsv"
        --output "$dataset_csv"
        --horizon-seconds "$HORIZON_SECONDS"
        --positive-reuse-threshold "$POSITIVE_REUSE_THRESHOLD"
        --candidate-cooldown-ms "$CANDIDATE_COOLDOWN_MS"
        --max-first-reuse-seconds "$MAX_FIRST_REUSE_SECONDS"
        --min-benefit-score "$MIN_BENEFIT_SCORE"
        --future-reuse-count-mode "$FUTURE_REUSE_COUNT_MODE"
        --data-block-types "$DATA_BLOCK_TYPES"
      )
      if [[ "$INCLUDE_NO_INSERT" == "1" ]]; then
        builder_args+=(--include-no-insert)
      fi
      append_builder_args "$WORKLOAD" builder_args

      echo "[DATASET] db=$db_label workload=$WORKLOAD cache=$cache_label seed=$seed"
      if python "$DATASET_BUILDER" "${builder_args[@]}" >"$dataset_log" 2>&1; then
        rebuilt=$((rebuilt + 1))
      else
        echo "[ERR] dataset build failed: $run_dir" >&2
        cat "$dataset_log" >&2
        exit 1
      fi
    done
  done
done

echo "[DONE] workload=$WORKLOAD total=$total rebuilt=$rebuilt skipped=$skipped"
