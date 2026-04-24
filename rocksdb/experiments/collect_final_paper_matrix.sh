#!/usr/bin/env bash
set -euo pipefail

# Final paper sampling matrix for cache-admission training.
#
# Goals:
# 1) Cover point read / batched read / seek-range / mixed read-write workloads
# 2) Cover multiple cache-pressure regimes
# 3) Produce training-ready datasets and a manifest in one pass
#
# Default matrix:
#   DB scales: 10M, 30M
#   workloads: readrandom, multireadrandom, seekrandom,
#              readwhilewriting, seekrandomwhilewriting
#   cache sizes: 32MB, 64MB, 128MB, 256MB, 512MB
#   seeds: 101, 202, 303
#
# Notes:
# - This script is intended for final paper data collection. Parameters are
#   explicit instead of relying on script defaults elsewhere.
# - The output datasets can be fed directly into train_export_logreg.py.
# - 1GB cache is intentionally not included by default. For 10M it is too close
#   to a saturation regime, and for 30M it increases runtime/storage cost
#   without adding much signal for a CCF-C / SCI 3-4 style paper.
# - Mixed write workloads will mutate the DB. This script therefore copies a
#   pristine source DB into a per-run working DB before each run.

DB_BENCH="${DB_BENCH:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench}"
TRACE_ANALYZER="${TRACE_ANALYZER:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/block_cache_trace_analyzer}"
DATASET_BUILDER="${DATASET_BUILDER:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset.py}"
BUILD_DIR="${BUILD_DIR:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release}"

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix}"

DB_PATHS_CSV="${DB_PATHS_CSV:-/yuejData/rocksdb_exp/db_10m_pristine,/yuejData/rocksdb_exp/db_30m_pristine}"
DB_LABELS_CSV="${DB_LABELS_CSV:-10M,30M}"
NUMS_CSV="${NUMS_CSV:-10000000,30000000}"
RUN_DB_ROOT="${RUN_DB_ROOT:-$OUT_ROOT/_run_dbs}"
PRESERVE_RUN_DB="${PRESERVE_RUN_DB:-0}"
ALLOW_EMPTY_DATASET="${ALLOW_EMPTY_DATASET:-1}"

WORKLOADS_CSV="${WORKLOADS_CSV:-readrandom,multireadrandom,seekrandom,readwhilewriting,seekrandomwhilewriting}"
CACHE_SIZES_CSV="${CACHE_SIZES_CSV:-33554432,67108864,134217728,268435456,536870912}"
SEEDS_CSV="${SEEDS_CSV:-101,202,303}"
READ_RANDOM_EXP_RANGE_CSV="${READ_RANDOM_EXP_RANGE_CSV:-0}"

THREADS="${THREADS:-16}"
READ_ONLY_DURATION="${READ_ONLY_DURATION:-180}"
MIXED_RW_DURATION="${MIXED_RW_DURATION:-300}"
KEY_SIZE="${KEY_SIZE:-20}"
VALUE_SIZE="${VALUE_SIZE:-100}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
CACHE_INDEX_AND_FILTER_BLOCKS="${CACHE_INDEX_AND_FILTER_BLOCKS:-false}"
USE_DIRECT_READS="${USE_DIRECT_READS:-true}"
USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION="${USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION:-true}"
STATISTICS="${STATISTICS:-1}"
HISTOGRAM="${HISTOGRAM:-1}"
TRACE_MAX_SIZE="${TRACE_MAX_SIZE:-68719476736}"
TRACE_SAMPLING_FREQUENCY="${TRACE_SAMPLING_FREQUENCY:-1}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1}"
MULTIREAD_BATCH_SIZE="${MULTIREAD_BATCH_SIZE:-16}"
SEEK_NEXTS="${SEEK_NEXTS:-8}"
REBUILD_BINARIES="${REBUILD_BINARIES:-1}"
KEEP_BLOCK_TRACE_BIN="${KEEP_BLOCK_TRACE_BIN:-0}"

# Freeze builder parameters for reproducibility.
HORIZON_SECONDS="${HORIZON_SECONDS:-5}"
POSITIVE_REUSE_THRESHOLD="${POSITIVE_REUSE_THRESHOLD:-8}"
CANDIDATE_COOLDOWN_MS="${CANDIDATE_COOLDOWN_MS:-3000}"
MAX_FIRST_REUSE_SECONDS="${MAX_FIRST_REUSE_SECONDS:-3}"
MIN_BENEFIT_SCORE="${MIN_BENEFIT_SCORE:-0.05}"

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

copy_db_dir() {
  local src="$1"
  local dst="$2"
  rm -rf "$dst"
  mkdir -p "$dst"

  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$src"/ "$dst"/
  else
    cp -a "$src"/. "$dst"/
  fi
}

split_csv() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "$csv"
}

human_cache() {
  case "$1" in
    33554432) echo "32MB" ;;
    67108864) echo "64MB" ;;
    134217728) echo "128MB" ;;
    268435456) echo "256MB" ;;
    536870912) echo "512MB" ;;
    *) echo "$1" ;;
  esac
}

exp_range_label() {
  echo "$1" | sed 's/-/m/g; s/\./p/g'
}

workload_supports_read_random_exp_range() {
  case "$1" in
    readrandom|multireadrandom|readwhilewriting|multireadwhilewriting)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

append_workload_args() {
  local workload="$1"
  local -n args_ref="$2"
  case "$workload" in
    multireadrandom|multireadwhilewriting)
      args_ref+=(--batch_size="$MULTIREAD_BATCH_SIZE")
      args_ref+=(--multiread_batched=true)
      ;;
    seekrandom|seekrandomwhilewriting)
      args_ref+=(--seek_nexts="$SEEK_NEXTS")
      ;;
  esac
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

workload_duration() {
  local workload="$1"
  case "$workload" in
    readwhilewriting|seekrandomwhilewriting|multireadwhilewriting)
      echo "$MIXED_RW_DURATION"
      ;;
    *)
      echo "$READ_ONLY_DURATION"
      ;;
  esac
}

require_file "$DATASET_BUILDER"

if [[ "$REBUILD_BINARIES" == "1" ]]; then
  require_dir "$BUILD_DIR"
  (
    cd "$BUILD_DIR"
    make db_bench block_cache_trace_analyzer -j20
  )
fi

require_file "$DB_BENCH"
require_file "$TRACE_ANALYZER"

split_csv "$WORKLOADS_CSV" WORKLOADS
split_csv "$CACHE_SIZES_CSV" CACHE_SIZES
split_csv "$SEEDS_CSV" SEEDS
split_csv "$READ_RANDOM_EXP_RANGE_CSV" READ_RANDOM_EXP_RANGES
split_csv "$DB_PATHS_CSV" DB_PATHS
split_csv "$DB_LABELS_CSV" DB_LABELS
split_csv "$NUMS_CSV" NUMS

if [[ "${#DB_PATHS[@]}" -ne "${#DB_LABELS[@]}" || "${#DB_PATHS[@]}" -ne "${#NUMS[@]}" ]]; then
  echo "[ERR] DB_PATHS_CSV, DB_LABELS_CSV and NUMS_CSV must have the same length" >&2
  exit 1
fi

for db_path in "${DB_PATHS[@]}"; do
  require_dir "$db_path"
done

MATRIX_ROOT="$OUT_ROOT"
MANIFEST_CSV="$MATRIX_ROOT/manifest.csv"
SUMMARY_CSV="$MATRIX_ROOT/manifest_summary.csv"
REPORT_MD="$MATRIX_ROOT/report.md"

mkdir -p "$MATRIX_ROOT"
mkdir -p "$RUN_DB_ROOT"

echo "db_label,workload,read_random_exp_range,read_random_exp_label,cache_size,cache_label,seed,duration_sec,status,rows,label_pos,label_neg,pos_ratio,unique_blocks,snapshot_rows,snapshot_l0_unique,snapshot_l1_unique,run_dir,source_db_path,work_db_dir" > "$MANIFEST_CSV"

echo "[INFO] MATRIX_ROOT=$MATRIX_ROOT"
echo "[INFO] DB_LABELS=${DB_LABELS[*]}"
echo "[INFO] WORKLOADS=${WORKLOADS[*]}"
echo "[INFO] READ_RANDOM_EXP_RANGES=${READ_RANDOM_EXP_RANGES[*]}"
echo "[INFO] CACHE_SIZES=${CACHE_SIZES[*]}"
echo "[INFO] SEEDS=${SEEDS[*]}"
echo "[INFO] NUMS=${NUMS[*]} THREADS=$THREADS READ_ONLY_DURATION=$READ_ONLY_DURATION MIXED_RW_DURATION=$MIXED_RW_DURATION"
echo "[INFO] cache_index_and_filter_blocks=$CACHE_INDEX_AND_FILTER_BLOCKS use_direct_reads=$USE_DIRECT_READS use_direct_io_for_flush_and_compaction=$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION"
echo "[INFO] RUN_DB_ROOT=$RUN_DB_ROOT PRESERVE_RUN_DB=$PRESERVE_RUN_DB ALLOW_EMPTY_DATASET=$ALLOW_EMPTY_DATASET"
echo "[INFO] builder_params: horizon=$HORIZON_SECONDS reuse=$POSITIVE_REUSE_THRESHOLD cooldown_ms=$CANDIDATE_COOLDOWN_MS max_first_reuse=$MAX_FIRST_REUSE_SECONDS min_benefit=$MIN_BENEFIT_SCORE"

for idx in "${!DB_PATHS[@]}"; do
  db_path="${DB_PATHS[$idx]}"
  db_label="${DB_LABELS[$idx]}"
  num="${NUMS[$idx]}"
  label_root="$MATRIX_ROOT/$db_label"
  run_db_label_root="$RUN_DB_ROOT/$db_label"
  mkdir -p "$label_root"
  mkdir -p "$run_db_label_root"

  for workload in "${WORKLOADS[@]}"; do
    for read_random_exp_range in "${READ_RANDOM_EXP_RANGES[@]}"; do
      if ! workload_supports_read_random_exp_range "$workload" && [[ "$read_random_exp_range" != "0" && "$read_random_exp_range" != "0.0" ]]; then
        echo "[SKIP] workload=$workload ignores read_random_exp_range=$read_random_exp_range"
        continue
      fi
      read_random_exp_label="$(exp_range_label "$read_random_exp_range")"
      for cache_size in "${CACHE_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
        cache_label="$(human_cache "$cache_size")"
        duration_sec="$(workload_duration "$workload")"
        run_dir="$label_root/$workload/exp_${read_random_exp_label}/cache_${cache_label}/seed_${seed}"
        mkdir -p "$run_dir"

        block_bin="$run_dir/block_trace.bin"
        block_txt="$run_dir/block_trace.txt"
        snapshot_csv="$run_dir/snapshot.csv"
        sst_tsv="$run_dir/sst_trace.tsv"
        dataset_csv="$run_dir/cache_admission_dataset.csv"
        dataset_log="$run_dir/dataset_build.log"
        stdout_log="$run_dir/stdout.log"
        stderr_log="$run_dir/stderr.log"
        command_sh="$run_dir/command.sh"
        work_db_dir="$run_db_label_root/$workload/exp_${read_random_exp_label}/cache_${cache_label}/seed_${seed}"

        rm -f "$block_bin" "$block_txt" "$snapshot_csv" "$sst_tsv" "$dataset_csv" "$dataset_log" "$stdout_log" "$stderr_log" "$command_sh"
        mkdir -p "$(dirname "$work_db_dir")"

        echo "[DBCOPY] src=$db_path dst=$work_db_dir"
        copy_db_dir "$db_path" "$work_db_dir"

        args=(
          --benchmarks="$workload"
          --db="$work_db_dir"
          --use_existing_db=true
          --num="$num"
          --threads="$THREADS"
          --duration="$duration_sec"
          --key_size="$KEY_SIZE"
          --value_size="$VALUE_SIZE"
          --cache_size="$cache_size"
          --cache_index_and_filter_blocks="$CACHE_INDEX_AND_FILTER_BLOCKS"
          --use_direct_reads="$USE_DIRECT_READS"
          --use_direct_io_for_flush_and_compaction="$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION"
          --compression_type="$COMPRESSION_TYPE"
          --read_random_exp_range="$read_random_exp_range"
          --seed="$seed"
          --statistics="$STATISTICS"
          --histogram="$HISTOGRAM"
          --block_cache_trace_file="$block_bin"
          --block_cache_trace_sampling_frequency="$TRACE_SAMPLING_FREQUENCY"
          --block_cache_trace_max_trace_file_size_in_bytes="$TRACE_MAX_SIZE"
          --cache_admission_snapshot_file="$snapshot_csv"
          --cache_admission_snapshot_interval_sec="$SNAPSHOT_INTERVAL"
          --cache_admission_sst_trace_file="$sst_tsv"
        )
        append_workload_args "$workload" args

        {
          printf '#!/usr/bin/env bash\n'
          printf '"%s"' "$DB_BENCH"
          for arg in "${args[@]}"; do
            printf ' \\\n  %q' "$arg"
          done
          printf '\n'
        } > "$command_sh"
        chmod +x "$command_sh"

        echo "[RUN] db=$db_label workload=$workload exp=$read_random_exp_range cache=$cache_label seed=$seed duration=$duration_sec"
        "$DB_BENCH" "${args[@]}" >"$stdout_log" 2>"$stderr_log"

        echo "[TRACE] db=$db_label workload=$workload exp=$read_random_exp_range cache=$cache_label seed=$seed"
        "$TRACE_ANALYZER" \
          --block_cache_trace_path="$block_bin" \
          --human_readable_trace_file_path="$block_txt"

        if [[ "$KEEP_BLOCK_TRACE_BIN" != "1" ]]; then
          rm -f "$block_bin"
        fi

        echo "[DATASET] db=$db_label workload=$workload exp=$read_random_exp_range cache=$cache_label seed=$seed"
        dataset_status="ok"
        builder_args=(
          --block-trace "$block_txt" \
          --snapshot-csv "$snapshot_csv" \
          --sst-trace-tsv "$sst_tsv" \
          --output "$dataset_csv" \
          --horizon-seconds "$HORIZON_SECONDS" \
          --positive-reuse-threshold "$POSITIVE_REUSE_THRESHOLD" \
          --candidate-cooldown-ms "$CANDIDATE_COOLDOWN_MS" \
          --max-first-reuse-seconds "$MAX_FIRST_REUSE_SECONDS" \
          --min-benefit-score "$MIN_BENEFIT_SCORE"
        )
        append_builder_args "$workload" builder_args

        if ! python "$DATASET_BUILDER" \
          "${builder_args[@]}" \
          >"$dataset_log" 2>&1; then
          if grep -q "No candidate samples were generated" "$dataset_log"; then
            dataset_status="empty_dataset"
            echo "[WARN] empty dataset: db=$db_label workload=$workload exp=$read_random_exp_range cache=$cache_label seed=$seed"
          else
            echo "[ERR] dataset build failed: db=$db_label workload=$workload exp=$read_random_exp_range cache=$cache_label seed=$seed" >&2
            cat "$dataset_log" >&2
            exit 1
          fi
        fi

        if [[ "$dataset_status" == "empty_dataset" && "$ALLOW_EMPTY_DATASET" != "1" ]]; then
          echo "[ERR] empty dataset and ALLOW_EMPTY_DATASET=0: db=$db_label workload=$workload exp=$read_random_exp_range cache=$cache_label seed=$seed" >&2
          cat "$dataset_log" >&2
          exit 1
        fi

        if [[ "$PRESERVE_RUN_DB" != "1" ]]; then
          rm -rf "$work_db_dir"
        fi

        python - "$dataset_csv" "$snapshot_csv" "$MANIFEST_CSV" "$db_label" "$workload" "$read_random_exp_range" "$read_random_exp_label" "$cache_size" "$cache_label" "$seed" "$duration_sec" "$run_dir" "$db_path" "$work_db_dir" "$dataset_status" <<'PY'
import csv
import os
import sys
import pandas as pd

dataset_csv, snapshot_csv, manifest_csv, db_label, workload, read_random_exp_range, read_random_exp_label, cache_size, cache_label, seed, duration_sec, run_dir, source_db_path, work_db_dir, dataset_status = sys.argv[1:]
if os.path.exists(dataset_csv):
    df = pd.read_csv(dataset_csv)
else:
    df = pd.DataFrame()
snap = pd.read_csv(snapshot_csv)

if {"sst_fd_number", "block_offset"}.issubset(df.columns):
    unique_blocks = int((df["sst_fd_number"].astype(str) + ":" + df["block_offset"].astype(str)).nunique())
else:
    unique_blocks = -1

label_pos = int((df["label"] == 1).sum()) if "label" in df.columns else -1
label_neg = int((df["label"] == 0).sum()) if "label" in df.columns else -1
pos_ratio = (label_pos / len(df)) if len(df) else float("nan")

row = [
    db_label,
    workload,
    read_random_exp_range,
    read_random_exp_label,
    cache_size,
    cache_label,
    seed,
    duration_sec,
    dataset_status,
    len(df),
    label_pos,
    label_neg,
    f"{pos_ratio:.6f}",
    unique_blocks,
    len(snap),
    snap["l0_files"].nunique() if "l0_files" in snap.columns else -1,
    snap["l1_files"].nunique() if "l1_files" in snap.columns else -1,
    run_dir,
    source_db_path,
    work_db_dir,
]

with open(manifest_csv, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
        done
      done
    done
  done
done

python - "$MANIFEST_CSV" "$SUMMARY_CSV" "$REPORT_MD" "$DB_LABELS_CSV" "$WORKLOADS_CSV" "$READ_RANDOM_EXP_RANGE_CSV" "$CACHE_SIZES_CSV" "$SEEDS_CSV" <<'PY'
import sys
import pandas as pd

manifest_csv, summary_csv, report_md, db_labels_csv, workloads_csv, read_random_exp_range_csv, cache_sizes_csv, seeds_csv = sys.argv[1:]
df = pd.read_csv(manifest_csv)

summary = (
    df.groupby(["db_label", "workload", "read_random_exp_range", "cache_label", "status"], as_index=False)
      .agg(
          runs=("rows", "count"),
          rows_mean=("rows", "mean"),
          pos_ratio_mean=("pos_ratio", "mean"),
          unique_blocks_mean=("unique_blocks", "mean"),
          l0_unique_mean=("snapshot_l0_unique", "mean"),
          l1_unique_mean=("snapshot_l1_unique", "mean"),
      )
)
summary.to_csv(summary_csv, index=False)

with open(report_md, "w", encoding="utf-8") as f:
    f.write("# Final Paper Sampling Report\n\n")
    f.write(f"- db_labels: `{db_labels_csv}`\n")
    f.write(f"- workloads: `{workloads_csv}`\n")
    f.write(f"- read_random_exp_range: `{read_random_exp_range_csv}`\n")
    f.write(f"- cache_sizes: `{cache_sizes_csv}`\n")
    f.write(f"- seeds: `{seeds_csv}`\n")
    f.write(f"- total_runs: `{len(df)}`\n")
    f.write(f"- total_rows: `{int(df['rows'].sum())}`\n")
    f.write(f"- avg_pos_ratio: `{df['pos_ratio'].mean():.6f}`\n\n")
    f.write("## Per Workload / Cache\n\n")
    f.write("| db | workload | exp | cache | status | runs | rows_mean | pos_ratio_mean | unique_blocks_mean | l0_unique_mean | l1_unique_mean |\n")
    f.write("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for _, row in summary.iterrows():
        f.write(
            f"| {row['db_label']} | {row['workload']} | {row['read_random_exp_range']} | {row['cache_label']} | {row['status']} | {int(row['runs'])} | "
            f"{row['rows_mean']:.1f} | {row['pos_ratio_mean']:.4f} | "
            f"{row['unique_blocks_mean']:.1f} | {row['l0_unique_mean']:.1f} | {row['l1_unique_mean']:.1f} |\n"
        )
PY

echo "[DONE] manifest: $MANIFEST_CSV"
echo "[DONE] summary:  $SUMMARY_CSV"
echo "[DONE] report:   $REPORT_MD"
