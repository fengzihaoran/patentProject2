#!/usr/bin/env bash
set -euo pipefail

# Focused online-eval script for one workload.
#
# Purpose:
# 1) verify whether the current online policy is collapsing to full data-block rejection
# 2) compare baseline vs a static all-reject "nodatacache" control vs low-threshold ML variants
# 3) keep the run matrix small before re-running the full paper matrix
#
# Notes:
# - "nodatacache" here is implemented as the same ML admission path with a threshold
#   above any probability score we currently observe, so data blocks are always rejected.
# - This is not a zero-overhead static engine baseline. It is an "all-reject under the
#   same admission path" control, which is the right next diagnostic step.

DB_BENCH="${DB_BENCH:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench}"
BUILD_DIR="${BUILD_DIR:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release}"
SUMMARY_PY="${SUMMARY_PY:-/home/qhsf5/yuej/patentProject2/python/scripts/rebuild_online_eval_reports.py}"

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/online_eval_one_workload_debug}"
RUN_DB_ROOT="${RUN_DB_ROOT:-$OUT_ROOT/_run_dbs}"

DB_PATH="${DB_PATH:-/yuejData/rocksdb_exp/db_10m_pristine}"
DB_LABEL="${DB_LABEL:-10M}"
NUM="${NUM:-10000000}"

WORKLOAD="${WORKLOAD:-readrandom}"
CACHE_SIZES_CSV="${CACHE_SIZES_CSV:-33554432,67108864,134217728,268435456,536870912}"
SEEDS_CSV="${SEEDS_CSV:-101,202,303}"
LOW_THRESHOLDS_CSV="${LOW_THRESHOLDS_CSV:-0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45}"
INCLUDE_NODATACACHE="${INCLUDE_NODATACACHE:-1}"
NODATACACHE_THRESHOLD="${NODATACACHE_THRESHOLD:-1.10}"

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
REBUILD_BINARIES="${REBUILD_BINARIES:-0}"
PRESERVE_RUN_DB="${PRESERVE_RUN_DB:-0}"
COLLECT_ML_SNAPSHOT="${COLLECT_ML_SNAPSHOT:-0}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-1}"

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

split_csv() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "$csv"
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

human_cache() {
  case "$1" in
    33554432) echo "32MB" ;;
    67108864) echo "64MB" ;;
    134217728) echo "128MB" ;;
    268435456) echo "256MB" ;;
    536870912) echo "512MB" ;;
    1073741824) echo "1GB" ;;
    *) echo "$1" ;;
  esac
}

workload_duration() {
  case "$1" in
    readwhilewriting|seekrandomwhilewriting|multireadwhilewriting)
      echo "$MIXED_RW_DURATION"
      ;;
    *)
      echo "$READ_ONLY_DURATION"
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

write_command_script() {
  local out_path="$1"
  local exe="$2"
  shift 2
  {
    printf '#!/usr/bin/env bash\n'
    printf '"%s"' "$exe"
    for arg in "$@"; do
      printf ' \\\n  %q' "$arg"
    done
    printf '\n'
  } > "$out_path"
  chmod +x "$out_path"
}

run_one() {
  local cache_size="$1"
  local cache_label="$2"
  local seed="$3"
  local duration_sec="$4"
  local variant="$5"
  local threshold="$6"

  local run_dir="$OUT_ROOT/$DB_LABEL/$WORKLOAD/cache_${cache_label}/seed_${seed}/${variant}"
  local work_db_dir="$RUN_DB_ROOT/$DB_LABEL/$WORKLOAD/cache_${cache_label}/seed_${seed}/${variant}_db"
  local stdout_log="$run_dir/stdout.log"
  local stderr_log="$run_dir/stderr.log"
  local command_sh="$run_dir/command.sh"
  local snapshot_csv=""

  rm -rf "$run_dir"
  mkdir -p "$run_dir"
  mkdir -p "$(dirname "$work_db_dir")"

  echo "[DBCOPY] src=$DB_PATH dst=$work_db_dir"
  copy_db_dir "$DB_PATH" "$work_db_dir"

  local args=(
    --benchmarks="$WORKLOAD"
    --db="$work_db_dir"
    --use_existing_db=true
    --num="$NUM"
    --threads="$THREADS"
    --duration="$duration_sec"
    --key_size="$KEY_SIZE"
    --value_size="$VALUE_SIZE"
    --cache_size="$cache_size"
    --cache_index_and_filter_blocks="$CACHE_INDEX_AND_FILTER_BLOCKS"
    --compression_type="$COMPRESSION_TYPE"
    --seed="$seed"
    --statistics="$STATISTICS"
    --histogram="$HISTOGRAM"
  )
  append_workload_args "$WORKLOAD" args

  if [[ "$variant" != "baseline" ]]; then
    args+=(--enable_ml_cache_admission=1)
    args+=(--ml_cache_admission_threshold="$threshold")
    if [[ "$COLLECT_ML_SNAPSHOT" == "1" ]]; then
      snapshot_csv="$run_dir/snapshot.csv"
      args+=(--cache_admission_snapshot_file="$snapshot_csv")
      args+=(--cache_admission_snapshot_interval_sec="$SNAPSHOT_INTERVAL")
    fi
  fi

  write_command_script "$command_sh" "$DB_BENCH" "${args[@]}"

  echo "[RUN] db=$DB_LABEL workload=$WORKLOAD cache=$cache_label seed=$seed variant=$variant duration=$duration_sec"
  "$DB_BENCH" "${args[@]}" >"$stdout_log" 2>"$stderr_log"

  if [[ "$PRESERVE_RUN_DB" != "1" ]]; then
    rm -rf "$work_db_dir"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DB_LABEL" "$WORKLOAD" "$cache_size" "$cache_label" "$seed" "$duration_sec" \
    "$variant" "$threshold" "$run_dir" "$stdout_log" "$stderr_log" "$snapshot_csv" \
    >> "$RUN_MANIFEST_CSV"
}

require_file "$DB_BENCH"
require_file "$SUMMARY_PY"
require_dir "$DB_PATH"

if [[ "$REBUILD_BINARIES" == "1" ]]; then
  require_dir "$BUILD_DIR"
  (
    cd "$BUILD_DIR"
    make db_bench -j20
  )
fi

split_csv "$CACHE_SIZES_CSV" CACHE_SIZES
split_csv "$SEEDS_CSV" SEEDS
split_csv "$LOW_THRESHOLDS_CSV" LOW_THRESHOLDS

mkdir -p "$OUT_ROOT"
mkdir -p "$RUN_DB_ROOT"

RUN_MANIFEST_CSV="$OUT_ROOT/run_manifest.csv"
RAW_RESULTS_CSV="$OUT_ROOT/raw_results.csv"
COMPARE_CSV="$OUT_ROOT/compare_to_baseline.csv"
THRESHOLD_SUMMARY_CSV="$OUT_ROOT/threshold_summary.csv"
WORKLOAD_THRESHOLD_SUMMARY_CSV="$OUT_ROOT/workload_threshold_summary.csv"
REPORT_MD="$OUT_ROOT/report.md"

echo "db_label,workload,cache_size,cache_label,seed,duration_sec,variant,threshold,run_dir,stdout_log,stderr_log,snapshot_csv" > "$RUN_MANIFEST_CSV"

DURATION_SEC="$(workload_duration "$WORKLOAD")"

echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] DB_PATH=$DB_PATH DB_LABEL=$DB_LABEL NUM=$NUM"
echo "[INFO] WORKLOAD=$WORKLOAD"
echo "[INFO] CACHE_SIZES=${CACHE_SIZES[*]}"
echo "[INFO] SEEDS=${SEEDS[*]}"
echo "[INFO] LOW_THRESHOLDS=${LOW_THRESHOLDS[*]}"
echo "[INFO] INCLUDE_NODATACACHE=$INCLUDE_NODATACACHE NODATACACHE_THRESHOLD=$NODATACACHE_THRESHOLD"
echo "[INFO] THREADS=$THREADS DURATION=$DURATION_SEC"

for cache_size in "${CACHE_SIZES[@]}"; do
  cache_label="$(human_cache "$cache_size")"
  for seed in "${SEEDS[@]}"; do
    run_one "$cache_size" "$cache_label" "$seed" "$DURATION_SEC" "baseline" ""
    if [[ "$INCLUDE_NODATACACHE" == "1" ]]; then
      run_one "$cache_size" "$cache_label" "$seed" "$DURATION_SEC" "ml_t${NODATACACHE_THRESHOLD}_nodatacache" "$NODATACACHE_THRESHOLD"
    fi
    for threshold in "${LOW_THRESHOLDS[@]}"; do
      run_one "$cache_size" "$cache_label" "$seed" "$DURATION_SEC" "ml_t${threshold}" "$threshold"
    done
  done
done

python "$SUMMARY_PY" \
  "$RUN_MANIFEST_CSV" \
  "$RAW_RESULTS_CSV" \
  "$COMPARE_CSV" \
  "$THRESHOLD_SUMMARY_CSV" \
  "$WORKLOAD_THRESHOLD_SUMMARY_CSV" \
  "$REPORT_MD"

echo "[DONE] focused online evaluation complete: $OUT_ROOT"
