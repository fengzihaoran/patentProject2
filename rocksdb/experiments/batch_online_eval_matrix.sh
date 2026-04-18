#!/usr/bin/env bash
set -euo pipefail

# Batch online evaluation for baseline vs ML cache admission.
#
# Goals:
# 1) Always evaluate from pristine source DB copies to avoid polluting source DBs
# 2) Run baseline and multiple ML thresholds under the same db_bench binary
# 3) Cover multiple DB scales / workloads / cache sizes / seeds in one pass
# 4) Auto-generate machine-readable summaries for later analysis and paper tables
#
# Outputs under OUT_ROOT:
# - run_manifest.csv
# - raw_results.csv
# - compare_to_baseline.csv
# - threshold_summary.csv
# - workload_threshold_summary.csv
# - report.md

DB_BENCH="${DB_BENCH:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench}"
BUILD_DIR="${BUILD_DIR:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release}"

OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/online_eval_matrix}"
RUN_DB_ROOT="${RUN_DB_ROOT:-$OUT_ROOT/_run_dbs}"

DB_PATHS_CSV="${DB_PATHS_CSV:-/yuejData/rocksdb_exp/db_10m_pristine,/yuejData/rocksdb_exp/db_30m_pristine}"
DB_LABELS_CSV="${DB_LABELS_CSV:-10M,30M}"
NUMS_CSV="${NUMS_CSV:-10000000,30000000}"

WORKLOADS_CSV="${WORKLOADS_CSV:-readrandom,multireadrandom,seekrandom,readwhilewriting,seekrandomwhilewriting}"
CACHE_SIZES_CSV="${CACHE_SIZES_CSV:-33554432,67108864,134217728,268435456,536870912}"
SEEDS_CSV="${SEEDS_CSV:-101,202,303}"
THRESHOLDS_CSV="${THRESHOLDS_CSV:-0.50,0.55,0.60}"

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
  local db_label="$1"
  local source_db="$2"
  local num="$3"
  local workload="$4"
  local cache_size="$5"
  local cache_label="$6"
  local seed="$7"
  local duration_sec="$8"
  local variant="$9"
  local threshold="${10}"

  local run_dir="$OUT_ROOT/$db_label/$workload/cache_${cache_label}/seed_${seed}/${variant}"
  local work_db_dir="$RUN_DB_ROOT/$db_label/$workload/cache_${cache_label}/seed_${seed}/${variant}_db"
  local stdout_log="$run_dir/stdout.log"
  local stderr_log="$run_dir/stderr.log"
  local command_sh="$run_dir/command.sh"
  local snapshot_csv=""

  rm -rf "$run_dir"
  mkdir -p "$run_dir"
  mkdir -p "$(dirname "$work_db_dir")"

  echo "[DBCOPY] src=$source_db dst=$work_db_dir"
  copy_db_dir "$source_db" "$work_db_dir"

  local args=(
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
    --seed="$seed"
    --statistics="$STATISTICS"
    --histogram="$HISTOGRAM"
  )
  append_workload_args "$workload" args

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

  echo "[RUN] db=$db_label workload=$workload cache=$cache_label seed=$seed variant=$variant duration=$duration_sec"
  "$DB_BENCH" "${args[@]}" >"$stdout_log" 2>"$stderr_log"

  if [[ "$PRESERVE_RUN_DB" != "1" ]]; then
    rm -rf "$work_db_dir"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$db_label" "$workload" "$cache_size" "$cache_label" "$seed" "$duration_sec" \
    "$variant" "$threshold" "$run_dir" "$stdout_log" "$stderr_log" "$snapshot_csv" \
    >> "$RUN_MANIFEST_CSV"
}

require_file "$DB_BENCH"

if [[ "$REBUILD_BINARIES" == "1" ]]; then
  require_dir "$BUILD_DIR"
  (
    cd "$BUILD_DIR"
    make db_bench -j20
  )
fi

split_csv "$DB_PATHS_CSV" DB_PATHS
split_csv "$DB_LABELS_CSV" DB_LABELS
split_csv "$NUMS_CSV" NUMS
split_csv "$WORKLOADS_CSV" WORKLOADS
split_csv "$CACHE_SIZES_CSV" CACHE_SIZES
split_csv "$SEEDS_CSV" SEEDS
split_csv "$THRESHOLDS_CSV" THRESHOLDS

if [[ "${#DB_PATHS[@]}" -ne "${#DB_LABELS[@]}" || "${#DB_PATHS[@]}" -ne "${#NUMS[@]}" ]]; then
  echo "[ERR] DB_PATHS_CSV, DB_LABELS_CSV and NUMS_CSV must have the same length" >&2
  exit 1
fi

for db_path in "${DB_PATHS[@]}"; do
  require_dir "$db_path"
done

mkdir -p "$OUT_ROOT"
mkdir -p "$RUN_DB_ROOT"

RUN_MANIFEST_CSV="$OUT_ROOT/run_manifest.csv"
RAW_RESULTS_CSV="$OUT_ROOT/raw_results.csv"
COMPARE_CSV="$OUT_ROOT/compare_to_baseline.csv"
THRESHOLD_SUMMARY_CSV="$OUT_ROOT/threshold_summary.csv"
WORKLOAD_THRESHOLD_SUMMARY_CSV="$OUT_ROOT/workload_threshold_summary.csv"
REPORT_MD="$OUT_ROOT/report.md"

echo "db_label,workload,cache_size,cache_label,seed,duration_sec,variant,threshold,run_dir,stdout_log,stderr_log,snapshot_csv" > "$RUN_MANIFEST_CSV"

echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] DB_LABELS=${DB_LABELS[*]}"
echo "[INFO] WORKLOADS=${WORKLOADS[*]}"
echo "[INFO] CACHE_SIZES=${CACHE_SIZES[*]}"
echo "[INFO] SEEDS=${SEEDS[*]}"
echo "[INFO] THRESHOLDS=${THRESHOLDS[*]}"
echo "[INFO] THREADS=$THREADS READ_ONLY_DURATION=$READ_ONLY_DURATION MIXED_RW_DURATION=$MIXED_RW_DURATION"
echo "[INFO] cache_index_and_filter_blocks=$CACHE_INDEX_AND_FILTER_BLOCKS use_direct_reads=$USE_DIRECT_READS use_direct_io_for_flush_and_compaction=$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION"
echo "[INFO] PRESERVE_RUN_DB=$PRESERVE_RUN_DB COLLECT_ML_SNAPSHOT=$COLLECT_ML_SNAPSHOT"

for idx in "${!DB_PATHS[@]}"; do
  db_path="${DB_PATHS[$idx]}"
  db_label="${DB_LABELS[$idx]}"
  num="${NUMS[$idx]}"

  for workload in "${WORKLOADS[@]}"; do
    for cache_size in "${CACHE_SIZES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        cache_label="$(human_cache "$cache_size")"
        duration_sec="$(workload_duration "$workload")"

        run_one "$db_label" "$db_path" "$num" "$workload" "$cache_size" "$cache_label" "$seed" "$duration_sec" "baseline" ""

        for threshold in "${THRESHOLDS[@]}"; do
          variant="ml_t${threshold}"
          run_one "$db_label" "$db_path" "$num" "$workload" "$cache_size" "$cache_label" "$seed" "$duration_sec" "$variant" "$threshold"
        done
      done
    done
  done
done

python - "$RUN_MANIFEST_CSV" "$RAW_RESULTS_CSV" "$COMPARE_CSV" "$THRESHOLD_SUMMARY_CSV" "$WORKLOAD_THRESHOLD_SUMMARY_CSV" "$REPORT_MD" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

import pandas as pd


manifest_csv = Path(sys.argv[1])
raw_results_csv = Path(sys.argv[2])
compare_csv = Path(sys.argv[3])
threshold_summary_csv = Path(sys.argv[4])
workload_threshold_summary_csv = Path(sys.argv[5])
report_md = Path(sys.argv[6])


BENCH_RE = re.compile(
    r"^(?P<bench>[A-Za-z0-9_]+)\s*:\s+"
    r"(?P<micros>[0-9.]+)\s+micros/op\s+"
    r"(?P<ops>[0-9.]+)\s+ops/sec\s+"
    r"(?P<seconds>[0-9.]+)\s+seconds\s+"
    r"(?P<operations>[0-9.]+)\s+operations"
)

COUNT_PATTERNS = {
    "data_hit_count": re.compile(r"rocksdb\.block\.cache\.data\.hit COUNT : (\d+)"),
    "data_miss_count": re.compile(r"rocksdb\.block\.cache\.data\.miss COUNT : (\d+)"),
    "data_add_count": re.compile(r"rocksdb\.block\.cache\.data\.add COUNT : (\d+)"),
}

STDERR_RE = re.compile(
    r"reject_ratio=(?P<reject_ratio>[0-9.]+).*?last_prob=(?P<last_prob>[0-9.]+)"
)

PERCENTILE_KEYS = ["P50", "P75", "P95", "P99", "P99.9", "P99.99"]


def parse_stdout(path: Path) -> dict:
    metrics = {
        "bench_name": "",
        "micros_per_op": math.nan,
        "ops_per_sec": math.nan,
        "elapsed_seconds": math.nan,
        "operations": math.nan,
        "p50_us": math.nan,
        "p75_us": math.nan,
        "p95_us": math.nan,
        "p99_us": math.nan,
        "p999_us": math.nan,
        "p9999_us": math.nan,
        "data_hit_count": math.nan,
        "data_miss_count": math.nan,
        "data_add_count": math.nan,
        "data_hit_ratio": math.nan,
    }
    if not path.exists():
      return metrics

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = BENCH_RE.search(line.strip())
        if m:
            metrics["bench_name"] = m.group("bench")
            metrics["micros_per_op"] = float(m.group("micros"))
            metrics["ops_per_sec"] = float(m.group("ops"))
            metrics["elapsed_seconds"] = float(m.group("seconds"))
            metrics["operations"] = float(m.group("operations"))
            break

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Percentiles:"):
            for key in PERCENTILE_KEYS:
                m = re.search(rf"{re.escape(key)}:\s*([0-9.]+)", line)
                if not m:
                    continue
                value = float(m.group(1))
                if key == "P50":
                    metrics["p50_us"] = value
                elif key == "P75":
                    metrics["p75_us"] = value
                elif key == "P95":
                    metrics["p95_us"] = value
                elif key == "P99":
                    metrics["p99_us"] = value
                elif key == "P99.9":
                    metrics["p999_us"] = value
                elif key == "P99.99":
                    metrics["p9999_us"] = value

    for key, pat in COUNT_PATTERNS.items():
        m = pat.search(text)
        if m:
            metrics[key] = float(m.group(1))

    hit = metrics["data_hit_count"]
    miss = metrics["data_miss_count"]
    if not math.isnan(hit) and not math.isnan(miss) and (hit + miss) > 0:
        metrics["data_hit_ratio"] = hit / (hit + miss)
    return metrics


def parse_stderr(path: Path) -> dict:
    metrics = {
        "reject_ratio": math.nan,
        "last_prob": math.nan,
    }
    if not path.exists():
        return metrics
    last_match = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STDERR_RE.search(line)
        if m:
            last_match = m
    if last_match:
        metrics["reject_ratio"] = float(last_match.group("reject_ratio"))
        metrics["last_prob"] = float(last_match.group("last_prob"))
    return metrics


manifest = pd.read_csv(manifest_csv)
rows = []
for rec in manifest.to_dict(orient="records"):
    stdout_metrics = parse_stdout(Path(rec["stdout_log"]))
    stderr_metrics = parse_stderr(Path(rec["stderr_log"]))
    row = dict(rec)
    row.update(stdout_metrics)
    row.update(stderr_metrics)
    rows.append(row)

raw = pd.DataFrame(rows)
raw.to_csv(raw_results_csv, index=False)

baseline = raw[raw["variant"] == "baseline"].copy()
baseline = baseline.rename(
    columns={
        "micros_per_op": "baseline_micros_per_op",
        "ops_per_sec": "baseline_ops_per_sec",
        "p50_us": "baseline_p50_us",
        "p75_us": "baseline_p75_us",
        "p95_us": "baseline_p95_us",
        "p99_us": "baseline_p99_us",
        "p999_us": "baseline_p999_us",
        "p9999_us": "baseline_p9999_us",
        "data_hit_ratio": "baseline_data_hit_ratio",
        "data_hit_count": "baseline_data_hit_count",
        "data_miss_count": "baseline_data_miss_count",
        "data_add_count": "baseline_data_add_count",
    }
)
baseline = baseline[
    [
        "db_label",
        "workload",
        "cache_size",
        "cache_label",
        "seed",
        "baseline_micros_per_op",
        "baseline_ops_per_sec",
        "baseline_p50_us",
        "baseline_p75_us",
        "baseline_p95_us",
        "baseline_p99_us",
        "baseline_p999_us",
        "baseline_p9999_us",
        "baseline_data_hit_ratio",
        "baseline_data_hit_count",
        "baseline_data_miss_count",
        "baseline_data_add_count",
    ]
]

ml = raw[raw["variant"] != "baseline"].copy()
compare = ml.merge(
    baseline,
    on=["db_label", "workload", "cache_size", "cache_label", "seed"],
    how="left",
)

def pct_delta(new, old):
    if pd.isna(new) or pd.isna(old) or old == 0:
        return math.nan
    return (new - old) / old * 100.0


compare["delta_ops_pct"] = compare.apply(
    lambda r: pct_delta(r["ops_per_sec"], r["baseline_ops_per_sec"]), axis=1
)
compare["delta_micros_pct"] = compare.apply(
    lambda r: pct_delta(r["micros_per_op"], r["baseline_micros_per_op"]), axis=1
)
compare["delta_p50_pct"] = compare.apply(
    lambda r: pct_delta(r["p50_us"], r["baseline_p50_us"]), axis=1
)
compare["delta_p99_pct"] = compare.apply(
    lambda r: pct_delta(r["p99_us"], r["baseline_p99_us"]), axis=1
)
compare["delta_hit_ratio"] = compare["data_hit_ratio"] - compare["baseline_data_hit_ratio"]
compare["delta_data_add_pct"] = compare.apply(
    lambda r: pct_delta(r["data_add_count"], r["baseline_data_add_count"]), axis=1
)
compare.to_csv(compare_csv, index=False)


agg_map = {
    "delta_ops_pct": "mean",
    "delta_micros_pct": "mean",
    "delta_p50_pct": "mean",
    "delta_p99_pct": "mean",
    "delta_hit_ratio": "mean",
    "delta_data_add_pct": "mean",
    "reject_ratio": "mean",
    "ops_per_sec": "mean",
    "micros_per_op": "mean",
    "p99_us": "mean",
}

threshold_summary = (
    compare.groupby("threshold", dropna=False)
    .agg(agg_map)
    .rename(columns={"variant": "num_runs"})
    .reset_index()
)
threshold_summary["num_runs"] = compare.groupby("threshold").size().values
threshold_summary.to_csv(threshold_summary_csv, index=False)

workload_threshold_summary = (
    compare.groupby(["db_label", "workload", "threshold"], dropna=False)
    .agg(agg_map)
    .reset_index()
)
workload_threshold_summary["num_runs"] = (
    compare.groupby(["db_label", "workload", "threshold"]).size().values
)
workload_threshold_summary.to_csv(workload_threshold_summary_csv, index=False)


with report_md.open("w", encoding="utf-8") as f:
    f.write("# Online Evaluation Report\n\n")
    f.write("## Matrix\n\n")
    f.write(f"- raw_runs: `{len(raw)}`\n")
    f.write(f"- baseline_runs: `{len(baseline)}`\n")
    f.write(f"- ml_runs: `{len(ml)}`\n")
    f.write(f"- db_labels: `{', '.join(sorted(raw['db_label'].astype(str).unique()))}`\n")
    f.write(f"- workloads: `{', '.join(sorted(raw['workload'].astype(str).unique()))}`\n")
    f.write(f"- cache_labels: `{', '.join(sorted(raw['cache_label'].astype(str).unique()))}`\n")
    f.write(f"- seeds: `{', '.join(str(x) for x in sorted(raw['seed'].astype(int).unique()))}`\n\n")

    f.write("## Threshold Summary\n\n")
    for _, row in threshold_summary.sort_values("threshold").iterrows():
        threshold = row["threshold"]
        f.write(
            f"- threshold=`{threshold}`: "
            f"avg_ops_pct=`{row['delta_ops_pct']:.4f}`, "
            f"avg_micros_pct=`{row['delta_micros_pct']:.4f}`, "
            f"avg_p99_pct=`{row['delta_p99_pct']:.4f}`, "
            f"avg_hit_ratio_diff=`{row['delta_hit_ratio']:.6f}`, "
            f"avg_reject_ratio=`{row['reject_ratio']:.6f}`, "
            f"num_runs=`{int(row['num_runs'])}`\n"
        )

    if not threshold_summary.empty:
        best_ops = threshold_summary.sort_values(
            ["delta_ops_pct", "threshold"], ascending=[False, True]
        ).iloc[0]
        best_p99 = threshold_summary.sort_values(
            ["delta_p99_pct", "threshold"], ascending=[True, True]
        ).iloc[0]
        f.write("\n## Best Thresholds\n\n")
        f.write(
            f"- best_by_ops: threshold=`{best_ops['threshold']}` "
            f"(avg_ops_pct=`{best_ops['delta_ops_pct']:.4f}`)\n"
        )
        f.write(
            f"- best_by_p99: threshold=`{best_p99['threshold']}` "
            f"(avg_p99_pct=`{best_p99['delta_p99_pct']:.4f}`)\n"
        )

    f.write("\n## Artifacts\n\n")
    f.write("- `run_manifest.csv`\n")
    f.write("- `raw_results.csv`\n")
    f.write("- `compare_to_baseline.csv`\n")
    f.write("- `threshold_summary.csv`\n")
    f.write("- `workload_threshold_summary.csv`\n")

print(f"Wrote: {raw_results_csv}")
print(f"Wrote: {compare_csv}")
print(f"Wrote: {threshold_summary_csv}")
print(f"Wrote: {workload_threshold_summary_csv}")
print(f"Wrote: {report_md}")
PY

echo "[DONE] online evaluation matrix complete: $OUT_ROOT"
