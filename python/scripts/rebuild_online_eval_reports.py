# python /home/qhsf5/yuej/patentProject2/python/scripts/rebuild_online_eval_reports.py   /yuejData/rocksdb_exp/online_eval_matrix/run_manifest.csv   /yuejData/rocksdb_exp/online_eval_matrix/raw_results.csv   /yuejData/rocksdb_exp/online_eval_matrix/compare_to_baseline.csv   /yuejData/rocksdb_exp/online_eval_matrix/threshold_summary.csv   /yuejData/rocksdb_exp/online_eval_matrix/workload_threshold_summary.csv   /yuejData/rocksdb_exp/online_eval_matrix/report.md
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

key_columns = ["db_label", "workload", "cache_size", "cache_label", "seed"]
if "read_random_exp_range" in raw.columns:
    key_columns.append("read_random_exp_range")

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
    key_columns
    + [
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
    on=key_columns,
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

workload_group_columns = ["db_label", "workload"]
if "read_random_exp_range" in compare.columns:
    workload_group_columns.append("read_random_exp_range")
workload_threshold_summary = (
    compare.groupby(workload_group_columns + ["threshold"], dropna=False)
    .agg(agg_map)
    .reset_index()
)
workload_threshold_summary["num_runs"] = (
    compare.groupby(workload_group_columns + ["threshold"], dropna=False).size().values
)
workload_threshold_summary.to_csv(workload_threshold_summary_csv, index=False)

variant_summary_csv = compare_csv.with_name("variant_summary.csv")
variant_summary = (
    compare.groupby(["variant", "threshold"], dropna=False)
    .agg(agg_map)
    .reset_index()
)
variant_summary["num_runs"] = (
    compare.groupby(["variant", "threshold"], dropna=False).size().values
)
variant_summary.to_csv(variant_summary_csv, index=False)

workload_variant_summary_csv = compare_csv.with_name("workload_variant_summary.csv")
workload_variant_summary = (
    compare.groupby(workload_group_columns + ["variant", "threshold"], dropna=False)
    .agg(agg_map)
    .reset_index()
)
workload_variant_summary["num_runs"] = (
    compare.groupby(workload_group_columns + ["variant", "threshold"], dropna=False)
    .size()
    .values
)
workload_variant_summary.to_csv(workload_variant_summary_csv, index=False)


with report_md.open("w", encoding="utf-8") as f:
    f.write("# Online Evaluation Report\n\n")
    f.write("## Matrix\n\n")
    f.write(f"- raw_runs: `{len(raw)}`\n")
    f.write(f"- baseline_runs: `{len(baseline)}`\n")
    f.write(f"- ml_runs: `{len(ml)}`\n")
    f.write(f"- db_labels: `{', '.join(sorted(raw['db_label'].astype(str).unique()))}`\n")
    f.write(f"- workloads: `{', '.join(sorted(raw['workload'].astype(str).unique()))}`\n")
    if "read_random_exp_range" in raw.columns:
        f.write(
            f"- read_random_exp_ranges: `{', '.join(sorted(raw['read_random_exp_range'].astype(str).unique()))}`\n"
        )
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
    f.write("- `variant_summary.csv`\n")
    f.write("- `workload_variant_summary.csv`\n")

print(f"Wrote: {raw_results_csv}")
print(f"Wrote: {compare_csv}")
print(f"Wrote: {threshold_summary_csv}")
print(f"Wrote: {workload_threshold_summary_csv}")
print(f"Wrote: {variant_summary_csv}")
print(f"Wrote: {workload_variant_summary_csv}")
print(f"Wrote: {report_md}")

print("[DONE] online evaluation matrix complete: $OUT_ROOT")
