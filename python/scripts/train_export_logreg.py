#!/usr/bin/env python3
"""
Train and export a logistic-regression cache-admission model aligned with the
online C++ admission gate.

Outputs:
- final_model_params.json: intercept, means, scales, weights, feature order
- ml_cache_admission_params.inc: C++ snippet for ml_cache_admission.h
- oof_predictions.csv: leave-one-run-out out-of-fold predictions
- threshold_metrics.csv: pooled metrics for requested thresholds
- report.md: compact training/evaluation summary

Sample:
python /home/qhsf5/yuej/patentProject2/python/scripts/train_export_logreg.py \
  --data-root /yuejData/rocksdb_exp/final_paper_matrix_directio \
  --include-workloads readrandom,multireadrandom,readwhilewriting \
  --output-dir /home/qhsf5/yuej/patentProject2/python/output/train_export_logreg_pointlookup_directio \
  --thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80,0.90 \
  --class-weight balanced \
  --c 1.0 \
  --solver lbfgs \
  --max-iter 1000 \
  --train-max-rows-per-run 50000 \
  --read-sample-count-lines \
  --oof-max-rows-per-run 10000

  这条命令的含义：
    最终模型每个 run 最多用 50000 行，54 个 run 大约最多 270万 行，足够训练 10 维逻辑回归。
    离线 OOF 指标每个 run 用 10000 行，避免 54 折全量训练拖死。
    如果还慢，把 --train-max-rows-per-run 改成 20000，--oof-max-rows-per-run 改成 5000。

  如果你想更稳，可以后面把：
    --train-max-rows-per-run 100000
    --oof-max-rows-per-run 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "label"
RUN_COLUMN = "_run_id"
DATASET_COLUMN = "_dataset_path"
WORKLOAD_COLUMN = "_workload"

DEFAULT_DATA_ROOT = Path("rocksdb_exp/confirm_v2_10m/10M")
DEFAULT_DATASET_GLOB = "**/cache_admission_dataset.csv"

ONLINE_LITE_FEATURES = [
    "block_size",
    "level",
    "l0_files",
    "l1_files",
    "estimate_pending_compaction_bytes",
    "num_running_compactions",
    "num_running_flushes",
    "cur_size_all_mem_tables",
    "block_cache_usage",
    "block_cache_pinned_usage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/export a logistic-regression admission model."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory containing datasets. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--dataset-glob",
        default=DEFAULT_DATASET_GLOB,
        help=f"Glob under --data-root. Default: {DEFAULT_DATASET_GLOB}",
    )
    parser.add_argument(
        "--dataset-files",
        nargs="*",
        default=[],
        help="Optional explicit dataset files. Overrides --data-root/--dataset-glob.",
    )
    parser.add_argument(
        "--include-workloads",
        default="",
        help="Optional comma-separated workload allowlist, inferred from path.",
    )
    parser.add_argument(
        "--exclude-workloads",
        default="",
        help="Optional comma-separated workload denylist, inferred from path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python/output/train_export_logreg"),
        help="Directory for exported artifacts.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.50,0.55,0.60",
        help="Comma-separated decision thresholds to evaluate.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Max iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["balanced", "none"],
        help="Class weighting strategy.",
    )
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=1,
        help="Skip leave-one-run-out fold when test rows are below this threshold.",
    )
    parser.add_argument(
        "--solver",
        default="liblinear",
        choices=["liblinear", "lbfgs"],
        help="LogisticRegression solver.",
    )
    parser.add_argument(
        "--oof-max-rows-per-run",
        type=int,
        default=0,
        help=(
            "Use at most this many rows per run for leave-one-run-out OOF "
            "evaluation. 0 means full OOF. Final exported model still trains "
            "on the full loaded training data unless --train-max-rows-per-run "
            "is set. Default: 0."
        ),
    )
    parser.add_argument(
        "--train-max-rows-per-run",
        type=int,
        default=0,
        help=(
            "Use at most this many rows per run for final model training. "
            "0 means use all rows. Default: 0."
        ),
    )
    parser.add_argument(
        "--read-sample-count-lines",
        action="store_true",
        help=(
            "When --train-max-rows-per-run is set, count CSV rows first and "
            "parse only randomly selected row numbers. This avoids parsing "
            "multi-GB CSV files just to sample them."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def find_dataset_files(args: argparse.Namespace) -> List[Path]:
    if args.dataset_files:
        paths = [Path(p).resolve() for p in args.dataset_files]
    else:
        paths = sorted(args.data_root.resolve().glob(args.dataset_glob))
    files = [p for p in paths if p.is_file()]
    if not files:
        raise FileNotFoundError("No dataset files found.")
    return files


def filter_dataset_paths_by_workload(
    paths: Sequence[Path],
    data_root: Path,
    include_workloads: Sequence[str],
    exclude_workloads: Sequence[str],
) -> List[Path]:
    filtered: List[Path] = []
    for path in paths:
        workload = infer_workload(path, data_root)
        if include_workloads and workload not in include_workloads:
            continue
        if exclude_workloads and workload in exclude_workloads:
            continue
        filtered.append(path)
    if not filtered:
        raise ValueError("No dataset files left after workload path filtering.")
    return filtered


def infer_run_id(dataset_path: Path, data_root: Path) -> str:
    try:
        rel = dataset_path.resolve().relative_to(data_root.resolve())
        return str(rel.parent).replace("\\", "/")
    except Exception:
        return dataset_path.parent.name


def infer_workload(dataset_path: Path, data_root: Path) -> str:
    try:
        rel = dataset_path.resolve().relative_to(data_root.resolve())
        parts = rel.parts
        if len(parts) >= 2:
            return parts[1]
        return parts[0] if parts else "unknown"
    except Exception:
        return "unknown"


def dataset_dtypes(feature_names: Sequence[str]) -> Dict[str, str]:
    dtypes = {name: "float32" for name in feature_names}
    dtypes[LABEL_COLUMN] = "int8"
    return dtypes


def stable_path_seed(path: Path, random_state: int) -> int:
    digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ int(random_state)) & 0xFFFFFFFF


def count_csv_data_rows(path: Path) -> int:
    total_lines = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            total_lines += chunk.count(b"\n")
    return max(total_lines - 1, 0)


def sampled_skiprows(path: Path, sample_size: int, random_state: int):
    total_rows = count_csv_data_rows(path)
    if sample_size <= 0 or total_rows <= sample_size:
        return None, total_rows
    rng = np.random.default_rng(stable_path_seed(path, random_state))
    # pandas passes row index 0 for the header, and 1..N for data rows.
    keep_rows = set(
        int(row_idx) for row_idx in rng.choice(total_rows, size=sample_size, replace=False) + 1
    )

    def should_skip(row_idx: int) -> bool:
        return row_idx != 0 and row_idx not in keep_rows

    return should_skip, total_rows


def load_datasets(
    paths: Sequence[Path],
    data_root: Path,
    feature_names: Sequence[str],
    random_state: int,
    train_max_rows_per_run: int = 0,
    read_sample_count_lines: bool = False,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    required_columns = set(feature_names) | {LABEL_COLUMN}
    dtypes = dataset_dtypes(feature_names)
    for idx, path in enumerate(paths, start=1):
        print(f"[LOAD] {idx}/{len(paths)} {path}", flush=True)
        skiprows = None
        if train_max_rows_per_run > 0 and read_sample_count_lines:
            skiprows, total_rows = sampled_skiprows(
                path, train_max_rows_per_run, random_state=random_state
            )
            if skiprows is not None:
                print(
                    f"[LOAD] sampling {train_max_rows_per_run}/{total_rows} rows from {path.name}",
                    flush=True,
                )
        df = pd.read_csv(
            path,
            usecols=lambda col: col in required_columns,
            dtype=dtypes,
            low_memory=False,
            skiprows=skiprows,
        )
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"{path} missing required column: {LABEL_COLUMN}")
        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            raise ValueError(f"{path} missing feature columns: {missing_features}")
        if (
            train_max_rows_per_run > 0
            and not read_sample_count_lines
            and len(df) > train_max_rows_per_run
        ):
            df = df.sample(n=train_max_rows_per_run, random_state=random_state)
        df[RUN_COLUMN] = infer_run_id(path, data_root)
        df[DATASET_COLUMN] = str(path)
        df[WORKLOAD_COLUMN] = infer_workload(path, data_root)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged[LABEL_COLUMN] = pd.to_numeric(merged[LABEL_COLUMN], errors="coerce")
    merged = merged.dropna(subset=[LABEL_COLUMN]).copy()
    merged[LABEL_COLUMN] = merged[LABEL_COLUMN].astype(int)
    return merged


def filter_workloads(
    df: pd.DataFrame, include_workloads: Sequence[str], exclude_workloads: Sequence[str]
) -> pd.DataFrame:
    filtered = df
    if include_workloads:
        filtered = filtered[filtered[WORKLOAD_COLUMN].isin(include_workloads)]
    if exclude_workloads:
        filtered = filtered[~filtered[WORKLOAD_COLUMN].isin(exclude_workloads)]
    if filtered.empty:
        raise ValueError("No rows left after workload filtering.")
    return filtered.copy()


def to_numeric_if_present(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")


def sample_rows_per_run(
    df: pd.DataFrame, max_rows_per_run: int, random_state: int
) -> pd.DataFrame:
    if max_rows_per_run <= 0:
        return df
    sampled = (
        df.groupby(RUN_COLUMN, group_keys=False, sort=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_rows_per_run),
                random_state=random_state,
            )
        )
        .reset_index(drop=True)
    )
    return sampled


def make_pipeline(args: argparse.Namespace) -> Pipeline:
    class_weight = None if args.class_weight == "none" else args.class_weight
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=args.c,
                    max_iter=args.max_iter,
                    class_weight=class_weight,
                    solver=args.solver,
                    random_state=args.random_state,
                ),
            ),
        ]
    )


def evaluate_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / len(y_true) if len(y_true) else float("nan")
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_leave_one_run_out(
    df: pd.DataFrame, feature_names: Sequence[str], args: argparse.Namespace
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x = df[list(feature_names)].copy()
    y = df[LABEL_COLUMN].to_numpy()
    groups = df[RUN_COLUMN].to_numpy()

    logo = LeaveOneGroupOut()
    rows: List[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(x, y, groups), start=1):
        if len(test_idx) < args.min_test_rows:
            continue
        print(
            f"[OOF] fold={fold_idx} train_rows={len(train_idx)} test_rows={len(test_idx)}",
            flush=True,
        )
        pipe = make_pipeline(args)
        pipe.fit(x.iloc[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(x.iloc[test_idx])[:, 1]

        fold_df = df.iloc[test_idx][[RUN_COLUMN, DATASET_COLUMN, WORKLOAD_COLUMN]].copy()
        fold_df["fold"] = fold_idx
        fold_df["y_true"] = y[test_idx]
        fold_df["y_prob"] = y_prob
        rows.append(fold_df)

    if not rows:
        raise ValueError("No valid folds produced.")

    oof = pd.concat(rows, ignore_index=True)
    y_true = oof["y_true"].to_numpy()
    y_prob = oof["y_prob"].to_numpy()

    summary = {
        "num_rows": float(len(df)),
        "num_runs": float(df[RUN_COLUMN].nunique()),
        "num_workloads": float(df[WORKLOAD_COLUMN].nunique()),
        "label_pos_ratio": float(df[LABEL_COLUMN].mean()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    return oof, summary


def fit_final_model(
    df: pd.DataFrame, feature_names: Sequence[str], args: argparse.Namespace
) -> Pipeline:
    pipe = make_pipeline(args)
    pipe.fit(df[list(feature_names)], df[LABEL_COLUMN].to_numpy())
    return pipe


def export_params(
    pipe: Pipeline, feature_names: Sequence[str]
) -> Dict[str, object]:
    scaler: StandardScaler = pipe.named_steps["scaler"]
    model: LogisticRegression = pipe.named_steps["model"]
    return {
        "feature_names": list(feature_names),
        "intercept": float(model.intercept_[0]),
        "means": [float(x) for x in scaler.mean_.tolist()],
        "scales": [float(x if x != 0.0 else 1.0) for x in scaler.scale_.tolist()],
        "weights": [float(x) for x in model.coef_[0].tolist()],
    }


def format_cpp_array(values: Sequence[float], indent: str = "      ") -> str:
    parts = [f"{v:.17g}" for v in values]
    lines: List[str] = []
    for i in range(0, len(parts), 3):
        chunk = ", ".join(parts[i : i + 3])
        suffix = "," if i + 3 < len(parts) else ""
        lines.append(f"{indent}{chunk}{suffix}")
    return "\n".join(lines)


def build_cpp_snippet(params: Dict[str, object]) -> str:
    features = params["feature_names"]
    comments = "\n".join(
        f"  // [{idx}] {name}" for idx, name in enumerate(features)
    )
    return (
        "// Auto-generated by python/scripts/train_export_logreg.py\n"
        "// Feature order must match MLCacheAdmissionFeatures.\n"
        f"{comments}\n"
        f"  static constexpr double kIntercept = {params['intercept']:.17g};\n\n"
        "  static constexpr double kMeans[] = {\n"
        f"{format_cpp_array(params['means'])}}};\n\n"
        "  static constexpr double kScales[] = {\n"
        f"{format_cpp_array(params['scales'])}}};\n\n"
        "  static constexpr double kWeights[] = {\n"
        f"{format_cpp_array(params['weights'])}}};\n"
    )


def write_report(
    output_dir: Path,
    summary: Dict[str, float],
    threshold_metrics: pd.DataFrame,
    params: Dict[str, object],
    workloads: Sequence[str],
    dataset_files: Sequence[Path],
) -> None:
    report_path = output_dir / "report.md"
    best_row = threshold_metrics.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Logistic Regression Export Report\n\n")
        f.write("## Data\n\n")
        f.write(f"- datasets: `{len(dataset_files)}`\n")
        f.write(f"- workloads: `{', '.join(workloads)}`\n")
        f.write(f"- rows: `{int(summary['num_rows'])}`\n")
        if "num_oof_rows" in summary:
            f.write(f"- oof_rows: `{int(summary['num_oof_rows'])}`\n")
        f.write(f"- runs: `{int(summary['num_runs'])}`\n")
        f.write(f"- positive_ratio: `{summary['label_pos_ratio']:.6f}`\n\n")
        f.write("## Pooled OOF Metrics\n\n")
        f.write(f"- roc_auc: `{summary['roc_auc']:.6f}`\n")
        f.write(f"- pr_auc: `{summary['pr_auc']:.6f}`\n")
        f.write(
            f"- best_threshold_by_f1: `{best_row['threshold']:.2f}` "
            f"(f1=`{best_row['f1']:.6f}`, precision=`{best_row['precision']:.6f}`, "
            f"recall=`{best_row['recall']:.6f}`)\n\n"
        )
        f.write("## Feature Order\n\n")
        for idx, name in enumerate(params["feature_names"]):
            f.write(f"- `{idx}`: `{name}`\n")
        f.write("\n## Artifacts\n\n")
        f.write("- `final_model_params.json`\n")
        f.write("- `ml_cache_admission_params.inc`\n")
        f.write("- `oof_predictions.csv`\n")
        f.write("- `threshold_metrics.csv`\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root.resolve()
    include_workloads = parse_csv_list(args.include_workloads)
    exclude_workloads = parse_csv_list(args.exclude_workloads)
    thresholds = [float(x) for x in parse_csv_list(args.thresholds)]

    all_dataset_files = find_dataset_files(args)
    dataset_files = filter_dataset_paths_by_workload(
        all_dataset_files, data_root, include_workloads, exclude_workloads
    )
    print(
        f"[INFO] datasets={len(dataset_files)} "
        f"(found={len(all_dataset_files)}) output_dir={output_dir}",
        flush=True,
    )
    df = load_datasets(
        dataset_files,
        data_root,
        ONLINE_LITE_FEATURES,
        random_state=args.random_state,
        train_max_rows_per_run=args.train_max_rows_per_run,
        read_sample_count_lines=args.read_sample_count_lines,
    )
    df = filter_workloads(df, include_workloads, exclude_workloads)
    to_numeric_if_present(df, ONLINE_LITE_FEATURES)
    print(
        f"[INFO] loaded rows={len(df)} runs={df[RUN_COLUMN].nunique()} "
        f"workloads={','.join(sorted(df[WORKLOAD_COLUMN].unique().tolist()))} "
        f"positive_ratio={df[LABEL_COLUMN].mean():.6f}",
        flush=True,
    )

    oof_df = sample_rows_per_run(
        df, args.oof_max_rows_per_run, random_state=args.random_state
    )
    if args.oof_max_rows_per_run > 0:
        print(
            f"[INFO] OOF sampled rows={len(oof_df)} "
            f"max_rows_per_run={args.oof_max_rows_per_run}",
            flush=True,
        )
    oof, summary = run_leave_one_run_out(oof_df, ONLINE_LITE_FEATURES, args)
    summary["num_rows"] = float(len(df))
    summary["num_oof_rows"] = float(len(oof_df))
    summary["num_runs"] = float(df[RUN_COLUMN].nunique())
    summary["label_pos_ratio"] = float(df[LABEL_COLUMN].mean())
    y_true = oof["y_true"].to_numpy()
    y_prob = oof["y_prob"].to_numpy()

    threshold_rows = [evaluate_at_threshold(y_true, y_prob, thr) for thr in thresholds]
    threshold_metrics = pd.DataFrame(threshold_rows).sort_values("threshold")
    threshold_metrics.to_csv(output_dir / "threshold_metrics.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    final_pipe = fit_final_model(df, ONLINE_LITE_FEATURES, args)
    params = export_params(final_pipe, ONLINE_LITE_FEATURES)
    with (output_dir / "final_model_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    (output_dir / "ml_cache_admission_params.inc").write_text(
        build_cpp_snippet(params), encoding="utf-8"
    )

    workloads = sorted(df[WORKLOAD_COLUMN].unique().tolist())
    write_report(output_dir, summary, threshold_metrics, params, workloads, dataset_files)

    print(f"Wrote: {output_dir / 'final_model_params.json'}")
    print(f"Wrote: {output_dir / 'ml_cache_admission_params.inc'}")
    print(f"Wrote: {output_dir / 'oof_predictions.csv'}")
    print(f"Wrote: {output_dir / 'threshold_metrics.csv'}")
    print(f"Wrote: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
