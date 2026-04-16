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
"""

from __future__ import annotations

import argparse
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
        return parts[0] if parts else "unknown"
    except Exception:
        return "unknown"


def load_datasets(paths: Sequence[Path], data_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"{path} missing required column: {LABEL_COLUMN}")
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

    dataset_files = find_dataset_files(args)
    data_root = args.data_root.resolve()
    include_workloads = parse_csv_list(args.include_workloads)
    exclude_workloads = parse_csv_list(args.exclude_workloads)
    thresholds = [float(x) for x in parse_csv_list(args.thresholds)]

    df = load_datasets(dataset_files, data_root)
    df = filter_workloads(df, include_workloads, exclude_workloads)
    to_numeric_if_present(df, ONLINE_LITE_FEATURES)

    oof, summary = run_leave_one_run_out(df, ONLINE_LITE_FEATURES, args)
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
