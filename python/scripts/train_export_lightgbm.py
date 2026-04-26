#!/usr/bin/env python3
"""
Train a LightGBM cache-admission model for offline comparison.

This script intentionally uses the same lightweight online feature set as
train_export_logreg.py, so the offline result is directly comparable with the
current deployed logistic-regression gate. It does not emit a RocksDB .inc
deployment file yet; tree-model deployment should be decided only after the
offline metrics are clearly better.

Outputs:
- lightgbm_model.txt: LightGBM text model
- lightgbm_model.json: LightGBM dumped tree structure
- final_model_params.json: feature order and hyperparameters
- feature_importance.csv: gain/split importances
- oof_predictions.csv: grouped out-of-fold predictions
- threshold_metrics.csv: pooled threshold metrics
- report.md: compact training/evaluation summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


LABEL_COLUMN = "label"
BENEFIT_SCORE_COLUMN = "benefit_score"
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
    "block_cache_capacity",
    "block_cache_usage",
    "block_cache_pinned_usage",
    "block_cache_usage_ratio",
    "block_cache_pinned_usage_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM admission model for offline evaluation."
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
        default=Path("python/output/train_export_lightgbm"),
        help="Directory for exported artifacts.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated decision thresholds to evaluate.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=15,
        help="LightGBM num_leaves. Keep small for possible online deployment.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="LightGBM max_depth. Keep small for possible online deployment.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=80,
        help="Number of boosting trees.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Boosting learning rate.",
    )
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=200,
        help="Minimum samples in a leaf.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsampling fraction.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Feature subsampling fraction.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 regularization.",
    )
    parser.add_argument(
        "--class-weight",
        default="balanced",
        choices=["balanced", "none"],
        help="Class weighting strategy.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        default="none",
        choices=["none", "positive_benefit_log", "positive_benefit_linear"],
        help=(
            "Optional training-time sample weighting. positive_benefit_log "
            "upweights positive rows by log1p(benefit_score) so the model "
            "focuses on high-benefit cached blocks. Default: none."
        ),
    )
    parser.add_argument(
        "--sample-weight-max",
        type=float,
        default=8.0,
        help="Clip sample weights to this max before mean-normalization. Default: 8.0.",
    )
    parser.add_argument(
        "--oof-mode",
        default="group-kfold",
        choices=["group-kfold", "leave-one-run-out", "none"],
        help="Grouped out-of-fold evaluation mode.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits when --oof-mode=group-kfold.",
    )
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=1,
        help="Skip OOF fold when test rows are below this threshold.",
    )
    parser.add_argument(
        "--oof-max-rows-per-run",
        type=int,
        default=30000,
        help="Use at most this many rows per run for OOF evaluation. 0 means full OOF.",
    )
    parser.add_argument(
        "--train-max-rows-per-run",
        type=int,
        default=200000,
        help="Use at most this many rows per run for final model training. 0 means full.",
    )
    parser.add_argument(
        "--read-sample-count-lines",
        action="store_true",
        help=(
            "When --train-max-rows-per-run is set, count CSV rows first and "
            "parse only randomly selected row numbers."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="LightGBM training threads.",
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
        if len(parts) >= 2:
            return parts[1]
        return parts[0] if parts else "unknown"
    except Exception:
        return "unknown"


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


def dataset_dtypes(
    feature_names: Sequence[str], extra_columns: Sequence[str]
) -> Dict[str, str]:
    dtypes = {name: "float32" for name in feature_names}
    for name in extra_columns:
        dtypes[name] = "float32"
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
    keep_rows = set(
        int(row_idx)
        for row_idx in rng.choice(total_rows, size=sample_size, replace=False) + 1
    )

    def should_skip(row_idx: int) -> bool:
        return row_idx != 0 and row_idx not in keep_rows

    return should_skip, total_rows


def load_datasets(
    paths: Sequence[Path],
    data_root: Path,
    feature_names: Sequence[str],
    extra_columns: Sequence[str],
    random_state: int,
    train_max_rows_per_run: int = 0,
    read_sample_count_lines: bool = False,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    required_columns = set(feature_names) | set(extra_columns) | {LABEL_COLUMN}
    dtypes = dataset_dtypes(feature_names, extra_columns)
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
        missing_extra = [col for col in extra_columns if col not in df.columns]
        if missing_extra:
            raise ValueError(f"{path} missing required extra columns: {missing_extra}")
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


def sample_weight_columns(args: argparse.Namespace) -> List[str]:
    if args.sample_weight_mode == "none":
        return []
    return [BENEFIT_SCORE_COLUMN]


def compute_sample_weights(
    df: pd.DataFrame, args: argparse.Namespace
) -> Optional[np.ndarray]:
    if args.sample_weight_mode == "none":
        return None
    if BENEFIT_SCORE_COLUMN not in df.columns:
        raise ValueError(
            f"--sample-weight-mode={args.sample_weight_mode} requires {BENEFIT_SCORE_COLUMN}"
        )

    y = df[LABEL_COLUMN].to_numpy()
    benefit = pd.to_numeric(df[BENEFIT_SCORE_COLUMN], errors="coerce").fillna(0.0)
    benefit = benefit.clip(lower=0.0).to_numpy(dtype=float)
    weights = np.ones(len(df), dtype=float)
    positive_mask = y == 1
    if args.sample_weight_mode == "positive_benefit_log":
        weights[positive_mask] = 1.0 + np.log1p(benefit[positive_mask])
    elif args.sample_weight_mode == "positive_benefit_linear":
        weights[positive_mask] = 1.0 + benefit[positive_mask]
    else:
        raise ValueError(f"Unsupported sample_weight_mode: {args.sample_weight_mode}")

    weights = np.clip(weights, 1.0, args.sample_weight_max)
    mean = float(weights.mean())
    if mean > 0:
        weights /= mean
    return weights


def make_model(args: argparse.Namespace) -> LGBMClassifier:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - exercised on server env.
        raise SystemExit(
            "Missing dependency: lightgbm. Install it in the training environment, "
            "for example: pip install lightgbm"
        ) from exc

    class_weight = None if args.class_weight == "none" else args.class_weight
    return LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1 if args.subsample < 1.0 else 0,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        class_weight=class_weight,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbosity=-1,
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


def build_group_splits(
    x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, args: argparse.Namespace
):
    unique_groups = np.unique(groups)
    if args.oof_mode == "none":
        return []
    if args.oof_mode == "leave-one-run-out":
        return list(LeaveOneGroupOut().split(x, y, groups))
    n_splits = min(args.n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two groups for grouped OOF evaluation.")
    return list(GroupKFold(n_splits=n_splits).split(x, y, groups))


def run_grouped_oof(
    df: pd.DataFrame, feature_names: Sequence[str], args: argparse.Namespace
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x = df[list(feature_names)].copy()
    y = df[LABEL_COLUMN].to_numpy()
    groups = df[RUN_COLUMN].to_numpy()
    splits = build_group_splits(x, y, groups, args)
    if not splits:
        raise ValueError("OOF evaluation disabled; no OOF metrics to report.")

    rows: List[pd.DataFrame] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(test_idx) < args.min_test_rows:
            continue
        print(
            f"[OOF] fold={fold_idx} train_rows={len(train_idx)} test_rows={len(test_idx)}",
            flush=True,
        )
        model = make_model(args)
        model.fit(
            x.iloc[train_idx],
            y[train_idx],
            sample_weight=compute_sample_weights(df.iloc[train_idx], args),
        )
        y_prob = model.predict_proba(x.iloc[test_idx])[:, 1]
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
        "sample_weight_mode": args.sample_weight_mode,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    return oof, summary


def fit_final_model(
    df: pd.DataFrame, feature_names: Sequence[str], args: argparse.Namespace
) -> LGBMClassifier:
    model = make_model(args)
    model.fit(
        df[list(feature_names)],
        df[LABEL_COLUMN].to_numpy(),
        sample_weight=compute_sample_weights(df, args),
    )
    return model


def write_feature_importance(
    model: LGBMClassifier, feature_names: Sequence[str], output_dir: Path
) -> None:
    booster = model.booster_
    rows = []
    for name, gain, split in zip(
        feature_names,
        booster.feature_importance(importance_type="gain"),
        booster.feature_importance(importance_type="split"),
    ):
        rows.append({"feature": name, "importance_gain": gain, "importance_split": split})
    pd.DataFrame(rows).sort_values(
        ["importance_gain", "importance_split"], ascending=[False, False]
    ).to_csv(output_dir / "feature_importance.csv", index=False)


def model_metadata(
    model: LGBMClassifier, feature_names: Sequence[str], args: argparse.Namespace
) -> Dict[str, object]:
    return {
        "model_type": "lightgbm",
        "feature_names": list(feature_names),
        "objective": "binary",
        "num_trees": int(model.booster_.num_trees()),
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "class_weight": args.class_weight,
        "sample_weight_mode": args.sample_weight_mode,
        "sample_weight_max": args.sample_weight_max,
    }


def write_report(
    output_dir: Path,
    summary: Dict[str, float],
    threshold_metrics: pd.DataFrame,
    params: Dict[str, object],
    workloads: Sequence[str],
    dataset_files: Sequence[Path],
    args: argparse.Namespace,
) -> None:
    report_path = output_dir / "report.md"
    best_row = threshold_metrics.sort_values(
        ["f1", "threshold"], ascending=[False, True]
    ).iloc[0]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# LightGBM Training Report\n\n")
        f.write("## Data\n\n")
        f.write(f"- datasets: `{len(dataset_files)}`\n")
        f.write(f"- workloads: `{', '.join(workloads)}`\n")
        f.write(f"- rows: `{int(summary['num_rows'])}`\n")
        if "num_oof_rows" in summary:
            f.write(f"- oof_rows: `{int(summary['num_oof_rows'])}`\n")
        f.write(f"- runs: `{int(summary['num_runs'])}`\n")
        f.write(f"- positive_ratio: `{summary['label_pos_ratio']:.6f}`\n")
        f.write(f"- sample_weight_mode: `{summary.get('sample_weight_mode', 'none')}`\n\n")

        f.write("## Pooled OOF Metrics\n\n")
        f.write(f"- oof_mode: `{args.oof_mode}`\n")
        f.write(f"- roc_auc: `{summary['roc_auc']:.6f}`\n")
        f.write(f"- pr_auc: `{summary['pr_auc']:.6f}`\n")
        f.write(
            f"- best_threshold_by_f1: `{best_row['threshold']:.2f}` "
            f"(f1=`{best_row['f1']:.6f}`, precision=`{best_row['precision']:.6f}`, "
            f"recall=`{best_row['recall']:.6f}`)\n\n"
        )

        f.write("## Model\n\n")
        for key in [
            "num_trees",
            "num_leaves",
            "max_depth",
            "n_estimators",
            "learning_rate",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "class_weight",
        ]:
            f.write(f"- {key}: `{params[key]}`\n")

        f.write("\n## Feature Order\n\n")
        for idx, name in enumerate(params["feature_names"]):
            f.write(f"- `{idx}`: `{name}`\n")

        f.write("\n## Artifacts\n\n")
        f.write("- `lightgbm_model.txt`\n")
        f.write("- `lightgbm_model.json`\n")
        f.write("- `final_model_params.json`\n")
        f.write("- `feature_importance.csv`\n")
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
        sample_weight_columns(args),
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
    oof, summary = run_grouped_oof(oof_df, ONLINE_LITE_FEATURES, args)
    summary["num_rows"] = float(len(df))
    summary["num_oof_rows"] = float(len(oof_df))
    summary["num_runs"] = float(df[RUN_COLUMN].nunique())
    summary["label_pos_ratio"] = float(df[LABEL_COLUMN].mean())
    summary["sample_weight_mode"] = args.sample_weight_mode

    y_true = oof["y_true"].to_numpy()
    y_prob = oof["y_prob"].to_numpy()
    threshold_rows = [evaluate_at_threshold(y_true, y_prob, thr) for thr in thresholds]
    threshold_metrics = pd.DataFrame(threshold_rows).sort_values("threshold")
    threshold_metrics.to_csv(output_dir / "threshold_metrics.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)

    final_model = fit_final_model(df, ONLINE_LITE_FEATURES, args)
    booster = final_model.booster_
    booster.save_model(str(output_dir / "lightgbm_model.txt"))
    with (output_dir / "lightgbm_model.json").open("w", encoding="utf-8") as f:
        json.dump(booster.dump_model(), f)
    params = model_metadata(final_model, ONLINE_LITE_FEATURES, args)
    with (output_dir / "final_model_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    write_feature_importance(final_model, ONLINE_LITE_FEATURES, output_dir)

    workloads = sorted(df[WORKLOAD_COLUMN].unique().tolist())
    write_report(
        output_dir,
        summary,
        threshold_metrics,
        params,
        workloads,
        dataset_files,
        args,
    )

    print(f"Wrote: {output_dir / 'lightgbm_model.txt'}")
    print(f"Wrote: {output_dir / 'lightgbm_model.json'}")
    print(f"Wrote: {output_dir / 'final_model_params.json'}")
    print(f"Wrote: {output_dir / 'feature_importance.csv'}")
    print(f"Wrote: {output_dir / 'oof_predictions.csv'}")
    print(f"Wrote: {output_dir / 'threshold_metrics.csv'}")
    print(f"Wrote: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
