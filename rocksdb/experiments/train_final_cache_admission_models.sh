#!/usr/bin/env bash
set -euo pipefail

# Train final cache-admission models from the main paper dataset.
#
# Logistic regression is the deployable online model because it exports
# ml_cache_admission_params.inc directly. LightGBM is trained as an offline
# comparison candidate; only deploy it if its gain justifies adding tree
# inference to RocksDB's hot path.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix_main_k24v400}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/qhsf5/yuej/patentProject2/python/output}"
INCLUDE_WORKLOADS="${INCLUDE_WORKLOADS:-readrandom,multireadrandom,readwhilewriting}"
THRESHOLDS="${THRESHOLDS:-0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80}"

TRAIN_MAX_ROWS_PER_RUN="${TRAIN_MAX_ROWS_PER_RUN:-200000}"
OOF_MAX_ROWS_PER_RUN="${OOF_MAX_ROWS_PER_RUN:-30000}"
SAMPLE_WEIGHT_MODE="${SAMPLE_WEIGHT_MODE:-positive_benefit_log}"
SAMPLE_WEIGHT_MAX="${SAMPLE_WEIGHT_MAX:-8.0}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"

TRAIN_LOGREG="${TRAIN_LOGREG:-1}"
TRAIN_LIGHTGBM="${TRAIN_LIGHTGBM:-1}"

LOGREG_OUT="${LOGREG_OUT:-$OUTPUT_ROOT/train_export_logreg_main_k24v400_13f_benefit200k}"
LIGHTGBM_OUT="${LIGHTGBM_OUT:-$OUTPUT_ROOT/train_export_lightgbm_main_k24v400_13f_benefit200k}"

LOGREG_SCRIPT="${LOGREG_SCRIPT:-/home/qhsf5/yuej/patentProject2/python/scripts/train_export_logreg.py}"
LIGHTGBM_SCRIPT="${LIGHTGBM_SCRIPT:-/home/qhsf5/yuej/patentProject2/python/scripts/train_export_lightgbm.py}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERR] missing DATA_ROOT: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$LOGREG_SCRIPT" ]]; then
  echo "[ERR] missing LOGREG_SCRIPT: $LOGREG_SCRIPT" >&2
  exit 1
fi
if [[ "$TRAIN_LIGHTGBM" == "1" && ! -f "$LIGHTGBM_SCRIPT" ]]; then
  echo "[ERR] missing LIGHTGBM_SCRIPT: $LIGHTGBM_SCRIPT" >&2
  exit 1
fi

echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] INCLUDE_WORKLOADS=$INCLUDE_WORKLOADS"
echo "[INFO] THRESHOLDS=$THRESHOLDS"
echo "[INFO] TRAIN_MAX_ROWS_PER_RUN=$TRAIN_MAX_ROWS_PER_RUN OOF_MAX_ROWS_PER_RUN=$OOF_MAX_ROWS_PER_RUN"
echo "[INFO] SAMPLE_WEIGHT_MODE=$SAMPLE_WEIGHT_MODE SAMPLE_WEIGHT_MAX=$SAMPLE_WEIGHT_MAX CLASS_WEIGHT=$CLASS_WEIGHT"

if [[ "$TRAIN_LOGREG" == "1" ]]; then
  echo "[TRAIN] logistic regression -> $LOGREG_OUT"
  "$PYTHON_BIN" "$LOGREG_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --include-workloads "$INCLUDE_WORKLOADS" \
    --output-dir "$LOGREG_OUT" \
    --thresholds "$THRESHOLDS" \
    --class-weight "$CLASS_WEIGHT" \
    --sample-weight-mode "$SAMPLE_WEIGHT_MODE" \
    --sample-weight-max "$SAMPLE_WEIGHT_MAX" \
    --c 1.0 \
    --solver lbfgs \
    --max-iter 1000 \
    --train-max-rows-per-run "$TRAIN_MAX_ROWS_PER_RUN" \
    --read-sample-count-lines \
    --oof-max-rows-per-run "$OOF_MAX_ROWS_PER_RUN"
fi

if [[ "$TRAIN_LIGHTGBM" == "1" ]]; then
  echo "[TRAIN] LightGBM -> $LIGHTGBM_OUT"
  "$PYTHON_BIN" "$LIGHTGBM_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --include-workloads "$INCLUDE_WORKLOADS" \
    --output-dir "$LIGHTGBM_OUT" \
    --thresholds "$THRESHOLDS" \
    --class-weight "$CLASS_WEIGHT" \
    --sample-weight-mode "$SAMPLE_WEIGHT_MODE" \
    --sample-weight-max "$SAMPLE_WEIGHT_MAX" \
    --num-leaves 15 \
    --max-depth 4 \
    --n-estimators 80 \
    --learning-rate 0.05 \
    --min-child-samples 200 \
    --subsample 0.8 \
    --colsample-bytree 0.9 \
    --reg-lambda 1.0 \
    --train-max-rows-per-run "$TRAIN_MAX_ROWS_PER_RUN" \
    --read-sample-count-lines \
    --oof-max-rows-per-run "$OOF_MAX_ROWS_PER_RUN" \
    --n-jobs 16
fi

echo "[DONE] training complete"
