#!/usr/bin/env bash
set -euo pipefail

# Copy every cache_admission_dataset.csv under final_paper_matrix_directio while
# preserving the directory tree. The result is:
#   /yueData5T/final_paper_matrix_directio/...

SRC_ROOT="${SRC_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix_directio}"
DEST_PARENT="${DEST_PARENT:-/yueData5T}"
DEST_ROOT="${DEST_ROOT:-$DEST_PARENT/$(basename "$SRC_ROOT")}"

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "[ERR] source directory does not exist: $SRC_ROOT" >&2
  exit 1
fi

mkdir -p "$DEST_ROOT"

echo "[INFO] source: $SRC_ROOT"
echo "[INFO] target: $DEST_ROOT"

dir_count=0
while IFS= read -r -d '' dir; do
  rel="${dir#$SRC_ROOT}"
  mkdir -p "$DEST_ROOT$rel"
  ((dir_count += 1))
done < <(find "$SRC_ROOT" -type d -print0)

file_count=0
while IFS= read -r -d '' file; do
  rel="${file#$SRC_ROOT/}"
  dest="$DEST_ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -p "$file" "$dest"
  ((file_count += 1))
done < <(find "$SRC_ROOT" -type f -name 'cache_admission_dataset.csv' -print0)

if [[ "$file_count" -eq 0 ]]; then
  echo "[WARN] no cache_admission_dataset.csv files found under $SRC_ROOT" >&2
fi

echo "[DONE] created $dir_count directories and copied $file_count dataset files."
