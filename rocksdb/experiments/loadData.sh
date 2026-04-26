#!/usr/bin/env bash
set -euo pipefail

# Build pristine DBs for the final cache-admission experiments.
#
# Main experiment:
#   K=24, V=400, DB=20M/40M
#
# Scale robustness experiment:
#   K=24, V=400, DB=30M
#
# Output DB layout:
#   $DATA_ROOT/db_k24_v400_20m_pristine
#   $DATA_ROOT/db_k24_v400_30m_pristine
#   $DATA_ROOT/db_k24_v400_40m_pristine

DB_BENCH="${DB_BENCH:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench}"
BUILD_DIR="${BUILD_DIR:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release}"
DATA_ROOT="${DATA_ROOT:-/yuejData/rocksdb_exp}"

LOAD_THREADS="${LOAD_THREADS:-16}"
LOAD_SEED="${LOAD_SEED:-101}"
REBUILD_BINARIES="${REBUILD_BINARIES:-1}"
BACKUP_EXISTING="${BACKUP_EXISTING:-1}"
FORCE_REBUILD_DB="${FORCE_REBUILD_DB:-1}"

CACHE_INDEX_AND_FILTER_BLOCKS="${CACHE_INDEX_AND_FILTER_BLOCKS:-false}"
USE_DIRECT_READS="${USE_DIRECT_READS:-true}"
USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION="${USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION:-true}"
COMPRESSION_TYPE="${COMPRESSION_TYPE:-none}"
DISABLE_WAL="${DISABLE_WAL:-1}"
STATISTICS="${STATISTICS:-1}"
HISTOGRAM="${HISTOGRAM:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
MANIFEST_CSV="$DATA_ROOT/pristine_db_manifest.csv"

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

backup_or_remove_existing_db() {
  local db_path="$1"
  if [[ ! -d "$db_path" ]]; then
    return 0
  fi

  if [[ "$FORCE_REBUILD_DB" != "1" ]]; then
    echo "[SKIP] existing DB: $db_path"
    echo "       set FORCE_REBUILD_DB=1 to rebuild it" >&2
    return 1
  fi

  if [[ "$BACKUP_EXISTING" == "1" ]]; then
    local backup_path="${db_path}_backup_${TS}"
    echo "[BACKUP] $db_path -> $backup_path"
    mv "$db_path" "$backup_path"
  else
    echo "[REMOVE] $db_path"
    rm -rf "$db_path"
  fi
}

build_one_db() {
  local label="$1"
  local num="$2"
  local key_size="$3"
  local value_size="$4"
  local db_path="$DATA_ROOT/db_${label}_pristine"
  local log_prefix="$DATA_ROOT/db_${label}_pristine_build"

  if ! backup_or_remove_existing_db "$db_path"; then
    return 0
  fi

  mkdir -p "$db_path"
  echo "[LOAD] label=$label num=$num key_size=$key_size value_size=$value_size db=$db_path"

  {
    printf '#!/usr/bin/env bash\n'
    printf '"%s" \\\n' "$DB_BENCH"
    printf '  --benchmarks=fillrandom \\\n'
    printf '  --db=%q \\\n' "$db_path"
    printf '  --use_existing_db=false \\\n'
    printf '  --num=%q \\\n' "$num"
    printf '  --threads=%q \\\n' "$LOAD_THREADS"
    printf '  --key_size=%q \\\n' "$key_size"
    printf '  --value_size=%q \\\n' "$value_size"
    printf '  --cache_index_and_filter_blocks=%q \\\n' "$CACHE_INDEX_AND_FILTER_BLOCKS"
    printf '  --use_direct_reads=%q \\\n' "$USE_DIRECT_READS"
    printf '  --use_direct_io_for_flush_and_compaction=%q \\\n' "$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION"
    printf '  --compression_type=%q \\\n' "$COMPRESSION_TYPE"
    printf '  --disable_wal=%q \\\n' "$DISABLE_WAL"
    printf '  --seed=%q \\\n' "$LOAD_SEED"
    printf '  --statistics=%q \\\n' "$STATISTICS"
    printf '  --histogram=%q\n' "$HISTOGRAM"
  } > "${log_prefix}.command.sh"
  chmod +x "${log_prefix}.command.sh"

  "$DB_BENCH" \
    --benchmarks=fillrandom \
    --db="$db_path" \
    --use_existing_db=false \
    --num="$num" \
    --threads="$LOAD_THREADS" \
    --key_size="$key_size" \
    --value_size="$value_size" \
    --cache_index_and_filter_blocks="$CACHE_INDEX_AND_FILTER_BLOCKS" \
    --use_direct_reads="$USE_DIRECT_READS" \
    --use_direct_io_for_flush_and_compaction="$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION" \
    --compression_type="$COMPRESSION_TYPE" \
    --disable_wal="$DISABLE_WAL" \
    --seed="$LOAD_SEED" \
    --statistics="$STATISTICS" \
    --histogram="$HISTOGRAM" \
    > "${log_prefix}.log" \
    2> "${log_prefix}.err"

  echo "$label,$num,$key_size,$value_size,$db_path,${log_prefix}.log,${log_prefix}.err" >> "$MANIFEST_CSV"
  echo "[DONE] $label"
}

mkdir -p "$DATA_ROOT"

if [[ "$REBUILD_BINARIES" == "1" ]]; then
  require_dir "$BUILD_DIR"
  (
    cd "$BUILD_DIR"
    make db_bench -j20
  )
fi
require_file "$DB_BENCH"

echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] LOAD_THREADS=$LOAD_THREADS LOAD_SEED=$LOAD_SEED"
echo "[INFO] cache_index_and_filter_blocks=$CACHE_INDEX_AND_FILTER_BLOCKS use_direct_reads=$USE_DIRECT_READS use_direct_io_for_flush_and_compaction=$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION compression=$COMPRESSION_TYPE"
echo "[INFO] FORCE_REBUILD_DB=$FORCE_REBUILD_DB BACKUP_EXISTING=$BACKUP_EXISTING"

echo "label,num,key_size,value_size,db_path,stdout_log,stderr_log" > "$MANIFEST_CSV"

#build_one_db "k24_v400_20m" 20000000 24 400
build_one_db "k24_v400_30m" 30000000 24 400
build_one_db "k24_v400_40m" 40000000 24 400

echo "[DONE] manifest=$MANIFEST_CSV"
