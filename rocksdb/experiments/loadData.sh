#重新生成 pristine DB
set -euo pipefail

DB_BENCH=/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench
DATA_ROOT=/yuejData/rocksdb_exp
TS=$(date +%Y%m%d_%H%M%S)
CACHE_INDEX_AND_FILTER_BLOCKS="${CACHE_INDEX_AND_FILTER_BLOCKS:-false}"
USE_DIRECT_READS="${USE_DIRECT_READS:-true}"
USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION="${USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION:-true}"

mkdir -p "$DATA_ROOT"

if [ -d "$DATA_ROOT/db_10m_pristine" ]; then
  mv "$DATA_ROOT/db_10m_pristine" "$DATA_ROOT/db_10m_pristine_backup_$TS"
fi

if [ -d "$DATA_ROOT/db_30m_pristine" ]; then
  mv "$DATA_ROOT/db_30m_pristine" "$DATA_ROOT/db_30m_pristine_backup_$TS"
fi

"$DB_BENCH" \
  --benchmarks=fillrandom \
  --db="$DATA_ROOT/db_10m_pristine" \
  --num=10000000 \
  --threads=16 \
  --key_size=20 \
  --value_size=100 \
  --cache_index_and_filter_blocks="$CACHE_INDEX_AND_FILTER_BLOCKS" \
  --use_direct_reads="$USE_DIRECT_READS" \
  --use_direct_io_for_flush_and_compaction="$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION" \
  --compression_type=none \
  --disable_wal=1 \
  --seed=101 \
  --statistics=1 \
  --histogram=1 \
  > "$DATA_ROOT/db_10m_pristine_build.log" \
  2> "$DATA_ROOT/db_10m_pristine_build.err"

"$DB_BENCH" \
  --benchmarks=fillrandom \
  --db="$DATA_ROOT/db_30m_pristine" \
  --num=30000000 \
  --threads=16 \
  --key_size=20 \
  --value_size=100 \
  --cache_index_and_filter_blocks="$CACHE_INDEX_AND_FILTER_BLOCKS" \
  --use_direct_reads="$USE_DIRECT_READS" \
  --use_direct_io_for_flush_and_compaction="$USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION" \
  --compression_type=none \
  --disable_wal=1 \
  --seed=101 \
  --statistics=1 \
  --histogram=1 \
  > "$DATA_ROOT/db_30m_pristine_build.log" \
  2> "$DATA_ROOT/db_30m_pristine_build.err"
