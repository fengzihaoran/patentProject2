#重新生成 pristine DB
set -euo pipefail

DB_BENCH=/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench
DATA_ROOT=/yuejData/rocksdb_exp
TS=$(date +%Y%m%d_%H%M%S)

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
  --compression_type=none \
  --disable_wal=1 \
  --seed=101 \
  --statistics=1 \
  --histogram=1 \
  > "$DATA_ROOT/db_30m_pristine_build.log" \
  2> "$DATA_ROOT/db_30m_pristine_build.err"

