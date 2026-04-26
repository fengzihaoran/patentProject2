#!/usr/bin/env bash
set -euo pipefail

# Resume main final-paper trace collection without building datasets.
#
# This script runs db_bench and block_cache_trace_analyzer only. It does not
# call build_cache_admission_dataset.py because BUILD_DATASETS=0 below.
# Existing runs with block_trace.txt + snapshot.csv + sst_trace.tsv are skipped.
# readrandom/multireadrandom are collected for exp=2,4,6. readwhilewriting is
# collected once under exp_na because read_random_exp_range does not apply to it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DB_BENCH="${DB_BENCH:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/db_bench}" \
TRACE_ANALYZER="${TRACE_ANALYZER:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release/block_cache_trace_analyzer}" \
DATASET_BUILDER="${DATASET_BUILDER:-/home/qhsf5/yuej/patentProject2/python/scripts/build_cache_admission_dataset.py}" \
BUILD_DIR="${BUILD_DIR:-/home/qhsf5/yuej/patentProject2/rocksdb/cmake-build-release}" \
OUT_ROOT="${OUT_ROOT:-/yuejData/rocksdb_exp/final_paper_matrix_main_k24v400}" \
DB_PATHS_CSV="${DB_PATHS_CSV:-/yuejData/rocksdb_exp/db_k24_v400_20m_pristine,/yuejData/rocksdb_exp/db_k24_v400_40m_pristine}" \
DB_LABELS_CSV="${DB_LABELS_CSV:-k24_v400_20m,k24_v400_40m}" \
NUMS_CSV="${NUMS_CSV:-20000000,40000000}" \
WORKLOADS_CSV="${WORKLOADS_CSV:-readrandom,multireadrandom,readwhilewriting}" \
READ_RANDOM_EXP_RANGE_CSV="${READ_RANDOM_EXP_RANGE_CSV:-2,4,6}" \
CACHE_SIZES_CSV="${CACHE_SIZES_CSV:-33554432,134217728,268435456,536870912}" \
SEEDS_CSV="${SEEDS_CSV:-101,202,303}" \
THREADS="${THREADS:-16}" \
READ_ONLY_DURATION="${READ_ONLY_DURATION:-180}" \
MIXED_RW_DURATION="${MIXED_RW_DURATION:-300}" \
KEY_SIZE="${KEY_SIZE:-24}" \
VALUE_SIZE="${VALUE_SIZE:-400}" \
TRACE_LOADER="${TRACE_LOADER:-polars}" \
CACHE_INDEX_AND_FILTER_BLOCKS="${CACHE_INDEX_AND_FILTER_BLOCKS:-false}" \
USE_DIRECT_READS="${USE_DIRECT_READS:-true}" \
USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION="${USE_DIRECT_IO_FOR_FLUSH_AND_COMPACTION:-true}" \
REBUILD_BINARIES="${REBUILD_BINARIES:-0}" \
TRACE_SAMPLING_FREQUENCY="${TRACE_SAMPLING_FREQUENCY:-1}" \
COPY_DB_FOR_READ_ONLY="${COPY_DB_FOR_READ_ONLY:-0}" \
DEFER_TRACE_ANALYSIS="${DEFER_TRACE_ANALYSIS:-0}" \
KEEP_BLOCK_TRACE_BIN="${KEEP_BLOCK_TRACE_BIN:-0}" \
PRESERVE_RUN_DB="${PRESERVE_RUN_DB:-0}" \
BUILD_DATASETS=0 \
SKIP_COMPLETED_RUNS=1 \
"$SCRIPT_DIR/collect_final_paper_matrix.sh"
