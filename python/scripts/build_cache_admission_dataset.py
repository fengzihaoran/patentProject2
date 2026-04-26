#!/usr/bin/env python3
"""
Build a supervised-learning dataset for RocksDB block-cache admission.

Inputs:
1. Human-readable block cache trace converted from RocksDB block cache tracing.
2. Optional periodic LSM snapshots in CSV form.
3. Optional SST lineage TSV generated from flush/compaction listeners.

The output is one CSV row per candidate admission event:
  "user-accessed data block, cache miss, candidate for insertion".

This script is intentionally conservative:
- It only uses fields that are stable in RocksDB's public trace format.
- It treats an SST compaction input event as an invalidation signal.
- It keeps time-based joins simple and explicit.

Recommended workflow:
1. Collect block-cache trace during db_bench or your target workload.
2. Convert the binary trace into human-readable CSV-like text.
3. Collect 1s LSM snapshots in a separate CSV.
4. Collect SST lineage events if available.
5. Run this script to produce the first-pass training dataset.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


# RocksDB human-readable block cache trace fields, in order.
TRACE_FIELDS = [
    "access_ts_us",
    "block_id",
    "block_type",
    "block_size",
    "cf_id",
    "cf_name",
    "level",
    "sst_fd_number",
    "caller",
    "no_insert",
    "get_id",
    "get_key_id",
    "referenced_data_size",
    "is_cache_hit",
    "referenced_key_exist_in_block",
    "num_keys_in_block",
    "table_id",
    "sequence_number",
    "block_key_size",
    "referenced_key_size",
    "block_offset",
]

IDX_ACCESS_TS_US = 0
IDX_BLOCK_TYPE = 2
IDX_BLOCK_SIZE = 3
IDX_CF_ID = 4
IDX_CF_NAME = 5
IDX_LEVEL = 6
IDX_SST_FD_NUMBER = 7
IDX_CALLER = 8
IDX_NO_INSERT = 9
IDX_GET_ID = 10
IDX_REFERENCED_DATA_SIZE = 12
IDX_IS_CACHE_HIT = 13
IDX_REFERENCED_KEY_EXIST_IN_BLOCK = 14
IDX_NUM_KEYS_IN_BLOCK = 15
IDX_BLOCK_OFFSET = 20

TRACE_USED_COLUMNS = [
    "access_ts_us",
    "block_type",
    "block_size",
    "cf_id",
    "cf_name",
    "level",
    "sst_fd_number",
    "caller",
    "no_insert",
    "get_id",
    "referenced_data_size",
    "is_cache_hit",
    "referenced_key_exist_in_block",
    "num_keys_in_block",
    "block_offset",
]
TRACE_USED_COLUMN_INDICES = [
    IDX_ACCESS_TS_US,
    IDX_BLOCK_TYPE,
    IDX_BLOCK_SIZE,
    IDX_CF_ID,
    IDX_CF_NAME,
    IDX_LEVEL,
    IDX_SST_FD_NUMBER,
    IDX_CALLER,
    IDX_NO_INSERT,
    IDX_GET_ID,
    IDX_REFERENCED_DATA_SIZE,
    IDX_IS_CACHE_HIT,
    IDX_REFERENCED_KEY_EXIST_IN_BLOCK,
    IDX_NUM_KEYS_IN_BLOCK,
    IDX_BLOCK_OFFSET,
]

BASE_FIELDNAMES = [
    "ts_us",
    "cf_id",
    "cf_name",
    "sst_fd_number",
    "block_offset",
    "block_size",
    "level",
    "caller",
    "no_insert",
    "referenced_data_size",
    "referenced_key_exist_in_block",
    "num_keys_in_block",
    "recent_block_hits_10s",
    "recent_block_hits_60s",
    "recent_sst_hits_10s",
    "recent_sst_hits_60s",
    "recent_cf_hits_10s",
    "recent_cf_hits_60s",
    "recent_block_decay_10s",
    "recent_block_decay_60s",
    "recent_sst_decay_10s",
    "recent_sst_decay_60s",
    "recent_cf_decay_10s",
    "recent_cf_decay_60s",
    "future_reuse_raw_count",
    "future_reuse_unique_get_count",
    "future_reuse_count",
    "first_reuse_delta_us",
    "invalidated_within_horizon",
    "invalidation_delta_us",
    "survived_until_first_reuse",
    "benefit_score",
    "block_cache_usage_ratio",
    "block_cache_pinned_usage_ratio",
    "label",
]


# In this RocksDB checkout, TraceType::kBlockTraceDataBlock = 9.
# 7 and 8 are index/filter blocks, which are not the intended training target
# for this script.
TRACE_TYPE_DATA_BLOCK = 9

# TableReaderCaller enum values come from include/rocksdb/table_reader_caller.h.
# RocksDB uses 1-based values:
#   kUserGet = 1, kUserMultiGet = 2, kUserIterator = 3, ...
# Keep these aligned with upstream, otherwise whole workloads such as
# multireadrandom can be filtered out incorrectly.
CALLER_NAMES = {
    1: "user_get",
    2: "user_mget",
    3: "user_iterator",
    10: "compaction",
}

USER_CALLERS = {1, 2, 3}


@dataclass(frozen=True, slots=True)
class BlockKey:
    cf_id: int
    sst_fd_number: int
    block_offset: int


@dataclass(slots=True)
class TraceEvent:
    ts_us: int
    block: BlockKey
    block_type: int
    block_size: int
    cf_name: str
    level: int
    caller: int
    no_insert: int
    get_id: int
    is_cache_hit: int
    referenced_data_size: int
    referenced_key_exist_in_block: int
    num_keys_in_block: int


@dataclass(slots=True)
class SnapshotPoint:
    ts_us: int
    values: Dict[str, float]


@dataclass(slots=True)
class InvalidationEvent:
    ts_us: int
    cf_name: str
    input_files: List[Tuple[int, int]]


@dataclass(slots=True)
class TraceLoadStats:
    total_rows: int
    kept_events: int
    bad_rows: int
    block_type_counts: Counter[int]
    caller_counts: Counter[int]


@dataclass(slots=True)
class SstTraceLoadStats:
    sst_trace_rows: int
    invalidation_event_rows: int
    invalidation_file_refs: int


@dataclass(slots=True)
class DecayedCounter:
    value: float = 0.0
    last_ts_us: Optional[int] = None

    def score_at(self, ts_us: int, decay_us: int) -> float:
        if self.last_ts_us is None:
            return 0.0
        if decay_us <= 0:
            return self.value
        delta = max(ts_us - self.last_ts_us, 0)
        return self.value * math.exp(-delta / decay_us)

    def add(self, ts_us: int, decay_us: int) -> None:
        self.value = self.score_at(ts_us, decay_us) + 1.0
        self.last_ts_us = ts_us


@dataclass(slots=True)
class WriteStats:
    row_count: int
    label_pos: int
    label_neg: int
    build_dataset_sec: float = 0.0
    write_csv_sec: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cache-admission training dataset from RocksDB traces."
    )
    parser.add_argument(
        "--block-trace",
        required=True,
        help="Human-readable block cache trace file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file for training samples.",
    )
    parser.add_argument(
        "--snapshot-csv",
        default="",
        help="Optional periodic snapshot CSV joined by the latest timestamp <= event time.",
    )
    parser.add_argument(
        "--sst-trace-tsv",
        default="",
        help="Optional flush/compaction lineage TSV used to estimate invalidation risk.",
    )
    parser.add_argument(
        "--data-block-types",
        default=str(TRACE_TYPE_DATA_BLOCK),
        help=(
            "Comma-separated trace block_type values treated as data blocks. "
            f"Default: {TRACE_TYPE_DATA_BLOCK}."
        ),
    )
    parser.add_argument(
        "--trace-loader",
        choices=("csv", "polars"),
        default="polars",
        help=(
            "Trace parser backend. polars is faster on large traces and falls "
            "back to csv if unavailable or if strict parsing fails. Default: polars."
        ),
    )
    parser.add_argument(
        "--horizon-seconds",
        type=int,
        default=5,
        help="Prediction horizon in seconds. Default: 5.",
    )
    parser.add_argument(
        "--positive-reuse-threshold",
        type=int,
        default=8,
        help="Label is positive only if future_reuse_count >= threshold. Default: 8.",
    )
    parser.add_argument(
        "--candidate-cooldown-ms",
        type=int,
        default=3000,
        help=(
            "Emit at most one candidate sample for the same block within this "
            "cooldown window. Default: 3000."
        ),
    )
    parser.add_argument(
        "--max-first-reuse-seconds",
        type=int,
        default=3,
        help=(
            "Label is positive only if the first reuse happens within this many "
            "seconds. Default: 3."
        ),
    )
    parser.add_argument(
        "--min-benefit-score",
        type=float,
        default=0.05,
        help=(
            "Label is positive only if benefit_score >= threshold. "
            "benefit_score is future_reuse_count per KiB of cached block. "
            "Default: 0.05."
        ),
    )
    parser.add_argument(
        "--future-reuse-count-mode",
        choices=("unique_get", "access"),
        default="unique_get",
        help=(
            "How to count future reuse for labels. unique_get counts at most "
            "one reuse from the same get_id, reducing duplicate inflation in "
            "MultiGet and repeated block touches. access preserves the old raw "
            "event-count behavior. Default: unique_get."
        ),
    )
    parser.add_argument(
        "--include-no-insert",
        action="store_true",
        help=(
            "Include cache misses whose trace no_insert flag is set. Off by "
            "default because those misses are not real cache-admission decisions."
        ),
    )
    parser.add_argument(
        "--recent-short-seconds",
        type=int,
        default=10,
        help="Short historical window in seconds. Default: 10.",
    )
    parser.add_argument(
        "--recent-long-seconds",
        type=int,
        default=60,
        help="Long historical window in seconds. Default: 60.",
    )
    parser.add_argument(
        "--allow-iterator",
        action="store_true",
        help="Include iterator accesses as user accesses. Off by default for conservative labeling.",
    )
    return parser.parse_args()


def parse_int_set(raw: str) -> set[int]:
    values = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.add(int(token))
    if not values:
        raise ValueError("integer set cannot be empty")
    return values


def parse_int(raw: str) -> int:
    raw = raw.strip()
    if raw == "":
        return 0
    return int(raw)


def parse_float(raw: str) -> float:
    raw = raw.strip()
    if raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def load_block_trace_csv(
    path: Path, allow_iterator: bool, data_block_types: set[int]
) -> Tuple[List[TraceEvent], TraceLoadStats]:
    events: List[TraceEvent] = []
    allowed_callers = set(USER_CALLERS)
    if not allow_iterator:
        allowed_callers.discard(3)
    block_type_counts: Counter[int] = Counter()
    caller_counts: Counter[int] = Counter()
    total_rows = 0
    bad_rows = 0
    in_time_order = True
    prev_kept_ts_us = -1

    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        for row_num, row in enumerate(reader, start=1):
            if not row:
                continue
            total_rows += 1
            if len(row) != len(TRACE_FIELDS):
                bad_rows += 1
                continue
            try:
                block_type = parse_int(row[IDX_BLOCK_TYPE])
                caller = parse_int(row[IDX_CALLER])
            except ValueError:
                bad_rows += 1
                continue
            block_type_counts[block_type] += 1
            caller_counts[caller] += 1
            if block_type not in data_block_types:
                continue
            if caller not in allowed_callers:
                continue
            try:
                ts_us = parse_int(row[IDX_ACCESS_TS_US])
                event = TraceEvent(
                    ts_us=ts_us,
                    block=BlockKey(
                        cf_id=parse_int(row[IDX_CF_ID]),
                        sst_fd_number=parse_int(row[IDX_SST_FD_NUMBER]),
                        block_offset=parse_int(row[IDX_BLOCK_OFFSET]),
                    ),
                    block_type=block_type,
                    block_size=parse_int(row[IDX_BLOCK_SIZE]),
                    cf_name=row[IDX_CF_NAME],
                    level=parse_int(row[IDX_LEVEL]),
                    caller=caller,
                    no_insert=parse_int(row[IDX_NO_INSERT]),
                    get_id=parse_int(row[IDX_GET_ID]),
                    is_cache_hit=parse_int(row[IDX_IS_CACHE_HIT]),
                    referenced_data_size=parse_int(row[IDX_REFERENCED_DATA_SIZE]),
                    referenced_key_exist_in_block=parse_int(
                        row[IDX_REFERENCED_KEY_EXIST_IN_BLOCK]
                    ),
                    num_keys_in_block=parse_int(row[IDX_NUM_KEYS_IN_BLOCK]),
                )
            except ValueError:
                bad_rows += 1
                continue
            if prev_kept_ts_us > ts_us:
                in_time_order = False
            prev_kept_ts_us = ts_us
            events.append(event)

    if not in_time_order:
        events.sort(key=lambda e: e.ts_us)
    return events, TraceLoadStats(
        total_rows=total_rows,
        kept_events=len(events),
        bad_rows=bad_rows,
        block_type_counts=block_type_counts,
        caller_counts=caller_counts,
    )


def load_block_trace_polars(
    path: Path, allow_iterator: bool, data_block_types: set[int]
) -> Tuple[List[TraceEvent], TraceLoadStats]:
    try:
        import polars as pl
    except ImportError:
        print("[WARN] polars not available, falling back to csv loader")
        return load_block_trace_csv(path, allow_iterator, data_block_types)

    schema_overrides = {
        "access_ts_us": pl.Int64,
        "block_type": pl.Int32,
        "block_size": pl.Int32,
        "cf_id": pl.Int32,
        "cf_name": pl.Utf8,
        "level": pl.Int32,
        "sst_fd_number": pl.Int64,
        "caller": pl.Int32,
        "no_insert": pl.Int32,
        "get_id": pl.Int64,
        "referenced_data_size": pl.Int64,
        "is_cache_hit": pl.Int32,
        "referenced_key_exist_in_block": pl.Int32,
        "num_keys_in_block": pl.Int32,
        "block_offset": pl.Int64,
    }

    read_csv_kwargs = dict(
        has_header=False,
        new_columns=TRACE_USED_COLUMNS,
        columns=TRACE_USED_COLUMN_INDICES,
        null_values=[""],
    )
    try:
        try:
            df = pl.read_csv(path, schema_overrides=schema_overrides, **read_csv_kwargs)
        except TypeError:
            # Older Polars versions used dtypes before schema_overrides.
            df = pl.read_csv(path, dtypes=schema_overrides, **read_csv_kwargs)
    except Exception as exc:
        print(f"[WARN] polars trace loader failed ({exc}); falling back to csv loader")
        return load_block_trace_csv(path, allow_iterator, data_block_types)

    total_rows = df.height
    numeric_columns = [name for name in TRACE_USED_COLUMNS if name != "cf_name"]
    df = df.with_columns(
        [pl.col(name).fill_null(0) for name in numeric_columns]
        + [pl.col("cf_name").fill_null("")]
    )

    block_type_counts = Counter(int(x) for x in df["block_type"].to_list())
    caller_counts = Counter(int(x) for x in df["caller"].to_list())

    allowed_callers = set(USER_CALLERS)
    if not allow_iterator:
        allowed_callers.discard(3)

    df = df.filter(
        pl.col("block_type").is_in(sorted(data_block_types))
        & pl.col("caller").is_in(sorted(allowed_callers))
    )

    cols = {name: df[name].to_list() for name in TRACE_USED_COLUMNS}
    events: List[TraceEvent] = []
    in_time_order = True
    prev_kept_ts_us = -1
    for (
        ts_us,
        block_type,
        block_size,
        cf_id,
        cf_name,
        level,
        sst_fd_number,
        caller,
        no_insert,
        get_id,
        referenced_data_size,
        is_cache_hit,
        referenced_key_exist_in_block,
        num_keys_in_block,
        block_offset,
    ) in zip(*(cols[name] for name in TRACE_USED_COLUMNS)):
        ts_us = int(ts_us)
        if prev_kept_ts_us > ts_us:
            in_time_order = False
        prev_kept_ts_us = ts_us
        events.append(
            TraceEvent(
                ts_us=ts_us,
                block=BlockKey(
                    cf_id=int(cf_id),
                    sst_fd_number=int(sst_fd_number),
                    block_offset=int(block_offset),
                ),
                block_type=int(block_type),
                block_size=int(block_size),
                cf_name=str(cf_name),
                level=int(level),
                caller=int(caller),
                no_insert=int(no_insert),
                get_id=int(get_id),
                is_cache_hit=int(is_cache_hit),
                referenced_data_size=int(referenced_data_size),
                referenced_key_exist_in_block=int(referenced_key_exist_in_block),
                num_keys_in_block=int(num_keys_in_block),
            )
        )

    if not in_time_order:
        events.sort(key=lambda e: e.ts_us)
    return events, TraceLoadStats(
        total_rows=total_rows,
        kept_events=len(events),
        bad_rows=0,
        block_type_counts=block_type_counts,
        caller_counts=caller_counts,
    )


def load_block_trace(
    path: Path,
    allow_iterator: bool,
    data_block_types: set[int],
    trace_loader: str,
) -> Tuple[List[TraceEvent], TraceLoadStats]:
    if trace_loader == "csv":
        return load_block_trace_csv(path, allow_iterator, data_block_types)
    return load_block_trace_polars(path, allow_iterator, data_block_types)


def load_snapshot_csv(path: Path) -> Tuple[List[int], List[SnapshotPoint]]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        if "ts_us" not in reader.fieldnames:
            raise ValueError(f"{path}: snapshot CSV must contain a ts_us column")

        rows: List[SnapshotPoint] = []
        for row_num, row in enumerate(reader, start=2):
            ts_us = parse_int(row["ts_us"])
            values: Dict[str, float] = {}
            for key, value in row.items():
                if key == "ts_us":
                    continue
                values[key] = parse_float(value or "")
            rows.append(SnapshotPoint(ts_us=ts_us, values=values))

    rows.sort(key=lambda s: s.ts_us)
    return [row.ts_us for row in rows], rows


def parse_file_infos(raw: str) -> List[Tuple[int, int]]:
    results: List[Tuple[int, int]] = []
    raw = raw.strip()
    if not raw:
        return results
    for token in raw.split(";"):
        token = token.strip()
        if not token or ":" not in token:
            continue
        level_raw, file_raw = token.split(":", 1)
        try:
            results.append((int(level_raw), int(file_raw)))
        except ValueError:
            continue
    return results


def load_sst_trace_tsv(
    path: Path,
) -> Tuple[Dict[Tuple[str, int], List[int]], SstTraceLoadStats]:
    invalidations: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    sst_trace_rows = 0
    invalidation_event_rows = 0
    invalidation_file_refs = 0
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing TSV header")
        required = {"event", "ts_us", "cf_name"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing TSV columns: {sorted(missing)}")

        for row in reader:
            sst_trace_rows += 1
            event_name = (row.get("event") or "").strip().lower()
            if "compaction" not in event_name:
                continue
            invalidation_event_rows += 1
            ts_us = parse_int(row["ts_us"])
            cf_name = row["cf_name"]
            file_infos = parse_file_infos(row.get("input_file_infos", ""))
            invalidation_file_refs += len(file_infos)
            for _, file_number in file_infos:
                invalidations[(cf_name, file_number)].append(ts_us)

    for ts_list in invalidations.values():
        ts_list.sort()
    return invalidations, SstTraceLoadStats(
        sst_trace_rows=sst_trace_rows,
        invalidation_event_rows=invalidation_event_rows,
        invalidation_file_refs=invalidation_file_refs,
    )


def lookup_snapshot(
    ts_us: int,
    snapshot_ts: Sequence[int],
    snapshots: Sequence[SnapshotPoint],
) -> Dict[str, float]:
    if not snapshot_ts:
        return {}
    idx = bisect.bisect_right(snapshot_ts, ts_us) - 1
    if idx < 0:
        return {}
    return snapshots[idx].values


def next_invalidation_ts(
    ts_us: int,
    event: TraceEvent,
    invalidations: Dict[Tuple[str, int], List[int]],
) -> Optional[int]:
    key = (event.cf_name, event.block.sst_fd_number)
    ts_list = invalidations.get(key)
    if not ts_list:
        return None
    idx = bisect.bisect_right(ts_list, ts_us)
    if idx >= len(ts_list):
        return None
    return ts_list[idx]


def prune_old(queue: Deque[int], min_ts_us: int) -> None:
    while queue and queue[0] < min_ts_us:
        queue.popleft()


def count_unique_request_ids(request_ids: Sequence[int], start: int, end: int) -> int:
    if end <= start:
        return 0
    count = end - start
    if count == 1:
        return 1
    if count <= 4:
        first = request_ids[start]
        second = request_ids[start + 1]
        if count == 2:
            return 1 if first == second else 2
        seen = {first, second}
        if count >= 3:
            seen.add(request_ids[start + 2])
        if count == 4:
            seen.add(request_ids[start + 3])
        return len(seen)
    return len(set(request_ids[start:end]))


def safe_ratio(numerator: float, denominator: float) -> float:
    if math.isnan(numerator) or math.isnan(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def build_dataset(
    events: Sequence[TraceEvent],
    snapshot_ts: Sequence[int],
    snapshots: Sequence[SnapshotPoint],
    invalidations: Dict[Tuple[str, int], List[int]],
    horizon_seconds: int,
    positive_reuse_threshold: int,
    candidate_cooldown_ms: int,
    max_first_reuse_seconds: int,
    min_benefit_score: float,
    future_reuse_count_mode: str,
    include_no_insert: bool,
    recent_short_seconds: int,
    recent_long_seconds: int,
) -> Iterable[Dict[str, object]]:
    horizon_us = horizon_seconds * 1_000_000
    candidate_cooldown_us = candidate_cooldown_ms * 1000
    max_first_reuse_us = max_first_reuse_seconds * 1_000_000
    short_us = recent_short_seconds * 1_000_000
    long_us = recent_long_seconds * 1_000_000

    block_key_to_id: Dict[Tuple[int, int, int], int] = {}
    id_to_block_key: List[BlockKey] = []
    event_block_ids: List[int] = []

    def intern_block_key(block: BlockKey) -> int:
        raw_key = (block.cf_id, block.sst_fd_number, block.block_offset)
        block_id = block_key_to_id.get(raw_key)
        if block_id is None:
            block_id = len(id_to_block_key)
            block_key_to_id[raw_key] = block_id
            id_to_block_key.append(block)
        return block_id

    by_block_ts: Dict[int, List[int]] = defaultdict(list)
    by_block_request_ids: Dict[int, List[int]] = defaultdict(list)
    for idx, event in enumerate(events):
        block_id = intern_block_key(event.block)
        event_block_ids.append(block_id)
        by_block_ts[block_id].append(event.ts_us)
        request_id = event.get_id if event.get_id > 0 else -(idx + 1)
        by_block_request_ids[block_id].append(request_id)

    recent_block_short: Dict[int, Deque[int]] = defaultdict(deque)
    recent_block_long: Dict[int, Deque[int]] = defaultdict(deque)
    recent_sst_short: Dict[Tuple[str, int], Deque[int]] = defaultdict(deque)
    recent_sst_long: Dict[Tuple[str, int], Deque[int]] = defaultdict(deque)
    recent_cf_short: Dict[str, Deque[int]] = defaultdict(deque)
    recent_cf_long: Dict[str, Deque[int]] = defaultdict(deque)
    decay_block_short: Dict[int, DecayedCounter] = defaultdict(DecayedCounter)
    decay_block_long: Dict[int, DecayedCounter] = defaultdict(DecayedCounter)
    decay_sst_short: Dict[Tuple[str, int], DecayedCounter] = defaultdict(DecayedCounter)
    decay_sst_long: Dict[Tuple[str, int], DecayedCounter] = defaultdict(DecayedCounter)
    decay_cf_short: Dict[str, DecayedCounter] = defaultdict(DecayedCounter)
    decay_cf_long: Dict[str, DecayedCounter] = defaultdict(DecayedCounter)

    last_candidate_ts_by_block: Dict[int, int] = {}
    unique_count_cache: Dict[Tuple[int, int, int], int] = {}

    for event_idx, event in enumerate(events):
        block_id = event_block_ids[event_idx]
        sst_key = (event.cf_name, event.block.sst_fd_number)
        block_short = recent_block_short[block_id]
        block_long = recent_block_long[block_id]
        sst_short = recent_sst_short[sst_key]
        sst_long = recent_sst_long[sst_key]
        cf_short = recent_cf_short[event.cf_name]
        cf_long = recent_cf_long[event.cf_name]

        prune_old(block_short, event.ts_us - short_us)
        prune_old(block_long, event.ts_us - long_us)
        prune_old(sst_short, event.ts_us - short_us)
        prune_old(sst_long, event.ts_us - long_us)
        prune_old(cf_short, event.ts_us - short_us)
        prune_old(cf_long, event.ts_us - long_us)

        block_decay_short = decay_block_short[block_id].score_at(
            event.ts_us, short_us
        )
        block_decay_long = decay_block_long[block_id].score_at(
            event.ts_us, long_us
        )
        sst_decay_short = decay_sst_short[sst_key].score_at(event.ts_us, short_us)
        sst_decay_long = decay_sst_long[sst_key].score_at(event.ts_us, long_us)
        cf_decay_short = decay_cf_short[event.cf_name].score_at(
            event.ts_us, short_us
        )
        cf_decay_long = decay_cf_long[event.cf_name].score_at(event.ts_us, long_us)

        is_candidate = event.is_cache_hit == 0 and (
            include_no_insert or event.no_insert == 0
        )
        if is_candidate:
            last_candidate_ts = last_candidate_ts_by_block.get(block_id)
            if (
                last_candidate_ts is not None
                and event.ts_us - last_candidate_ts < candidate_cooldown_us
            ):
                is_candidate = False

        if is_candidate:
            last_candidate_ts_by_block[block_id] = event.ts_us
            block_ts = by_block_ts[block_id]
            block_request_ids = by_block_request_ids[block_id]
            future_start = bisect.bisect_right(block_ts, event.ts_us)
            future_end = bisect.bisect_right(
                block_ts, event.ts_us + horizon_us, lo=future_start
            )
            future_reuse_raw_count = future_end - future_start
            unique_cache_key = (block_id, future_start, future_end)
            if unique_cache_key in unique_count_cache:
                future_reuse_unique_get_count = unique_count_cache[unique_cache_key]
            else:
                future_reuse_unique_get_count = count_unique_request_ids(
                    block_request_ids, future_start, future_end
                )
                unique_count_cache[unique_cache_key] = future_reuse_unique_get_count
            if future_reuse_count_mode == "access":
                future_reuse_count = future_reuse_raw_count
            else:
                future_reuse_count = future_reuse_unique_get_count
            first_reuse_delta_us = (
                block_ts[future_start] - event.ts_us
                if future_start < future_end
                else -1
            )

            invalidation_ts = next_invalidation_ts(event.ts_us, event, invalidations)
            invalidated_within_horizon = 0
            invalidation_delta_us = -1
            if invalidation_ts is not None:
                invalidation_delta_us = invalidation_ts - event.ts_us
                if invalidation_delta_us <= horizon_us:
                    invalidated_within_horizon = 1

            survived_until_first_reuse = 1
            if (
                invalidation_delta_us >= 0
                and first_reuse_delta_us >= 0
                and invalidation_delta_us <= first_reuse_delta_us
            ):
                survived_until_first_reuse = 0

            benefit_score = future_reuse_count / max(event.block_size / 1024.0, 1.0)
            positive_label = int(
                future_reuse_count >= positive_reuse_threshold
                and first_reuse_delta_us >= 0
                and first_reuse_delta_us <= max_first_reuse_us
                and benefit_score >= min_benefit_score
                and survived_until_first_reuse == 1
            )

            snapshot = lookup_snapshot(event.ts_us, snapshot_ts, snapshots)
            block_cache_capacity = snapshot.get("block_cache_capacity", math.nan)
            block_cache_usage = snapshot.get("block_cache_usage", math.nan)
            block_cache_pinned_usage = snapshot.get(
                "block_cache_pinned_usage", math.nan
            )
            row: Dict[str, object] = {
                "ts_us": event.ts_us,
                "cf_id": event.block.cf_id,
                "cf_name": event.cf_name,
                "sst_fd_number": event.block.sst_fd_number,
                "block_offset": event.block.block_offset,
                "block_size": event.block_size,
                "level": event.level,
                "caller": CALLER_NAMES.get(event.caller, str(event.caller)),
                "no_insert": event.no_insert,
                "referenced_data_size": event.referenced_data_size,
                "referenced_key_exist_in_block": event.referenced_key_exist_in_block,
                "num_keys_in_block": event.num_keys_in_block,
                "recent_block_hits_10s": len(block_short),
                "recent_block_hits_60s": len(block_long),
                "recent_sst_hits_10s": len(sst_short),
                "recent_sst_hits_60s": len(sst_long),
                "recent_cf_hits_10s": len(cf_short),
                "recent_cf_hits_60s": len(cf_long),
                "recent_block_decay_10s": block_decay_short,
                "recent_block_decay_60s": block_decay_long,
                "recent_sst_decay_10s": sst_decay_short,
                "recent_sst_decay_60s": sst_decay_long,
                "recent_cf_decay_10s": cf_decay_short,
                "recent_cf_decay_60s": cf_decay_long,
                "future_reuse_raw_count": future_reuse_raw_count,
                "future_reuse_unique_get_count": future_reuse_unique_get_count,
                "future_reuse_count": future_reuse_count,
                "first_reuse_delta_us": first_reuse_delta_us,
                "invalidated_within_horizon": invalidated_within_horizon,
                "invalidation_delta_us": invalidation_delta_us,
                "survived_until_first_reuse": survived_until_first_reuse,
                "benefit_score": benefit_score,
                "block_cache_usage_ratio": safe_ratio(
                    block_cache_usage, block_cache_capacity
                ),
                "block_cache_pinned_usage_ratio": safe_ratio(
                    block_cache_pinned_usage, block_cache_capacity
                ),
                "label": positive_label,
            }
            for key, value in snapshot.items():
                row[key] = value
            yield row

        block_short.append(event.ts_us)
        block_long.append(event.ts_us)
        sst_short.append(event.ts_us)
        sst_long.append(event.ts_us)
        cf_short.append(event.ts_us)
        cf_long.append(event.ts_us)
        decay_block_short[block_id].add(event.ts_us, short_us)
        decay_block_long[block_id].add(event.ts_us, long_us)
        decay_sst_short[sst_key].add(event.ts_us, short_us)
        decay_sst_long[sst_key].add(event.ts_us, long_us)
        decay_cf_short[event.cf_name].add(event.ts_us, short_us)
        decay_cf_long[event.cf_name].add(event.ts_us, long_us)


def format_counter(counter: Counter[int], value_names: Dict[int, str]) -> str:
    if not counter:
        return "(none)"
    parts: List[str] = []
    for key, count in sorted(counter.items()):
        label = value_names.get(key, str(key))
        parts.append(f"{label}={count}")
    return ", ".join(parts)


def write_rows(
    path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]
) -> WriteStats:
    row_count = 0
    label_pos = 0
    label_neg = 0
    build_dataset_sec = 0.0
    write_csv_sec = 0.0
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        write_start = time.perf_counter()
        writer.writeheader()
        write_csv_sec += time.perf_counter() - write_start

        row_iter = iter(rows)
        while True:
            build_start = time.perf_counter()
            try:
                row = next(row_iter)
            except StopIteration:
                build_dataset_sec += time.perf_counter() - build_start
                break
            build_dataset_sec += time.perf_counter() - build_start

            write_start = time.perf_counter()
            writer.writerow(row)
            write_csv_sec += time.perf_counter() - write_start
            row_count += 1
            if int(row.get("label", 0)) == 1:
                label_pos += 1
            else:
                label_neg += 1
    if row_count == 0:
        path.unlink(missing_ok=True)
    return WriteStats(
        row_count=row_count,
        label_pos=label_pos,
        label_neg=label_neg,
        build_dataset_sec=build_dataset_sec,
        write_csv_sec=write_csv_sec,
    )


def main() -> int:
    total_start = time.perf_counter()
    args = parse_args()

    block_trace = Path(args.block_trace)
    output = Path(args.output)

    snapshot_ts: List[int] = []
    snapshots: List[SnapshotPoint] = []
    invalidations: Dict[Tuple[str, int], List[int]] = {}

    data_block_types = parse_int_set(args.data_block_types)
    trace_load_start = time.perf_counter()
    events, trace_stats = load_block_trace(
        block_trace,
        allow_iterator=args.allow_iterator,
        data_block_types=data_block_types,
        trace_loader=args.trace_loader,
    )
    trace_load_sec = time.perf_counter() - trace_load_start

    snapshot_load_sec = 0.0
    if args.snapshot_csv:
        snapshot_load_start = time.perf_counter()
        snapshot_ts, snapshots = load_snapshot_csv(Path(args.snapshot_csv))
        snapshot_load_sec = time.perf_counter() - snapshot_load_start

    sst_trace_load_sec = 0.0
    if args.sst_trace_tsv:
        sst_trace_load_start = time.perf_counter()
        invalidations, sst_stats = load_sst_trace_tsv(Path(args.sst_trace_tsv))
        sst_trace_load_sec = time.perf_counter() - sst_trace_load_start
        print(
            "[SST] "
            f"sst_trace_rows={sst_stats.sst_trace_rows} "
            f"invalidation_event_rows={sst_stats.invalidation_event_rows} "
            f"invalidation_file_refs={sst_stats.invalidation_file_refs}"
        )
        if sst_stats.invalidation_file_refs == 0:
            print(
                "[WARN] sst trace was provided but no compaction invalidation "
                "file refs were loaded"
            )

    snapshot_fieldnames: List[str] = []
    if snapshots:
        snapshot_keys = set()
        for snapshot in snapshots:
            snapshot_keys.update(snapshot.values.keys())
        snapshot_fieldnames = [
            key for key in sorted(snapshot_keys) if key not in BASE_FIELDNAMES
        ]
    fieldnames = BASE_FIELDNAMES + snapshot_fieldnames

    rows = build_dataset(
        events=events,
        snapshot_ts=snapshot_ts,
        snapshots=snapshots,
        invalidations=invalidations,
        horizon_seconds=args.horizon_seconds,
        positive_reuse_threshold=args.positive_reuse_threshold,
        candidate_cooldown_ms=args.candidate_cooldown_ms,
        max_first_reuse_seconds=args.max_first_reuse_seconds,
        min_benefit_score=args.min_benefit_score,
        future_reuse_count_mode=args.future_reuse_count_mode,
        include_no_insert=args.include_no_insert,
        recent_short_seconds=args.recent_short_seconds,
        recent_long_seconds=args.recent_long_seconds,
    )
    write_stats = write_rows(output, rows, fieldnames)
    build_dataset_sec = write_stats.build_dataset_sec
    write_csv_sec = write_stats.write_csv_sec
    if write_stats.row_count == 0:
        raise ValueError(
            "No candidate samples were generated. "
            f"Loaded {trace_stats.kept_events} user-access data-block events "
            f"from {trace_stats.total_rows} trace rows after filtering "
            f"(bad_rows={trace_stats.bad_rows}, data_block_types={sorted(data_block_types)}). "
            f"block_type counts: "
            f"{format_counter(trace_stats.block_type_counts, {7: 'index', 8: 'filter', 9: 'data', 10: 'uncompression_dict', 11: 'range_deletion'})}. "
            f"caller counts: {format_counter(trace_stats.caller_counts, CALLER_NAMES)}."
        )

    pos_ratio = write_stats.label_pos / write_stats.row_count
    total_sec = time.perf_counter() - total_start
    print(
        f"Wrote {write_stats.row_count} training samples to {output} "
        f"(positive={write_stats.label_pos}, negative={write_stats.label_neg}, "
        f"pos_ratio={pos_ratio:.6f}, future_reuse_count_mode={args.future_reuse_count_mode}, "
        f"include_no_insert={int(args.include_no_insert)}, "
        f"data_block_types={sorted(data_block_types)}, bad_rows={trace_stats.bad_rows})"
    )
    print(
        "[TIMING] "
        f"trace_load_sec={trace_load_sec:.2f} "
        f"snapshot_load_sec={snapshot_load_sec:.2f} "
        f"sst_trace_load_sec={sst_trace_load_sec:.2f} "
        f"build_dataset_sec={build_dataset_sec:.2f} "
        f"write_csv_sec={write_csv_sec:.2f} "
        f"total_sec={total_sec:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
