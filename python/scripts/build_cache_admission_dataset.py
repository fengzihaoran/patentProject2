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

BASE_FIELDNAMES = [
    "ts_us",
    "cf_id",
    "cf_name",
    "sst_fd_number",
    "block_offset",
    "block_size",
    "level",
    "caller",
    "referenced_data_size",
    "referenced_key_exist_in_block",
    "num_keys_in_block",
    "recent_block_hits_10s",
    "recent_block_hits_60s",
    "recent_sst_hits_10s",
    "recent_sst_hits_60s",
    "recent_cf_hits_10s",
    "recent_cf_hits_60s",
    "future_reuse_count",
    "first_reuse_delta_us",
    "invalidated_within_horizon",
    "invalidation_delta_us",
    "survived_until_first_reuse",
    "benefit_score",
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


@dataclass(frozen=True)
class BlockKey:
    cf_id: int
    sst_fd_number: int
    block_offset: int


@dataclass
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


@dataclass
class SnapshotPoint:
    ts_us: int
    values: Dict[str, float]


@dataclass
class InvalidationEvent:
    ts_us: int
    cf_name: str
    input_files: List[Tuple[int, int]]


@dataclass
class TraceLoadStats:
    total_rows: int
    kept_events: int
    block_type_counts: Counter[int]
    caller_counts: Counter[int]


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


def parse_int(raw: str) -> int:
    raw = raw.strip()
    if raw == "":
        return 0
    return int(raw)


def parse_float(raw: str) -> float:
    raw = raw.strip()
    if raw == "":
        return math.nan
    return float(raw)


def load_block_trace(
    path: Path, allow_iterator: bool
) -> Tuple[List[TraceEvent], TraceLoadStats]:
    events: List[TraceEvent] = []
    allowed_callers = set(USER_CALLERS)
    if not allow_iterator:
        allowed_callers.discard(3)
    block_type_counts: Counter[int] = Counter()
    caller_counts: Counter[int] = Counter()
    total_rows = 0
    in_time_order = True
    prev_kept_ts_us = -1

    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        for row_num, row in enumerate(reader, start=1):
            if not row:
                continue
            total_rows += 1
            if len(row) != len(TRACE_FIELDS):
                raise ValueError(
                    f"{path}:{row_num}: expected {len(TRACE_FIELDS)} fields, got {len(row)}"
                )
            block_type = parse_int(row[IDX_BLOCK_TYPE])
            caller = parse_int(row[IDX_CALLER])
            block_type_counts[block_type] += 1
            caller_counts[caller] += 1
            if block_type != TRACE_TYPE_DATA_BLOCK:
                continue
            if caller not in allowed_callers:
                continue
            ts_us = parse_int(row[IDX_ACCESS_TS_US])
            if prev_kept_ts_us > ts_us:
                in_time_order = False
            prev_kept_ts_us = ts_us
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
            events.append(event)

    if not in_time_order:
        events.sort(key=lambda e: e.ts_us)
    return events, TraceLoadStats(
        total_rows=total_rows,
        kept_events=len(events),
        block_type_counts=block_type_counts,
        caller_counts=caller_counts,
    )


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


def load_sst_trace_tsv(path: Path) -> Dict[Tuple[str, int], List[int]]:
    invalidations: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing TSV header")
        required = {"event", "ts_us", "cf_name"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing TSV columns: {sorted(missing)}")

        for row in reader:
            if row["event"] != "compaction":
                continue
            ts_us = parse_int(row["ts_us"])
            cf_name = row["cf_name"]
            for _, file_number in parse_file_infos(row.get("input_file_infos", "")):
                invalidations[(cf_name, file_number)].append(ts_us)

    for ts_list in invalidations.values():
        ts_list.sort()
    return invalidations


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
    recent_short_seconds: int,
    recent_long_seconds: int,
) -> Iterable[Dict[str, object]]:
    horizon_us = horizon_seconds * 1_000_000
    candidate_cooldown_us = candidate_cooldown_ms * 1000
    max_first_reuse_us = max_first_reuse_seconds * 1_000_000
    short_us = recent_short_seconds * 1_000_000
    long_us = recent_long_seconds * 1_000_000

    by_block_ts: Dict[BlockKey, List[int]] = defaultdict(list)
    for event in events:
        by_block_ts[event.block].append(event.ts_us)

    recent_block_short: Dict[BlockKey, Deque[int]] = defaultdict(deque)
    recent_block_long: Dict[BlockKey, Deque[int]] = defaultdict(deque)
    recent_sst_short: Dict[Tuple[str, int], Deque[int]] = defaultdict(deque)
    recent_sst_long: Dict[Tuple[str, int], Deque[int]] = defaultdict(deque)
    recent_cf_short: Dict[str, Deque[int]] = defaultdict(deque)
    recent_cf_long: Dict[str, Deque[int]] = defaultdict(deque)

    last_candidate_ts_by_block: Dict[BlockKey, int] = {}

    for event in events:
        sst_key = (event.cf_name, event.block.sst_fd_number)
        block_short = recent_block_short[event.block]
        block_long = recent_block_long[event.block]
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

        is_candidate = event.is_cache_hit == 0
        if is_candidate:
            last_candidate_ts = last_candidate_ts_by_block.get(event.block)
            if (
                last_candidate_ts is not None
                and event.ts_us - last_candidate_ts < candidate_cooldown_us
            ):
                is_candidate = False

        if is_candidate:
            last_candidate_ts_by_block[event.block] = event.ts_us
            block_ts = by_block_ts[event.block]
            future_start = bisect.bisect_right(block_ts, event.ts_us)
            future_end = bisect.bisect_right(
                block_ts, event.ts_us + horizon_us, lo=future_start
            )
            future_reuse_count = future_end - future_start
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
            row: Dict[str, object] = {
                "ts_us": event.ts_us,
                "cf_id": event.block.cf_id,
                "cf_name": event.cf_name,
                "sst_fd_number": event.block.sst_fd_number,
                "block_offset": event.block.block_offset,
                "block_size": event.block_size,
                "level": event.level,
                "caller": CALLER_NAMES.get(event.caller, str(event.caller)),
                "referenced_data_size": event.referenced_data_size,
                "referenced_key_exist_in_block": event.referenced_key_exist_in_block,
                "num_keys_in_block": event.num_keys_in_block,
                "recent_block_hits_10s": len(block_short),
                "recent_block_hits_60s": len(block_long),
                "recent_sst_hits_10s": len(sst_short),
                "recent_sst_hits_60s": len(sst_long),
                "recent_cf_hits_10s": len(cf_short),
                "recent_cf_hits_60s": len(cf_long),
                "future_reuse_count": future_reuse_count,
                "first_reuse_delta_us": first_reuse_delta_us,
                "invalidated_within_horizon": invalidated_within_horizon,
                "invalidation_delta_us": invalidation_delta_us,
                "survived_until_first_reuse": survived_until_first_reuse,
                "benefit_score": benefit_score,
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
) -> int:
    row_count = 0
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            row_count += 1
    if row_count == 0:
        path.unlink(missing_ok=True)
    return row_count


def main() -> int:
    args = parse_args()

    block_trace = Path(args.block_trace)
    output = Path(args.output)

    snapshot_ts: List[int] = []
    snapshots: List[SnapshotPoint] = []
    invalidations: Dict[Tuple[str, int], List[int]] = {}

    events, trace_stats = load_block_trace(
        block_trace, allow_iterator=args.allow_iterator
    )
    if args.snapshot_csv:
        snapshot_ts, snapshots = load_snapshot_csv(Path(args.snapshot_csv))
    if args.sst_trace_tsv:
        invalidations = load_sst_trace_tsv(Path(args.sst_trace_tsv))

    snapshot_fieldnames: List[str] = []
    if snapshots:
        snapshot_fieldnames = [
            key for key in snapshots[0].values.keys() if key not in BASE_FIELDNAMES
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
        recent_short_seconds=args.recent_short_seconds,
        recent_long_seconds=args.recent_long_seconds,
    )
    row_count = write_rows(output, rows, fieldnames)
    if row_count == 0:
        raise ValueError(
            "No candidate samples were generated. "
            f"Loaded {trace_stats.kept_events} user-access data-block events "
            f"from {trace_stats.total_rows} trace rows after filtering. "
            f"block_type counts: "
            f"{format_counter(trace_stats.block_type_counts, {7: 'index', 8: 'filter', 9: 'data', 10: 'uncompression_dict', 11: 'range_deletion'})}. "
            f"caller counts: {format_counter(trace_stats.caller_counts, CALLER_NAMES)}."
        )

    print(f"Wrote {row_count} training samples to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
