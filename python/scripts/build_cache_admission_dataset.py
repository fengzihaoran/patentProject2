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
from collections import defaultdict, deque
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


# TraceType enum values are stable enough for first-pass filtering.
# We only need the data-block one.
TRACE_TYPE_DATA_BLOCK = 7

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


def load_block_trace(path: Path, allow_iterator: bool) -> List[TraceEvent]:
    events: List[TraceEvent] = []
    allowed_callers = set(USER_CALLERS)
    if not allow_iterator:
        allowed_callers.discard(3)

    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        for row_num, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != len(TRACE_FIELDS):
                raise ValueError(
                    f"{path}:{row_num}: expected {len(TRACE_FIELDS)} fields, got {len(row)}"
                )
            data = dict(zip(TRACE_FIELDS, row))
            caller = parse_int(data["caller"])
            event = TraceEvent(
                ts_us=parse_int(data["access_ts_us"]),
                block=BlockKey(
                    cf_id=parse_int(data["cf_id"]),
                    sst_fd_number=parse_int(data["sst_fd_number"]),
                    block_offset=parse_int(data["block_offset"]),
                ),
                block_type=parse_int(data["block_type"]),
                block_size=parse_int(data["block_size"]),
                cf_name=data["cf_name"],
                level=parse_int(data["level"]),
                caller=caller,
                no_insert=parse_int(data["no_insert"]),
                get_id=parse_int(data["get_id"]),
                is_cache_hit=parse_int(data["is_cache_hit"]),
                referenced_data_size=parse_int(data["referenced_data_size"]),
                referenced_key_exist_in_block=parse_int(
                    data["referenced_key_exist_in_block"]
                ),
                num_keys_in_block=parse_int(data["num_keys_in_block"]),
            )
            if event.block_type != TRACE_TYPE_DATA_BLOCK:
                continue
            if caller not in allowed_callers:
                continue
            events.append(event)

    events.sort(key=lambda e: e.ts_us)
    return events


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

    by_block: Dict[BlockKey, List[int]] = defaultdict(list)
    for idx, event in enumerate(events):
        by_block[event.block].append(idx)

    recent_block: Dict[BlockKey, Deque[int]] = defaultdict(deque)
    recent_sst: Dict[Tuple[str, int], Deque[int]] = defaultdict(deque)
    recent_cf: Dict[str, Deque[int]] = defaultdict(deque)

    next_pos_per_block: Dict[BlockKey, int] = defaultdict(int)
    last_candidate_ts_by_block: Dict[BlockKey, int] = {}

    for idx, event in enumerate(events):
        block_hist = recent_block[event.block]
        sst_hist = recent_sst[(event.cf_name, event.block.sst_fd_number)]
        cf_hist = recent_cf[event.cf_name]

        prune_old(block_hist, event.ts_us - long_us)
        prune_old(sst_hist, event.ts_us - long_us)
        prune_old(cf_hist, event.ts_us - long_us)

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
            block_indices = by_block[event.block]
            pos = next_pos_per_block[event.block]
            while pos < len(block_indices) and block_indices[pos] <= idx:
                pos += 1
            next_pos_per_block[event.block] = pos

            future_reuse_count = 0
            first_reuse_delta_us = -1
            j = pos
            while j < len(block_indices):
                future_idx = block_indices[j]
                future_event = events[future_idx]
                delta_us = future_event.ts_us - event.ts_us
                if delta_us > horizon_us:
                    break
                future_reuse_count += 1
                if first_reuse_delta_us < 0:
                    first_reuse_delta_us = delta_us
                j += 1

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
                "recent_block_hits_10s": count_recent(block_hist, event.ts_us - short_us),
                "recent_block_hits_60s": len(block_hist),
                "recent_sst_hits_10s": count_recent(sst_hist, event.ts_us - short_us),
                "recent_sst_hits_60s": len(sst_hist),
                "recent_cf_hits_10s": count_recent(cf_hist, event.ts_us - short_us),
                "recent_cf_hits_60s": len(cf_hist),
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

        block_hist.append(event.ts_us)
        sst_hist.append(event.ts_us)
        cf_hist.append(event.ts_us)


def count_recent(queue: Deque[int], min_ts_us: int) -> int:
    count = 0
    for ts in reversed(queue):
        if ts < min_ts_us:
            break
        count += 1
    return count


def write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    materialized = list(rows)
    if not materialized:
        raise ValueError("No candidate samples were generated.")

    fieldnames: List[str] = []
    seen = set()
    for row in materialized:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def main() -> int:
    args = parse_args()

    block_trace = Path(args.block_trace)
    output = Path(args.output)

    snapshot_ts: List[int] = []
    snapshots: List[SnapshotPoint] = []
    invalidations: Dict[Tuple[str, int], List[int]] = {}

    events = load_block_trace(block_trace, allow_iterator=args.allow_iterator)
    if args.snapshot_csv:
        snapshot_ts, snapshots = load_snapshot_csv(Path(args.snapshot_csv))
    if args.sst_trace_tsv:
        invalidations = load_sst_trace_tsv(Path(args.sst_trace_tsv))

    row_count = write_rows(
        output,
        build_dataset(
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
        ),
    )

    print(f"Wrote {row_count} training samples to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
