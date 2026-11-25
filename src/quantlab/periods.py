"""Period building for backtests (webapp, rnd, tests).

Supports:
- three_block: config-defined p1/p2/p3 blocks
- yearly: 2020, 2021, 2022, ..., current-year->today
- holdout: extra holdout blocks, added in both modes

Reads from config['periods'].

We intentionally keep this small and dependency-free.
"""

from __future__ import annotations

import datetime as dt
from typing import Any


def _parse_date(s: str) -> dt.date:
    if s == "today":
        return dt.date.today()
    return dt.date.fromisoformat(s)


def _block(item: dict[str, Any], default_name: str = "") -> tuple[str, dt.date, dt.date]:
    return (str(item.get("name", default_name)), _parse_date(str(item.get("start"))), _parse_date(str(item.get("end"))))


def _holdout_blocks(cfg: dict[str, Any] | None) -> list[tuple[str, dt.date, dt.date]]:
    return [_block(h) for h in (cfg or {}).get("periods", {}).get("holdout") or []]


def holdout_names(cfg: dict[str, Any] | None) -> list[str]:
    """Periods that are never scored: the extra `periods.holdout` blocks + `periods.score_exclude`.

    The webapp shows them outside the header stats; quantlab-rnd doesn't run them while tuning.
    """
    return [name for name, _, _ in _holdout_blocks(cfg)] + list((cfg or {}).get("periods", {}).get("score_exclude") or [])


def build_periods(cfg: dict[str, Any] | None) -> list[tuple[str, dt.date, dt.date]]:
    """Build periods list (name, start, end).

    Mode defaults to three_block if not provided.

    Notes:
    - three_block requires cfg['periods']['three_block'] with p1/p2/p3.
    - Includes the holdout blocks; a period that overlaps one raises.
    - Order: by date; three_block puts the holdouts last (2020-2022, 2023-2025, 2019, 2026).
    """
    periods, holdout = _mode_periods(cfg), _holdout_blocks(cfg)
    for name, start, end in periods:
        for h_name, h_start, h_end in holdout:
            if start <= h_end and h_start <= end:
                raise ValueError(f"period {name} ({start}..{end}) overlaps the holdout {h_name} ({h_start}..{h_end})")
    periods = sorted(periods + holdout, key=lambda p: p[1])
    if (cfg or {}).get("periods", {}).get("mode", "three_block") == "three_block":
        never_scored = set(holdout_names(cfg))
        periods.sort(key=lambda p: p[0] in never_scored)
    return periods


def _mode_periods(cfg: dict[str, Any] | None) -> list[tuple[str, dt.date, dt.date]]:

    p = (cfg or {}).get("periods", {})
    mode = p.get("mode", "three_block")

    if mode == "yearly":
        start_year = 2020
        end_date = dt.date.today()
        periods: list[tuple[str, dt.date, dt.date]] = []

        for y in range(start_year, end_date.year + 1):
            start = dt.date(y, 1, 1)
            end = dt.date(y, 12, 31)
            if y == end_date.year:
                end = end_date
            periods.append((str(y), start, end))

        return periods

    # default: three_block
    tb = p.get("three_block")
    if not (isinstance(tb, dict) and tb.get("p1") and tb.get("p2") and tb.get("p3")):
        raise ValueError("periods.mode=three_block requires periods.three_block.p1/p2/p3 in config")

    return [_block(tb[key], key) for key in ("p1", "p2", "p3")]
