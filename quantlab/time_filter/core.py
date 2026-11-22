from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class SessionWindow:
    """A daily time-of-day window.

    Times are interpreted in `tz` when building the mask.

    Example
    -------
    Block Asia (roughly): SessionWindow("00:00", "06:59") in UTC.
    """

    start_hhmm: str  # inclusive
    end_hhmm: str  # inclusive; supports wrap-around (e.g. 22:00..01:00)
    days: tuple[int, ...] | None = None  # 0=Mon..6=Sun; None -> all days


@dataclass(frozen=True)
class EventWindow:
    """A single event timestamp with a blocking window around it."""

    ts: pd.Timestamp
    pre: dt.timedelta
    post: dt.timedelta


def _parse_hhmm(s: str) -> tuple[int, int]:
    try:
        hh, mm = s.split(":")
        return int(hh), int(mm)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid HH:MM time: {s!r}") from e


def build_allow_mask_from_sessions(
    index: pd.DatetimeIndex,
    *,
    tz: str = "UTC",
    block: list[SessionWindow] | None = None,
    allow: list[SessionWindow] | None = None,
) -> pd.Series:
    """Build a boolean allow-mask for a price/position index.

    Exactly one of (block, allow) should be provided.

    - If `block` is provided: allow=True except inside any blocked session.
    - If `allow` is provided: allow=True only inside at least one allowed session.

    Wrap-around windows are supported (e.g. 22:00..01:00).
    """
    if block is not None and allow is not None:
        raise ValueError("Provide only one of block or allow")
    if block is None and allow is None:
        # default: allow everything
        return pd.Series(True, index=index)

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_local = idx.tz_convert(tz)

    # minutes since midnight
    mins = idx_local.hour * 60 + idx_local.minute
    dow = idx_local.dayofweek

    def in_window(w: SessionWindow) -> pd.Series:
        sh, sm = _parse_hhmm(w.start_hhmm)
        eh, em = _parse_hhmm(w.end_hhmm)
        smin = sh * 60 + sm
        emin = eh * 60 + em

        if w.days is None:
            day_ok = pd.Series(True, index=idx_local)
        else:
            day_ok = pd.Series(dow.isin(list(w.days)), index=idx_local)

        if smin <= emin:
            t_ok = (mins >= smin) & (mins <= emin)
        else:
            # wrap-around (e.g. 22:00..01:00)
            t_ok = (mins >= smin) | (mins <= emin)

        return pd.Series(t_ok, index=idx_local) & day_ok

    windows = block if block is not None else allow
    m = pd.DataFrame({f"w{i}": in_window(w) for i, w in enumerate(windows)}).any(axis=1)

    if block is not None:
        return (~m).reindex(idx_local).astype(bool)
    return m.reindex(idx_local).astype(bool)


def build_allow_mask_from_months(
    index: pd.DatetimeIndex,
    *,
    block: list[int] | tuple[int, ...] | set[int] | None = None,
) -> pd.Series:
    """Build an allow-mask that blocks entire calendar months.

    Parameters
    ----------
    block:
      Months to block (1-12). Example: block=[6] blocks June entirely.

    Semantics
    ---------
    - allow=True outside blocked months
    - allow=False inside blocked months

    Notes
    -----
    - Uses the timestamp index month in UTC (or whatever tz is already on the index).
    - This is intended as a blunt seasonal *force-flat* filter.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    months = sorted({int(m) for m in (block or [])})
    if not months:
        return pd.Series(True, index=idx)

    allow = ~pd.Index(idx.month).isin(months)
    return pd.Series(allow, index=idx).astype(bool)


def build_allow_mask_from_events(index: pd.DatetimeIndex, *, events: list[EventWindow]) -> pd.Series:
    """Allow-mask that blocks within any event's [ts-pre, ts+post] interval."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    allow = pd.Series(True, index=idx)
    for e in events:
        ts = pd.Timestamp(e.ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        start = ts - e.pre
        end = ts + e.post
        allow.loc[(idx >= start) & (idx <= end)] = False
    return allow.astype(bool)


# A configured event type with no event this close to the period's start or end means
# the calendar doesn't cover the period (the longest normal gap, between FOMC meetings,
# is about 7 weeks).
CALENDAR_COVERAGE_SLACK = dt.timedelta(days=60)


NEW_YORK = ZoneInfo("America/New_York")


def new_york_to_utc(date: str, time_et: str) -> pd.Timestamp:
    """New York local date + HH:MM -> UTC (daylight saving aware: 14:00 ET is 18:00 or 19:00 UTC)."""
    d = dt.date.fromisoformat(date)
    hh, mm = _parse_hhmm(time_et)
    return pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=NEW_YORK)).tz_convert("UTC")


def load_calendar_events(
    csv_path: str | Path,
    *,
    windows: dict[str, tuple[float, float]],
    start: dt.date,
    end: dt.date,
) -> list[EventWindow]:
    """Event windows for `start..end` from the calendar CSV (date, time_et, event; New York time).

    windows: event code -> (hours before, hours after). Only these codes are used.

    Fails loudly instead of silently filtering nothing: raises if the file is missing,
    a configured code isn't in it, or the calendar doesn't cover the period for a code.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Event calendar not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["date", "time_et", "event"], dtype=str)
    df["ts_utc"] = [new_york_to_utc(d, t) for d, t in zip(df["date"], df["time_et"])]

    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    events: list[EventWindow] = []
    for code, (pre_hours, post_hours) in windows.items():
        ts = df.loc[df["event"] == code, "ts_utc"]
        if ts.empty:
            raise ValueError(f"No {code!r} events in {csv_path.name} (codes there: {sorted(df['event'].unique())})")
        if ts.min() - CALENDAR_COVERAGE_SLACK > lo or ts.max() + CALENDAR_COVERAGE_SLACK < hi:
            raise ValueError(
                f"{csv_path.name} has {code} events from {ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d}, "
                f"which doesn't cover {start}..{end}"
            )
        pre, post = dt.timedelta(hours=pre_hours), dt.timedelta(hours=post_hours)
        events += [EventWindow(ts=t, pre=pre, post=post) for t in ts if lo - post <= t < hi + pre]
    return events


def apply_time_filter(
    pos_size: pd.Series,
    allow_mask: pd.Series,
) -> pd.Series:
    """Apply a time filter to a position/size series.

    Canonical semantics: **force_flat**.

    - allow_mask=True  -> position unchanged
    - allow_mask=False -> force position to 0

    (Other semantics were removed to avoid confusing multi-shift interactions with
    engine lag and to keep the backtest wiring unambiguous.)
    """
    if not isinstance(pos_size, pd.Series):
        raise TypeError("pos_size must be a Series")

    pos = pos_size.copy().astype(float)
    idx = pos.index
    m = pd.Series(allow_mask, index=idx).reindex(idx).fillna(False).astype(bool)

    return pos.where(m, 0.0).astype(float)
