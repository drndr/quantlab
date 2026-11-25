"""time_filter: blocked months + event windows from the calendar; fails loudly, never silently."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from quantlab.paths import EVENT_CALENDAR
from quantlab.pipeline import load_time_filter_mask
from quantlab.time_filter.core import new_york_to_utc

DAY = dt.date(2024, 7, 31)
FOMC_2H = {"FOMC": {"pre_hours": 2.0, "post_hours": 0.5}}


def _calendar(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Calendar file in the real format: date, time_et (New York), event, note."""
    p = tmp_path / "events.csv"
    pd.DataFrame([(*r, "") for r in rows], columns=["date", "time_et", "event", "note"]).to_csv(p, index=False)
    return p


def _mask(events: dict | None, calendar: Path, idx: pd.DatetimeIndex, start=DAY, end=DAY, months=None):
    tf = {k: v for k, v in (("events", events), ("months", months)) if v}
    return load_time_filter_mask(idx, start, end, cfg={"time_filter": tf} if tf else {}, calendar=calendar)


SUMMER_FOMC = [("2024-06-12", "14:00", "FOMC"), ("2024-07-31", "14:00", "FOMC"), ("2024-09-18", "14:00", "FOMC")]


def test_event_window_uses_new_york_time(tmp_path: Path) -> None:
    # 14:00 New York in July is 18:00 UTC, so 2h before / 30min after blocks 16:00-18:30 UTC.
    idx = pd.date_range("2024-07-31 15:50", "2024-07-31 18:40", freq="5min", tz="UTC")
    m = _mask(FOMC_2H, _calendar(tmp_path, SUMMER_FOMC), idx)
    blocked = idx[~m.to_numpy()]
    assert blocked[0] == pd.Timestamp("2024-07-31 16:00", tz="UTC")
    assert blocked[-1] == pd.Timestamp("2024-07-31 18:30", tz="UTC")


def test_only_configured_event_types_are_blocked(tmp_path: Path) -> None:
    cal = _calendar(tmp_path, [*SUMMER_FOMC, ("2024-07-31", "08:30", "CPI")])
    idx = pd.date_range("2024-07-31 12:00", "2024-07-31 13:00", freq="5min", tz="UTC")
    assert _mask(FOMC_2H, cal, idx).all()  # CPI is in the file but not configured


def test_months_block_without_events(tmp_path: Path) -> None:
    idx = pd.date_range("2026-05-31", periods=4, freq="1D", tz="UTC")
    m = _mask(None, tmp_path / "unused.csv", idx, dt.date(2026, 5, 31), dt.date(2026, 6, 3), months={"block": [6]})
    assert m.tolist() == [True, False, False, False]


def test_no_time_filter_means_no_mask(tmp_path: Path) -> None:
    idx = pd.date_range("2024-07-31", periods=3, freq="5min", tz="UTC")
    assert _mask(None, tmp_path / "unused.csv", idx) is None


def test_missing_calendar_file_raises(tmp_path: Path) -> None:
    idx = pd.date_range("2024-07-31", periods=3, freq="5min", tz="UTC")
    with pytest.raises(FileNotFoundError):
        _mask(FOMC_2H, tmp_path / "nope.csv", idx)


def test_unknown_event_type_raises(tmp_path: Path) -> None:
    idx = pd.date_range("2024-07-31", periods=3, freq="5min", tz="UTC")
    with pytest.raises(ValueError, match="'CPI'"):
        _mask({"CPI": {"pre_hours": 1.0}}, _calendar(tmp_path, SUMMER_FOMC), idx)


def test_calendar_not_covering_the_period_raises(tmp_path: Path) -> None:
    # Like the old FOMC file, which started in 2021 and silently left 2020 unfiltered.
    cal = _calendar(tmp_path, [("2021-01-27", "14:00", "FOMC"), ("2021-03-17", "14:00", "FOMC")])
    idx = pd.date_range("2020-01-02", periods=3, freq="5min", tz="UTC")
    with pytest.raises(ValueError, match="doesn't cover"):
        _mask(FOMC_2H, cal, idx, dt.date(2020, 1, 1), dt.date(2020, 12, 31))


def test_new_york_to_utc_handles_daylight_saving() -> None:
    assert new_york_to_utc("2024-07-31", "14:00") == pd.Timestamp("2024-07-31 18:00", tz="UTC")  # EDT
    assert new_york_to_utc("2024-12-18", "14:00") == pd.Timestamp("2024-12-18 19:00", tz="UTC")  # EST


def test_real_calendar_has_every_fomc_statement_2019_to_2026() -> None:
    df = pd.read_csv(EVENT_CALENDAR, dtype=str)
    fomc = df[df["event"] == "FOMC"]
    per_year = fomc["date"].str[:4].value_counts().sort_index().to_dict()
    assert per_year == {"2019": 9, "2020": 10, "2021": 8, "2022": 8, "2023": 8, "2024": 8, "2025": 8, "2026": 8}
    # Missing from the old calendar: three 2023-24 meetings and the 2020 emergency actions.
    assert {"2023-02-01", "2023-11-01", "2024-05-01", "2020-03-03", "2020-03-15", "2020-03-23"} <= set(fomc["date"])
