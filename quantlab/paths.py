"""Repo paths.

    <repo>/quantlab/    code (this file is <repo>/quantlab/paths.py)
    <repo>/data/        datasets (not tracked) + econ_calendar inputs
    <repo>/reports/     generated reports + decision bundles

Relative paths in configs (e.g. `data/econ_calendar/...`) are relative to <repo>.
"""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]  # the repo root
QUANTLAB = WORKSPACE / "quantlab"
DATA = WORKSPACE / "data"
REPORTS = WORKSPACE / "reports"

EVENT_CALENDAR = DATA / "econ_calendar" / "events.csv"  # time_filter events (New York time)

DATA_ROOTS = {
    "5m_ohlc": DATA / "dukascopy_5m_ohlc",
    "15m_ohlc": DATA / "dukascopy_15m_ohlc",
    "5m": DATA / "dukascopy_5m",
}
