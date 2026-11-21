"""Workspace and data paths.

The repo is used as a package named `quantlab` inside a workspace folder:
    <workspace>/quantlab/...    (this file is <workspace>/quantlab/paths.py)
"""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
QUANTLAB = WORKSPACE / "quantlab"

DATA_ROOTS = {
    "5m_ohlc": QUANTLAB / "data" / "dukascopy_5m_ohlc",
    "15m_ohlc": QUANTLAB / "data" / "dukascopy_15m_ohlc",
    "5m": QUANTLAB / "data" / "dukascopy_5m",
}
