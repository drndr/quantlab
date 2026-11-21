"""Runs load only the inputs the configured gates read."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import yaml

from quantlab.pipeline import PreparedPeriod, data_needs, run_config_periods

EMA_SEP = {"gate": "ema_sep", "params": {"ema_fast": 50, "ema_slow": 300, "atr_n": 20, "sep_k": 0.07}}
CORR = {"gate": "corr", "params": {}}
TIME = {"gate": "time_filter", "params": {}}


def _cfg(*gates: dict) -> dict:
    return {"trend": {"fast": 30, "slow": 110, "ma_kind": "ema"}, "pipeline": list(gates)}


def test_htf_gate_needs_15m_bars_only() -> None:
    assert data_needs(_cfg(EMA_SEP, TIME)) == {"need_htf": True, "need_corr": False}


def test_corr_gate_needs_cross_asset_prices() -> None:
    assert data_needs(_cfg(TIME, CORR)) == {"need_htf": False, "need_corr": True}


def test_time_filter_only_needs_nothing_extra() -> None:
    assert data_needs(_cfg(TIME)) == {"need_htf": False, "need_corr": False}


def test_canonical_config_does_not_need_xag_or_eur() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "trend_based" / "current.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert data_needs(cfg) == {"need_htf": True, "need_corr": False}


def test_sweep_candidate_needing_unloaded_data_fails_loudly() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="5min", tz="UTC")
    prepared = [
        PreparedPeriod(
            name="P",
            start=dt.date(2026, 1, 1),
            end=dt.date(2026, 1, 1),
            prices=pd.Series(100.0, index=idx),
            context={"bars_15m": None, "prices_xag": None, "prices_eur": None},
        )
    ]
    with pytest.raises(ValueError, match="prices_xag"):
        run_config_periods(_cfg(TIME, CORR), prepared=prepared)
