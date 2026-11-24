"""rnd simulates and scores on the same starting balance (run and sweep). Needs no market data."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import quantlab.rnd as rnd
from quantlab.pipeline import PreparedPeriod

CFG = {
    "periods": {
        "mode": "three_block",
        "three_block": {
            "p1": {"name": "A", "start": "2024-01-01", "end": "2024-01-31"},
            "p2": {"name": "B", "start": "2024-02-01", "end": "2024-02-29"},
            "p3": {"name": "C", "start": "2024-03-01", "end": "2024-03-31"},
        },
    },
    "costs": {"fee_per_lot": 3.5, "spread_per_lot": 3.5},
    "trend": {"fast": 10, "slow": 30, "ma_kind": "ema"},
    "pipeline": [{"gate": "time_filter", "params": {}}],  # no time_filter block: does nothing
}


@pytest.fixture(autouse=True)
def synthetic_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """One period of waves around $2,000 (1 oz needs about $100 margin at 1:20)."""
    idx = pd.date_range("2024-01-01", periods=12 * 24 * 10, freq="5min", tz="UTC")
    t = np.arange(len(idx))
    prices = pd.Series(2000.0 + 0.01 * t + 15.0 * np.sin(t / 60.0) + 4.0 * np.sin(t / 7.0), index=idx)
    prepared = [PreparedPeriod(name="A", start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 10), prices=prices, context={})]
    monkeypatch.setattr(rnd, "prepare_periods", lambda *a, **k: prepared)


def _run(capital: float, tmp_path: Path, capsys: pytest.CaptureFixture) -> dict:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG), encoding="utf-8")
    rnd.main(["run", "--config", str(cfg_path), "--initial-capital", str(capital)])
    out = json.loads(capsys.readouterr().out)
    assert out["initial_capital"] == capital
    return {"pnl": out["score"]["sum_pnl"], "dd": out["score"]["worst_maxdd"]}


def _sweep(capital: float, tmp_path: Path, capsys: pytest.CaptureFixture) -> dict:
    cfg_path, sweep_path = tmp_path / "cfg.yaml", tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG), encoding="utf-8")
    sweep = {"base": str(cfg_path), "initial_capital": capital, "grid": {"trend.slow": [30]}}
    sweep_path.write_text(yaml.safe_dump(sweep), encoding="utf-8")
    rnd.main(["sweep", "--sweep", str(sweep_path)])
    out = json.loads(capsys.readouterr().out)
    assert out["initial_capital"] == capital
    return {"pnl": out["topk"][0]["sum_pnl"], "dd": out["topk"][0]["worst_maxdd"]}


COMMANDS = pytest.mark.parametrize("command", [_run, _sweep], ids=["run", "sweep"])


@COMMANDS
def test_doubling_the_balance_keeps_the_dollar_profit(command, tmp_path: Path, capsys) -> None:
    one, two = command(1000.0, tmp_path, capsys), command(2000.0, tmp_path, capsys)
    assert abs(one["pnl"]) > 1.0  # it trades
    assert two["pnl"] * 2000.0 == pytest.approx(one["pnl"] * 1000.0)  # fixed lots: same dollars, half the percent
    assert one["dd"] < two["dd"] < 0.0  # same dollar drawdown on a larger account


@COMMANDS
def test_balance_reaches_the_margin_check(command, tmp_path: Path, capsys) -> None:
    # $50 can't cover the margin for 1 oz: no trades, so exactly 0%, not a fake loss against $50.
    assert command(50.0, tmp_path, capsys) == {"pnl": 0.0, "dd": 0.0}


@pytest.mark.parametrize("bad", [0, -1000, float("nan"), float("inf")])
def test_balance_must_be_positive(bad: float, tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        _run(bad, tmp_path, capsys)
    assert "--initial-capital" in capsys.readouterr().err
    with pytest.raises(ValueError, match="positive number"):
        _sweep(bad, tmp_path, capsys)
