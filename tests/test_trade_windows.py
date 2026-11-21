"""Trade stats must cover a trade's full P&L window: entry bar through the first flat bar.

The engine books the P&L of holding over bar t-1 -> t on bar t, so a trade's final
price move and its exit costs land on the first bar where position is already 0.
"""

from __future__ import annotations

import math

import pandas as pd

from quantlab.engine.backtest import backtest_positions_account_margin
from quantlab.engine.trades import build_trade_ledger, extract_trade_log, trade_summary


def _bt_losing_trade() -> pd.DataFrame:
    # 1 oz long on bars 1-3, flat from bar 4; $1 cost per side (0.01 lot at $100/lot).
    # P&L by bar: -1 (entry cost), +2, +1, -3 (103 -> 101 while still held, plus exit cost).
    idx = pd.date_range("2026-01-01", periods=6, freq="5min")
    prices = pd.Series([100, 100, 102, 103, 101, 101], index=idx, dtype=float)
    pos = pd.Series([0, 1, 1, 1, 0, 0], index=idx, dtype=float)
    return backtest_positions_account_margin(
        prices=prices, positions_size=pos, lag=0, fee_per_lot=100.0, spread_per_lot=0.0, use_numba=False
    )


def test_engine_books_example_as_expected() -> None:
    assert _bt_losing_trade()["pnl"].round(9).tolist() == [0.0, -1.0, 2.0, 1.0, -3.0, 0.0]


def test_trade_log_covers_entry_cost_final_move_and_exit_cost() -> None:
    bt = _bt_losing_trade()
    log = extract_trade_log(bt)
    assert len(log) == 1
    t = log.iloc[0]
    assert abs(t["pnl"] - (-1.0)) < 1e-9  # was 0.0 (entry cost dropped)
    assert abs(t["trade_return"] - (999.0 / 1000.0 - 1.0)) < 1e-12  # was +0.002 (a "winner")
    assert t["bars"] == 3
    # Per-trade P&L reconciles with the account.
    assert abs(log["pnl"].sum() - (bt["equity"].iloc[-1] - bt["equity"].iloc[0])) < 1e-9


def test_ledger_costs_include_both_sides() -> None:
    bt = _bt_losing_trade()
    tl = build_trade_ledger(bt)
    assert abs(tl.loc[0, "costs_total"] - bt["costs"].sum()) < 1e-9  # 2.0, was 1.0
    assert not bool(tl.loc[0, "win"])


def test_trade_summary_uses_full_window() -> None:
    s = trade_summary(build_trade_ledger(_bt_losing_trade()))
    assert s["n_trades"] == 1
    assert s["win_rate"] == 0.0  # was 1.0 before the window fix
    assert s["avg_loss"] < 0
    assert math.isnan(s["avg_win"])  # no winning trades
