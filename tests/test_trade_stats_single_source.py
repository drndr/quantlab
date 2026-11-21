"""Every trade statistic has one definition (engine.trades.trade_summary), and the
equity report, the trades report and the per-bucket tables all read it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.engine.backtest import backtest_positions_account_margin
from quantlab.engine.trades import agg_trade_table, build_trade_ledger, trade_summary
from quantlab.reporting.generate_equity_report import report_periods_equity_only
from quantlab.reporting.generate_trades_report import report_periods_trades_html


def _bt_three_trades() -> pd.DataFrame:
    # 1 oz, $1 cost per side (0.01 lot at $100/lot). Net trade P&L: +3, -5, +4.
    idx = pd.date_range("2026-01-01", periods=12, freq="5min")
    prices = pd.Series([100, 100, 103, 105, 105, 105, 104, 102, 102, 102, 106, 108], index=idx, dtype=float)
    pos = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], index=idx, dtype=float)
    return backtest_positions_account_margin(
        prices=prices, positions_size=pos, lag=0, fee_per_lot=100.0, spread_per_lot=0.0, use_numba=False
    )


def test_summary_reconciles_with_account() -> None:
    bt = _bt_three_trades()
    s = trade_summary(build_trade_ledger(bt))
    assert s["n_trades"] == 3
    assert abs(s["win_rate"] - 2 / 3) < 1e-12
    assert abs(s["sum_pnl"] - (bt["equity"].iloc[-1] - bt["equity"].iloc[0])) < 1e-9  # +2
    assert abs(s["profit_factor"] - (3 + 4) / 5) < 1e-9  # dollars
    assert abs(s["sum_costs"] - bt["costs"].sum()) < 1e-9


def test_bucket_table_uses_the_same_definitions() -> None:
    tl = build_trade_ledger(_bt_three_trades())
    s = trade_summary(tl)
    row = agg_trade_table(tl.assign(bucket="all"), "bucket").iloc[0]
    for k, v in s.items():
        assert abs(float(row[k]) - v) < 1e-12, k


def test_equity_and_trades_reports_show_the_same_win_rate(tmp_path: Path) -> None:
    bt = _bt_three_trades()
    eq_html = report_periods_equity_only(
        periods={"P": bt}, out_path=tmp_path / "eq.html", title="T", initial_capital=1000.0
    ).read_text(encoding="utf-8")
    tr_html = report_periods_trades_html(periods={"P": bt}, out_path=tmp_path / "tr.html", title="T").read_text(
        encoding="utf-8"
    )
    assert "66.67%" in eq_html and "66.67%" in tr_html
