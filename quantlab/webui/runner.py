"""Backtest execution using Strategy classes."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from .config import CONFIG_PATH, StrategyInfo, get_strategies
from ..configs.schema import validate_config_dict
from ..periods import build_periods
from ..pipeline import load_period_data, run_config_periods
from ..strategies import (
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    TrendStrategy,
    BacktestConfig,
)
from .. import report_periods_equity_only, report_robustness
from ..reporting.generate_trades_report import report_periods_trades_html


def run_backtest(
    strategy_id: str,
    *,
    variant: str | None = None,  # None | "robustness"
    record_executions: bool = False,
) -> tuple[bool, str, Path]:
    """Run a backtest using Strategy classes.
    
    Args:
        strategy_id: Strategy key from get_strategies()
    
    Returns:
        (success, output, report_path)
    """
    strategies = get_strategies()
    if strategy_id not in strategies:
        return False, f"Unknown strategy: {strategy_id}", Path(".")
    
    info = strategies[strategy_id]
    
    try:
        # Define periods (config-driven)
        cfg_path = CONFIG_PATH
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

        # Fail fast on typos / unknown keys.
        cfg = validate_config_dict(cfg)

        # Variant controls the report surface.
        # Robustness currently uses a yearly breakdown under the hood.
        if variant not in {None, "robustness"}:
            raise ValueError(f"Invalid variant: {variant!r}")

        if variant == "robustness":
            cfg = dict(cfg)
            cfg_periods = dict(cfg.get("periods", {}) or {})
            cfg_periods["mode"] = "yearly"
            cfg["periods"] = cfg_periods

        periods = build_periods(cfg)

        costs = cfg.get("costs", {}) or {}
        config = BacktestConfig(
            fee_per_lot=float(costs.get("fee_per_lot", 0.0) or 0.0),
            spread_per_lot=float(costs.get("spread_per_lot", 0.0) or 0.0),
            record_executions=bool(record_executions),
        )
        
        # Run based on strategy type
        if info.strategy_type == "buy_and_hold":
            results = _run_buy_and_hold(periods, config)
        
        elif info.strategy_type == "trend":
            params = info.params or {}
            results = _run_trend_baseline(periods, config, **params)
        
        elif info.strategy_type == "mean_reversion":
            results = _run_mean_reversion(periods, config)
        
        elif info.strategy_type == "best_trend":
            results = _run_best_trend(periods, config, info.config_path)
        
        else:
            return False, f"Unknown strategy type: {info.strategy_type}", Path(".")
        
        # Generate report (variant naming)
        # `variant` is provided by the caller (UI).
        report_path = get_report_path(strategy_id, variant=variant, kind="equity") or (info.output_dir / info.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        baseline_results = None
        if info.strategy_type == "best_trend":
            # Buy & Hold baseline over the same periods (for excess return vs baseline).
            baseline_results = _run_buy_and_hold(periods, config)

        final_title = _generate_report(results, info, report_path, cfg=cfg, baseline_results=baseline_results, variant=variant)

        # Also generate trade breakdown report (convenience for debugging / diagnosis)
        #
        # Policy: we do NOT generate a separate "robustness trades" report.
        # The normal per-period trade report is sufficient, and yearly/robustness
        # runs should stay table-only.
        trades_path = None
        if variant is None:
            trades_path = get_report_path(strategy_id, variant=None, kind="trades")
            if trades_path is not None:
                try:
                    periods_df = results
                    # Trade breakdown report should keep a clean title (avoid dumping the
                    # hyperparam-enriched string into the page title/header).
                    base_title = (final_title.split(" + ", 1)[0].strip() if final_title else info.name)
                    score_exclude = list(((cfg.get("periods", {}) or {}).get("score_exclude") or []) or [])
                    report_periods_trades_html(
                        periods=periods_df,
                        out_path=trades_path,
                        title=base_title + " — trade breakdown",
                        score_exclude=score_exclude,
                    )
                except Exception:
                    # Don't fail the whole UI run if trade report fails.
                    trades_path = None

        output = f"Generated: {report_path}" + (f"\nGenerated: {trades_path}" if trades_path is not None else "")
        return True, output, report_path
    
    except Exception as e:
        import traceback
        return False, f"Error: {e}\n{traceback.format_exc()}", Path(".")


def _buy_and_hold_config(prices: pd.Series, config: BacktestConfig) -> BacktestConfig:
    """Engine config for the buy-and-hold baseline: ~100% of capital in gold, unlevered.

    - lot_per_size makes position_size=1 worth ~initial_capital at the first close.
    - leverage=None turns off the margin check. The baseline is a yardstick, not an
      account. With a 1x check, any uptick before the (lagged) entry bar makes the
      position worth more than equity, so entry is skipped until price returns to the
      first close, or never, if it doesn't.
    """
    px0 = float(prices.iloc[0]) if len(prices) else 1.0
    lot_per_size = float(config.initial_capital) / max(1e-12, px0 * float(config.contract_size_per_lot))
    return dataclasses.replace(config, leverage=None, lot_per_size=lot_per_size)


def _run_buy_and_hold(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
) -> dict[str, pd.DataFrame]:
    """Run buy and hold strategy (baseline: ~100% notional at entry, unlevered).

    See `_buy_and_hold_config` for the sizing and why there is no margin check.
    """
    results = {}

    for name, start, end in periods:
        data = load_period_data("XAUUSD", start, end)
        prices = data["prices"].astype(float)
        result = BuyAndHoldStrategy().run_backtest(prices, config=_buy_and_hold_config(prices, config))
        results[name] = result.df

    return results


def _run_trend_baseline(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
    fast: int = 20,
    slow: int = 100,
) -> dict[str, pd.DataFrame]:
    """Run simple trend strategy (no filters)."""
    results = {}
    
    for name, start, end in periods:
        data = load_period_data("XAUUSD", start, end)
        strategy = TrendStrategy(fast=fast, slow=slow)
        result = strategy.run_backtest(data["prices"], config=config)
        
        results[name] = result.df
    
    return results


def _run_mean_reversion(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
) -> dict[str, pd.DataFrame]:
    """Run mean reversion strategy."""
    results = {}
    
    for name, start, end in periods:
        data = load_period_data("XAUUSD", start, end)
        strategy = MeanReversionStrategy(lookback=50, entry_z=1.0, exit_z=0.0)
        result = strategy.run_backtest(data["prices"], config=config)
        
        results[name] = result.df
    
    return results


def _run_best_trend(
    periods: list[tuple[str, dt.date, dt.date]],
    config: BacktestConfig,
    config_path: Path | None,
) -> dict[str, pd.DataFrame]:
    """Run the best_trend config over `periods` (shared pipeline: quantlab/pipeline.py)."""
    cfg = validate_config_dict(yaml.safe_load(Path(config_path or CONFIG_PATH).read_text(encoding="utf-8")) or {})
    return run_config_periods(cfg, periods, bt_config=config)


def _generate_report(
    results: dict[str, pd.DataFrame],
    info: StrategyInfo,
    report_path: Path,
    *,
    cfg: dict | None = None,
    baseline_results: dict[str, pd.DataFrame] | None = None,
    variant: str | None = None,
) -> str:
    """Generate HTML report.

    Returns the final title string used in the report.
    """
    periods = results

    initial_capital = {name: 1000.0 for name in periods}
    
    title = info.name

    # (removed) legacy title enrichment: configs are pipeline-based; keep titles stable.
    score_exclude = list(((cfg.get("periods", {}) or {}).get("score_exclude") or []) or []) if isinstance(cfg, dict) else []

    baseline_periods = baseline_results or None

    report_fn = report_robustness if (variant == "robustness") else report_periods_equity_only

    report_fn(
        periods=periods,
        baseline_periods=baseline_periods,
        out_path=report_path,
        title=title,
        initial_capital=initial_capital,
        score_exclude=score_exclude,
    )

    return title


def get_report_path(strategy_id: str, *, variant: str | None = None, kind: str = "equity") -> Path | None:
    """Get the expected report path for a strategy.

    kind:
      - "equity": default performance report
      - "trades": trade breakdown report

    variant:
      - None: default report
      - "robustness": robustness report surface (suffix "_robustness")
    """
    strategies = get_strategies()
    if strategy_id not in strategies:
        return None

    info = strategies[strategy_id]
    base = info.output_dir / info.output

    # kind suffix
    if kind.strip().lower() in {"trades", "trade", "trade_breakdown"}:
        base = base.with_name(base.stem + "_trades" + base.suffix)

    # Robustness report is equity-only.
    if variant == "robustness":
        if kind.strip().lower() in {"trades", "trade", "trade_breakdown"}:
            return None
        return base.with_name(base.stem + "_robustness" + base.suffix)

    if variant is not None:
        # Fail fast on unknown variants.
        return None

    return base


def report_exists(strategy_id: str, *, variant: str | None = None, kind: str = "equity") -> bool:
    """Check if a report already exists."""
    path = get_report_path(strategy_id, variant=variant, kind=kind)
    return path is not None and path.exists()


def report_mtime(strategy_id: str, *, variant: str | None = None, kind: str = "equity") -> float | None:
    """Return report modification time (unix seconds) or None."""
    path = get_report_path(strategy_id, variant=variant, kind=kind)
    if path is None or not path.exists():
        return None
    try:
        return path.stat().st_mtime
    except Exception:
        return None