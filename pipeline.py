"""Run a strategy config over periods: one implementation for the webapp, rnd and tests.

    prepare_periods(cfg)          load each period's inputs once (only the data the gates read)
    run_config_periods(cfg, ...)  time filter -> strategy -> engine, for each period

Every caller (webui runner, rnd CLI and sweeps, regression tests) goes through these
functions, so they all run exactly the same pipeline.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data.loaders import load_dukascopy_ohlc
from .paths import DATA_ROOTS, WORKSPACE
from .periods import build_periods
from .strategies import BacktestConfig, TrendStrategyWithGates
from .time_filter import (
    EventWindow,
    build_allow_mask_from_econ_calendar,
    build_allow_mask_from_events,
    build_allow_mask_from_months,
)

Period = tuple[str, dt.date, dt.date]

# The warm-up window may open on a weekend or market holiday, so its first bar can be
# a few days after the requested load start without any data being missing.
WARMUP_SLACK_DAYS = 3


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_period_data(
    symbol: str,
    start: dt.date,
    end: dt.date,
    *,
    need_htf: bool = False,
    need_corr: bool = False,
    corr_symbol: str = "XAGUSD",
    corr2_symbol: str = "EURUSD",
) -> dict[str, Any]:
    """Load all data needed for a period.

    Returns:
        dict with keys: prices, bars_5m, bars_15m (optional), prices_xag, prices_eur
    """
    root_5m = DATA_ROOTS["5m_ohlc"]
    root_15m = DATA_ROOTS["15m_ohlc"]

    bars_5m = load_dukascopy_ohlc(symbol=symbol, start=start, end=end, root=root_5m)
    prices = bars_5m["close"].astype(float)

    data = {
        "prices": prices,
        "bars_5m": bars_5m,
    }

    if need_htf:
        bars_15m = load_dukascopy_ohlc(symbol=symbol, start=start, end=end, root=root_15m)
        data["bars_15m"] = bars_15m

    if need_corr:
        xag = load_dukascopy_ohlc(symbol=corr_symbol, start=start, end=end, root=root_5m)
        data["prices_xag"] = xag["close"].astype(float)

        eur = load_dukascopy_ohlc(symbol=corr2_symbol, start=start, end=end, root=root_5m)
        data["prices_eur"] = eur["close"].astype(float)

    return data


def data_needs(cfg: dict) -> dict[str, bool]:
    """Which optional inputs the configured gate pipeline reads.

    Derived from the strategy's own data_requirements, so data for gates that are not
    in the pipeline (e.g. XAGUSD/EURUSD for `corr`) is never loaded or required.
    Returns keyword arguments for `load_period_data`.
    """
    reqs = TrendStrategyWithGates.from_config(cfg).data_requirements
    return {"need_htf": "bars_15m" in reqs, "need_corr": "prices_xag" in reqs}


@dataclass(frozen=True)
class PreparedPeriod:
    """One period's inputs: main prices plus the context the gates read."""

    name: str
    start: dt.date
    end: dt.date
    prices: pd.Series
    context: dict[str, Any]
    load_start: dt.date | None = None  # earlier than `start` when warming up indicators


def prepare_periods(
    cfg: dict,
    periods: list[Period] | None = None,
    *,
    warmup_days: int | None = None,
) -> list[PreparedPeriod]:
    """Load each period's inputs once (sweeps reuse them across candidates).

    `periods` defaults to the config's own periods (`build_periods(cfg)`).
    `warmup_days` defaults to the config's `periods.warmup_days` (0 = cold start).
    """
    periods = build_periods(cfg) if periods is None else periods
    if warmup_days is None:
        warmup_days = int((cfg.get("periods", {}) or {}).get("warmup_days", 0) or 0)
    symbol = cfg.get("symbol", "XAUUSD")
    needs = data_needs(cfg)

    out: list[PreparedPeriod] = []
    for name, start, end in periods:
        load_start = start - dt.timedelta(days=warmup_days)
        data = load_period_data(
            symbol,
            load_start,
            end,
            **needs,
            corr_symbol=cfg.get("corr_symbol", "XAGUSD"),
            corr2_symbol=cfg.get("corr2_symbol", "EURUSD"),
        )
        # Guard: the data may not reach back far enough (e.g. the dataset starts on the
        # first period's start). Run with whatever warm-up exists, and say so.
        first_day = data["prices"].index[0].date()
        if warmup_days and first_day > load_start + dt.timedelta(days=WARMUP_SLACK_DAYS):
            have = max(0, (start - first_day).days)
            warnings.warn(
                f"{name}: warm-up wants {warmup_days} days of {symbol} data from {load_start}, "
                f"but it starts {first_day}; running with {have} days of warm-up",
                stacklevel=2,
            )
            load_start = min(first_day, start)
        context = {
            "bars_15m": data.get("bars_15m"),
            "prices_xag": data.get("prices_xag"),
            "prices_eur": data.get("prices_eur"),
        }
        out.append(
            PreparedPeriod(name=name, start=start, end=end, prices=data["prices"], context=context, load_start=load_start)
        )
    return out


# ---------------------------------------------------------------------------
# Time filter
# ---------------------------------------------------------------------------


def load_fomc_mask(
    index: pd.DatetimeIndex,
    start: dt.date,
    end: dt.date,
    fomc_days_path: Path,
    fomc_cfg: dict | None = None,
) -> pd.Series | None:
    """Build FOMC allow mask for the period."""
    if not fomc_days_path.exists():
        return None

    fomc_cfg = fomc_cfg or {}
    utc_hhmm = fomc_cfg.get("utc_hhmm", "19:00")
    pre_hours = fomc_cfg.get("pre_hours", 2.0)
    post_hours = fomc_cfg.get("post_hours", 0.5)

    df = pd.read_csv(fomc_days_path)
    if "date" not in df.columns:
        return None

    days = [dt.date.fromisoformat(str(x)) for x in df["date"].tolist()]
    days = [d for d in days if start <= d <= end]

    if not days:
        return None

    hh, mm = (int(x) for x in utc_hhmm.split(":"))
    events = [
        EventWindow(
            ts=pd.Timestamp(dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=dt.UTC)),
            pre=dt.timedelta(hours=pre_hours),
            post=dt.timedelta(hours=post_hours),
        )
        for d in days
    ]

    return build_allow_mask_from_events(index, events=events)


def load_time_filter_mask(
    index: pd.DatetimeIndex,
    start: dt.date,
    end: dt.date,
    *,
    cfg: dict,
    workspace: Path,
) -> pd.Series | None:
    """Load allow mask for the configured time_filter kind.

    Supported kinds:
    - fomc
    - econ_calendar
    - months (months-only force-flat; no event windows)
    """
    tf = cfg.get("time_filter", {}) or {}
    kind = (tf.get("kind") or "fomc").strip().lower()

    # Optional month blocking overlay (force-flat semantics).
    months_cfg = tf.get("months", {}) or {}
    block_months = months_cfg.get("block") or []
    m_mask = build_allow_mask_from_months(index, block=block_months)

    if kind == "months":
        return pd.Series(m_mask, index=index).astype(bool)

    if kind == "fomc":
        fomc_cfg = tf.get("fomc", {}) or {}
        days_csv = fomc_cfg.get("days_csv", "quantlab/data/econ_calendar/fomc_decision_days.csv")
        mask = load_fomc_mask(index, start, end, workspace / str(days_csv), fomc_cfg)

        if mask is None:
            return m_mask
        return (pd.Series(mask, index=index).astype(bool) & pd.Series(m_mask, index=index).astype(bool)).astype(bool)

    if kind == "econ_calendar":
        ec = tf.get("econ_calendar", {}) or {}
        csv_rel = ec.get("csv", "quantlab/data/econ_calendar/usd_important_events.csv")
        rules = ec.get("rules", {}) or {}

        csv_path = str((workspace / str(csv_rel)).resolve())

        mask = build_allow_mask_from_econ_calendar(
            index,
            start=start,
            end=end,
            csv_path=csv_path,
            rules=rules,
        )

        if mask is None:
            return m_mask
        return (pd.Series(mask, index=index).astype(bool) & pd.Series(m_mask, index=index).astype(bool)).astype(bool)

    # Unknown kind -> no mask
    return None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def backtest_config(cfg: dict, *, record_executions: bool = False) -> BacktestConfig:
    """Engine settings from a config (costs); everything else uses BacktestConfig defaults."""
    costs = cfg.get("costs", {}) or {}
    return BacktestConfig(
        fee_per_lot=float(costs.get("fee_per_lot", 0.0) or 0.0),
        spread_per_lot=float(costs.get("spread_per_lot", 0.0) or 0.0),
        record_executions=bool(record_executions),
    )


def run_config_periods(
    cfg: dict,
    periods: list[Period] | None = None,
    *,
    prepared: list[PreparedPeriod] | None = None,
    bt_config: BacktestConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run `cfg` over each period and return {period name: backtest dataframe}.

    cfg:       config dict (callers validate it; see configs.schema)
    periods:   [(name, start, end)]; defaults to the config's own periods
    prepared:  inputs from `prepare_periods` to reuse (sweeps); loaded when omitted
    bt_config: engine settings; defaults to `backtest_config(cfg)`
    """
    if prepared is None:
        prepared = prepare_periods(cfg, periods)
    if bt_config is None:
        bt_config = backtest_config(cfg)

    # Reused inputs (sweeps) were loaded for the base config. Fail loudly if this
    # config's gates need data that wasn't loaded (the gates would silently no-op).
    needs = data_needs(cfg)
    for p in prepared:
        missing = [
            k for k, need in (("bars_15m", needs["need_htf"]), ("prices_xag", needs["need_corr"]))
            if need and p.context.get(k) is None
        ]
        if missing:
            raise ValueError(f"{p.name}: config needs {missing}, which were not loaded for this run")

    out: dict[str, pd.DataFrame] = {}
    for p in prepared:
        load_start = p.load_start or p.start
        mask = load_time_filter_mask(pd.DatetimeIndex(p.prices.index), load_start, p.end, cfg=cfg, workspace=WORKSPACE)
        strategy = TrendStrategyWithGates.from_config(cfg, allow_mask=mask)
        result = strategy.run_backtest(
            p.prices,
            context={**p.context, "allow_mask": mask},
            config=bt_config,
            # With warm-up, trade and measure only from the period start.
            evaluate_from=pd.Timestamp(p.start, tz="UTC") if load_start < p.start else None,
        )
        out[p.name] = result.df
    return out
