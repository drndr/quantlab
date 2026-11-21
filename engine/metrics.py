from __future__ import annotations

# (dataclass not used)

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (pd.DataFrame,)):
        raise TypeError("Expected 1D series-like, got DataFrame")
    return pd.Series(x)


def daily_returns_from_equity(
    equity,
    *,
    tz: str | None = "UTC",
) -> pd.Series:
    """Compute daily close-to-close returns from an intraday equity curve.

    This is the canonical return series for Sharpe in Quantlab.

    Notes
    -----
    - Daily grouping is done by resampling to calendar days ("1D") on the equity
      index and taking the last value as the daily close.
    - If you need NY-5pm trading-day boundaries, implement that explicitly and
      keep this function as the single choke point.
    """
    e = _to_series(equity).dropna().astype(float)
    if e.empty:
        return pd.Series(dtype=float)

    if not isinstance(e.index, (pd.DatetimeIndex,)):
        raise TypeError("equity must be indexed by timestamps (DatetimeIndex)")

    # Normalize timestamps to a known timezone for resampling.
    idx = pd.to_datetime(e.index)
    if tz is not None:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
    e = e.copy()
    e.index = idx
    e = e.sort_index()

    daily_close = e.resample("1D").last().dropna()
    return daily_close.pct_change().dropna()


def sharpe(
    equity,
    *,
    annual_days: int = 252,
    tz: str | None = "UTC",
) -> float:
    """Annualized Sharpe ratio computed on **daily** returns.

    Canonical definition in this codebase:
      1) Aggregate equity to daily closes
      2) Compute daily returns (close-to-close)
      3) Sharpe = sqrt(252) * mean(daily_ret) / std(daily_ret)

    This intentionally avoids intraday annualization debates.
    """
    r = daily_returns_from_equity(equity, tz=tz)
    return sharpe_from_returns(r, annual_days=annual_days)


def sharpe_from_returns(
    r: pd.Series,
    *,
    annual_days: int = 252,
) -> float:
    """Annualized Sharpe from a 1D return series."""
    r = _to_series(r).dropna().astype(float)
    if r.empty:
        return float("nan")

    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig == 0.0 or not np.isfinite(sig):
        return float("nan")

    return float(np.sqrt(float(annual_days)) * mu / sig)


def sharpe_bootstrap(
    equity,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    block_size: int = 21,
    annual_days: int = 252,
    tz: str | None = "UTC",
    seed: int | None = 42,
) -> dict[str, float | int]:
    """Block-bootstrap CI for annualized Sharpe (daily equity returns).

    Uses a circular block bootstrap on the *daily* returns derived from equity.

    Returns keys:
      point, lo, hi, ci, n_boot, block_size
    """
    r = daily_returns_from_equity(equity, tz=tz)
    r = r.dropna().astype(float)

    out: dict[str, float | int] = {
        "point": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "ci": float(ci),
        "n_boot": int(n_boot),
        "block_size": int(block_size),
    }

    if len(r) < max(5, 2 * int(block_size)):
        return out

    values = r.to_numpy(dtype=float)
    n = int(values.shape[0])
    bs = int(block_size)
    n_blocks = int(np.ceil(n / bs))

    rng = np.random.default_rng(seed)
    sharpes = np.empty(int(n_boot), dtype=float)

    for i in range(int(n_boot)):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([((np.arange(s, s + bs) % n)) for s in starts])[:n]
        sample = values[idx]
        sharpes[i] = sharpe_from_returns(sample, annual_days=annual_days)

    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return out

    alpha = (1.0 - float(ci)) / 2.0
    out["point"] = float(sharpe_from_returns(values, annual_days=annual_days))
    out["lo"] = float(np.percentile(sharpes, 100.0 * alpha))
    out["hi"] = float(np.percentile(sharpes, 100.0 * (1.0 - alpha)))
    return out


def max_drawdown(equity) -> float:
    e = _to_series(equity).dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())


# (removed) CAGR / volatility / performance_summary wrapper — kept metrics minimal.


# --- Exposure metrics ---


def exposure_from_position(
    bt: pd.DataFrame,
    *,
    pos_col: str = "position",
) -> float:
    """Percent of bars with non-zero exposure.

    Canonical definition (per your spec):
      Exposure% = mean(|position| > 0) * 100

    Notes
    -----
    - Computed on the native bar frequency of the backtest dataframe.
    - Position is treated as "in market" iff abs(position) > 0.
    """
    if bt is None or len(bt) == 0 or pos_col not in bt.columns:
        return float("nan")

    pos = bt[pos_col].fillna(0.0).astype(float)
    return float((pos.abs() > 0.0).mean() * 100.0)


# Trade-level stats (trade count, win rate, avg win/loss, profit factor) live in
# engine/trades.py: build_trade_ledger() + trade_summary(). This module keeps only
# equity-curve and position metrics, so each statistic has one definition.
