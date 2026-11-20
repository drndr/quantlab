#!/usr/bin/env python3
"""Download Dukascopy 1-minute candles and build 1m / 5m / 15m mid OHLC parquet.

Why this exists
---------------
The tick path (`dukascopy.py` + `download.py`) needs 24 HTTP requests per
symbol-day. Dukascopy's free feed throttles hard (parallel requests get
HTTP 503), so a multi-year backfill takes days. Minute candles need 2 requests
per symbol-day (BID + ASK), i.e. 12x fewer.

Usage
-----
    # From the workspace root (the folder that contains `quantlab/`):
    .venv/Scripts/python -m quantlab.data.dukascopy_candles --symbols XAUUSD --start 2025-01-01

    # --end defaults to *yesterday* (UTC). Today's file is incomplete, and the
    # raw cache is never re-fetched, so caching it would freeze a partial day.

Output (same layout + schema the backtest loaders read)
-------------------------------------------------------
- data/dukascopy_1m/{SYMBOL}/{YYYY}/{YYYY-MM-DD}.parquet
    ts (minute START, UTC), open, high, low, close (mid),
    bid_close, ask_close, spread_close          (same schema as dukascopy.build_daily_1m)
- data/dukascopy_5m_ohlc/... and data/dukascopy_15m_ohlc/...
    ts (bar END, UTC: closed='left', label='right', as in data/resampling.py),
    open, high, low, close                       (read by data.loaders.load_dukascopy_ohlc)

Semantics vs the tick path
--------------------------
- open / close: mid of the first / last quote in the minute. Same as tick-derived bars.
- high / low:   (bid_high + ask_high) / 2 and (bid_low + ask_low) / 2. Bid and ask
  extremes can occur on different ticks, so this can overstate the true mid range by
  up to the spread change within a minute. The tick path instead takes extremes of
  1-second *last* quotes, which misses intra-second extremes (understates range).
- Minutes with zero volume on either side are dropped. Dukascopy pads closed-market
  minutes (weekends, the daily break) with flat zero-volume candles; the tick path
  produces no bars there, and neither does this.
"""

from __future__ import annotations

import argparse
import datetime as dt
import lzma
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

from quantlab.data.dukascopy import DUKASCOPY_BASE, SYMBOL_SPECS, iter_days

WORKSPACE = Path(__file__).resolve().parents[2]
QUANTLAB = WORKSPACE / "quantlab"

SIDES = ("BID", "ASK")

# Waits (seconds) after a failed request. Dukascopy's throttle windows last tens of
# seconds; retrying quickly just extends them.
BACKOFF_S = (5, 10, 20, 30, 60, 60, 90, 120)

_CANDLE_DTYPE = np.dtype(
    [
        ("t", ">i4"),  # seconds since 00:00 UTC of the file's day (minute start)
        ("open", ">i4"),
        ("close", ">i4"),
        ("low", ">i4"),
        ("high", ">i4"),
        ("vol", ">f4"),
    ]
)

_EMPTY_1M = {
    "ts": pl.Datetime("ns", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "bid_close": pl.Float64,
    "ask_close": pl.Float64,
    "spread_close": pl.Float64,
}


def candle_url(symbol: str, day: dt.date, side: str) -> str:
    # Dukascopy months are 0-indexed in the URL.
    return f"{DUKASCOPY_BASE}/{symbol}/{day.year}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def candle_cache_path(cache_dir: Path, symbol: str, day: dt.date, side: str) -> Path:
    # Same tree as the tick cache ({symbol}/{YYYY}/{MM0}/{DD}/...), different file name.
    return cache_dir / symbol / str(day.year) / f"{day.month - 1:02d}" / f"{day.day:02d}" / f"{side}_candles_min_1.bi5"


def fetch(session: requests.Session, url: str, out_path: Path, *, timeout_s: float = 60.0) -> None:
    """Download `url` to `out_path` atomically, backing off on throttling/errors."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    last_err: Exception | None = None
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            r = session.get(url, timeout=timeout_s)
            r.raise_for_status()
            tmp_path.write_bytes(r.content)
            os.replace(tmp_path, out_path)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < len(BACKOFF_S):
                time.sleep(BACKOFF_S[attempt])

    raise RuntimeError(f"Failed after {len(BACKOFF_S) + 1} tries: {url} ({last_err})") from last_err


def parse_candles_bi5(path: Path, *, price_scale: int) -> dict[str, np.ndarray]:
    """Parse a Dukascopy minute-candle .bi5 file (field order: open, close, low, high)."""
    raw = path.read_bytes()
    if not raw:
        return {k: np.empty(0) for k in ("t", "open", "high", "low", "close", "vol")}

    data = lzma.decompress(raw, format=lzma.FORMAT_AUTO)
    if len(data) % _CANDLE_DTYPE.itemsize != 0:
        raise ValueError(f"Unexpected decompressed size {len(data)} for {path}")

    a = np.frombuffer(data, dtype=_CANDLE_DTYPE)
    scale = float(10**price_scale)
    out = {
        "t": a["t"].astype(np.int64),
        "open": a["open"] / scale,
        "high": a["high"] / scale,
        "low": a["low"] / scale,
        "close": a["close"] / scale,
        "vol": a["vol"].astype(np.float64),
    }

    # Sanity: with the (open, close, low, high) field order, low/high must bracket open/close.
    bad = (out["low"] > np.minimum(out["open"], out["close"]) + 1e-9) | (
        out["high"] < np.maximum(out["open"], out["close"]) - 1e-9
    )
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} candles violate low<=open,close<=high (field order changed?)")
    return out


def build_daily_1m_mid(
    *,
    session: requests.Session,
    symbol: str,
    day: dt.date,
    cache_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Download BID+ASK minute candles for one day and write the 1m mid parquet.

    Returns (out_path, n_bars). Days with no trading are written as empty files so
    they are not re-processed.
    """
    out_path = out_dir / symbol / str(day.year) / f"{day.isoformat()}.parquet"
    if out_path.exists() and not overwrite:
        return out_path, -1

    spec = SYMBOL_SPECS.get(symbol)
    if spec is None:
        raise KeyError(f"Unknown symbol spec for {symbol}. Add to dukascopy.SYMBOL_SPECS.")

    sides: dict[str, dict[str, np.ndarray]] = {}
    for side in SIDES:
        p = candle_cache_path(cache_dir, symbol, day, side)
        if not p.exists():
            fetch(session, candle_url(symbol, day, side), p)
        sides[side] = parse_candles_bi5(p, price_scale=spec.price_scale)

    bid, ask = sides["BID"], sides["ASK"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(bid["t"]) == 0 or len(ask["t"]) == 0:
        pl.DataFrame(schema=_EMPTY_1M).write_parquet(out_path, compression="zstd")
        return out_path, 0

    if not np.array_equal(bid["t"], ask["t"]):
        raise ValueError(f"{symbol} {day}: BID/ASK minute grids differ")

    keep = (bid["vol"] > 0) & (ask["vol"] > 0)
    day_start_ns = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000

    df = pl.DataFrame(
        {
            "ts_ns": day_start_ns + bid["t"][keep] * 1_000_000_000,
            "open": (bid["open"][keep] + ask["open"][keep]) / 2.0,
            "high": (bid["high"][keep] + ask["high"][keep]) / 2.0,
            "low": (bid["low"][keep] + ask["low"][keep]) / 2.0,
            "close": (bid["close"][keep] + ask["close"][keep]) / 2.0,
            "bid_close": bid["close"][keep],
            "ask_close": ask["close"][keep],
        }
    )
    df = (
        df.with_columns(
            pl.from_epoch("ts_ns", time_unit="ns").dt.replace_time_zone("UTC").alias("ts"),
            (pl.col("ask_close") - pl.col("bid_close")).alias("spread_close"),
        )
        .drop("ts_ns")
        .select(list(_EMPTY_1M))
        .sort("ts")
    )

    df.write_parquet(out_path, compression="zstd")
    return out_path, df.height


def resample_1m_to_ohlc(
    *,
    symbol: str,
    start: dt.date,
    end: dt.date,
    rule: str,
    root_in: Path,
    root_out: Path,
    overwrite: bool = False,
) -> int:
    """Resample daily 1m mid OHLC files to `rule` bars (closed='left', label='right')."""
    n_written = 0
    for day in iter_days(start, end):
        in_path = root_in / symbol / str(day.year) / f"{day.isoformat()}.parquet"
        out_path = root_out / symbol / str(day.year) / f"{day.isoformat()}.parquet"
        if not in_path.exists() or (out_path.exists() and not overwrite):
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.read_parquet(in_path).select(["ts", "open", "high", "low", "close"]).sort("ts")
        bars = (
            df.group_by_dynamic(index_column="ts", every=rule, closed="left", label="right")
            .agg(
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
            )
            .drop_nulls()
            .sort("ts")
        )
        bars.write_parquet(out_path)
        n_written += 1
    return n_written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download Dukascopy 1m candles -> 1m/5m/15m mid OHLC parquet")
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD"])
    ap.add_argument("--start", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None, help="YYYY-MM-DD (default: yesterday UTC)")
    ap.add_argument("--cache-dir", type=Path, default=QUANTLAB / "data" / "dukascopy_raw")
    ap.add_argument("--out-1m", type=Path, default=QUANTLAB / "data" / "dukascopy_1m")
    ap.add_argument("--out-5m-ohlc", type=Path, default=QUANTLAB / "data" / "dukascopy_5m_ohlc")
    ap.add_argument("--out-15m-ohlc", type=Path, default=QUANTLAB / "data" / "dukascopy_15m_ohlc")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild parquet outputs (raw cache is always reused)")
    args = ap.parse_args(argv)

    yesterday = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    end = args.end or yesterday
    if end > yesterday:
        print(f"NOTE: clamping --end {end} to {yesterday} (today's candles are incomplete)")
        end = yesterday
    if end < args.start:
        print("Nothing to do: end < start")
        return 0

    days = list(iter_days(args.start, end))
    print(f"CANDLES_START symbols={args.symbols} start={args.start} end={end} days={len(days)}", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; quantlab/0.1)", "Accept": "*/*"})

    errors: list[tuple[str, dt.date, str]] = []
    for symbol in args.symbols:
        t0 = time.monotonic()
        n_done = n_new = 0
        for i, day in enumerate(days, start=1):
            try:
                _, n_bars = build_daily_1m_mid(
                    session=session,
                    symbol=symbol,
                    day=day,
                    cache_dir=args.cache_dir,
                    out_dir=args.out_1m,
                    overwrite=args.overwrite,
                )
                n_done += 1
                n_new += int(n_bars >= 0)
            except Exception as e:  # noqa: BLE001
                errors.append((symbol, day, str(e)))
                print(f"ERROR {symbol} {day}: {e}", flush=True)

            if i % 10 == 0 or i == len(days):
                el = time.monotonic() - t0
                eta = (el / max(n_new, 1)) * (len(days) - i) if n_new else float("nan")
                print(
                    f"{symbol}: {i}/{len(days)} days  (built {n_new}, errors {len(errors)})  "
                    f"elapsed {el / 60:.1f} min  eta ~{eta / 60:.0f} min",
                    flush=True,
                )

        n5 = resample_1m_to_ohlc(symbol=symbol, start=args.start, end=end, rule="5m",
                                 root_in=args.out_1m, root_out=args.out_5m_ohlc, overwrite=args.overwrite)
        n15 = resample_1m_to_ohlc(symbol=symbol, start=args.start, end=end, rule="15m",
                                  root_in=args.out_1m, root_out=args.out_15m_ohlc, overwrite=args.overwrite)
        print(f"{symbol}: {n_done}/{len(days)} days ok; resampled {n5} days to 5m, {n15} days to 15m", flush=True)

    print(f"CANDLES_END errors={len(errors)}", flush=True)
    for symbol, day, err in errors:
        print(f"FAILED {symbol} {day}: {err}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
