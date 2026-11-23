"""Dukascopy datafeed: download raw .bi5 files and build daily parquet files.

Two sources for the same feed:
- ticks:           {HH}h_ticks.bi5, one file per hour (24 requests per symbol-day)
                   -> build_daily_1s (1s last quotes), build_daily_1m
- minute candles:  {BID,ASK}_candles_min_1.bi5 (2 requests per symbol-day)
                   -> build_daily_1m_from_candles

Dukascopy's free feed throttles hard (parallel requests get HTTP 503), so a
multi-year backfill from ticks takes days; candles need 12x fewer requests.

CLI: python -m quantlab.data.download --source candles|ticks
"""

from __future__ import annotations

import datetime as dt
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lzma
import numpy as np
import polars as pl
import requests


DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"


@dataclass(frozen=True)
class DukascopySymbolSpec:
    symbol: str
    price_scale: int  # price = int_price / 10**price_scale


SYMBOL_SPECS: dict[str, DukascopySymbolSpec] = {
    "XAUUSD": DukascopySymbolSpec("XAUUSD", price_scale=3),
    "XAGUSD": DukascopySymbolSpec("XAGUSD", price_scale=3),
    # FX majors
    "EURUSD": DukascopySymbolSpec("EURUSD", price_scale=5),
}


def dukascopy_tick_url(symbol: str, day: dt.date, hour: int) -> str:
    # Dukascopy months are 0-indexed in the URL.
    month0 = day.month - 1
    return f"{DUKASCOPY_BASE}/{symbol}/{day.year}/{month0:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"


def _cache_path(cache_dir: Path, symbol: str, day: dt.date, hour: int) -> Path:
    month0 = day.month - 1
    return cache_dir / symbol / str(day.year) / f"{month0:02d}" / f"{day.day:02d}" / f"{hour:02d}h_ticks.bi5"


# Waits (seconds) after a failed request. Dukascopy's throttle windows last tens of
# seconds; retrying quickly just extends them.
BACKOFF_S = (5, 10, 20, 30, 60, 60, 90, 120)

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """One HTTP session for all downloads (reuses the connection)."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Dukascopy sometimes behaves better with a browser-ish UA.
        _session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; quantlab/0.1)", "Accept": "*/*"})
    return _session


def download_bi5(url: str, out_path: Path, *, timeout_s: float = 60.0) -> None:
    """Download `url` to `out_path` atomically, backing off on throttling/errors."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    last_err: Exception | None = None
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            r = _get_session().get(url, timeout=timeout_s)
        except requests.RequestException as e:  # connection error, timeout
            last_err = e
        else:
            if r.ok:
                tmp_path.write_bytes(r.content)
                os.replace(tmp_path, out_path)
                return
            last_err = requests.HTTPError(f"HTTP {r.status_code}", response=r)
            if r.status_code < 500 and r.status_code != 429:
                break  # e.g. 404 for an hour that hasn't happened yet: retrying won't help
        if attempt < len(BACKOFF_S):
            time.sleep(BACKOFF_S[attempt])

    raise RuntimeError(f"Download failed: {url} ({last_err})") from last_err


_TICK_DTYPE = np.dtype(
    [
        ("ms", ">i4"),
        ("ask", ">i4"),
        ("bid", ">i4"),
        ("ask_vol", ">f4"),
        ("bid_vol", ">f4"),
    ]
)


def parse_ticks_bi5(path: Path) -> np.ndarray:
    """Parse a Dukascopy *tick* .bi5 file.

    Returns a numpy structured array with fields: ms, ask, bid, ask_vol, bid_vol.

    Notes:
    - Empty files (0 bytes) are common on weekends/holidays.
    """
    raw = path.read_bytes()
    if len(raw) == 0:
        return np.empty((0,), dtype=_TICK_DTYPE)

    # FORMAT_AUTO handles Dukascopy .bi5 (typically LZMA "alone" format)
    decompressed = lzma.decompress(raw, format=lzma.FORMAT_AUTO)
    if len(decompressed) == 0:
        return np.empty((0,), dtype=_TICK_DTYPE)

    if len(decompressed) % _TICK_DTYPE.itemsize != 0:
        raise ValueError(f"Unexpected decompressed size: {len(decompressed)} bytes")

    return np.frombuffer(decompressed, dtype=_TICK_DTYPE)


def ticks_to_1s_quotes(
    ticks: np.ndarray,
    *,
    symbol: str,
    day: dt.date,
    hour: int,
) -> pl.DataFrame:
    """Convert tick array -> 1s last-quote DataFrame for that hour.

    Output columns: ts (UTC), bid, ask, mid, spread
    """
    if ticks.size == 0:
        return pl.DataFrame(schema={
            "ts": pl.Datetime("ns", time_zone="UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "mid": pl.Float64,
            "spread": pl.Float64,
        })

    spec = SYMBOL_SPECS.get(symbol)
    if spec is None:
        raise KeyError(f"Unknown symbol spec for {symbol}. Add to SYMBOL_SPECS.")

    base = dt.datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=dt.UTC)
    base_ns = int(base.timestamp() * 1_000_000_000)

    ms = ticks["ms"].astype(np.int64)
    ts_ns = base_ns + ms * 1_000_000

    scale = 10 ** spec.price_scale
    bid = ticks["bid"].astype(np.float64) / scale
    ask = ticks["ask"].astype(np.float64) / scale

    df = pl.DataFrame({
        "ts_ns": ts_ns,
        "bid": bid,
        "ask": ask,
    })
    df = df.with_columns(
        pl.col("ts_ns").cast(pl.Datetime("ns", time_zone="UTC")).alias("ts"),
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
    ).drop("ts_ns")

    # Dukascopy ticks are typically ordered, but sort just in case.
    df = df.sort("ts")

    # 1-second last quote
    df = (
        df.with_columns(pl.col("ts").dt.truncate("1s").alias("ts_1s"))
        .group_by("ts_1s")
        .agg(
            pl.col("bid").last().alias("bid"),
            pl.col("ask").last().alias("ask"),
            pl.col("mid").last().alias("mid"),
            pl.col("spread").last().alias("spread"),
        )
        .rename({"ts_1s": "ts"})
        .sort("ts")
    )

    return df


def iter_days(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    if end < start:
        raise ValueError("end must be >= start")
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def build_daily_1s(
    *,
    symbol: str,
    day: dt.date,
    cache_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    timeout_s: float = 60.0,
) -> Path:
    """Download ticks (if needed) and write a daily 1s parquet file.

    Output file: {out_dir}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet
    """
    out_path = out_dir / symbol / f"{day.year}" / f"{day.isoformat()}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return out_path

    hour_frames: list[pl.DataFrame] = []
    for hour in range(24):
        url = dukascopy_tick_url(symbol, day, hour)
        p = _cache_path(cache_dir, symbol, day, hour)
        if not p.exists():
            download_bi5(url, p, timeout_s=timeout_s)
        ticks = parse_ticks_bi5(p)
        df1s = ticks_to_1s_quotes(ticks, symbol=symbol, day=day, hour=hour)
        if df1s.height:
            hour_frames.append(df1s)

    if hour_frames:
        df_day = pl.concat(hour_frames).unique(subset=["ts"], keep="last").sort("ts")
    else:
        df_day = pl.DataFrame(schema={
            "ts": pl.Datetime("ns", time_zone="UTC"),
            "bid": pl.Float64,
            "ask": pl.Float64,
            "mid": pl.Float64,
            "spread": pl.Float64,
        })

    df_day.write_parquet(out_path, compression="zstd")
    return out_path


# Daily 1m parquet schema, from ticks (build_daily_1m) or candles (build_daily_1m_from_candles).
_SCHEMA_1M = {
    "ts": pl.Datetime("ns", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "bid_close": pl.Float64,
    "ask_close": pl.Float64,
    "spread_close": pl.Float64,
}


def build_daily_1m(
    *,
    symbol: str,
    day: dt.date,
    cache_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    timeout_s: float = 60.0,
) -> Path:
    """Download ticks (if needed) and write a daily 1-minute OHLC parquet file.

    Same output as `build_daily_1m_from_candles` (which needs 2 requests, not 24).

    Output file: {out_dir}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet

    Columns:
      ts (UTC minute), open, high, low, close, bid_close, ask_close, spread_close

    Notes:
    - This is still "direct" from Dukascopy ticks (not resampling your 1s files).
    - Minutes with no ticks are omitted for now (we can forward-fill later if desired).
    """
    out_path = out_dir / symbol / f"{day.year}" / f"{day.isoformat()}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return out_path

    minute_frames: list[pl.DataFrame] = []
    for hour in range(24):
        url = dukascopy_tick_url(symbol, day, hour)
        p = _cache_path(cache_dir, symbol, day, hour)
        if not p.exists():
            download_bi5(url, p, timeout_s=timeout_s)
        ticks = parse_ticks_bi5(p)
        if ticks.size == 0:
            continue

        spec = SYMBOL_SPECS.get(symbol)
        if spec is None:
            raise KeyError(f"Unknown symbol spec for {symbol}. Add to SYMBOL_SPECS.")

        base = dt.datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=dt.UTC)
        base_ns = int(base.timestamp() * 1_000_000_000)
        ms = ticks["ms"].astype(np.int64)
        ts_ns = base_ns + ms * 1_000_000

        scale = 10 ** spec.price_scale
        bid = ticks["bid"].astype(np.float64) / scale
        ask = ticks["ask"].astype(np.float64) / scale
        mid = (bid + ask) / 2.0
        spread = ask - bid

        df = pl.DataFrame({
            "ts_ns": ts_ns,
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "spread": spread,
        }).with_columns(
            pl.col("ts_ns").cast(pl.Datetime("ns", time_zone="UTC")).alias("ts"),
        ).drop("ts_ns").sort("ts")

        # 1-minute bucket OHLC on mid; keep last bid/ask/spread for close.
        df = (
            df.with_columns(pl.col("ts").dt.truncate("1m").alias("ts_1m"))
            .group_by("ts_1m")
            .agg(
                pl.col("mid").first().alias("open"),
                pl.col("mid").max().alias("high"),
                pl.col("mid").min().alias("low"),
                pl.col("mid").last().alias("close"),
                pl.col("bid").last().alias("bid_close"),
                pl.col("ask").last().alias("ask_close"),
                pl.col("spread").last().alias("spread_close"),
            )
            .rename({"ts_1m": "ts"})
            .sort("ts")
        )
        if df.height:
            minute_frames.append(df)

    if minute_frames:
        df_day = pl.concat(minute_frames).unique(subset=["ts"], keep="last").sort("ts")
    else:
        df_day = pl.DataFrame(schema=_SCHEMA_1M)

    df_day.write_parquet(out_path, compression="zstd")
    return out_path


def latest_utc_date() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


# ---------------------------------------------------------------------------
# Minute candles (alternative to ticks for 1m bars)
# ---------------------------------------------------------------------------

CANDLE_SIDES = ("BID", "ASK")

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


def dukascopy_candle_url(symbol: str, day: dt.date, side: str) -> str:
    # Dukascopy months are 0-indexed in the URL.
    month0 = day.month - 1
    return f"{DUKASCOPY_BASE}/{symbol}/{day.year}/{month0:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def _candle_cache_path(cache_dir: Path, symbol: str, day: dt.date, side: str) -> Path:
    # Same tree as the tick cache, different file name.
    month0 = day.month - 1
    return cache_dir / symbol / str(day.year) / f"{month0:02d}" / f"{day.day:02d}" / f"{side}_candles_min_1.bi5"


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


def build_daily_1m_from_candles(
    *,
    symbol: str,
    day: dt.date,
    cache_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    timeout_s: float = 60.0,
) -> tuple[Path, int]:
    """Download BID+ASK minute candles (if needed) and write the daily 1m mid parquet.

    Same output as `build_daily_1m` (from ticks):
      {out_dir}/{symbol}/{YYYY}/{YYYY-MM-DD}.parquet
      ts (minute start, UTC), open, high, low, close (mid), bid_close, ask_close, spread_close

    Returns (out_path, n_bars); n_bars is -1 when the file already existed. Days with
    no trading are written as empty files so they are not re-processed.

    Semantics vs the tick path:
    - open / close: mid of the first / last quote in the minute, as from ticks.
    - high / low: (bid_high + ask_high) / 2 and (bid_low + ask_low) / 2. Bid and ask
      extremes can occur on different ticks, so this can overstate the true mid range
      by up to the spread change within a minute.
    - Minutes with zero volume on either side are dropped. Dukascopy pads closed-market
      minutes (weekends, the daily break) with flat zero-volume candles; ticks produce
      no bars there either.
    """
    out_path = out_dir / symbol / str(day.year) / f"{day.isoformat()}.parquet"
    if out_path.exists() and not overwrite:
        return out_path, -1

    spec = SYMBOL_SPECS.get(symbol)
    if spec is None:
        raise KeyError(f"Unknown symbol spec for {symbol}. Add to SYMBOL_SPECS.")

    sides: dict[str, dict[str, np.ndarray]] = {}
    for side in CANDLE_SIDES:
        p = _candle_cache_path(cache_dir, symbol, day, side)
        if not p.exists():
            download_bi5(dukascopy_candle_url(symbol, day, side), p, timeout_s=timeout_s)
        sides[side] = parse_candles_bi5(p, price_scale=spec.price_scale)

    bid, ask = sides["BID"], sides["ASK"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(bid["t"]) == 0 or len(ask["t"]) == 0:
        pl.DataFrame(schema=_SCHEMA_1M).write_parquet(out_path, compression="zstd")
        return out_path, 0

    if not np.array_equal(bid["t"], ask["t"]):
        raise ValueError(f"{symbol} {day}: BID/ASK minute grids differ")

    keep = (bid["vol"] > 0) & (ask["vol"] > 0)
    day_start_ns = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()) * 1_000_000_000

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
        .select(list(_SCHEMA_1M))
        .sort("ts")
    )

    df.write_parquet(out_path, compression="zstd")
    return out_path, df.height
