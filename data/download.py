#!/usr/bin/env python3
"""Download Dukascopy data and build the parquet files the backtests read.

Two sources (see dukascopy.py):
- candles (default): 1m BID/ASK candles -> 1m mid OHLC -> 5m/15m OHLC.
  2 requests per symbol-day. --end defaults to *yesterday* (UTC): today's candles
  are incomplete, and the raw cache is never re-fetched, so caching them would
  freeze a partial day.
- ticks: hourly tick files -> 1s quotes -> 5m/15m mid and OHLC.
  24 requests per symbol-day, so backfills are much slower.

Runs incrementally: skips existing files, reuses the raw cache, and keeps going past
days that fail (they are listed at the end; rerun the same command to retry them).

Usage (from the workspace folder, the parent of `quantlab/`):
    # Backfill a period from candles
    .venv/bin/python -m quantlab.data.download --symbols XAUUSD --start 2020-01-01 --end 2020-12-31

    # Last 7 days from ticks
    .venv/bin/python -m quantlab.data.download --source ticks --symbols XAUUSD XAGUSD EURUSD --days 7

Cron example (download recent data daily at 6am UTC):
    0 6 * * * cd /path/to/workspace && .venv/bin/python -m quantlab.data.download --symbols XAUUSD --days 3 >> /var/log/quantlab_download.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from quantlab.data.dukascopy import build_daily_1m_from_candles, build_daily_1s, iter_days, latest_utc_date
from quantlab.data.resampling import (
    resample_dukascopy_1m_to_ohlc,
    resample_dukascopy_1s_to_bars,
    resample_dukascopy_1s_to_ohlc,
)

# Anchor default paths to the workspace root (like the WebUI does).
# This avoids accidental writes to e.g. quantlab/quantlab/data when running from
# inside the quantlab/ folder.
WORKSPACE = Path(__file__).resolve().parents[2]
QUANTLAB = WORKSPACE / "quantlab"

Errors = list[tuple[str, dt.date, str]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download Dukascopy data and build parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--source", choices=("candles", "ticks"), default="candles",
                    help="candles: 1m candles, 2 requests/day (default); ticks: hourly tick files, 24 requests/day")
    ap.add_argument("--symbols", nargs="+", default=["XAUUSD"], help="Symbols to download")
    ap.add_argument("--start", type=dt.date.fromisoformat, default=None, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None,
                    help="End date (YYYY-MM-DD; default: yesterday for candles, today for ticks)")
    ap.add_argument("--days", type=int, default=None, help="Download last N days (alternative to --start/--end)")
    ap.add_argument("--cache-dir", type=Path, default=QUANTLAB / "data" / "dukascopy_raw", help="Raw .bi5 cache")
    ap.add_argument("--out-1m", type=Path, default=QUANTLAB / "data" / "dukascopy_1m", help="1m output dir (candles)")
    ap.add_argument("--out-1s", type=Path, default=QUANTLAB / "data" / "dukascopy_1s", help="1s output dir (ticks)")
    ap.add_argument("--out-5m", type=Path, default=QUANTLAB / "data" / "dukascopy_5m", help="5m mid output dir (ticks)")
    ap.add_argument("--out-5m-ohlc", type=Path, default=QUANTLAB / "data" / "dukascopy_5m_ohlc", help="5m OHLC output dir")
    ap.add_argument("--out-15m", type=Path, default=QUANTLAB / "data" / "dukascopy_15m", help="15m mid output dir (ticks)")
    ap.add_argument("--out-15m-ohlc", type=Path, default=QUANTLAB / "data" / "dukascopy_15m_ohlc", help="15m OHLC output dir")
    ap.add_argument("--no-mid", action="store_true", help="Skip mid files, only OHLC (ticks)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet files (the raw cache is always reused)")
    ap.add_argument("--max-days", type=int, default=None, help="Max days per run (for cron safety)")
    ap.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = ap.parse_args(argv)

    # Determine date range
    if args.source == "candles":
        yesterday = latest_utc_date() - dt.timedelta(days=1)
        end = args.end or yesterday
        if end > yesterday:
            print(f"NOTE: clamping --end {end} to {yesterday} (today's candles are incomplete)")
            end = yesterday
    else:
        end = args.end or latest_utc_date()

    if args.days is not None:
        start = end - dt.timedelta(days=args.days - 1)
    elif args.start is not None:
        start = args.start
    else:
        # Default: last 7 days
        start = end - dt.timedelta(days=6)

    # Limit days if requested (for cron safety)
    if args.max_days is not None:
        max_end = start + dt.timedelta(days=args.max_days - 1)
        end = min(end, max_end)

    if end < start:
        print("Nothing to do: end < start")
        return 0
    days = list(iter_days(start, end))

    if not args.quiet:
        print(
            "DOWNLOAD_START "
            f"source={args.source} symbols={args.symbols} start={start} end={end} days={len(days)} "
            f"out_5m_ohlc={args.out_5m_ohlc} out_15m_ohlc={args.out_15m_ohlc}",
            flush=True,
        )

    errors: Errors = []
    for symbol in args.symbols:
        if not args.quiet:
            print(f"\n== {symbol} ==", flush=True)
        if args.source == "candles":
            errors += _download_candles(symbol, days, args)
        else:
            errors += _download_ticks(symbol, days, args)

    # Summary
    if not args.quiet:
        print(f"\nDOWNLOAD_END errors={len(errors)}")

    if errors:
        for symbol, day, err in errors:
            print(f"FAILED {symbol} {day}: {err}", file=sys.stderr)
        return 1

    return 0


def _download_candles(symbol: str, days: list[dt.date], args: argparse.Namespace) -> Errors:
    """Minute candles -> daily 1m mid OHLC, then 5m/15m OHLC."""
    errors: Errors = []
    t0 = time.monotonic()
    n_new = 0
    for i, day in enumerate(days, start=1):
        try:
            _, n_bars = build_daily_1m_from_candles(
                symbol=symbol,
                day=day,
                cache_dir=args.cache_dir,
                out_dir=args.out_1m,
                overwrite=args.overwrite,
            )
            n_new += int(n_bars >= 0)
        except Exception as e:  # noqa: BLE001
            errors.append((symbol, day, f"download: {e}"))
            if not args.quiet:
                print(f"ERROR {symbol} {day}: {e}", flush=True)

        if not args.quiet and (i % 10 == 0 or i == len(days)):
            el = time.monotonic() - t0
            eta = (el / n_new) * (len(days) - i) if n_new else float("nan")
            print(
                f"{symbol}: {i}/{len(days)} days  (built {n_new}, errors {len(errors)})  "
                f"elapsed {el / 60:.1f} min  eta ~{eta / 60:.0f} min",
                flush=True,
            )

    start, end = days[0], days[-1]
    n_5m = resample_dukascopy_1m_to_ohlc(
        symbol=symbol, start=start, end=end, rule="5m",
        root_in=args.out_1m, root_out=args.out_5m_ohlc, overwrite=args.overwrite,
    )
    n_15m = resample_dukascopy_1m_to_ohlc(
        symbol=symbol, start=start, end=end, rule="15m",
        root_in=args.out_1m, root_out=args.out_15m_ohlc, overwrite=args.overwrite,
    )
    if not args.quiet:
        print(f"{symbol}: {len(days) - len(errors)}/{len(days)} days ok; "
              f"resampled {n_5m} 5m OHLC, {n_15m} 15m OHLC", flush=True)
    return errors


def _download_ticks(symbol: str, days: list[dt.date], args: argparse.Namespace) -> Errors:
    """Ticks -> daily 1s quotes, then 5m/15m mid and OHLC."""
    errors: Errors = []
    start, end = days[0], days[-1]

    # Step 1: Download tick data → build 1s parquet
    total_1s = 0
    for i, day in enumerate(days, start=1):
        try:
            build_daily_1s(
                symbol=symbol,
                day=day,
                cache_dir=args.cache_dir,
                out_dir=args.out_1s,
                overwrite=args.overwrite,
            )
            total_1s += 1

            if not args.quiet and i % 10 == 0:
                print(f"{symbol}: {i}/{len(days)} days downloaded")
        except Exception as e:
            errors.append((symbol, day, f"download: {e}"))
            if not args.quiet:
                print(f"ERROR {symbol} {day}: {e}")

    if not args.quiet:
        print(f"{symbol}: downloaded {total_1s} days")

    # Step 2: Resample to 5m/15m mid and OHLC
    if not args.no_mid:
        n_5m = resample_dukascopy_1s_to_bars(
            symbol=symbol,
            start=start,
            end=end,
            rule="5m",
            root_in=args.out_1s,
            root_out=args.out_5m,
            overwrite=args.overwrite,
        )
        n_15m = resample_dukascopy_1s_to_bars(
            symbol=symbol,
            start=start,
            end=end,
            rule="15m",
            root_in=args.out_1s,
            root_out=args.out_15m,
            overwrite=args.overwrite,
        )
        if not args.quiet:
            print(f"{symbol}: resampled {n_5m} 5m mid, {n_15m} 15m mid")

    n_5m_ohlc = resample_dukascopy_1s_to_ohlc(
        symbol=symbol,
        start=start,
        end=end,
        rule="5m",
        root_in=args.out_1s,
        root_out=args.out_5m_ohlc,
        overwrite=args.overwrite,
    )
    n_15m_ohlc = resample_dukascopy_1s_to_ohlc(
        symbol=symbol,
        start=start,
        end=end,
        rule="15m",
        root_in=args.out_1s,
        root_out=args.out_15m_ohlc,
        overwrite=args.overwrite,
    )
    if not args.quiet:
        print(f"{symbol}: resampled {n_5m_ohlc} 5m OHLC, {n_15m_ohlc} 15m OHLC")

    return errors


if __name__ == "__main__":
    sys.exit(main())
