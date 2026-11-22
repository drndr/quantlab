# Event calendar

`events.csv` lists the events the time filter can flatten around (`time_filter.events` in the
config). It is maintained by hand, one row per event, in **New York time** exactly as the
official source states it; the filter converts to UTC (daylight saving included) when loading.

| Column | Meaning |
|---|---|
| `date` | Release date (New York) |
| `time_et` | Release time (New York), `HH:MM` |
| `event` | Event code used in the config, e.g. `FOMC` |
| `note` | Optional (e.g. why an unscheduled event happened) |

The filter refuses to run (rather than silently filtering nothing) if a configured event code is
missing from this file, or if the file doesn't cover the backtest period for that code.

## FOMC

Every FOMC policy statement 2020–2026, from the Federal Reserve's meeting calendars
(https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm and `fomchistorical<year>.htm`),
with the release time from each statement's press release: 14:00 for scheduled meetings; the
2020 emergency actions at their actual times. New meetings: add a row.

## Adding other events

Add rows with a new code (e.g. `CPI`, `NFP`) and their release times (US data releases come out
at 08:30 New York time), then give the code a window in the config. FRED's release-date API
(https://fred.stlouisfed.org/docs/api/fred/) lists the dates for CPI, NFP, PCE, GDP and retail sales.
