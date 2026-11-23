# Golden regression

Checks that the code still trades exactly as before. Headline metrics can stay similar while the
execution semantics drift, so this compares **position and equity series**, bar by bar.

`test_golden.py` runs two weeks of XAUUSD (29 July – 9 August 2024) with one small case per rule:
each gate, the time filter, the margin check, the warm-up, and several gates together. The settings
are simple round numbers, independent of the live config, so tuning the strategy never touches this
test. When it fails, the failing case names the rule that changed.

- `position` must match exactly, `equity` to 1e-10; `returns_net` must be the return of `equity`.
- Every gate in the schema needs a case (`corr` is the exception: it needs XAGUSD and EURUSD data).
- Every case must trade differently from its reference case, so no case can pass by doing nothing.

## Run

Needs local XAUUSD 5m/15m OHLC data for 24 July – 9 August 2024 (the series check is skipped without it).

```bash
pytest tests/regression
```

## Update

Only when you *intend* to change behaviour. Say in the commit message which semantic change it was.
If you're unsure, don't update: look at the first differing bar in the failure message.

```bash
python tests/regression/test_golden.py --update
```
