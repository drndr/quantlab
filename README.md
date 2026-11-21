# Quantlab - Trading Strategy R&D

Quantlab is a modular trading-strategy development and backtesting framework with a unified execution engine and a config-driven gate pipeline.
It includes a WebUI runner for HTML reports and an agent-friendly `quantlab.rnd` CLI.

## Architecture

- **Single backtest engine** — `engine/backtest.py` handles all position accounting, margin, and PnL computation
- **Composable gates** — Strategy behavior is assembled from small, testable gate classes
- **Config-as-truth** — YAML configs define the full pipeline; no hidden logic in code
- **Agent-friendly CLI** — `quantlab.rnd` outputs JSON, designed for automated R&D loops

## File Structure

```
quantlab/
├── engine/                  # Backtest engine + metrics + trade extraction
│   ├── backtest.py          # Core loop (python + optional numba)
│   ├── metrics.py           # Equity-curve metrics: Sharpe, max-DD, exposure
│   └── trades.py            # Trade log, ledger, trade stats (trade_summary)
├── strategies/              # Strategy classes + gates
│   ├── base.py              # StrategyBase, BacktestResult, BacktestConfig
│   ├── buy_and_hold.py
│   ├── mean_reversion.py
│   ├── trend_following.py   # TrendStrategy, TrendStrategyWithGates
│   └── gates/               # Composable gate implementations
├── time_filter/             # Time-blocking infra (FOMC + econ calendar + month blocks)
├── data/
│   ├── dukascopy.py         # Dukascopy download: ticks (1s/1m) or minute candles (1m)
│   ├── resampling.py        # 5m/15m mid + OHLC
│   ├── download.py          # CLI: download + resample (--source candles|ticks)
│   ├── loaders.py           # OHLC parquet loader used by backtests
│   └── validate.py          # CLI: check integrity
├── configs/
│   ├── schema.py            # Pydantic validation for canonical config
│   └── trend_based/
│       ├── reference/       # Archived reference configs
│       └── current.yaml     # Canonical config
├── reporting/               # Report generation (HTML)
├── reports/                 # Output reports + decision bundles
├── tests/                   # Unit tests + regression (golden series)
├── webui/                   # Browser interface
├── rnd.py                   # Low-token CLI runner (agent-friendly)
├── pipeline.py              # Shared run pipeline: data -> time filter -> strategy -> engine
├── periods.py / paths.py    # Period building, workspace + data paths
└── check.sh                 # Unit + regression checks (one command)
```

## Quick Start

Run commands from the workspace folder: the parent of this repo, which must be checked out as `quantlab/` (the package name), with `.venv/` next to it.

### WebUI (human inspection)
```bash
.venv/bin/python -m quantlab.webui.backtest_ui --port 8080
```

### Report generation (canonical HTML outputs)
```bash
.venv/bin/python -c "from quantlab.webui.runner import run_backtest; run_backtest('best_trend')"
```

### CLI R&D loop (agent-friendly)
```bash
# JSON by default; use --format text for humans.

# Configs
# - canonical run:     quantlab/configs/trend_based/current.yaml
# - reference:         quantlab/configs/trend_based/reference/*.yaml

# Single scoring run
.venv/bin/python -m quantlab.rnd run --config quantlab/configs/trend_based/current.yaml

# (Optional) Sweeps
# Sweeps are driven by `quantlab.rnd sweep` and dotted-path keys (see `--help`).
# Decision bundles are written under `quantlab/reports/trend_based/decisions/`.
```

## Data Management

```bash
# Download specific period (1m candles -> 1m + 5m/15m OHLC, what the backtests read)
.venv/bin/python -m quantlab.data.download --symbols XAUUSD --start 2020-01-01 --end 2020-12-31

# Download recent data from ticks instead (1s + 5m/15m; 24 requests per day vs 2)
.venv/bin/python -m quantlab.data.download --source ticks --symbols XAUUSD XAGUSD EURUSD --days 7

# Validate integrity
.venv/bin/python -m quantlab.data.validate
```

**Data formats:**
| Folder | Columns |
|--------|---------|
| `dukascopy_1s` | ts, bid, ask, mid, spread |
| `dukascopy_1m` | ts, open, high, low, close, bid_close, ask_close, spread_close |
| `dukascopy_5m` | ts, bid, ask, mid, spread |
| `dukascopy_5m_ohlc` | ts, open, high, low, close |
| `dukascopy_15m` | ts, bid, ask, mid, spread |
| `dukascopy_15m_ohlc` | ts, open, high, low, close |

## Strategy Classes

- `TrendStrategyWithGates`: main strategy (MA crossover + configurable gate pipeline from YAML).
  Supports SMA or EMA via `ma_kind` param in config.
- Baselines: `BuyAndHoldStrategy`, `TrendStrategy` (simple MA crossover), `MeanReversionStrategy`.

## Composable Gates

`TrendStrategyWithGates` applies a gate pipeline on top of a base trend signal.

### Gate Reference

| Gate | Category | Purpose |
|------|----------|---------|
| `ema_sep` | Entry filter | Require EMA separation > ATR × k (avoid chop) |
| `nochop` | Entry filter | Require N closes above EMA in lookback (sustained trend) |
| `htf_confirm` | Entry filter | Require HTF SMA fast > slow |
| `corr` | Entry filter + sizing | Correlation stability filter + confirm sizing (XAG/EUR) |
| `time_filter` | Time filter | Block FOMC windows, force-flat specific months |
| `ema_strength_sizing` | Sizing | Size up when HTF EMA separation is strong |
| `seasonality_cap` | Sizing | Cap position size by calendar month |
| `churn` | Churn control | Entry debounce + re-entry cooldown |
| `shock_exit` | Exit | Force-flat on large adverse bar moves |
| `mid_loss_limiter` | Exit | Exit losers in toxic mid-duration band |
| `no_recovery_exit` | Exit | Exit if no new equity high within N bars |
| `profit_milestone` | Exit | Partial exit at profit milestones |
| `rolling_max_exit` | Exit | Exit below rolling max threshold |

### Registering new gates:
- Add a new gate class under `quantlab/strategies/gates/` and register it:

```python
from quantlab.strategies.gates.registry import register_gate

@register_gate("my_gate")
class MyGate:
    def __init__(self, **params): ...
    @property
    def name(self) -> str: ...
    def __call__(self, positions, prices, context=None): ...
```

- Then reference it from YAML:

```yaml
pipeline:
  - gate: my_gate
    params: {}
```


## Configuration

Canonical config (tracked):
- `quantlab/configs/trend_based/current.yaml`

Gate pipeline config:
- `current.yaml` uses `pipeline:`, a list of `{gate, params}` entries.
- Gate is ON if it appears in `pipeline:`.
- Gate order is the list order.

Indicator warm-up:
- `periods.warmup_days` loads that many extra days before each period, so indicators start warmed up. Trading and metrics still start at the period start (0 = cold start).
- If the data doesn't reach back that far, the run warns and uses the warm-up that exists.

### Config validation (Pydantic)
To avoid silent misconfig (typos / wrong param names), Quantlab validates the canonical config schema (including *fully typed* gate params).
The WebUI runner and `quantlab.rnd` CLI validate configs on load and will fail fast on unknown keys.

Validate a YAML file directly:
```bash
.venv/bin/python -m quantlab.configs.schema quantlab/configs/trend_based/current.yaml
```

## Reports

**Canonical performance snapshot:** `TRADING_INSIGHTS.md`
**Best strategy documentation:** `reports/trend_based/BEST_TREND_STRATEGY.md`

### HTML generation code
- `quantlab/reporting/generate_equity_report.py` (multi-period single-file equity/performance HTML)
- `quantlab/reporting/generate_robustness_report.py` (robustness surface; currently yearly breakdown table, will expand)
- `quantlab/reporting/generate_trades_report.py` (multi-period single-file trade breakdown HTML)

### Outputs
```
quantlab/reports/
├── baselines/           # Simple strategy reports (overwritten)
│   ├── buy_and_hold.html
│   ├── trend.html
│   └── mean_reversion.html
├── trend_based/         # Best trend variants (overwritten)
│   ├── best_trend.html
│   ├── BEST_TREND_STRATEGY.md
│   └── decisions/       # Decision bundles (artifacts we keep)
│       └── YYYY-MM-DD_<slug>/
│           ├── DECISION.md
│           ├── best.yaml
│           ├── results.csv
│           ├── top_k.csv        # sweep only
│           ├── notes.json
│           └── raw/
```

## Development

- **API Key:** `FRED_API_KEY` for economic calendar
- **Design:** Strategy classes with composable gates, single backtest engine

### Checks

```bash
./check.sh                    # unit tests + regression
./check.sh --unit-only        # fast unit tests only
./check.sh --regression-only  # golden series only
```

### Regression & parity (mandatory when refactoring execution semantics)

When touching anything that can alter signals/execution (gates, costs, time-filtering, return math, trade extraction), regress on **position + equity series**, not just headline metrics.

Golden series regression (best_trend 2024–2025):

```bash
# run regression
.venv/bin/python -m pytest -q quantlab/tests/regression/test_best_trend_2024_2025_series.py

# update golden (ONLY when the change is intentional)
PYTHONPATH=. .venv/bin/python quantlab/tests/regression/update_golden_best_trend_2024_2025.py
```

Notes:
- Golden artifacts live under `tests/regression/golden/`.
- The golden config includes the `corr` gate, so the regression needs XAGUSD and EURUSD data as well as XAUUSD.
- We bias toward CLI/WebUI parity; the webapp, `quantlab.rnd` and the regression test all run through `pipeline.py`.

## Requirements
- Runtime: `quantlab/requirements.txt`
- Dev/test: `quantlab/requirements-dev.txt`

## Repo history
- **February 2026:** GPT-5.2 + OpenClaw
- **November 2026:** Opus 5.5 + Claude Code
