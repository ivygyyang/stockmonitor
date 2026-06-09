# stockmonitor

A CLI tool that scans stocks for variance patterns that may precede breakouts, spikes, or pullbacks. Uses [yfinance](https://github.com/ranaroussi/yfinance) for market data, scores each ticker across multiple technical indicators, explains results in plain English, **learns from its own predictions over time**, and includes a walk-forward ML backtesting engine, live paper-trading simulator, and a multi-stock virtual portfolio powered by machine learning.

## Authors

- **Richie Friedland** ([RichieJr1111](https://github.com/RichieJr1111)) — ML engine, walk-forward backtesting, live paper trading, portfolio simulator, observation space design, model architecture (Random Forest & Gradient Boosting), feature engineering
- **Robert Lewis** — architecture, CLI framework, scheduling, setup
- **Ivy Lewis** — indicator scoring, self-improvement loop, ELI5 summaries, watchlist

## Features

- **Multi-indicator scoring** — RSI, MACD crossovers, Bollinger Bands, volume spikes, gap detection, ATR
- **Plain-English (ELI5) summaries** — every scan explains what the signals mean in simple terms
- **Watchlist management** — persistent per-user watchlist stored at `~/.stockmonitor/watchlist.json`
- **Daily prediction logging** — save each day's signals as a timestamped snapshot
- **Next-day grading** — compare yesterday's predictions against actual price moves
- **Self-improvement journal** — every grade writes structured improvement directives to a running journal
- **Auto-tuning thresholds** — RSI, volume, and other thresholds adjust automatically based on hit rate
- **ML walk-forward backtesting** — train a Random Forest on historical data, test it day-by-day with no look-ahead bias
- **Live paper trading** — virtual $1,000 portfolio that trades in real time using daily ML predictions
- **Scheduled automation** — one command sets up Windows Task Scheduler to run everything daily
- **Rich output** — color-coded terminal table with bullish (green) / bearish (red) signal bias
- **CSV export** — save scan results for further analysis

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/ivygyyang/stockmonitor.git
cd stockmonitor
pip install -e .
pip install scikit-learn   # required for ML commands
```

## Usage

### Scan

```bash
stockmonitor scan                          # scan watchlist with ELI5 explanation
stockmonitor scan AAPL TSLA NVDA           # scan specific tickers
stockmonitor scan --alerts-only            # only show tickers with active signals
stockmonitor scan --min-score 3            # high-conviction setups only
stockmonitor scan --no-eli5               # table only
stockmonitor scan -o results.csv           # export to CSV
```

### Daily Self-Improvement Loop

```bash
# 1. Save today's predictions
stockmonitor log

# 2. Next day: grade predictions + auto-update journal + auto-tune thresholds
stockmonitor grade

# 3. Read accumulated improvement notes
stockmonitor journal

# 4. See overall accuracy stats
stockmonitor accuracy

# 5. Automate everything (runs once -- sets up Windows Task Scheduler)
stockmonitor schedule
```

After running `schedule`, the tool will:
- **6:30 AM daily** — save predictions automatically
- **9:00 AM daily** — grade them, update the journal, and tune thresholds

### Watchlist

```bash
stockmonitor watch list
stockmonitor watch add GOOG
stockmonitor watch remove SPY
```

### ML Backtesting

Walk-forward backtest: the model trains on a rolling window of past data and predicts each day's direction sequentially. No look-ahead bias — it only ever sees data that existed before the day it predicts.

```bash
# Backtest AAPL using all data from 2000 to today
stockmonitor ml-backtest AAPL --start 2000-01-01

# Multiple tickers, specific date range
stockmonitor ml-backtest AAPL MSFT SPY --start 2000-01-01 --end 2023-01-01

# Use maximum available history
stockmonitor ml-backtest SPY QQQ --period max

# Tune the model
stockmonitor ml-backtest AAPL --start 2000-01-01 --train-window 504 --n-estimators 200
```

### Live Paper Trading

Train on historical data, then simulate real trading from today forward with a virtual portfolio. State is saved between sessions so you can check in anytime.

```bash
# Start a new session (trains on 2000-today, starts with $1,000)
stockmonitor live-start AAPL --train-start 2000-01-01 --cash 1000

# Run daily after market close to advance the simulation
stockmonitor live-update              # updates all active sessions
stockmonitor live-update AAPL        # or a specific one

# Check your dashboard anytime
stockmonitor live-check              # all sessions
stockmonitor live-check AAPL        # just one

# Start multiple tickers
stockmonitor live-start MSFT --train-start 2000-01-01 --cash 1000
stockmonitor live-start NVDA --train-start 2000-01-01 --cash 1000
```

The dashboard shows: current portfolio value, P&L vs buy-and-hold, recent trade log, portfolio chart, and tomorrow's predicted signal.

### Paper-Trade Simulation (historical)

Simulate trading over a past period to evaluate strategy performance:

```bash
# Train on 2000-2023, trade 2023-today with $1,000
stockmonitor paper-trade AAPL --train-start 2000-01-01 --train-end 2023-01-01

# Test a specific historical window (dot-com bubble)
stockmonitor paper-trade MSFT --train-start 1995-01-01 --train-end 2000-01-01 --trade-end 2003-01-01

# Higher confidence threshold = fewer but higher-quality trades
stockmonitor paper-trade AAPL --train-start 2000-01-01 --train-end 2023-01-01 --confidence 0.70
```

## The Self-Improvement Loop

Every time `stockmonitor grade` runs, the tool:

1. Fetches actual next-day prices for all predicted tickers
2. Scores each prediction correct/incorrect
3. Runs the **Advisor** — a full analysis across all graded history:
   - Per-signal accuracy (is RSI oversold reliable? is BB squeeze predictive?)
   - Per-ticker accuracy (which tickers is the model good/bad at?)
   - Signal combination accuracy (does RSI + volume spike together outperform each alone?)
   - Trend analysis (is accuracy improving or degrading recently?)
4. Writes two types of directives into the grade file and journal:
   - **AUTO** — applied immediately (threshold tightening/relaxing)
   - **MANUAL** — insights requiring a developer to update the scoring logic
5. Appends a full entry to `~/.stockmonitor/journal.md`

Run `stockmonitor journal` to read the accumulated notes. The journal grows smarter over time.

## ML Observation Space

The Random Forest model receives 13 features for each trading day:

| Feature | Description |
|---------|-------------|
| `rsi` | 14-day Relative Strength Index (momentum, 0–100) |
| `macd_line` | MACD line — 12/26 EMA difference (trend direction) |
| `macd_hist` | MACD histogram — line minus signal (momentum shift) |
| `bb_pct_b` | Bollinger %B — where price sits within the bands (0–1) |
| `bb_width` | Bollinger Band width (volatility expansion/compression) |
| `atr_pct` | ATR as % of price (daily volatility magnitude) |
| `vol_ratio` | Volume vs 20-day average (unusual activity) |
| `gap_pct` | Overnight gap % — open vs prior close |
| `ret_1d` | 1-day return % |
| `ret_3d` | 3-day return % |
| `ret_5d` | 5-day return % |
| `close_vs_sma20` | % above/below 20-day simple moving average |
| `close_vs_sma50` | % above/below 50-day simple moving average |

**Label:** `1` if next-day close > today's close, `0` otherwise (binary classification).

The model uses a Random Forest classifier (default: 100 trees, max depth 6, min 10 samples per leaf) retrained daily on a rolling window of the most recent N trading days.

## Signals & Scoring

| Signal | Condition | Score |
|--------|-----------|-------|
| RSI | ≤ 35 (oversold) | +2 |
| RSI | ≥ 65 (overbought) | −2 |
| MACD | Histogram crosses above zero | +2 |
| MACD | Histogram crosses below zero | −2 |
| Bollinger Band | Price breaks above upper band | +2 |
| Bollinger Band | Price breaks below lower band | −2 |
| Bollinger Band | Squeeze (width < 75% of 50-day avg) | +1 |
| Volume | Spike ≥ 2× 20-day average | ±2 (amplifies existing bias) |
| Gap | Gap up ≥ 1.5% | +1 |
| Gap | Gap down ≤ −1.5% | −1 |
| ATR | ATR% ≥ 3% | flagged (elevated volatility) |

## Data Storage

All data is stored locally in `~/.stockmonitor/`:

| Path | Contents |
|------|----------|
| `watchlist.json` | Saved tickers |
| `config.json` | Auto-tuned threshold settings |
| `logs/YYYY-MM-DD.json` | Daily prediction snapshots |
| `grades/YYYY-MM-DD.json` | Graded results + self-improvement directives |
| `journal.md` | Cumulative improvement journal |
| `live/<TICKER>.json` | Live paper-trade session state |

## Options Reference

```
stockmonitor scan [OPTIONS] [TICKERS]...
  --period TEXT         yfinance period (1mo, 3mo, 6mo, 1y) [default: 6mo]
  --rsi-oversold INT    RSI oversold threshold              [default: 35]
  --rsi-overbought INT  RSI overbought threshold            [default: 65]
  --vol-spike FLOAT     Volume spike ratio vs 20-day avg   [default: 2.0]
  --gap-pct FLOAT       Minimum gap % to flag              [default: 1.5]
  --atr-pct FLOAT       ATR% threshold for elevated vol    [default: 3.0]
  --min-score INT       Minimum absolute score to display  [default: 0]
  --alerts-only, -a     Only show tickers with alerts
  --no-eli5             Skip plain-English explanation
  --output, -o PATH     Export results to CSV

stockmonitor ml-backtest [OPTIONS] [TICKERS]...
  --period TEXT         History to download (2y, 5y, 10y, max) [default: 5y]
  --start TEXT          Start date as YYYY-MM-DD
  --end TEXT            End date as YYYY-MM-DD
  --train-window INT    Days of history per model iteration  [default: 252]
  --n-estimators INT    Random Forest trees                  [default: 100]
  --top-features INT    Features to display                  [default: 5]
  --no-show-curve       Hide rolling accuracy chart

stockmonitor live-start TICKER [OPTIONS]
  --train-start TEXT    Start of training history            [default: 2000-01-01]
  --train-end TEXT      End of training / go-live date       [default: today]
  --cash FLOAT          Starting virtual cash                [default: 1000.0]
  --confidence FLOAT    Min model confidence to trade        [default: 0.55]
  --train-window INT    Rolling days for daily retraining    [default: 252]

stockmonitor live-update [TICKERS]...   Advance all (or named) sessions
stockmonitor live-check  [TICKERS]...   Show dashboard for all (or named) sessions
  --last-trades INT     Recent trades to show               [default: 10]

stockmonitor paper-trade [OPTIONS] [TICKERS]...
  --train-start TEXT    Start of training data              [default: 2000-01-01]
  --train-end TEXT      End of training / start of trading  [default: 2023-01-01]
  --trade-end TEXT      End of trading period               [default: today]
  --cash FLOAT          Starting virtual cash               [default: 1000.0]
  --confidence FLOAT    Min model confidence to buy         [default: 0.55]

stockmonitor log [TICKERS]...     Save today's predictions
stockmonitor grade [PREDICT_DATE] [ACTUAL_DATE]
  --no-auto-tune        Skip threshold adjustment after grading
stockmonitor journal [-n N]       View last N journal entries (default: 5)
  --path                Print journal file path
stockmonitor accuracy             Show historical hit rate + current thresholds
stockmonitor schedule             Set up Windows Task Scheduler for daily automation
stockmonitor watch add|remove|list
```

## Dependencies

| Package | Purpose |
|---------|---------|
| yfinance | Market data (OHLCV) — free, no API key required |
| pandas | Data manipulation |
| numpy | Numerical computation |
| rich | Terminal table and dashboard rendering |
| typer | CLI framework |
| scikit-learn | Random Forest classifier for ML commands |
