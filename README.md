# stockmonitor

A CLI tool that scans stocks for variance patterns that may precede breakouts, spikes, or pullbacks. Uses [yfinance](https://github.com/ranaroussi/yfinance) for market data, scores each ticker across multiple technical indicators, explains results in plain English, and **learns from its own predictions over time**.

## Features

- **Multi-indicator scoring** — RSI, MACD crossovers, Bollinger Bands, volume spikes, gap detection, ATR
- **Plain-English (ELI5) summaries** — every scan explains what the signals mean in simple terms
- **Watchlist management** — persistent per-user watchlist stored at `~/.stockmonitor/watchlist.json`
- **Daily prediction logging** — save each day's signals as a timestamped snapshot
- **Next-day grading** — compare yesterday's predictions against actual price moves
- **Self-improvement journal** — every grade writes structured improvement directives to a running journal
- **Auto-tuning thresholds** — RSI, volume, and other thresholds adjust automatically based on hit rate
- **Scheduled automation** — one command sets up Windows Task Scheduler to run everything daily
- **Rich output** — color-coded terminal table with bullish (green) / bearish (red) signal bias
- **CSV export** — save scan results for further analysis

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/ivygyyang/stockmonitor.git
cd stockmonitor
pip install -e .
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

# 5. Automate everything (runs once — sets up Windows Task Scheduler)
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
| yfinance | Market data (OHLCV) |
| pandas | Data manipulation |
| numpy | Numerical computation |
| rich | Terminal table rendering |
| typer | CLI framework |
