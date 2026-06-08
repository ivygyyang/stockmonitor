# stockmonitor

A CLI tool that scans stocks for variance patterns that may precede breakouts, spikes, or pullbacks. Uses [yfinance](https://github.com/ranaroussi/yfinance) for market data, scores each ticker across multiple technical indicators, explains results in plain English, and learns from its own predictions over time.

## Features

- **Multi-indicator scoring** — RSI, MACD crossovers, Bollinger Bands, volume spikes, gap detection, ATR
- **Plain-English (ELI5) summaries** — every scan explains what the signals mean in simple terms
- **Watchlist management** — persistent per-user watchlist stored at `~/.stockmonitor/watchlist.json`
- **Daily prediction logging** — save each day's signals as a timestamped snapshot
- **Next-day grading** — compare yesterday's predictions against actual price moves
- **Self-tuning thresholds** — automatically adjusts RSI, volume, and other thresholds based on historical accuracy
- **Accuracy tracking** — see how well each signal type has performed over time
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
# Scan your saved watchlist
stockmonitor scan

# Scan specific tickers
stockmonitor scan AAPL TSLA NVDA MSFT

# Show only tickers with at least one active signal
stockmonitor scan --alerts-only

# Only show high-conviction setups (absolute score ≥ 3)
stockmonitor scan --min-score 3

# Export results to CSV
stockmonitor scan -o results.csv

# Table only — skip the plain-English explanation
stockmonitor scan --no-eli5

# Customize thresholds
stockmonitor scan --rsi-oversold 30 --rsi-overbought 70 --vol-spike 2.5
```

### Daily Prediction Log & Self-Improvement

```bash
# Save today's predictions to disk
stockmonitor log

# Next day: grade yesterday's predictions against actual price moves
stockmonitor grade

# See full historical accuracy + current auto-tuned thresholds
stockmonitor accuracy
```

**Recommended daily workflow:**
1. Run `stockmonitor log` each morning (or automate it)
2. Run `stockmonitor grade` the next morning — it scores each prediction and auto-adjusts thresholds
3. After ~2 weeks of data, run `stockmonitor accuracy` to see which signals are most reliable

### Watchlist

```bash
stockmonitor watch list
stockmonitor watch add GOOG
stockmonitor watch remove SPY
```

## Signals & Scoring

Each ticker is scored across the following indicators. Positive score = bullish bias, negative = bearish bias.

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

## Self-Tuning

After grading at least 10 predictions, the tool automatically adjusts its thresholds:

- If a signal's hit rate drops below **48%**, the threshold tightens (e.g. RSI oversold moves from 35 → 33)
- If a signal's hit rate exceeds **65%**, the threshold relaxes slightly to catch more setups
- Adjusted thresholds are saved to `~/.stockmonitor/config.json` and used on all future scans
- Run `stockmonitor accuracy` to see current thresholds and per-signal performance

## Data Storage

All data is stored locally in `~/.stockmonitor/`:

| Path | Contents |
|------|----------|
| `watchlist.json` | Your saved tickers |
| `config.json` | Auto-tuned threshold settings |
| `logs/YYYY-MM-DD.json` | Daily prediction snapshots |
| `grades/YYYY-MM-DD.json` | Graded accuracy results |

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

stockmonitor log [TICKERS]...

  --period TEXT         yfinance period string             [default: 6mo]

stockmonitor grade [PREDICT_DATE] [ACTUAL_DATE]

  --no-auto-tune        Skip threshold adjustment after grading

stockmonitor accuracy   Show historical hit rate and current thresholds
stockmonitor watch      add | remove | list
```

## Dependencies

| Package | Purpose |
|---------|---------|
| yfinance | Market data (OHLCV) |
| pandas | Data manipulation |
| numpy | Numerical computation |
| rich | Terminal table rendering |
| typer | CLI framework |
