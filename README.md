# stockmonitor

A CLI tool that scans stocks for variance patterns that may precede breakouts, spikes, or pullbacks. Uses [yfinance](https://github.com/ranaroussi/yfinance) for market data and scores each ticker across multiple technical indicators.

## Features

- **Multi-indicator scoring** — RSI, MACD crossovers, Bollinger Bands, volume spikes, gap detection, ATR
- **Watchlist management** — persistent per-user watchlist stored at `~/.stockmonitor/watchlist.json`
- **Flexible scanning** — scan your watchlist or pass tickers directly on the command line
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

# Customize thresholds
stockmonitor scan --rsi-oversold 30 --rsi-overbought 70 --vol-spike 2.5
```

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

## Options Reference

```
stockmonitor scan [OPTIONS] [TICKERS]...

Options:
  --period TEXT         yfinance period (1mo, 3mo, 6mo, 1y) [default: 6mo]
  --rsi-oversold INT    RSI oversold threshold              [default: 35]
  --rsi-overbought INT  RSI overbought threshold            [default: 65]
  --vol-spike FLOAT     Volume spike ratio vs 20-day avg   [default: 2.0]
  --gap-pct FLOAT       Minimum gap % to flag              [default: 1.5]
  --atr-pct FLOAT       ATR% threshold for elevated vol    [default: 3.0]
  --min-score INT       Minimum absolute score to display  [default: 0]
  --alerts-only, -a     Only show tickers with alerts
  --output, -o PATH     Export results to CSV
```

## Dependencies

| Package | Purpose |
|---------|---------|
| yfinance | Market data (OHLCV) |
| pandas | Data manipulation |
| numpy | Numerical computation |
| rich | Terminal table rendering |
| typer | CLI framework |
