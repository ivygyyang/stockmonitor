import json
from pathlib import Path

DEFAULT_PATH = Path.home() / ".stockmonitor" / "watchlist.json"
DEFAULT_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "SPY", "QQQ"]


def _load_raw(path: Path) -> list[str]:
    if path.exists():
        return json.loads(path.read_text())
    return list(DEFAULT_TICKERS)


def load(path: Path = DEFAULT_PATH) -> list[str]:
    return _load_raw(path)


def save(tickers: list[str], path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(t.upper() for t in tickers)), indent=2))


def add(ticker: str, path: Path = DEFAULT_PATH) -> list[str]:
    tickers = load(path)
    ticker = ticker.upper()
    if ticker not in tickers:
        tickers.append(ticker)
        save(tickers, path)
    return sorted(tickers)


def remove(ticker: str, path: Path = DEFAULT_PATH) -> list[str]:
    tickers = load(path)
    ticker = ticker.upper()
    tickers = [t for t in tickers if t != ticker]
    save(tickers, path)
    return sorted(tickers)
