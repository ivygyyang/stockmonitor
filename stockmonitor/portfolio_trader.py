"""
Multi-stock portfolio paper trader.

One shared cash pool (default $1,000) allocated across multiple tickers.
Each day after market close:
  1. Fetch latest prices for all tracked tickers
  2. Retrain each ticker's model on its rolling window
  3. Sell any held positions the model rates DOWN
  4. Buy the top-confidence UP signals with available cash (equal-weight)

State is stored at ~/.stockmonitor/portfolio.json
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .live_trader import (
    FEATURE_NAMES, AVAILABLE_MODELS,
    _build_features, _train_model, _predict,
)

warnings.filterwarnings("ignore", category=FutureWarning)

STATE_PATH = Path.home() / ".stockmonitor" / "portfolio.json"


@dataclass
class Position:
    ticker: str
    shares: float
    buy_price: float
    buy_date: str


@dataclass
class PortfolioTrade:
    date: str
    ticker: str
    action: str       # BUY | SELL
    price: float
    shares: float
    value: float
    proba_up: float
    pnl: float = 0.0  # realized P&L for SELL trades


@dataclass
class PortfolioState:
    tickers: list[str]
    starting_cash: float
    cash: float
    positions: dict[str, Position]   # ticker -> Position
    train_start: str
    started_on: str
    last_updated: str
    train_window: int
    n_estimators: int
    model_type: str
    confidence_threshold: float
    max_positions: int
    trades: list[PortfolioTrade] = field(default_factory=list)
    daily_values: list[tuple[str, float]] = field(default_factory=list)
    # buy-and-hold reference: equal-weight all tickers from day 1
    bh_shares: dict[str, float] = field(default_factory=dict)
    bh_start_prices: dict[str, float] = field(default_factory=dict)

    def total_value(self, prices: dict[str, float]) -> float:
        held = sum(pos.shares * prices.get(t, pos.buy_price)
                   for t, pos in self.positions.items())
        return self.cash + held

    def bh_value(self, prices: dict[str, float]) -> float:
        return sum(self.bh_shares.get(t, 0) * prices.get(t, p)
                   for t, p in self.bh_start_prices.items())

    def save(self) -> Path:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        STATE_PATH.write_text(json.dumps(data, indent=2))
        return STATE_PATH

    @classmethod
    def load(cls) -> "PortfolioState":
        if not STATE_PATH.exists():
            raise FileNotFoundError(
                "No portfolio session found. "
                "Run `stockmonitor portfolio-start` first."
            )
        data = json.loads(STATE_PATH.read_text())
        data["positions"] = {
            t: Position(**p) for t, p in data.get("positions", {}).items()
        }
        data["trades"] = [PortfolioTrade(**tr) for tr in data.get("trades", [])]
        return cls(**data)

    @classmethod
    def exists(cls) -> bool:
        return STATE_PATH.exists()


# ── Public API ────────────────────────────────────────────────────────────────

def _fetch_price(ticker: str) -> tuple[Optional[float], Optional[pd.DataFrame]]:
    """Download recent history for a ticker. Returns (latest_close, full_raw_df)."""
    try:
        raw = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if raw.empty:
            return None, None
        return float(raw["Close"].iloc[-1]), raw
    except Exception:
        return None, None


def _signal(raw: pd.DataFrame, train_window: int, n_estimators: int,
            model_type: str) -> tuple[int, float]:
    """Return (prediction 0/1, proba_up) for the latest row."""
    try:
        feat = _build_features(raw)
        if len(feat) < train_window:
            return 1, 0.5
        clf, scaler = _train_model(feat, train_window, n_estimators, model_type)
        if clf is None:
            return 1, 0.5
        pred, proba = _predict(clf, scaler, feat[FEATURE_NAMES].values[-1])
        return pred, proba
    except Exception:
        return 1, 0.5


def start(
    tickers: list[str],
    starting_cash: float = 1000.0,
    train_start: str = "2000-01-01",
    train_window: int = 252,
    n_estimators: int = 100,
    model_type: str = "rf",
    confidence_threshold: float = 0.55,
    max_positions: int = 3,
) -> PortfolioState:
    """
    Initialise a portfolio session. Downloads history for all tickers,
    records buy-and-hold baselines, makes the first set of predictions.
    Does NOT make any trades on day 0 — first trades happen on live-update.
    """
    today = str(date.today())
    cash_per_bh = starting_cash / len(tickers)

    bh_shares: dict[str, float] = {}
    bh_prices: dict[str, float] = {}

    console_lines: list[str] = []
    for ticker in tickers:
        price, _ = _fetch_price(ticker)
        if price and price > 0:
            bh_shares[ticker] = cash_per_bh / price
            bh_prices[ticker] = price
            console_lines.append(f"    {ticker}: ${price:,.2f}")
        else:
            console_lines.append(f"    {ticker}: skipped (no data)")

    state = PortfolioState(
        tickers=tickers,
        starting_cash=starting_cash,
        cash=starting_cash,
        positions={},
        train_start=train_start,
        started_on=today,
        last_updated=today,
        train_window=train_window,
        n_estimators=n_estimators,
        model_type=model_type,
        confidence_threshold=confidence_threshold,
        max_positions=max_positions,
        bh_shares=bh_shares,
        bh_start_prices=bh_prices,
    )
    state.daily_values.append((today, starting_cash))
    state.save()
    return state, console_lines


def update() -> tuple[PortfolioState, list[str]]:
    """
    Fetch latest prices, sell losers, buy top-confidence winners.
    Returns (state, log_lines).
    """
    state = PortfolioState.load()
    log: list[str] = []

    # ── Fetch latest prices and signals for all tickers ───────────────────────
    prices: dict[str, float] = {}
    signals: dict[str, tuple[int, float]] = {}   # ticker -> (pred, proba_up)

    for ticker in state.tickers:
        price, raw = _fetch_price(ticker)
        if price is None:
            log.append(f"  [yellow]{ticker}: skipped (no data)[/yellow]")
            continue
        prices[ticker] = price
        pred, proba = _signal(raw, state.train_window, state.n_estimators,
                              state.model_type)
        signals[ticker] = (pred, proba)

    if not prices:
        return state, ["  [red]Could not fetch any prices. Check internet.[/red]"]

    latest_date = str(date.today())
    if latest_date == state.last_updated:
        return state, ["  Already up to date for today."]

    # ── Sell positions where model predicts DOWN ──────────────────────────────
    to_sell = [
        t for t, pos in state.positions.items()
        if t in signals and signals[t][0] == 0
    ]
    for ticker in to_sell:
        pos = state.positions[ticker]
        price = prices[ticker]
        proceeds = pos.shares * price
        pnl = proceeds - (pos.shares * pos.buy_price)
        state.cash += proceeds
        state.trades.append(PortfolioTrade(
            date=latest_date, ticker=ticker, action="SELL",
            price=price, shares=pos.shares, value=proceeds,
            proba_up=signals[ticker][1], pnl=pnl,
        ))
        del state.positions[ticker]
        direction = "profit" if pnl >= 0 else "loss"
        log.append(f"  SELL {ticker} @ ${price:,.2f}  ({direction}: ${pnl:+,.2f})")

    # ── Buy top-confidence UP signals with available cash ─────────────────────
    slots_free = state.max_positions - len(state.positions)
    if slots_free > 0 and state.cash > 1.0:
        candidates = [
            (t, proba)
            for t, (pred, proba) in signals.items()
            if pred == 1
            and proba >= state.confidence_threshold
            and t not in state.positions
            and t in prices
        ]
        # Sort by confidence descending, take top N
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:slots_free]

        if candidates:
            cash_each = state.cash / len(candidates)
            for ticker, proba in candidates:
                price = prices[ticker]
                shares = cash_each / price
                state.cash -= cash_each
                state.positions[ticker] = Position(
                    ticker=ticker, shares=shares,
                    buy_price=price, buy_date=latest_date,
                )
                state.trades.append(PortfolioTrade(
                    date=latest_date, ticker=ticker, action="BUY",
                    price=price, shares=shares, value=cash_each,
                    proba_up=proba,
                ))
                log.append(f"  BUY  {ticker} @ ${price:,.2f}  "
                           f"({shares:.4f} shares, confidence {proba:.1%})")

    if not to_sell and not [t for t in log if "BUY" in t]:
        held = ", ".join(state.positions.keys()) if state.positions else "none"
        log.append(f"  HOLD — no trades today  (positions: {held})")

    total = state.total_value(prices)
    state.daily_values.append((latest_date, total))
    state.last_updated = latest_date
    state.save()
    return state, log


def get_status() -> tuple[PortfolioState, dict[str, float]]:
    """Load state and fetch current prices for all held + tracked tickers."""
    state = PortfolioState.load()
    prices: dict[str, float] = {}
    all_tickers = list(set(state.tickers) | set(state.positions.keys()))
    for ticker in all_tickers:
        try:
            raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if not raw.empty:
                prices[ticker] = float(raw["Close"].iloc[-1])
        except Exception:
            pass
    return state, prices
