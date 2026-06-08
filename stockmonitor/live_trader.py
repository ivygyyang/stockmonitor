"""
Persistent live paper trader.

State is stored in ~/.stockmonitor/live/<TICKER>.json so it survives between
sessions. Each call to `update()` fetches the latest price, executes the
pending trade decision, then generates a fresh prediction for tomorrow.

Workflow:
    live-start AAPL          # train on history, save state, first prediction
    live-update AAPL         # run daily (or whenever) to advance the simulation
    live-check AAPL          # read-only dashboard — never modifies state
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .indicators import atr, bollinger_bands, macd, rsi, volume_ratio, gap_pct

warnings.filterwarnings("ignore", category=FutureWarning)

FEATURE_NAMES = [
    "rsi", "macd_line", "macd_hist", "bb_pct_b", "bb_width",
    "atr_pct", "vol_ratio", "gap_pct",
    "ret_1d", "ret_3d", "ret_5d",
    "close_vs_sma20", "close_vs_sma50",
]

STATE_DIR = Path.home() / ".stockmonitor" / "live"


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["rsi"] = rsi(df)
    ml_line, _, hist = macd(df)
    feat["macd_line"] = ml_line
    feat["macd_hist"] = hist
    _, _, _, width, pct_b = bollinger_bands(df)
    feat["bb_pct_b"] = pct_b
    feat["bb_width"] = width
    atr_s = atr(df)
    feat["atr_pct"] = atr_s / df["Close"] * 100
    feat["vol_ratio"] = volume_ratio(df)
    feat["gap_pct"] = gap_pct(df)
    close = df["Close"]
    feat["ret_1d"] = close.pct_change(1) * 100
    feat["ret_3d"] = close.pct_change(3) * 100
    feat["ret_5d"] = close.pct_change(5) * 100
    feat["close_vs_sma20"] = (close / close.rolling(20).mean() - 1) * 100
    feat["close_vs_sma50"] = (close / close.rolling(50).mean() - 1) * 100
    feat["label"] = (close.shift(-1) > close).astype(int)
    return feat.dropna()


def _train_model(feat_df: pd.DataFrame, train_window: int, n_estimators: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    X = feat_df[FEATURE_NAMES].values[-train_window:]
    y = feat_df["label"].values[-train_window:]

    if len(set(y)) < 2:
        return None, None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_s, y)
    return clf, scaler


def _predict(clf, scaler, feat_row: np.ndarray):
    x_s = scaler.transform(feat_row.reshape(1, -1))
    pred = int(clf.predict(x_s)[0])
    proba_up = float(clf.predict_proba(x_s)[0][1])
    return pred, proba_up


# ── Persistent state ──────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    date: str
    action: str          # BUY | SELL | HOLD
    price: float
    shares: float
    cash: float
    portfolio_value: float
    proba_up: float


@dataclass
class LiveState:
    ticker: str
    starting_cash: float
    cash: float
    shares: float
    train_start: str
    train_end: str
    started_on: str
    last_updated: str
    pending_action: str   # what the model said to do NEXT market day
    pending_proba: float
    confidence_threshold: float
    train_window: int
    n_estimators: int
    trades: list[TradeRecord] = field(default_factory=list)
    daily_values: list[tuple[str, float]] = field(default_factory=list)
    # buy-and-hold reference
    bh_shares: float = 0.0
    bh_start_price: float = 0.0

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> Path:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path = STATE_DIR / f"{self.ticker}.json"
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, ticker: str) -> "LiveState":
        path = STATE_DIR / f"{ticker.upper()}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No live session found for {ticker.upper()}. "
                f"Run `stockmonitor live-start {ticker.upper()}` first."
            )
        data = json.loads(path.read_text())
        data["trades"] = [TradeRecord(**t) for t in data.get("trades", [])]
        return cls(**data)

    @classmethod
    def exists(cls, ticker: str) -> bool:
        return (STATE_DIR / f"{ticker.upper()}.json").exists()

    def current_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def bh_value(self, price: float) -> float:
        return self.bh_shares * price


# ── Public API ────────────────────────────────────────────────────────────────

def start(
    ticker: str,
    train_start: str,
    train_end: Optional[str],
    starting_cash: float = 1000.0,
    train_window: int = 252,
    n_estimators: int = 100,
    confidence_threshold: float = 0.55,
) -> LiveState:
    """
    Download history, train model, record initial state, make first prediction.
    train_end=None means "up to today" (model trains on all available data).
    """
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    t_end = train_end or str(date.today())

    raw = yf.download(ticker, start=train_start, end=t_end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    if raw.empty or len(raw) < train_window + 60:
        raise ValueError(f"Not enough data for {ticker} ({len(raw)} rows, need {train_window + 60})")

    feat_df = _build_features(raw)
    clf, scaler = _train_model(feat_df, train_window, n_estimators)
    if clf is None:
        raise ValueError("Training failed — only one class in training window.")

    # Predict on the very last row of training data (what to do tomorrow)
    last_row = feat_df[FEATURE_NAMES].values[-1]
    pred, proba_up = _predict(clf, scaler, last_row)

    action = "HOLD"
    if pred == 1 and proba_up >= confidence_threshold:
        action = "BUY"
    elif pred == 0:
        action = "SELL"   # means: if holding, sell tomorrow

    last_price = float(raw["Close"].iloc[-1])
    today_str = str(date.today())

    state = LiveState(
        ticker=ticker.upper(),
        starting_cash=starting_cash,
        cash=starting_cash,
        shares=0.0,
        train_start=train_start,
        train_end=t_end,
        started_on=today_str,
        last_updated=today_str,
        pending_action=action,
        pending_proba=proba_up,
        confidence_threshold=confidence_threshold,
        train_window=train_window,
        n_estimators=n_estimators,
        bh_shares=starting_cash / last_price,
        bh_start_price=last_price,
    )
    state.daily_values.append((today_str, starting_cash))
    state.save()
    return state


def update(ticker: str) -> tuple[LiveState, str, float]:
    """
    Fetch latest price, execute yesterday's pending action, retrain, new prediction.
    Returns (updated_state, executed_action, price_used).
    """
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    state = LiveState.load(ticker)

    # Download enough history for retraining + latest price
    raw = yf.download(
        state.ticker,
        start=state.train_start,
        auto_adjust=True, progress=False
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    latest_price = float(raw["Close"].iloc[-1])
    latest_date = str(raw.index[-1].date())

    if latest_date == state.last_updated:
        return state, "ALREADY_CURRENT", latest_price

    # ── Execute yesterday's pending action ────────────────────────────────────
    executed = "HOLD"
    if state.pending_action == "BUY" and state.cash > 0:
        state.shares = state.cash / latest_price
        state.cash = 0.0
        executed = "BUY"
    elif state.pending_action == "SELL" and state.shares > 0:
        state.cash = state.shares * latest_price
        state.shares = 0.0
        executed = "SELL"

    portfolio_val = state.current_value(latest_price)
    state.daily_values.append((latest_date, portfolio_val))

    if executed != "HOLD":
        state.trades.append(TradeRecord(
            date=latest_date,
            action=executed,
            price=latest_price,
            shares=state.shares if executed == "BUY" else 0.0,
            cash=state.cash,
            portfolio_value=portfolio_val,
            proba_up=state.pending_proba,
        ))

    # ── Retrain and predict next action ──────────────────────────────────────
    feat_df = _build_features(raw)
    clf, scaler = _train_model(feat_df, state.train_window, state.n_estimators)
    if clf is not None:
        last_row = feat_df[FEATURE_NAMES].values[-1]
        pred, proba_up = _predict(clf, scaler, last_row)
        if pred == 1 and proba_up >= state.confidence_threshold:
            state.pending_action = "BUY"
        elif pred == 0:
            state.pending_action = "SELL"
        else:
            state.pending_action = "HOLD"
        state.pending_proba = proba_up
    else:
        state.pending_action = "HOLD"
        state.pending_proba = 0.5

    state.last_updated = latest_date
    state.save()
    return state, executed, latest_price


def get_status(ticker: str) -> tuple[LiveState, float]:
    """Load state and fetch the latest price for display (read-only)."""
    state = LiveState.load(ticker)
    raw = yf.download(state.ticker, period="5d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    latest_price = float(raw["Close"].iloc[-1]) if not raw.empty else 0.0
    return state, latest_price


def list_sessions() -> list[str]:
    if not STATE_DIR.exists():
        return []
    return [p.stem for p in sorted(STATE_DIR.glob("*.json"))]
