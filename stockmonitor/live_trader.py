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
    # Momentum
    "rsi", "rsi_slope",
    "macd_line", "macd_hist", "macd_slope",
    # Volatility / bands
    "bb_pct_b", "bb_width", "atr_pct", "atr_slope",
    # Volume
    "vol_ratio", "vol_ratio_5d",
    # Gap
    "gap_pct",
    # Short-term returns
    "ret_1d", "ret_3d", "ret_5d",
    # Medium-term returns
    "ret_10d", "ret_20d", "ret_60d",
    # Trend
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma200",
    "sma20_vs_sma50",
    # Volatility regime
    "vol_regime",      # rolling 10d std / rolling 60d std  (>1 = expanding vol)
    "up_days_10",      # fraction of up days in last 10
]

AVAILABLE_MODELS = ("rf", "gbm")   # random forest | gradient boosting

STATE_DIR = Path.home() / ".stockmonitor" / "live"


# ── Feature builder ───────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Momentum
    rsi_s = rsi(df)
    feat["rsi"] = rsi_s
    feat["rsi_slope"] = rsi_s.diff(3)           # RSI trending up or down

    ml_line, _, hist = macd(df)
    feat["macd_line"] = ml_line
    feat["macd_hist"] = hist
    feat["macd_slope"] = ml_line.diff(3)

    # Volatility / bands
    _, _, _, width, pct_b = bollinger_bands(df)
    feat["bb_pct_b"] = pct_b
    feat["bb_width"] = width
    atr_s = atr(df)
    atr_pct = atr_s / close * 100
    feat["atr_pct"] = atr_pct
    feat["atr_slope"] = atr_pct.diff(5)         # volatility expanding or contracting

    # Volume
    vr = volume_ratio(df)
    feat["vol_ratio"] = vr
    feat["vol_ratio_5d"] = df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean()

    # Gap
    feat["gap_pct"] = gap_pct(df)

    # Short-term returns
    feat["ret_1d"] = close.pct_change(1) * 100
    feat["ret_3d"] = close.pct_change(3) * 100
    feat["ret_5d"] = close.pct_change(5) * 100

    # Medium-term returns (new)
    feat["ret_10d"] = close.pct_change(10) * 100
    feat["ret_20d"] = close.pct_change(20) * 100
    feat["ret_60d"] = close.pct_change(60) * 100

    # Trend
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    feat["close_vs_sma20"]  = (close / sma20  - 1) * 100
    feat["close_vs_sma50"]  = (close / sma50  - 1) * 100
    feat["close_vs_sma200"] = (close / sma200 - 1) * 100
    feat["sma20_vs_sma50"]  = (sma20 / sma50  - 1) * 100  # golden/death cross proximity

    # Volatility regime: short-term vol vs long-term vol
    std10 = close.pct_change().rolling(10).std()
    std60 = close.pct_change().rolling(60).std()
    feat["vol_regime"] = std10 / std60.replace(0, np.nan)

    # Fraction of up-days in last 10 sessions
    feat["up_days_10"] = (close.diff() > 0).rolling(10).mean()

    feat["label"] = (close.shift(-1) > close).astype(int)
    return feat.dropna()


def _train_model(feat_df: pd.DataFrame, train_window: int, n_estimators: int,
                 model_type: str = "rf"):
    from sklearn.preprocessing import StandardScaler

    X = feat_df[FEATURE_NAMES].values[-train_window:]
    y = feat_df["label"].values[-train_window:]

    if len(set(y)) < 2:
        return None, None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    if model_type == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
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
    model_type: str = "rf"   # "rf" or "gbm"
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
    model_type: str = "rf",
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
    clf, scaler = _train_model(feat_df, train_window, n_estimators, model_type)
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
        model_type=model_type,
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
    clf, scaler = _train_model(feat_df, state.train_window, state.n_estimators,
                               getattr(state, "model_type", "rf"))
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


def retrain(
    ticker: str,
    train_start: Optional[str] = None,
    train_window: Optional[int] = None,
    n_estimators: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    model_type: Optional[str] = None,
) -> tuple[LiveState, dict]:
    """
    Retrain an existing session's model with new settings without resetting
    the portfolio. Portfolio history, trades, cash, and shares are preserved.
    Returns (updated_state, changes_dict) where changes_dict describes what changed.
    """
    state = LiveState.load(ticker)
    changes = {}

    if train_start and train_start != state.train_start:
        changes["train_start"] = (state.train_start, train_start)
        state.train_start = train_start
    if train_window and train_window != state.train_window:
        changes["train_window"] = (state.train_window, train_window)
        state.train_window = train_window
    if n_estimators and n_estimators != state.n_estimators:
        changes["n_estimators"] = (state.n_estimators, n_estimators)
        state.n_estimators = n_estimators
    if confidence_threshold and confidence_threshold != state.confidence_threshold:
        changes["confidence"] = (state.confidence_threshold, confidence_threshold)
        state.confidence_threshold = confidence_threshold
    if model_type and model_type != getattr(state, "model_type", "rf"):
        changes["model_type"] = (getattr(state, "model_type", "rf"), model_type)
        state.model_type = model_type

    # Re-download full history with potentially new train_start
    raw = yf.download(state.ticker, start=state.train_start,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    if raw.empty or len(raw) < state.train_window + 60:
        raise ValueError(f"Not enough data for new settings ({len(raw)} rows)")

    feat_df = _build_features(raw)
    clf, scaler = _train_model(feat_df, state.train_window, state.n_estimators,
                               getattr(state, "model_type", "rf"))
    if clf is None:
        raise ValueError("Retraining failed — only one class in window.")

    last_row = feat_df[FEATURE_NAMES].values[-1]
    pred, proba_up = _predict(clf, scaler, last_row)

    if pred == 1 and proba_up >= state.confidence_threshold:
        state.pending_action = "BUY"
    elif pred == 0:
        state.pending_action = "SELL"
    else:
        state.pending_action = "HOLD"
    state.pending_proba = proba_up
    state.train_end = str(raw.index[-1].date())

    state.save()
    return state, changes


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
