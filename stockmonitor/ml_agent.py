"""
Walk-forward ML backtesting agent.

For each ticker it:
  1. Downloads historical OHLCV data via yfinance
  2. Engineers features from existing indicators (RSI, MACD, BB, ATR, volume)
  3. Labels each day: 1 if next-day close > today's close, 0 otherwise
  4. Runs a walk-forward simulation: train on the past N days, predict the next
     day, slide the window forward one day, repeat
  5. Logs per-day predictions and accuracy as the window slides, so you can see
     how the model "learns" over time

No look-ahead bias: the model only ever sees data that would have been
available before the day it is predicting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .indicators import atr, bollinger_bands, macd, rsi, volume_ratio, gap_pct

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Feature engineering ───────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rsi",
    "macd_line",
    "macd_hist",
    "bb_pct_b",
    "bb_width",
    "atr_pct",
    "vol_ratio",
    "gap_pct",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "close_vs_sma20",
    "close_vs_sma50",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["rsi"] = rsi(df)

    ml, sl, hist = macd(df)
    feat["macd_line"] = ml
    feat["macd_hist"] = hist

    _, _, _, width, pct_b = bollinger_bands(df)
    feat["bb_pct_b"] = pct_b
    feat["bb_width"] = width

    atr_series = atr(df)
    feat["atr_pct"] = atr_series / df["Close"] * 100

    feat["vol_ratio"] = volume_ratio(df)
    feat["gap_pct"] = gap_pct(df)

    close = df["Close"]
    feat["ret_1d"] = close.pct_change(1) * 100
    feat["ret_3d"] = close.pct_change(3) * 100
    feat["ret_5d"] = close.pct_change(5) * 100
    feat["close_vs_sma20"] = (close / close.rolling(20).mean() - 1) * 100
    feat["close_vs_sma50"] = (close / close.rolling(50).mean() - 1) * 100

    # label: 1 if next day's close > today's close
    feat["label"] = (close.shift(-1) > close).astype(int)

    return feat.dropna()


# ── Walk-forward engine ───────────────────────────────────────────────────────

@dataclass
class DayResult:
    date: str
    predicted: int          # 0 = down, 1 = up
    actual: int
    correct: bool
    proba_up: float         # model confidence


@dataclass
class BacktestResult:
    ticker: str
    start: str
    end: str
    train_window: int
    total_days: int
    correct: int
    accuracy: float
    feature_importances: dict[str, float]
    day_results: list[DayResult] = field(default_factory=list)
    rolling_accuracy: list[float] = field(default_factory=list)  # accuracy every 20 steps


def run_backtest(
    ticker: str,
    period: str = "5y",
    start: Optional[str] = None,
    end: Optional[str] = None,
    train_window: int = 252,   # ~1 trading year
    step: int = 1,             # predict 1 day ahead each slide
    n_estimators: int = 100,
    min_test_days: int = 60,
) -> Optional[BacktestResult]:
    """
    Walk-forward backtest for one ticker.

    train_window: how many historical days to train on each iteration.
    step: how many days to slide the window forward each iteration.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required for ML backtesting.\n"
            "Install it with:  pip install scikit-learn"
        )

    if start or end:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    else:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty or len(raw) < train_window + min_test_days + 60:
        return None

    # yfinance sometimes returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    feat_df = _build_features(raw)
    if len(feat_df) < train_window + min_test_days:
        return None

    X = feat_df[FEATURE_NAMES].values
    y = feat_df["label"].values
    dates = feat_df.index

    day_results: list[DayResult] = []
    importances_accum = np.zeros(len(FEATURE_NAMES))
    n_models = 0
    rolling_acc_log: list[float] = []

    start_idx = train_window
    end_idx = len(X) - 1  # last row has no next-day label

    for i in range(start_idx, end_idx, step):
        X_train = X[i - train_window : i]
        y_train = y[i - train_window : i]
        X_test = X[i : i + 1]
        y_test = y[i]

        # skip if only one class in training window
        if len(set(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_s, y_train)

        pred = int(clf.predict(X_test_s)[0])
        proba = float(clf.predict_proba(X_test_s)[0][1])
        correct = pred == int(y_test)

        day_results.append(
            DayResult(
                date=str(dates[i].date()),
                predicted=pred,
                actual=int(y_test),
                correct=correct,
                proba_up=proba,
            )
        )

        importances_accum += clf.feature_importances_
        n_models += 1

        # log rolling accuracy every 20 predictions
        if len(day_results) % 20 == 0:
            recent = day_results[-20:]
            rolling_acc_log.append(sum(r.correct for r in recent) / 20 * 100)

    if not day_results:
        return None

    total = len(day_results)
    correct_count = sum(r.correct for r in day_results)
    accuracy = correct_count / total * 100

    avg_importances = importances_accum / max(n_models, 1)
    fi = dict(sorted(zip(FEATURE_NAMES, avg_importances), key=lambda x: -x[1]))

    return BacktestResult(
        ticker=ticker,
        start=day_results[0].date,
        end=day_results[-1].date,
        train_window=train_window,
        total_days=total,
        correct=correct_count,
        accuracy=accuracy,
        feature_importances=fi,
        day_results=day_results,
        rolling_accuracy=rolling_acc_log,
    )


# ── Paper trading simulator ───────────────────────────────────────────────────

@dataclass
class Trade:
    date: str
    action: str          # "BUY" | "SELL" | "HOLD"
    price: float
    shares: float
    cash: float
    portfolio_value: float
    proba_up: float


@dataclass
class PaperTradeResult:
    ticker: str
    train_start: str
    train_end: str
    trade_start: str
    trade_end: str
    starting_cash: float
    final_value: float
    buy_and_hold_value: float
    total_return_pct: float
    buy_hold_return_pct: float
    num_trades: int
    win_trades: int
    trades: list[Trade]
    daily_values: list[tuple[str, float]]   # (date, portfolio_value)


def run_paper_trade(
    ticker: str,
    train_start: str,
    train_end: str,
    trade_start: Optional[str] = None,
    trade_end: Optional[str] = None,
    starting_cash: float = 1000.0,
    train_window: int = 252,
    n_estimators: int = 100,
    min_confidence: float = 0.55,   # only act when model is this confident
) -> Optional[PaperTradeResult]:
    """
    Train on [train_start, train_end], then simulate daily trading on
    [trade_start, trade_end] (defaults to train_end -> today).

    Strategy: long-only.
      - If model predicts UP with >= min_confidence: buy (use all cash).
      - If model predicts DOWN: sell everything, hold cash.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required.\n"
            "Install it with:  pip install scikit-learn"
        )

    # Download full training history
    raw_train = yf.download(ticker, start=train_start, end=train_end,
                            auto_adjust=True, progress=False)
    if isinstance(raw_train.columns, pd.MultiIndex):
        raw_train.columns = raw_train.columns.droplevel(1)

    if raw_train.empty or len(raw_train) < train_window + 60:
        return None

    # Download trading period data
    td_start = trade_start or train_end
    td_end = trade_end  # None = today
    raw_trade = yf.download(ticker, start=td_start, end=td_end,
                            auto_adjust=True, progress=False)
    if isinstance(raw_trade.columns, pd.MultiIndex):
        raw_trade.columns = raw_trade.columns.droplevel(1)

    if raw_trade.empty or len(raw_trade) < 2:
        return None

    # Build features on the combined dataset so indicators have enough lookback
    raw_all = pd.concat([raw_train, raw_trade])
    raw_all = raw_all[~raw_all.index.duplicated(keep="last")]
    raw_all.sort_index(inplace=True)

    feat_all = _build_features(raw_all)

    # Identify the boundary index in the feature frame
    train_mask = feat_all.index < pd.Timestamp(td_start)
    trade_mask = ~train_mask

    feat_train = feat_all[train_mask]
    feat_trade = feat_all[trade_mask]

    if len(feat_train) < train_window or len(feat_trade) < 1:
        return None

    X_train_full = feat_train[FEATURE_NAMES].values
    y_train_full = feat_train["label"].values
    X_trade = feat_trade[FEATURE_NAMES].values
    trade_dates = feat_trade.index

    # Get actual close prices for the trading period
    close_trade = raw_trade["Close"].reindex(feat_trade.index).ffill()

    # Train final model on all training data
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_full[-train_window:])
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr_s, y_train_full[-train_window:])

    # Simulate trading day by day, retraining each day on a rolling window
    # that now includes days from the trading period as they pass
    X_rolling = list(X_train_full)
    y_rolling = list(y_train_full)

    cash = starting_cash
    shares = 0.0
    trades: list[Trade] = []
    daily_values: list[tuple[str, float]] = []
    buy_and_hold_shares = starting_cash / float(close_trade.iloc[0]) if close_trade.iloc[0] > 0 else 0

    for i, (date, row_x) in enumerate(zip(trade_dates, X_trade)):
        price = float(close_trade.loc[date]) if date in close_trade.index else None
        if price is None or price <= 0:
            continue

        # Retrain on rolling window each day
        X_win = np.array(X_rolling[-train_window:])
        y_win = np.array(y_rolling[-train_window:])

        if len(set(y_win)) >= 2:
            scaler_d = StandardScaler()
            X_win_s = scaler_d.fit_transform(X_win)
            clf_d = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            clf_d.fit(X_win_s, y_win)
            x_s = scaler_d.transform(row_x.reshape(1, -1))
            pred = int(clf_d.predict(x_s)[0])
            proba_up = float(clf_d.predict_proba(x_s)[0][1])
        else:
            pred = 1
            proba_up = 0.5

        # Trading logic
        action = "HOLD"
        if pred == 1 and proba_up >= min_confidence and cash > 0:
            # Buy with all available cash
            shares = cash / price
            cash = 0.0
            action = "BUY"
        elif pred == 0 and shares > 0:
            # Sell everything
            cash = shares * price
            shares = 0.0
            action = "SELL"

        portfolio_value = cash + shares * price
        daily_values.append((str(date.date()), portfolio_value))

        if action != "HOLD":
            trades.append(Trade(
                date=str(date.date()),
                action=action,
                price=price,
                shares=shares if action == "BUY" else 0.0,
                cash=cash,
                portfolio_value=portfolio_value,
                proba_up=proba_up,
            ))

        # Add this day's data to rolling history for tomorrow's retrain
        label_val = int(feat_trade["label"].iloc[i]) if i < len(feat_trade) else 1
        X_rolling.append(row_x)
        y_rolling.append(label_val)

    # Final liquidation
    final_price = float(close_trade.iloc[-1])
    final_value = cash + shares * final_price

    buy_hold_final = buy_and_hold_shares * final_price
    total_ret = (final_value - starting_cash) / starting_cash * 100
    bh_ret = (buy_hold_final - starting_cash) / starting_cash * 100

    sell_trades = [t for t in trades if t.action == "SELL"]
    buy_prices = {t.date: t.price for t in trades if t.action == "BUY"}
    win_trades = 0
    buy_price_val = None
    for t in trades:
        if t.action == "BUY":
            buy_price_val = t.price
        elif t.action == "SELL" and buy_price_val is not None:
            if t.price > buy_price_val:
                win_trades += 1
            buy_price_val = None

    return PaperTradeResult(
        ticker=ticker,
        train_start=train_start,
        train_end=train_end,
        trade_start=str(trade_dates[0].date()),
        trade_end=str(trade_dates[-1].date()),
        starting_cash=starting_cash,
        final_value=final_value,
        buy_and_hold_value=buy_hold_final,
        total_return_pct=total_ret,
        buy_hold_return_pct=bh_ret,
        num_trades=len(sell_trades),
        win_trades=win_trades,
        trades=trades,
        daily_values=daily_values,
    )
