from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from .indicators import atr, bollinger_bands, gap_pct, macd, rsi, volume_ratio

warnings.filterwarnings("ignore")


@dataclass
class Signal:
    ticker: str
    price: float
    change_pct: float
    rsi_val: float
    macd_cross: str          # "bullish" | "bearish" | "—"
    bb_position: str         # "squeeze" | "upper_breakout" | "lower_breakout" | "—"
    vol_spike: float         # ratio vs 20-day avg
    gap: float               # gap % vs prior close
    atr_pct: float           # ATR as % of price
    alerts: list[str] = field(default_factory=list)
    score: int = 0           # positive = bullish bias, negative = bearish bias

    def to_dict(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Price": f"${self.price:.2f}",
            "Chg%": f"{self.change_pct:+.2f}%",
            "RSI": f"{self.rsi_val:.1f}",
            "MACD": self.macd_cross,
            "BB": self.bb_position,
            "VolRatio": f"{self.vol_spike:.2f}x",
            "Gap%": f"{self.gap:+.2f}%",
            "ATR%": f"{self.atr_pct:.2f}%",
            "Score": f"{self.score:+d}",
            "Alerts": ", ".join(self.alerts) if self.alerts else "—",
        }


def _score_signal(sig: Signal, cfg: dict) -> None:
    alerts = []
    score = 0

    # RSI
    if sig.rsi_val <= cfg["rsi_oversold"]:
        alerts.append(f"RSI oversold ({sig.rsi_val:.0f})")
        score += 2
    elif sig.rsi_val >= cfg["rsi_overbought"]:
        alerts.append(f"RSI overbought ({sig.rsi_val:.0f})")
        score -= 2

    # MACD crossover
    if sig.macd_cross == "bullish":
        alerts.append("MACD bullish cross")
        score += 2
    elif sig.macd_cross == "bearish":
        alerts.append("MACD bearish cross")
        score -= 2

    # Bollinger Band
    if sig.bb_position == "squeeze":
        alerts.append("BB squeeze (volatility coiling)")
        score += 1
    elif sig.bb_position == "upper_breakout":
        alerts.append("BB upper breakout")
        score += 2
    elif sig.bb_position == "lower_breakout":
        alerts.append("BB lower breakout")
        score -= 2

    # Volume spike
    if sig.vol_spike >= cfg["vol_spike_ratio"]:
        alerts.append(f"Volume spike ({sig.vol_spike:.1f}x avg)")
        score += 2 if score >= 0 else -2  # amplify existing bias

    # Gap
    if sig.gap >= cfg["gap_up_pct"]:
        alerts.append(f"Gap up ({sig.gap:+.1f}%)")
        score += 1
    elif sig.gap <= -cfg["gap_up_pct"]:
        alerts.append(f"Gap down ({sig.gap:+.1f}%)")
        score -= 1

    # High ATR = elevated volatility environment
    if sig.atr_pct >= cfg["atr_pct_threshold"]:
        alerts.append(f"High ATR ({sig.atr_pct:.2f}%)")

    sig.alerts = alerts
    sig.score = score


def _detect_macd_cross(hist_df: pd.DataFrame) -> str:
    _, _, histogram = macd(hist_df)
    if len(histogram.dropna()) < 2:
        return "—"
    last = histogram.iloc[-1]
    prev = histogram.iloc[-2]
    if prev < 0 and last > 0:
        return "bullish"
    if prev > 0 and last < 0:
        return "bearish"
    return "—"


def _detect_bb_position(hist_df: pd.DataFrame, cfg: dict) -> str:
    _, _, _, width, pct_b = bollinger_bands(hist_df)
    if width.isna().all():
        return "—"
    cur_width = width.iloc[-1]
    cur_pct_b = pct_b.iloc[-1]
    avg_width = width.rolling(50).mean().iloc[-1]

    if not np.isnan(avg_width) and cur_width < avg_width * cfg["bb_squeeze_ratio"]:
        return "squeeze"
    if cur_pct_b >= 1.0:
        return "upper_breakout"
    if cur_pct_b <= 0.0:
        return "lower_breakout"
    return "—"


def scan_ticker(ticker: str, cfg: dict) -> Signal | None:
    try:
        data = yf.download(
            ticker,
            period=cfg.get("period", "6mo"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 30:
            return None

        # Flatten MultiIndex columns if present (yfinance >=0.2.38)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data["Close"]
        price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (price - prev_close) / prev_close * 100

        rsi_series = rsi(data)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0

        vol_series = volume_ratio(data)
        vol_spike = float(vol_series.iloc[-1]) if not vol_series.isna().all() else 1.0

        gap_series = gap_pct(data)
        gap = float(gap_series.iloc[-1]) if not gap_series.isna().all() else 0.0

        atr_series = atr(data)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.isna().all() else 0.0
        atr_pct = atr_val / price * 100

        macd_cross = _detect_macd_cross(data)
        bb_pos = _detect_bb_position(data, cfg)

        sig = Signal(
            ticker=ticker,
            price=price,
            change_pct=change_pct,
            rsi_val=rsi_val,
            macd_cross=macd_cross,
            bb_position=bb_pos,
            vol_spike=vol_spike,
            gap=gap,
            atr_pct=atr_pct,
        )
        _score_signal(sig, cfg)
        return sig

    except Exception:
        return None


DEFAULT_CFG = {
    "period": "6mo",
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "vol_spike_ratio": 2.0,
    "gap_up_pct": 1.5,
    "atr_pct_threshold": 3.0,
    "bb_squeeze_ratio": 0.75,
    "min_score": 0,
}


def scan_all(tickers: list[str], cfg: dict | None = None) -> list[Signal]:
    effective_cfg = {**DEFAULT_CFG, **(cfg or {})}
    results = []
    for ticker in tickers:
        sig = scan_ticker(ticker, effective_cfg)
        if sig is not None:
            results.append(sig)
    results.sort(key=lambda s: abs(s.score), reverse=True)
    return results
