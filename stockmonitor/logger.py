from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from .scanner import Signal

LOGS_DIR = Path.home() / ".stockmonitor" / "logs"


def _log_path(day: date) -> Path:
    return LOGS_DIR / f"{day.isoformat()}.json"


def save_snapshot(signals: list[Signal], day: date | None = None) -> Path:
    day = day or date.today()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _log_path(day)
    records = [
        {
            "ticker": s.ticker,
            "price": s.price,
            "change_pct": s.change_pct,
            "rsi": s.rsi_val,
            "macd": s.macd_cross,
            "bb": s.bb_position,
            "vol_ratio": s.vol_spike,
            "atr_pct": s.atr_pct,
            "score": s.score,
            "alerts": s.alerts,
            # prediction: positive score = expects up, negative = expects down
            "predicted_direction": "up" if s.score > 0 else ("down" if s.score < 0 else "neutral"),
        }
        for s in signals
    ]
    payload = {"date": day.isoformat(), "saved_at": datetime.now().isoformat(), "signals": records}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_snapshot(day: date) -> dict | None:
    path = _log_path(day)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_log_dates() -> list[date]:
    if not LOGS_DIR.exists():
        return []
    dates = []
    for f in sorted(LOGS_DIR.glob("????-??-??.json")):
        try:
            dates.append(date.fromisoformat(f.stem))
        except ValueError:
            pass
    return dates
