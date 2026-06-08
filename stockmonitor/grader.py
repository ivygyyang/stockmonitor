from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

from .logger import load_snapshot, list_log_dates

GRADES_DIR = Path.home() / ".stockmonitor" / "grades"


def _grade_path(day: date) -> Path:
    return GRADES_DIR / f"{day.isoformat()}.json"


def _fetch_actual_change(ticker: str, after: date) -> float | None:
    """Return the actual % change on `after` date."""
    try:
        data = yf.download(
            ticker,
            start=after.isoformat(),
            end=(after + timedelta(days=7)).isoformat(),  # buffer for weekends/holidays
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None
        if isinstance(data.columns, __import__("pandas").MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # first available trading day on or after `after`
        row = data.iloc[0]
        prev = data["Close"].shift(1).iloc[0]
        if prev != prev:  # NaN on first row, use Open as proxy
            return float((row["Close"] - row["Open"]) / row["Open"] * 100)
        return float((row["Close"] - prev) / prev * 100)
    except Exception:
        return None


def grade_day(predict_date: date, actual_date: date | None = None) -> dict | None:
    """
    Grade predictions made on `predict_date` against prices on `actual_date`.
    actual_date defaults to the next calendar day (tool finds next trading day).
    """
    snapshot = load_snapshot(predict_date)
    if not snapshot:
        return None

    actual_date = actual_date or (predict_date + timedelta(days=1))
    GRADES_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for rec in snapshot["signals"]:
        ticker = rec["ticker"]
        predicted = rec["predicted_direction"]
        actual_chg = _fetch_actual_change(ticker, actual_date)

        if actual_chg is None:
            outcome = "unknown"
            correct = None
        else:
            actual_dir = "up" if actual_chg > 0.5 else ("down" if actual_chg < -0.5 else "flat")
            if predicted == "neutral":
                correct = None
                outcome = "no_prediction"
            elif predicted == actual_dir:
                correct = True
                outcome = "correct"
            elif actual_dir == "flat":
                correct = None
                outcome = "flat"
            else:
                correct = False
                outcome = "incorrect"

        results.append(
            {
                "ticker": ticker,
                "predicted_direction": predicted,
                "score": rec["score"],
                "alerts": rec["alerts"],
                "actual_change_pct": round(actual_chg, 2) if actual_chg is not None else None,
                "outcome": outcome,
                "correct": correct,
            }
        )

    # Import here to avoid circular imports at module load
    from . import advisor as _advisor
    from . import journal as _journal

    directives = _advisor.generate_directives(actual_date)

    grade_record = {
        "predict_date": predict_date.isoformat(),
        "actual_date": actual_date.isoformat(),
        "results": results,
        "self_improvement": directives,
    }
    _grade_path(predict_date).write_text(json.dumps(grade_record, indent=2))
    _journal.append_entry(directives)
    return grade_record


def load_all_grades() -> list[dict]:
    if not GRADES_DIR.exists():
        return []
    grades = []
    for f in sorted(GRADES_DIR.glob("????-??-??.json")):
        try:
            grades.append(json.loads(f.read_text()))
        except Exception:
            pass
    return grades


def compute_accuracy() -> dict:
    """Aggregate accuracy stats across all graded days."""
    grades = load_all_grades()
    if not grades:
        return {}

    signal_stats: dict[str, dict] = {}
    overall_correct = 0
    overall_total = 0

    for grade in grades:
        for r in grade["results"]:
            if r["correct"] is None:
                continue
            overall_total += 1
            if r["correct"]:
                overall_correct += 1

            for alert in r["alerts"]:
                # normalise alert label to first word(s)
                key = alert.split("(")[0].strip()
                if key not in signal_stats:
                    signal_stats[key] = {"correct": 0, "total": 0}
                signal_stats[key]["total"] += 1
                if r["correct"]:
                    signal_stats[key]["correct"] += 1

    signal_accuracy = {
        k: {
            "correct": v["correct"],
            "total": v["total"],
            "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
        }
        for k, v in sorted(signal_stats.items(), key=lambda x: -x[1]["total"])
    }

    return {
        "days_graded": len(grades),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_pct": round(overall_correct / overall_total * 100, 1) if overall_total else 0,
        "by_signal": signal_accuracy,
    }
