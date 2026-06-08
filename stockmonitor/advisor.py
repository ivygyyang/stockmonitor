"""
Analyzes graded prediction history and generates self-improvement directives.

Directives fall into two categories:
  - AUTO: can be applied immediately by tuner.py (threshold changes, weight nudges)
  - MANUAL: pattern insights that require a developer to update scoring logic
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from .grader import load_all_grades

MIN_SAMPLES = 5  # minimum predictions before drawing conclusions about a signal
RECENT_WINDOW = 7  # days to consider "recent" for trend analysis


def _signal_stats(grades: list[dict]) -> dict[str, dict]:
    """Aggregate correct/total per signal label across all grades."""
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "recent_correct": 0, "recent_total": 0})
    cutoff = sorted(g["predict_date"] for g in grades)[-RECENT_WINDOW:] if grades else []

    for grade in grades:
        is_recent = grade["predict_date"] in cutoff
        for r in grade["results"]:
            if r["correct"] is None:
                continue
            for alert in r["alerts"]:
                key = alert.split("(")[0].strip()
                stats[key]["total"] += 1
                if r["correct"]:
                    stats[key]["correct"] += 1
                if is_recent:
                    stats[key]["recent_total"] += 1
                    if r["correct"]:
                        stats[key]["recent_correct"] += 1
    return dict(stats)


def _ticker_stats(grades: list[dict]) -> dict[str, dict]:
    """Aggregate correct/total per ticker."""
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for grade in grades:
        for r in grade["results"]:
            if r["correct"] is None:
                continue
            stats[r["ticker"]]["total"] += 1
            if r["correct"]:
                stats[r["ticker"]]["correct"] += 1
    return dict(stats)


def _combo_stats(grades: list[dict]) -> dict[str, dict]:
    """Check accuracy when two signals fire together."""
    combos: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for grade in grades:
        for r in grade["results"]:
            if r["correct"] is None or len(r["alerts"]) < 2:
                continue
            keys = sorted(set(a.split("(")[0].strip() for a in r["alerts"]))
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    combo = f"{keys[i]} + {keys[j]}"
                    combos[combo]["total"] += 1
                    if r["correct"]:
                        combos[combo]["correct"] += 1
    return dict(combos)


def _trend(stats: dict) -> str:
    """Is the signal getting better or worse recently?"""
    if stats["recent_total"] < 3:
        return "insufficient recent data"
    overall_pct = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
    recent_pct = stats["recent_correct"] / stats["recent_total"] * 100
    delta = recent_pct - overall_pct
    if delta > 10:
        return f"improving (recent {recent_pct:.0f}% vs overall {overall_pct:.0f}%)"
    if delta < -10:
        return f"degrading (recent {recent_pct:.0f}% vs overall {overall_pct:.0f}%)"
    return f"stable (recent {recent_pct:.0f}% vs overall {overall_pct:.0f}%)"


def generate_directives(today: date | None = None) -> dict:
    """
    Run full analysis and return structured improvement directives.
    Returns a dict with 'auto' and 'manual' directive lists, plus raw stats.
    """
    today = today or date.today()
    grades = load_all_grades()

    auto_directives: list[str] = []
    manual_directives: list[str] = []

    if not grades:
        return {
            "date": today.isoformat(),
            "auto": ["No graded history yet. Run `stockmonitor log` then `stockmonitor grade` the next day."],
            "manual": [],
            "signal_stats": {},
            "ticker_stats": {},
            "combo_stats": {},
        }

    sig_stats = _signal_stats(grades)
    ticker_stats = _ticker_stats(grades)
    combo_stats = _combo_stats(grades)

    total_graded = sum(
        1 for g in grades for r in g["results"] if r["correct"] is not None
    )
    total_correct = sum(
        1 for g in grades for r in g["results"] if r["correct"] is True
    )
    overall_pct = total_correct / total_graded * 100 if total_graded else 0

    # ── Per-signal analysis ───────────────────────────────────────────────────
    for signal, s in sig_stats.items():
        if s["total"] < MIN_SAMPLES:
            continue
        pct = s["correct"] / s["total"] * 100
        trend = _trend(s)

        if pct < 40:
            auto_directives.append(
                f"AUTO | TIGHTEN | {signal}: {pct:.0f}% accuracy over {s['total']} samples ({trend}). "
                f"Raise the bar for this signal to fire."
            )
            manual_directives.append(
                f"MANUAL | REVIEW | {signal} is underperforming ({pct:.0f}%). "
                f"Consider whether this signal adds value or needs a confirming condition before scoring."
            )
        elif pct > 68:
            auto_directives.append(
                f"AUTO | RELAX | {signal}: {pct:.0f}% accuracy over {s['total']} samples ({trend}). "
                f"This signal is reliable — consider relaxing threshold slightly to catch more setups."
            )
        elif 40 <= pct < 48 and s["total"] >= 10:
            auto_directives.append(
                f"AUTO | MONITOR | {signal}: {pct:.0f}% accuracy ({trend}). "
                f"Borderline performance — watching for further degradation."
            )

    # ── Per-ticker analysis ───────────────────────────────────────────────────
    for ticker, s in ticker_stats.items():
        if s["total"] < MIN_SAMPLES:
            continue
        pct = s["correct"] / s["total"] * 100
        if pct < 38:
            manual_directives.append(
                f"MANUAL | TICKER | {ticker}: only {pct:.0f}% accurate over {s['total']} predictions. "
                f"This ticker may need custom thresholds or a different indicator set."
            )
        elif pct > 72:
            manual_directives.append(
                f"MANUAL | TICKER | {ticker}: {pct:.0f}% accurate over {s['total']} predictions. "
                f"Model works well on this ticker — good candidate for higher position sizing."
            )

    # ── Signal combination analysis ───────────────────────────────────────────
    for combo, s in combo_stats.items():
        if s["total"] < MIN_SAMPLES:
            continue
        pct = s["correct"] / s["total"] * 100
        if pct > 70:
            manual_directives.append(
                f"MANUAL | COMBO | '{combo}' fires together with {pct:.0f}% accuracy over {s['total']} cases. "
                f"Consider giving this combination a bonus score multiplier in scanner.py."
            )
        elif pct < 38:
            manual_directives.append(
                f"MANUAL | COMBO | '{combo}' together is only {pct:.0f}% accurate over {s['total']} cases. "
                f"These signals may cancel each other out — consider reducing score when both fire."
            )

    # ── Overall health check ──────────────────────────────────────────────────
    if overall_pct < 45 and total_graded >= 20:
        manual_directives.append(
            f"MANUAL | SYSTEM | Overall accuracy is {overall_pct:.0f}% across {total_graded} predictions. "
            f"Consider revisiting the core scoring logic in scanner.py — the current signal weights may not suit current market conditions."
        )
    elif overall_pct > 65 and total_graded >= 20:
        manual_directives.append(
            f"MANUAL | SYSTEM | Overall accuracy is {overall_pct:.0f}% across {total_graded} predictions. "
            f"Model is performing well. Consider expanding the watchlist to more volatile tickers."
        )

    if not auto_directives:
        auto_directives.append(
            f"AUTO | OK | All signals within acceptable range ({overall_pct:.0f}% overall). No threshold changes needed."
        )
    if not manual_directives:
        manual_directives.append(
            f"MANUAL | OK | No structural changes recommended at this time ({total_graded} predictions graded)."
        )

    return {
        "date": today.isoformat(),
        "overall_accuracy_pct": round(overall_pct, 1),
        "total_graded": total_graded,
        "auto": auto_directives,
        "manual": manual_directives,
        "signal_stats": {
            k: {**v, "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0}
            for k, v in sig_stats.items()
        },
        "ticker_stats": {
            k: {**v, "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0}
            for k, v in ticker_stats.items()
        },
        "combo_stats": {
            k: {**v, "pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0}
            for k, v in combo_stats.items()
        },
    }
