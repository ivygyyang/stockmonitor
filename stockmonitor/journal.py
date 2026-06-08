"""
Maintains a running improvement journal at ~/.stockmonitor/journal.md.

Every time `stockmonitor grade` runs, the advisor's directives are appended
here in human-readable form. This journal is the tool's long-term memory —
it accumulates insights over time so developers can read it and implement
improvements that go beyond automatic threshold tuning.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

JOURNAL_PATH = Path.home() / ".stockmonitor" / "journal.md"


def append_entry(directives: dict) -> None:
    """Append a new journal entry from advisor directives."""
    day = directives.get("date", date.today().isoformat())
    overall = directives.get("overall_accuracy_pct", "—")
    total = directives.get("total_graded", 0)
    auto = directives.get("auto", [])
    manual = directives.get("manual", [])
    sig_stats = directives.get("signal_stats", {})
    ticker_stats = directives.get("ticker_stats", {})
    combo_stats = directives.get("combo_stats", {})

    lines = [
        f"\n---\n",
        f"## {day}  |  Overall accuracy: {overall}%  |  Total graded: {total}\n",
    ]

    if sig_stats:
        lines.append("### Signal Performance\n")
        lines.append("| Signal | Correct | Total | Hit Rate | Trend |\n")
        lines.append("|--------|---------|-------|----------|---------|\n")
        for sig, s in sorted(sig_stats.items(), key=lambda x: -x[1]["total"]):
            trend_raw = s.get("recent_correct", 0)
            trend_total = s.get("recent_total", 0)
            trend_str = f"{trend_raw}/{trend_total} recent" if trend_total else "—"
            lines.append(f"| {sig} | {s['correct']} | {s['total']} | {s['pct']}% | {trend_str} |\n")
        lines.append("\n")

    if ticker_stats:
        lines.append("### Per-Ticker Performance\n")
        lines.append("| Ticker | Correct | Total | Hit Rate |\n")
        lines.append("|--------|---------|-------|----------|\n")
        for ticker, s in sorted(ticker_stats.items(), key=lambda x: -x[1]["total"]):
            lines.append(f"| {ticker} | {s['correct']} | {s['total']} | {s['pct']}% |\n")
        lines.append("\n")

    if combo_stats:
        lines.append("### Signal Combination Performance\n")
        lines.append("| Combo | Correct | Total | Hit Rate |\n")
        lines.append("|-------|---------|-------|----------|\n")
        for combo, s in sorted(combo_stats.items(), key=lambda x: -x[1]["total"]):
            if s["total"] >= 3:
                lines.append(f"| {combo} | {s['correct']} | {s['total']} | {s['pct']}% |\n")
        lines.append("\n")

    if auto:
        lines.append("### Auto-Applied Directives\n")
        for d in auto:
            lines.append(f"- {d}\n")
        lines.append("\n")

    if manual:
        lines.append("### Manual Improvement Notes\n")
        lines.append("_These require a developer to review and implement in the codebase._\n\n")
        for d in manual:
            lines.append(f"- {d}\n")
        lines.append("\n")

    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write header if this is a new journal
    if not JOURNAL_PATH.exists():
        header = (
            "# StockMonitor Improvement Journal\n\n"
            "This file is automatically maintained by `stockmonitor grade`.\n"
            "It records daily prediction accuracy and self-improvement directives.\n\n"
            "**AUTO** directives are applied immediately by the tuner.\n"
            "**MANUAL** directives require a developer to review and implement.\n"
        )
        JOURNAL_PATH.write_text(header)

    with open(JOURNAL_PATH, "a") as f:
        f.writelines(lines)


def read_journal(last_n: int | None = None) -> str:
    if not JOURNAL_PATH.exists():
        return "No journal entries yet. Run `stockmonitor log` then `stockmonitor grade` the next day."
    text = JOURNAL_PATH.read_text()
    if last_n is None:
        return text
    # Return only the last N entries (split on the --- dividers)
    entries = text.split("\n---\n")
    return "\n---\n".join(entries[-last_n:])


def journal_path() -> Path:
    return JOURNAL_PATH
