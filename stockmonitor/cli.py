from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from datetime import date, timedelta
from . import scanner, watchlist, eli5, logger, grader, tuner

app = typer.Typer(
    name="stockmonitor",
    help="Scan stocks for breakout/spike/pullback signals using yfinance data.",
    add_completion=False,
)
console = Console()

SCORE_COLORS = {
    "bullish": "green",
    "bearish": "red",
    "neutral": "dim",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_color(score: int) -> str:
    if score >= 3:
        return "bold green"
    if score > 0:
        return "green"
    if score <= -3:
        return "bold red"
    if score < 0:
        return "red"
    return "dim"


def _build_table(signals: list[scanner.Signal]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, highlight=True)
    cols = ["Ticker", "Price", "Chg%", "RSI", "MACD", "BB", "VolRatio", "Gap%", "ATR%", "Score", "Alerts"]
    for col in cols:
        table.add_column(col, justify="right" if col not in ("Ticker", "MACD", "BB", "Alerts") else "left")

    for sig in signals:
        d = sig.to_dict()
        color = _score_color(sig.score)
        table.add_row(
            f"[bold]{d['Ticker']}[/bold]",
            d["Price"],
            f"[{color}]{d['Chg%']}[/{color}]",
            d["RSI"],
            d["MACD"],
            d["BB"],
            d["VolRatio"],
            d["Gap%"],
            d["ATR%"],
            f"[{color}]{d['Score']}[/{color}]",
            f"[{color}]{d['Alerts']}[/{color}]",
        )
    return table


# ── Commands ─────────────────────────────────────────────────────────────────

@app.command()
def scan(
    tickers: Optional[list[str]] = typer.Argument(None, help="Tickers to scan (defaults to watchlist)"),
    period: str = typer.Option("6mo", help="yfinance period string (1mo, 3mo, 6mo, 1y)"),
    rsi_oversold: int = typer.Option(35, help="RSI oversold threshold"),
    rsi_overbought: int = typer.Option(65, help="RSI overbought threshold"),
    vol_spike: float = typer.Option(2.0, help="Volume spike ratio vs 20-day avg"),
    gap_pct: float = typer.Option(1.5, help="Minimum gap % to flag"),
    atr_pct: float = typer.Option(3.0, help="ATR% threshold to flag elevated volatility"),
    min_score: int = typer.Option(0, help="Minimum absolute score to display"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Export results to CSV"),
    alerts_only: bool = typer.Option(False, "--alerts-only", "-a", help="Show only tickers with at least one alert"),
    no_eli5: bool = typer.Option(False, "--no-eli5", help="Skip the plain-English explanation"),
):
    """Scan stocks for breakout, spike, and pullback signals."""
    target = [t.upper() for t in tickers] if tickers else watchlist.load()

    # Start from auto-tuned config, then apply any explicit CLI overrides
    cfg = tuner.load_config()
    cfg.update({
        "period": period,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "vol_spike_ratio": vol_spike,
        "gap_up_pct": gap_pct,
        "atr_pct_threshold": atr_pct,
        "min_score": min_score,
    })

    console.print(f"\n[bold]Scanning {len(target)} ticker(s)…[/bold]\n")
    signals = scanner.scan_all(target, cfg)

    if alerts_only:
        signals = [s for s in signals if s.alerts]
    if min_score > 0:
        signals = [s for s in signals if abs(s.score) >= min_score]

    if not signals:
        console.print("[yellow]No signals matched the current thresholds.[/yellow]")
        raise typer.Exit()

    console.print(_build_table(signals))

    if not no_eli5:
        console.print("\n[bold underline]Plain-English Summary[/bold underline]\n")
        console.print(eli5.explain_all(signals))
        console.print()

    if output:
        rows = [s.to_dict() for s in signals]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def watch(
    action: str = typer.Argument(..., help="add | remove | list"),
    ticker: Optional[str] = typer.Argument(None, help="Ticker symbol"),
):
    """Manage your watchlist."""
    if action == "list":
        tickers = watchlist.load()
        console.print("[bold]Watchlist:[/bold]", ", ".join(tickers))
    elif action == "add":
        if not ticker:
            console.print("[red]Provide a ticker to add.[/red]")
            raise typer.Exit(1)
        tickers = watchlist.add(ticker)
        console.print(f"[green]Added {ticker.upper()}.[/green] Watchlist: {', '.join(tickers)}")
    elif action == "remove":
        if not ticker:
            console.print("[red]Provide a ticker to remove.[/red]")
            raise typer.Exit(1)
        tickers = watchlist.remove(ticker)
        console.print(f"[yellow]Removed {ticker.upper()}.[/yellow] Watchlist: {', '.join(tickers)}")
    else:
        console.print(f"[red]Unknown action '{action}'. Use add, remove, or list.[/red]")
        raise typer.Exit(1)


@app.command()
def log(
    tickers: Optional[list[str]] = typer.Argument(None, help="Tickers to scan and log (defaults to watchlist)"),
    period: str = typer.Option("6mo", help="yfinance period string"),
):
    """Scan stocks and save today's predictions to disk for next-day grading."""
    target = [t.upper() for t in tickers] if tickers else watchlist.load()
    cfg = tuner.load_config()
    cfg["period"] = period

    console.print(f"\n[bold]Scanning and logging {len(target)} ticker(s)…[/bold]\n")
    signals = scanner.scan_all(target, cfg)

    if not signals:
        console.print("[yellow]No signals to log.[/yellow]")
        raise typer.Exit()

    console.print(_build_table(signals))
    path = logger.save_snapshot(signals)
    console.print(f"\n[green]Predictions saved → {path}[/green]")
    console.print("[dim]Run [bold]stockmonitor grade[/bold] tomorrow to see how accurate they were.[/dim]\n")


@app.command()
def grade(
    predict_date: Optional[str] = typer.Argument(None, help="Date to grade as YYYY-MM-DD (defaults to yesterday)"),
    actual_date: Optional[str] = typer.Argument(None, help="Date to compare against as YYYY-MM-DD (defaults to today)"),
    auto_tune: bool = typer.Option(True, help="Automatically tune thresholds after grading"),
):
    """Grade yesterday's predictions against today's actual price moves."""
    pdate = date.fromisoformat(predict_date) if predict_date else date.today() - timedelta(days=1)
    adate = date.fromisoformat(actual_date) if actual_date else date.today()

    console.print(f"\n[bold]Grading predictions from {pdate} vs actual prices on {adate}…[/bold]\n")
    result = grader.grade_day(pdate, adate)

    if not result:
        console.print(f"[red]No prediction log found for {pdate}. Run [bold]stockmonitor log[/bold] first.[/red]")
        raise typer.Exit(1)

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    for col in ["Ticker", "Predicted", "Score", "Actual Chg%", "Outcome"]:
        table.add_column(col, justify="left" if col in ("Ticker", "Predicted", "Outcome") else "right")

    for r in result["results"]:
        outcome = r["outcome"]
        color = "green" if outcome == "correct" else ("red" if outcome == "incorrect" else "dim")
        chg = f"{r['actual_change_pct']:+.2f}%" if r["actual_change_pct"] is not None else "—"
        table.add_row(
            f"[bold]{r['ticker']}[/bold]",
            r["predicted_direction"],
            f"{r['score']:+d}",
            chg,
            f"[{color}]{outcome}[/{color}]",
        )

    console.print(table)

    scored = [r for r in result["results"] if r["correct"] is not None]
    if scored:
        correct = sum(1 for r in scored if r["correct"])
        pct = correct / len(scored) * 100
        color = "green" if pct >= 60 else ("yellow" if pct >= 45 else "red")
        console.print(f"\n[bold]Hit rate today:[/bold] [{color}]{correct}/{len(scored)} ({pct:.0f}%)[/{color}]\n")

    if auto_tune:
        _, changes = tuner.tune()
        if changes and changes[0] != "Not enough graded data yet (need at least 10 graded predictions).":
            console.print("[bold underline]Threshold Adjustments[/bold underline]")
            for c in changes:
                console.print(f"  • {c}")
            console.print()


@app.command()
def accuracy():
    """Show historical prediction accuracy and current threshold settings."""
    stats = grader.compute_accuracy()

    if not stats:
        console.print("[yellow]No graded history yet. Run [bold]stockmonitor log[/bold] then [bold]stockmonitor grade[/bold] the next day.[/yellow]")
        raise typer.Exit()

    pct = stats["overall_pct"]
    color = "green" if pct >= 60 else ("yellow" if pct >= 45 else "red")
    console.print(
        f"\n[bold]Overall accuracy:[/bold] [{color}]{stats['overall_correct']}/{stats['overall_total']} "
        f"({pct}%)[/{color}] across [bold]{stats['days_graded']}[/bold] graded day(s)\n"
    )

    if stats.get("by_signal"):
        table = Table(box=box.SIMPLE_HEAVY, title="Accuracy by Signal", show_lines=False)
        table.add_column("Signal", justify="left")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Hit Rate", justify="right")

        for sig, s in stats["by_signal"].items():
            p = s["pct"]
            color = "green" if p >= 60 else ("yellow" if p >= 45 else "red")
            table.add_row(sig, str(s["correct"]), str(s["total"]), f"[{color}]{p}%[/{color}]")

        console.print(table)

    cfg = tuner.load_config()
    console.print("\n[bold]Current thresholds (auto-tuned):[/bold]")
    console.print(f"  RSI oversold  : {cfg.get('rsi_oversold', 35)}")
    console.print(f"  RSI overbought: {cfg.get('rsi_overbought', 65)}")
    console.print(f"  Volume spike  : {cfg.get('vol_spike_ratio', 2.0)}x\n")


def main() -> None:
    app()
