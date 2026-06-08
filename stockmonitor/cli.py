from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from . import scanner, watchlist

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
):
    """Scan stocks for breakout, spike, and pullback signals."""
    target = [t.upper() for t in tickers] if tickers else watchlist.load()

    cfg = {
        "period": period,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "vol_spike_ratio": vol_spike,
        "gap_up_pct": gap_pct,
        "atr_pct_threshold": atr_pct,
        "min_score": min_score,
    }

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


def main() -> None:
    app()
