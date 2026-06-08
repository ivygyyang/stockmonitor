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
from . import scanner, watchlist, eli5, logger, grader, tuner, journal, ml_agent, live_trader

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


@app.command()
def journal_cmd(
    last: int = typer.Option(5, "--last", "-n", help="Show last N entries"),
    path: bool = typer.Option(False, "--path", help="Print journal file path and exit"),
):
    """View the self-improvement journal — accumulated insights from graded predictions."""
    if path:
        console.print(str(journal.journal_path()))
        raise typer.Exit()
    console.print(f"\n[bold underline]Improvement Journal[/bold underline] (last {last} entries)\n")
    text = journal.read_journal(last_n=last)
    # Render as plain text — markdown tables look fine in terminal
    console.print(text)


@app.command()
def schedule():
    """Set up Windows Task Scheduler to run stockmonitor log + grade automatically each day."""
    import subprocess, sys, shutil

    sm = shutil.which("stockmonitor")
    if not sm:
        console.print("[red]stockmonitor command not found in PATH. Make sure you ran `pip install -e .`[/red]")
        raise typer.Exit(1)

    # log task: runs at 6:30 AM daily
    log_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2024-01-01T06:30:00</StartBoundary>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>{sm}</Command>
      <Arguments>log</Arguments>
    </Exec>
  </Actions>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy></Settings>
</Task>"""

    # grade task: runs at 9:00 AM daily (after market open, after log)
    grade_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2024-01-01T09:00:00</StartBoundary>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>{sm}</Command>
      <Arguments>grade</Arguments>
    </Exec>
  </Actions>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy></Settings>
</Task>"""

    import tempfile, os
    tasks = [
        ("StockMonitor_DailyLog", log_xml, "6:30 AM"),
        ("StockMonitor_DailyGrade", grade_xml, "9:00 AM"),
    ]
    success = True
    for name, xml, time_str in tasks:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-16") as f:
            f.write(xml)
            tmp = f.name
        try:
            result = subprocess.run(
                ["schtasks", "/Create", "/TN", name, "/XML", tmp, "/F"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                console.print(f"[green]✓ Scheduled '{name}' at {time_str} daily[/green]")
            else:
                console.print(f"[red]✗ Failed to schedule '{name}': {result.stderr.strip()}[/red]")
                success = False
        finally:
            os.unlink(tmp)

    if success:
        console.print("\n[bold]All done.[/bold] stockmonitor will now:")
        console.print("  • Save predictions every morning at [bold]6:30 AM[/bold]")
        console.print("  • Grade them and update the journal every morning at [bold]9:00 AM[/bold]")
        console.print(f"\nView the journal anytime with: [bold]stockmonitor journal[/bold]")


@app.command(name="ml-backtest")
def ml_backtest_cmd(
    tickers: Optional[list[str]] = typer.Argument(None, help="Tickers to backtest (defaults to watchlist)"),
    period: str = typer.Option("5y", help="History to download: 2y, 5y, 10y, max (ignored if --start is set)"),
    start: Optional[str] = typer.Option(None, help="Start date as YYYY-MM-DD, e.g. 2000-01-01"),
    end: Optional[str] = typer.Option(None, help="End date as YYYY-MM-DD (defaults to today)"),
    train_window: int = typer.Option(252, help="Trading days used to train each model iteration (~1yr = 252)"),
    n_estimators: int = typer.Option(100, help="Number of trees in each Random Forest"),
    top_features: int = typer.Option(5, help="Number of top features to display"),
    show_curve: bool = typer.Option(True, help="Show rolling accuracy learning curve"),
):
    """Walk-forward ML backtest: train on old data, predict, measure, repeat."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        console.print("[red]scikit-learn is not installed.[/red]")
        console.print("Run:  [bold]pip install scikit-learn[/bold]")
        raise typer.Exit(1)

    target = [t.upper() for t in tickers] if tickers else watchlist.load()
    date_label = f"{start} to {end or 'today'}" if start else f"{period} of history"
    console.print(f"\n[bold]ML Walk-Forward Backtest[/bold] - {len(target)} ticker(s), {date_label}\n")
    console.print(
        f"[dim]Train window: {train_window} days | "
        f"Features: RSI, MACD, Bollinger, ATR, Volume, returns, SMAs[/dim]\n"
    )

    summary_table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    for col in ["Ticker", "Period", "Test Days", "Correct", "Accuracy", "vs Coin Flip"]:
        summary_table.add_column(col, justify="right" if col not in ("Ticker", "Period") else "left")

    results = []
    for ticker in target:
        console.print(f"  Backtesting [bold]{ticker}[/bold]…", end="")
        try:
            result = ml_agent.run_backtest(
                ticker,
                period=period,
                start=start,
                end=end,
                train_window=train_window,
                n_estimators=n_estimators,
            )
        except ImportError as e:
            console.print(f"\n[red]{e}[/red]")
            raise typer.Exit(1)

        if result is None:
            console.print(" [yellow]not enough data, skipped[/yellow]")
            continue

        console.print(f" done  ({result.total_days} test days)")
        results.append(result)

        acc = result.accuracy
        vs_coin = acc - 50.0
        acc_color = "green" if acc >= 55 else ("yellow" if acc >= 50 else "red")
        vs_color = "green" if vs_coin > 0 else "red"
        summary_table.add_row(
            f"[bold]{ticker}[/bold]",
            f"{result.start} to {result.end}",
            str(result.total_days),
            str(result.correct),
            f"[{acc_color}]{acc:.1f}%[/{acc_color}]",
            f"[{vs_color}]{vs_coin:+.1f}%[/{vs_color}]",
        )

    if not results:
        console.print("[yellow]No results — try a broader period or more tickers.[/yellow]")
        raise typer.Exit()

    console.print()
    console.print(summary_table)

    # Per-ticker feature importances and learning curve
    for result in results:
        console.print(f"\n[bold underline]{result.ticker} - Top {top_features} Predictive Features[/bold underline]")
        fi_items = list(result.feature_importances.items())[:top_features]
        max_imp = fi_items[0][1] if fi_items else 1.0
        for name, imp in fi_items:
            bar_len = int(imp / max_imp * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            console.print(f"  {name:<20} {bar} {imp*100:.1f}%")

        if show_curve and result.rolling_accuracy:
            console.print(f"\n  [dim]Rolling accuracy (every 20 predictions, oldest to newest):[/dim]")
            curve_vals = result.rolling_accuracy
            for i, v in enumerate(curve_vals):
                bar_len = int((v - 40) / 20 * 20) if v > 40 else 0  # scale 40-60% range to 0-20 chars
                bar_len = max(0, min(bar_len, 20))
                color = "green" if v >= 55 else ("yellow" if v >= 50 else "red")
                label = f"  [{i*20+1:>4}-{(i+1)*20:>4}]"
                console.print(f"{label}  [{color}]{'#' * bar_len}[/{color}] {v:.1f}%")

    console.print(
        "\n[dim]Interpretation: accuracy > 55% consistently suggests the features have "
        "predictive signal for that ticker. The learning curve shows whether the model "
        "improves as more data accumulates.[/dim]\n"
    )


@app.command(name="paper-trade")
def paper_trade_cmd(
    tickers: Optional[list[str]] = typer.Argument(None, help="Tickers to simulate (defaults to watchlist)"),
    train_start: str = typer.Option("2000-01-01", help="Start of training data (YYYY-MM-DD)"),
    train_end: str = typer.Option("2023-01-01", help="End of training / start of live trading (YYYY-MM-DD)"),
    trade_end: Optional[str] = typer.Option(None, help="End of trading period (YYYY-MM-DD, defaults to today)"),
    cash: float = typer.Option(1000.0, help="Starting virtual cash in dollars"),
    train_window: int = typer.Option(252, help="Days of history used to train each daily model"),
    n_estimators: int = typer.Option(100, help="Random Forest trees per model"),
    confidence: float = typer.Option(0.55, help="Minimum model confidence (0-1) required to buy"),
    show_trades: bool = typer.Option(True, help="Print every individual trade"),
    show_chart: bool = typer.Option(True, help="Show ASCII portfolio value chart"),
):
    """
    Train on historical data then paper-trade a virtual portfolio on recent data.

    The model trains on [train_start -> train_end], then trades day-by-day
    from train_end to today (or --trade-end). Every day it retrains on the
    latest rolling window and decides: buy, sell, or hold.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError:
        console.print("[red]scikit-learn is not installed. Run: pip install scikit-learn[/red]")
        raise typer.Exit(1)

    target = [t.upper() for t in tickers] if tickers else watchlist.load()

    console.print(f"\n[bold]Paper Trading Simulator[/bold]")
    console.print(f"  Train:  {train_start} to {train_end}")
    console.print(f"  Trade:  {train_end} to {trade_end or 'today'}")
    console.print(f"  Cash:   ${cash:,.2f} | Confidence threshold: {confidence:.0%}\n")

    for ticker in target:
        console.print(f"[bold underline]{ticker}[/bold underline]")
        console.print(f"  Training model on {train_start} to {train_end}…", end="")

        try:
            result = ml_agent.run_paper_trade(
                ticker,
                train_start=train_start,
                train_end=train_end,
                trade_end=trade_end,
                starting_cash=cash,
                train_window=train_window,
                n_estimators=n_estimators,
                min_confidence=confidence,
            )
        except ImportError as e:
            console.print(f"\n[red]{e}[/red]")
            raise typer.Exit(1)

        if result is None:
            console.print(" [yellow]not enough data, skipped[/yellow]\n")
            continue

        console.print(f" done\n")

        # ── Summary ──────────────────────────────────────────────────────────
        ml_color = "green" if result.total_return_pct >= 0 else "red"
        bh_color = "green" if result.buy_hold_return_pct >= 0 else "red"
        beat = result.total_return_pct > result.buy_hold_return_pct

        console.print(f"  Trading period : {result.trade_start} to {result.trade_end}")
        console.print(f"  Starting cash  : ${result.starting_cash:>10,.2f}")
        console.print(
            f"  ML final value : [{ml_color}]${result.final_value:>10,.2f}[/{ml_color}]  "
            f"([{ml_color}]{result.total_return_pct:+.1f}%[/{ml_color}])"
        )
        console.print(
            f"  Buy-and-hold   : [{bh_color}]${result.buy_and_hold_value:>10,.2f}[/{bh_color}]  "
            f"([{bh_color}]{result.buy_hold_return_pct:+.1f}%[/{bh_color}])"
        )

        verdict_color = "green" if beat else "red"
        verdict = "BEAT buy-and-hold" if beat else "LOST to buy-and-hold"
        diff = result.total_return_pct - result.buy_hold_return_pct
        console.print(
            f"\n  Verdict: [{verdict_color}][bold]{verdict}[/bold][/{verdict_color}] "
            f"by [{verdict_color}]{diff:+.1f}%[/{verdict_color}]"
        )
        console.print(
            f"  Trades : {result.num_trades} round-trips | "
            f"Won: {result.win_trades} / {result.num_trades}"
        )

        # ── Trade log ────────────────────────────────────────────────────────
        if show_trades and result.trades:
            console.print()
            trade_table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
            for col in ["Date", "Action", "Price", "Shares", "Cash", "Portfolio"]:
                trade_table.add_column(col, justify="right" if col not in ("Date", "Action") else "left")

            for t in result.trades:
                color = "green" if t.action == "BUY" else "red"
                trade_table.add_row(
                    t.date,
                    f"[{color}][bold]{t.action}[/bold][/{color}]",
                    f"${t.price:,.2f}",
                    f"{t.shares:.4f}" if t.action == "BUY" else "-",
                    f"${t.cash:,.2f}",
                    f"${t.portfolio_value:,.2f}",
                )
            console.print(trade_table)

        # ── ASCII chart ──────────────────────────────────────────────────────
        if show_chart and result.daily_values:
            console.print(f"\n  [dim]Portfolio value over trading period (${cash:,.0f} start):[/dim]")
            vals = [v for _, v in result.daily_values]
            dates_chart = [d for d, _ in result.daily_values]
            chart_width = 60
            chart_height = 10
            min_v = min(vals)
            max_v = max(vals)
            v_range = max_v - min_v if max_v != min_v else 1.0

            # Downsample to chart_width points
            step_c = max(1, len(vals) // chart_width)
            sampled = vals[::step_c][:chart_width]
            sampled_dates = dates_chart[::step_c][:chart_width]

            # Build grid
            grid = [[" "] * len(sampled) for _ in range(chart_height)]
            for col_i, v in enumerate(sampled):
                row_i = chart_height - 1 - int((v - min_v) / v_range * (chart_height - 1))
                row_i = max(0, min(chart_height - 1, row_i))
                color_char = "+" if v >= cash else "-"
                grid[row_i][col_i] = color_char

            for row_i, row in enumerate(grid):
                val_at_row = max_v - (row_i / (chart_height - 1)) * v_range
                label = f"  ${val_at_row:>8,.0f} |"
                line = "".join(row)
                console.print(label + line)

            # X-axis labels
            first_d = sampled_dates[0] if sampled_dates else ""
            last_d = sampled_dates[-1] if sampled_dates else ""
            console.print(f"           +{'-' * len(sampled)}")
            console.print(f"            {first_d:<20}{last_d:>20}\n")

        console.print()


@app.command(name="live-start")
def live_start_cmd(
    ticker: str = typer.Argument(..., help="Ticker symbol to start trading, e.g. AAPL"),
    train_start: str = typer.Option("2000-01-01", help="Start of training history (YYYY-MM-DD)"),
    train_end: Optional[str] = typer.Option(None, help="End of training / go-live date (YYYY-MM-DD, defaults to today)"),
    cash: float = typer.Option(1000.0, help="Starting virtual cash in dollars"),
    train_window: int = typer.Option(252, help="Rolling days used to retrain the model each day"),
    confidence: float = typer.Option(0.55, help="Minimum model confidence (0.0-1.0) required to buy"),
    n_estimators: int = typer.Option(100, help="Random Forest trees per model"),
):
    """
    Train the ML model on historical data and start a live paper-trading session.

    After this runs, use `live-update` each day to advance the simulation and
    `live-check` to view the dashboard anytime.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError:
        console.print("[red]scikit-learn required: pip install scikit-learn[/red]")
        raise typer.Exit(1)

    t = ticker.upper()
    if live_trader.LiveState.exists(t):
        console.print(f"[yellow]A session for {t} already exists.[/yellow]")
        console.print(f"  Run [bold]stockmonitor live-check {t}[/bold] to see it.")
        console.print(f"  Delete [bold]{live_trader.STATE_DIR / t}.json[/bold] to reset it.")
        raise typer.Exit()

    console.print(f"\n[bold]Starting live paper-trade session for {t}[/bold]")
    console.print(f"  Training on: {train_start} to {train_end or 'today'}")
    console.print(f"  Starting cash: ${cash:,.2f}  |  Confidence: {confidence:.0%}\n")
    console.print("  Downloading history and training model...", end="")

    try:
        state = live_trader.start(
            t,
            train_start=train_start,
            train_end=train_end,
            starting_cash=cash,
            train_window=train_window,
            n_estimators=n_estimators,
            confidence_threshold=confidence,
        )
    except (ValueError, ImportError) as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)

    action_color = "green" if state.pending_action == "BUY" else ("red" if state.pending_action == "SELL" else "dim")
    console.print(" done\n")
    console.print(f"  [green]Session saved.[/green] State file: {live_trader.STATE_DIR / (t + '.json')}\n")
    console.print(f"  [bold]Tomorrow's signal:[/bold] [{action_color}]{state.pending_action}[/{action_color}] "
                  f"(model confidence: {state.pending_proba:.1%})")
    console.print()
    console.print(f"  Run [bold]stockmonitor live-update {t}[/bold] each day to advance the simulation.")
    console.print(f"  Run [bold]stockmonitor live-check {t}[/bold] anytime to see your dashboard.\n")


@app.command(name="live-update")
def live_update_cmd(
    tickers: Optional[list[str]] = typer.Argument(None, help="Tickers to update (defaults to all active sessions)"),
):
    """
    Fetch today's price, execute yesterday's trade signal, and predict tomorrow.

    Run this once per day (after market close) to keep the simulation current.
    Safe to run multiple times -- it skips if already up to date.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError:
        console.print("[red]scikit-learn required: pip install scikit-learn[/red]")
        raise typer.Exit(1)

    targets = [t.upper() for t in tickers] if tickers else live_trader.list_sessions()
    if not targets:
        console.print("[yellow]No active sessions found. Run [bold]stockmonitor live-start TICKER[/bold] first.[/yellow]")
        raise typer.Exit()

    for t in targets:
        console.print(f"[bold]{t}[/bold] - updating...", end="")
        try:
            state, executed, price = live_trader.update(t)
        except FileNotFoundError as e:
            console.print(f"\n[red]{e}[/red]")
            continue

        if executed == "ALREADY_CURRENT":
            console.print(f" [dim]already up to date (last: {state.last_updated})[/dim]")
            continue

        exec_color = "green" if executed == "BUY" else ("red" if executed == "SELL" else "dim")
        sig_color = "green" if state.pending_action == "BUY" else ("red" if state.pending_action == "SELL" else "dim")
        pv = state.current_value(price)
        ret = (pv - state.starting_cash) / state.starting_cash * 100
        ret_color = "green" if ret >= 0 else "red"
        console.print(
            f" executed [{exec_color}]{executed}[/{exec_color}] @ ${price:,.2f} | "
            f"Portfolio: [{ret_color}]${pv:,.2f} ({ret:+.1f}%)[/{ret_color}] | "
            f"Next: [{sig_color}]{state.pending_action}[/{sig_color}] ({state.pending_proba:.1%})"
        )


@app.command(name="live-check")
def live_check_cmd(
    tickers: Optional[list[str]] = typer.Argument(None, help="Tickers to display (defaults to all active sessions)"),
    chart_height: int = typer.Option(12, help="Height of the portfolio chart in rows"),
    chart_width: int = typer.Option(70, help="Width of the portfolio chart in columns"),
    last_trades: int = typer.Option(10, help="Number of most recent trades to show (0 = all)"),
):
    """
    Dashboard: portfolio value, P&L, vs buy-and-hold, recent trades, chart.

    Read-only -- never modifies state. Run anytime to check how you are doing.
    """
    targets = [t.upper() for t in tickers] if tickers else live_trader.list_sessions()
    if not targets:
        console.print("[yellow]No active sessions. Run [bold]stockmonitor live-start TICKER[/bold] first.[/yellow]")
        raise typer.Exit()

    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text

    for t in targets:
        try:
            state, latest_price = live_trader.get_status(t)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            continue

        pv = state.current_value(latest_price)
        bh = state.bh_value(latest_price)
        total_ret = (pv - state.starting_cash) / state.starting_cash * 100
        bh_ret = (bh - state.starting_cash) / state.starting_cash * 100
        diff = total_ret - bh_ret
        days_running = (date.today() - date.fromisoformat(state.started_on)).days

        pv_color = "green" if total_ret >= 0 else "red"
        bh_color = "green" if bh_ret >= 0 else "red"
        beat_color = "green" if diff > 0 else "red"
        sig_color = "green" if state.pending_action == "BUY" else ("red" if state.pending_action == "SELL" else "yellow")

        position = f"${state.shares * latest_price:,.2f} in shares ({state.shares:.4f} sh @ ${latest_price:,.2f})" \
                   if state.shares > 0 else f"${state.cash:,.2f} in cash (no position)"

        # ── Header panel ─────────────────────────────────────────────────────
        console.rule(f"[bold] {t} Live Paper Trade Dashboard [/bold]")
        console.print()

        # Two-column summary
        left = Text()
        left.append(f"  Started       : {state.started_on}\n")
        left.append(f"  Last updated  : {state.last_updated}\n")
        left.append(f"  Running       : {days_running} days\n")
        left.append(f"  Train period  : {state.train_start} to {state.train_end}\n")
        left.append(f"  Confidence    : {state.confidence_threshold:.0%}\n")

        right = Text()
        right.append(f"  Starting cash : ${state.starting_cash:,.2f}\n")
        right.append(f"  Position      : {position}\n")
        right.append(f"  ML value now  : ", style="bold")
        right.append(f"${pv:,.2f}  ({total_ret:+.1f}%)\n", style=pv_color)
        right.append(f"  Buy-and-hold  : ", style="bold")
        right.append(f"${bh:,.2f}  ({bh_ret:+.1f}%)\n", style=bh_color)
        right.append(f"  vs B&H        : ", style="bold")
        right.append(f"{diff:+.1f}%\n", style=beat_color)

        console.print(Columns([left, right], equal=True))

        # Signal box
        sig_label = {
            "BUY":  "BUY tomorrow - model predicts price will rise",
            "SELL": "SELL tomorrow - model predicts price will fall",
            "HOLD": "HOLD - model confidence below threshold",
        }.get(state.pending_action, state.pending_action)
        console.print(
            Panel(
                f"[{sig_color}][bold]{sig_label}[/bold][/{sig_color}]"
                f"  (confidence {state.pending_proba:.1%})",
                title="[bold]Tomorrow's Signal[/bold]",
                border_style=sig_color,
                padding=(0, 2),
            )
        )

        # ── Trade log ────────────────────────────────────────────────────────
        console.print()
        trades_to_show = state.trades if last_trades == 0 else state.trades[-last_trades:]
        if trades_to_show:
            console.print(f"[bold]Recent Trades[/bold] (showing last {len(trades_to_show)} of {len(state.trades)} total)")
            t_table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
            for col in ["Date", "Action", "Price", "Shares", "Cash after", "Portfolio"]:
                t_table.add_column(col, justify="right" if col not in ("Date", "Action") else "left")

            for tr in trades_to_show:
                color = "green" if tr.action == "BUY" else "red"
                t_table.add_row(
                    tr.date,
                    f"[{color}][bold]{tr.action}[/bold][/{color}]",
                    f"${tr.price:,.2f}",
                    f"{tr.shares:.4f}" if tr.action == "BUY" else "-",
                    f"${tr.cash:,.2f}",
                    f"${tr.portfolio_value:,.2f}",
                )
            console.print(t_table)
        else:
            console.print("[dim]No trades yet.[/dim]\n")

        # ── Portfolio chart ──────────────────────────────────────────────────
        if state.daily_values:
            vals = [v for _, v in state.daily_values]
            dates_c = [d for d, _ in state.daily_values]
            # append current price
            vals.append(pv)
            dates_c.append(str(date.today()))

            min_v = min(vals)
            max_v = max(vals)
            v_range = max_v - min_v if max_v != min_v else 1.0

            step_c = max(1, len(vals) // chart_width)
            sampled_v = vals[::step_c][:chart_width]
            sampled_d = dates_c[::step_c][:chart_width]

            grid = [[" "] * len(sampled_v) for _ in range(chart_height)]
            for ci, v in enumerate(sampled_v):
                ri = chart_height - 1 - int((v - min_v) / v_range * (chart_height - 1))
                ri = max(0, min(chart_height - 1, ri))
                grid[ri][ci] = "+" if v >= state.starting_cash else "-"

            console.print(f"\n[bold]Portfolio Value Chart[/bold]  (+ above start, - below)")
            for ri, row in enumerate(grid):
                val_label = max_v - (ri / max(chart_height - 1, 1)) * v_range
                line = "".join(row)
                # colour runs of + green, - red
                rich_line = ""
                for ch in line:
                    if ch == "+":
                        rich_line += "[green]+[/green]"
                    elif ch == "-":
                        rich_line += "[red]-[/red]"
                    else:
                        rich_line += " "
                console.print(f"  ${val_label:>8,.0f} |{rich_line}")

            first_d = sampled_d[0] if sampled_d else ""
            last_d = sampled_d[-1] if sampled_d else ""
            console.print(f"           +{'-' * len(sampled_v)}")
            console.print(f"            {first_d}  ...  {last_d}\n")

        console.print()


# Register journal command under the name "journal"
app.command(name="journal")(journal_cmd)


def main() -> None:
    app()
