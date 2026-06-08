from __future__ import annotations

from .scanner import Signal


def _describe_change(chg: float) -> str:
    if chg <= -15:
        return f"got absolutely crushed, dropping {chg:.1f}% — something big happened"
    if chg <= -5:
        return f"had a rough day, falling {chg:.1f}%"
    if chg <= -2:
        return f"slipped {chg:.1f}% today"
    if chg < 0:
        return f"dipped a little ({chg:.1f}%)"
    if chg == 0:
        return "barely moved today"
    if chg < 2:
        return f"nudged up a tiny bit ({chg:+.1f}%)"
    if chg < 5:
        return f"had a decent day, rising {chg:+.1f}%"
    if chg < 15:
        return f"surged {chg:+.1f}% today"
    return f"exploded upward {chg:+.1f}% — something big happened"


def _describe_rsi(rsi: float) -> str:
    if rsi <= 25:
        return (
            f"RSI is {rsi:.0f} — extremely oversold. "
            "The stock has been beaten down so hard it's historically rare. "
            "Think of it like a spring compressed all the way — big bounce potential."
        )
    if rsi <= 35:
        return (
            f"RSI is {rsi:.0f} — oversold. "
            "The stock is on sale compared to where it's been. "
            "Like a ball bouncing near the floor."
        )
    if rsi >= 75:
        return (
            f"RSI is {rsi:.0f} — extremely overbought. "
            "Everyone and their grandma has already bought in. "
            "Could be running out of buyers — a pullback wouldn't be surprising."
        )
    if rsi >= 65:
        return (
            f"RSI is {rsi:.0f} — overbought. "
            "The stock has run up a lot lately. Getting stretched."
        )
    return f"RSI is {rsi:.0f} — in normal territory, no extreme reading."


def _describe_macd(cross: str) -> str | None:
    if cross == "bullish":
        return "MACD just flipped bullish — momentum is shifting upward. Buyers are taking control."
    if cross == "bearish":
        return "MACD just flipped bearish — momentum is shifting downward. Sellers are taking control."
    return None


def _describe_bb(bb: str) -> str | None:
    if bb == "squeeze":
        return (
            "Bollinger Bands are squeezing — volatility is coiling up like a compressed spring. "
            "A big move is coming, we just don't know which direction yet."
        )
    if bb == "upper_breakout":
        return "Price broke above the upper Bollinger Band — it's moving faster than normal to the upside."
    if bb == "lower_breakout":
        return "Price broke below the lower Bollinger Band — it's falling faster than normal, often a sign of panic."
    return None


def _describe_volume(vol: float) -> str | None:
    if vol >= 3:
        return f"Volume is {vol:.1f}x the normal level — massive unusual interest. People are paying attention."
    if vol >= 2:
        return f"Volume is {vol:.1f}x the normal level — notably more trading activity than usual."
    return None


def _describe_atr(atr_pct: float) -> str | None:
    if atr_pct >= 10:
        return (
            f"ATR is {atr_pct:.1f}% — this thing is a rollercoaster. "
            "It can swing double digits in a single day. Very high risk."
        )
    if atr_pct >= 3:
        return f"ATR is {atr_pct:.1f}% — elevated volatility. Expect big swings."
    return None


def _describe_score(score: int) -> str:
    if score >= 5:
        return "Overall: strong bullish setup. Multiple signals lining up to the upside."
    if score >= 3:
        return "Overall: moderate bullish lean. More reasons to expect a bounce than a continued drop."
    if score > 0:
        return "Overall: slight bullish tilt, but nothing screaming."
    if score <= -5:
        return "Overall: strong bearish setup. Multiple signals pointing down."
    if score <= -3:
        return "Overall: moderate bearish lean. Sellers appear to be in control."
    if score < 0:
        return "Overall: slight bearish tilt, but nothing definitive."
    return "Overall: neutral — signals are mixed or absent. No clear edge either way."


def explain(sig: Signal) -> str:
    lines = [f"[bold cyan]{sig.ticker}[/bold cyan] {_describe_change(sig.change_pct)}."]

    rsi_line = _describe_rsi(sig.rsi_val)
    lines.append(f"  • {rsi_line}")

    macd_line = _describe_macd(sig.macd_cross)
    if macd_line:
        lines.append(f"  • {macd_line}")

    bb_line = _describe_bb(sig.bb_position)
    if bb_line:
        lines.append(f"  • {bb_line}")

    vol_line = _describe_volume(sig.vol_spike)
    if vol_line:
        lines.append(f"  • {vol_line}")

    atr_line = _describe_atr(sig.atr_pct)
    if atr_line:
        lines.append(f"  • {atr_line}")

    lines.append(f"  → [bold]{_describe_score(sig.score)}[/bold]")
    return "\n".join(lines)


def explain_all(signals: list[Signal]) -> str:
    sections = [explain(s) for s in signals]

    # Big picture summary
    bullish = [s for s in signals if s.score >= 3]
    bearish = [s for s in signals if s.score <= -3]
    squeezing = [s for s in signals if s.bb_position == "squeeze"]

    summary_parts = []
    if bullish:
        names = ", ".join(s.ticker for s in bullish)
        summary_parts.append(f"[green]Strong bullish setups: {names}.[/green]")
    if bearish:
        names = ", ".join(s.ticker for s in bearish)
        summary_parts.append(f"[red]Strong bearish setups: {names}.[/red]")
    if squeezing:
        names = ", ".join(s.ticker for s in squeezing)
        summary_parts.append(
            f"[yellow]{names} {'are' if len(squeezing) > 1 else 'is'} squeezing — "
            "expect a big move soon, direction TBD.[/yellow]"
        )

    result = "\n\n".join(sections)
    if summary_parts:
        result += "\n\n[bold]Big picture:[/bold] " + " ".join(summary_parts)
    return result
