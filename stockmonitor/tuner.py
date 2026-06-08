from __future__ import annotations

import json
from pathlib import Path

from .grader import compute_accuracy
from .scanner import DEFAULT_CFG

CONFIG_PATH = Path.home() / ".stockmonitor" / "config.json"

# Minimum graded predictions before a signal is eligible for tuning
MIN_SAMPLES = 10

# If a signal's accuracy drops below this, tighten its threshold
ACCURACY_FLOOR = 0.48

# If accuracy is strong, we can relax slightly
ACCURACY_CEILING = 0.65

# How much to nudge thresholds each tuning cycle
RSI_STEP = 2
VOL_STEP = 0.25


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return dict(DEFAULT_CFG)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def tune() -> tuple[dict, list[str]]:
    """
    Analyse graded history and nudge thresholds to improve accuracy.
    Returns (new_config, list_of_changes_made).
    """
    stats = compute_accuracy()
    cfg = load_config()
    changes = []

    if not stats or stats.get("overall_total", 0) < MIN_SAMPLES:
        return cfg, ["Not enough graded data yet (need at least 10 graded predictions)."]

    by_signal = stats.get("by_signal", {})

    # ── RSI oversold ──────────────────────────────────────────────────────────
    rsi_os = by_signal.get("RSI oversold", {})
    if rsi_os.get("total", 0) >= MIN_SAMPLES:
        acc = rsi_os["pct"] / 100
        current = cfg.get("rsi_oversold", 35)
        if acc < ACCURACY_FLOOR and current > 20:
            cfg["rsi_oversold"] = current - RSI_STEP  # tighten (require more oversold)
            changes.append(
                f"RSI oversold: {current} → {cfg['rsi_oversold']} "
                f"(accuracy was {rsi_os['pct']}%, tightened threshold)"
            )
        elif acc > ACCURACY_CEILING and current < 40:
            cfg["rsi_oversold"] = current + RSI_STEP  # relax
            changes.append(
                f"RSI oversold: {current} → {cfg['rsi_oversold']} "
                f"(accuracy is {rsi_os['pct']}%, relaxed threshold)"
            )

    # ── RSI overbought ────────────────────────────────────────────────────────
    rsi_ob = by_signal.get("RSI overbought", {})
    if rsi_ob.get("total", 0) >= MIN_SAMPLES:
        acc = rsi_ob["pct"] / 100
        current = cfg.get("rsi_overbought", 65)
        if acc < ACCURACY_FLOOR and current < 80:
            cfg["rsi_overbought"] = current + RSI_STEP
            changes.append(
                f"RSI overbought: {current} → {cfg['rsi_overbought']} "
                f"(accuracy was {rsi_ob['pct']}%, tightened threshold)"
            )
        elif acc > ACCURACY_CEILING and current > 60:
            cfg["rsi_overbought"] = current - RSI_STEP
            changes.append(
                f"RSI overbought: {current} → {cfg['rsi_overbought']} "
                f"(accuracy is {rsi_ob['pct']}%, relaxed threshold)"
            )

    # ── Volume spike ──────────────────────────────────────────────────────────
    vol = by_signal.get("Volume spike", {})
    if vol.get("total", 0) >= MIN_SAMPLES:
        acc = vol["pct"] / 100
        current = cfg.get("vol_spike_ratio", 2.0)
        if acc < ACCURACY_FLOOR and current < 4.0:
            cfg["vol_spike_ratio"] = round(current + VOL_STEP, 2)
            changes.append(
                f"Volume spike ratio: {current} → {cfg['vol_spike_ratio']} "
                f"(accuracy was {vol['pct']}%, raised bar)"
            )
        elif acc > ACCURACY_CEILING and current > 1.5:
            cfg["vol_spike_ratio"] = round(current - VOL_STEP, 2)
            changes.append(
                f"Volume spike ratio: {current} → {cfg['vol_spike_ratio']} "
                f"(accuracy is {vol['pct']}%, lowered bar)"
            )

    if not changes:
        changes.append("All thresholds are performing within acceptable range. No changes made.")

    save_config(cfg)
    return cfg, changes
