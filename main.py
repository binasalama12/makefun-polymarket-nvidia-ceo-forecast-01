#!/usr/bin/env python3
"""Forecaster for Polymarket's "Next CEO of NVIDIA?" event (make.fun
polymarket-nvidia-ceo-forecast-01).

Reads one case from the input directory: every market of the event with
its Yes price at each minute up to the checkpoint. Writes, for each
market and each of the 240 minutes after the checkpoint, five quantiles
(5%, 25%, 50%, 75%, 95%) of that minute's price.

Model (per market, everything estimated from the case itself):

- The median is persistence: the last observed price. These markets
  move in steps and level shifts are close to unpredictable, so the
  last price is the honest centre.
- The spread blends two views of the market's own history:
  1. Empirical quantiles of h-minute price changes, computed directly
     at anchor horizons (1, 5, 15, 30, 60, 120, 240 minutes) over the
     recent window and interpolated in sqrt(h). This captures the
     heavy tails, the mass of zero moves, and the mean reversion of
     one-minute spikes, none of which a Gaussian sees.
  2. A Gaussian random walk whose per-minute sigma comes from the
     market's mean absolute one-minute change (robust to the rare
     huge prints that wreck a plain standard deviation), floored and
     capped so quiet and wild markets both stay sane.
- A volatility-regime factor compares the last 6 hours of absolute
  movement with the whole history and widens or narrows the spread
  when the market has clearly heated up or gone quiet.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. The program is deterministic, uses only the standard
library and runs in well under the 8 second per-case limit.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

BASELINE_SIGMA = 0.0025
WINDOW = 1440            # minutes of recent history for vol estimates
ANCHORS = [1, 5, 15, 30, 60, 120, 240]
EMP_WEIGHT = 0.5         # blend weight on the empirical quantiles
SIGMA_FLOOR = 0.0012
SIGMA_CAP = 0.004
MIN_EMP_HISTORY = 300    # below this, fall back to the Gaussian only


def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def price6(v):
    return round(clamp01(v) * 1e6) / 1e6 + 0.0


def quantile_sorted(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def mean_abs_diff(prices):
    if len(prices) < 2:
        return 0.0
    total = 0.0
    for i in range(len(prices) - 1):
        total += abs(prices[i + 1] - prices[i])
    return total / (len(prices) - 1)


def anchor_quantiles(hist):
    """Empirical quantiles of h-step changes at each anchor horizon."""
    out = {}
    n = len(hist)
    for h in ANCHORS:
        if h >= n:
            continue
        changes = sorted(hist[i + h] - hist[i] for i in range(n - h))
        out[h] = [quantile_sorted(changes, tau) for tau in QUANTILES]
    return out


def empirical_row(anchors_q, h):
    """Interpolated empirical change quantiles for horizon h."""
    keys = sorted(anchors_q)
    if not keys:
        return None
    if h <= keys[0]:
        return anchors_q[keys[0]]
    if h >= keys[-1]:
        return anchors_q[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= h <= hi:
            f = (math.sqrt(h) - math.sqrt(lo)) / (math.sqrt(hi) - math.sqrt(lo))
            return [anchors_q[lo][j] * (1.0 - f) + anchors_q[hi][j] * f
                    for j in range(len(QUANTILES))]
    return anchors_q[keys[-1]]


def market_sigma(prices):
    window = prices[-WINDOW:] if len(prices) > WINDOW else prices
    mad = mean_abs_diff(window)
    return min(max(1.2533 * mad, SIGMA_FLOOR), SIGMA_CAP)


def regime_scale(prices):
    """Recent 6 h of movement vs the whole history: >1 widens the
    spread when the market has heated up, <1 narrows it when quiet."""
    if len(prices) <= 720:
        return 1.0
    overall = mean_abs_diff(prices)
    if overall <= 1e-9:
        return 1.0
    recent = mean_abs_diff(prices[-361:])
    ratio = recent / overall
    return min(1.6, max(0.7, ratio))


def forecast_market(prices, horizon, sigma):
    x = prices[-1]
    window = prices[-WINDOW:] if len(prices) > WINDOW else list(prices)
    anchors_q = anchor_quantiles(window) if len(window) >= MIN_EMP_HISTORY else {}
    scale = regime_scale(prices)

    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        gauss = [x + z * sigma * sq for z in Z]
        emp = empirical_row(anchors_q, h)
        if emp is None:
            row = gauss
        else:
            row = [EMP_WEIGHT * (x + emp[j]) + (1.0 - EMP_WEIGHT) * gauss[j]
                   for j in range(len(QUANTILES))]
        row = sorted(price6(x + (v - x) * scale) for v in row)
        rows.append(row)
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text())
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"{path.name} asks for quantiles "
                             f"{case['quantiles']}, not {QUANTILES}")
        horizon = int(case["horizon_minutes"])
        parsed = []
        for market in case["markets"]:
            prices = [float(p) for p in market["prices"]]
            if not prices:
                raise ValueError(f"market {market['id']} has no prices")
            parsed.append((market["id"], prices))
        # A market with too little history of its own (it listed just
        # before the checkpoint) borrows the typical sigma of the
        # case's established markets instead of a wild guess.
        sigmas = {mid: market_sigma(prices) for mid, prices in parsed
                  if len(prices) >= MIN_EMP_HISTORY}
        if sigmas:
            ordered = sorted(sigmas.values())
            fallback_sigma = ordered[len(ordered) // 2]
        else:
            fallback_sigma = BASELINE_SIGMA
        forecasts = []
        for mid, prices in parsed:
            sigma = sigmas.get(mid, fallback_sigma)
            forecasts.append({"id": mid,
                              "q": forecast_market(prices, horizon, sigma)})
        out = json.dumps({"forecasts": forecasts}) + "\n"
        (OUTPUT_DIR / f"{case_id}.out").write_text(out)


if __name__ == "__main__":
    main()
