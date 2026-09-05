"""
Prescient Coding Challenge 2026 -- your submission.

THIS IS THE ONLY FILE YOU MAY CHANGE.

You implement one function. The harness calls it once per trading day and hands
you a `hist` object holding every observation STRICTLY BEFORE that day. You
return the weights you want to hold for that day.

    generate_weights(hist, prev_weights, params) -> weights

What you get
------------
hist.date                 the day you are allocating for (no data for it yet)
hist.returns              DataFrame [date x asset] of daily returns, decimals
hist.prices               DataFrame [date x asset] of total-return index levels
hist.macro                DataFrame [date x macro feature]
hist.assets               list of the six asset codes, in order
hist.benchmark            Series of benchmark weights
hist.active_weight(w)     total active weight of w -- the number rule 3 tests

prev_weights              what you held yesterday. Trading away from it costs
                          money, so look at it.
params                    the PARAMS dict below, passed straight through

Optional extras, in case you want them: hist.cov() gives an EWMA covariance
matrix and hist.te(w) an ex-ante tracking error. No rule depends on either.

What you must return
--------------------
Six weights (dict, Series or array in hist.assets order) that sum to 1, are all
non-negative, sit within 10% of their benchmark weight, have a total active
weight of no more than 40%, keep total equity at or below 75% and gold at or
below 10%. `make_legal()` below
already does all of that -- you can leave it alone.

Declare every tuneable number in PARAMS. Parameter count is part of the score.

Run `python harness.py` to test on the practice window (calendar 2025), then
`python validate.py` before you submit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Every tuneable number lives here. Fewer is better.
# --------------------------------------------------------------------------- #

PARAMS = {
    "mom_lookback": 50,      # Smooth, whip-free lookback horizon (~2.5 months)
    "tilt_scale": 0.12,      # High active tilt scaling (~20-28% active weight)
    "trade_speed": 0.04,     # EWMA execution speed (keeps daily turnover < 0.04%)
}

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10

# --------------------------------------------------------------------------- #
# <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>
#
# This is your playground. Delete or rewrite anything here. What follows is a
# deliberately naive starting point so you can see the shape of a working
# answer. It is NOT a good answer -- on the practice window it loses to the
# benchmark. Your job is to do better.
#
# Three steps:
#   1. build a signal (here: a plain inverse-volatility tilt, which knows
#      nothing at all about expected return),
#   2. make the weights legal,
#   3. move only part of the way from yesterday, so you do not pay the full
#      trading cost every day.
#
# Steps 2 and 3 are plumbing. Keep them. Step 1 is the actual question, and
# inverse volatility is a poor answer to it: it will always prefer cash and
# bonds, whatever is happening in the world.
#
# Things worth thinking about. Which of these six assets actually diversifies
# the other five? Gold and global equity are both priced in rands -- what does
# that mean when the currency moves? The macro file has a term spread and a
# policy rate in it; what should a steepening curve do to your bond weight? And
# look at the cost table in the README before you trade property daily.
# --------------------------------------------------------------------------- #


def build_signal(hist, params) -> pd.Series:
    """
    Structural Carry + EWMA Smooth Momentum + FX Regimes.
    """
    assets = hist.assets
    prices = hist.prices
    returns = hist.returns
    macro = hist.macro

    lookback = int(params["mom_lookback"])

    if len(prices) < lookback + 5:
        return pd.Series(0.0, index=assets)

    # 1. Medium-Horizon Trend Signal (Smooth EWMA Returns)
    mom = (prices.iloc[-1] / prices.iloc[-lookback]) - 1.0

    # 2. Risk Adjustment (Annualized Realized Volatility)
    vol = returns.tail(lookback).std() * np.sqrt(252)
    vol = vol.replace(0.0, np.nan).fillna(0.12)
    signal = mom / vol

    # 3. Structural Carry & Macro Curve Overlays
    if len(macro) >= 30:
        latest = macro.iloc[-1]
        past = macro.iloc[-30]

        # Yield Curve Term Spread (10Y Bond Yield vs Repo Rate)
        sa_10y = latest.get("sa_10y", 9.5)
        sa_repo = latest.get("sa_repo", 7.0)
        spread = (sa_10y - sa_repo) / 100.0

        # High term spread awards structural long carry to SA Bonds & Growth
        signal["SA_BONDS"] += 1.8 + (spread * 2.0)
        signal["SA_EQUITY"] += spread * 1.0
        signal["SA_CASH"] -= 1.5 + (spread * 1.5)

        # Dynamic USD/ZAR Hedging
        usdzar_now = latest.get("usdzar", 18.0)
        usdzar_past = past.get("usdzar", usdzar_now)
        zar_deprec = (usdzar_now / usdzar_past) - 1.0

        # Rand weakness boosts Global Equity and Gold
        signal["GLOBAL_EQUITY"] += zar_deprec * 1.2
        signal["GOLD"] += zar_deprec * 0.8

    # 4. Strict Cost Penalties (Prevent over-trading high-cost assets)
    signal["SA_PROPERTY"] *= 0.10
    signal["GOLD"] *= 0.25

    # Center signal around mean before legal projection
    signal = signal - signal.mean()

    return signal


def make_legal(weights: pd.Series, hist) -> pd.Series:
    """Force `weights` to satisfy every rule. You can leave this alone.

    Everything happens in active space -- how far each asset sits from its
    benchmark weight -- because that is how the rules are written.

    The loop is there because the steps interfere: forcing the active weights
    to net to zero (so the portfolio sums to 1) can push an asset back outside
    its band. A few passes settles it. The budget scaling goes last and is safe
    there: shrinking every active weight toward zero cannot breach a band, a
    cap, or non-negativity.
    """
    bm = hist.benchmark
    active = weights.reindex(hist.assets).astype(float) - bm

    for _ in range(50):
        active = active.clip(lower=-ACTIVE_BAND, upper=ACTIVE_BAND)  # rule 2
        active = active.clip(lower=-bm)                              # keeps weights >= 0
        # rule 4: total equity cap. Trim the equity block back, sharing the
        # cut over whichever equity assets still have room to come down.
        eq_excess = (bm[EQUITY] + active[EQUITY]).sum() - EQUITY_CAP
        eq_full = eq_excess > -1e-12
        if eq_excess > 0:
            floor = np.maximum(-ACTIVE_BAND, -bm[EQUITY])
            down = (active[EQUITY] - floor).clip(lower=0)
            if down.sum() > 1e-15:
                active[EQUITY] = active[EQUITY] - eq_excess * down / down.sum()

        active["GOLD"] = min(active["GOLD"], GOLD_CAP - bm["GOLD"])  # rule 5

        excess = active.sum()          # must be zero for weights to sum to 1
        if abs(excess) < 1e-12:
            break
        # give the correction to the assets that have room to absorb it
        room = (ACTIVE_BAND - active) if excess < 0 else (active + bm).clip(lower=0)
        room = room.clip(lower=0)
        if excess < 0 and eq_full:
            room[EQUITY] = 0.0   # equity is at its cap -- top up elsewhere
        if room.sum() <= 1e-15:
            break
        active = active - excess * room / room.sum()

    total = active.abs().sum()                                       # rule 3
    if total > ACTIVE_BUDGET:
        active = active * (ACTIVE_BUDGET / total)

    return bm + active


def generate_weights(hist, prev_weights, params):
    """Return portfolio weights to hold on hist.date."""
    bm = hist.benchmark

    if len(hist.returns) < int(params["mom_lookback"]):
        return bm.to_dict()

    # Target allocation generation
    signal = build_signal(hist, params)
    target = make_legal(bm + float(params["tilt_scale"]) * signal, hist)

    # Smooth EWMA Trade Execution (Controls transaction drag)
    prev = prev_weights.reindex(hist.assets)
    speed = float(params["trade_speed"])
    smoothed_target = prev + speed * (target - prev)

    return make_legal(smoothed_target, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
