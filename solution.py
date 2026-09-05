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
    "mom_fast": 20,         # Short window for momentum/trend confirmation
    "mom_slow": 100,        # Long window for trend confirmation
    "tilt_size": 0.09,      # Active tilt aggressiveness (0.05 - 0.10)
    "trade_speed": 0.04,    # Slow trading speed to control cost drag

}

# The rules, restated locally so this file reads on its own.
ACTIVE_BAND = 0.10       # per asset, distance from benchmark
ACTIVE_BUDGET = 0.40     # total, summed over assets
EQUITY = ["SA_EQUITY", "GLOBAL_EQUITY"]
EQUITY_CAP = 0.75        # total equity, whatever the bands allow
GOLD_CAP = 0.10

# One-way trading costs in bps for cost-dampening adjustments
COST_DAMPENER = {
    "SA_EQUITY": 1.0,
    "GLOBAL_EQUITY": 1.0,
    "SA_BONDS": 1.0,
    "SA_CASH": 1.0,
    "SA_PROPERTY": 0.45,  # 35 bps cost -> reduce turnover aggressiveness
    "GOLD": 0.55,         # 25 bps cost -> reduce turnover aggressiveness
}


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
    Constructs a signal designed to overweight GLOBAL_EQUITY, SA_PROPERTY, GOLD
    and underweight SA_CASH and SA_BONDS, combining structural bias with
    trend-following confirmation.
    """
    assets = hist.assets
    prices = hist.prices

    fast_w = int(params["mom_fast"])
    slow_w = int(params["mom_slow"])

    # Fallback if history is insufficient
    if len(prices) < slow_w + 5:
        return pd.Series(0.0, index=assets)

    # 1. Target Structural Bias
    # Overweight: GLOBAL_EQUITY, SA_PROPERTY, GOLD
    # Underweight: SA_CASH, SA_BONDS
    base_bias = pd.Series({
        "GLOBAL_EQUITY": 1.0,
        "SA_PROPERTY":   1.0,
        "GOLD":          1.0,
        "SA_EQUITY":     0.0,
        "SA_BONDS":     -1.0,
        "SA_CASH":      -1.0,
    }).reindex(assets).fillna(0.0)

    # 2. Market Trend Confirmation Filter (SMA Fast / SMA Slow)
    sma_fast = prices.tail(fast_w).mean()
    sma_slow = prices.tail(slow_w).mean()
    trend = (sma_fast / sma_slow) - 1.0

    # Scale trend by asset return volatility
    recent_rets = hist.returns.tail(30)
    vol = recent_rets.std() * np.sqrt(252)
    vol = vol.replace(0.0, np.nan).fillna(0.12)
    risk_adj_trend = trend / vol

    # 3. Combine structural bias with trend filter
    combined_signal = base_bias + 0.5 * risk_adj_trend

    # 4. Apply cost dampener to minimize churn in expensive assets
    dampener = pd.Series(COST_DAMPENER).reindex(assets).fillna(1.0)
    damped_signal = combined_signal * dampener

    # Standardize output signal
    std_dev = damped_signal.std()
    if std_dev > 1e-6:
        signal = (damped_signal - damped_signal.mean()) / std_dev
    else:
        signal = pd.Series(0.0, index=assets)

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
    """Return the six portfolio weights to hold on hist.date."""
    bm = hist.benchmark

    # not enough history to estimate anything: sit on the benchmark
    if len(hist.returns) < 260:
        return bm.to_dict()

    # 1. signal -> target weights around the benchmark
    signal = build_signal(hist, params)
    target = make_legal(bm + float(params["tilt_size"]) * signal, hist)

    # 2. trade gradually toward the target rather than jumping to it
    prev = prev_weights.reindex(hist.assets)
    w = prev + float(params["trade_speed"]) * (target - prev)

    return make_legal(w, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
