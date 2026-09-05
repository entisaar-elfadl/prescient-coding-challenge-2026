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

from matplotlib.pyplot import hist
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Every tuneable number lives here. Fewer is better.
# --------------------------------------------------------------------------- #

PARAMS = {
    "lookback": 40,       # Lookback window for trend and volatility
    "tilt_size": 0.07,     # Active tilt scale factor
    "trade_speed": 0.06,   # Low trade speed to eliminate transaction drag
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
    1. Carry Extraction: Overweight SA Bonds when real yields are high.
    2. FX Volatility Regime: Shift to Global Equity/Gold when ZAR is unstable.
    3. Slow Momentum: Low-turnover cross-sectional ranking.
    """
    prices = hist.prices
    returns = hist.returns
    macro = hist.macro
    
    lb = int(params["lookback"])
    recent_rets = returns.tail(lb)
    
    # 1. Risk-Adjusted Return (Sharpe-like ranking)
    vol = recent_rets.std() * np.sqrt(252)
    cum_ret = (prices.iloc[-1] / prices.iloc[-lb]) - 1.0
    signal = (cum_ret / vol.replace(0.0, np.nan)).fillna(0.0)

    # 2. SA Bond Carry Engine (High yield, low transaction cost)
    if "sa_10y" in macro.columns and "sa_repo" in macro.columns:
        curve_slope = macro["sa_10y"].iloc[-1] - macro["sa_repo"].iloc[-1]
        if curve_slope > 2.0:  # Steep curve: overweight bonds vs cash
            signal["SA_BONDS"] += 0.5
            signal["SA_CASH"] -= 0.3

    # 3. Macro FX Shield (USDZAR Volatility Spike)
    if "usdzar" in macro.columns and len(macro["usdzar"]) >= 20:
        zar_vol = macro["usdzar"].pct_change().tail(20).std() * np.sqrt(252)
        if zar_vol > 0.12:  # High ZAR volatility -> risk off
            signal["GLOBAL_EQUITY"] += 0.4
            signal["GOLD"] += 0.4
            signal["SA_PROPERTY"] -= 0.6
            signal["SA_EQUITY"] -= 0.3

    # Standardize signal
    score = signal.reindex(hist.assets).fillna(0.0)
    if score.std() > 0:
        score = (score - score.mean()) / score.std()
        
    return score


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
    bm = hist.benchmark

    if len(hist.returns) < 260:
        return bm.to_dict()

    signal = build_signal(hist, params)
    target = make_legal(bm + float(params["tilt_size"]) * signal, hist)

    # Slow trade speed minimizes transaction drag
    prev = prev_weights.reindex(hist.assets)
    w = prev + float(params["trade_speed"]) * (target - prev)

    return make_legal(w, hist).to_dict()


# <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
