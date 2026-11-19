"""gta6-trophy-rarity-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%,
95%) per trophy, scored by mean pinball loss on the logit scale.

Method (count-aware logistic drift with a refresh-dilution prior)
---------------------------------------------------------------
Exophase does not refresh a game's counts continuously: the house saw the
launch-night page frozen for hours (638 players tracked, every rate the
same), then one refresh multiply the tracked players ~6x and push every
rate but the 100% one down. Two forces therefore move a trophy's rate:

  1. Progression -- players earn trophies, so the count of earners E_i(t)
     grows. On the logit scale, log-time is the natural clock: we fit, per
     trophy, a least-squares line of logit(rate) against ln(1 + hours since
     the release day began), over the recent history, and read it at the
     target. The slope is shrunk toward its tier's mean slope (empirical
     Bayes: trophies of a tier -- platinum/gold/silver/bronze -- progress
     together, and a single trophy's steppy refresh history is noisy).
  2. Dilution -- new players syncing at a refresh hold almost nothing
     beyond the intro trophy, so a refresh drags every other rate down.
     When the history is a single frozen page (no rate ever changed), the
     page is stale and a refresh is pending: the median is shifted down
     and the spread widened, with both effects decaying with the horizon
     (a refresh dominates a 90-minute horizon; over a week, progression
     dominates and the fitted trend already nets dilution against
     progression). When the history already moves, the trend fit carries
     the net effect and no extra shift is applied.

Spread: the logit standard deviation combines the baseline random walk
(0.5 per sqrt-day), the trend fit's uncertainty (slope standard error
times the extrapolation distance, plus residual scatter) and, for a stale
page, the refresh uncertainty. Quantiles are normal quantiles on the
logit scale around the median, rounded to 4 decimals.

Special cases: a trophy at ~100% (the intro trophy, earned by every
tracked player) stays there -- it gets a tiny spread and no dilution
shift. A trophy with no usable rate falls back to its tier's last median.

The program reads every <id>.in in the input directory and writes
<id>.out: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, quantiles non-decreasing, each in [0.01, 99.99].
Deterministic (same case -> same bytes), no network, no clock, no
unseeded randomness, standard library only, well under a second a case.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (fixed in advance; the dilution ones are calibrated
# on the practice refresh documented in the kit's competition.yaml, not
# on any hidden answer: at the first refresh players tracked went
# 638 -> 3,868 and every rate but the 100% one fell) ---
RW_SIGMA = 0.5          # logit random-walk sigma per sqrt-day (house baseline)
DILUTION_SHIFT = 1.5    # logit drop if a refresh lands (N ~3x at a refresh)
DILUTION_DECAY_H = 8.0  # hours over which the refresh prior fades
REFRESH_SPREAD = 0.75   # extra logit spread while a refresh is pending
SHRINK_K = 3.0          # empirical-Bayes pseudo-count for tier shrinkage
MAX_DRIFT = 2.5         # cap on |logit drift| from the last read
DRIFT_DAMP_X = 0.6      # log-time distance over which extrapolation is halved:
                        # a logit-linear trend bends as rates saturate, so
                        # far extrapolations are damped toward persistence
SPREAD_FLOOR = 0.30     # minimum logit spread (never trust a point forecast)
SPREAD_CAP = 2.2        # maximum logit spread
INTRO_SPREAD = 0.35     # tight spread for a trophy at ~100%
WINDOW_HOURS = 48.0     # trend-fit window (recent history only)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the kit's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(case, when):
    """Hours from the start of the release day (2026-11-19T00:00Z)."""
    release = case.get("game", {}).get("release_date", "2026-11-19")
    day0 = datetime.fromisoformat(release + "T00:00:00+00:00")
    return max(0.0, (when - day0).total_seconds() / 3600.0)


def log_time(hours):
    return math.log1p(hours)


def ols_slope(xs, ys, ws):
    """Weighted least-squares slope/intercept; returns (slope, resid_sd, se)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    n = len(xs)
    if n > 2:
        rss = sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        resid_var = rss / (sw * (n - 2) / n) if sw > 0 else 0.0
        resid_sd = math.sqrt(max(0.0, resid_var))
        se = math.sqrt(max(0.0, resid_var) / sxx) if sxx > 0 else 0.0
    else:
        resid_sd, se = 0.0, 0.0
    return slope, resid_sd, se


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0

    last = history[-1]
    checkpoint = parse_time(case["at"])
    x_target = log_time(hours_since_release(case, parse_time(case["target"])))
    x_last = log_time(hours_since_release(case, checkpoint))
    dx = max(0.0, x_target - x_last)

    # Staleness: how many distinct rate-vectors does the history hold? A
    # single frozen page means Exophase has not refreshed since the first
    # read and a refresh is pending.
    seen = set()
    for read in history:
        seen.add(tuple(read["rates"]))
    stale = len(seen) <= 1

    # Per-trophy series: (x, logit rate), recent window, recency-weighted.
    # Weights ramp linearly from 0.25 (oldest in window) to 1 (newest).
    cutoff = checkpoint.timestamp() - WINDOW_HOURS * 3600
    per_trophy = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        pts = []
        for read in history:
            r = read["rates"][i]
            if r is None:
                continue
            t = parse_time(read["at"])
            if t.timestamp() < cutoff:
                continue
            pts.append((log_time(hours_since_release(case, t)), logit(r),
                        t.timestamp()))
        if pts:
            t0, t1 = pts[0][2], pts[-1][2]
            span = max(1.0, t1 - t0)
            for x, y, ts in pts:
                w = 0.25 + 0.75 * (ts - t0) / span
                xs.append(x)
                ys.append(y)
                ws.append(w)
        per_trophy.append((xs, ys, ws))

    # Last usable rate per trophy (walk back over nulls), tier fallback.
    last_rate = []
    for i, trophy in enumerate(trophies):
        r = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                r = read["rates"][i]
                break
        last_rate.append(r)
    tier_last = {}
    for trophy, r in zip(trophies, last_rate):
        if r is not None:
            tier_last.setdefault(trophy["tier"], []).append(r)
    tier_median = {t: sorted(v)[len(v) // 2] for t, v in tier_last.items()}
    for i, trophy in enumerate(trophies):
        if last_rate[i] is None:
            last_rate[i] = tier_median.get(trophy["tier"], 1.0)

    # Fit slopes, then shrink each toward its tier's mean slope.
    slopes, resids, ses = [], [], []
    for xs, ys, ws in per_trophy:
        b, sd, se = ols_slope(xs, ys, ws)
        slopes.append(b)
        resids.append(sd)
        ses.append(se)
    tier_slope = {}
    for tier in {t["tier"] for t in trophies}:
        idx = [i for i, t in enumerate(trophies) if t["tier"] == tier]
        # weight by how much movement a trophy actually showed
        num = sum(slopes[i] * max(1.0, len(per_trophy[i][0])) for i in idx)
        den = sum(max(1.0, len(per_trophy[i][0])) for i in idx)
        tier_slope[tier] = num / den if den else 0.0
    shrunk = []
    for i, trophy in enumerate(trophies):
        n_distinct = len(set(per_trophy[i][1]))
        w = float(n_distinct)
        b = (w * slopes[i] + SHRINK_K * tier_slope[trophy["tier"]]) / (w + SHRINK_K)
        shrunk.append(b)

    # Refresh prior: only when the page is stale, fading with the horizon.
    decay = math.exp(-horizon_h / DILUTION_DECAY_H) if stale else 0.0
    shift = -DILUTION_SHIFT * decay
    refresh_var = (REFRESH_SPREAD * decay) ** 2
    rw_var = (RW_SIGMA * math.sqrt(horizon_d)) ** 2

    rows = []
    for i, trophy in enumerate(trophies):
        r = last_rate[i]
        if r >= 99.5:
            # Intro trophy: everybody holds it, refreshes leave it at 100%.
            lg = logit(r)
            qs = [round4(clamp_pct(inv_logit(lg + z * INTRO_SPREAD))) for z in Z]
            rows.append({"id": trophy["id"], "q": qs})
            continue
        drift = max(-MAX_DRIFT, min(MAX_DRIFT, shrunk[i] * dx))
        # Damp far extrapolations: a saturating curve bends, so a straight
        # logit-log-time line overshoots the further it is read out.
        drift *= 1.0 / (1.0 + dx / DRIFT_DAMP_X)
        med_logit = logit(r) + drift + shift
        fit_var = (ses[i] * dx) ** 2 + resids[i] ** 2
        spread = math.sqrt(rw_var + fit_var + refresh_var)
        spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread))
        qs = [round4(clamp_pct(inv_logit(med_logit + z * spread))) for z in Z]
        # enforce non-decreasing after rounding
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
