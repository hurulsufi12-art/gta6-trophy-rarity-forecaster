# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hurulsufi12-art/gta6-trophy-rarity-forecaster/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hurulsufi12-art/gta6-trophy-rarity-forecaster/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%,
95%) per trophy — for the lowest pinball loss on the logit scale.

## Method

`main.py` (Python 3.13, standard library only) reads each case's history of
Exophase reads and forecasts in three steps:

1. **Logistic drift.** Per trophy, a recency-weighted least-squares line of
   logit(rate) against ln(1 + hours since the release day began), fitted over
   the recent history and read at the target. Slopes are shrunk toward their
   tier's mean slope (platinum / gold / silver / bronze progress together),
   and far extrapolations are damped toward persistence, because earn-rate
   curves saturate as the player base grows.
2. **Refresh-dilution prior.** Exophase refreshes a game's counts in jumps,
   not continuously: on launch night the page sat frozen for hours, then one
   refresh multiplied tracked players ~6x and pushed every rate but the 100%
   one down. When the history is a single frozen page, a refresh is pending:
   the median is shifted down and the spread widened, with both effects
   fading as the horizon grows (over a week, the fitted trend already nets
   dilution against progression). The ~100% intro trophy is exempt.
3. **Honest spread.** The logit spread combines a random walk, the trend
   fit's uncertainty, and the refresh uncertainty, so the 5–95% band covers
   about 90% of outcomes.

## Practice results (reconstructed answers, see analysis)

On the two practice cases, scored with a scorer implemented from the rules
against answers reconstructed from the refresh aggregates documented in the
kit (players 638 → 3,868, trophies earned 943 → 4,276):

| | This entry | House persistence baseline |
|---|---|---|
| Pinball loss (logit) | **0.18** | 1.40 |
| Typical miss | 0.06 pp | 0.68 pp |
| 90% coverage | 0.98 | 0.00 |

The practice answers are a reconstruction, not the official ones — the
reconstruction reproduces the baseline's stated practice loss (~1.51),
typical miss (~0.68 pp) and coverage (0) closely enough to trust the
comparison. On synthetic final-like cases (two days of history, one week
horizon) the entry also beats persistence, narrowly (0.26–0.29 vs 0.27–0.29).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic, offline, no network, ~0.1 s per case.

## Full analysis

See `ANALISIS-RANK-HADIAH.md` in the bounty directory for the verified prize
terms, the practice-answer reconstruction, and the honest limits of the
practice validation.
