# Project Canary Model Registry

The authoritative deployed bundle is `../models/trish_v19/`. Read `TRISH_V19_INTEGRATION.md` before changing model behavior.

## Live models

| ID | Business question | Algorithm | Evidence window | App role |
|---|---|---|---|---|
| M1 | Where might the end-of-cycle recovery proxy finish? | Extra Trees | Daily through Day 14 | Recovery-proxy outlook; held afterward |
| M3 | Where might average bodyweight finish on Day 35? | CatBoost | Days 7, 14, and 21 | Bodyweight outlook; held between and afterward |

Models 2, 4, 5, and 6 from earlier research are archived and are not live prediction routes.

## Held-out performance

| Model | Overall MAE | R² | Naive improvement |
|---|---:|---:|---:|
| M1 recovery proxy | 1.58 percentage points | 0.373 | 33.7% |
| M3 Day 35 bodyweight | 122.1 g | 0.241 | 18.2% |

M1 checkpoint MAE is 1.62 percentage points at Day 7 and 1.55 at Day 14. M3 checkpoint MAE is 122.6 g at Day 7, 105.7 g at Day 14, and 105.7 g at Day 21.

These are saved leave-one-building-flock-out validation results. They are not leave-one-complete-cycle-out metrics. Other buildings from the same production cycle may remain in a fold's training data.

## Source of truth

`../models/trish_v19/manifest.json` records:

- artifact filename and SHA-256 hash;
- MLflow run ID;
- feature count;
- target definition;
- validation performance;
- refresh policy;
- pilot-stage status.

The app must read these values from the manifest instead of manually retyping them.

## Non-negotiable boundaries

- Recovery is a last-recorded-population proxy unless verified harvest reconciliation is available.
- M3 does not refresh on Day 28.
- The forecast error band is an empirical held-out error reference, not a formal confidence interval.
- Forecasts never change risk points.
- LOFO importance describes association, not causation or direction of intervention.
- A generic future-flock forecast must remain unavailable until the raw-data-to-85-feature transformer is packaged and validated.
