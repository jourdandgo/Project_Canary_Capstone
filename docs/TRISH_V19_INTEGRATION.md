# Trish v19 Dashboard Integration

## Live scope

Project Canary uses two final handoff models:

| Model | Outcome | Algorithm | Refresh |
|---|---|---|---|
| Model 1 | End-of-cycle recovery proxy | Extra Trees | Daily through Day 14; held afterward |
| Model 3 | Day 35 bodyweight | CatBoost | Days 7, 14, and 21; held between and afterward |

Models 2, 4, 5, and 6 are not part of live v19 inference or owner-facing output.

## Source-backed replay rule

The final serialized artifacts were fitted on all 34 building-flocks. Historical and 2026-3 app screens therefore display the saved leave-one-building-flock-out prediction for the selected building/day instead of rescoring that same flock with the all-data fitted artifact.

The replay join is:

`harvest_cycle + building + eligible evidence day → exact model-ready row + saved held-out prediction`

Model 1 selects the latest daily row through Day 14. Model 3 selects Day 7, 14, or 21 and holds the latest eligible checkpoint. A current record without a measured checkpoint weight does not receive Model 3.

## Traceability contract

Every forecast detail shows:

1. model and target definition;
2. evidence day and held/recalculated status;
3. exact 85-feature model-ready row;
4. algorithm, artifact bundle, and MLflow run ID;
5. saved held-out prediction;
6. checkpoint MAE and empirical 80%-error band;
7. recorded outcome for completed replay cycles;
8. global LOFO association evidence and its non-causal boundary.

Every observed-risk detail separately shows raw observations, calculation, threshold, threshold source, points, total equation, label band or override, rule version, approval state, and evidence status.

Forecasts never add or remove risk points.

## Integrity files

The authoritative runtime bundle is `models/trish_v19/`. Its `manifest.json` records artifact hashes, MLflow run IDs, model metrics, target definitions, and refresh policies. The app checks that each artifact hash matches the manifest and that its feature and held-out-prediction files exist.

## Current limitation

The handoff does not include the complete raw-data-to-85-feature transformer as a deployable function. The current app is therefore a pilot-stage source-backed replay. It must display forecast unavailable for an arbitrary future flock rather than silently route to an unrelated fallback.
