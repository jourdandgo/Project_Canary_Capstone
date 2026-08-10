# Project Canary — Teammate Model Comparison Protocol

## Required package

- The notebook that prepares data, trains candidates, and evaluates them
- The `.pkl` or `.joblib` artifact
- Any requirements or environment file
- The exact feature list or fitted preprocessing pipeline, if separate

The artifact must be a trusted file created by the teammate. A serialized model is never treated as evidence by itself.

## Fair evaluation rules

1. Reproduce the teammate’s workflow on Canary’s canonical workbook and exported training rows.
2. Use the same Y definition and the same information cutoff for every model.
3. Hold out one complete harvest cycle at a time. Do not randomly divide related building-day rows.
4. Fit imputers, scalers, encoders, and feature selection inside each training fold.
5. Reject inputs that reveal a future checkpoint weight, final population, recovery label, Day 35 label, or another row from the held-out cycle.
6. Report overall, Day 14, forecast-horizon, and cycle-by-cycle results.

## Comparison metrics

The primary metric is **cycle-macro mean absolute error (MAE)**: calculate MAE separately for every held-out cycle, then average the cycle results so large cycles do not dominate.

- Both models: MAE, RMSE, bias, fold variability, and uncertainty coverage
- Recovery: percentage-point error and the 95% target-side confusion matrix
- Weight: gram error, percentage within 200 g, and the 1,800 g target-side confusion matrix

## Champion rule

- Recovery: select the simplest reproducible method within 10% of the best cycle-macro MAE.
- Day 35 weight: select the simplest reproducible method within 5% of the best cycle-macro MAE.
- A small apparent gain is not accepted if it depends on leakage, an unreproducible pickle, or an unstable split.

## Safe intake command

Run `python -m scripts.inspect_teammate_model teammate.ipynb --model teammate.pkl`. This inventories the notebook and pickle without executing the pickle. A human must then review the workflow before any trusted loading or reproduction.

## Status

The comparison framework is ready. Champion comparison remains pending until the teammate’s files are supplied.
