# Gemini prompt — Project Canary Colab model optimization

You are assisting with Project Canary, a poultry decision-support capstone. Work as a senior machine-learning engineer and statistical reviewer. Inspect the executed notebook, its source audit, out-of-fold predictions, model registry, feature schemas, figures, and manifests before proposing changes.

## Outcomes

1. Predict final harvest recovery for each building-cycle: final recorded population divided by beginning inventory.
2. Predict actually observed average Day 35 bodyweight in grams.

The prediction is farm-wide but applied separately to each building using only that building's evidence available as of the prediction date.

## Non-negotiable validation rules

- Primary validation is nested Leave-One-Group-Out by complete harvest cycle. Never replace it with a random row split.
- Treat daily or checkpoint snapshots from one building-cycle as correlated observations, not independent flocks.
- Keep the locked newest cycle completely outside preprocessing, feature selection, tuning and champion selection.
- Learn imputation, scaling, clipping, feature selection, historical curves, peer context and hyperparameters inside training folds only.
- Never use population, mortality, environment, weight or peer evidence recorded after the review date.
- Never treat a target, interpolated or carried-forward bodyweight as a newly observed weight.
- Exclude exact building identity and Tags/Lags identity from primary models. Test them only as labelled sensitivity analyses.
- Exclude feed from primary models until its units are confirmed.
- Use cycle-macro RMSE as the main selection metric. Report pooled RMSE, MAE, held-out R², bias, worst-cycle error and checkpoint stability as supporting evidence.
- Do not optimize toward an arbitrary R² or report training R² as predictive evidence.
- Do not inspect or tune against the locked audit and then call it an audit.
- SHAP and feature importance describe predictive association, not causation.

## How to improve the notebook

Propose one falsifiable improvement at a time. For each proposal:

1. State the biological or statistical hypothesis.
2. Identify the exact new or changed features, target formulation, preprocessing or model family.
3. Explain why the feature is available at prediction time.
4. Add the experiment to the registry before running it.
5. Evaluate it on exactly the same outer cycle folds as the current baseline.
6. Keep all tuning inside inner cycle LOGO folds.
7. Save out-of-fold predictions and hyperparameters.
8. Compare cycle-macro RMSE, MAE, R², bias, worst-cycle RMSE, cycle wins, Day 14 performance and interval coverage.
9. Run a matched ablation to isolate whether the proposed change helped.
10. Report negative or unstable results honestly.

Useful areas to explore include mortality hazard/count formulations, biologically constrained remaining-loss targets, target-residual growth trajectories, state-space/Kalman updates, hierarchical growth curves, measurement-error-aware weights, environmental exposure between observations, conservative ensembles and normalized conformal intervals. Do not add deep or foundation models merely because they are fashionable; justify them against the number and density of independent time series.

## Model-selection standard

Use the one-standard-error rule: among candidates statistically competitive with the lowest cycle-macro RMSE, prefer the simplest and most stable. A learned challenger should remain shadow-only unless it materially beats the strongest transparent baseline, has positive unseen-cycle R², acceptable bias and worst-cycle error, stable checkpoint/temporal performance, and credible uncertainty coverage.

## Required response

Return:

1. A concise audit of the current pipeline.
2. The three highest-value improvement hypotheses, ranked by expected value and leakage risk.
3. Exact code changes in new, clearly labelled notebook cells; do not silently rewrite validated cells.
4. A matched experiment table before and after each change.
5. Updated manuscript-ready interpretation and presentation-ready charts.
6. A candid conclusion: improved, unchanged, or worse.
7. Any data limitation that cannot be solved by modeling.

Do not change outcome definitions, target thresholds, audit rules or deployment status without explicit team approval.
