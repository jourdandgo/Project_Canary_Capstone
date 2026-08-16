# Project Canary — Platform-Neutral Modeling Prompt

Read `docs/PROJECT_CANARY_EXTERNAL_MODELING_CONTEXT.md` completely before beginning. Then inspect the latest files in `data/`, `outputs/model_ready/`, and the current model manifests in `models/`.

Act as an independent senior machine-learning engineer reviewing Project Canary's two prediction problems:

1. Final harvest-recovery proxy versus the 95% goal.
2. Observed average bodyweight on Day 35 versus the 1,800 g milestone.

Develop the strongest honest, reproducible, and explainable modeling solution supported by the available data. Treat the existing Canary pipelines as benchmarks, not templates. Independently determine and justify the analytical grain, data preparation, target formulation, features, validation design, algorithms, tuning, uncertainty method, and champion-selection logic.

The context document describes the business definitions, data sources, current counts, known quality risks, and expected evidence. Preserve temporal integrity: a forecast may use only information that would have been available on its review date. Repeated daily or checkpoint rows from one building-cycle are not additional independent flocks.

For each outcome, produce:

- A fully reproducible notebook or code pipeline.
- Source and model-ready data audits.
- The final Y and X definitions with timing and leakage notes.
- EDA relevant to prediction and business interpretation.
- A fair comparison of at least five justified approaches, including a simple benchmark.
- A concise comparison table containing MAE, RMSE, R², bias, stability, and the outcome-specific metrics defined in the context document.
- Results by forecast age/checkpoint, including Day 14.
- Actual-versus-predicted, residual, stability, and error-by-age visualizations.
- Feature importance and SHAP analysis where appropriate, at both global and individual-building levels.
- Uncertainty estimates and an honest assessment of empirical reliability.
- A clearly separated later-cycle audit after the modeling approach is finalized.
- Exported champion artifacts, features, hyperparameters, predictions, metrics, and figures.
- A plain-language interpretation for the farm owner and a technical explanation suitable for capstone panelists.

Do not assume that a complex machine-learning model must win. Do not optimize for an impressive training result. Challenge the existing formulation when the evidence supports an alternative, and state clearly if the available data cannot support a reliable forecast.

At the end, provide:

1. The champion recommendation for each prediction problem.
2. A side-by-side comparison with the current Canary benchmark.
3. The five most important predictive drivers for each outcome, with direction where defensible.
4. The key limitations and leakage checks.
5. The highest-value data-collection improvements.
6. A short recommendation on whether either new model is ready to replace the current application model.

