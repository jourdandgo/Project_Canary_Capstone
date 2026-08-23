# How Project Canary's Forecasts Work

Project Canary now evaluates **three model engines supporting two business outcomes**:

- Model 1 · Extra Trees predicts the end-of-cycle recovery proxy.
- Model 3 · XGBoost predicts Day 35 bodyweight from Day 21 evidence.
- The checkpoint historical-remaining-gain method predicts Day 35 bodyweight at Days 7, 14, 21, and 28.

The complete plain-language explanation, performance comparison, use guide, and speaker notes are in:

**[Project Canary: Three-Model Explainer](THREE_MODEL_SIMPLE_EXPLAINER.md)**

The checkpoint bodyweight method is the default app selection. Model 3 remains available through an explicitly labelled toggle. Forecasts never change the independent rules-based review-priority score.
