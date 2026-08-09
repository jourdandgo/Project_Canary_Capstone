# Sprint 5 Report Source Notes

- Audience: product stakeholders / farm owner.
- Delivery: self-contained portable HTML report with a Markdown companion.
- Question: Is the Project Canary capstone decision flow reproducible and ready for stakeholder demonstration?
- Assessment standard: all five required outputs, all six building states, continuous operation after Day 14, and explicit caveats.
- Main evidence: `artifacts/capstone_validation.json`, automated tests, `models/MODEL_CARD.md`, and `docs/OPEN_ITEMS.md`.
- Visual omission choice: model performance charts were omitted because the report’s decision is capstone acceptance, not candidate-model selection. Exact model metrics remain in the model card.

## Chart map

| Report section | Question | Family / type | Fields | Supported claim | Palette | Source |
|---|---|---|---|---|---|---|
| Continuous-cycle behavior | Did every representative scenario pass its acceptance checks? | Category comparison / single-series bar | scenario, checks passed, checks run | All five walkthroughs passed nine of nine checks | Single-root blue, no redundant legend | `artifacts/capstone_validation.json` |

The chart uses five meaningful categories, a zero-based magnitude comparison, direct category labels, and an adjacent explanatory paragraph. Exact scenario coverage remains available in the following table.
