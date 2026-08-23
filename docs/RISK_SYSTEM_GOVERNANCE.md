# Project Canary observed-risk governance

**Rule version:** `risk-rules-0.6.0-score-bands-only`
**Status:** Proposed for farm validation — not approved for routine use

## What the score is for

Canary's 0–12 observed-concern score ranks buildings for inspection using four facts that the farm can verify directly: growth progress, cumulative population loss, latest daily mortality, and the current environmental condition. It is not a disease diagnosis, treatment instruction, causal claim, or probability of missing a harvest target. Forecast models remain separate from this score.

## Why these four dimensions

| Dimension | Operational question | Why separate |
|---|---|---|
| Weight gap | Is measured growth behind the farm's age target? | Growth can lag even when survival is stable. |
| Population loss | How much of the placed flock has already been lost? | Cumulative loss measures realized supply erosion. |
| Daily mortality | Is there an urgent current loss event? | A spike can require review before cumulative loss becomes large. |
| Environmental conditions | Is a fresh temperature or humidity observation outside its age band? | It gives management a concrete condition to inspect, not proof of cause. |

Population loss and daily mortality are shown separately because they answer different timing questions. Their points contribute to the same transparent total without any special label override.

## Calculation and labels

Each available dimension receives 0–3 points. The sum is the observed-concern total; the maximum is 12 when all four dimensions are available.

| Total | Label |
|---:|---|
| 0–2 | Low |
| 3–5 | Medium |
| 6–8 | High |
| 9–12 | Critical |

The label follows this table exactly. The same total always produces the same label, regardless of which dimensions contributed the points. This leaves **Critical** for totals of 9–12 and makes every card directly reconcilable.

## Severe conditions and problem patterns

Canary continues to show every 3/3 dimension, detected problem pattern, persistent watch, and matched inspection guide. These signals help management decide what to inspect first, but they do not override the score-band label.

## Evidence coverage

- **Four dimensions scored:** Complete evidence.
- **Three dimensions scored:** score-band label with a Reduced evidence flag.
- **Fewer than three:** score-band label with an Insufficient evidence warning.
- Missing or stale environmental readings never receive zero points silently.

## Threshold provenance and control

Weight, population-loss, and daily-mortality candidate thresholds originate in the Farm Validation Workbook. Temperature and humidity use the supplied age-specific tropical reference bands. The severity distances and label bands are still proposals. Before routine adoption, Doc Raymond should approve the rule version after the team reviews a one-to-two-cycle shadow-pilot log containing score, evidence coverage, inspection taken, outcome, and any threshold exception.

Every active-building view must display the raw observation and date, target or reference band, calculation, points, threshold source, total, score-band label, evidence status, rule version, and approval status.
