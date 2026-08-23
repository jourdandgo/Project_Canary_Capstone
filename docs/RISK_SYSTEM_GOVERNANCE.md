# Project Canary observed-risk governance

**Rule version:** `risk-rules-0.5.0-banded-hybrid`
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

Population loss and daily mortality share a **survivability** domain for override logic. They are shown separately because they answer different timing questions, but they cannot be counted as two independent severe domains.

## Calculation and labels

Each available dimension receives 0–3 points. The sum is the observed-concern total; the maximum is 12 when all four dimensions are available.

| Base total | Base label |
|---:|---|
| 0–2 | Low |
| 3–5 | Medium |
| 6–8 | High |
| 9–12 | Critical |

The previous `Critical 6–12` band was retired because it made the critical label too broad. The base bands are even three-point intervals, which makes the total easy to explain and leaves **Critical** for substantially accumulated concern.

## Safeguard overrides

The base label can be elevated only when a documented rule applies:

1. **Critical:** daily mortality is 3/3.
2. **Critical:** population loss is 3/3.
3. **Critical:** two distinct domains each reach 3/3. The domains are growth, survivability, and environment; two survivability components alone do not satisfy this rule.
4. **High floor:** any other single dimension reaches 3/3 but the total would otherwise be Low or Medium.

These overrides ensure an acute, independently serious condition is not hidden by a low total while avoiding accidental double counting.

## Evidence coverage

- **Four dimensions scored:** Complete evidence.
- **Three dimensions scored:** normal label with a Reduced evidence flag.
- **Fewer than three:** Insufficient evidence, unless an acute population-loss or daily-mortality override applies.
- Missing or stale environmental readings never receive zero points silently.

## Threshold provenance and control

Weight, population-loss, and daily-mortality candidate thresholds originate in the Farm Validation Workbook. Temperature and humidity use the supplied age-specific tropical reference bands. The severity distances, label bands, and overrides are still proposals. Before routine adoption, Doc Raymond should approve the rule version after the team reviews a one-to-two-cycle shadow-pilot log containing score, evidence coverage, inspection taken, outcome, and any threshold exception.

Every active-building view must display the raw observation and date, target or reference band, calculation, points, threshold source, total, evidence status, priority-rule identifier, rule version, and approval status.
