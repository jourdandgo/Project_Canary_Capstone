# Project Canary — Open Items and Decisions to Revisit

This is the durable follow-up list for the capstone. Update the **Status**, **Owner**, and **Decision / evidence** columns whenever an item is resolved. Items marked “Prototype can proceed” do not block a capstone demonstration, but their limitation must remain visible.

## Decisions already locked

| Decision | Agreed position |
|---|---|
| Final harvest recovery | Survived birds / beginning population |
| Harvest-recovery goal | At least 95% |
| Primary liveweight goal | At least 2.0 kg on Day 35; birds may continue growing after Day 35 |
| Operating window | Day 1 until the building is actually harvested; Days 1–14 retain special early-warning importance |
| Risk logic | Rules-based and independent of model forecasts |
| Predicted outcomes | Separate harvest-recovery forecast and Day 35 average-weight projection; a naïve model is acceptable when it validates best |
| Primary source | `FARM HARVEST DATA.xlsx` |
| Action logic | Deterministic mapping from identified problem pattern and severity to an owner-approved action |
| Day 14 weight target | 400 grams; use the same weighing method across buildings |
| Workbook End Date | Maximum date in the daily records; not confirmed harvest or completion date |
| Historical display convention | Every cycle before the latest placement cycle is treated as completed; each building's maximum daily-record date is displayed as its capstone completion date |
| Recovery accounting scope | For the capstone, no separate adjustment for transfers, culls, partial harvests, or missing birds |

## Capstone decisions still open

| ID | Priority | Item to resolve | Why it matters | Current safe prototype behavior | Owner | Status |
|---|---|---|---|---|---|---|
| OPEN-001 | Before farm-policy claim | Approve or revise the four risk-dimension thresholds and Low/Medium/High/Critical score bands. | The calculations work, but the cutoffs are not yet confirmed farm policy. | Every threshold and score is traceable; the interface labels version `risk-rules-0.2.0-provisional` as provisional. | Doc Raymond / farm operations | Open |
| OPEN-002 | Before farm-approved recommendations | Review all seven action-playbook rules, urgency wording, inspection checks, and escalation triggers. | Recommendations should reflect the farm owner’s operating practice. | All guidance is visibly preliminary and editable; no diagnosis or automatic treatment is given. | Doc Raymond | Open |
| OPEN-003 | Useful for secondary EDA | Confirm the two suspicious Lagundi 2026-1 summary rows and provide 2026-2/2026-3 final weights when available. | These values could strengthen secondary analysis of Day 14 versus later farm outcomes, but they are not the primary Day 35 prediction target. | Suspect rows are excluded; no missing future cycle is imputed. | Farm data owner | Open |
| OPEN-004 | Useful for secondary EDA | Confirm how Ave Live Weight (kg) is calculated, including partial harvests and which birds are included. | This determines how safely the summary can be used in secondary harvest-weight analysis. | The primary weight projection uses recorded Day 35 weights from Farm Harvest Data and does not derive Day 35 through linear scaling. | Farm data owner / mentor | Open |
| OPEN-005 | Before production or independently verified harvest claim | Identify a source of true harvest-complete date/status and final surviving population if Canary moves beyond the capstone convention. | `End Date` is only the maximum daily-record date and cannot independently prove harvest completion. | For the capstone, every earlier cycle is displayed as completed on its last recorded building date; the interface discloses this convention. | Doc Raymond / farm data owner | Open; non-blocking for capstone |
| OPEN-006 | Resolved for capstone scope | Recovery uses ending/surviving population divided by beginning population, with no additional adjustment for transfers, culls, partial harvests, or missing birds. | Locks the simple capstone accounting formula. | Keep the formula; still resolve what source record qualifies as the ending population under OPEN-005. | Doc Raymond / farm data owner | Confirmed |
| OPEN-007 | Before operational rollout | Agree on the bodyweight sampling protocol and minimum acceptable sample. | A stale or unrepresentative sample can mislead both the risk score and weight forecast. | The dashboard shows the measurement day and staleness and never calls a carried-forward weight “today’s weight.” | Farm operations | Open |
| OPEN-008 | Before capstone defense | Confirm whether every missing cycle-building combination means the building was not used, or whether its records are absent. | The workbook contains 3–6 buildings depending on cycle; without confirmation, “not recorded” cannot be interpreted as “not operated.” | Canary always shows all six physical buildings and explicitly marks missing historical building data rather than treating it as a healthy or completed flock. | Farm data owner | Open |
| OPEN-009 | Before farm-value claim | Confirm the live-chicken price per kg, expected sale weight, realistic recoverable improvement, and cycles per year. | These inputs materially change the estimated revenue opportunity. | The calculator is fully adjustable and labels its starting values as unsourced planning placeholders. It reports gross revenue—not profit or guaranteed savings. | Doc Raymond / farm finance | Open |
| OPEN-010 | Before farm-policy claim | Confirm the expected survival path from placement to the 95% target day. | Canary currently distributes the allowed 5% loss linearly through Day 35; the final target is agreed, but the daily path is an assumption. | The path is visible and editable in Data & Settings and remains labeled provisional. | Doc Raymond / farm operations | Open |
| OPEN-011 | Before farm-policy claim | Decide whether the mortality risk dimension should respond to absolute daily mortality, worsening trend, or both. | A trend-only score can miss mortality that is persistently high but no longer increasing. | Absolute 0.1%/0.2%/0.3% daily-mortality alerts remain separate operational checks; the core point score currently uses worsening trend. | Doc Raymond / farm operations | Open |
| OPEN-012 | Before farm-policy claim | Decide whether peer comparison should add 0–3 points, add a smaller modifier, or remain supporting context only. | Peer context can help identify building-specific drift, but it can also repeat the same weight, survival, or mortality signal already scored. | The current 0–3 peer contribution and exact evidence are disclosed; thresholds are editable and provisional. | Doc Raymond / mentor | Open |

## Useful later improvements — not capstone blockers

| ID | Item | Reason to revisit | Suggested trigger | Status |
|---|---|---|---|---|
| LATER-001 | Agree on acceptable forecast error and target-miss usefulness. | Model metrics are reported, but the business acceptance threshold is not yet defined. | Stakeholder model review. | Backlog |
| LATER-002 | Confirm feed units and improve temperature/humidity coverage. | These features may become more useful as source completeness improves. | Next workbook redesign. | Backlog |
| LATER-003 | Define a retraining and model-review schedule. | The recovery model has five complete training cycles; more cycles should improve confidence. | After each additional completed cycle or material process change. | Backlog |
| LATER-004 | Approve veterinary escalation wording and responsible contact path. | The prototype should guide inspection and escalation without diagnosing disease. | Before live farm use. | Backlog |
| LATER-005 | Add authentication, backups, audit retention, and deployment controls. | These are production concerns, not capstone requirements. | If the prototype is promoted to a shared production tool. | Backlog |

## Resolution rule

An item is closed only when the decision and its evidence are recorded in this file and the affected configuration, model card, or application wording is updated. Verbal approval alone should not silently change dashboard behavior.
