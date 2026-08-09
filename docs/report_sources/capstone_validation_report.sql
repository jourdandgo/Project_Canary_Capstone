-- Reviewed report snapshot derived from artifacts/capstone_validation.json.
-- The JSON remains the detailed building-level source of truth.

SELECT 45 AS checks_passed, 45 AS checks_total, 5 AS scenarios, 1666 AS canonical_rows;

SELECT 'Day 14' AS scenario_short, 9 AS checks_passed, 9 AS checks_run, 'Pass' AS result
UNION ALL SELECT 'Day 22', 9, 9, 'Pass'
UNION ALL SELECT 'Mixed states', 9, 9, 'Pass'
UNION ALL SELECT 'Day 48', 9, 9, 'Pass'
UNION ALL SELECT 'Missing day', 9, 9, 'Pass';

SELECT 'A. Risk rating' AS output, 'Pass' AS result, 'Farm approval of provisional thresholds remains open.' AS qualification
UNION ALL SELECT 'B. Why', 'Pass', 'Raw observations, thresholds, scores, freshness, equation, and label rule are traceable.'
UNION ALL SELECT 'C. Predicted recovery', 'Pass as prototype', 'Five completed training cycles; cycle-held-out MAE 1.26 percentage points.'
UNION ALL SELECT 'D. Predicted average weight', 'Experimental farm baseline', '17 accepted final-harvest labels; MAE 0.093 kg; not personalized and target-side accuracy is 32.2%.'
UNION ALL SELECT 'E. Recommended action', 'Pass as preliminary guidance', 'Seven deterministic rules remain pending Doc Raymond review.';
