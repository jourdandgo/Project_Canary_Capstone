-- Reviewed stakeholder summary derived from docs/OPEN_ITEMS.md.

SELECT 'Risk thresholds and rating bands' AS decision, 'Versioned and visibly provisional.' AS current_behavior
UNION ALL SELECT 'Seven recommended-action rules', 'Visibly preliminary and editable with explicit approval controls.'
UNION ALL SELECT 'Final average-liveweight definition and denominator', 'Accepted labels are used; eligible buildings receive a visibly non-personalized farm baseline.'
UNION ALL SELECT 'Missing building-cycle records', 'All six physical buildings remain visible and missing records are explicitly labeled.'
UNION ALL SELECT 'Actual versus planned End Date', 'Default uses latest complete daily data; selected End Date is treated as harvested.'
UNION ALL SELECT 'Culls, transfers, and partial harvests', 'Recovery uses the agreed simple beginning-to-ending inventory formula.';
