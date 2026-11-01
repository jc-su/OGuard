# days_to_year Implementation

## Approach
Iterative year-by-year calculation from epoch (1970-01-01). For positive days, advance forward; for negative days, go backward. Uses leap year logic: divisible by 400 (leap), 100 (not leap), 4 (leap).

## Status
✓ Completed

## Implementation Details
- Epoch is 1970-01-01 (day 0)
- Handles both positive (future) and negative (past) day counts
- Proper leap year calculation (Gregorian calendar)
- Linear iteration (could be optimized with binary search if needed)
