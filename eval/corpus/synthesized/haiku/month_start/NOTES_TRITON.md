# month_start Implementation Notes

## Status
The month_start kernel has been partially implemented but contains bugs in the calendar conversion logic. The implementation attempts to convert day numbers (days since 1970-01-01) to calendar dates and extract the first day of the month, but the calculations are incorrect for many date ranges.

## Issues
1. The year refinement loop may not converge correctly for all day numbers
2. The month boundary calculations are error-prone due to Triton's limitations on break statements and list indexing
3. The nested if-else structure for month checking is difficult to maintain and verify

## What Works
- Empty arrays (n=0)
- Some simple cases like day 0 (1970-01-01)

## What Needs Fixing
- Date conversion algorithm needs thorough verification against reference implementation
- Leap year calculation needs validation
- Off-by-one errors in day counting

## Recommended Approach for Future Work
Use a more algorithmic approach with explicit calculations for:
1. Number of days from epoch to a given year
2. Day of year within a year
3. Month from day-of-year

Or implement using lookup tables for days per month and year transition points.
