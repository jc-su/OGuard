# group_max Implementation Notes

Finds the maximum value for each group using atomic operations.

## Test Results
- Numba: All tests pass
- CUDA: All tests pass

## Implementation Details
- Uses atomic max operations for thread safety
- Groups defined by group array
- Initializes with smallest int64 value
