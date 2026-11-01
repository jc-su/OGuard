# segment_max Implementation Notes

Computes the maximum value within each segment defined by offset boundaries.

## Test Results
- Numba: All tests pass (n: 0, 1, 1000, 1000003)
- CUDA: All tests pass (n: 0, 1, 1000, 1000003)

## Key Implementation Details
- Uses segment offsets to define boundaries
- Returns LLONG_MIN for empty segments
- Proper handling of non-decreasing offsets
