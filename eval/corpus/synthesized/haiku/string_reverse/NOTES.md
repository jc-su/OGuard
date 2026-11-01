# string_reverse Implementation Notes

Reverses bytes within variable-length strings.

## Test Results
- Numba: All tests pass
- CUDA: All tests pass

## Implementation Details
- Handles variable-length strings via offsets
- Reverses within each string's byte boundaries
- Maintains string structure in output
