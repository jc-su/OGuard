# masked_sum Implementation Notes

Sums values where mask is nonzero using atomic operations.

## Test Results
- Numba: All tests pass
- CUDA: All tests pass

## Implementation Details
- Uses atomic add for thread-safe accumulation
- Only includes rows where mask != 0
- Accumulates to single output element
