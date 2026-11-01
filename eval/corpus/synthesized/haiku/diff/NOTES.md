# diff Implementation Notes

Computes consecutive differences between elements.

## Test Results
- Numba: All tests pass
- CUDA: All tests pass

## Implementation Details
- First element equals original value
- Subsequent elements are difference from previous
- Simple element-wise parallel computation
