# window_max Implementation Notes

## Implementation
Computes sliding window maximum for each position, finding the maximum of values[max(0, i - w + 1) : i + 1].

## Test Results
### Numba (GPU 1, Numba CUDA)
- n=0, w=1: PASS
- n=1, w=1: PASS
- n=100, w=5: PASS
- n=1000, w=32: PASS
- n=1000003, w=256: PASS

### CUDA C++ (GPU 1, CUDA 13.0, sm_86)
- n=0, w=1: PASS
- n=1, w=1: PASS
- n=100, w=5: PASS
- n=1000, w=32: PASS
- n=1000003, w=256: PASS

All tests pass successfully.
