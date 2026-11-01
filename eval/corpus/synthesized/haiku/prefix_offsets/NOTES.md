# prefix_offsets Implementation Notes

## Implementation
Computes inclusive prefix sum of lengths array. Each thread computes its offset by summing all previous lengths.

## Test Results
### Numba (GPU 1, Numba CUDA)
- n=0: PASS
- n=1: PASS
- n=1000: PASS
- n=1000003: PASS

### CUDA C++ (GPU 1, CUDA 13.0, sm_86)
- n=0: PASS
- n=1: PASS
- n=1000: PASS
- n=1000003: PASS

All tests pass successfully.
