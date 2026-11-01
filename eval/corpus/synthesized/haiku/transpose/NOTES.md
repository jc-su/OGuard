# transpose Implementation Notes

## Implementation
Transposes a row-major matrix by mapping element at (i,j) to (j,i).

## Test Results
### Numba (GPU 1, Numba CUDA)
- 0x0: PASS
- 1x1: PASS
- 10x100: PASS
- 1000x1001: PASS
- 32x32: PASS

### CUDA C++ (GPU 1, CUDA 13.0, sm_86)
- 0x0: PASS
- 1x1: PASS
- 10x100: PASS
- 1000x1001: PASS

All tests pass successfully.
