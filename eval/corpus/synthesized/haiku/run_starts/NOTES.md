# run_starts Implementation Notes

## Task Description
Identifies the first row of each run of equal values.
- `flags[i] = 1` when `i = 0` or `values[i] != values[i - 1]`, else 0
- Simple element-wise comparison operation

## Implementation Details

### Numba Version (numba.py)
- Simple kernel: each thread checks if it's the first element or if value changed
- Special handling for n=0 case (return early without launching kernel)
- Device memory management: copy to device, launch, copy back
- 256 threads per block configuration

### CUDA C++ Version (kernel.cu)
- Direct CUDA kernel with raw pointers
- Ternary operator for conditional assignment
- Same logic as Numba version
- Test includes random value generation and CPU reference

## Test Results
- n=0: PASS (empty input handled correctly)
- n=1: PASS (single element always has flag=1)
- n=1000: PASS
- n=1000003: PASS (large input)

All tests verify against NumPy/CPU reference implementation.
