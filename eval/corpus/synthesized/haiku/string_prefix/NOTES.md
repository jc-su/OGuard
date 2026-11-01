# string_prefix Implementation Notes

## Implementation
The kernel extracts the first k bytes from each string in a variable-length string column. Strings are stored as a character array with an offset array indicating where each string starts.

## Correctness
- Handles variable-length strings correctly
- Pads with zeros when string is shorter than k
- Handles n=0 correctly

## Test Results
### Numba (GPU 1, Numba CUDA)
All tests pass for n in {0, 1, 1000, 1000003} and k in {1, 2, 32, 64}.

### CUDA C++ (GPU 1, CUDA 13.0, sm_86)
- n=0, k=1: PASS
- n=1, k=1: PASS
- n=1000, k=32: PASS
- n=1000003, k=64: PASS

All tests pass successfully.
