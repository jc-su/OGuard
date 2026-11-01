# scatter_by_offsets notes

The task is complete. The Numba, CUDA, and Triton kernels pass their GPU tests for n in {0, 1, 1000, 1000003}.

## Decisions

- The kernel does not read `total`. It is the length of `out`, and the kernel relies on the precondition that `offsets` is the exclusive prefix sum of `flags`, so every written position is below `total`.
- The tests use flags of 0 or 1. With a flag larger than 1, the positions between `offsets[i] + 1` and `offsets[i + 1]` are not written.
- The environment has Numba 0.64 with numba-cuda 0.30.4, not Numba 0.65.

## Tests

Run the tests from the parent directory. The `-P` flag keeps this directory, which holds `numba.py` and `triton.py`, off `sys.path`.

```
CUDA_VISIBLE_DEVICES=1 python -P scatter_by_offsets/test_numba.py
CUDA_VISIBLE_DEVICES=1 python -P scatter_by_offsets/test_triton.py
nvcc -arch=sm_86 -O3 scatter_by_offsets/test.cu -o test_scatter_by_offsets && CUDA_VISIBLE_DEVICES=1 ./test_scatter_by_offsets
```

The tests use flag densities of 0, 0.001, 0.5, and 1, and check that no write goes past `total`.
