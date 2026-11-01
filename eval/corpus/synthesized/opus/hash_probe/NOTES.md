# hash_probe notes

The task is complete. The Numba, CUDA, and Triton kernels pass their GPU tests for m in {0, 1, 1000, 1000003}.

## Decisions

- The kernel uses the hash and probe order of `hash_build`: the home slot of a key is `fmix64(key) & (capacity - 1)`.
- The kernel checks for an empty slot before it compares keys, so a probe key of -1 never matches.
- Each warp gets its output positions with one atomic add on `out_count`. The warp vote uses a lane mask computed from `blockDim.x`, so the kernel is correct for any 1-D block size, including sizes that are not a multiple of 32.
- `out_count` receives the number of matches, also when it is larger than `out_cap`. The kernel writes only the pairs at positions below `out_cap`.
- In `signature.json`, `table_rows` and `out_build` have `indexes_into` set to null. Their values are row ids of the build-side keys (`keys` of `hash_build`), which are not an argument of this kernel.
- `out_count` has the kind `inout` because it must hold 0 on entry.
- `run` raises `ValueError` if `capacity` is not a power of two.
- The environment has Numba 0.64 with numba-cuda 0.30.4, not Numba 0.65.

## Tests

Run the tests from the parent directory. The `-P` flag keeps this directory, which holds `numba.py` and `triton.py`, off `sys.path`.

```
CUDA_VISIBLE_DEVICES=1 python -P hash_probe/test_numba.py
CUDA_VISIBLE_DEVICES=1 python -P hash_probe/test_triton.py
nvcc -arch=sm_86 -O3 hash_probe/test.cu -o test_hash_probe && CUDA_VISIBLE_DEVICES=1 ./test_hash_probe
```

The tests build the table on the CPU, then compare the pairs with a CPU join. They cover `out_cap` smaller than the number of matches, probe keys that repeat, a probe key of -1, block sizes 32, 100, 256, and 1024, and keys that all have the same home slot near the table end. The Triton test replaces the block sizes with tiles of 128 keys in 4 warps and 1024 keys in 4 warps, next to the tile of `run` (32 keys in 1 warp).
