# min_max: Triton

Status: complete. `test_triton.py` passes for n in {0, 1, 1000, 1000003} with mixed-sign, all-negative, all-positive values and extremes (1e300, -1e300) at the ends; with n = 0 no program runs and the result stays (+inf, -inf).

Approach: each program reduces 8 unrolled tiles of 1024 values in registers (min and max vectors), then one `tl.min` / `tl.max` and one relaxed fp64 `atomic_min` / `atomic_max` on the two result cells. Triton lowers the fp64 atomics to signed and unsigned 64-bit integer atomics, which is exact for values without NaN. A grid-stride loop variant was 40% slower than the unrolled tiles.

Time at n = 1000003: 20 us (8 MB read, about 400 GB/s with the L2 flushed between runs).
