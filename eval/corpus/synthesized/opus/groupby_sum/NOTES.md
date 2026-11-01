# groupby_sum notes

The task is complete. The Numba, CUDA, and Triton kernels pass their GPU tests for n in {0, 1, 1000, 1000003}.

## Decisions

- TASKS.md gives no row-count scalar for this task, but the CUDA kernel needs one. Both versions take `n` (int64) as the last argument: `kernel(group, value, table_keys, table_sums, capacity, n)`.
- A group value of -1 is not supported, because -1 marks an empty slot.
- The home slot of a group is `fmix32(group) & (capacity - 1)`, where fmix32 is the MurmurHash3 32-bit finalizer.
- In each warp, the lanes that hold the same group (`match_any_sync`) add their values in a tree reduction. Then one lane per group adds the partial sum to the table. This reduces atomic contention when there are few groups. The warp operations use a lane mask computed from `blockDim.x`, so the kernel is correct for any 1-D block size.
- The order of the floating-point additions depends on thread timing, so the sums can differ from a sequential sum by rounding.
- `table_keys` and `table_sums` have the kind `inout` because the kernel reads the values that they hold on entry (-1 and 0).
- `run` raises `ValueError` if `capacity` is not a power of two.
- The environment has Numba 0.64 with numba-cuda 0.30.4, not Numba 0.65.

## Triton version

- Triton has no `match_any_sync`. Without a pre-aggregation, every row adds its value to the table, and with 1 group and 1 million rows the kernel took 1.7 ms, against 0.06 ms for the Numba version.
- `triton.py` therefore adds up the values of each group within a tile of 512 rows first. It sorts the rows of the tile by a 32-bit key: the group hash with its low 9 bits replaced by the row's lane. The rows of a group then form a run, and a segmented scan sums each run. The last row of each run adds the run sum to the table. A hash collision of two groups can interleave their rows; that only splits a group into more runs. A sort by the full group with a 64-bit key gave the same aggregation but was slower.
- Measured on the RTX A6000 with 1 million and 4.2 million rows, and 1, 32, 1000, and n / 2 groups in random and sorted order, the Triton version takes 0.19 to 1.42 times the time of the Numba version. It is faster with few groups in random order (32 groups: 0.19 ms against 0.97 ms for 4.2 million rows) and slower with 1000 groups in random order, where a tile of 512 rows has few rows of the same group (0.26 ms against 0.18 ms).
- As in `hash_build`, the Triton version claims a slot with a predicated inline PTX compare-and-swap (`atom.global.relaxed.gpu.cas.b32`), because `tl.atomic_cas` has no mask. `BLOCK` must be a multiple of `32 * NUM_WARPS`, and `triton.py` raises `ValueError` at import if it is not.

## Tests

Run the tests from the parent directory. The `-P` flag keeps this directory, which holds `numba.py` and `triton.py`, off `sys.path`.

```
CUDA_VISIBLE_DEVICES=1 python -P groupby_sum/test_numba.py
CUDA_VISIBLE_DEVICES=1 python -P groupby_sum/test_triton.py
nvcc -arch=sm_86 -O3 groupby_sum/test.cu -o test_groupby_sum && CUDA_VISIBLE_DEVICES=1 ./test_groupby_sum
```

The tests use 1 group, 1000 groups, and n / 2 groups, with random and sorted group order, negative group values, block sizes 32, 100, 256, and 1024, and groups that all have the same home slot near the table end. A sum passes if its error is at most 1e-9 times the sum of the absolute values of its group. The Triton test replaces the block sizes with tiles of 128 rows in 4 warps and 1024 rows in 8 warps, and adds groups whose rows the sort of a tile interleaves.
