# hash_build notes

The task is complete. The Numba, CUDA, and Triton kernels pass their GPU tests for n in {0, 1, 1000, 1000003}.

## Decisions

- TASKS.md gives no row-count scalar for this task, but the CUDA kernel needs one. Both versions take `n` (int64) as the last argument: `kernel(keys, table_keys, table_rows, capacity, n)`.
- The home slot of a key is `fmix64(key) & (capacity - 1)`, where fmix64 is the MurmurHash3 64-bit finalizer. `hash_probe` uses the same hash and probe order.
- `table_keys` has the kind `inout` because the kernel reads the -1 values that the slots hold on entry.
- `run` raises `ValueError` if `capacity` is not a power of two of at least `max(1, 2 * n)`.
- The environment has Numba 0.64 with numba-cuda 0.30.4, not Numba 0.65.

## Triton version

- `tl.atomic_cas` has no mask. With it, every lane of a tile issues a compare-and-swap in every probe step, until the last key of the tile is in the table. On the RTX A6000, that made the build 2.2 to 5.2 times slower than the Numba version (4.2 million keys: 6.2 ms to 14.4 ms for tiles of 32 to 1024 keys, against 2.8 ms). A plain read of the slot before an unmasked CAS helped little (5.6 ms).
- `triton.py` therefore issues the CAS as predicated inline PTX (`tl.inline_asm_elementwise` with `@p atom.global.relaxed.gpu.cas.b64`), only in the lanes whose slot read EMPTY. The build then takes 2.4 ms for 4.2 million keys and 0.40 ms for 1 million keys, against 2.8 ms and 0.47 ms for the Numba version. The inline PTX limits the Triton kernel to NVIDIA GPUs.
- The asm runs once for each element that a thread holds. A layout that replicates an element across threads would issue its CAS more than once, so `BLOCK` must be a multiple of `32 * NUM_WARPS`. `triton.py` raises `ValueError` at import if it is not.
- Each program is one warp with one key per lane (`BLOCK = 32`, `NUM_WARPS = 1`), so the loop condition of a program is a warp reduction.

## Tests

Run the tests from the parent directory. The `-P` flag keeps this directory, which holds `numba.py` and `triton.py`, off `sys.path`.

```
CUDA_VISIBLE_DEVICES=1 python -P hash_build/test_numba.py
CUDA_VISIBLE_DEVICES=1 python -P hash_build/test_triton.py
nvcc -arch=sm_86 -O3 hash_build/test.cu -o test_hash_build && CUDA_VISIBLE_DEVICES=1 ./test_hash_build
```

The tests check that each key is in exactly one slot with its row id, and that a linear probe from the home slot of each key reaches the key. The tests also use keys that all have the same home slot near the table end, so that the probes wrap around.
