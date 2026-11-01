# Third set of held-out GPU query operator tasks

You write GPU kernels for a columnar query engine that runs on NVIDIA GPUs (RTX A6000, sm_86, CUDA 13.0). Columns live in device memory as flat arrays in the Apache Arrow layout. Write each kernel the way you would for a production engine: correct on every input size, including 0 and sizes that are not a multiple of the block size, and fast.

Write every task three times, in the task's directory:

- `numba.py`: a Numba CUDA kernel (`from numba import cuda`). Define the kernel as a plain Python function named `kernel` decorated with nothing, a module-level `TYPES` tuple of Numba types for `cuda.compile_ptx`, and a function `run(...)` that launches `cuda.jit(kernel)` on device arrays with 256 threads per block. Arrays are 1-D Numba arrays (`float64[:]`, `int32[:]`, and so on).
- `kernel.cu`: a CUDA C++ kernel `extern "C" __global__ void kernel(...)` with raw pointers and scalar arguments, launched with 256 threads per block, plus a separate `test.cu` with a `main` that allocates inputs, launches the kernel, and checks the result against a CPU reference.
- `triton.py`: a Triton kernel `@triton.jit def kernel(...)`, followed by any `tl.constexpr` arguments, such as a block size. Define at module level `TYPES`, a dict from each argument that is not a `tl.constexpr` to its Triton type (`"*fp64"`, `"*i64"`, `"*i32"`, `"*u32"`, `"*i8"`, or `"*u8"` for a pointer to that element type, and `"i64"` for a scalar); `CONSTANTS`, a dict from each `tl.constexpr` argument to its value; `NUM_WARPS`, the number of warps of each program; and `run(...)`, which launches `kernel[grid](...)` with `num_warps=NUM_WARPS` on PyTorch CUDA tensors.

Each task lists the kernel's arguments in order. Every version takes them in that order and with those names. Scalars are `int64`.

For each task directory, also write `signature.json`: one entry per kernel argument, in order, with the fields `name`, `kind` (`"input"`, `"output"`, `"inout"`, or `"scalar"`), `dtype`, `length` (an expression over the other arguments' names, for example `"n"` or `"g + 1"`; for scalars, the value range as `"[lo, hi]"` or a meaning such as `"rows of keys"`), and `indexes_into` (the name of the array whose positions this argument's values denote, or null).

Test every kernel on the GPU against a NumPy reference, with n in {0, 1, 1000, 1000003}: the Numba kernel in `test_numba.py`, the CUDA kernel in `test.cu`, and the Triton kernel in `test_triton.py`. Record in `NOTES.md` any task you could not complete and why.

## Tasks

1. `run_starts(values, flags, n)`: `values` (int64, n), output `flags` (int8, n). `flags[i] = 1` when `i = 0` or `values[i] != values[i - 1]`, and 0 otherwise: the first row of each run of equal values.
2. `segment_max(offsets, values, maxes, g)`: `offsets` (int32, g + 1; nondecreasing, `offsets[g] = n`), `values` (int64, n), output `maxes` (int64, g). `maxes[j]` is the largest of `values[offsets[j]:offsets[j + 1]]`, and the smallest int64 value for an empty segment.
3. `lower_bound(keys, sorted, out, n, s)`: `keys` (int64, n), `sorted` (int64, s; sorted ascending), output `out` (int32, n). `out[i]` is the number of elements of `sorted` that are less than `keys[i]`, a value in `[0, s]`.
4. `group_max(group, value, best, n, capacity)`: `group` (int32, n; every value in `[0, capacity)`), `value` (int64, n), output `best` (int64, capacity; the smallest int64 value on entry). `best[k]` becomes the largest `value[i]` with `group[i] = k`, and stays as it is for a group with no row.
5. `string_reverse(offsets, chars, out, n)`: a string column as `offsets` (int32, n + 1) and `chars` (uint8, `offsets[n]` bytes), output `out` (uint8, `offsets[n]` bytes). Write each string's bytes to `out` in reverse order, at the string's own offsets, so that `offsets` with `out` is the column of byte-reversed strings.
6. `masked_sum(values, mask, result, n)`: `values` (float64, n), `mask` (int8, n), output `result` (float64, 1 element; zero on entry). Add to `result[0]` the sum of `values[i]` over the rows with `mask[i] != 0`.
7. `bucket_histogram(values, bounds, counts, n, b)`: `values` (float64, n, no NaN), `bounds` (float64, b; sorted ascending), output `counts` (int32, b + 1; zero on entry). For each row, add 1 to `counts[c]`, where `c` is the number of bounds that are less than or equal to `values[i]`.
8. `diff(values, out, n)`: `values` (int64, n), output `out` (int64, n). `out[0] = values[0]`, and `out[i] = values[i] - values[i - 1]` for `i > 0`.
9. `scatter_add(idx, values, out, m, n)`: `idx` (int32, m; every value is a row id of `out`), `values` (float64, m), output `out` (float64, n; zero on entry). For each `i`, add `values[i]` to `out[idx[i]]`.
10. `tile_reverse(values, out, n)`: `values` (int32, n), output `out` (int32, n). The rows form tiles of 1024 consecutive rows, the last of which can be shorter. Write each tile's values to the same rows of `out` in reverse order within the tile.
