# Fourth set of held-out GPU query operator tasks

You write GPU kernels for a columnar query engine that runs on NVIDIA GPUs (RTX A6000, sm_86, CUDA 13.0). Columns live in device memory as flat arrays in the Apache Arrow layout. Write each kernel the way you would for a production engine: correct on every input size, including 0 and sizes that are not a multiple of the block size, and fast.

Write every task three times, in the task's directory:

- `numba.py`: a Numba CUDA kernel (`from numba import cuda`). Define the kernel as a plain Python function named `kernel` decorated with nothing, a module-level `TYPES` tuple of Numba types for `cuda.compile_ptx`, and a function `run(...)` that launches `cuda.jit(kernel)` on device arrays with 256 threads per block. Arrays are 1-D Numba arrays (`float64[:]`, `int32[:]`, and so on).
- `kernel.cu`: a CUDA C++ kernel `extern "C" __global__ void kernel(...)` with raw pointers and scalar arguments, launched with 256 threads per block, plus a separate `test.cu` with a `main` that allocates inputs, launches the kernel, and checks the result against a CPU reference.
- `triton.py`: a Triton kernel `@triton.jit def kernel(...)`, followed by any `tl.constexpr` arguments, such as a block size. Define at module level `TYPES`, a dict from each argument that is not a `tl.constexpr` to its Triton type (`"*fp64"`, `"*i64"`, `"*i32"`, `"*u32"`, `"*i8"`, or `"*u8"` for a pointer to that element type, and `"i64"` for a scalar); `CONSTANTS`, a dict from each `tl.constexpr` argument to its value; `NUM_WARPS`, the number of warps of each program; and `run(...)`, which launches `kernel[grid](...)` with `num_warps=NUM_WARPS` on PyTorch CUDA tensors.

Each task lists the kernel's arguments in order. Every version takes them in that order and with those names. Scalars are `int64`.

For each task directory, also write `signature.json`: one entry per kernel argument, in order, with the fields `name`, `kind` (`"input"`, `"output"`, `"inout"`, or `"scalar"`), `dtype`, `length` (an expression over the other arguments' names, for example `"n"` or `"g + 1"`; for scalars, the value range as `"[lo, hi]"` or a meaning such as `"rows of keys"`), and `indexes_into` (the name of the array whose positions this argument's values denote, or null).

Test every kernel on the GPU against a NumPy reference, with n in {0, 1, 1000, 1000003}: the Numba kernel in `test_numba.py`, the CUDA kernel in `test.cu`, and the Triton kernel in `test_triton.py`. Record in `NOTES.md` any task you could not complete and why.

## Tasks

1. `stream_compact(values, keep, out, count, n)`: `values` (int64, n), `keep` (int8, n), output `out` (int64, n), output `count` (int32, 1 element; zero on entry). Write the values of the rows with `keep[i] != 0` to the first `count[0]` elements of `out`, in any order, and set `count[0]` to the number of such rows.
2. `string_prefix(offsets, chars, out, n, k)`: a string column as `offsets` (int32, n + 1) and `chars` (uint8, `offsets[n]` bytes), output `out` (uint8, `n * k`), with `k` in `[1, 64]`. Write the first `min(k, length)` bytes of string `i` to `out[i * k : i * k + k]`, and 0 in the rest of the `k` bytes.
3. `join_count(build, probe, counts, b, p)`: `build` (int64, b; sorted ascending), `probe` (int64, p), output `counts` (int32, p). `counts[i]` is the number of elements of `build` equal to `probe[i]`.
4. `null_count(valid, count, n)`: an Arrow validity bitmap `valid` (uint8, `(n + 7) / 8` bytes; bit `i % 8` of byte `i / 8` is row `i`), output `count` (int64, 1 element; zero on entry). Add to `count[0]` the number of rows in `[0, n)` whose bit is 0.
5. `prefix_offsets(lengths, offsets, n)`: `lengths` (int32, n; every value in `[0, 2^20]`), output `offsets` (int32, n + 1). `offsets[0] = 0` and `offsets[i + 1] = offsets[i] + lengths[i]`.
6. `transpose(source, out, rows, cols)`: `source` (int32, `rows * cols`, row-major), output `out` (int32, `rows * cols`). `out[j * rows + i] = source[i * cols + j]`.
7. `hash_partition(keys, part, counts, n, p)`: `keys` (int64, n), output `part` (int32, n), output `counts` (int32, p; zero on entry), with `p` in `[1, 1024]`. `part[i]` is `h(keys[i]) % p` for `h(x) = (x * 0x9E3779B97F4A7C15) >> 32` computed on the unsigned 64-bit value, and `counts[k]` becomes the number of rows with `part[i] = k`.
8. `month_start(days, out, n)`: `days` (int32, n; days since 1970-01-01, every value in `[-719162, 2932896]`), output `out` (int32, n). `out[i]` is the day number of the first day of the month that contains day `days[i]`, in the proleptic Gregorian calendar.
9. `window_max(values, out, n, w)`: `values` (int64, n), output `out` (int64, n), with `w` in `[1, 1024]`. `out[i]` is the largest of `values[max(0, i - w + 1) : i + 1]`.
10. `pair_sum(left_idx, right_idx, left, right, out, m, l, r)`: `left_idx` (int32, m; every value a row of `left`), `right_idx` (int32, m; every value a row of `right`), `left` (float64, l), `right` (float64, r), output `out` (float64, m). `out[i] = left[left_idx[i]] + right[right_idx[i]]`.
