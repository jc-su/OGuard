# Held-out GPU query operator tasks

You write GPU kernels for a columnar query engine that runs on NVIDIA GPUs (RTX A6000, sm_86, CUDA 13.0). Columns live in device memory as flat arrays in the Apache Arrow layout. Write each kernel the way you would for a production engine: correct on every input size, including 0 and sizes that are not a multiple of the block size, and fast.

Write every task three times, in the task's directory:

- `numba.py`: a Numba CUDA kernel (`from numba import cuda`). Define the kernel as a plain Python function named `kernel` decorated with nothing, a module-level `TYPES` tuple of Numba types for `cuda.compile_ptx`, and a function `run(...)` that launches `cuda.jit(kernel)` on device arrays with 256 threads per block. Arrays are 1-D Numba arrays (`float64[:]`, `int32[:]`, and so on).
- `kernel.cu`: a CUDA C++ kernel `extern "C" __global__ void kernel(...)` with raw pointers and scalar arguments, launched with 256 threads per block, plus a separate `test.cu` with a `main` that allocates inputs, launches the kernel, and checks the result against a CPU reference.
- `triton.py`: a Triton kernel `@triton.jit def kernel(...)`, followed by any `tl.constexpr` arguments, such as a block size. Define at module level `TYPES`, a dict from each argument that is not a `tl.constexpr` to its Triton type (`"*fp64"`, `"*i64"`, `"*i32"`, `"*u32"`, `"*i8"`, or `"*u8"` for a pointer to that element type, and `"i64"` for a scalar); `CONSTANTS`, a dict from each `tl.constexpr` argument to its value; `NUM_WARPS`, the number of warps of each program; and `run(...)`, which launches `kernel[grid](...)` with `num_warps=NUM_WARPS` on PyTorch CUDA tensors.

Each task lists the kernel's arguments in order. Every version takes them in that order and with those names. Scalars are `int64`.

For each task directory, also write `signature.json`: one entry per kernel argument, in order, with the fields `name`, `kind` (`"input"`, `"output"`, `"inout"`, or `"scalar"`), `dtype`, `length` (an expression over the other arguments' names, for example `"n"` or `"m + p"`; for scalars, the value range as `"[lo, hi]"` or a meaning such as `"rows of keys"`), and `indexes_into` (the name of the array whose positions this argument's values denote, or null).

Test every kernel on the GPU against a NumPy reference, with n in {0, 1, 1000, 1000003}: the Numba kernel in `test_numba.py`, the CUDA kernel in `test.cu`, and the Triton kernel in `test_triton.py`. Record in `NOTES.md` any task you could not complete and why.

## Tasks

1. `valid_rows(valid, rows, count, n)`: `valid` (uint8, `(n + 7) / 8` bytes) is an Arrow validity bitmap: row `i` is valid when bit `i % 8` of byte `i / 8` is 1, the least significant bit first. Outputs `rows` (int32, capacity n) and `count` (int32, 1 element, zero on entry). Write into `rows` the ids of the valid rows, in any order, and their number into `count`.
2. `dictionary_decode(codes, dictionary, out, n, d)`: `codes` (int32, n; every value is in `[0, d)`), `dictionary` (float64, d), output `out` (float64, n). `out[i] = dictionary[codes[i]]`.
3. `run_end_decode(run_values, run_ends, out, r, n)`: an Arrow run-end encoded column of n rows: `run_values` (float64, r) and `run_ends` (int32, r; strictly increasing, `run_ends[r - 1] = n`); run `j` covers the rows from `run_ends[j - 1]` (0 for `j = 0`) up to `run_ends[j]`. Output `out` (float64, n), the decoded column.
4. `string_contains(offsets, chars, pattern, match, n, k)`: a string column as `offsets` (int32, n + 1) and `chars` (uint8, `offsets[n]` bytes), a pattern `pattern` (uint8, k), output `match` (int8, n). `match[i] = 1` when the pattern occurs in string `i` as a byte substring, and 0 otherwise; the empty pattern occurs in every string.
5. `min_max(values, result, n)`: `values` (float64, n, no NaN), output `result` (float64, 2 elements; `+inf` and `-inf` on entry). Store the minimum of `values` in `result[0]` and the maximum in `result[1]`; with n = 0 the result stays as it is.
6. `rolling_sum(values, out, n, window)`: `values` (float64, n), output `out` (float64, n), scalar `window` (1 to 1024). `out[i]` is the sum of `values[j]` over `max(0, i - window + 1) <= j <= i`.
7. `radix_scatter(keys, cursors, out, n, shift)`: `keys` (uint32, n), `cursors` (int32, 256; on entry, the exclusive prefix sum of the counts of the byte `(key >> shift) & 255` over the keys, as a histogram pass of a radix sort gives), output `out` (int32, n), scalar `shift` (0 to 24). Write each row id `i` into `out`, at the position its byte's cursor gives, advancing the cursor, so that `out` holds the row ids grouped by byte, in any order within a byte.
8. `days_to_year(days, year, n)`: `days` (int32, n; days since 1970-01-01 in the proleptic Gregorian calendar, of dates in the years 1 to 9999), output `year` (int32, n), the year of each date.
9. `merge_sorted(a, b, out, m, p)`: `a` (int64, m) and `b` (int64, p), each sorted ascending, output `out` (int64, m + p): the merge of `a` and `b`, sorted ascending, with an element of `a` before an equal element of `b`.
10. `sorted_contains(keys, members, match, n, s)`: `keys` (int64, n), `members` (int64, s; sorted ascending, distinct), output `match` (int8, n). `match[i] = 1` when `keys[i]` is one of `members`, and 0 otherwise.
