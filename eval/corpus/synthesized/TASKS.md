# GPU query operator tasks

You write GPU kernels for a columnar query engine that runs on NVIDIA GPUs (RTX A6000, sm_86, CUDA 13.0). Columns live in device memory as flat arrays in the Apache Arrow layout. Write each kernel the way you would for a production engine: correct on every input size, including 0 and sizes that are not a multiple of the block size, and fast.

Write every task twice:

- `numba.py`: a Numba CUDA kernel (`from numba import cuda`, Numba 0.65). Define the kernel as a plain Python function named `kernel` decorated with nothing, a module-level `TYPES` tuple of Numba types for `cuda.compile_ptx`, and a function `run(...)` that launches `cuda.jit(kernel)` on device arrays. Arrays are 1-D Numba arrays (`float64[:]`, `int32[:]`, and so on); scalars are `int64` unless stated.
- `kernel.cu`: a CUDA C++ kernel `extern "C" __global__ void kernel(...)` with raw pointers and scalar arguments in the same order as the Numba version, plus a separate `test.cu` with a `main` that allocates inputs, launches the kernel, and checks the result against a CPU reference.

For each task directory, also write `signature.json`: one entry per kernel argument, in order, with the fields
`name`, `kind` (`"input"`, `"output"`, `"inout"`, or `"scalar"`), `dtype`, `length` (an expression over the other arguments' names, for example `"n"` or `"capacity"`; for scalars, the value range as `"[lo, hi]"` or a meaning such as `"rows of keys"`), and `indexes_into` (the name of the array whose positions this argument's values denote, or null).

Test every kernel on the GPU against a NumPy reference, with n in {0, 1, 1000, 1000003}. Compile each Numba kernel to PTX with `cuda.compile_ptx(kernel, TYPES, cc=(8, 6))` and save it as `numba.ptx`; compile each CUDA kernel with `nvcc -ptx -arch=sm_86 -O3 kernel.cu -o kernel.ptx`. Record in `NOTES.md` any task you could not complete and why.

## Tasks

1. `filter_select`: `quantity` (float64, n), scalar `threshold` (float64), outputs `sel` (int32, capacity n) and `count` (int32, 1 element). Write into `sel` the row ids `i` with `quantity[i] < threshold`, in any order, and the number of such rows into `count`.
2. `gather`: `col` (float64, n), `sel` (int32, m; every value is a row id of `col`), output `out` (float64, m), scalars `m`, `n`. `out[i] = col[sel[i]]`.
3. `projection`: `price`, `discount` (float64, n), output `out` (float64, n). `out[i] = price[i] * (1 - discount[i])`.
4. `q6_sum`: TPC-H Q6. `shipdate` (int32 days, n), `discount`, `quantity`, `price` (float64, n), scalars `date_lo`, `date_hi` (int32), `disc_lo`, `disc_hi`, `qty_max` (float64), output `result` (float64, 1 element, zero on entry). Add `price[i] * discount[i]` over the rows with `date_lo <= shipdate < date_hi`, `disc_lo <= discount <= disc_hi`, `quantity < qty_max`.
5. `hash_build`: `keys` (int64, n, all distinct and not equal to -1), outputs `table_keys` (int64, capacity, filled with -1 on entry) and `table_rows` (int32, capacity), scalar `capacity` (a power of two, at least 2n). Insert each row into an open-addressing hash table with linear probing: the slot holds the key and the row id.
6. `hash_probe`: `probe_keys` (int64, m), the table from `hash_build` (`table_keys`, `table_rows`, `capacity`), outputs `out_build` and `out_probe` (int32, capacity `out_cap`), `out_count` (int32, 1 element, zero on entry), scalars `m`, `out_cap`. For each probe row that matches a build key, emit the pair (build row id, probe row id). Write at most `out_cap` pairs; `out_count` holds the number of matches.
7. `groupby_sum`: `group` (int32, n), `value` (float64, n), outputs `table_keys` (int32, capacity, filled with -1 on entry), `table_sums` (float64, capacity, zero on entry), scalar `capacity` (a power of two, at least 2 times the number of groups). Hash aggregation: sum `value` per `group`.
8. `scatter_by_offsets`: `flags` (int8, n), `offsets` (int32, n; the exclusive prefix sum of `flags`), output `out` (int32, capacity `total`), scalars `n`, `total` (the sum of `flags`). For each `i` with `flags[i] != 0`, `out[offsets[i]] = i`.
9. `string_length`: a string column as `offsets` (int32, n + 1) and `chars` (uint8, `offsets[n]` bytes), output `lengths` (int32, n). The number of UTF-8 code points of each string.
10. `starts_with`: the same string column, a pattern `prefix` (uint8, k), output `match` (int8, n). `match[i] = 1` when string `i` starts with the pattern.
11. `histogram`: `keys` (uint32, n), output `counts` (int32, 256, zero on entry), scalar `shift` (0 to 24). Count the keys by the byte `(key >> shift) & 255`, as one pass of a radix sort.
12. `segmented_sum`: `offsets` (int32, g + 1, nondecreasing, `offsets[g] = n`), `values` (float64, n), output `sums` (float64, g), scalar `g`. `sums[j]` is the sum of `values[offsets[j]:offsets[j+1]]`.
