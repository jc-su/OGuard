# Haiku: kernels of HELDOUT4.md left out of the corpus

- `string_prefix`, Numba, CUDA C++, and Triton: the adapter cannot read the
  `signature.json`. It gives `out` the length `n * k`, a product of two
  sizes, which is not a sum of size names and constants, and not a length
  such as `offsets[n]` or a quotient that gives the buffer its own size
  symbol. `eval.corpus.sources` wrote no PTX for them.
- `transpose`, Numba, CUDA C++, and Triton: the adapter cannot read the
  `signature.json`, which gives `source` and `out` the length `rows * cols`.
  `eval.corpus.sources` wrote no PTX for them.
- `stream_compact`, `join_count`, `null_count`, `prefix_offsets`,
  `hash_partition`, `month_start`, `window_max`, and `pair_sum`, Triton: the
  module does not load. `triton.py` does not import `triton` (its first
  comment expects the loader to inject `triton`, `tl`, and `torch`), so
  `@triton.jit` raises `NameError: name 'triton' is not defined`. The
  kernels' arguments also differ from the `signature.json`: each buffer
  argument has a `_ptr` suffix, so `stream_compact` takes
  `(values_ptr, keep_ptr, out_ptr, count_ptr, n)`, not
  `(values, keep, out, count, n)`. The Triton kernels of `string_prefix` and
  `transpose` have the same two faults. Haiku's `month_start/NOTES_TRITON.md`
  also states that its Triton kernel computes wrong months.
  `eval.corpus.sources triton` wrote no PTX for them.
