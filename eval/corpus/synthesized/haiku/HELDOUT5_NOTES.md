# Haiku: kernels of HELDOUT5.md left out of the corpus

- All 20 Triton kernels: the arguments differ from the `signature.json`.
  Each buffer argument has a `_ptr` suffix, so `between_bitmap` takes
  `(values_ptr, valid_ptr, out_ptr, n, lo, hi)`, not
  `(values, valid, out, n, lo, hi)`. Two also differ in more than the
  suffix: `range_window_sum` names its last scalar `range_val`, not
  `range`, and `bit_unpack` takes `(packed_ptr, out_ptr, n, bits)` and
  omits `w`. `eval.corpus.sources triton` wrote no PTX for them.
- `range_window_sum`, Numba and CUDA C++: the kernel names its last
  argument `range_val`, and the `signature.json` and the task name it
  `range`. `eval.corpus.sources` wrote no PTX for them.
- `bit_unpack`, Numba and CUDA C++: the kernel takes
  `(packed, out, n, bits)` and omits the argument `w` that the task and the
  `signature.json` state, so the arguments differ in number and order.
  `eval.corpus.sources` wrote no PTX for them.
- `add_months`, Numba: the source does not compile. The kernel calls the
  module's functions `is_leap_year` and `days_in_month`, which are plain
  Python functions, not `cuda.jit(device=True)` functions, so Numba's type
  inference fails with "Untyped global name 'is_leap_year'". No PTX was
  written.
