# Triton implementation notes

All 10 tasks were completed in Triton and verified on GPU 1 (RTX A6000)
against a NumPy/Python reference for n in {0, 1, 1000, 1000003}. No task
was left incomplete.

A few implementation notes that affected the design, in case they matter
for review:

- Triton's `//` and `%` on signed integers truncate toward zero (C
  semantics), not Python's floor semantics. `days_to_year` relies on this
  directly (the Howard Hinnant `civil_from_days` algorithm is written for
  truncating division).
- A bare Python `int`/`float` kernel argument (as opposed to one produced
  by `tl.load` or tensor arithmetic) is usable in ordinary Python
  expressions and as a `range()` bound/step inside the `@triton.jit`
  function, but it is not itself a `tl` value: calling a tensor method
  such as `.to(...)` directly on it fails. Where a scalar argument (e.g.
  `r`, `s`, `m`, `p`, `window`) needed to seed a per-lane vector, it was
  broadcast via `tl.zeros(...) + scalar` instead of `scalar.to(dtype)`.
- Triton's `for` loops support a data-dependent (runtime) trip count and
  can be nested, as long as the bound is a scalar shared by every lane in
  the program (not a per-lane vector). This is used for:
  - `rolling_sum`: the window-offset loop bound is the runtime `window`.
  - `string_contains`: one CTA handles one row, so the per-row shift
    range and pattern-chunk range are both uniform scalars for that
    program; the inner BLOCK_K-wide chunk comparison still parallelizes
    the pattern-byte compare across lanes.
  - `merge_sorted`, `sorted_contains`, `run_end_decode`: binary search
    uses a fixed `ITERS = 32` (comfortably >= ceil(log2(max array size)))
    rather than a size-dependent bound, since the search state (lo, hi)
    provably freezes once converged, so extra iterations are free of
    effect.
- `tl.atomic_min`/`tl.atomic_max` work directly on `float64` pointers on
  this GPU/Triton build, so `min_max` needs no bit-trick reinterpretation
  to do float atomics.
- `radix_scatter` and `string_contains` use per-lane scatter atomics /
  gather loads (vector pointer offsets), which Triton supports directly.
