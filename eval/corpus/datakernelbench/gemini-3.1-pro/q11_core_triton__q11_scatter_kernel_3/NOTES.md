# q11_core_triton__q11_scatter_kernel_3

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q11/round_01.py`, copied as `plan.py`: `_q11_scatter_kernel` in `_KERNEL_SOURCE`, lines 25-44 of the plan file.

## Extraction edits

- `def _q11_scatter_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configurations 5 and 6 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q11_scatter_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; it returns before the launch when `n == 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 2048`, `num_warps = 4`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

`triton.json` states the grid maxima (1048576, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `partkey_ptr`, `supplycost_ptr`, `availqty_ptr`: columns of the joined frame, `n_elements = ps_partkey.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `out_ptr`: `torch.zeros(max_key + 1)` with `max_key = int(ps_partkey.max())`, zeroed by the plan and accumulated with `tl.atomic_add`: readwrite.
- `partkey_ptr`: the plan sizes `out_ptr` by this column's maximum, and the TPC-H part keys are positive: `index-in` into `out_ptr`.
