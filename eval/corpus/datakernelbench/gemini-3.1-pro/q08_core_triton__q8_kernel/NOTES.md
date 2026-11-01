# q08_core_triton__q8_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q08/round_02.py`, copied as `plan.py`: `_q8_kernel` in `_KERNEL_SOURCE`, lines 25-52 of the plan file.

## Extraction edits

- `def _q8_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configurations 1 and 2 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K._q8_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; it returns before the launch when `n == 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 1024`, `num_warps = 4`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

`triton.json` states the grid maxima (2097152, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `o_year_ptr`, `volume_ptr`, `is_brazil_ptr`: the year of `o_orderdate`, the volume, and the nation flag (cast to int8) of the joined rows, `n_elements = o_year.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `sum_brazil_ptr`, `sum_total_ptr`, `present_ptr`: `torch.zeros(num_groups)` with `num_groups = max(o_year) - min(o_year) + 1`, zeroed by the plan and updated with `tl.atomic_add` and `tl.atomic_max`: readwrite; `num_groups` is at least 1.
- `min_year = min(o_year)`, so `o_year - min_year` lies in `[0, num_groups)` for the plan's data, a relation between the column, the scalar, and the outputs' length that the signature does not state: `o_year_ptr` is `read`.
