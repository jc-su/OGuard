# q17_triton_pass1_kernel_7

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q17/round_04.py`, copied as `plan.py`: `pass1_kernel` in `_KERNEL_SOURCE`, lines 23-51 of the plan file.

## Extraction edits

- `def pass1_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-22) is replaced by the values of configurations 13 and 14 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K.pass1_kernel[grid](...)` with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`; it returns before the launch when the filtered parts or the lineitem rows are empty.

This corpus kernel is the configuration `BLOCK_SIZE = 8192`, `num_warps = 4`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

## Lengths and ranges

- `l_partkey_ptr`, `l_quantity_ptr`: columns of `lineitem`, `n_elements` rows each, at least 1 (i32 at scale factor 10).
- `is_valid_ptr` (uint8), `sum_qty_ptr`, `cnt_qty_ptr`: `torch.zeros(max_pk + 1)` with `max_pk = max(part_keys.max(), l_partkey.max())`; the two sums are zeroed and accumulated with `tl.atomic_add`: readwrite. `max_pk` is the argument (size).
- `l_partkey_ptr`: `max_pk` is at least this column's maximum, and the TPC-H keys are positive: `index-in` into `is_valid_ptr`.
