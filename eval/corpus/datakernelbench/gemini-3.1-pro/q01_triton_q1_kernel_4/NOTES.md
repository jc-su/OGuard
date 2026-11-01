# q01_triton_q1_kernel_4

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q01/round_03.py`, copied as `plan.py`: `q1_kernel` in `_KERNEL_SOURCE`, lines 27-83 of the plan file.

## Extraction edits

- `def q1_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-26) is replaced by the values of configurations 7 and 8 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`run_query` launches `_K.q1_kernel[grid](...)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)` and `NUM_GROUPS=num_groups`, `num_groups = num_rf * num_ls` from the category counts of `l_returnflag` and `l_linestatus`: 6 at the plan's TPC-H launch (A, N, R; F, O).

This corpus kernel is the configuration `NUM_GROUPS = 6`, `BLOCK_SIZE = 2048`, `num_warps = 8`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

## Lengths and ranges

- `rf_ptr`, `ls_ptr`, `qty_ptr`, `price_ptr`, `disc_ptr`, `tax_ptr`, `shipdate_ptr`: columns of `lineitem`, `n_elements = l_quantity_t.numel()` rows each (i32 at scale factor 10).
- The six outputs: `torch.zeros(num_groups)` with `num_groups = NUM_GROUPS = 6`, zeroed by the plan and accumulated with `tl.atomic_add`: readwrite, 6 elements.
- `num_ls = len(ls_categories)`, a divisor of `NUM_GROUPS = 6`: in [1, 6]; `shipdate_hi` is any date.
- `rf_ptr` and `ls_ptr` hold category codes, which the kernel compares with the group numbers and does not use in an address: `read`.
