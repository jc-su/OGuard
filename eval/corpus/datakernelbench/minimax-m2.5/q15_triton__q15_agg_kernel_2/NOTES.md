# q15_triton__q15_agg_kernel_2

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q15/round_08.py`, copied as `plan.py`.

## Extraction

- The kernel `_q15_agg_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 2 of 8, `BLOCK_SIZE = 1024` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_q15_agg` launches `_K._q15_agg_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `revenue_ptr`: `n_elements` elements: the revenue of each lineitem row that passes the date filter and has a supplier.
- `pos_ptr`: `n_elements` elements: `torch.searchsorted(sorted_s, suppkey_f)`, kept only where `pos < num_suppliers` and the key matches, so each value lies in [0, num_suppliers): an index into `out_ptr`.
- `out_ptr`: `torch.zeros(num_suppliers)` with `num_suppliers = s_suppkey_t.shape[0]`; the kernel adds to it with `tl.atomic_add`. The plan's `clamp(max=num_suppliers - 1)` and indexing need a supplier, so its least value is 1.
- `n_elements`: `n = revenue.shape[0]`, the size symbol of `revenue_ptr` and `pos_ptr`.
