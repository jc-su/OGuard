# q05_core_triton__agg_revenue_kernel_2

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q05/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_agg_revenue_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 2 of 8, `BLOCK_SIZE = 1024` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_agg_revenue` launches `_K._agg_revenue_kernel[grid]` with `grid = (triton.cdiv(n_rows, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `nation_id_ptr`: `n_elements` elements, the `nation_id` of each row of `final_df`, which the plan assigns as `range(len(nations_unique))` and joins back, so each value lies in [0, num_nations): an index into `out_ptr`.
- `price_ptr`: `n_elements` elements: one element per row of `final_df`.
- `discount_ptr`: `n_elements` elements: one element per row of `final_df`.
- `out_ptr`: `torch.zeros(num_nations)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n_rows = nation_ids.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
- `num_nations`: `len(nations_unique)`, the size symbol of `out_ptr`. A launch runs a program only when `n_rows` is positive, and then the region has a nation, so its least value is 1.
