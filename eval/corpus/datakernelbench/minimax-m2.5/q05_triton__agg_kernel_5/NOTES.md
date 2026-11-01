# q05_triton__agg_kernel_5

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q05/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_agg_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 5 of 8, `BLOCK_SIZE = 4096` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_agg` launches `_K._agg_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `nation_id_ptr`: `n_elements` elements, the `nation_id` of each row of `final_df`, which the plan assigns as `range(len(nations_unique))` and joins back, so each value lies in [0, num_nations): an index into `out_ptr`.
- `price_ptr`: `n_elements` elements: one element per row of `final_df`.
- `discount_ptr`: `n_elements` elements: one element per row of `final_df`.
- `out_ptr`: `torch.zeros(num_nations)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n = nation_ids.numel()`, the size symbol of the row buffers.
- `num_nations`: `len(nations_unique)`, the size symbol of `out_ptr`. A launch runs a program only when `n` is positive, and then the region has a nation, so its least value is 1.
