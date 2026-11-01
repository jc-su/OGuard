# q14_core_triton__promo_revenue_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q14/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_promo_revenue_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 1 of 8, `BLOCK_SIZE = 1024` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_promo_revenue` launches `_K._promo_revenue_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `shipdate_ptr`: `n_elements` elements: `l_shipdate` of the lineitem-part join as days since 1970-01-01, one element per joined row.
- `price_ptr`: `n_elements` elements: one element per joined row.
- `discount_ptr`: `n_elements` elements: one element per joined row.
- `promo_ptr`: `n_elements` elements: the promo flag, one element per joined row.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `out_total_ptr`: `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `out_promo_ptr`: `torch.zeros(1)`, as `out_total_ptr`.
- `n_elements`: `n = l_extendedprice.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
