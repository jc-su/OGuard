# q06_triton__q6_kernel_4

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q06/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_q6_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 4 of 8, `BLOCK_SIZE = 2048` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`run_query` launches `_K._q6_kernel[grid]` with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `price_ptr`: `n_elements` elements: `l_extendedprice`, one element per lineitem row.
- `discount_ptr`: `n_elements` elements: one element per lineitem row.
- `quantity_ptr`: `n_elements` elements: one element per lineitem row.
- `shipdate_ptr`: `n_elements` elements: one element per lineitem row.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `thresh_ptr`: `torch.tensor([discount_lo, discount_hi, quantity_hi])`, 3 elements.
- `out_ptr`: `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `l_extendedprice_t.numel()`, below 2^31 at scale factor 10, the size symbol of the row buffers.
