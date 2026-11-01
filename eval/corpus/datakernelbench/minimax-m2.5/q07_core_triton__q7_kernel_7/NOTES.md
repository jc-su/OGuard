# q07_core_triton__q7_kernel_7

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q07/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_q7_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 7 of 8, `BLOCK_SIZE = 8192` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_q7` launches `_K._q7_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `supp_ptr`: `n_elements` elements, the rows of the joined frame: the supplier nation code, which the plan builds as `(supp_nation == nation_b).astype(int32)`, 0 or 1; the signature cannot state the range.
- `cust_ptr`: `n_elements` elements: the customer nation code, 0 or 1 in the same way.
- `shipdate_ptr`: `n_elements` elements: `l_shipdate` as days since 1970-01-01, one element per joined row.
- `price_ptr`: `n_elements` elements: one element per joined row.
- `discount_ptr`: `n_elements` elements: one element per joined row.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `out_ptr`: `torch.zeros(8)`; the kernel adds to it with `tl.atomic_add` at `key = supp * 4 + cust * 2 + year_offset`, where the plan expects `supp` and `cust` in {0, 1} and, by its decoding of the groups as the years 1995 and 1996, the shipdate window inside those two years.
- `n_elements`: `n = supp.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
