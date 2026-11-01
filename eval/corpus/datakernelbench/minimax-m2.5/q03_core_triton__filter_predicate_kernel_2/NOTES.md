# q03_core_triton__filter_predicate_kernel_2

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q03/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `_filter_predicate_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 2 of 6, `BLOCK_SIZE = 1024` and `num_warps = 8`.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_triton_q3` launches `_K._filter_predicate_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `l_shipdate_ptr`: `n_elements` elements: `l_shipdate` of the joined frame `col` as days since 1970-01-01, one element per row.
- `o_orderdate_ptr`: `n_elements` elements: `o_orderdate` as days, one element per row of `col`.
- `shipdate_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `orderdate_hi`: a date bound, days since 1970-01-01 that the plan computes from its `orderdate_hi` parameter; any int32 value; not used in an address.
- `mask_ptr`: `torch.zeros(n, int32)`: `n_elements` elements, which the kernel writes.
- `n_elements`: `n = l_orderkey.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of every buffer.
