# q10_core_triton__agg_kernel_7

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q10/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_agg_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 7 of 8, `BLOCK_SIZE = 8192` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._agg_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`, only when `n > 0`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `custkey_ptr`: `n_elements` elements: `c_custkey` of each row of the joined frame `loc`; the plan sizes `out_ptr` from its least and greatest values, which the signature cannot state.
- `price_ptr`: `n_elements` elements: one element per joined row.
- `discount_ptr`: `n_elements` elements: one element per joined row.
- `out_ptr`: `torch.zeros(num_keys)` with `num_keys = max_key - min_key + 1`; the kernel adds to it with `tl.atomic_add` at `cust - min_key`.
- `min_key`: `int(c_custkey.min())`, below 2^31 at scale factor 10 (`i32`); the kernel subtracts it from each key.
- `n_elements`: `n = c_custkey.shape[0]`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers; the plan returns before the launch when it is 0, so its least value is 1.
- `num_keys`: `max_key - min_key + 1`, at least 1, the size symbol of `out_ptr`.
