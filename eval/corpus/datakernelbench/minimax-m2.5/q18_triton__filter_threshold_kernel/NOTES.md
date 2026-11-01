# q18_triton__filter_threshold_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q18/round_06.py`, copied as `plan.py`.

## Extraction

- The kernel `_filter_threshold_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 1 of 6, `BLOCK_SIZE = 1024` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 3), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._filter_threshold_kernel[grid](sums, float(qty_threshold), valid_mask, n)` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `sums_ptr`: `n_elements` elements: `sums`, one element per distinct order key.
- `threshold`: `float(qty_threshold)`, a Python float, which Triton passes as `fp32`; not used in an address.
- `out_ptr`: `n_elements` elements: `valid_mask = torch.zeros(n, dtype=torch.int32)`.
- `n_elements`: `n = sums.numel()`, the size symbol of both buffers.
