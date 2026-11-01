# q10_triton__revenue_kernel_7

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q10/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_revenue_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 7 of 8, `BLOCK_SIZE = 8192` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_compute_revenue` launches `_K._revenue_kernel[grid]` with `grid = (triton.cdiv(n_rows, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `custkey_ptr`: `n_elements` elements, the `o_custkey` of each joined row. The plan sizes `revenue_ptr` from the greatest of these keys plus 1, and TPC-H customer keys are positive, so each value lies in [0, max_key): an index into `revenue_ptr`.
- `price_ptr`: `n_elements` elements: one element per joined row.
- `discount_ptr`: `n_elements` elements: one element per joined row.
- `revenue_ptr`: `torch.zeros(max_key)` with `max_key = int(custkey_tensor.max()) + 1`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n_rows = custkey_tensor.shape[0]`; `custkey_tensor.max()` fails on an empty tensor, so its least value is 1.
