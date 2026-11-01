# q12_triton__q12_kernel_7

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q12/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_q12_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 7 of 8, `BLOCK_SIZE = 8192` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_q12` launches `_K._q12_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `shipmode_ptr`: `n_elements` elements: the ship mode code, 0 or 1, one element per joined row.
- `priority_ptr`: `n_elements` elements: the high-priority flag, 0 or 1, one element per joined row.
- `commit_ptr`: `n_elements` elements: one element per joined row.
- `receipt_ptr`: `n_elements` elements: one element per joined row.
- `ship_ptr`: `n_elements` elements: one element per joined row.
- `receipt_lo`: a date bound, days since 1970-01-01 that the plan computes from its `receipt_lo` parameter; any int32 value; not used in an address.
- `receipt_hi`: a date bound, days since 1970-01-01 that the plan computes from its `receipt_hi` parameter; any int32 value; not used in an address.
- `high_ptr`: `torch.zeros(2, dtype=torch.int64)`; the kernel adds to elements 0 and 1 with `tl.atomic_add`.
- `low_ptr`: `torch.zeros(2, dtype=torch.int64)`, as `high_ptr`.
- `n_elements`: `n = shipmode_t.shape[0]`; `_launch_q12` returns before the launch when it is 0, so its least value is 1.
