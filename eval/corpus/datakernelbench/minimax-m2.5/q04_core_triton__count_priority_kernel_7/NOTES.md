# q04_core_triton__count_priority_kernel_7

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q04/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_count_priority_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 7 of 8, `BLOCK_SIZE = 8192` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_count_priority` launches `_K._count_priority_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `priority_ptr`: `n_elements` elements: `final_priority`, the priority category codes of the qualifying orders, in [0, num_priorities), which the kernel compares and does not use in an address.
- `out_ptr`: `torch.zeros(num_priorities, int64)`; the kernel adds to it with `tl.atomic_add` at `out_ptr + p` for `p` in `range(num_priorities)`.
- `n_elements`: `n = priority.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of `priority_ptr`.
- `num_priorities`: `len(priority_categories)`, the distinct order priorities (5 in TPC-H), the size symbol of `out_ptr`.
