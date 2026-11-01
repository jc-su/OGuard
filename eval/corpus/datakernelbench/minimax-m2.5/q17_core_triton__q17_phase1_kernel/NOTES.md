# q17_core_triton__q17_phase1_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q17/round_05.py`, copied as `plan.py`.

## Extraction

- The kernel `_q17_phase1_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 1 of 8, `BLOCK_SIZE = 1024` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- Left out the other kernels of the module: `_q17_phase2_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._q17_phase1_kernel[grid]` with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`, only when `n_parts > 0`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `part_idx_ptr`: `n_elements` elements: `idx_clamped`, the `searchsorted` position of each lineitem's part key in `sorted_keys` clamped to `n_parts - 1`, so every value lies in [0, n_parts): an index into `sum_qty_ptr`.
- `mask_ptr`: `n_elements` elements: the part-membership flag as uint8, one element per lineitem row.
- `qty_ptr`: `n_elements` elements: `l_quantity`, one element per lineitem row.
- `sum_qty_ptr`: `torch.zeros(n_parts)`; the kernel adds to it with `tl.atomic_add`.
- `cnt_qty_ptr`: `n_parts` elements, as `sum_qty_ptr`.
- `n_elements`: `l_partkey.numel()`, the lineitem rows, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers; the plan states it with the annotation `int`, which the extraction keeps.
- `n_parts`: `sorted_keys.numel()`, the filtered parts, the size symbol of the accumulators; the plan returns before the launch when it is 0, so its least value is 1.
