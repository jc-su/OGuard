# q19_core_triton__q19_kernel_4

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q19/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_q19_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 4 of 8, `BLOCK_SIZE = 2048` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_q19` launches `_K._q19_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `qty_ptr`: `n_elements` elements: `l_quantity` of the lineitem-part join, `n = l_quantity.numel()` rows.
- `price_ptr`: `n_elements` elements: one element per joined row.
- `discount_ptr`: `n_elements` elements: one element per joined row.
- `brand_id_ptr`: `n_elements` elements: the brand code 12, 23, 34, or 0, one element per joined row.
- `container_grp_ptr`: `n_elements` elements: the container group 1, 2, 3, or 0, one element per joined row.
- `size_ptr`: `n_elements` elements: `p_size`, one element per joined row.
- `brand12_id`, `brand23_id`, `brand34_id`: the constants 12, 23, and 34; not used in an address.
- `sm_grp`, `md_grp`, `lg_grp`: the constants 1, 2, and 3; not used in an address.
- `out_ptr`: `torch.zeros(1)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n = l_quantity.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
