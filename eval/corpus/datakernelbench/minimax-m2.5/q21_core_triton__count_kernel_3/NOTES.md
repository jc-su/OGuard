# q21_core_triton__count_kernel_3

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q21/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `_count_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 3 of 8, `BLOCK_SIZE = 2048` and `num_warps = 4`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._count_kernel[grid]` with `grid = (triton.cdiv(n_rows, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `supp_ids_ptr`: `n_elements` elements, the rows of `l1`: `supp_id`, which the plan assigns as `range(len(supp_n))` and joins to `l1` on `s_suppkey`, so every value lies in [0, num_suppliers): an index into `out_ptr`.
- `mask_ptr`: `n_elements` elements: the Q21 predicate of each `l1` row.
- `out_ptr`: `torch.zeros(num_suppliers, int64)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements`: `n_rows = supp_ids.shape[0]`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
- `num_suppliers`: `len(supp_n)`, the size symbol of `out_ptr`. A launch runs a program only when `n_rows` is positive, and then `l1` has a supplier of the nation, so its least value is 1.
