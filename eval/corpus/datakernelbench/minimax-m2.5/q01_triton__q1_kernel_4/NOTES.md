# q01_triton__q1_kernel_4

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q01/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_q1_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 4 of 8, `BLOCK_SIZE = 2048` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_run_fused_aggregation` launches `_K._q1_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `rf_ptr`: `n_elements` elements: `rf.shape[0]`, the `l_returnflag` category codes, one per lineitem row; the codes are in [0, 3), which the signature cannot state.
- `ls_ptr`: `n_elements` elements: the `l_linestatus` category codes, one per lineitem row, in [0, 2).
- `qty_ptr`: `n_elements` elements: one element per lineitem row.
- `price_ptr`: `n_elements` elements: one element per lineitem row.
- `discount_ptr`: `n_elements` elements: one element per lineitem row.
- `tax_ptr`: `n_elements` elements: one element per lineitem row.
- `shipdate_ptr`: `n_elements` elements: one element per lineitem row.
- `shipdate_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `sum_qty_ptr`: `torch.zeros(max_groups)` with `max_groups = 256 * 256`; the kernel adds to it with `tl.atomic_add`.
- `sum_base_price_ptr`: 65536 elements, as `sum_qty_ptr`.
- `sum_disc_price_ptr`: 65536 elements, as `sum_qty_ptr`.
- `sum_charge_ptr`: 65536 elements, as `sum_qty_ptr`.
- `sum_disc_ptr`: 65536 elements, as `sum_qty_ptr`.
- `count_ptr`: 65536 elements of int64, as `sum_qty_ptr`.
- `n_elements`: `n = rf.shape[0]`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
