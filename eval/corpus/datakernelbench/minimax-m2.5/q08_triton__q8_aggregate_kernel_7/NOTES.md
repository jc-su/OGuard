# q08_triton__q8_aggregate_kernel_7

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/full/queries/q08/round_07.py`, copied as `plan.py`.

## Extraction

- The kernel `_q8_aggregate_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 7 of 8, `BLOCK_SIZE = 8192` and `num_warps = 4` (`CONSTANTS` also holds `min_year = 1995`).
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- `min_year` is a `tl.constexpr` that the plan sets to `int(o_year.min())`; `CONSTANTS` holds 1995, the least order year of the rows that Q8 selects (order dates from 1995-01-01 to 1996-12-31).
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_launch_q8_aggregate` launches `_K._q8_aggregate_kernel[grid]` with `grid = (triton.cdiv(n, BLOCK_SIZE),)` and `min_year` as a `constexpr`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `year_ptr`: `n_elements` elements: the order year, one element per joined row.
- `volume_ptr`: `n_elements` elements: one element per joined row.
- `is_brazil_ptr`: `n_elements` elements: one element per joined row.
- `sum_total_ptr`: `torch.zeros(2)`; the kernel adds to elements 0 and 1 with `tl.atomic_add`.
- `sum_brazil_ptr`: `torch.zeros(2)`, as `sum_total_ptr`.
- `n_elements`: `n = o_year.numel()`; `o_year.min()` before the launch fails on an empty tensor, so its least value is 1.
