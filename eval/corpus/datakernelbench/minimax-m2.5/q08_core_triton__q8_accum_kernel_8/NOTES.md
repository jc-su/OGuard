# q08_core_triton__q8_accum_kernel_8

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/triton/core/queries/q08/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `_q8_accum_kernel` is renamed `kernel`; its body and parameters are the plan's.
- The `@triton.autotune` decorator is replaced by the values of one configuration: configuration 8 of 8, `BLOCK_SIZE = 8192` and `num_warps = 8`.
- The autotuner's configurations also vary `num_stages` (2 and 4), which the front end does not take; the configurations that differ only in `num_stages` give this one kernel.
- Copied the module's imports `import triton` and `import triton.language as tl`.
- `TYPES` states the Triton type of each argument that is not a `tl.constexpr`, from the value that the plan passes.
- Left out the host code.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`_query_core` launches `_K._q8_accum_kernel[grid]` with `grid = (triton.cdiv(n_rows, BLOCK_SIZE),)`; the autotuner picks one of its configurations at run time.

## Lengths and ranges

- `year_ptr`: `n_rows` elements: the order year of each row of the joined frame.
- `volume_ptr`: `n_rows` elements: `l_extendedprice * (1 - l_discount)`, one element per joined row.
- `is_brazil_ptr`: `n_rows` elements: the nation flag, one element per joined row.
- `min_year`: `unique_years[0].item()`, the least order year; the kernel subtracts it from each year and tests the difference against [0, num_groups) before it indexes.
- `num_groups`: `unique_years.shape[0]`, the distinct order years, the size symbol of the accumulators; `unique_years[0]` raises on an empty input, so the launch has at least one group and one row.
- `sum_brazil_ptr`: `torch.zeros(num_groups)`; the kernel adds to it with `tl.atomic_add`.
- `sum_total_ptr`: `num_groups` elements, as `sum_brazil_ptr`.
- `n_rows`: `o_year.numel()`, below 2^31 at scale factor 10 (`i32`), the size symbol of the row buffers.
