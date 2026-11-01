# q11_triton__q11_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/full/queries/q11/round_02.py`, copied as `plan.py`: `_q11_kernel` in `_KERNEL_SOURCE`, lines 24-52 of the plan file.

## Extraction edits

- `def _q11_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-23) is replaced by the values of configuration 1 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`run_query` launches `_K._q11_kernel[grid](...)` with `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`; it returns before the launch when `partsupp` is empty.

This corpus kernel is the configuration `BLOCK_SIZE = 1024`, `num_warps = 4`, `num_stages = 2`. The corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4.

## Lengths and ranges

- `ps_suppkey_ptr`, `ps_partkey_ptr`, `ps_supplycost_ptr`, `ps_availqty_ptr`: columns of `partsupp`, `n_elements = suppkey_t.numel()` rows each, at least 1 (i32 at scale factor 10).
- `is_nation_supp_ptr`: `torch.zeros(max_suppkey + 1)` of int32 with `max_suppkey` at least `ps_suppkey.max()`.
- `out_sums_ptr`: `torch.zeros(max_partkey + 1)` with `max_partkey = ps_partkey.max()`, zeroed and accumulated with `tl.atomic_add`: readwrite.
- `ps_suppkey_ptr` and `ps_partkey_ptr`: the plan sizes `is_nation_supp_ptr` and `out_sums_ptr` by these columns' maxima, and the TPC-H keys are positive: `index-in`.
