# q04_core_triton_count_priorities_kernel_3

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q04/round_02.py`, copied as `plan.py`: `count_priorities_kernel` in `_KERNEL_SOURCE`, lines 54-81 of the plan file.

## Extraction edits

- `def count_priorities_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 43-52) is replaced by the values of configurations 5 and 6 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- `NUM_PRIORITIES` is the `tl.constexpr` that the launch passes as `num_priorities`, the count of distinct `o_orderpriority` values, which TPC-H fixes at 5 ('1-URGENT' to '5-LOW'): `CONSTANTS` states 5.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K.count_priorities_kernel[grid2](..., NUM_PRIORITIES=num_priorities)` with `grid2 = (triton.cdiv(n_orders, BLOCK_SIZE),)`, when `n_orders > 0`, after `mark_exists_kernel`.

This corpus kernel is the configuration `BLOCK_SIZE = 2048`, `num_warps = 4`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

`triton.json` states the grid maxima (1048576, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `o_okey_ptr`, `o_odate_ptr`, `o_priority_ptr`: columns of `orders`, `n_elements = o_orderkey.numel()` rows each, at least 1 at the launch (i32 at scale factor 10); `o_priority_ptr` holds the category codes of `o_orderpriority`.
- `exists_map_ptr`: the `max_key + 1` flags of the previous launch; `max_key` is the argument of its length (size), below 2^31 at scale factor 10.
- `counts_out_ptr`: `torch.zeros(num_priorities)` with `num_priorities = 5`, zeroed by the plan and accumulated with `tl.atomic_add`: readwrite, 5 elements.
- `o_okey_ptr` is checked against `max_key` by the kernel: `read`.
