# q04_core_triton_mark_exists_kernel_6

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/triton/core/queries/q04/round_02.py`, copied as `plan.py`: `mark_exists_kernel` in `_KERNEL_SOURCE`, lines 24-40 of the plan file.

## Extraction edits

- `def mark_exists_kernel(` is named `def kernel(`.
- The `@triton.autotune` decorator (lines 14-22) is replaced by the values of configurations 11 and 12 of its list: `CONSTANTS` and `NUM_WARPS`.
- The module states `TYPES`, `CONSTANTS`, and `NUM_WARPS`; the imports `triton` and `triton.language as tl` are the plan's.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`_query_core` launches `_K.mark_exists_kernel[grid1](...)` with `grid1 = (triton.cdiv(n_line, BLOCK_SIZE),)`, when `n_line > 0`.

This corpus kernel is the configuration `BLOCK_SIZE = 4096`, `num_warps = 8`. The autotuner lists this pair with `num_stages` 2 and 4; the corpus module does not state `num_stages`, which sets the pipelining of `tl.range` loops that the kernel does not use, and its PTX is the same for `num_stages` 2, 3 (the front end's default), and 4, so the two configurations are one corpus kernel.

`triton.json` states the grid maxima (524288, 1, 1): `2^31 / BLOCK_SIZE` programs in x, where the 32-bit offset `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps, and 1 in y and z, which the kernel ignores.

## Lengths and ranges

- `l_okey_ptr`, `l_commit_ptr`, `l_receipt_ptr`: columns of `lineitem`, `n_elements = l_orderkey.numel()` rows each, at least 1 at the launch (i32 at scale factor 10).
- `exists_map_ptr`: `torch.zeros(max_key + 1)` with `max_key = max(o_orderkey.max(), l_orderkey.max())`, zeroed by the plan before the launch and set by the kernel: readwrite.
- `l_okey_ptr`: the plan sizes `exists_map_ptr` by the maximum of this column (and of `o_orderkey`), and the TPC-H order keys are positive, so every value lies in `[0, max_key]`: `index-in` into `exists_map_ptr`.
