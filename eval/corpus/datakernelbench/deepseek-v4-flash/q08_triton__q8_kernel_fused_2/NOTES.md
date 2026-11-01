# q08_triton__q8_kernel_fused_2

`triton.py` is the kernel `_q8_kernel_fused` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/triton/full/queries/q08/round_02.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `_q8_kernel_fused` is renamed `kernel`; the module keeps the plan's imports of `triton` and `triton.language`.
- The `@triton.autotune` decorator is replaced by the values of this configuration, `BLOCK_SIZE = 1024`, 8 warps. The autotuner lists 16 configurations: `BLOCK_SIZE` 1024, 2048, 4096, 8192, `num_warps` 4 and 8, `num_stages` 2 and 4. The module states `CONSTANTS` and `NUM_WARPS` and no `num_stages`, and the kernel has no loop that Triton pipelines (the front end's PTX is the same for `num_stages` 2, 4, and Triton's default), so the two configurations that differ only in `num_stages` are one corpus kernel. The eight kernels are the configurations in the autotuner's order, the first without a suffix and the others `_2` to `_8`; this is number 2.
- Added `TYPES`, the Triton type of each argument that is not a `tl.constexpr`, as the plan's values have them (a Python integer below 2^31 is `i32`, a Python float `fp32`).
- Left out: the host code.
- The kernel's body and parameters are the plan's.

## Launch

`_launch_q8` launches `_q8_kernel_fused[(cdiv(n, BLOCK_SIZE),)]` once, through the autotuner.

## Lengths and ranges

- `year_ptr` (read, length `n_elements`): the order year of each joined row; `n_elements` is `year.numel()`.
- `volume_ptr` (read, length `n_elements`): `price * (1 - discount)`, one per row.
- `is_brazil_ptr` (read, length `n_elements`): `(li["n_name"] == nation_name)`, a bool tensor.
- `out_brazil_ptr` (readwrite, length `num_groups`): `torch.zeros(num_groups)`; the kernel adds to it with `tl.atomic_add`.
- `out_total_ptr` (readwrite, length `num_groups`): `torch.zeros(num_groups)`; the kernel adds to it with `tl.atomic_add`.
- `n_elements` (size): `year.numel()`, below 2^31 at scale factor 10.
- `num_groups` (size): `len(torch.unique(year))`, the length of both outputs.
- `year_offset` (scalar): `int(unique_years.min())`; no range.

Least values of the size symbols: `n_elements` 0, `num_groups` 0. The plan launches on whatever rows its joins and filters give, which can be none.
