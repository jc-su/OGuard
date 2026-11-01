# q19_core_cuda_q19_revenue_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q19/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q19_revenue_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q19_revenue_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch and c10 includes and the host code, which the kernel does not use.

## Launch

`q19_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>`: block (1024, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `l_quantity`, `l_extendedprice`, `l_discount` (`float64`), `p_brand_id`, `p_container_grp`, `p_size` (`int32`): the joined columns, `n = l_quantity.numel()` rows each (checked by the launcher).
- `n`: size symbol with least value 1; the launcher returns before the launch when `n == 0`.
- `out`: `torch::zeros({1})`, one element, added to with `atomicAdd` (`readwrite`).
- `brand12_id`, `brand23_id`, `brand34_id`, `sm_grp`, `md_grp`, `lg_grp`: the codes 12, 23, 34, 1, 2, 3 of `run_query`, compared only.
