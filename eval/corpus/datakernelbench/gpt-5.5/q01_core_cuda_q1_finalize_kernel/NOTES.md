# q01_core_cuda_q1_finalize_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q01/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q1_finalize_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q1_finalize_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macros `MAX_RF`, `MAX_LS`, and `MAX_GROUPS` (`MAX_LS` is used by the kernel); left out `BLOCK_SIZE`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q1_launcher` launches the kernel once, when `n_groups > 0`, with
`<<<(n_groups + 127) / 128, 128, 0, stream>>>`: block (128, 1, 1), no dynamic
shared memory.

## Lengths and ranges

- `group_ids`: `gids_dev`, the `n_groups` group ids `gid = rf * MAX_LS + ls` of the nonempty groups, which the host builds from `g_count` with `rf < 16` and `ls < 16`, so every value lies in `[0, 255]`: `index-in` `g_count`.
- `g_sum_qty`, `g_sum_base_price`, `g_sum_disc_price`, `g_sum_charge`, `g_sum_disc`, `g_count`: the 256-element accumulators of `q1_aggregate_kernel`; read only here.
- `out_rf`, `out_ls`, `out_count` (`int64`) and the seven `float64` outputs: `torch::empty({n_groups})` each, `n_groups` elements, which the kernel only writes.
- `n_groups`: the count of nonempty groups (at most 256), size symbol with least value 1; the launcher launches only when `n_groups > 0`.
