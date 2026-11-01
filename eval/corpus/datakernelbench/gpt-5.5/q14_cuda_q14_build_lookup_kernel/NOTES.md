# q14_cuda_q14_build_lookup_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q14/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q14_build_lookup_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q14_build_lookup_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and CUDA runtime
  includes, the macro `BLOCK_SIZE`, and the host code, which the kernel does
  not use.

## Launch

`q14_build_lookup`, called by `_get_promo_lookup` once per `part` table,
launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `part_keys`: `p_partkey` of `part`, `n` rows; `n = part_keys.numel()`.
- `promo_flags`: a `bool` tensor of `p_type LIKE 'PROMO%'`, `n` rows; the
  launcher checks `part_keys.numel() == promo_flags.numel()`.
- `n`: size symbol with least value 1; the launch happens only when `n > 0`.
- `lookup`: `torch::zeros({range})` of `uint8`, with
  `range = max_key - min_key + 1`, where `_get_promo_lookup` takes `min_key`
  and `max_key` as the min and max of `p_partkey`; the launcher checks
  `range > 0`. The length is a difference, which the signature cannot state,
  so it is the size symbol `lookup_len` with least value 1. The kernel only
  writes it.
- `min_key`: the min of `p_partkey`; the plan states no range of its own.
  The kernel writes `lookup[part_keys[idx] - min_key]` with no bound check,
  relying on this relation.
