# q02_cuda_q2_membership_mask_i64_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q02/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q2_membership_mask_i64_kernel` of
`_CUDA_SOURCE`.

## Extraction edits

- Renamed `q2_membership_mask_i64_kernel` to `kernel` and declared it
  `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, and c10 includes, the
  macro `BLOCK_SIZE` (used only by the host code), the other kernels, and the
  host code.

## Launch

`q2_membership_mask_launcher` launches the kernel when `n > 0` with
`<<<(n + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory. `_cuda_membership_mask` calls it with `ps_suppkey` and the
bitmap that `q2_mark_i64_keys_kernel` built.

## Lengths and ranges

- `keys`: `n = keys.numel()` rows, `ps_suppkey` of partsupp. The bitmap has
  `max(s_suppkey, ps_suppkey) + 1` bytes, so every key is at most
  `bitmap_size - 1`, and TPC-H keys are positive: `index-in` into `bitmap`.
- `bitmap`: `bitmap_size = bitmap.numel()` bytes, read only.
- `out`: `torch::empty({n})` of `bool`, written only.
- `n`: size symbol with least value 1 (the launch needs `n > 0`).
- `bitmap_size`: size symbol with least value 1. The plan returns before the
  call when no European supplier exists, so `max_supp` is a positive key.
