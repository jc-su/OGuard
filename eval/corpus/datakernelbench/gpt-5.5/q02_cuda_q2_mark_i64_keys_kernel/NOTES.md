# q02_cuda_q2_mark_i64_keys_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q02/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q2_mark_i64_keys_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q2_mark_i64_keys_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, and c10 includes, the
  macro `BLOCK_SIZE` (used only by the host code), the other kernels, and the
  host code.

## Launch

`q2_build_bitmap_launcher` launches the kernel when `n > 0 && bitmap_size > 0`
with `<<<(n + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no
dynamic shared memory. `_cuda_membership_mask` calls it with the `s_suppkey`
of the European suppliers and `bitmap_size = max_supp + 1`.

## Lengths and ranges

- `keys`: `n_keys = keys.numel()` rows, the `s_suppkey` of the suppliers joined
  to the European nations. `max_supp` is the maximum of `s_suppkey` and
  `ps_suppkey`, so every key is at most `bitmap_size - 1`, and TPC-H keys are
  positive: `index-in` into `bitmap`.
- `bitmap`: `torch::zeros({bitmap_size})` bytes, which the kernel only stores 1
  into (`write`).
- `n_keys`, `bitmap_size`: size symbols with least value 1 (the launch needs
  both positive).
