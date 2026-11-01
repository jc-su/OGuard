# q02_cuda_q2_eq_i64_mask_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q02/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q2_eq_i64_mask_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q2_eq_i64_mask_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, and c10 includes, the
  macro `BLOCK_SIZE` (used only by the host code), the other kernels, and the
  host code.

## Launch

`q2_eq_i64_mask_launcher` launches the kernel when `n > 0` with
`<<<(n + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory. `_cuda_eq_i64_mask` calls it with `p_size` and `size_val`.

## Lengths and ranges

- `values`: `n = values.numel()` rows, `p_size` of part as int64.
- `out`: `torch::empty({n})` of `bool`, written only.
- `n`: size symbol with least value 1 (the launch needs `n > 0`).
- `target`: the size to match; comparison only.
