# q01_cuda_q1_kernel_generic

Source: `azure__gpt-5.5/cuda/full/queries/q01/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q1_kernel_generic` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q1_kernel_generic` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macros `BLOCK_SIZE` and `MAX_GROUPS` that
  the kernel uses; left out the Torch, ATen, c10, `<algorithm>`, and `<vector>`
  includes, the other macros and kernel, and the host code.

## Launch

`q1_launcher` launches the kernel in the `else` branch of `n_groups <= 6`
(plan.py line 299) with `<<<min((n + 511) / 512, 8192), BLOCK_SIZE, 0, stream>>>`:
block (512, 1, 1), no dynamic shared memory. At scale factor 10 `n_groups` is 6
and the plan takes the `q1_kernel6` branch; the launch is kept because the plan
states it.

## Lengths and ranges

- The seven lineitem columns: `n` rows each, as for `q1_kernel6`
  (`n = quantity.numel()`, the other six checked equal).
- `n`: size symbol with least value 1; the launcher returns before the launch
  when `n == 0`.
- `n_groups`: `n_rf * n_ls`, the length of each output. `TORCH_CHECK` bounds it
  by `MAX_GROUPS` (64), and the branch is taken only when it exceeds 6: least
  value 7, `size_maxima` 64 in `launch.json`.
- The six outputs: `torch::zeros({n_groups64})`, `n_groups` elements each, which
  the kernel adds to with `atomicAdd` (`readwrite`).
- `n_ls`: in [1, 64], since `n_groups = n_rf * n_ls` is in [7, 64].
- `returnflag`, `linestatus`: category codes; the kernel bounds the group id.
- `shipdate_hi_days`: comparison threshold only.
