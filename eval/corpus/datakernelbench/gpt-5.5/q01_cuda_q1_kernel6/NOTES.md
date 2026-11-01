# q01_cuda_q1_kernel6

Source: `azure__gpt-5.5/cuda/full/queries/q01/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q1_kernel6` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q1_kernel6` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macros `BLOCK_SIZE`, `ADD_ROW_TO_GROUP`, and
  `REDUCE_AND_ATOMIC` that the kernel uses; left out the Torch, ATen, c10,
  `<algorithm>`, and `<vector>` includes, the macros `MAX_GROUPS` and
  `MAX_BLOCKS`, the other kernel, and the host code.

## Launch

`q1_launcher` launches the kernel when `n_groups <= 6` (plan.py line 278) with
`<<<min((n + 511) / 512, 8192), BLOCK_SIZE, 0, stream>>>`: block (512, 1, 1), no
dynamic shared memory. At scale factor 10, `l_returnflag` has 3 categories and
`l_linestatus` 2, so `n_groups` is 6 and the plan takes this branch.

## Lengths and ranges

- The seven lineitem columns (`returnflag`, `linestatus`, `quantity`,
  `extendedprice`, `discount`, `tax`, `shipdate_days`): `n` rows each; the
  launcher sets `n = quantity.numel()` and checks that the other six have the
  same `numel()`.
- `n`: size symbol with least value 1; the launcher returns before the launch
  when `n == 0`.
- `n_groups`: `n_rf * n_ls`, the length of each output. The launcher returns
  before the launch when it is 0, and this branch is taken only when it is at
  most 6: least value 1, `size_maxima` 6 in `launch.json`.
- `out_qty`, `out_price`, `out_disc_price`, `out_charge`, `out_disc`,
  `out_count`: `torch::zeros({n_groups64})`, `n_groups` elements each, which the
  kernel adds to with `atomicAdd` (`readwrite`).
- `n_ls`: the count of `l_linestatus` categories; `n_groups = n_rf * n_ls` is in
  [1, 6], so `n_ls` is in [1, 6].
- `returnflag`, `linestatus`: category codes; the kernel bounds the group id
  they form itself, so they have no index role.
- `shipdate_hi_days`: comparison threshold only.
