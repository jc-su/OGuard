# q21_core_cuda_q21_count_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q21/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q21_count_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q21_count_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q21_count_launcher` launches the kernel once, when `n_l1`, `n_orders`, and
`num_suppliers` are all positive, with `<<<min((n_l1 + 255) / 256, 65535), BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory, a grid-stride loop.

## Lengths and ranges

- `l1_orderkey` (`int64`), `l1_late` (`bool`), `supp_ids` (`int64`): the `l1` rows, `n_l1` each (checked by the launcher).
- `supp_ids`: `supp_id = range(len(supp_n))` by construction and `num_suppliers = len(supp_n)`, so every value lies in `[0, num_suppliers)`: `index-in` `counts`.
- `lo_orderkeys`, `lo_distinct_supp`, `lo_late_distinct_supp`: the per-order aggregate, `n_orders` rows each (checked by the launcher); `lo_orderkeys` is sorted, which the review does not rely on.
- `counts`: `torch::zeros({num_suppliers})`, added to with `atomicAdd` (`readwrite`).
- `n_l1`, `n_orders`, `num_suppliers`: size symbols with least value 1; the launch runs only when all three are positive.
