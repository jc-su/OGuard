# q21_cuda_q21_set_supplier_map_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q21/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q21_set_supplier_map_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q21_set_supplier_map_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`; left out the Torch, ATen, and c10 includes, the
  macro `BLOCK_SIZE` (not used by the kernel), and the host code.

## Launch

`q21_grouped_launcher` launches the kernel once, when `n_supp > 0`, with
`<<<(n_supp + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no
dynamic shared memory.

## Lengths and ranges

- `supp_keys`, `supp_ids`: `n` rows each, the `s_suppkey` of the suppliers of
  the nation, sorted by name, and their positions `0 .. n - 1` as int32.
- `n`: `supp_keys.numel()`, size symbol with least value 1 (the launch is inside
  `if (n_supp > 0)`, and `run_query` returns earlier when there are none).
- `max_suppkey`: `int(df_supplier["s_suppkey"].max())`, size symbol with least
  value 0.
- `supp_map`: `torch::full({max_suppkey + 1}, -1)` of int32, which the kernel
  only stores into (`write`).
