# q10_cuda_q10_aggregate_revenue_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q10_aggregate_revenue_kernel` of
`_CUDA_SOURCE`.

## Extraction edits

- Renamed `q10_aggregate_revenue_kernel` to `kernel` and declared it
  `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and
  `cuda_runtime.h` includes, the macro `BLOCK_SIZE`, and the host code, which
  the kernel does not use.

## Launch

`q10_aggregate_revenue_launcher` launches the kernel once, after returning
when `n_rows == 0 || out_size == 0`, with
`<<<(n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `custkey_ptr`, `price_ptr`, `discount_ptr`: `n_rows` rows each, the columns
  of the lineitem and orders join. The launcher checks equal `numel()` and sets
  `n_rows = price.numel()`.
- `n_rows`: size symbol with least value 1 (the launcher returns for 0).
- `out_revenue_ptr`: `torch::zeros({out_size})`, which the kernel adds to with
  `atomicAdd` (`readwrite`); `out_size = max c_custkey + 1` is its size symbol,
  least value 1 (the launcher returns for 0).
- `custkey_ptr` holds `o_custkey` of the join; the plan sizes the output from
  `c_custkey`, another table, and the kernel checks `0 <= ck < out_size`
  itself, so it is `read`.
