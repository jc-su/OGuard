# q14_cuda_q14_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q14/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q14_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q14_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch,
  ATen, c10, and CUDA runtime includes and the host code, which the kernel
  does not use.

## Launch

`q14_launcher` launches the kernel once, when `n > 0`, with
`<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `price`, `discount`, `shipdate_days`, `part_keys`: `n` rows each, the
  `lineitem` columns; `n = price.numel()` and the launcher checks the other
  three have the same `numel()`.
- `n`: size symbol with least value 1; the launch happens only when `n > 0`.
- `promo_lookup`: the `uint8` lookup that `q14_build_lookup_kernel` wrote.
- `lookup_n`: `promo_lookup.numel()`, the length of `promo_lookup`; size
  symbol with least value 1 (the lookup has `max_key - min_key + 1 > 0`
  entries).
- `min_key`: the min of `p_partkey`; no range.
- `date_lo`, `date_hi`: comparison thresholds only.
- `out`: `torch::zeros({2})` of `float64`, which the kernel adds to with
  `atomicAdd` (`readwrite`).
