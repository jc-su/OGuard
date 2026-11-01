# q15_cuda_q15_aggregate_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q15/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q15_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`, the macro `EMPTY_KEY`, and the device functions
  `q15_hash_i64` and `q15_lookup_supplier`; left out the Torch, ATen, c10, and CUDA runtime includes, `<vector>`, the host code, and the other macros and
  includes, which the kernel does not use.

## Launch

`q15_launcher` launches the kernel once, when `agg_blocks > 0` (so
`n_line >= 1`), with `<<<(n_line + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `l_suppkey`, `l_extendedprice`, `l_discount`, `l_shipdate_days`: `n_line`
  rows each; `n_line = l_suppkey.numel()` and the launcher checks the other
  three have the same `numel()`.
- `n_line`: size symbol with least value 1.
- `hash_keys`, `hash_vals`: the tables of `q15_build_hash_kernel`,
  `hash_mask + 1` entries each; read only. `hash_vals` is `torch::empty`,
  and the build kernel stores a supplier row id below `n_sup` in each slot it
  fills; the plan relies on this, which the signature cannot state (a slot
  without a store holds any value).
- `hash_mask`: the least power of two `>= 2 * n_sup`, minus one; size symbol
  with least value 0 (the launch does not require `n_sup > 0`).
- `totals`: `torch::zeros({n_sup})` of `float64`, which the kernel adds to
  with `atomicAdd` (`readwrite`); `present`: `torch::zeros({n_sup})` of
  `uint8`, which the kernel only stores 1 into (`write`). `n_sup` is
  `s_suppkey.numel()`, not an argument of this kernel; size symbol with
  least value 0.
- `shipdate_lo_days`, `shipdate_hi_days`: comparison thresholds only.
