# q15_cuda_q15_reduce_max_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q15/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q15_reduce_max_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_reduce_max_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`, `#include <cfloat>` (for `DBL_MAX`), and the
  macro `BLOCK_SIZE`; left out the Torch, ATen, c10, and CUDA runtime includes, `<vector>`, the host code, and the other macros and device functions, which
  the kernel does not use.

## Launch

`q15_launcher` launches the kernel once, unconditionally, with
`<<<reduce_blocks, BLOCK_SIZE, 0, stream>>>`, where `reduce_blocks` is
`(n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE` clamped to [1, 4096]: block
(1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `totals`, `present`: the `n_sup` per-supplier sums and flags of
  `q15_aggregate_kernel`; read only.
- `n_sup`: `s_suppkey.numel()`, size symbol with least value 0 (the launch is
  unconditional).
- `block_max`: `torch::empty({reduce_blocks})`, one element per block of the
  plan's grid. The signature states the quotient `(n_sup + 1023) / 1024`,
  which gives the buffer its own size symbol; its least value is 1 (the plan
  clamps the count to at least 1). The kernel only writes it.
