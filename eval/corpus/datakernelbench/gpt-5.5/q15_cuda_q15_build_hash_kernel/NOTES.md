# q15_cuda_q15_build_hash_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q15/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q15_build_hash_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_build_hash_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`, the macro `EMPTY_KEY`, and the device function
  `q15_hash_i64`; left out the Torch, ATen, c10, and CUDA runtime includes, `<vector>`, the host code, and the other macros and includes, which the kernel
  does not use.

## Launch

`q15_launcher` launches the kernel once, when `build_blocks > 0` (so
`n_sup >= 1`), with `<<<(n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `s_suppkey`: `s_suppkey` of `supplier`, `n_sup` rows; `n_sup = s_suppkey.numel()`.
- `n_sup`: size symbol with least value 1 (the launch needs `build_blocks > 0`).
- `hash_keys`: `torch::full({hash_size}, EMPTY_KEY)`, and `hash_vals`:
  `torch::empty({hash_size})`, where `hash_size` is the least power of two
  `>= 2 * n_sup` and `hash_mask = hash_size - 1`; both have `hash_mask + 1`
  entries. The kernel CASes `hash_keys` (`readwrite`) and only stores to
  `hash_vals` (`write`).
- `hash_mask`: size symbol with least value 1 (`n_sup >= 1` makes
  `hash_size >= 2`).
