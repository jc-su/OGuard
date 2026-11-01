# q17_core_cuda_q17_build_hash_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q17_build_hash_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q17_build_hash_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`, `#include <limits.h>`, the macro `EMPTY_KEY`, and the device function `q17_hash_u64`; left out `BLOCK_SIZE` and `q17_lookup_key`, `<algorithm>` (host `std::min` and `std::max` only), the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q17_launcher` launches the kernel once, when `n_line > 0` and `n_parts > 0`, with
`<<<build_blocks, BLOCK_SIZE, 0, stream>>>`, `build_blocks = max(1, min((n_parts + 1023) / 1024, 8192))`:
block (1024, 1, 1), no dynamic shared memory, a grid-stride loop.

## Lengths and ranges

- `part_keys`: the filtered `p_partkey` values, `n_parts` rows; hashed only, so `read`.
- `hash_keys`: the `hash_size` entries that `q17_init_kernel` set to `EMPTY_KEY`, probed with `atomicCAS` (`readwrite`); `hash_vals`: `torch::empty({hash_size})`, written only.
- `n_parts`: size symbol with least value 1; `hash_size`: a power of two of at least `4 * n_parts` and at least 2, least value 2.
