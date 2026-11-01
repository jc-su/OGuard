# q17_core_cuda_q17_accumulate_avg_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q17_accumulate_avg_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q17_accumulate_avg_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`, `#include <limits.h>`, the macro `EMPTY_KEY`, and the device functions `q17_hash_u64` and `q17_lookup_key`; left out `BLOCK_SIZE`, `<algorithm>` (host `std::min` and `std::max` only), the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q17_launcher` launches the kernel once, when `n_line > 0` and `n_parts > 0`, with
`<<<scan_blocks, BLOCK_SIZE, 0, stream>>>`, `scan_blocks = max(1, min((n_line + 1023) / 1024, 8192))`:
block (1024, 1, 1), no dynamic shared memory, a grid-stride loop.

## Lengths and ranges

- `l_partkey` (`int64`), `l_quantity` (`float64`): the lineitem columns, `n_line` rows each (checked by the launcher); `l_partkey` is hashed only, so `read`.
- `hash_keys`, `hash_vals`: the `hash_size` entries that `q17_build_hash_kernel` filled; read only here.
- `sum_qty`, `cnt_qty`: the `n_parts` accumulators that `q17_init_kernel` zeroed, added to with `atomicAdd` (`readwrite`); `n_parts` is a size symbol that the kernel does not receive, least value 1.
- `n_line`: size symbol with least value 1; `hash_size`: least value 2.
