# q20_core_cuda_insert_partsupp_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `insert_partsupp_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `insert_partsupp_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>` and the device functions `mix_u64` and `hash_pair_i64`; left out `hash_i64`, the macro `BLOCK_SIZE`, `<algorithm>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q20_launcher` launches the kernel once, when `nps`, `ns`, and `nli` are all positive,
with `<<<(nps + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `ps_partkey`, `ps_suppkey`: the filtered partsupp keys, `nps` rows each (checked by the launcher); hashed only, so `read`.
- `ht_part`, `ht_supp`: `torch::empty({pair_cap})` of `int64`, written only; `ht_idx`: `torch::full({pair_cap}, -1)`, probed with `atomicCAS` (`readwrite`); `cap = pair_cap` entries each.
- `cap`: the smallest power of two of at least `max(16, 4 * nps)`: size symbol with least value 16. `nps`: least value 1.
