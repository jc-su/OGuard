# q20_core_cuda_aggregate_lineitem_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `aggregate_lineitem_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `aggregate_lineitem_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>` and the device functions `mix_u64` and `hash_pair_i64`; left out `hash_i64`, the macro `BLOCK_SIZE`, `<algorithm>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q20_launcher` launches the kernel once, when `nps`, `ns`, and `nli` are all positive,
with `<<<(nli + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `l_partkey`, `l_suppkey` (`int64`), `l_quantity` (`float64`): the date-filtered lineitem columns, `nli` rows each (checked by the launcher); the keys are hashed only, so `read`.
- `ht_part`, `ht_supp`, `ht_idx`: the `cap` entries that `insert_partsupp_kernel` filled; read only here.
- `sums`: `torch::zeros({nps})`, `nps` elements, added to with `atomicAdd` (`readwrite`); `nps` is a size symbol that the kernel does not receive, least value 1.
- `nli`: size symbol with least value 1; `cap`: a power of two of at least 16.
