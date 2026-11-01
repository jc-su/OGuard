# q20_core_cuda_qualify_partsupp_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `qualify_partsupp_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `qualify_partsupp_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>` and the device functions `mix_u64` and `hash_i64`; left out `hash_pair_i64`, the macro `BLOCK_SIZE`, `<algorithm>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q20_launcher` launches the kernel once, when `nps`, `ns`, and `nli` are all positive,
with `<<<(nps + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `ps_suppkey`, `ps_availqty` (`int64`), `sums` (`float64`, the sums of `aggregate_lineitem_kernel`): `nps` elements each; read only.
- `qual_supp_set`: `torch::full({supp_cap}, -1)` of `int64`, `supp_cap = pair_cap` entries, probed with `atomicCAS` (`readwrite`).
- `supp_cap`: the smallest power of two of at least `max(16, 4 * nps)`: size symbol with least value 16. `nps`: least value 1.
