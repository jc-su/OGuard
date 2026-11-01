# q20_cuda_build_ps_hash_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `build_ps_hash_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `build_ps_hash_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the macro `EMPTY_KEY`, and the device functions
  `mix64` and `make_pair_key_i64`; left out the Torch, ATen, c10, and
  `cuda_runtime.h` includes, the macro `BLOCK_SIZE` (not used by the kernel),
  and the host code.

## Launch

`q20_launcher` launches the kernel once with
`<<<(nps + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory. Before it, `cudaMemsetAsync` fills `table_keys` with 0xff bytes
(every slot `EMPTY_KEY`).

## Lengths and ranges

- `ps_partkey`, `ps_suppkey`: `nps` rows each, the partsupp rows whose part
  name starts with the prefix; the launcher checks `ps_suppkey.numel() == nps`.
- `nps`: size symbol with least value 1; the launcher returns before any launch
  when `nps == 0` (and `run_query` returns earlier when the filtered partsupp is
  empty).
- `table_keys`, `table_vals`: `torch::empty({hash_size})` of int64, passed as
  `uint64_t*` and `int64_t*`, with `hash_size = max(1024, next power of two of
  4 * nps)` and `table_mask = hash_size - 1`. The length is `table_mask + 1`,
  with `table_mask` a size symbol of least value 1023. The kernel CASes
  `table_keys` (`readwrite`) and only stores `table_vals` (`write`).
- `key_base`: `max_suppkey + 1`, where the launcher returns for
  `max_suppkey < 0`, so at least 1; it only forms the pair key.
