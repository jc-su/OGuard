# q20_core_cuda_select_supplier_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `select_supplier_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `select_supplier_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>` and the device functions `mix_u64` and `hash_i64`; left out `hash_pair_i64`, the macro `BLOCK_SIZE`, `<algorithm>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q20_launcher` launches the kernel once, when `nps`, `ns`, and `nli` are all positive,
with `<<<(ns + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory, `out_count = torch::zeros({1})`.

## Lengths and ranges

- `s_suppkey`, `s_row_idx`: the keys and row numbers of the suppliers of the nation, `ns` rows each (checked by the launcher); read only (the row numbers are stored, not used as indices).
- `qual_supp_set`: the `supp_cap` entries that `qualify_partsupp_kernel` filled; read only.
- `out_idx`: `torch::empty({ns})` of `int64`, `ns` elements, written only; the plan narrows it to `out_count` afterwards.
- `out_count`: `torch::zeros({1})` of `int64`, used as an atomic cursor through an `unsigned long long` cast (`readwrite`).
- `ns`: size symbol with least value 1; `supp_cap`: a power of two of at least 16.
