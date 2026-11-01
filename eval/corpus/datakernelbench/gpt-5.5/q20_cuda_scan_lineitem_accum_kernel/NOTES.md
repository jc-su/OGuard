# q20_cuda_scan_lineitem_accum_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q20/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `scan_lineitem_accum_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `scan_lineitem_accum_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`, the macro `EMPTY_KEY`, and the device functions
  `mix64` and `make_pair_key_i64`; left out the Torch, ATen, c10, and
  `cuda_runtime.h` includes, the macro `BLOCK_SIZE` (not used by the kernel),
  and the host code.

## Launch

`q20_launcher` launches the kernel once, when `nli > 0`, with
`<<<(nli + 255) / 256, BLOCK_SIZE, 0, stream>>>`: block (256, 1, 1), no dynamic
shared memory, after `build_ps_hash_kernel` on the same stream.

## Lengths and ranges

- `l_partkey`, `l_suppkey`, `l_quantity`, `l_shipdate_days`: `nli` rows
  each, the columns of `lineitem`; the launcher checks the three lengths
  against `nli = l_partkey.numel()`.
- `nli`: size symbol with least value 1 (the launch is inside `if (nli > 0)`).
- `table_keys`, `table_vals`: the hash table of `build_ps_hash_kernel`,
  `table_mask + 1` entries each, `table_mask` a size symbol of least value
  1023. The kernel reads both. `table_vals` holds the partsupp row ids that the
  build kernel stored, and uninitialized values in empty slots (`torch::empty`);
  the plan does not bound them as a whole, so its role is `read`, not an index.
- `ps_sum_qty`: `torch::zeros({nps})`, `nps` a size symbol of least value 1
  (the launcher returns for `nps == 0`); the kernel adds to it with `atomicAdd`
  (`readwrite`).
- `key_base`: `max_suppkey + 1`, at least 1; `date_lo`, `date_hi`: epoch
  days, compared only.
