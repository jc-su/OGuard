# q22_core_cuda_final_group_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `final_group_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `final_group_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `MAX_CODE`; left out `<climits>`, `BLOCK_SIZE` (not used by the kernel), `<algorithm>`, `<vector>`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q22_launcher` launches the kernel once, when `n_customers > 0`, with
`<<<cust_blocks, BLOCK_SIZE, 0, stream>>>`, `cust_blocks = (n_customers + 255) / 256`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `c_custkey` (`int64`), `c_acctbal` (`float64`), `cntrycode` (`int32`), `in_mask` (`bool`): the customer columns, `n = n_customers` rows each (checked by the launcher).
- `has_order`: the `key_space` bytes that `mark_orders_kernel` wrote; read only. `key_space = max(0, max(c_custkey)) + 1` is sized from the maximum of the same `c_custkey` column, and TPC-H keys are positive, so `c_custkey` is `index-in` `has_order`.
- `avg_sum`, `avg_count`: the 1-element results of `avg_acctbal_kernel`; read at 0.
- `global_counts` (`int64`), `global_sums` (`float64`): `torch::zeros({MAX_CODE})` each, 100 elements, added to with `atomicAdd` (`readwrite`).
- `n`, `key_space`: size symbols with least value 1.
