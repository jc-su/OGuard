# q22_core_cuda_avg_acctbal_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `avg_acctbal_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `avg_acctbal_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<climits>`, `MAX_CODE`, `<algorithm>`, `<vector>`, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q22_launcher` launches the kernel once, when `n_customers > 0`, with
`<<<cust_blocks, BLOCK_SIZE, 0, stream>>>`, `cust_blocks = (n_customers + 255) / 256`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `c_acctbal` (`float64`), `in_mask` (`bool`): the customer columns, `n = n_customers` rows each (checked by the launcher).
- `avg_sum`: `torch::zeros({1})`, `avg_count`: `torch::zeros({1})` of `int64` (added through an `unsigned long long` cast), one element each (`readwrite`).
- `n`: size symbol with least value 1.
