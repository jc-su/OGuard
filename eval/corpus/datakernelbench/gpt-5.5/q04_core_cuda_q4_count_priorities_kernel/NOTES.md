# q04_core_cuda_q4_count_priorities_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q04/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q4_count_priorities_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q4_count_priorities_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out `<algorithm>`, `<vector>`, the macro `Q4_BLOCK_SIZE`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q4_launcher` launches the kernel once, when `n_orders > 0` and `n_lineitem > 0`,
with `<<<(n_orders + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE, Q4_BLOCK_SIZE, shared_bytes, stream>>>`,
`shared_bytes = num_priorities * sizeof(unsigned long long)`: block (256, 1, 1),
dynamic shared memory `8 * num_priorities` bytes (`launch.json` terms
`{"num_priorities": 8}`).

## Lengths and ranges

- `o_orderkey` (`int64`), `o_orderdate_days`, `o_orderpriority_codes` (`int32`): the orders columns, `n_orders = o_orderkey.numel()` rows each.
- `has_late_lineitem`: the `max_key + 1` bytes that `q4_mark_late_orders_kernel` wrote; read only here. `max_key` is the maximum of `o_orderkey` and `l_orderkey`, checked `>= 0` (least value 0), so `o_orderkey` is `index-in` `has_late_lineitem`.
- `o_orderpriority_codes`: cuDF category codes of `o_orderpriority`, in `[0, num_priorities)` by construction: `index-in` `counts`.
- `counts`: `torch::zeros({num_priorities})` of `int64`, `num_priorities` elements, which the kernel adds to with `atomicAdd` (`readwrite`).
- `num_priorities`: the number of categories (5 at scale factor 10), checked `>= 0` and returned on before the launch when 0: size symbol with least value 1. The launch requests `8 * num_priorities` bytes of dynamic shared memory and fails (`C10_CUDA_KERNEL_LAUNCH_CHECK`) above the 48 KiB that a launch may request without opting in, so the kernel runs only for `num_priorities <= 6144` (`size_maxima`).
- `n_orders`: size symbol with least value 1; the launcher returns before the launch when it is 0.
- `date_lo_days`, `date_hi_days`: comparison thresholds only.
