# q18_core_cuda_dense_accumulate_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `dense_accumulate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `dense_accumulate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, `<vector>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`dense_groupby_launcher` launches the kernel once with
`<<<line_blocks, BLOCK_SIZE, 0, stream>>>`, `line_blocks = (n + 1023) / 1024`: block
(1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `keys` (`int64`), `qty` (`float64`): `l_orderkey` and `l_quantity`, `n` rows each. `max_key_kernel` clamps keys below 1 to 0, so the plan does not bound negative keys by `key_range`: `keys` is `read`, and the kernel bounds each key itself.
- `sums`: `torch::zeros({key_range})`, added to with `atomicAdd`; `flags`: `torch::zeros({key_range})` of `int32`, set with `atomicExch` (both `readwrite`).
- `key_range`: `host_max + 1` with `host_max >= 0`: size symbol with least value 1. `n`: least value 1.

## Branch

The plan takes the dense path only when `sorted_count_segments_kernel` found a
decrease in `l_orderkey`, which the benchmark's lineitem table at scale factor 10
probably does not have.
