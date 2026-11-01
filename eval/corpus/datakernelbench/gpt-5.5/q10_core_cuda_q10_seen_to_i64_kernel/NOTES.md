# q10_core_cuda_q10_seen_to_i64_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q10_seen_to_i64_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q10_seen_to_i64_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, the Thrust and CUDA runtime includes, `<vector>`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q10_groupby_launcher` launches the kernel once with
`<<<dense_blocks, BLOCK_SIZE, 0, stream>>>`, `dense_blocks = (dense_size + 1023) / 1024`
(checked to fit an `int`): block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `seen`: the `dense_size` bytes that `q10_dense_aggregate_kernel` marked; read only.
- `prefix`: `torch::empty({dense_size})` of `int64`, written only (Thrust scans it in place afterwards).
- `dense_size`: size symbol with least value 1.
