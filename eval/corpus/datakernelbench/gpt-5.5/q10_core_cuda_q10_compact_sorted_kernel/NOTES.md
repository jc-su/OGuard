# q10_core_cuda_q10_compact_sorted_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q10/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q10_compact_sorted_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q10_compact_sorted_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, the Thrust and CUDA runtime includes, `<vector>`, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q10_groupby_launcher` launches the kernel once, when `unique_count > 0`, with
`<<<dense_blocks, BLOCK_SIZE, 0, stream>>>`, `dense_blocks = (dense_size + 1023) / 1024`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `seen` (`uint8`), `prefix` (`int64`, the exclusive scan of `seen` by Thrust), `revenue_dense` (`float64`): `dense_size` elements each; read only.
- `keys_out`, `revenue_out`: `torch::empty({unique_count})` each, `unique_count = prefix[dense_size - 1] + seen[dense_size - 1]` elements, the number of seen keys; `unique_count` is a size symbol that the kernel does not receive, with least value 1 at this launch. The kernel only writes them.
- `dense_size`: size symbol with least value 1.
