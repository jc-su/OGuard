# q18_core_cuda_sorted_compact_segments_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `sorted_compact_segments_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `sorted_compact_segments_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q18_launcher` launches the kernel once, on the sorted path (no violation), with
`<<<line_blocks, BLOCK_SIZE, 0, stream>>>`, `line_blocks = (n + 1023) / 1024`: block
(1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `keys` (`int64`), `qty` (`float64`): `l_orderkey` and `l_quantity`, `n` rows each; read only.
- `unique_offsets`, `big_offsets`: the exclusive scans of the per-block counts, `line_blocks = (n + 1023) / 1024` elements each; the quotient gives the two buffers one size symbol (`unique_offsets`) with least value 1. Read only.
- `unique_out`, `big_keys_out` (`int64`), `big_sums_out` (`float64`): `torch::empty({n})` each, `n` elements, written only; the plan slices the first `totals` entries afterwards.
- `n`: size symbol with least value 1. `threshold`: a comparison threshold only.

## Branch

The plan takes this path when `sorted_count_segments_kernel` found `l_orderkey`
nondecreasing, which holds for the benchmark's lineitem table at scale factor 10.
