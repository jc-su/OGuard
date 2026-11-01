# q18_core_cuda_dense_compact_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `dense_compact_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `dense_compact_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`dense_groupby_launcher` launches the kernel once with
`<<<key_blocks, BLOCK_SIZE, 0, stream>>>`, `key_blocks = (key_range + 1023) / 1024`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `flags` (`int32`), `sums` (`float64`): the `key_range` entries of `dense_accumulate_kernel`; read only.
- `unique_offsets`, `big_offsets`: the exclusive scans of the per-block counts, `key_blocks = (key_range + 1023) / 1024` elements each; the quotient gives the two buffers one size symbol (`unique_offsets`) with least value 1. Read only.
- `unique_out`, `big_keys_out` (`int64`), `big_sums_out` (`float64`): `torch::empty({n})` each, `n` the lineitem row count, a size symbol that the kernel does not receive, least value 1. Written only.
- `key_range`: size symbol with least value 1. `threshold`: a comparison threshold only.

## Branch

The plan takes the dense path only when `sorted_count_segments_kernel` found a
decrease in `l_orderkey`, which the benchmark's lineitem table at scale factor 10
probably does not have.
