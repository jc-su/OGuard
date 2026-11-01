# q18_core_cuda_scan_counts_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `scan_counts_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `scan_counts_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`exclusive_scan_counts` launches the kernel once per call with
`<<<scan_blocks, BLOCK_SIZE, 0, stream>>>`, `scan_blocks = (n + 1023) / 1024`: block
(1024, 1, 1), no dynamic shared memory. `q18_launcher` calls it for the unique
counts and for the big counts on the sorted path, and `dense_groupby_launcher` on
the dense path; every launch has this configuration, so it is one corpus kernel.

## Lengths and ranges

- `in`: the `n` per-block counts of the counting kernel; read only. `out`: `torch::empty({n})`, the exclusive scan, written only.
- `block_sums`: `torch::empty({scan_blocks})`, one element per block of the plan's grid, `(n + 1023) / 1024`; the quotient gives the buffer its own size symbol with least value 1. The kernel only writes it.
- `n`: `line_blocks` or `key_blocks`, at least 1: size symbol with least value 1.
