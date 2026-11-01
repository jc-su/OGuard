# q18_core_cuda_add_block_offsets_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `add_block_offsets_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `add_block_offsets_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, `<vector>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`exclusive_scan_counts` launches the kernel once per call with
`<<<scan_blocks, BLOCK_SIZE, 0, stream>>>`, `scan_blocks = (n + 1023) / 1024`: block
(1024, 1, 1), no dynamic shared memory. Every call has this configuration, so it
is one corpus kernel.

## Lengths and ranges

- `offsets`: the `n` block-local exclusive scans of `scan_counts_kernel`, which the kernel adds to in place (`readwrite`).
- `block_offsets`: the `scan_blocks = (n + 1023) / 1024` scanned block sums of `scan_small_kernel`; the quotient gives the buffer its own size symbol with least value 1, derived from `n` as `(n + 1023) // 1024`, which the plan computes (corrected 2026-10-02; it was first an independent symbol). Read only.
- `n`: size symbol with least value 1.
