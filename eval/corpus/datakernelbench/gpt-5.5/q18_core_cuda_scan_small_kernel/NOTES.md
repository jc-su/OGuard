# q18_core_cuda_scan_small_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q18/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `scan_small_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `scan_small_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`exclusive_scan_counts` launches the kernel once per call with
`<<<1, BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1), no dynamic shared memory. Every
call (two on the sorted path, two on the dense path) has this configuration, so it
is one corpus kernel.

## Lengths and ranges

- `in`: the `n = scan_blocks` block sums of `scan_counts_kernel`; read only. `out`: `torch::empty({n})`, written only.
- `n`: `scan_blocks`, size symbol with least value 1.
