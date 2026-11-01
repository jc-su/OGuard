# q22_core_cuda_ker_grp

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q22/round_03.py`, copied as `plan.py`.

## Extraction

- The kernel `ker_grp` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros and device functions that the kernel uses: `#define BLOCK_SIZE 1024` (which the kernel does not use, kept beside `MAX_CODE`), `#define MAX_CODE 100`, and `exists_key`.
- Left out the other kernels of the source: `ker_avg`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`cuda_grp` launches `ker_grp<<<(ck.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and no dynamic shared memory, after `ker_avg`; `launch.json` states the block.

## Lengths and ranges

- `ck`: `n` elements: `c_custkey`, one element per customer row, `n = ck.numel()`.
- `bal`: `n` elements: `c_acctbal`, one element per customer row.
- `cd`: `n` elements: the country code, the first two characters of `c_phone` as int32, one element per customer row.
- `ms`: `n` elements: the country-code membership mask, one element per customer row.
- `ok`: `no` elements: the sorted distinct `o_custkey` values.
- `no`: `ok.numel()`, the size symbol of `ok`.
- `av`: the average balance; not used in an address.
- `cnts`: `torch::zeros({MAX_CODE}, kInt64)`: 100 elements, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `sums`: `torch::zeros({MAX_CODE})`: 100 elements, as `cnts`.
- `n`: `ck.numel()`, the size symbol of the row buffers.
