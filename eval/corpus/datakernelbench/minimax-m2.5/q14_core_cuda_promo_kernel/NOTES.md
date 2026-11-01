# q14_core_cuda_promo_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q14/round_04.py`, copied as `plan.py`.

## Extraction

- The kernel `promo_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- The plan takes `uint8_t` from `#include <torch/extension.h>`, which nvcc cannot include without PyTorch's headers; the extraction includes `<cstdint>` in its place.
- Copied the macros that the kernel uses: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`promo_launcher` launches `promo_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `p`: `n` elements: `l_extendedprice` of the lineitem-part join, `n = P.numel()` rows.
- `d`: `n` elements: `l_discount`, one element per joined row.
- `sd`: `n` elements: `l_shipdate` as days since 1970-01-01, one element per joined row.
- `pm`: `n` elements: the promo flag as uint8, one element per joined row.
- `lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `tot`: `torch::zeros({1})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `pr`: `torch::zeros({1})`, as `tot`.
- `n`: `n = P.numel()`, the size symbol of the row buffers.
