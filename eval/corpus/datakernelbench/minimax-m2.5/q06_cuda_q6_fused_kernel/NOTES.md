# q06_cuda_q6_fused_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q06/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `q6_fused_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros that the kernel uses: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q6_launcher` launches `q6_fused_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `price_ptr`: `n_elements` elements: `price.numel()`, one element per lineitem row.
- `discount_ptr`: `n_elements` elements: one element per lineitem row.
- `quantity_ptr`: `n_elements` elements: one element per lineitem row.
- `shipdate_ptr`: `n_elements` elements: one element per lineitem row.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `disc_lo`: the discount_lo parameter, 0.05; not used in an address.
- `disc_hi`: the discount_hi parameter, 0.07; not used in an address.
- `qty_hi`: the quantity_hi parameter, 24.0; not used in an address.
- `out_ptr`: `torch::zeros({1})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n_elements`: `n = price.numel()`, the size symbol of the row buffers.
