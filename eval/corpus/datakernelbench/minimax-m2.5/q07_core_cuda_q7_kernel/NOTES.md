# q07_core_cuda_q7_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q07/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `q7_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q7_launcher` launches `q7_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `supp`: `n` elements, the rows of the joined frame: the supplier nation code, which the plan builds as `(supp_nation == nation_b).astype(int32)`, 0 or 1; the signature cannot state the range.
- `cust`: `n` elements: the customer nation code, 0 or 1 in the same way.
- `shipdate`: `n` elements: `l_shipdate` as days since 1970-01-01, one element per joined row.
- `price`: `n` elements: one element per joined row.
- `discount`: `n` elements: one element per joined row.
- `date_lo`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_lo` parameter; any int32 value; not used in an address.
- `date_hi`: a date bound, days since 1970-01-01 that the plan computes from its `shipdate_hi` parameter; any int32 value; not used in an address.
- `revenue`: `torch::zeros({8})`, zeroed before the launch; the kernel adds to it with `atomicAdd` at `group = s * 4 + c * 2 + year_offset`, where the plan expects `s` and `c` in {0, 1} and, by its decoding of the groups as the years 1995 and 1996, the shipdate window inside those two years.
- `n`: `n = price.numel()`, the size symbol of the row buffers.
