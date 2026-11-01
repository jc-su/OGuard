# q19_cuda_q19_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q19/round_02.py`, copied as `plan.py`.

## Extraction

- The kernel `q19_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Kept the includes that the kernel uses: `#include <cstdint>`.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`.
- Copied the macros that the kernel uses: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q19_launcher` launches `q19_kernel<<<blocks, BLOCK_SIZE>>>` with `blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE`, `BLOCK_SIZE = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `l_quantity`: `n` elements: `l_quantity.numel()`, one element per joined row.
- `l_extendedprice`: `n` elements: one element per joined row.
- `l_discount`: `n` elements: one element per joined row.
- `p_brand_id`: `n` elements: one element per joined row.
- `p_container_grp`: `n` elements: one element per joined row.
- `p_size`: `n` elements: one element per joined row.
- `out`: `torch::zeros({1})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n`: `n = l_quantity.numel()`, the size symbol of the row buffers.
- `brand12_id`: the plan passes the constant 12.
- `brand23_id`: the plan passes the constant 23.
- `brand34_id`: the plan passes the constant 34.
- `sm_grp`: the plan passes the constant 1.
- `md_grp`: the plan passes the constant 2.
- `lg_grp`: the plan passes the constant 3.
- `q1_lo`: the constant 1.0; not used in an address.
- `q1_hi`: the constant 11.0; not used in an address.
- `q2_lo`: the constant 10.0; not used in an address.
- `q2_hi`: the constant 20.0; not used in an address.
- `q3_lo`: the constant 20.0; not used in an address.
- `q3_hi`: the constant 30.0; not used in an address.
- `s1_lo`: the plan passes the constant 1.
- `s1_hi`: the plan passes the constant 5.
- `s2_lo`: the plan passes the constant 1.
- `s2_hi`: the plan passes the constant 10.
- `s3_lo`: the plan passes the constant 1.
- `s3_hi`: the plan passes the constant 15.
