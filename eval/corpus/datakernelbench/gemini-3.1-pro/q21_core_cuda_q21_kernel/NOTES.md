# q21_core_cuda_q21_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q21/round_01.py`, copied as `plan.py`: `q21_kernel` in `_CUDA_SOURCE`, lines 13-57 of the plan file.

## Extraction edits

- `__global__ void q21_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q21_launcher` launches `q21_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 512` and `blocks = (num_l1 + 511) / 512`, when `num_l1 > 0`.

`launch.json` states the block (512, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l1_orderkey`, `l1_late`, `supp_ids`: columns of `l1`, the lineitems of the nation's suppliers in orders of the status, `num_l1 = l1_orderkey.numel()` rows each, at least 1 at the launch.
- `lo_orderkeys`, `lo_distinct_supp`, `lo_late_distinct_supp`: columns of `agg`, one row per distinct order key, `num_lo = lo_orderkeys.numel()` rows each, possibly none.
- `counts`: `torch::zeros({num_suppliers})` of int64, passed as `unsigned long long int*`, zeroed by the plan and accumulated with `atomicAdd`: readwrite; `num_suppliers = len(supp_n)`, at least 1 when `l1` has a row.
- `supp_ids`: the row numbers `range(len(supp_n))` of the nation's suppliers, joined into `l1` by `s_suppkey`, so every value lies in `[0, num_suppliers)`, the length of `counts`: `index-in` into `counts`.
