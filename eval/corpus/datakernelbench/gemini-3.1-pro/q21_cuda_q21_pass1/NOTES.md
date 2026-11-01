# q21_cuda_q21_pass1

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q21/round_02.py`, copied as `plan.py`: `q21_pass1` in `_CUDA_SOURCE`, lines 11-45 of the plan file.

## Extraction edits

- `__global__ void q21_pass1(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- `#include <cstdint>` is added: the kernel uses `uint32_t`, which the plan's `<torch/extension.h>` declares through `<cstdint>`; the front end compiles without the PyTorch include path, and the CUDA headers that nvcc includes do not declare it.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q21_launcher` launches `q21_pass1<<<blocks1, 1024>>>` with `blocks1 = (n_lineitems + 1023) / 1024`, when `blocks1 > 0`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_orderkey`, `l_suppkey`, `l_receiptdate`, `l_commitdate`: columns of `lineitem`, `n_lineitems` rows each, at least 1 at the launch.
- `order_map`: `torch.full((max_okey + 1,), -1)` scattered with the dense indices `arange(num_orders_f)` of the F orders; `max_orderkey = order_map.numel() - 1` is the argument of its length (size).
- `counts`: `torch::zeros({num_orders_f})`; `lineitems`: `torch::zeros({num_orders_f * 16})`; both zeroed by the plan before the launch: readwrite. `num_orders_f` is the count of F orders, at least 1 (the plan returns earlier when there are none).
- `l_orderkey` values are not bounded by the F orders' maximum (the kernel checks them): `read`. `order_map` holds -1, so it is not an index buffer: `read`.
