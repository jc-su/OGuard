# q10_cuda_probe_and_agg_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q10/round_02.py`, copied as `plan.py`: `probe_and_agg_kernel` in `_CUDA_SOURCE`, lines 34-56 of the plan file.

## Extraction edits

- `__global__ void probe_and_agg_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q10_launcher` launches `probe_and_agg_kernel<<<blocks_li, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` and `blocks_li = (n_lineitems + 1023) / 1024`, when `n_lineitems > 0`, after `build_order_map_kernel`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_orderkey`, `l_extendedprice`, `l_discount`: columns of `lineitem` filtered to `l_returnflag = 'R'`, `n = l_orderkey.numel()` rows each, at least 1 at the launch.
- `order_map`: the `max_orderkey + 1` elements of the previous launch, holding -1 or an `o_custkey` value; `max_orderkey` is the argument of that length (size).
- `customer_revenue`: `torch::zeros({max_custkey + 1})` with `max_custkey = int(df_customer["c_custkey"].max())`, zeroed by the plan and accumulated with `atomicAdd`: readwrite.
- `l_orderkey` refers to `orders` by a foreign key, which the plan does not check against `max_orderkey` (the kernel does): `read`. `order_map` holds -1, so it is not an index buffer: `read`.
