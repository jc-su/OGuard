# q21_cuda_q21_pass2

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q21/round_02.py`, copied as `plan.py`: `q21_pass2` in `_CUDA_SOURCE`, lines 47-111 of the plan file.

## Extraction edits

- `__global__ void q21_pass2(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q21_launcher` launches `q21_pass2<<<blocks2, 1024>>>` with `blocks2 = (num_orders_f + 1023) / 1024`, when `blocks2 > 0`, after `q21_pass1`.

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `counts`, `lineitems`: the `num_orders_f` counts and `16 * num_orders_f` packed rows of the previous launch; `num_orders_f` is the argument (size), at least 1.
- `supp_map`: `torch.full((max_skey + 1,), -1)` scattered with the dense indices `arange(num_supp_n)` of the nation's suppliers; `max_suppkey = supp_map.numel() - 1` is the argument of its length (size).
- `out_counts`: `torch::zeros({num_supp_n})`, zeroed and accumulated with `atomicAdd`: readwrite; `num_supp_n` is the nation's supplier count, at least 1 (the plan returns earlier when there are none).
- `supp_map` holds -1, so it is not an index buffer: `read`.
