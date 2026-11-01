# q10_core_cuda_q10_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/core/queries/q10/round_02.py`, copied as `plan.py`: `q10_kernel` in `_CUDA_SOURCE`, lines 11-29 of the plan file.

## Extraction edits

- `__global__ void q10_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q10_launcher` launches `q10_kernel<<<blocks, threads>>>` with `threads = 1024` and `blocks = (n + 1023) / 1024`, when `n > 0` (`_query_core` also returns earlier when the frame is empty).

`launch.json` states the block (1024, 1, 1) and the dynamic shared memory 0 bytes.

## Lengths and ranges

- `custkey_ptr`, `extprice_ptr`, `discount_ptr`: columns of the joined frame, `n = custkey.numel()` rows each, at least 1 at the launch.
- `rev_out`, `present_out`: `torch.zeros(max_key + 1)` with `max_key = int(c_custkey.max())` of the same frame, zeroed by the plan before the launch and accumulated with `atomicAdd` or set: readwrite.
- `custkey_ptr`: the plan sizes `rev_out` and `present_out` by this column's maximum, and the TPC-H customer keys are positive, so every value lies in `[0, max_key]`: `index-in` into `rev_out`.
