# q22_core_cuda_avg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q22/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `avg_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `avg_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#define BLOCK_SIZE 1024` and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The kernel's fixed-width integer types are declared for the plan by `<torch/extension.h>` through `<cstdint>`; the nvcc front end has no torch include path, so `kernel.cu` includes `<cstdint>` in its place.

## Launch

`avg_launcher` launches `avg_kernel<<<blocks, BLOCK_SIZE, shared_mem>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and `shared_mem = BLOCK_SIZE * (sizeof(double) + sizeof(unsigned long long)) = 16384` bytes of dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 16384 bytes. `launch.json` also states the grid claim `[2097152, 1, 1]`: the kernel holds the row id `blockIdx.x * blockDim.x + threadIdx.x` in an `int`, which is at most 2^31 - 1 for 2097152 = 2^31 / 1024 blocks in x and negative beyond.

## Lengths and ranges

- `acctbal`, `mask`: `c_acctbal` and the country-code membership mask (bool) of `customer`, `n = acctbal.numel()` rows.
- `out_sum`, `out_cnt`: `torch::zeros({1})` (float64, and int64 passed as `unsigned long long *`), which each block adds into atomically.
- The dynamic shared memory holds `s_sum` (1024 doubles) followed by `s_cnt` (1024 unsigned long long), one entry each per thread of the block.
