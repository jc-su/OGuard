# q22_core_cuda_agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q22/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `agg_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `agg_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#define MAX_CODE 100` and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The kernel's fixed-width integer types are declared for the plan by `<torch/extension.h>` through `<cstdint>`; the nvcc front end has no torch include path, so `kernel.cu` includes `<cstdint>` in its place.

## Launch

`agg_launcher` launches `agg_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes. `launch.json` also states the grid claim `[2097152, 1, 1]`: the kernel holds the row id `blockIdx.x * blockDim.x + threadIdx.x` in an `int`, which is at most 2^31 - 1 for 2097152 = 2^31 / 1024 blocks in x and negative beyond.

## Lengths and ranges

- `custkey`, `acctbal`, `cntrycode`, `mask`: columns of `customer` (the country code as int32, the membership mask as bool), `n = custkey.numel()` rows.
- `orders`: the sorted distinct `o_custkey` values, `n_orders = orders_unique.numel()` entries; the kernel binary-searches it.
- `out_cnt`, `out_sum`: `torch::zeros({MAX_CODE})` with `MAX_CODE = 100` (int64 passed as `unsigned long long *`, and float64); the kernel adds into them atomically at a code it tests against `[0, 100)`.
- `avg`: the average balance of the first kernel, any double.
