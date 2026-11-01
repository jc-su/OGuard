# q08_cuda_q8_agg_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q08/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q8_agg_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q8_agg_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, `#include <pybind11/pybind11.h>`, `#include <ATen/cuda/CUDAContext.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The kernel's `uint8_t` is declared for the plan by `<torch/extension.h>` through `<cstdint>`; the nvcc front end has no torch include path, so `kernel.cu` includes `<cstdint>` in its place. This is the one edit beyond the protocol's list.

## Launch

`q8_launcher` launches `q8_agg_kernel<<<blocks, BLOCK_SIZE, shared_bytes, stream>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and `shared_bytes = sizeof(double) * 2 * num_years` bytes of dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 16 * num_years bytes, size maxima `num_years <= 3072`.

## Lengths and ranges

- `years`, `volume`, `is_brazil`: columns of the joined table `li` (the order year, `l_extendedprice * (1 - l_discount)`, and the supplier-nation flag as uint8), `N = years.numel()` rows.
- `sum_total`, `sum_brazil`: `torch::zeros({num_years})`, which the kernel adds into atomically.
- `num_years`: `year_hi - year_lo + 1` from the two date parameters (2 by default). The dynamic shared memory is `16 * num_years` bytes; a launch above the default limit of 48 KiB does not run, so every launch has `num_years <= 3072`, the size maximum in `launch.json`.
- `year_lo`: the year of `orderdate_lo`, any int32.
