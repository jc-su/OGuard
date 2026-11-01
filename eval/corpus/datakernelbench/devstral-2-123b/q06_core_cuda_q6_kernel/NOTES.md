# q06_core_cuda_q6_kernel

Kernel `q6_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q06/round_04.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 4`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q6_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#include <cuda_runtime.h>`, `#include <device_launch_parameters.h>` and `#include <cstdio>`, which only the host code uses.
- Kept `#define BLOCK_SIZE 1024`, the size of the kernel's shared array.

## Launch

The `forward` binding (plan lines 54-89) launches `q6_kernel<<<(n + 1023) / 1024, 1024>>>` (lines 74-82) with 0 bytes of dynamic shared memory, `n = price.numel()` (line 72); `_launch_q6` calls it at lines 123-127 and `run_query` at lines 178-188.

## Lengths and ranges

- `price_ptr` (read, float64, length `n_elements`): `l_extendedprice` (line 176); the four columns come from the same `lineitem`.
- `discount_ptr` (read, float64, length `n_elements`): `l_discount` (line 174).
- `quantity_ptr` (read, float64, length `n_elements`): `l_quantity` (line 175).
- `shipdate_ptr` (read, int32, length `n_elements`): `l_shipdate` in days (lines 171-173).
- `date_lo`, `date_hi` (scalar, float64): the day counts of the parameters `shipdate_lo` and `shipdate_hi` (lines 168-169), passed as `float(...)` (line 125).
- `disc_lo`, `disc_hi`, `qty_hi` (scalar, float64): the parameters `discount_lo`, `discount_hi` and `quantity_hi` (line 126).
- `out_ptr` (readwrite, float64, length `1`): `torch::zeros({1})` (line 73); the kernel adds into it with `atomicAdd`.
- `n_elements` (size, int64): `price.numel()` (line 72).
- size symbol `n_elements`, least value 0.
