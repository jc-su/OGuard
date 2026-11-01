# q06_cuda_q6_kernel

Kernel `q6_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q06/round_01.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 1`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q6_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.

## Launch

`q6_launcher` (plan lines 74-89) launches `q6_kernel<<<(n + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory, `n = price.numel()`; `run_query` calls it at lines 147-151.

## Lengths and ranges

- `price_ptr` (read, float64, length `n_elements`): `l_extendedprice` of `lineitem` (line 143); `n_elements = price.numel()` (line 74).
- `discount_ptr` (read, float64, length `n_elements`): `l_discount` of the same `lineitem` (line 144).
- `quantity_ptr` (read, float64, length `n_elements`): `l_quantity` of the same `lineitem` (line 145).
- `shipdate_ptr` (read, int32, length `n_elements`): `l_shipdate` in days, of the same `lineitem` (line 146).
- `date_lo` (scalar, int32): from the date parameter `shipdate_lo` (line 141); the plan does not bound it.
- `date_hi` (scalar, int32): from the date parameter `shipdate_hi` (line 142).
- `disc_lo` (scalar, float64): the parameter `discount_lo`.
- `disc_hi` (scalar, float64): the parameter `discount_hi`.
- `qty_hi` (scalar, float64): the parameter `quantity_hi`.
- `out_ptr` (readwrite, float64, length `1`): `torch::zeros({1})` (line 75); the kernel adds into it with `atomicAdd`.
- `n_elements` (size, int64): `price.numel()` (line 74).
- size symbol `n_elements`, least value 0.
