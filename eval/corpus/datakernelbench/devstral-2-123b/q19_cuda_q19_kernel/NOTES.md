# q19_cuda_q19_kernel

Kernel `q19_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q19/round_01.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 1`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q19_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.

## Launch

`q19_launcher` (plan lines 84-96) launches `q19_kernel<<<(n + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory, `n = l_quantity.numel()`; `run_query` calls it at lines 198-202.

## Lengths and ranges

- `l_quantity` (read, float64, length `n_elements`): `joined["l_quantity"]` (line 190).
- `l_extendedprice` (read, float64, length `n_elements`): `joined["l_extendedprice"]` (line 191).
- `l_discount` (read, float64, length `n_elements`): `joined["l_discount"]` (line 192).
- `p_brand_id` (read, int32, length `n_elements`): the brand code of the same rows (lines 166-170, 193).
- `p_container_grp` (read, int32, length `n_elements`): the container group of the same rows (lines 174-187, 194).
- `p_size` (read, int32, length `n_elements`): `joined["p_size"]` (line 195).
- `brand12_id` (scalar, int32): the constant 12 (line 165).
- `brand23_id` (scalar, int32): the constant 23 (line 165).
- `brand34_id` (scalar, int32): the constant 34 (line 165).
- `sm_grp` (scalar, int32): the constant 1 (line 173).
- `md_grp` (scalar, int32): the constant 2 (line 173).
- `lg_grp` (scalar, int32): the constant 3 (line 173).
- `out_ptr` (readwrite, float64, length `1`): `torch::zeros({1})` (line 85); the kernel adds into it with `atomicAdd`.
- `n_elements` (size, int64): `l_quantity.numel()` (line 84); the six columns come from the same `joined`.
- size symbol `n_elements`, least value 0.
