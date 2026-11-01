# q19_core_cuda_q19_kernel

Kernel `q19_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q19/round_01.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 1`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q19_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Kept `#define BLOCK_SIZE 1024`, the size of the kernel's shared array.

## Launch

`q19_launcher` (plan lines 65-111) launches `q19_kernel<<<(n + 1023) / 1024, 1024>>>` (lines 99-109) with 0 bytes of dynamic shared memory, `n = l_quantity.numel()` (line 97); `_query_core` calls it at lines 150-157 and `run_query` at lines 223-236.

## Lengths and ranges

- `l_quantity_ptr` (read, float64, length `n_elements`): `joined["l_quantity"]` (line 216); the six columns come from the same `joined` (line 194).
- `l_extendedprice_ptr` (read, float64, length `n_elements`): line 217.
- `l_discount_ptr` (read, float64, length `n_elements`): line 218.
- `p_brand_id_ptr` (read, int32, length `n_elements`): `brand_encoded` (lines 196-201, 219).
- `p_container_grp_ptr` (read, int32, length `n_elements`): `container_grp` (lines 203-214, 220).
- `p_size_ptr` (read, int32, length `n_elements`): `joined["p_size"]` (line 221).
- `brand12_id`, `brand23_id`, `brand34_id` (scalar, int32): 12, 23, 34 (line 196).
- `sm_grp`, `md_grp`, `lg_grp` (scalar, int32): 1, 2, 3 (line 203).
- `out_ptr` (readwrite, float64, length `1`): `torch::zeros({1})` (line 98); the kernel adds into it with `atomicAdd`.
- `n_elements` (size, int64): `l_quantity.numel()` (line 97).
- size symbol `n_elements`, least value 0.
