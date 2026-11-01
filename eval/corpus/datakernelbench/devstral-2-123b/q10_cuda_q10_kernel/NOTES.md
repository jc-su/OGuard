# q10_cuda_q10_kernel

Kernel `q10_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q10/round_10.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 10`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q10_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Kept `#include <cuda_runtime.h>`, which declares the kernel's `atomicAdd`.
- Left out `#include <cooperative_groups.h>`, `using namespace cooperative_groups;` and `#define BLOCK_SIZE 1024`, which the kernel does not use.

## Launch

`q10_launcher` (plan lines 51-63) launches `q10_kernel<<<(n + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory, `n = price.numel()`; `run_query` calls it at line 143.

## Lengths and ranges

- `price_ptr` (read, float64, length `n_elements`): `lo_f["l_extendedprice"]` (line 138); the launcher checks that the three inputs have `n` elements (lines 48-51).
- `discount_ptr` (read, float64, length `n_elements`): `lo_f["l_discount"]` (line 139).
- `custkey_ptr` (read, int64, length `n_elements`): `lo_f["o_custkey"]` (line 140), the customer keys of the joined orders; the plan does not bound them and relies on each `o_custkey` being at most `max(c_custkey)`, so the role is `read`.
- `revenue_ptr` (readwrite, float64, length `max_custkey + 1`): `torch::zeros({max_custkey + 1})` (line 55); the kernel adds into it with `atomicAdd`.
- `n_elements` (size, int64): `price.numel()` (line 51).
- size symbol `n_elements`, least value 0.
- size symbol `max_custkey`, least value 0.
