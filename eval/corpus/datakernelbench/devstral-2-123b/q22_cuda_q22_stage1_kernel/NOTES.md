# q22_cuda_q22_stage1_kernel

Kernel `q22_stage1_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q22/round_03.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 3`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q22_stage1_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#define BLOCK_SIZE 1024`, which the kernel does not use, and the plan's second kernel `q22_stage2_kernel` (a separate corpus kernel).

## Launch

The `q22_stage1` binding (plan lines 93-106) launches `q22_stage1_kernel<<<(n_customer + 1023) / 1024, 1024, 2 * 1024 * sizeof(double)>>>`, 16384 bytes of dynamic shared memory; `run_query` calls it at line 195.

## Lengths and ranges

- `c_acctbal_ptr` (read, float64, length `n_customer`): `customer["c_acctbal"]` (line 187); `n_customer = c_custkey_t.numel()` (line 191), the rows of `customer`.
- `in_cntrycode_mask_ptr` (read, bool, length `n_customer`): the country code test of each customer (lines 180, 189).
- `avg_placeholder_ptr` (readwrite, float64, length `2`): `torch.zeros(2)` (line 194); the kernel adds into both elements with `atomicAdd`.
- `n_customer` (size, int64): `c_custkey_t.numel()` (line 191).
- size symbol `n_customer`, least value 0.
