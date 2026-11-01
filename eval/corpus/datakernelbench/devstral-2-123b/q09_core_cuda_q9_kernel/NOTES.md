# q09_core_cuda_q9_kernel

Kernel `q9_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q09/round_06.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 6`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q9_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#include <cuda_runtime.h>`, which only the host code uses, and `#define BLOCK_SIZE 1024`, which the kernel does not use.

## Launch

`q9_launcher` (plan lines 28-61) launches `q9_kernel<<<(n + 1023) / 1024, 1024>>>` (lines 51-59) with 0 bytes of dynamic shared memory, `n = l_extendedprice_contig.numel()` (line 50); `_launch_q9` calls it at line 92 and `run_query` at lines 145-150.

## Lengths and ranges

- `l_extendedprice` (read, float64, length `n_elements`): `joined["l_extendedprice"]` (line 146); the four columns come from the same `joined` (lines 138-142).
- `l_discount` (read, float64, length `n_elements`): line 147.
- `ps_supplycost` (read, float64, length `n_elements`): line 148.
- `l_quantity` (read, float64, length `n_elements`): line 149.
- `amount_out` (write, float64, length `n_elements`): `torch::empty_like(l_extendedprice_contig)` (line 49).
- `n_elements` (size, int64): `l_extendedprice_contig.numel()` (line 50).
- size symbol `n_elements`, least value 0.
