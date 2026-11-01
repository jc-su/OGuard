# q17_cuda_q17_second_pass

Kernel `q17_second_pass` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q17/round_10.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 10`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q17_second_pass` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Kept `#include <cuda_runtime.h>` and the device function `atomicAdd_double`, which the kernel uses.
- Left out `#define BLOCK_SIZE 1024` and `#define MAX_PARTS_PER_BLOCK 1024`, which the kernel does not use, and the plan's first kernel `q17_first_pass` (a separate corpus kernel).

## Launch

`q17_kernel_launcher` (plan lines 120-131) launches `q17_second_pass<<<(n_items + 1023) / 1024, 1024, 1024 * sizeof(double)>>>`, 8192 bytes of dynamic shared memory; `run_query` calls it at line 190.

## Lengths and ranges

- `l_partkey_ptr` (read, int64, length `n_items`): `lineitem["l_partkey"]` (line 185); `n_items = l_partkey.numel()` (line 100).
- `l_quantity_ptr` (read, float64, length `n_items`): `lineitem["l_quantity"]` (line 186).
- `l_extendedprice_ptr` (read, float64, length `n_items`): `lineitem["l_extendedprice"]` (line 187).
- `part_keys_ptr` (read, int64, length `n_parts`): the `p_partkey` of the filtered parts (line 184); `n_parts = part_keys.numel()` (line 101).
- `sum_qty_ptr` (read, float64, length `n_parts`): `torch::zeros({n_parts})` (line 103), written by `q17_first_pass`.
- `cnt_qty_ptr` (read, float64, length `n_parts`): `torch::zeros({n_parts})` (line 104), written by `q17_first_pass`.
- `sum_price_ptr` (readwrite, float64, length `1`): `torch.zeros(1)` (line 189); the kernel adds into it with a compare-and-swap loop.
- `n_items` (size, int64): `l_partkey.numel()` (line 100).
- `n_parts` (size, int64): `part_keys.numel()` (line 101).
- size symbol `n_items`, least value 0.
- size symbol `n_parts`, least value 0.
