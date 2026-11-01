# q07_core_cuda_q7_kernel

Kernel `q7_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q07/round_03.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 3`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q7_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Kept `#define BLOCK_SIZE 512` and `#define NUM_GROUPS 8`, the sizes of the kernel's shared and register arrays (48 KiB of static shared memory, the limit).

## Launch

The `q7_launch` binding (plan lines 98-143) launches `q7_kernel<<<(n + 511) / 512, 512>>>` (lines 129-140) with 0 bytes of dynamic shared memory, `n = supp_nation_code.numel()` (line 123); `_query_core` calls it at lines 184-188 and `run_query` at lines 286-288.

## Lengths and ranges

- `supp_nation_code` (read, int32, length `n_elements`): `supp_code` (lines 275, 280), 0 or 1; the five columns come from the same `joined` (lines 265-272).
- `cust_nation_code` (read, int32, length `n_elements`): `cust_code` (lines 276, 281), 0 or 1.
- `l_shipdate_days` (read, int32, length `n_elements`): `l_shipdate` in days (lines 278, 282).
- `l_extendedprice` (read, float64, length `n_elements`): line 283.
- `l_discount` (read, float64, length `n_elements`): line 284.
- `shipdate_lo_days`, `shipdate_hi_days` (scalar, int32): the day counts of the parameters `shipdate_lo` and `shipdate_hi` (lines 254-255).
- `out_revenue` (readwrite, float64, length `8`): `torch::zeros({NUM_GROUPS})` (line 126); the kernel adds into it with `atomicAdd`.
- `out_count` (readwrite, int32, length `8`): `torch::zeros({NUM_GROUPS})` as int32 (line 127); the kernel adds into it with `atomicAdd`.
- `n_elements` (size, int64): `supp_nation_code.numel()` (line 123).
- size symbol `n_elements`, least value 0.
- The nation codes and the year only build a key compared with the loop variable `g`; no buffer is indexed by them.
