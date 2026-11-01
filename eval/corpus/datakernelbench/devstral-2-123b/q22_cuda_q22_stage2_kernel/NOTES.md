# q22_cuda_q22_stage2_kernel

Kernel `q22_stage2_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/full/queries/q22/round_03.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 3`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `q22_stage2_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Left out `#define BLOCK_SIZE 1024`, which the kernel does not use, and the plan's first kernel `q22_stage1_kernel` (a separate corpus kernel).

## Launch

The `q22_stage2` binding (plan lines 108-133) launches `q22_stage2_kernel<<<(n_customer + 1023) / 1024, 1024>>>` with 0 bytes of dynamic shared memory; `run_query` calls it at lines 202-208.

## Lengths and ranges

- `c_custkey_ptr` (read, int64, length `n_customer`): `customer["c_custkey"]` (line 186); `n_customer = c_custkey_t.numel()` (line 191).
- `c_acctbal_ptr` (read, float64, length `n_customer`): `customer["c_acctbal"]` (line 187).
- `cntrycode_ptr` (read, int32, length `n_customer`): the first two characters of `c_phone` as an integer (lines 178, 188); the plan does not bound it, so the role is `read`.
- `in_cntrycode_mask_ptr` (read, bool, length `n_customer`): `cntrycode` in the requested codes (lines 180, 189).
- `o_custkey_ptr` (read, int64, length `n_orders`): the sorted distinct `o_custkey` (lines 182-184); `n_orders = o_custkey_sorted.numel()` (line 192).
- `avg_val` (scalar, float64): `avg_placeholder[0] / avg_placeholder[1]` (line 197).
- `numcust_buff` (readwrite, float64, length `100`): `torch.zeros(100)` (line 199); the kernel adds into it with `atomicAdd`.
- `totacctbal_buff` (readwrite, float64, length `100`): `torch.zeros(100)` (line 200), as `numcust_buff`.
- `n_customer` (size, int64): `c_custkey_t.numel()` (line 191).
- `n_orders` (size, int64): `o_custkey_sorted.numel()` (line 192).
- size symbol `n_customer`, least value 0.
- size symbol `n_orders`, least value 0.
