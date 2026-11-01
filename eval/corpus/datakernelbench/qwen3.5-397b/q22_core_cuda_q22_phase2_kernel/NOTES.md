# q22_core_cuda_q22_phase2_kernel

The kernel `q22_phase2_kernel` of the CUDA plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q22/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q22_phase2_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` and `#include <algorithm>` (host binding code and headers the kernel does not use), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `binary_search`, `get_code_index`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 164: `q22_phase2_kernel<<<blocks, BLOCK_SIZE>>>` in `q22_launcher`, after `q22_phase1_kernel`, with `blocks = (n_customers + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The four customer columns have `n = c_custkey.numel()` rows; `o_custkey_sorted` is the sorted distinct `o_custkey`, `n_orders = o_custkey_sorted.numel()`; the gate does not establish sortedness, and the binary search's reads stay inside `o_custkey_sorted` whatever the order.
- `target_codes` is `torch.tensor([13, 31, 23, 29, 30, 18, 17])`, 7 elements, read by `get_code_index`.
- `group_codes`, `group_numcust`, and `group_totacctbal` are `torch::zeros({MAX_CNTRY_CODES})`, 7 elements each; `group_numcust` and `group_totacctbal` are `atomicAdd` accumulators (`readwrite`), `group_codes` is written: `write`.
