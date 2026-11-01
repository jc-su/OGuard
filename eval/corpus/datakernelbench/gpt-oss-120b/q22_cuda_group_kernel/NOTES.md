# q22_cuda_group_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q22/round_04.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `group_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `group_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- The declaration states the plan's parameters in order, with their types and qualifiers, without the comments that the plan's parameter list has: the corpus reads a kernel's parameters from its declaration (`eval/corpus/sources.py`, `c_parameters`), and a comment in the list, such as one with a comma or a parenthesis, is read as part of a parameter.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`group_by_cntry` launches `group_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `custkey`, `bal`, `cntrycode`, `in_mask`: columns of `customer`, `N = custkey.numel()` rows.
- `sorted_orders`: the sorted distinct `o_custkey` values, `orders_n = sorted_orders.numel()`.
- `target_codes`: the sorted target country codes, `n_codes = target_codes.numel()` (7 by default).
- `grp_counts`, `grp_sums`: `torch::zeros({n_codes})` each (int64 passed as `unsigned long long *`, and float64), which the kernel adds into atomically.
