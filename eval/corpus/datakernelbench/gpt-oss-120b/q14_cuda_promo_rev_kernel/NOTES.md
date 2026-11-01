# q14_cuda_promo_rev_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q14/round_03.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `promo_rev_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `promo_rev_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- The declaration states the plan's parameters in order, with their types and qualifiers, without the comments that the plan's parameter list has: the corpus reads a kernel's parameters from its declaration (`eval/corpus/sources.py`, `c_parameters`), and a comment in the list, such as one with a comma or a parenthesis, is read as part of a parameter.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024`, the device function `block_reduce` (with its `#ifndef` guard), and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`promo_rev_launcher` launches `promo_rev_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `extprice`, `discount`, `shipdate`, `promo`: columns of the joined table `joined`, `N = extprice.numel()` rows.
- `out_total`, `out_promo`: `torch::zeros({1})` each, added into once per block.
