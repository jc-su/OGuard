# q07_cuda_q7_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q07/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q7_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q7_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- The declaration states the plan's parameters in order, with their types and qualifiers, without the comments that the plan's parameter list has: the corpus reads a kernel's parameters from its declaration (`eval/corpus/sources.py`, `c_parameters`), and a comment in the list, such as one with a comma or a parenthesis, is read as part of a parameter.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q7_launcher` launches `q7_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `supp`, `cust`, `shipday`, `price`, `disc`: columns of the joined table `joined`, `N = price.numel()` rows.
- `supp`, `cust`: `(joined[...] == nation_b).astype("int32")`, so the plan's values are 0 or 1. No buffer of the kernel has 2 entries, so the range is not an index role: both are `read`.
- `out`: `torch::zeros({8})`, which the kernel adds into atomically.
