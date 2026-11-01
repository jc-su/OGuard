# q12_cuda_q12_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q12/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q12_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The plan declares the entry kernel `extern "C" __global__ void q12_kernel(...)`; it is renamed `kernel` and declared on one line, `extern "C" __global__ void kernel(...)`.
- The declaration states the plan's parameters in order, with their types and qualifiers, without the comments that the plan's parameter list has: the corpus reads a kernel's parameters from its declaration (`eval/corpus/sources.py`, `c_parameters`), and a comment in the list, such as one with a comma or a parenthesis, is read as part of a parameter.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (with its `#ifndef` guard), and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q12_launcher` launches `q12_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `shipmode`, `priority`, `commit`, `receipt`, `ship`: columns of the joined table `joined`, `N = shipmode.numel()` rows.
- `high_out`, `low_out`: `torch.zeros(2, dtype=int64)`, checked by `TORCH_CHECK(numel() == 2)` and zeroed again by the launcher, passed as `unsigned long long *`; the kernel adds into them atomically.
