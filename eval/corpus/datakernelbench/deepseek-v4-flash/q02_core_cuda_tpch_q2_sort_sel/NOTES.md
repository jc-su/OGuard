# q02_core_cuda_tpch_q2_sort_sel

`kernel.cu` is the kernel `tpch_q2_sort_sel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q02/round_01.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `tpch_q2_sort_sel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cstdint>`, `#define BLOCK_SIZE 256` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding).
- The kernel's body and parameters are the plan's.

## Launch

`q2_launcher`, called once by `_launch_q2` when `final` has rows, launches `tpch_q2_sort_sel<<<1, BLOCK_SIZE, shared_mem>>>` with `shared_mem = 2 * n * sizeof(int64_t)` clamped to [1024, 131072] bytes: 256 threads and one block. `launch.json` states the block (256, 1, 1) and `16 * n` bytes of dynamic shared memory with `n` at most 3072: for `n <= 8192` the plan's request is at least `16 * n` bytes, and a request above 48 KiB (`n > 3072`) exceeds the limit of a launch without an opt-in, so the plan's kernel does not run then.

## Lengths and ranges

- `s_acctbal_ptr` (read, length `n`): `s_acctbal` of the sorted final frame, one per row; `n` is `s_acctbal.numel()`.
- `p_partkey_ptr` (read, length `n`): `p_partkey` of the final frame, one per row; the kernel does not access it.
- `n` (size): `s_acctbal.numel()`; the plan launches only when `len(final) > 0`.
- `limit` (scalar): the query parameter `limit` (100); the launcher returns before the launch when `min(limit, n)` is 0, so the launch has `limit >= 1`.
- `out_idx_ptr` (write, length `k`): `torch::empty({k})` with `k = min(limit, n)`, which is not affine in `limit` and `n`; the signature names its length `k`, a size symbol of its own.

Least values of the size symbols: `n` 1, `k` 1. The plan launches only when the final frame has rows and `min(limit, n)` is positive.
