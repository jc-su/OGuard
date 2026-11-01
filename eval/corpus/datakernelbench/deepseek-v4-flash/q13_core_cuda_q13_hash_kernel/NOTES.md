# q13_core_cuda_q13_hash_kernel

`kernel.cu` is the kernel `q13_hash_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q13/round_04.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q13_hash_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's other kernel (`q13_gather_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`do_hash_launcher`, called once by `_launch_q13`, launches `q13_hash_kernel<<<(n_orders + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory, then `q13_gather_kernel`. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `c_custkey` (read, length `n_customers`): `c_custkey`, one per customer, `n_customers = c_custkey.numel()`; the kernel does not access it.
- `num_customers` (size): `int(df_customer["c_custkey"].max())`; `order_counts` has `num_customers + 1` elements.
- `o_custkey` (read, length `n_orders`): `o_custkey` of the orders whose comment does not match; `n_orders` is `o_custkey.numel()`. The kernel bounds it before indexing, so no index role.
- `n_orders` (size): `o_custkey.numel()`.
- `order_counts` (readwrite, length `num_customers + 1`): `torch.zeros(num_customers + 1, int64)` in `_launch_q13`, passed as `unsigned long long*`; the kernel adds to it with `atomicAdd`.

Least values of the size symbols: `n_customers` 0, `num_customers` 0, `n_orders` 0. The plan launches on whatever orders its filter keeps, which can be none; `num_customers` is the largest customer key, at least 1 in TPC-H, which the plan does not check.
