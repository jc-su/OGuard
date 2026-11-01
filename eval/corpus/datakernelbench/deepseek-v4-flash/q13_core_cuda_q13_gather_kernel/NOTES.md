# q13_core_cuda_q13_gather_kernel

`kernel.cu` is the kernel `q13_gather_kernel` of `plan.py`, a copy of `logs/kernel_plans/tpch/sf10/h100/deepseek-ai__deepseek-v4-flash/cuda/core/queries/q13/round_04.py` of DataKernelBench at commit `43c361d`.

## Edits

- The kernel `q13_gather_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Copied unchanged from the plan's source, in its order: `#include <cuda_runtime.h>`, `#include <cstdint>`, `#define BLOCK_SIZE 1024` (the includes declare the fixed-width integer types and the CUDA runtime that the kernel uses).
- Left out: `#include <torch/extension.h>` (the PyTorch host API), which only the host code uses (nvcc's implicit `cuda_runtime.h` declares the kernel's integer types and atomics); the host code (the launcher and the module binding); the plan's other kernel (`q13_hash_kernel`).
- The kernel's body and parameters are the plan's.

## Launch

`do_hash_launcher`, called once by `_launch_q13`, launches `q13_gather_kernel<<<(n_cust + 1023) / 1024, BLOCK_SIZE>>>` with no dynamic shared memory, after `q13_hash_kernel`. `launch.json` states the block (1024, 1, 1) and 0 bytes of dynamic shared memory.

## Lengths and ranges

- `c_custkey` (read, length `n_customers`): `c_custkey`, one per customer; `n_customers` is `c_custkey.numel()`. The largest value is `num_customers`, but the plan does not check that the keys are non-negative, so no index role.
- `order_counts` (read, length `num_customers + 1`): `order_counts`, `torch.zeros(num_customers + 1)` with `num_customers = max(c_custkey)`, filled by `q13_hash_kernel`.
- `n_customers` (size): `c_custkey.numel()`.
- `c_count` (write, length `n_customers`): `torch::zeros({n_cust})`.

Least values of the size symbols: `n_customers` 0, `num_customers` 0. The plan launches on the whole `customer` table, which can be empty; `num_customers` is its largest key.
