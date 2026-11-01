# q20_core_cuda_pack_key_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q20/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `pack_key_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `pack_key_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cstdint>` and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the Thrust includes, the launcher (which sorts, reduces, and searches with Thrust between the launches), the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q20_mask_launcher` launches `pack_key_kernel<<<blocks, BLOCK_SIZE>>>` twice, on the filtered lineitem (`N = li_n`) and on the filtered partsupp (`N = ps_n`), each with `BLOCK_SIZE = 1024` threads, `blocks = (N + 1023) / 1024`, and no dynamic shared memory: one configuration, so one corpus kernel. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `part`, `supp`: `l_partkey` and `l_suppkey` of the filtered lineitem, or `ps_partkey` and `ps_suppkey` of the filtered partsupp, `N` rows.
- `out`: `torch::empty({N}, uint64)` (`li_key` or `ps_key`), written at every row.
