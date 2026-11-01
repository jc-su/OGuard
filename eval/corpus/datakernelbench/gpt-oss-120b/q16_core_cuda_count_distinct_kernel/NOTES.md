# q16_core_cuda_count_distinct_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q16/round_02.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `count_distinct_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `count_distinct_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: the kernel's body, with the comment of its parameter list.
- Left out as host code: `#include <torch/extension.h>`, the Thrust includes, the launcher (which sorts the pairs with Thrust before the launch), and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.
- The kernel's fixed-width integer types are declared for the plan by `<torch/extension.h>` through `<cstdint>`; the nvcc front end has no torch include path, so `kernel.cu` includes `<cstdint>` in its place.

## Launch

`distinct_count_launcher` launches `count_distinct_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `group`, `supp`: the `(group id, ps_suppkey)` pairs of the joined table, sorted by the launcher, `n = group_ids.numel()` rows.
- `group`: `_gid = range(len(group_df))` merged onto the joined rows, so every value lies in `[0, num_groups)` with `num_groups = len(group_df)`: an index into `out`.
- `out`: `torch::zeros({num_groups}, int64)` passed as `unsigned long long *`; the kernel adds into it atomically.
