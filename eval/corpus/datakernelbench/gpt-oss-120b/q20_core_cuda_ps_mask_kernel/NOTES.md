# q20_core_cuda_ps_mask_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q20/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `ps_mask_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `ps_mask_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cstdint>` and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the Thrust includes, the launcher (which sorts, reduces, and searches with Thrust between the launches), the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q20_mask_launcher` launches `ps_mask_kernel<<<blocks_ps, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (ps_n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `ps_key`, `lower_idx`, `ps_availqty`, `mask`: the packed partsupp keys, their `thrust::lower_bound` positions among the distinct lineitem keys, `ps_availqty` (as int64), and the uint8 output, `n_ps = ps_n` rows each; the kernel writes `mask` at every row.
- `uniq_key`, `uniq_sum`: `torch::empty({li_n})` each, `li_n` the rows of the filtered lineitem, of which `thrust::reduce_by_key` fills the first `n_groups`; their length is the size symbol `li_n`.
- `n_groups`: the distinct keys that `reduce_by_key` wrote, at most `li_n`; a scalar, since it is the length of no buffer.
- `lower_idx`: `thrust::lower_bound` returns positions in `[0, n_groups]`, a range the plan establishes only through that call and that can reach `li_n`, so `lower_idx` is `read`, not an index.
