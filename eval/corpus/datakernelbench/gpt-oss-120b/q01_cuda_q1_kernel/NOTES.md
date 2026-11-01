# q01_cuda_q1_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q01/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q1_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q1_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- The declaration states the plan's parameters in order, with their types and qualifiers, without the comments that the plan's parameter list has: the corpus reads a kernel's parameters from its declaration (`eval/corpus/sources.py`, `c_parameters`), and a comment in the list, such as one with a comma or a parenthesis, is read as part of a parameter.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q1_launcher` launches `q1_kernel<<<blocks, threads>>>` with `threads = BLOCK_SIZE = 1024`, `blocks = (n + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `rf_code`, `ls_code`, `qty`, `price`, `disc`, `tax`, `shipdate_days`: columns of `lineitem` (`l_returnflag` and `l_linestatus` as cuDF category codes cast to int32, the numeric columns, and `l_shipdate` as int32 days), all with the `n = qty.numel()` rows of the table.
- `sum_qty`, `sum_base_price`, `sum_disc_price`, `sum_charge`, `sum_disc`, `cnt`: `torch.zeros(_MAX_GROUPS)` with `_MAX_GROUPS = 256 * 256 = 65536`, checked by `TORCH_CHECK(numel() == MAX_GROUPS)`; the kernel adds into them atomically, so each is read and written. `cnt` is the int64 tensor passed as `unsigned long long *`.
- `shipdate_hi`: days since epoch of the `shipdate_hi` parameter, any int32.
- The plan does not bound the category codes: `rf_code` and `ls_code` have the role `read`. The group index `rf_code * 256 + ls_code` stays below 65536 only while each column has at most 256 categories, which the plan assumes (TPC-H has 3 and 2).
