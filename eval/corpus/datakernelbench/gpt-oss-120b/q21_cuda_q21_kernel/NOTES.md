# q21_cuda_q21_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q21/round_05.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q21_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The plan declares the entry kernel `extern "C" __global__ void q21_kernel(...)`; it is renamed `kernel` and declared on one line, `extern "C" __global__ void kernel(...)`.
- The declaration states the plan's parameters in order, with their types and qualifiers, without the comments that the plan's parameter list has: the corpus reads a kernel's parameters from its declaration (`eval/corpus/sources.py`, `c_parameters`), and a comment in the list, such as one with a comma or a parenthesis, is read as part of a parameter.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q21_launcher` launches `q21_kernel<<<blocks, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n_rows + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `orderkey`, `supp_id`, `is_late`: columns of `l1` (lineitem joined with the nation's suppliers and the status-filtered orders), `n_rows = orderkey.numel()` rows.
- `supp_id`: `supp_n["supp_id"] = range(len(supp_n))`, merged into `l1` by an inner join, so every value lies in `[0, num_suppliers)`: an index into `out_counts`.
- `map`: `torch.full((max_key + 1,), -1)` with `max_key = max(lo_orderkeys)`, its entries at `lo_orderkeys` set to `0 .. len(lo_orderkeys) - 1`; its length is the size symbol `map_size`. The plan's values lie in `[-1, n_orders)`, which is not an index range, so `map` is `read`.
- `distinct`, `late_dist`: the per-order aggregates of `agg`, `n_orders = len(lo_orderkeys)` rows.
- `orderkey`: the `l1` order keys are a subset of `lo_orderkeys` through the plan's joins, so they lie at most at `max_key`, a relation between buffers; the plan sets no lower bound. It is `read`.
- `out_counts`: `torch::zeros({num_suppliers})` of uint32, which the kernel adds into atomically. `num_suppliers` is not a kernel argument.
