# q19_cuda_q19_full_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q19/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q19_full_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q19_full_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, c10,
  `cuda_runtime.h`, and `<algorithm>` includes, `compute_blocks`, and the host
  code, which the kernel does not use.

## Launch

`q19_full_launcher` launches the kernel once with
`<<<compute_blocks(n), BLOCK_SIZE>>>`: block (1024, 1, 1), no dynamic shared
memory. `compute_blocks` takes `min(ceil(n / 1024), 8 * SM count)` blocks; the
kernel strides over the rows. `run_query` takes this path through
`_run_query_exact_reference_semantics` when two of the three brand literals are
equal.

## Lengths and ranges

- `quantity_ptr`, `price_ptr`, `discount_ptr`, `brand_ptr`, `container_ptr`,
  `size_ptr`: `n_elements` rows each, the columns of the join of the filtered
  `lineitem` with the filtered `part` rows. The launcher sets
  `n = quantity.numel()` and checks that the other five have the same `numel()`.
- `n_elements`: size symbol with least value 1; the launcher returns before the
  launch when `n == 0`.
- `brand12_id`, `brand23_id`, `brand34_id`, `sm_grp`, `md_grp`, `lg_grp`: the
  plan passes the constants 12, 23, 34, 1, 2, 3.
- `out_ptr`: `torch::zeros({1})`, one element, which the kernel adds to with
  `atomicAdd` (`readwrite`).
