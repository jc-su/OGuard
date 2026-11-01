# q19_cuda_q19_case_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q19/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q19_case_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q19_case_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, c10,
  `cuda_runtime.h`, and `<algorithm>` includes, `compute_blocks`, and the host
  code, which the kernel does not use.

## Launch

`q19_case_launcher` launches the kernel once with
`<<<compute_blocks(n), BLOCK_SIZE>>>`: block (1024, 1, 1), no dynamic shared
memory. `compute_blocks` takes `min(ceil(n / 1024), 8 * SM count)` blocks; the
kernel strides over the rows. `run_query` takes this path when the three brand
literals are distinct (the default parameters).

## Lengths and ranges

- `quantity_ptr`, `price_ptr`, `discount_ptr`, `qcase_ptr`: `n_elements` rows
  each, the columns of the join of the filtered `lineitem` with the filtered
  `part` rows. The launcher sets `n = quantity.numel()` and checks that the other
  three have the same `numel()`.
- `n_elements`: size symbol with least value 1; the launcher returns before the
  launch when `n == 0`.
- `out_ptr`: `torch::zeros({1})`, one element, which the kernel adds to with
  `atomicAdd` (`readwrite`).
- `qcase_ptr` holds the case id 1, 2, or 3; the kernel only compares it.
