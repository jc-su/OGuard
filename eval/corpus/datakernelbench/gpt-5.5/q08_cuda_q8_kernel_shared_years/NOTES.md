# q08_cuda_q8_kernel_shared_years

Source: `azure__gpt-5.5/cuda/full/queries/q08/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q8_kernel_shared_years` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q8_kernel_shared_years` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `MAX_YEARS`; left out the Torch, ATen, `cuda_runtime.h`,
  `<algorithm>`, and `<vector>` includes, the other kernels, and the host
  code, which the kernel does not use.

## Launch

`q8_launcher` launches one of three kernels by `n_years`, after returning
when `n == 0 || n_years == 0`: `q8_kernel_two_year` when `n_years == 2`,
`q8_kernel_shared_years` when `n_years <= MAX_YEARS` (128), and
`q8_kernel_global_years` otherwise. This kernel runs when the order-date bounds span 1 or 3 to 128 years; the default parameters do not reach it. The launch is
`<<<min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096), BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `year_ptr`, `price_ptr`, `discount_ptr`, `suppkey_ptr`: `n_elements` rows
  each, the columns of the joined lineitem rows. The launcher sets
  `n = years.numel()` and checks that the other three have the same `numel()`.
- `n_elements`: size symbol with least value 1; the launcher returns before the
  launch when `n == 0`.
- `target_supp_ptr`: `_build_supplier_target_flags` allocates
  `torch.zeros(max_suppkey + 1, dtype=torch.uint8)`, and the launcher passes
  `target_supp_flags.numel()` as `target_supp_len`, so `target_supp_len` is the
  size symbol of its length (least value 1). The values of `suppkey_ptr` are
  `l_suppkey` of the join; the plan does not bound them by `max_suppkey`, and
  the kernel checks `sk >= 0 && sk < target_supp_len` itself, so `suppkey_ptr`
  is `read`.
- `base_year`: `lo_ts.year`, the year of `orderdate_lo`; it only shifts the year.
- `n_years`: `max(0, hi_ts.year - base_year + 1)`, the size symbol of the three
  outputs; least value 1 (the launcher returns for 0) and, in this branch, at
  most 128, which `launch.json` states as `size_maxima`.
- `out_target_ptr`, `out_total_ptr`, `out_count_ptr`: `torch::zeros({n_years})`,
  `n_years` elements each, which the kernel adds to with `atomicAdd`
  (`readwrite`).
