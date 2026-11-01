# q22_cuda_final_group_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `final_group_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `final_group_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the device function `find_code_idx`; left out
  the Torch, ATen, c10, and `<vector>` includes, the macro `BLOCK_SIZE` (not
  used by the kernel), and the host code.

## Launch

`q22_execute` launches the kernel once with
`<<<(n_customers + 1023) / 1024, BLOCK_SIZE, shared_bytes, stream>>>`: block
(1024, 1, 1), dynamic shared memory
`shared_bytes = num_codes * (sizeof(unsigned long long) + sizeof(double))`,
that is `16 * num_codes` bytes (`launch.json` terms `{"num_codes": 16}`).

## Lengths and ranges

- `c_custkey`, `c_acctbal`, `cntrycode`: `n_customers` rows each; the launcher
  checks the lengths against `n_customers = c_acctbal.numel()`.
- `n_customers`: size symbol with least value 1; the launcher returns for 0.
- `order_flags`: `torch::zeros({flags_len})` of uint8, filled by
  `build_order_flags_kernel`; read only here.
- `flags_len`: `max(1, max(c_custkey) + 1)`, checked `> 0` by the launcher, a
  size symbol with least value 1.
- `codes`: `num_codes` int32 entries; `num_codes` is a size symbol with least
  value 1 and the maximum 1024 that `TORCH_CHECK(num_codes <= 1024)`
  establishes (`size_maxima` in `launch.json`).
- `avg_stats`: the 2 doubles of `avg_acctbal_kernel`, read only here.
- `out_counts_i64`, `out_sums`: `torch::zeros({num_codes})` of int64 and
  float64, which the kernel adds to with `atomicAdd` (`readwrite`).
