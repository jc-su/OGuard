# q05_core_cuda_q5_aggregate_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q05/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q5_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q5_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macros `BLOCK_SIZE` and `MAX_NATIONS`; left out `<algorithm>`, the pybind11 include, the namespace alias, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q5_launcher` launches the kernel once, when `num_nations > 0` and `n > 0`, with
`<<<min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096), BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory, a grid-stride loop over the rows.

## Lengths and ranges

- `nation_id` (`int64`), `price`, `discount` (`float64`): the joined columns, `n_elements = l_extendedprice.numel()` rows each (the launcher checks the lengths).
- `nation_id`: `nation_id` of `nations_unique`, `range(len(nations_unique))` by construction, and `num_nations = len(nations_unique)`, so every value lies in `[0, num_nations)`: `index-in` `out`.
- `out`: `torch::zeros({num_nations})`, `num_nations` elements, which the kernel adds to with `atomicAdd` (`readwrite`).
- `num_nations`: checked `0 <= num_nations <= MAX_NATIONS` (1024) by `TORCH_CHECK` (`size_maxima`), returned on before the launch when 0: size symbol with least value 1.
- `n_elements`: size symbol with least value 1; the launch runs only when `n > 0`.
