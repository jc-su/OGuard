# q15_core_cuda_q15_count_selected_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q15_count_selected_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_count_selected_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out `<math_constants.h>`, the macro `BLOCK_SIZE`, the device function `q15_find_supplier_index`, `<vector>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q15_launcher` launches the kernel once, when `blocks_sup > 0`, with
`<<<blocks_sup, BLOCK_SIZE, 0, stream>>>`, `blocks_sup = (n_sup + 255) / 256`: block
(256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `totals`, `present`: the `n_sup` totals and flags, `n` elements each; read only.
- `max_value`: `current`, the one remaining block maximum after the reduction loop; read at 0.
- `count`: `torch::zeros({1})` of `int64` reinterpreted as `unsigned long long`, one element, added to with `atomicAdd` (`readwrite`).
- `n`: `n_sup`, size symbol with least value 1; `blocks_sup > 0` requires `n_sup >= 1`.
