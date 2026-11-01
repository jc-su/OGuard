# q15_core_cuda_q15_fill_selected_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q15/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q15_fill_selected_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_fill_selected_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out `<math_constants.h>`, the macro `BLOCK_SIZE`, the device function `q15_find_supplier_index`, `<vector>`, the CUDA runtime include, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q15_launcher` launches the kernel once, when `h_count > 0` and `blocks_sup > 0`,
with `<<<blocks_sup, BLOCK_SIZE, 0, stream>>>`, `blocks_sup = (n_sup + 255) / 256`:
block (256, 1, 1), no dynamic shared memory, `fill_count = torch::zeros({1})` as
`count`.

## Lengths and ranges

- `totals`, `present`, `s_suppkey`: `n = n_sup` elements each; read only.
- `max_value`: the one remaining block maximum; read at 0.
- `count`: the one-element `fill_count`, zeroed before the launch, used as an atomic cursor (`readwrite`).
- `sel_idx`, `sel_keys` (`int64`), `sel_rev` (`float64`): `torch::empty({h_count})` each, `h_count` elements, the number of suppliers at the maximum that `q15_count_selected_kernel` counted; `h_count` is a size symbol that the kernel does not receive, with least value 1 at this launch. The kernel only writes them.
- `n`: size symbol with least value 1.
