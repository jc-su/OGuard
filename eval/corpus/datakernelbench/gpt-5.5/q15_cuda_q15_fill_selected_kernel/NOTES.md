# q15_cuda_q15_fill_selected_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q15/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q15_fill_selected_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q15_fill_selected_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`; left out the Torch, ATen, c10, and CUDA runtime includes, `<vector>`, the host code, and the macros, other includes, and device
  functions, which the kernel does not use.

## Launch

`q15_launcher` launches the kernel once, when `selected_count > 0` and
`count_blocks > 0`, with
`<<<(n_sup + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`:
block (1024, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `totals`, `present`: the `n_sup` per-supplier sums and flags; read only.
- `n_sup`: size symbol with least value 1.
- `max_value`: the one-element maximum.
- `counter`: `torch::zeros({1})` of `int64`, the output cursor, which the
  kernel increments with `atomicAdd` (`readwrite`).
- `selected_indices` (`int64`), `selected_revenues` (`float64`):
  `torch::empty({selected_count})` each, where `selected_count` is the count
  that `q15_count_selected_kernel` found for the same `totals`, `present`,
  and `max_value`; size symbol `selected_count` with least value 1 (the launch
  needs `selected_count > 0`). The kernel only writes them. The plan relies on
  the counter starting at zero and on each supplier being visited once, so that
  the number of positions taken equals `selected_count`.
