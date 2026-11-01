# q17_cuda_q17_second_pass_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q17_second_pass_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q17_second_pass_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, ATen,
  c10, and `<algorithm>` includes, the macro `MAX_BLOCKS_PER_SM`, and the host
  code, which the kernel does not use.

## Launch

`q17_launcher` launches the kernel once, after `q17_first_pass_kernel`, with
`<<<q17_blocks_for(n_lineitem), BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1), no
dynamic shared memory; the kernel strides over the rows.

## Lengths and ranges

- `l_partkey`, `l_quantity`, `l_extendedprice`: `n_lineitem = l_partkey.numel()`
  rows; the launcher checks the other two have the same `numel()`.
- `part_map`: `map_size` int32 elements, -1 or a row index below `n_parts`
  (written by `q17_build_map_kernel`); `read`.
- `sum_qty`, `cnt_qty`: `n_parts` elements each, filled by
  `q17_first_pass_kernel`; this kernel only reads them.
- `out`: `torch::zeros({1})`, one element, which the kernel adds to with
  `atomicAdd` (`readwrite`).
- `n_lineitem`, `map_size`, `n_parts`: size symbols with least value 1; the
  launcher returns for `n_lineitem == 0`, `n_parts == 0`, or `map_size <= 0`.
