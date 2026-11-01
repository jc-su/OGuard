# q17_cuda_q17_first_pass_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q17_first_pass_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q17_first_pass_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and `<algorithm>`
  includes, the macros `BLOCK_SIZE` and `MAX_BLOCKS_PER_SM`, and the host code,
  which the kernel does not use.

## Launch

`q17_launcher` launches the kernel once, after `q17_build_map_kernel`, with
`<<<q17_blocks_for(n_lineitem), BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1), no
dynamic shared memory; the kernel strides over the rows.

## Lengths and ranges

- `l_partkey`, `l_quantity`: the lineitem columns, `n_lineitem = l_partkey.numel()`
  rows; the launcher checks that `l_quantity` has the same `numel()`.
  `l_partkey` is `read`: `part_map` is sized from the filtered part keys, not from
  `l_partkey`.
- `part_map`: `map_size` int32 elements, -1 from `cudaMemsetAsync(0xff)` or a
  filtered-part row index below `n_parts` written by `q17_build_map_kernel`;
  it holds the sentinel -1, so `read`.
- `sum_qty`, `cnt_qty`: `torch::zeros({n_parts})`, which the kernel adds to with
  `atomicAdd` (`readwrite`); `n_parts` is the number of filtered part keys, which
  the kernel does not receive.
- `n_lineitem`, `map_size`, `n_parts`: size symbols with least value 1; the
  launcher returns for `n_lineitem == 0`, `n_parts == 0`, or `map_size <= 0`.
