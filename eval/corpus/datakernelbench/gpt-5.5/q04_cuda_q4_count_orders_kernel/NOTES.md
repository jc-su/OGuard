# q04_cuda_q4_count_orders_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q04/round_03.py` (`output_round` 3),
copied as `plan.py`. The kernel is `q4_count_orders_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q4_count_orders_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `MAX_PRIORITIES`; left out the Torch,
  ATen, and c10 includes, the macro `BLOCK_SIZE` (used only by the host code),
  the other kernel, and the host code.

## Launch

`q4_launcher` launches the kernel after `q4_mark_late_orders_kernel`, when
`n_orders > 0 && num_priorities > 0`, with
`<<<(n_orders + 1023) / 1024, BLOCK_SIZE, 0, stream>>>`: block (1024, 1, 1), no
dynamic shared memory.

## Lengths and ranges

- `o_orderkey`, `o_orderdate_days`, `o_priority_codes`: `n_orders` rows each;
  `n_orders = o_orderkey.numel()` and the launcher checks the other two.
- `late_bitmap`: `bitmap_size = max(l_orderkey) + 1` bytes, which the previous
  kernel wrote; read only. The bound of `o_orderkey` against it is a foreign
  key relation, so `o_orderkey` is `read`; the kernel checks the key itself.
- `counts`: `torch::zeros({num_priorities})` of int64, which the kernel adds to
  with `atomicAdd` (`readwrite`).
- `o_priority_codes`: the category codes of `o_orderpriority` (no nulls in
  TPC-H), in [0, num_priorities - 1] by construction: `index-in` into `counts`.
- `num_priorities`: the category count; `TORCH_CHECK` bounds it by
  `MAX_PRIORITIES` (256) and the launch needs it positive: least value 1,
  `size_maxima` 256 in `launch.json`. At scale factor 10 it is 5.
- `n_orders`, `bitmap_size`: size symbols with least value 1.
- `date_lo_days`, `date_hi_days`: comparison thresholds only.
