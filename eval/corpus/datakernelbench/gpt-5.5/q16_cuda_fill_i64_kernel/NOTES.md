# q16_cuda_fill_i64_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q16/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `fill_i64_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `fill_i64_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`; left out the Torch, ATen, c10, `cuda.h`, and
  `cuda_runtime.h` includes, the macros `BLOCK_SIZE` and `EMPTY_KEY`, the device
  function `mix64`, and the host code, which the kernel does not use.

## Launch

`q16_count_distinct_launcher` launches the kernel twice with the same
configuration, block (256, 1, 1) and no dynamic shared memory:

- `<<<(num_groups + 255) / 256, 256>>>` on `counts` (`torch::empty({num_groups})`)
  with `n = num_groups` and `value = 0`, only when `num_groups > 0`;
- `<<<(hash_size + 255) / 256, 256>>>` on `hash_table`
  (`torch::empty({hash_size})`) with `n = hash_size` and `value = -1`, after the
  launcher has returned for `n == 0` or `num_groups == 0`; `hash_size` is the
  least power of two at least `max(2 * n, 16)`.

## Lengths and ranges

- `ptr`: `n` elements in both launches.
- `n`: size symbol with least value 1 (`num_groups > 0` in the first launch,
  `hash_size >= 16` in the second).
- `value`: 0 or -1, so `[-1, 0]`.
