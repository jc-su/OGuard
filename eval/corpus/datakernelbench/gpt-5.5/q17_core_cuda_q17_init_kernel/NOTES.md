# q17_core_cuda_q17_init_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q17/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q17_init_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q17_init_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <stdint.h>`, `#include <limits.h>`, and the macro `EMPTY_KEY`; left out `BLOCK_SIZE` and the device functions `q17_hash_u64` and `q17_lookup_key` (not used by the kernel), `<algorithm>` (host `std::min` and `std::max` only), the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q17_launcher` launches the kernel once, when `n_line > 0` and `n_parts > 0`, with
`<<<init_blocks, BLOCK_SIZE, 0, stream>>>`, `init_blocks = max(1, min((n_init + 1023) / 1024, 8192))`:
block (1024, 1, 1), no dynamic shared memory, a grid-stride loop.

## Lengths and ranges

- `hash_keys`: `torch::empty({hash_size})` of `int64`, `hash_size` elements, written only.
- `sum_qty`, `cnt_qty`: `torch::empty({n_parts})` each, `n_parts` elements, written only.
- `out`: `torch::empty({})`, a 0-dim tensor of one element, written only.
- `hash_size`: the smallest power of two of at least `4 * n_parts` and at least 2: size symbol with least value 2. `n_parts`: `part_keys.numel()`, size symbol with least value 1.
- `n_init`: `max(hash_size, n_parts)`, the loop bound; every access has its own check, so it is a plain scalar, in `[0, 2^31 - 1]`: the maximum of two sizes, each in `[0, 2^31 - 1]` (corrected 2026-10-02; it was first stated without a range, which let the loop's index wrap).
