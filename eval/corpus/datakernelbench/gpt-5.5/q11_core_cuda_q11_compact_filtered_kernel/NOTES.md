# q11_core_cuda_q11_compact_filtered_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q11/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q11_compact_filtered_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q11_compact_filtered_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macro `BLOCK_SIZE`, `<cuda.h>`, the CUDA runtime include, `<vector>`, the Thrust includes, the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q11_launcher` launches the kernel once, when `out_count > 0`, with
`<<<(dense_len + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>`: block
(256, 1, 1), no dynamic shared memory, `write_tensor = torch::zeros({1})` as
`write_ptr`.

## Lengths and ranges

- `dense_values_ptr`, `present_ptr`: the `dense_len` sums and flags of `q11_aggregate_kernel`; read only.
- `dense_len`: `max(ps_partkey) + 1`, size symbol with least value 1.
- `out_keys_ptr`, `out_values_ptr`: `torch::empty({out_count})` each, `out_count` elements, the number of passing entries that `q11_count_filtered_kernel` counted; `out_count` is a size symbol that the kernel does not receive, with least value 1 at this launch. The kernel only writes them.
- `write_ptr`: the one-element `write_tensor`, zeroed before the launch, used as an atomic cursor (`readwrite`).
- `threshold`: a comparison threshold only.
