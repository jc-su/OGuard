# q05_core_cuda_q5_sort_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q05/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q5_sort_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q5_sort_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the macros `BLOCK_SIZE` and `MAX_NATIONS`, `<algorithm>`, the pybind11 include, the namespace alias, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q5_launcher` launches the kernel once, when `num_nations > 0`, with
`<<<1, 1, 0, stream>>>`: block (1, 1, 1), no dynamic shared memory; one thread
sorts the small nation vector.

## Lengths and ranges

- `revenue`: the `num_nations` sums of `q5_aggregate_kernel` (zeros when `n == 0`); read only.
- `sorted_ids`: `torch::empty({num_nations})` of `int64`, which the kernel writes and reads back during the insertion sort, `key_id = sorted_ids[i]` and `sorted_ids[j] > key_id` (`readwrite`).
- `sorted_revenue`: `torch::empty({num_nations})`, which the kernel writes and reads back during the insertion sort (`readwrite`).
- `num_nations`: size symbol with least value 1; the launcher returns before the launch when it is 0.
