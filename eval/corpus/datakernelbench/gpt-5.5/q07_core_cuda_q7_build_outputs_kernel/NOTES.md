# q07_core_cuda_q7_build_outputs_kernel

Source: `azure__gpt-5.5/cuda/core/queries/q07/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `q7_build_outputs_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q7_build_outputs_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `NUM_GROUPS`; left out `BLOCK_SIZE`, the device function `year_from_epoch_days`, `<vector>`, the CUDA runtime include, and the Torch, ATen, and c10 includes and the host code, which the kernel does not use.

## Launch

`q7_launcher` launches the kernel once with `<<<1, 1, 0, stream>>>`: block
(1, 1, 1), no dynamic shared memory; one thread compacts the eight group sums.

## Lengths and ranges

- `revenue8_ptr`: the 8 sums of `q7_scan_kernel` (zeros when `n == 0`); read only.
- `out_supp_ptr`, `out_cust_ptr`, `out_year_ptr` (`int32`), `out_revenue_ptr` (`float64`): `torch::empty({4})` each, 4 elements, written only.
- `count_ptr`: `torch::zeros({1})` of `int32`, one element, which the kernel only writes.
