# q21_core_cuda_k

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/core/queries/q21/round_04.py`, copied as `plan.py`.

## Extraction

- The kernel `k` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- The plan takes `uint8_t` from `#include <torch/extension.h>`, which nvcc cannot include without PyTorch's headers; the extraction includes `<cstdint>` in its place.
- Left out the macros that the kernel does not use: `#define B 1024`.
- Left out the host code (the launchers and the pybind11 module).
- No guard, bound, index, or type of a memory access is changed.

## Launch

`launch` launches `k<<<g, B>>>` with `g = (n + B - 1) / B`, `B = 1024`, and no dynamic shared memory; `launch.json` states the block.

## Lengths and ranges

- `idx`: `n` elements, the rows of `l1` (lineitem joined with the nation's suppliers and the orders of the status): `torch.searchsorted(lo_orderkeys, l1_orderkey)`, a position in [0, n_agg] that the plan does not clamp; it is below `n_agg` because every `l1` order key is in `agg` (both come from lineitem joined with the same orders), which the signature cannot state.
- `l`: `n` elements: the late flag as uint8, one element per `l1` row.
- `s`: `n` elements: `supp_id`, which the plan assigns as `range(len(supp_n))` and joins to `l1` on `s_suppkey`, so every value lies in [0, num_suppliers): an index into `c`.
- `d`: `n_agg` elements: `distinct_supp` of `agg`, one element per order key of `agg`.
- `ld`: `n_agg` elements: `late_distinct_supp` of `agg`.
- `c`: `torch.zeros(num_suppliers)`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `n`: `int(idx.numel())`, the size symbol of the `l1` buffers.
- `n_agg`: the rows of `agg`, the size symbol of `d` and `ld`.
- `num_suppliers`: `len(supp_n)`, the size symbol of `c`.
