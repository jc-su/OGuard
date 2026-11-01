# q08_core_cuda_market_share_kernel

Kernel `market_share_kernel` of the plan `logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/cuda/core/queries/q08/round_09.py` (DataKernelBench commit `43c361d`; `results.json` has `"correct": true` and `"output_round": 9`), copied here as `plan.py`.

## Extraction edits

- Renamed the kernel `market_share_kernel` to `kernel` and declared it `extern "C"`.
- Left out `#include <torch/extension.h>` with the host code that uses it (the launcher and `PYBIND11_MODULE`); the kernel's `int32_t` and `int64_t` are the fixed-width types that nvcc's implicit CUDA headers declare.
- Kept `#define BLOCK_SIZE 1024`, the size of the kernel's shared arrays.

## Launch

`market_share_launcher` (plan lines 54-72) launches `market_share_kernel<<<(n + 1023) / 1024, 1024>>>` (lines 62-71) with 0 bytes of dynamic shared memory, `n = o_year.numel()` (line 61); `_query_core` calls it at lines 107-113 and `run_query` at line 187.

## Lengths and ranges

- `o_year` (read, int32, length `n_elements`): `o_year_t` (line 181), the order year of each row of `li` (lines 179-180); the kernel does not access it.
- `volume` (read, float64, length `n_elements`): `price_t * (1.0 - disc_t)` (lines 182-184).
- `is_brazil` (read, bool, length `n_elements`): `li["n_name"] == nation_name` as bool (line 185).
- `sum_brazil` (readwrite, float64, length `(n_elements + 1023) / 1024`): `torch.zeros(blocks)` with `blocks = (n + 1023) // 1024` (lines 102-104), one element per block of the plan's grid; the kernel adds into it with `atomicAdd`. The quotient gives the buffer its own size symbol, `sum_brazil`, least value 0 (`n` can be 0).
- `sum_total` (readwrite, float64, length `(n_elements + 1023) / 1024`): `torch.zeros(blocks)` (line 105), the same length as `sum_brazil`, whose symbol it shares.
- `n_elements` (size, int64): `o_year.numel()` (line 61); the three columns come from the same `li`.
- size symbol `n_elements`, least value 0.
- size symbol `sum_brazil`, least value 0.
- The kernel is correct only for a grid of at most `(n_elements + 1023) / 1024` blocks, a bound in a size symbol, which the claim language does not take; it has no grid claim.
