# q08_cuda_hash_aggregate_kernel

Plan: `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/cuda/full/queries/q08/round_01.py`, copied as `plan.py`.

## Extraction

- The kernel `hash_aggregate_kernel` is renamed `kernel` and declared `extern "C"`; its body and parameters are the plan's.
- Left out the includes that the kernel does not use: `#include <torch/extension.h>`, `#include <unordered_map>`.
- Left out the macros that the kernel does not use: `#define BLOCK_SIZE 1024`.
- Left out the host code (the launchers and the pybind11 module).
- Left out the other kernels of the source: `q8_aggregate_kernel`, `count_years_kernel`, `aggregate_by_year_kernel`, `simple_aggregate_kernel`.
- No guard, bound, index, or type of a memory access is changed.

## Launch

`q8_aggregate_launcher` launches `hash_aggregate_kernel<<<256, BLOCK_SIZE, shared_mem>>>`: 256 blocks of `BLOCK_SIZE = 1024` threads with `shared_mem = num_years * sizeof(double) * 2` bytes of dynamic shared memory, which holds `sh_brazil` and `sh_total`; `launch.json` states both. The other four kernels of the source are not launched.

## Lengths and ranges

- `o_year`: `n_elements` elements: `o_year.numel()`, one element per joined row.
- `volume`: `n_elements` elements: one element per joined row.
- `is_brazil`: `n_elements` elements: one element per joined row.
- `n_elements`: `n = o_year.numel()`; `o_year.min()` before the launch fails on an empty tensor, so its least value is 1.
- `sum_brazil`: `torch::zeros({num_years})`, zeroed before the launch; the kernel adds to it with `atomicAdd`.
- `sum_total`: `torch::zeros({num_years})`, as `sum_brazil`.
- `min_year`: `int(o_year.min())`; the kernel subtracts it from each year and checks the difference against [0, num_years).
- `num_years`: `max_year - min_year + 1` from the years of `o_year`, the size symbol of the two outputs; at least 1. TPC-H order dates lie in the years 1992 to 1998, so the selected rows span at most 7 years (2 with the plan's dates): `launch.json` states the maximum 7, which also bounds the dynamic shared memory.
