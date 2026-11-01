# sorted_contains

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

One thread per key runs a binary search for the last member <= key and
compares. The step is a select, not a branch, so the step count depends only
on s and a warp stays converged; the top levels of the search are the same
members for every key and stay in L1. Measured alternatives, a shared-memory
table of the top levels (persistent blocks) and 2 to 8 interleaved searches
per thread, were no faster: the kernel is bound by the random reads of the
bottom levels.

## Tests

s in {0, 1, 2, 3, 1000, 1000003} and n in {0, 1, 1000, 1000003}, keys half
members, some member + 1, the rest random, with the int64 minimum and maximum
as keys; plus members at both ends of the int64 range. Exact comparison with
`np.isin` (CPU: `std::binary_search`).

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 190 / 2020 us, CUDA 182 / 1983 us for s = 1000 / 10^6 (the second is
bound by random reads of an 8 MB array).

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P sorted_contains/test_numba.py
    cd sorted_contains && nvcc -O3 -arch=sm_86 -o test_sorted_contains test.cu && CUDA_VISIBLE_DEVICES=1 ./test_sorted_contains

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
