# days_to_year

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

The civil-from-days algorithm (H. Hinnant) restricted to the year: shift the
epoch to 0000-03-01, split into 400-year eras of 146097 days, find the year of
the era, and add one for January and February. For the years 1 to 9999 all
the arithmetic is unsigned 32-bit and every division is by a constant; the
Numba version casts each intermediate to uint32 so that LLVM keeps 32-bit
operations (checked in the PTX). One thread per row; memory bound.

## Tests

n in {0, 1, 1000, 1000003} with random dates in the years 1 to 9999, and every
date from 0001-01-01 to 9999-12-31 (3652059 days). The NumPy reference uses
`datetime64`; the CUDA reference counts leap years and does not share the
kernel's algorithm.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 122 us, CUDA 120 us (660 GB/s).

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P days_to_year/test_numba.py
    cd days_to_year && nvcc -O3 -arch=sm_86 -o test_days_to_year test.cu && CUDA_VISIBLE_DEVICES=1 ./test_days_to_year

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
