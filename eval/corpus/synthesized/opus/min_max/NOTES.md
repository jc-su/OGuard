# min_max

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

A single-pass reduction on a persistent grid (as many blocks as can be
resident, from the occupancy of the compiled kernel, and at most one block per
256 rows). Each thread folds its grid-stride share into a running min and
max, the block combines them with warp shuffles and shared memory, and thread
0 applies them with one atomic min and one atomic max. The CUDA version builds
the float64 atomics from integer atomics on the bit pattern (signed max/min
when the sign bit is clear, unsigned min/max when it is set); Numba has
float64 `cuda.atomic.min/max`. With n = 0 nothing is launched and the result
stays +inf, -inf.

## Tests

n in {0, 1, 1000, 1000003} with normal values, all negative, all positive and
all equal; infinities inside the column; extremes at the first and last
position; signed zeros. Exact comparison.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 126 us, CUDA 127 us (630 GB/s).

## Signature notes

`result` is `inout`: the kernel reads the +inf, -inf it holds on entry.

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P min_max/test_numba.py
    cd min_max && nvcc -O3 -arch=sm_86 -o test_min_max test.cu && CUDA_VISIBLE_DEVICES=1 ./test_min_max

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
