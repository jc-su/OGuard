# radix_scatter

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

Each block owns 2048 consecutive rows. A warp groups its lanes by digit with
`match_any_sync`, and the lowest lane of each group adds the group size to the
block's shared histogram (one shared atomic per distinct digit per warp),
which gives each row its rank within its digit. Thread d scans the histogram
and reserves the digit's output range with one atomic add on `cursors[d]`
(one global atomic per nonzero digit per block). The block then sorts its
rows by digit in shared memory and writes them out in that order, so the
stores of a warp go to a few contiguous runs. The order within a digit is not
stable (the task allows any order).

## Tests

n in {0, 1, 1000, 1000003} with random keys at shifts 0, 8, 16, 24, a
constant key, sequential keys, and three distinct digits; n = 255, 256, 257,
2047, 2048, 2049 with random keys. The check requires a permutation of the
row ids, every row inside its digit's range, and every cursor advanced by its
digit count.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 180 us, CUDA 159 us, both including a 1 KB copy that resets the
cursors (about 500 GB/s of key reads and id writes).

## Signature notes

`cursors` is `inout` with `indexes_into: out`; `out` has `indexes_into: keys`
(it holds row ids).

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P radix_scatter/test_numba.py
    cd radix_scatter && nvcc -O3 -arch=sm_86 -o test_radix_scatter test.cu && CUDA_VISIBLE_DEVICES=1 ./test_radix_scatter

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
