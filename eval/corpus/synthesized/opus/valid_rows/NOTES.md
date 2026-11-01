# valid_rows

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

Each block owns 2048 consecutive rows and each warp 256 of them, which are 32
bytes of the bitmap: lane l reads byte l once and masks the padding bits past
row n. In step j (0 to 7) a ballot over bit j of the lanes' bytes gives a
32-bit mask, and a lane's output slot is the popcount of the mask bits below
it. The warp counts go through a block scan, and one atomic add on `count` per
block reserves the block's output range, so the stores of each step are
coalesced. The ids are not sorted (the task allows any order).

## Tests

n in {0, 1, 1000, 1000003} with densities 0, 0.03, 0.5, 0.97 and 1, plus
n = 7, 8, 9, 255, 256, 257, 2047, 2048, 2049 and 65539. The padding bits past
row n are set, so the tests check that the kernel ignores them. The check
compares `count` and the sorted ids with the reference.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 34 us, CUDA 36 us (about 600 GB/s), 50% valid rows.

## Signature notes

`rows` has `indexes_into: valid` because its values are row (bit) positions of
the bitmap, not byte positions. `count` is `inout`: it must be zero on entry
and the kernel adds to it.

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P valid_rows/test_numba.py
    cd valid_rows && nvcc -O3 -arch=sm_86 -o test_valid_rows test.cu && CUDA_VISIBLE_DEVICES=1 ./test_valid_rows

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
