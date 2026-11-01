# dictionary_decode

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

One thread per row: a coalesced read of the code, a gather from the dictionary
(a small dictionary stays in L1/L2, the CUDA version reads it through the
read-only cache), and a coalesced store. The CUDA kernel has a grid-stride
loop. The Numba kernel has none, because LLVM unrolls such a loop behind a
64-bit division that costs more than one row; `run` launches ceil(n / 256)
blocks, which covers every row.

## Tests

n in {0, 1, 1000, 1000003} with d in {1, 7, 1000, 100000}, plus n = 255, 256,
257, 65537 with d = 3. Exact comparison.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 182 us, CUDA 181 us (660 GB/s), d = 1000.

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P dictionary_decode/test_numba.py
    cd dictionary_decode && nvcc -O3 -arch=sm_86 -o test_dictionary_decode test.cu && CUDA_VISIBLE_DEVICES=1 ./test_dictionary_decode

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
