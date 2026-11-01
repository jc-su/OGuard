# rolling_sum

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

The van Herk / Gil-Werman scheme: with segments of `window` rows, every window
holds one segment start c, and out[i] = suffix(i - window + 1) + prefix(i), a
sum of two segment partial sums and never a difference, so there is no
cancellation (a difference of prefix sums fails the spike test below). Each
block computes 1024 outputs from the 1024 + window - 1 inputs its windows
cover, staged in shared memory. The segmented prefix and suffix sums come from
per-thread sequential scans of short chunks, one warp-level segmented scan of
the chunk totals per direction, and a carry added at output time only when an
output's segment crosses its chunk boundary. The RTX A6000 runs float64 adds
at 1/64 of the float32 rate (measured 2.6e11 adds/s), so the kernel keeps
them near 3 per output plus a few for the chunk scans; an earlier version
with about twice as many adds was up to 1.9 times slower. `run` raises `ValueError` for a window
outside [1, 1024].

## Tests

n in {0, 1, 1000} with windows 1, 2, 3, 7, 32, 255, 256, 257, 1000, 1023, 1024;
n = 1000003 with windows 1, 7, 256, 1024; n = 1023, 1024, 1025, 2047, 2048,
2049, 5000 with windows 1, 5, 1000, 1024; and 100000 values with 1e15 at row
5 (windows 1, 16, 1024), where the windows without the spike must stay
exact. The reference sums every window directly; tolerance 1e-9 absolute
plus 1e-9 relative.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 239 / 325 / 338 us, CUDA 239 / 297 / 319 us for windows 1 / 16 / 1024.
Window 1 runs at the memory bound (670 GB/s); larger windows are bound by the
float64 adds.

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P rolling_sum/test_numba.py
    cd rolling_sum && nvcc -O3 -arch=sm_86 -o test_rolling_sum test.cu && CUDA_VISIBLE_DEVICES=1 ./test_rolling_sum

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
