# string_contains

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

Each block handles 256 consecutive strings in two phases. (1) Thread per
string for strings of at most 128 bytes: the thread tests the start positions
one after another; a longer string goes onto a shared-memory queue. (2) Warp
per queued long string: a step covers 256 start positions, each lane loads
the first bytes of its 8 candidates together (the lanes read consecutive
bytes), the warp votes with a ballot and stops at the first match. A
candidate is compared further only when its first byte matches, and stops at
the first mismatch. The empty pattern matches every string. This two-phase
split measured 1.5 to 4 times faster than one group size per block chosen
from the mean string length.

## Tests

n in {0, 1, 1000, 1000003} for short strings (0 to 20 bytes) and for a mix
with 2% long strings (129 to 3000 bytes, the warp path), with the patterns "",
"a", "cab", "abcda" and "dddddddd" over a 4-letter alphabet. Also long strings
with the pattern only at the start or only at the end, patterns longer than
128 bytes, a pattern longer than the string, and a sliced column whose
offsets do not start at 0. The reference is Python's `bytes` search.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
CUDA 61 us for 20 MB of short strings, 356 us for 100 MB of long strings, 178 us
for a short/long mix (280 to 340 GB/s); Numba 88 / 444 / 239 us on similar data.

## Signature notes

`chars` has length `offsets[n]` as the task states; the kernel does not assume
`offsets[0] = 0`.

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P string_contains/test_numba.py
    cd string_contains && nvcc -O3 -arch=sm_86 -o test_string_contains test.cu && CUDA_VISIBLE_DEVICES=1 ./test_string_contains

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
