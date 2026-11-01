# run_end_decode

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

Each block decodes 1792 consecutive rows. A 256-ary search (one probe per
thread per round, `syncthreads_count`) finds the first run of the tile in
ceil(log256(r)) rounds; the first rounds probe the same positions in every
block, so they hit L2. The block stages the ends of the runs that cover the
tile in shared memory, one coalesced chunk of 256 at a time, until a chunk
reaches the tile end. Each thread decodes 7 consecutive rows: a binary search
over the staged ends for its first row, then one comparison per next row (the
next row is in the same run or the next one). The values go through shared
memory so the stores coalesce.

## Tests

n = 0 (r = 0), and n in {1, 1000, 1000003} with one run, n runs of length 1,
r = n / 1000, n / 10, n / 2 random cut points, and a skewed mix (runs of 1 to 4
rows, runs of about 2000 rows, runs of n / 3 rows). Also n = 1791, 1792, 1793,
2047, 2048, 2049, 4096 and 300000 around the tile size. Exact comparison.

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 119 / 145 / 327 us, CUDA 118 / 142 / 326 us for r = 1000 / 10^6 / 10^7
(610 to 680 GB/s).

## Signature notes

`run_ends` has `indexes_into: out` because its values are row positions (run
ends, exclusive) of the decoded column.

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P run_end_decode/test_numba.py
    cd run_end_decode && nvcc -O3 -arch=sm_86 -o test_run_end_decode test.cu && CUDA_VISIBLE_DEVICES=1 ./test_run_end_decode

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
