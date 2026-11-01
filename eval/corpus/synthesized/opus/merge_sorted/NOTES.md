# merge_sorted

## Status

Complete. Numba (`numba.py`): `test_numba.py` passed. CUDA C++ (`kernel.cu`):
`test.cu` passed. `triton.py` and `test_triton.py` come from another session.

## Approach

A merge-path merge. Each block produces 1792 outputs: lane 0 of warp 0 and
lane 0 of warp 1 find the splits of the tile's first and last diagonal with a
binary search (a before b on ties: a[i] <= b[k - 1 - i]), the block loads its
slices of a and b into shared memory with coalesced reads, each thread finds
its own diagonal's split in shared memory and merges 7 outputs into
registers, and the tile goes back through shared memory so the stores
coalesce. A 256-ary search for the splits measured 3 times slower: its first
rounds read hundreds of scattered sectors per block, while the probes of a
binary search close in on data the block reads next.

## Tests

For n in {0, 1, 1000, 1000003}: (m, p) = (n, 0), (0, n), (n/2, n - n/2),
(n/3, n - n/3), (n, n) and (n, 3), each with many ties (values in [-50, 50])
and with few ties (values in [-2^62, 2^62]). Also disjoint ranges in both
orders, interleaved arrays, all-equal arrays, and int64 extremes. Exact
comparison with a stable sort (CPU: `std::merge`).

## Measured time

RTX A6000 (GPU 1, not shared at the time), n = 10^7 rows unless stated:
Numba 277 us, CUDA 269 us for m = p = 5 * 10^6 (about 590 GB/s).

## How to run

From the model directory (`opus/`), with GPU 1 only:

    CUDA_VISIBLE_DEVICES=1 python -P merge_sorted/test_numba.py
    cd merge_sorted && nvcc -O3 -arch=sm_86 -o test_merge_sorted test.cu && CUDA_VISIBLE_DEVICES=1 ./test_merge_sorted

`python -P` keeps the task directory off `sys.path`, where `numba.py` would
shadow the numba package. Under pytest, use `--import-mode=importlib` for the
same reason. `test.cu` includes `kernel.cu`.

## Checks

`ruff format` and `ruff check` pass on `numba.py` and `test_numba.py`. `ty`
passes on `test_numba.py`; on `numba.py` it reports the Numba device
intrinsics (`cuda.threadIdx`, `cuda.popc`, and so on) because Numba types them
as stubs for its own compiler, not for static checkers. `cuda.compile_ptx(kernel, TYPES)`
compiles, and the PTX uses no local memory.
