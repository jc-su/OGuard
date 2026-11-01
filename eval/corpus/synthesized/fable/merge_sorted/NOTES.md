# merge_sorted

Approach (Numba and CUDA): merge path. Each block owns 1024 consecutive
outputs (256 threads, 4 per thread). Every thread runs the two merge-path
searches that bound the block's slices of `a` and `b` (a binary search on
the diagonal over the global arrays; the search takes `a[i] <= b[j]` as
"a first", which gives the required order for equal elements), the block
copies both slices to shared memory with coalesced loads, each thread finds
its 4-output slice with a merge-path search in shared memory and merges it
sequentially into a shared staging buffer, and the block stores the staging
buffer with coalesced stores. Works for m = 0, p = 0 and m + p = 0.

Tests (n in 0, 1, 1000, 1000003; (m, p) in (n, n), (n, n/3), (n/3, n),
(n, 0), (0, n); values from a range of 100 (many duplicates) and 2^63
(nearly distinct); exact comparison against a stable sort of the union):

    CUDA_VISIBLE_DEVICES=1 python -P merge_sorted/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
