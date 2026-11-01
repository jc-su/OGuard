# radix_scatter

Approach (Numba and CUDA): one block per tile of 2048 keys (8 per thread,
coalesced loads). Each key is ranked inside its tile with an atomic add on
a shared-memory 256-bin histogram; then one thread per bin claims the bin's
range in `out` with a single global atomic add on the cursor (at most 256
global atomics per tile instead of one per key); then each row id is stored
at bin base + rank. Order inside a bin is arbitrary, as allowed. On exit the
cursors hold the bin ends, which is what the next radix pass expects.

Tests (n in 0, 1, 1000, 1000003; shifts 0, 8, 16, 24; uniform, skewed (four
distinct keys) and sorted keys). The check verifies that `out` is a
permutation of 0..n-1, that the bytes of keys[out] are non-decreasing, and
that every cursor advanced by its bin count.

    CUDA_VISIBLE_DEVICES=1 python -P radix_scatter/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
