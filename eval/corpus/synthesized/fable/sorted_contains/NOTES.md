# sorted_contains

Approach (Numba and CUDA): one thread per key, lower-bound binary search in
`members` followed by an equality test; s = 0 gives no match. The first
levels of the search tree are read by every thread and stay in L1/L2. A
hash set would be the choice for a very large member list, but the input is
given sorted, so the search needs no extra build step.

Tests (n in 0, 1, 1000, 1000003; member counts 0, 1, 100, 100000; half of
the keys drawn from the members; exact comparison against np.isin):

    CUDA_VISIBLE_DEVICES=1 python -P sorted_contains/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
