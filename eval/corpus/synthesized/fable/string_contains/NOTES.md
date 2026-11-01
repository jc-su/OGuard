# string_contains

Approach (Numba and CUDA): one thread per string; naive substring search
that compares the first pattern byte before the rest and stops at the first
occurrence. The empty pattern matches every string, and strings shorter than
the pattern never match. The pattern is read through L1 (small, read-only).
Long strings of very unequal length would favour a warp-per-string layout,
but the per-string loop is the standard choice for short strings.

Tests (n in 0, 1, 1000, 1000003; string lengths 0 to 24 over a 3-letter
alphabet; pattern lengths 0, 1, 2, 3, 6; exact comparison):

    CUDA_VISIBLE_DEVICES=1 python -P string_contains/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
