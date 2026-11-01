# rolling_sum

Approach (Numba and CUDA): one block per 256 outputs. The block loads its
tile plus the window - 1 preceding values (at most 1279 doubles) into shared
memory, computes an inclusive prefix sum over them (up to five block-wide
scan passes: warp shuffle scan, cross-warp carry), and writes
out[i] = P[i] - P[i - window] (P[i] alone when the window starts at row 0).
Cost per output is two shared reads instead of `window` reads, which matters
for windows near 1024.

Numerics: the prefix sums are block-local, so each is a sum of at most 1279
terms and the rounding error of the difference is of the same order as the
error of summing the window directly (about 1e-13 for values in [-1, 1]).
The test compares against 80-bit prefix sums with rtol 1e-12 and atol 1e-9.

Tests (n in 0, 1, 1000, 1000003; windows 1, 2, 7, 256, 1000, 1024):

    CUDA_VISIBLE_DEVICES=1 python -P rolling_sum/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
