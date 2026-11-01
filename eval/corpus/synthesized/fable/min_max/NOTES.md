# min_max

Approach (Numba and CUDA): grid-stride loop over a grid capped at 2048
blocks; register min/max per thread, xor-shuffle reduction per warp, shared
memory reduction over the eight warps, then one atomic min and one atomic max
per block from thread 0. A block that saw no element skips its atomics, so
with n = 0 the result keeps +inf and -inf.

Numba uses cuda.atomic.min/max on float64. CUDA has no float64 atomic
min/max, so kernel.cu maps them onto 64-bit integer atomics: with the sign
bit clear the signed int64 order equals the double order, with the sign bit
set the unsigned int64 order is the reverse. The branch tests the sign bit of
the bit pattern rather than `v >= 0`, which keeps -0.0 ordered as a negative.

Tests (n in 0, 1, 1000, 1000003; mixed-sign, all-positive and all-negative
data; exact comparison):

    CUDA_VISIBLE_DEVICES=1 python -P min_max/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
