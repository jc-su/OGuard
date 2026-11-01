# dictionary_decode

Approach (Numba and CUDA): one thread per row, out[i] = dictionary[codes[i]].
Code loads and output stores are coalesced; the dictionary gather is served
by the L1/L2 caches. The kernel trusts the contract that every code is in
[0, d) and does no bounds check.

Tests (n in 0, 1, 1000, 1000003, d in 1, 7, 4096; exact comparison):

    CUDA_VISIBLE_DEVICES=1 python -P dictionary_decode/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
