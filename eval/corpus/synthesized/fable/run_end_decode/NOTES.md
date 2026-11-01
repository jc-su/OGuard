# run_end_decode

Approach (Numba and CUDA): one thread per output row; row i belongs to the
first run j with run_ends[j] > i (upper bound). Each block first finds the run
range covering its 256 rows with two full searches (one in warp 0, one in
warp 1), then every thread searches only inside that range. With long runs the
per-row search is one or two steps; with unit runs it is the full log2(256).

Tests (n in 0, 1, 1000, 1000003; run shapes: single run, unit runs, short
random runs, long random runs; exact comparison against np.repeat):

    CUDA_VISIBLE_DEVICES=1 python -P run_end_decode/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
