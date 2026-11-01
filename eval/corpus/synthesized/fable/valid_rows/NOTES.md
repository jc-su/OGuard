# valid_rows

Approach (Numba and CUDA): one thread per row, block-level stream compaction.
Each warp ballots its valid lanes; the block prefix-sums the eight warp counts;
thread 0 claims one range of `rows` with a single atomic add on `count`; each
valid lane stores its id at block base + warp offset + rank in warp. Stores
of a warp are contiguous. Output order is by block claim order, so the test
sorts `rows[:count]` before comparing.

Tests (n in 0, 1, 1000, 1000003, densities 0%, 50%, 97%, 100%):

    CUDA_VISIBLE_DEVICES=1 python -P valid_rows/test_numba.py
    nvcc -O3 -arch=sm_86 -o test kernel.cu test.cu && CUDA_VISIBLE_DEVICES=1 ./test

Status: Numba and CUDA versions complete and passing.
