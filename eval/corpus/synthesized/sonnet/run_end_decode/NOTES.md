# run_end_decode

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003}, all pass.

Approach: binary search over run_ends. One thread per output row i finds
the smallest run index j with run_ends[j] > i (an upper_bound), via
binary search over the r run boundaries, then reads run_values[j]. This
is O(n log r) total work, fully parallel across output rows with no
cross-thread state.
