# rolling_sum

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003} across several window sizes, including window = 1,
window > n, and window = 1024 (the max), all pass.

Approach: shared-memory tiling. Each 256-thread block loads its 256
output rows' worth of input plus the `window - 1` rows of halo before
them into dynamic shared memory once (zero-padded past the start of the
array), then every thread sums its own window directly out of shared
memory. This reads every input element from global memory exactly once,
regardless of how much windows overlap, instead of re-reading up to
1024 elements per output from global memory.
