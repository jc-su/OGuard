# min_max

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003}, all pass.

Approach: block-level tree reduction in shared memory (each thread loads
at most one element), then only the block's single min and single max are
atomically merged into the global result -- ceil(n / 256) atomics instead
of n. CUDA C++ has no native atomicMin/atomicMax for double, so those are
implemented with a small atomicCAS retry loop on the bit pattern; Numba's
`cuda.atomic.min`/`cuda.atomic.max` support float64 natively. For n = 0
the launch is skipped and result keeps its +inf/-inf sentinel values, per
spec.
