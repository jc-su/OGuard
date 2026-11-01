# radix_scatter

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003} and several shifts (0, 8, 16, 24), all pass.

Approach: one thread per row, `atomicAdd` on its byte's cursor to claim a
slot, then a direct write. Contention is spread across 256 independent
counters (one per byte value) rather than a single global counter, so
per-bucket contention stays proportional to n / 256 even at n ~ 1e6.
