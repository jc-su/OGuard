# dictionary_decode

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003}, all pass.

Approach: one thread per row, a direct gather `out[i] = dictionary[codes[i]]`.
Purely memory-bound and embarrassingly parallel; `codes` and `out` accesses
are coalesced across a warp, and the CUDA C++ version reads `dictionary`
through `__ldg` (read-only cache) since it's shared read-only input.
