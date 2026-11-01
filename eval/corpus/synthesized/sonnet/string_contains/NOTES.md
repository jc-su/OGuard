# string_contains

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003}, plus an empty-pattern case, all pass.

Approach: one thread per row does a brute-force byte-substring scan
(O(len_i * k)) of its own string. `pattern` is small and read by every
thread in the block, so each block stages it into dynamic shared memory
once (a cooperative strided copy) instead of every thread re-reading it
from global memory on every candidate start offset.
