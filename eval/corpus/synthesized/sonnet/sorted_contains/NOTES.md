# sorted_contains

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003} and various member-set sizes s (including s = 0
and s ~ n/5), all pass.

Approach: binary search over sorted, distinct `members`. One thread per
key, O(n log s) total work, fully parallel with no cross-thread state.
