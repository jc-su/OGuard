// sorted_contains: membership of each key in a sorted, distinct member list.
//
// One thread per key: lower-bound binary search in `members`, then an
// equality test. The upper levels of the search tree are shared by every
// thread and stay in L1/L2; s = 0 gives no match.

extern "C" __global__ void kernel(const long long* __restrict__ keys,
                                  const long long* __restrict__ members,
                                  signed char* __restrict__ match,
                                  long long n,
                                  long long s)
{
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const long long key = keys[i];
    long long lo = 0, hi = s;
    while (lo < hi) {
        const long long mid = (lo + hi) >> 1;
        if (members[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    match[i] = (lo < s && members[lo] == key) ? 1 : 0;
}
