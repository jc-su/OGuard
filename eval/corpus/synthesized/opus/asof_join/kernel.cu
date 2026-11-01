// asof_join: out[i] = largest j with right[j] <= left[i] and left[i] - right[j] <= tolerance,
// or -1.
//
// The largest j with right[j] <= left[i] is upper_bound(right, left[i]) - 1. Any
// smaller j has right[j] at most as large, so it is farther from left[i]: if the
// largest j is out of tolerance, there is no match. Values are in [-2^61, 2^61],
// so left[i] - right[j] does not overflow.

extern "C" __global__ void kernel(const long long* __restrict__ left,
                                  const long long* __restrict__ right,
                                  int* __restrict__ out,
                                  long long n, long long r, long long tolerance)
{
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long x = left[i];
        long long lo = 0, hi = r;
        while (lo < hi) {
            const long long mid = lo + ((hi - lo) >> 1);
            if (right[mid] <= x) lo = mid + 1;
            else hi = mid;
        }
        const long long j = lo - 1;
        out[i] = (j >= 0 && x - right[j] <= tolerance) ? (int)j : -1;
    }
}
