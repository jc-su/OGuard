// run_end_decode: expand an Arrow run-end encoded column.
//
// One thread per output row. Row i belongs to the first run j with
// run_ends[j] > i, found by an upper-bound binary search over all of
// run_ends; the upper levels of the search are shared by every thread and
// stay in L1/L2.
//
// Revision: the block-level narrowing of the search range through shared
// memory is gone, so the search bounds start at 0 and r for every thread
// and no index that reaches memory passes through shared memory. Every
// probe of `run_ends` and the gather from `run_values` test the exact index
// they use against [0, r); under the task's contract those tests always
// pass.

#define THREADS 256

extern "C" __global__ void kernel(const double* __restrict__ run_values,
                                  const int* __restrict__ run_ends,
                                  double* __restrict__ out,
                                  long long r,
                                  long long n)
{
    const long long i = (long long)blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    long long lo = 0, hi = r;
    while (lo < hi) {
        const long long mid = (lo + hi) >> 1;
        if (mid < 0 || mid >= r) break;
        if (run_ends[mid] <= i) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= 0 && lo < r) out[i] = run_values[lo];
}
