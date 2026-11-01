// run_end_decode: expand an Arrow run-end encoded column.
//
// One thread per output row. Row i belongs to the first run j with
// run_ends[j] > i (an upper bound search). Each block first finds the run
// range of its 256 rows with two searches over all of run_ends (one in warp
// 0, one in warp 1), then every thread searches only inside that range,
// which is a few runs wide when runs are long.

__device__ __forceinline__ long long upper_bound(const int* __restrict__ run_ends,
                                                 long long lo, long long hi, long long row)
{
    while (lo < hi) {
        const long long mid = (lo + hi) >> 1;
        if (run_ends[mid] <= row) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

extern "C" __global__ void kernel(const double* __restrict__ run_values,
                                  const int* __restrict__ run_ends,
                                  double* __restrict__ out,
                                  long long r,
                                  long long n)
{
    __shared__ long long bounds[2];
    const int tid = threadIdx.x;
    const long long t0 = (long long)blockIdx.x * 256;
    if (t0 >= n) return;
    const long long last = min(t0 + 256, n) - 1;

    if (tid == 0) bounds[0] = upper_bound(run_ends, 0, r, t0);
    else if (tid == 32) bounds[1] = upper_bound(run_ends, 0, r, last) + 1;
    __syncthreads();

    const long long i = t0 + tid;
    if (i <= last) {
        const long long j = upper_bound(run_ends, bounds[0], bounds[1], i);
        out[i] = run_values[j];
    }
}
