// min_max: minimum and maximum of a float64 column without NaN.
//
// Grid-stride loop with a capped grid: each thread folds its elements into a
// register min and max, each warp reduces with xor shuffles, the eight warp
// results are reduced through shared memory, and thread 0 issues one atomic
// min and one atomic max per block. A block that saw no element (n = 0)
// skips the atomics, so the result keeps its entry values.
//
// float64 atomic min/max: for doubles with the sign bit clear the signed
// int64 order equals the double order; for doubles with the sign bit set the
// unsigned int64 order is the reverse of the double order. The sign bit is
// tested on the bit pattern so that -0.0 is ordered like a negative.

#include <math_constants.h>

__device__ __forceinline__ void atomic_min_double(double* addr, double v)
{
    const long long bits = __double_as_longlong(v);
    if (bits >= 0) atomicMin((long long*)addr, bits);
    else atomicMax((unsigned long long*)addr, (unsigned long long)bits);
}

__device__ __forceinline__ void atomic_max_double(double* addr, double v)
{
    const long long bits = __double_as_longlong(v);
    if (bits >= 0) atomicMax((long long*)addr, bits);
    else atomicMin((unsigned long long*)addr, (unsigned long long)bits);
}

extern "C" __global__ void kernel(const double* __restrict__ values,
                                  double* __restrict__ result,
                                  long long n)
{
    __shared__ double warp_min[8];
    __shared__ double warp_max[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    double lo = CUDART_INF;
    double hi = -CUDART_INF;
    const long long stride = (long long)gridDim.x * 256;
    for (long long i = (long long)blockIdx.x * 256 + tid; i < n; i += stride) {
        const double v = values[i];
        lo = fmin(lo, v);
        hi = fmax(hi, v);
    }

    for (int off = 16; off > 0; off >>= 1) {
        lo = fmin(lo, __shfl_xor_sync(0xffffffffu, lo, off));
        hi = fmax(hi, __shfl_xor_sync(0xffffffffu, hi, off));
    }
    if (lane == 0) { warp_min[wid] = lo; warp_max[wid] = hi; }
    __syncthreads();

    if (tid == 0) {
        for (int w = 1; w < 8; ++w) {
            lo = fmin(lo, warp_min[w]);
            hi = fmax(hi, warp_max[w]);
        }
        if (lo <= hi) {
            atomic_min_double(result, lo);
            atomic_max_double(result + 1, hi);
        }
    }
}
