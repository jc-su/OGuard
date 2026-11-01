// q6_sum: TPC-H Q6 filtered sum of price * discount.
//
// Each thread accumulates a private sum over a grid-stride loop, evaluating the
// cheap date predicate first so the other columns are read only for candidate
// rows. The block then reduces with warp shuffles and shared memory, and one
// atomicAdd per block accumulates into result.
//
// result must be zero on entry.

extern "C" __global__ void kernel(const int* __restrict__ shipdate,
                                  const double* __restrict__ discount,
                                  const double* __restrict__ quantity,
                                  const double* __restrict__ price,
                                  int date_lo,
                                  int date_hi,
                                  double disc_lo,
                                  double disc_hi,
                                  double qty_max,
                                  double* __restrict__ result,
                                  long long n) {
    double acc = 0.0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int d = shipdate[i];
        if (d >= date_lo && d < date_hi) {
            const double disc = discount[i];
            const double qty = quantity[i];
            if (disc >= disc_lo && disc <= disc_hi && qty < qty_max) acc += price[i] * disc;
        }
    }

#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);

    __shared__ double warp_sums[32];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0u) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0u) {
        acc = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0u) atomicAdd(result, acc);
    }
}
