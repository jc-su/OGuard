#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const double* __restrict__ qty,
    const double* __restrict__ price,
    const double* __restrict__ disc,
    const double* __restrict__ tax,
    const int*    __restrict__ rf,
    const int*    __restrict__ ls,
    const int*    __restrict__ shipdate,
    const int      shipdate_hi,
    double* __restrict__ sum_qty,
    double* __restrict__ sum_price,
    double* __restrict__ sum_disc_price,
    double* __restrict__ sum_charge,
    double* __restrict__ sum_disc,
    unsigned long long* __restrict__ cnt,
    const int64_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (shipdate[idx] <= shipdate_hi) {
        int key = rf[idx] * 256 + ls[idx];   // encodes group

        double q   = qty[idx];
        double p   = price[idx];
        double d   = disc[idx];
        double t   = tax[idx];
        double dp  = p * (1.0 - d);          // discounted price
        double ch  = dp * (1.0 + t);         // charge

        atomicAdd(&sum_qty[key],        q);
        atomicAdd(&sum_price[key],      p);
        atomicAdd(&sum_disc_price[key], dp);
        atomicAdd(&sum_charge[key],     ch);
        atomicAdd(&sum_disc[key],       d);
        atomicAdd(&cnt[key], 1ULL);
    }
}
