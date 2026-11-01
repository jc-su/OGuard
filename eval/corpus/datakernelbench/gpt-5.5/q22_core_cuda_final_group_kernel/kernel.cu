#include <cstdint>

#define MAX_CODE 100

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_mask,
    const uint8_t* __restrict__ has_order,
    const double* __restrict__ avg_sum,
    const int64_t* __restrict__ avg_count,
    int64_t key_space,
    int64_t* __restrict__ global_counts,
    double* __restrict__ global_sums,
    int64_t n
) {
    __shared__ unsigned long long sh_counts[MAX_CODE];
    __shared__ double sh_sums[MAX_CODE];

    for (int i = threadIdx.x; i < MAX_CODE; i += blockDim.x) {
        sh_counts[i] = 0ULL;
        sh_sums[i] = 0.0;
    }
    __syncthreads();

    unsigned long long cnt = (unsigned long long)avg_count[0];
    double avg = avg_sum[0] / (double)cnt;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && in_mask[idx]) {
        double bal = c_acctbal[idx];
        if (bal > avg) {
            int64_t key = c_custkey[idx];
            bool no_order = true;
            if (key >= 0 && key < key_space) {
                no_order = (has_order[key] == 0);
            }
            if (no_order) {
                int32_t code = cntrycode[idx];
                if (code >= 0 && code < MAX_CODE) {
                    atomicAdd(&sh_counts[code], 1ULL);
                    atomicAdd(&sh_sums[code], bal);
                }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < MAX_CODE) {
        unsigned long long c = sh_counts[threadIdx.x];
        double s = sh_sums[threadIdx.x];
        if (c != 0ULL) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&global_counts[threadIdx.x]), c);
            atomicAdd(&global_sums[threadIdx.x], s);
        }
    }
}
