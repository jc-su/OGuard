#include <cstdint>

__device__ __forceinline__ int find_code_idx(
    int32_t code,
    const int32_t* __restrict__ codes,
    int num_codes
) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        if (i >= num_codes) break;
        if (codes[i] == code) return i;
    }
    for (int i = 16; i < num_codes; ++i) {
        if (codes[i] == code) return i;
    }
    return -1;
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    const double* __restrict__ c_acctbal,
    const int32_t* __restrict__ cntrycode,
    const uint8_t* __restrict__ order_flags,
    int64_t flags_len,
    const int32_t* __restrict__ codes,
    int num_codes,
    const double* __restrict__ avg_stats,
    int64_t* __restrict__ out_counts_i64,
    double* __restrict__ out_sums,
    int64_t n_customers
) {
    extern __shared__ unsigned char shmem[];
    unsigned long long* sh_counts = reinterpret_cast<unsigned long long*>(shmem);
    double* sh_sums = reinterpret_cast<double*>(shmem + num_codes * sizeof(unsigned long long));

    for (int i = threadIdx.x; i < num_codes; i += blockDim.x) {
        sh_counts[i] = 0ULL;
        sh_sums[i] = 0.0;
    }
    __syncthreads();

    double avg_val = avg_stats[0] / avg_stats[1];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        double bal = c_acctbal[idx];
        if (bal > avg_val) {
            int32_t code = cntrycode[idx];
            int code_idx = find_code_idx(code, codes, num_codes);
            if (code_idx >= 0) {
                int64_t custkey = c_custkey[idx];
                bool no_order = true;
                if (custkey >= 0 && custkey < flags_len) {
                    no_order = (order_flags[custkey] == static_cast<uint8_t>(0));
                }
                if (no_order) {
                    atomicAdd(&sh_counts[code_idx], 1ULL);
                    atomicAdd(&sh_sums[code_idx], bal);
                }
            }
        }
    }
    __syncthreads();

    unsigned long long* out_counts = reinterpret_cast<unsigned long long*>(out_counts_i64);
    for (int i = threadIdx.x; i < num_codes; i += blockDim.x) {
        unsigned long long cnt = sh_counts[i];
        double sum = sh_sums[i];
        if (cnt != 0ULL) {
            atomicAdd(&out_counts[i], cnt);
            atomicAdd(&out_sums[i], sum);
        }
    }
}
