#include <cstdint>

#define HIST_SHARED_CAP 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey,
    int64_t n_customers,
    const int32_t* __restrict__ counts,
    int64_t* __restrict__ hist,
    int64_t hist_size,
    int64_t num_customers
) {
    __shared__ unsigned long long shist[HIST_SHARED_CAP];

    bool use_shared = hist_size <= HIST_SHARED_CAP;

    if (use_shared) {
        for (int64_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
            shist[i] = 0ULL;
        }
        __syncthreads();
    }

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_customers) {
        int64_t key = c_custkey[idx];
        if (key >= 0 && key <= num_customers) {
            int32_t cnt = counts[key];
            if (cnt >= 0 && static_cast<int64_t>(cnt) < hist_size) {
                if (use_shared) {
                    atomicAdd(&shist[cnt], 1ULL);
                } else {
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(hist + cnt),
                        1ULL
                    );
                }
            }
        }
    }

    if (use_shared) {
        __syncthreads();
        for (int64_t i = threadIdx.x; i < hist_size; i += blockDim.x) {
            unsigned long long v = shist[i];
            if (v != 0ULL) {
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(hist + i),
                    v
                );
            }
        }
    }
}
