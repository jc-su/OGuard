// group_sum_valid: counts[k] += number of valid rows of group k, sums[k] += their values
// (modulo 2^64).
//
// g <= 1024, so each block accumulates into a shared-memory table of g counters
// and sums and adds the nonzero entries to global memory once at the end. A warp
// whose 32 rows all fall in one group (sorted input or few groups) first reduces
// them with shuffles and issues one shared atomic instead of 32.

#define MAX_GROUPS 1024

extern "C" __global__ void kernel(const int* __restrict__ group,
                                  const long long* __restrict__ value,
                                  const unsigned char* __restrict__ valid,
                                  long long* __restrict__ counts,
                                  long long* __restrict__ sums,
                                  long long n, long long g)
{
    __shared__ unsigned long long s_count[MAX_GROUPS];
    __shared__ unsigned long long s_sum[MAX_GROUPS];
    for (long long k = threadIdx.x; k < g; k += blockDim.x) {
        s_count[k] = 0;
        s_sum[k] = 0;
    }
    __syncthreads();

    const unsigned int full = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const long long stride = (long long)gridDim.x * blockDim.x;
    // base is the same for all lanes of a warp, so the loop is warp-uniform.
    for (long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) - lane; base < n;
         base += stride) {
        const long long i = base + lane;
        const bool in_range = i < n;
        const int own = in_range ? group[i] : 0;
        // Lanes past the end take lane 0's group (row base < n) and contribute nothing.
        // The shuffle runs in every lane, as the full mask requires.
        const int lane0 = __shfl_sync(full, own, 0);
        const int k = in_range ? own : lane0;
        const bool use = in_range && ((valid[i >> 3] >> (i & 7)) & 1);
        unsigned long long c = use ? 1ull : 0ull;
        unsigned long long v = use ? (unsigned long long)value[i] : 0ull;
        if (__all_sync(full, k == __shfl_sync(full, k, 0))) {
            for (int d = 16; d > 0; d >>= 1) {
                c += __shfl_xor_sync(full, c, d);
                v += __shfl_xor_sync(full, v, d);
            }
            if (lane == 0 && c != 0) {
                atomicAdd(&s_count[k], c);
                atomicAdd(&s_sum[k], v);
            }
        } else if (use) {
            atomicAdd(&s_count[k], 1ull);
            atomicAdd(&s_sum[k], v);
        }
    }
    __syncthreads();

    for (long long k = threadIdx.x; k < g; k += blockDim.x) {
        if (s_count[k] != 0) {
            atomicAdd((unsigned long long*)&counts[k], s_count[k]);
            atomicAdd((unsigned long long*)&sums[k], s_sum[k]);
        }
    }
}
