#define BLOCK_SIZE 512

__device__ inline unsigned long long int atomicAdd_ull(unsigned long long int* address, unsigned long long int val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
}

extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double* __restrict__ values,
    int64_t* __restrict__ unique_keys,
    double* __restrict__ out_values,
    size_t* unique_count,
    size_t max_unique,
    size_t n
) {
    __shared__ int64_t s_keys[BLOCK_SIZE];
    __shared__ double s_vals[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    if (idx < n) {
        s_keys[tid] = keys[idx];
        s_vals[tid] = values[idx];
    } else {
        s_keys[tid] = -1;
        s_vals[tid] = 0.0;
    }
    __syncthreads();

    if (tid < blockDim.x && idx < n) {
        int64_t key = s_keys[tid];
        if (key != -1) {
            bool is_leader = true;
            for (int i = 0; i < blockDim.x; i++) {
                if (i != tid && s_keys[i] == key) {
                    is_leader = false;
                    break;
                }
            }

            if (is_leader) {
                double sum_val = s_vals[tid];
                for (int i = 0; i < blockDim.x; i++) {
                    if (i != tid && s_keys[i] == key) {
                        sum_val += s_vals[i];
                        s_keys[i] = -1;
                    }
                }

                size_t pos = atomicAdd_ull(reinterpret_cast<unsigned long long int*>(unique_count), 1);
                if (pos < max_unique) {
                    unique_keys[pos] = key;
                    out_values[pos] = sum_val;
                }
            }
        }
    }
}
