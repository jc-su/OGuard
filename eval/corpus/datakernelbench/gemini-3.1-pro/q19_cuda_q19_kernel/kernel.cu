#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const double* __restrict__ price,
    const double* __restrict__ discount,
    const double* __restrict__ quantity,
    const int8_t* __restrict__ part_match,
    double* __restrict__ out,
    int64_t n
) {
    __shared__ double block_rev[BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_rev = 0.0;
    
    if (idx < n) {
        int8_t pm = part_match[idx];
        double qty = quantity[idx];
        bool keep = false;
        
        if (pm == 1) {
            keep = (qty >= 1.0 && qty <= 11.0);
        } else if (pm == 2) {
            keep = (qty >= 10.0 && qty <= 20.0);
        } else if (pm == 3) {
            keep = (qty >= 20.0 && qty <= 30.0);
        }
        
        if (keep) {
            my_rev = price[idx] * (1.0 - discount[idx]);
        }
    }
    
    block_rev[threadIdx.x] = my_rev;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_rev[threadIdx.x] += block_rev[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(out, block_rev[0]);
    }
}
