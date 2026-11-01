#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ shipmode_ptr,
    const int32_t* __restrict__ priority_ptr,
    const int32_t* __restrict__ commitdate_ptr,
    const int32_t* __restrict__ receiptdate_ptr,
    const int32_t* __restrict__ shipdate_ptr,
    int32_t receipt_lo,
    int32_t receipt_hi,
    int64_t* __restrict__ out_high,
    int64_t* __restrict__ out_low,
    int64_t n_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Thread-local accumulation for 2 shipmodes (0=MAIL, 1=SHIP)
    int64_t h[2] = {0, 0};
    int64_t l[2] = {0, 0};
    
    // Grid-stride loop
    while (idx < n_elements) {
        int32_t commit = commitdate_ptr[idx];
        int32_t receipt = receiptdate_ptr[idx];
        int32_t ship = shipdate_ptr[idx];
        
        if (commit < receipt && ship < commit && receipt >= receipt_lo && receipt < receipt_hi) {
            int32_t sm = shipmode_ptr[idx];
            int32_t pr = priority_ptr[idx];
            
            if (pr == 1) {
                h[sm]++;
            } else {
                l[sm]++;
            }
        }
        idx += stride;
    }

    // Block-level reduction in shared memory
    __shared__ int64_t sh_h0[BLOCK_SIZE];
    __shared__ int64_t sh_h1[BLOCK_SIZE];
    __shared__ int64_t sh_l0[BLOCK_SIZE];
    __shared__ int64_t sh_l1[BLOCK_SIZE];

    sh_h0[threadIdx.x] = h[0];
    sh_h1[threadIdx.x] = h[1];
    sh_l0[threadIdx.x] = l[0];
    sh_l1[threadIdx.x] = l[1];
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh_h0[threadIdx.x] += sh_h0[threadIdx.x + offset];
            sh_h1[threadIdx.x] += sh_h1[threadIdx.x + offset];
            sh_l0[threadIdx.x] += sh_l0[threadIdx.x + offset];
            sh_l1[threadIdx.x] += sh_l1[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Global atomics from thread 0
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)&out_high[0], (unsigned long long)sh_h0[0]);
        atomicAdd((unsigned long long*)&out_high[1], (unsigned long long)sh_h1[0]);
        atomicAdd((unsigned long long*)&out_low[0], (unsigned long long)sh_l0[0]);
        atomicAdd((unsigned long long*)&out_low[1], (unsigned long long)sh_l1[0]);
    }
}
