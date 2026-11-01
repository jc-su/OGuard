#define MAX_NATIONS 32

extern "C" __global__ void kernel(
    const int64_t* __restrict__ nation_ids,
    const double* __restrict__ price,
    const double* __restrict__ discount,
    int num_nations,
    double* __restrict__ out,
    int64_t n
) {
    // Shared memory for per‑nation block‑level partial sums
    __shared__ double block_sums[MAX_NATIONS];

    // Zero the shared memory for the relevant nations
    if (threadIdx.x < MAX_NATIONS) {
        block_sums[threadIdx.x] = 0.0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int nid = static_cast<int>(nation_ids[idx]); // 0 .. num_nations-1
        double rev = price[idx] * (1.0 - discount[idx]);
        if (nid >= 0 && nid < num_nations) {
            atomicAdd(&block_sums[nid], rev);
        }
    }
    __syncthreads();

    // Write block partial sums to global output (one atomic per nation)
    if (threadIdx.x < num_nations) {
        atomicAdd(&out[threadIdx.x], block_sums[threadIdx.x]);
    }
}
