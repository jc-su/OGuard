extern "C" __global__ void kernel(
    const int64_t* __restrict__ keys,
    const double*   __restrict__ sums,
    int64_t* __restrict__ out_keys,
    double*   __restrict__ out_sums,
    double threshold,
    int64_t n,
    unsigned long long* __restrict__ count_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (sums[idx] > threshold) {
        unsigned long long pos = atomicAdd(count_out, 1ULL);
        out_keys[pos] = keys[idx];
        out_sums[pos] = sums[idx];
    }
}
