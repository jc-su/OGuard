extern "C" __global__ void kernel(const int* left_idx, const int* right_idx, const double* left,
                                  const double* right, double* out, long long m, long long l, long long r) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride)
        out[i] = left[left_idx[i]] + right[right_idx[i]];
}
