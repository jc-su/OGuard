extern "C" __global__ void kernel(int* offsets, unsigned char* chars, unsigned char* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int start = offsets[idx];
        int end = offsets[idx + 1];

        // Reverse the string
        int length = end - start;
        for (int i = 0; i < length; i++) {
            out[start + i] = chars[end - 1 - i];
        }
    }
}
