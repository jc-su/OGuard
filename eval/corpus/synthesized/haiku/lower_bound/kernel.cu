extern "C" __global__ void kernel(long long* keys, long long* sorted_arr, int* out, long long n, long long s) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long key = keys[idx];
        // Count how many elements in sorted < key
        long long count = 0;
        for (long long i = 0; i < s; i++) {
            if (sorted_arr[i] < key) {
                count++;
            } else {
                break;  // Array is sorted
            }
        }
        out[idx] = (int)count;
    }
}
