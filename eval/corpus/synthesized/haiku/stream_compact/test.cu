#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* values,
    const int8_t* keep,
    int64_t* out,
    int32_t* count,
    int64_t n
);

void cpu_reference(const int64_t* values, const int8_t* keep, int64_t n,
                   int64_t* out_ref, int32_t* count_ref) {
    *count_ref = 0;
    for (int64_t i = 0; i < n; i++) {
        if (keep[i] != 0) {
            out_ref[*count_ref] = values[i];
            (*count_ref)++;
        }
    }
}

void test_stream_compact(int64_t n) {
    // Allocate host memory
    int64_t* values_h = new int64_t[n];
    int8_t* keep_h = new int8_t[n];
    int64_t* out_h = new int64_t[n];
    int32_t count_h = 0;

    int64_t* out_ref = new int64_t[n];
    int32_t count_ref = 0;

    // Initialize input data
    for (int64_t i = 0; i < n; i++) {
        values_h[i] = i;
        keep_h[i] = (i % 3 != 0) ? 1 : 0;
    }

    // Allocate device memory
    int64_t* values_d;
    int8_t* keep_d;
    int64_t* out_d;
    int32_t* count_d;

    cudaMalloc(&values_d, n * sizeof(int64_t));
    cudaMalloc(&keep_d, n * sizeof(int8_t));
    cudaMalloc(&out_d, n * sizeof(int64_t));
    cudaMalloc(&count_d, sizeof(int32_t));

    // Copy to device
    cudaMemcpy(values_d, values_h, n * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(keep_d, keep_h, n * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemset(count_d, 0, sizeof(int32_t));

    // Launch kernel
    int grid_size = (n + 255) / 256;
    kernel<<<grid_size, 256>>>(values_d, keep_d, out_d, count_d, n);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(out_h, out_d, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count_h, count_d, sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(values_h, keep_h, n, out_ref, &count_ref);

    // Check results
    assert(count_h == count_ref);

    // Sort both arrays and compare
    std::sort(out_h, out_h + count_h);
    std::sort(out_ref, out_ref + count_ref);

    for (int32_t i = 0; i < count_h; i++) {
        assert(out_h[i] == out_ref[i]);
    }

    std::cout << "Test passed for n=" << n << ": " << count_h << " elements" << std::endl;

    // Cleanup
    cudaFree(values_d);
    cudaFree(keep_d);
    cudaFree(out_d);
    cudaFree(count_d);

    delete[] values_h;
    delete[] keep_h;
    delete[] out_h;
    delete[] out_ref;
}

int main() {
    for (int64_t n : {0, 1, 1000, 1000003}) {
        test_stream_compact(n);
    }
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
