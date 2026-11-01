#include <cstdint>
#include <cstring>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* build,
    const int64_t* probe,
    int32_t* counts,
    int64_t b,
    int64_t p
);

void cpu_reference(const int64_t* build, const int64_t* probe, int64_t b, int64_t p,
                   int32_t* counts_ref) {
    for (int64_t i = 0; i < p; i++) {
        counts_ref[i] = 0;
        for (int64_t j = 0; j < b; j++) {
            if (build[j] == probe[i]) {
                counts_ref[i]++;
            }
        }
    }
}

void test_join_count(int64_t b, int64_t p) {
    // Allocate host memory
    int64_t* build_h = new int64_t[b];
    int64_t* probe_h = new int64_t[p];
    int32_t* counts_h = new int32_t[p];
    int32_t* counts_ref = new int32_t[p];

    // Initialize data
    for (int64_t i = 0; i < b; i++) {
        build_h[i] = (i * 7 + 3) % 100;
    }
    std::sort(build_h, build_h + b);

    for (int64_t i = 0; i < p; i++) {
        probe_h[i] = (i * 11 + 5) % 100;
    }

    memset(counts_h, 0, p * sizeof(int32_t));

    // Allocate device memory
    int64_t* build_d;
    int64_t* probe_d;
    int32_t* counts_d;

    cudaMalloc(&build_d, b * sizeof(int64_t));
    cudaMalloc(&probe_d, p * sizeof(int64_t));
    cudaMalloc(&counts_d, p * sizeof(int32_t));

    // Copy to device
    cudaMemcpy(build_d, build_h, b * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(probe_d, probe_h, p * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemset(counts_d, 0, p * sizeof(int32_t));

    // Launch kernel
    int grid_size = (p + 255) / 256;
    if (grid_size == 0) grid_size = 1;
    kernel<<<grid_size, 256>>>(build_d, probe_d, counts_d, b, p);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(counts_h, counts_d, p * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpu_reference(build_h, probe_h, b, p, counts_ref);

    // Check results
    for (int64_t i = 0; i < p; i++) {
        assert(counts_h[i] == counts_ref[i]);
    }

    std::cout << "Test passed for b=" << b << ", p=" << p << std::endl;

    // Cleanup
    cudaFree(build_d);
    cudaFree(probe_d);
    cudaFree(counts_d);

    delete[] build_h;
    delete[] probe_h;
    delete[] counts_h;
    delete[] counts_ref;
}

int main() {
    test_join_count(0, 1);
    test_join_count(1, 1);
    test_join_count(1000, 100);
    test_join_count(1000003, 1000);
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
