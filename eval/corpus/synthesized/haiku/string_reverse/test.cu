#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void kernel(int* offsets, unsigned char* chars, unsigned char* out, int n) {
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

void cpu_reference(int* offsets, unsigned char* chars, unsigned char* out, int n) {
    for (int i = 0; i < n; i++) {
        int start = offsets[i];
        int end = offsets[i + 1];
        int length = end - start;
        for (int j = 0; j < length; j++) {
            out[start + j] = chars[end - 1 - j];
        }
    }
}

bool test_string_reverse(int n) {
    int* h_offsets = NULL;
    unsigned char* h_chars = NULL;
    unsigned char* h_out = NULL;
    unsigned char* h_ref = NULL;
    int total_chars = 0;

    if (n == 0) {
        h_offsets = (int*)malloc(sizeof(int));
        h_offsets[0] = 0;
        h_chars = (unsigned char*)malloc(sizeof(unsigned char));
        h_out = (unsigned char*)malloc(sizeof(unsigned char));
        h_ref = (unsigned char*)malloc(sizeof(unsigned char));
    } else {
        h_offsets = (int*)malloc((n + 1) * sizeof(int));
        h_offsets[0] = 0;

        // Create random string offsets
        for (int i = 0; i < n; i++) {
            int str_len = rand() % 100 + 1;
            total_chars += str_len;
            h_offsets[i + 1] = total_chars;
        }

        h_chars = (unsigned char*)malloc(total_chars * sizeof(unsigned char));
        h_out = (unsigned char*)malloc(total_chars * sizeof(unsigned char));
        h_ref = (unsigned char*)malloc(total_chars * sizeof(unsigned char));

        // Create random chars
        for (int i = 0; i < total_chars; i++) {
            h_chars[i] = (unsigned char)('A' + (rand() % 26));
        }
    }

    if (n > 0 || (n == 0 && total_chars == 0)) {
        // Allocate device memory
        int* d_offsets;
        unsigned char* d_chars;
        unsigned char* d_out;

        cudaMalloc(&d_offsets, (n + 1) * sizeof(int));
        if (total_chars > 0) {
            cudaMalloc(&d_chars, total_chars * sizeof(unsigned char));
            cudaMalloc(&d_out, total_chars * sizeof(unsigned char));

            cudaMemcpy(d_chars, h_chars, total_chars * sizeof(unsigned char), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out, h_out, total_chars * sizeof(unsigned char), cudaMemcpyHostToDevice);
        } else {
            // Allocate at least 1 byte
            cudaMalloc(&d_chars, sizeof(unsigned char));
            cudaMalloc(&d_out, sizeof(unsigned char));
        }

        cudaMemcpy(d_offsets, h_offsets, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel with 256 threads per block
        int threadsperblock = 256;
        int blockspergrid = (n + threadsperblock - 1) / threadsperblock;
        if (n > 0) {
            kernel<<<blockspergrid, threadsperblock>>>(d_offsets, d_chars, d_out, n);
        }

        // Copy result back
        if (total_chars > 0) {
            cudaMemcpy(h_out, d_out, total_chars * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        }

        // Get CPU reference
        cpu_reference(h_offsets, h_chars, h_ref, n);

        // Compare
        bool passed = (total_chars == 0) || (memcmp(h_out, h_ref, total_chars * sizeof(unsigned char)) == 0);
        printf("n=%d: %s\n", n, passed ? "PASS" : "FAIL");

        if (!passed && total_chars > 0) {
            printf("First 20 results (expected vs got):\n");
            for (int i = 0; i < (total_chars > 20 ? 20 : total_chars); i++) {
                printf("  [%d] expected %c, got %c\n", i, h_ref[i], h_out[i]);
            }
        }

        // Cleanup
        cudaFree(d_offsets);
        if (total_chars > 0) {
            cudaFree(d_chars);
            cudaFree(d_out);
        }
    } else {
        printf("n=%d: SKIP\n", n);
    }

    free(h_offsets);
    if (h_chars) free(h_chars);
    if (h_out) free(h_out);
    if (h_ref) free(h_ref);

    return true;
}

int main() {
    cudaSetDevice(0);

    int passed = 0, total = 0;

    for (int n : {0, 1, 1000, 10000}) {
        total++;
        if (test_string_reverse(n)) {
            passed++;
        }
    }

    printf("\nPassed %d/%d tests\n", passed, total);
    return (passed == total) ? 0 : 1;
}
