#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstring>
#include <cctype>

extern "C" __global__ void kernel(int32_t* offsets, uint8_t* chars, uint8_t* out, int64_t n);

void test_ascii_upper(int64_t n) {
    printf("Testing ascii_upper with n=%ld\n", n);

    // Build test strings
    const char* strings[] = {"hello", "World", "TeSt123", "abc"};
    int num_strs = 4;

    // Create offset and char arrays
    int32_t* h_offsets = new int32_t[n + 1];

    // Pre-calculate char buffer size
    int64_t char_total = 0;
    for (int64_t i = 0; i < n; i++) {
        const char* s = strings[i % num_strs];
        char_total += strlen(s);
    }

    uint8_t* h_chars = new uint8_t[char_total > 0 ? char_total : 1];
    uint8_t* h_out = new uint8_t[char_total > 0 ? char_total : 1];

    int64_t char_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        const char* s = strings[i % num_strs];
        h_offsets[i] = char_idx;
        int len = strlen(s);
        if (len > 0) memcpy(h_chars + char_idx, s, len);
        char_idx += len;
    }
    h_offsets[n] = char_idx;

    // Allocate device memory
    int32_t* d_offsets = nullptr;
    uint8_t* d_chars = nullptr;
    uint8_t* d_out = nullptr;

    cudaMalloc(&d_offsets, (n + 1) * sizeof(int32_t));
    cudaMalloc(&d_chars, char_idx * sizeof(uint8_t));
    cudaMalloc(&d_out, char_idx * sizeof(uint8_t));

    // Copy to device
    cudaMemcpy(d_offsets, h_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chars, h_chars, char_idx * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (char_idx + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_offsets, d_chars, d_out, n);

    // Copy result back
    cudaMemcpy(h_out, d_out, char_idx * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Verify result
    bool pass = true;
    for (int i = 0; i < char_idx; i++) {
        uint8_t expected = h_chars[i];
        if (expected >= 97 && expected <= 122) {
            expected = expected - 32;
        }

        if (h_out[i] != expected) {
            pass = false;
            if (char_idx <= 50) {
                printf("  Mismatch at i=%d: got %c, expected %c\n", i, h_out[i], expected);
            }
        }
    }

    if (pass) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    // Cleanup
    cudaFree(d_offsets);
    cudaFree(d_chars);
    cudaFree(d_out);
    delete[] h_offsets;
    delete[] h_chars;
    delete[] h_out;
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int i = 0; i < 4; i++) {
        test_ascii_upper(test_sizes[i]);
    }
    return 0;
}
