#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstring>

extern "C" __global__ void kernel(int32_t* a_offsets, uint8_t* a_chars, int32_t* b_offsets, uint8_t* b_chars, int8_t* match, int64_t n);

void test_string_equal(int64_t n) {
    printf("Testing string_equal with n=%ld\n", n);

    // Build test strings
    const char* strings[] = {"hello", "world", "test", "string"};
    int num_strs = 4;

    // Create offset and char arrays
    int32_t* h_a_offsets = new int32_t[n + 1];
    int32_t* h_b_offsets = new int32_t[n + 1];

    // Pre-calculate sizes
    int64_t a_char_total = 0, b_char_total = 0;
    for (int64_t i = 0; i < n; i++) {
        const char* a_str = strings[i % num_strs];
        const char* b_str = strings[(i + (i % 2)) % num_strs];
        a_char_total += strlen(a_str);
        b_char_total += strlen(b_str);
    }

    uint8_t* h_a_chars = new uint8_t[a_char_total > 0 ? a_char_total : 1];
    uint8_t* h_b_chars = new uint8_t[b_char_total > 0 ? b_char_total : 1];
    int8_t* h_match = n > 0 ? new int8_t[n] : new int8_t[1];

    int64_t a_char_idx = 0, b_char_idx = 0;

    for (int64_t i = 0; i < n; i++) {
        const char* a_str = strings[i % num_strs];
        const char* b_str = strings[(i + (i % 2)) % num_strs];

        h_a_offsets[i] = a_char_idx;
        int a_len = strlen(a_str);
        if (a_len > 0) memcpy(h_a_chars + a_char_idx, a_str, a_len);
        a_char_idx += a_len;

        h_b_offsets[i] = b_char_idx;
        int b_len = strlen(b_str);
        if (b_len > 0) memcpy(h_b_chars + b_char_idx, b_str, b_len);
        b_char_idx += b_len;
    }

    h_a_offsets[n] = a_char_idx;
    h_b_offsets[n] = b_char_idx;

    // Allocate device memory
    int32_t* d_a_offsets = nullptr;
    int32_t* d_b_offsets = nullptr;
    uint8_t* d_a_chars = nullptr;
    uint8_t* d_b_chars = nullptr;
    int8_t* d_match = nullptr;

    cudaMalloc(&d_a_offsets, (n + 1) * sizeof(int32_t));
    cudaMalloc(&d_b_offsets, (n + 1) * sizeof(int32_t));
    cudaMalloc(&d_a_chars, a_char_idx * sizeof(uint8_t));
    cudaMalloc(&d_b_chars, b_char_idx * sizeof(uint8_t));
    cudaMalloc(&d_match, n * sizeof(int8_t));

    // Copy to device
    cudaMemcpy(d_a_offsets, h_a_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_offsets, h_b_offsets, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_chars, h_a_chars, a_char_idx * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_chars, h_b_chars, b_char_idx * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    kernel<<<blocks, threads_per_block>>>(d_a_offsets, d_a_chars, d_b_offsets, d_b_chars, d_match, n);

    // Copy result back
    cudaMemcpy(h_match, d_match, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Verify result
    bool pass = true;
    for (int64_t i = 0; i < n; i++) {
        const char* a_str = strings[i % num_strs];
        const char* b_str = strings[(i + (i % 2)) % num_strs];
        int8_t expected = strcmp(a_str, b_str) == 0 ? 1 : 0;

        if (h_match[i] != expected) {
            pass = false;
            if (n <= 10) {
                printf("  Mismatch at i=%ld: got %d, expected %d\n", i, h_match[i], expected);
            }
        }
    }

    if (pass) {
        printf("  PASS\n");
    } else {
        printf("  FAIL\n");
    }

    // Cleanup
    cudaFree(d_a_offsets);
    cudaFree(d_b_offsets);
    cudaFree(d_a_chars);
    cudaFree(d_b_chars);
    cudaFree(d_match);
    delete[] h_a_offsets;
    delete[] h_b_offsets;
    delete[] h_a_chars;
    delete[] h_b_chars;
    delete[] h_match;
}

int main() {
    int64_t test_sizes[] = {0, 1, 1000, 1000003};
    for (int i = 0; i < 4; i++) {
        test_string_equal(test_sizes[i]);
    }
    return 0;
}
