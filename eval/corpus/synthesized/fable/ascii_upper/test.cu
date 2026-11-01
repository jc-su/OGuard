// Test the ascii_upper kernel against a CPU reference.
// Build: nvcc -arch=sm_86 -O2 -o ascii_upper_test kernel.cu test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

extern "C" __global__ void kernel(const int* offsets,
                                  const unsigned char* chars,
                                  unsigned char* out,
                                  long long n);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(err));                              \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

static bool check(long long n, std::mt19937_64& rng) {
    std::vector<int> offsets(n + 1, 0);
    std::vector<unsigned char> chars;
    for (long long i = 0; i < n; ++i) {
        int len = (int)(rng() % 17);
        for (int j = 0; j < len; ++j) {
            // Half the bytes are ASCII letters, the rest any byte value.
            unsigned char c = (rng() % 2) ? (unsigned char)(65 + rng() % 58) : (unsigned char)(rng() % 256);
            chars.push_back(c);
        }
        offsets[i + 1] = (int)chars.size();
    }
    size_t total = chars.size();
    std::vector<unsigned char> expected(total);
    for (size_t j = 0; j < total; ++j) {
        unsigned char c = chars[j];
        expected[j] = (c >= 'a' && c <= 'z') ? (unsigned char)(c - 32) : c;
    }

    int* d_offsets;
    unsigned char *d_chars, *d_out;
    CUDA_CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chars, total + 1));
    CUDA_CHECK(cudaMalloc(&d_out, total + 1));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_chars, chars.data(), total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, total + 1));

    unsigned blocks = (unsigned)((total + 255) / 256);
    if (blocks == 0) blocks = 1;
    kernel<<<blocks, 256>>>(d_offsets, d_chars, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> out(total);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, total, cudaMemcpyDeviceToHost));
    bool ok = total == 0 || std::memcmp(out.data(), expected.data(), total) == 0;

    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_chars));
    CUDA_CHECK(cudaFree(d_out));
    std::printf("n=%lld: %s (%zu bytes)\n", n, ok ? "PASS" : "FAIL", total);
    return ok;
}

int main() {
    std::mt19937_64 rng(4);
    const long long sizes[] = {0, 1, 1000, 1000003};
    int failures = 0;
    for (long long n : sizes) {
        if (!check(n, rng)) ++failures;
    }
    return failures ? 1 : 0;
}
