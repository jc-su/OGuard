// Test for string_length: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test
//
// Two average lengths per size exercise both kernel paths: about 12 code points
// per string (thread per string) and about 96 (warp per string). Code points are
// 1 to 4 bytes wide, so byte length and code point count differ.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const int* offsets, const uint8_t* chars, int* lengths, long long n);

#define CHECK(call)                                                                    \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static const int THREADS = 256;
static const int MAX_BLOCKS = 2048;
static int grid_for(long long n) {
    long long b = (n + THREADS - 1) / THREADS;
    if (b < 1) b = 1;
    if (b > MAX_BLOCKS) b = MAX_BLOCKS;
    return (int)b;
}

// Appends one random code point of 1 to 4 bytes (70% / 15% / 10% / 5%).
static void append_code_point(std::vector<uint8_t>& chars) {
    const unsigned w = (unsigned)(next_u64() % 20u);
    const int width = 1 + (w >= 14) + (w >= 17) + (w >= 19);
    static const uint8_t lead[5] = {0, 0, 0xC3, 0xE2, 0xF0};
    chars.push_back(width == 1 ? (uint8_t)(0x41 + next_u64() % 26u) : lead[width]);
    for (int b = 1; b < width; ++b) chars.push_back((uint8_t)(0x80 + next_u64() % 64u));
}

static bool run_case(long long n, int avg_cp, bool timed) {
    std::vector<int> offsets(n + 1);
    std::vector<int> expected(n);
    std::vector<uint8_t> chars;
    offsets[0] = 0;
    for (long long i = 0; i < n; ++i) {
        const int cp = (int)(next_u64() % (uint64_t)(2 * avg_cp + 1));
        for (int c = 0; c < cp; ++c) append_code_point(chars);
        expected[i] = cp;
        offsets[i + 1] = (int)chars.size();
    }

    int* d_offsets = nullptr;
    uint8_t* d_chars = nullptr;
    int* d_lengths = nullptr;
    CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_chars, std::max<size_t>(chars.size(), 1)));
    CHECK(cudaMalloc(&d_lengths, std::max<long long>(n, 1) * sizeof(int)));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_chars, chars.data(), chars.size(), cudaMemcpyHostToDevice));

    kernel<<<grid_for(n), THREADS>>>(d_offsets, d_chars, d_lengths, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<int> lengths(n);
    CHECK(cudaMemcpy(lengths.data(), d_lengths, n * sizeof(int), cudaMemcpyDeviceToHost));
    const bool ok = std::equal(lengths.begin(), lengths.end(), expected.begin());
    printf("  n=%lld bytes=%zu: %s\n", n, chars.size(), ok ? "ok" : "MISMATCH");

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(n), THREADS>>>(d_offsets, d_chars, d_lengths, n);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total += ms;
        }
        printf("  n=%lld bytes=%zu: %.1f us per launch\n", n, chars.size(), total / reps * 1e3f);
    }

    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_chars));
    CHECK(cudaFree(d_lengths));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    const int avgs[] = {12, 96};
    bool ok = true;
    for (long long n : sizes)
        for (int avg : avgs) ok = run_case(n, avg, n == 1000003) && ok;
    printf("%s string_length\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
