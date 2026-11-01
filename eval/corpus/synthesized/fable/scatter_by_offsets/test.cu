// Test for scatter_by_offsets: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const int8_t* flags, const int* offsets, int* out, long long n, long long total);

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

static bool run_case(long long n, bool timed) {
    std::vector<int8_t> flags(n);
    std::vector<int> offsets(n);
    std::vector<int> expected;
    int running = 0;
    for (long long i = 0; i < n; ++i) {
        flags[i] = (int8_t)(next_u64() & 1u);
        offsets[i] = running;
        if (flags[i]) {
            expected.push_back((int)i);
            ++running;
        }
    }
    const long long total = running;

    int8_t* d_flags = nullptr;
    int* d_offsets = nullptr;
    int* d_out = nullptr;
    CHECK(cudaMalloc(&d_flags, std::max<long long>(n, 1) * sizeof(int8_t)));
    CHECK(cudaMalloc(&d_offsets, std::max<long long>(n, 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_out, std::max<long long>(total, 1) * sizeof(int)));
    CHECK(cudaMemcpy(d_flags, flags.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0xff, std::max<long long>(total, 1) * sizeof(int)));

    kernel<<<grid_for(n), THREADS>>>(d_flags, d_offsets, d_out, n, total);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<int> out(total);
    CHECK(cudaMemcpy(out.data(), d_out, total * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = std::equal(out.begin(), out.end(), expected.begin());
    printf("  n=%lld: %s (total=%lld)\n", n, ok ? "ok" : "MISMATCH", total);

    if (ok && timed) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        const int reps = 20;
        float total_ms = 0.0f;
        for (int r = 0; r < reps; ++r) {
            CHECK(cudaEventRecord(start));
            kernel<<<grid_for(n), THREADS>>>(d_flags, d_offsets, d_out, n, total);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        printf("  n=%lld: %.1f us per launch\n", n, total_ms / reps * 1e3f);
    }

    CHECK(cudaFree(d_flags));
    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_out));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, n == 1000003) && ok;
    printf("%s scatter_by_offsets\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
