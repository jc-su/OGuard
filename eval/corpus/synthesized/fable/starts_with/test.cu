// Test for starts_with: launches kernel.cu and checks against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 kernel.cu test.cu -o test
//
// Strings are random ASCII of length 0 to 20; about 30% are forced to begin
// with the pattern. Every size is checked with k=3 ("abc"), k=0 (every string
// matches) and k=25 (longer than any string: nothing matches).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void kernel(const int* offsets, const uint8_t* chars, const uint8_t* prefix,
                                  int8_t* match, long long n, long long k);

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
    const uint8_t pattern[3] = {'a', 'b', 'c'};
    std::vector<int> offsets(n + 1);
    std::vector<uint8_t> chars;
    offsets[0] = 0;
    for (long long i = 0; i < n; ++i) {
        const int len = (int)(next_u64() % 21u);
        const bool force = (next_u64() % 10u) < 3u;
        for (int j = 0; j < len; ++j) {
            const uint8_t c = (force && j < 3) ? pattern[j] : (uint8_t)('a' + next_u64() % 26u);
            chars.push_back(c);
        }
        offsets[i + 1] = (int)chars.size();
    }

    int* d_offsets = nullptr;
    uint8_t* d_chars = nullptr;
    uint8_t* d_prefix = nullptr;
    int8_t* d_match = nullptr;
    CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_chars, std::max<size_t>(chars.size(), 1)));
    CHECK(cudaMalloc(&d_prefix, 32));
    CHECK(cudaMalloc(&d_match, std::max<long long>(n, 1) * sizeof(int8_t)));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_chars, chars.data(), chars.size(), cudaMemcpyHostToDevice));

    bool ok = true;
    const long long ks[] = {3, 0, 25};
    for (long long k : ks) {
        std::vector<uint8_t> prefix(k);
        for (long long j = 0; j < k; ++j) prefix[j] = j < 3 ? pattern[j] : (uint8_t)'x';
        std::vector<int8_t> expected(n);
        for (long long i = 0; i < n; ++i) {
            bool m = (long long)(offsets[i + 1] - offsets[i]) >= k;
            for (long long j = 0; m && j < k; ++j) m = chars[offsets[i] + j] == prefix[j];
            expected[i] = m ? 1 : 0;
        }
        CHECK(cudaMemcpy(d_prefix, prefix.data(), k, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_match, 0x7f, std::max<long long>(n, 1) * sizeof(int8_t)));

        kernel<<<grid_for(n), THREADS>>>(d_offsets, d_chars, d_prefix, d_match, n, k);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        std::vector<int8_t> match(n);
        CHECK(cudaMemcpy(match.data(), d_match, n * sizeof(int8_t), cudaMemcpyDeviceToHost));
        const bool case_ok = std::equal(match.begin(), match.end(), expected.begin());
        long long hits = 0;
        for (long long i = 0; i < n; ++i) hits += expected[i];
        printf("  n=%lld k=%lld: %s (matches=%lld)\n", n, k, case_ok ? "ok" : "MISMATCH", hits);
        ok = ok && case_ok;

        if (case_ok && timed && k == 3) {
            cudaEvent_t start, stop;
            CHECK(cudaEventCreate(&start));
            CHECK(cudaEventCreate(&stop));
            const int reps = 20;
            float total = 0.0f;
            for (int r = 0; r < reps; ++r) {
                CHECK(cudaEventRecord(start));
                kernel<<<grid_for(n), THREADS>>>(d_offsets, d_chars, d_prefix, d_match, n, k);
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                float ms = 0.0f;
                CHECK(cudaEventElapsedTime(&ms, start, stop));
                total += ms;
            }
            printf("  n=%lld: %.1f us per launch\n", n, total / reps * 1e3f);
        }
    }

    CHECK(cudaFree(d_offsets));
    CHECK(cudaFree(d_chars));
    CHECK(cudaFree(d_prefix));
    CHECK(cudaFree(d_match));
    return ok;
}

int main() {
    const long long sizes[] = {0, 1, 1000, 1000003};
    bool ok = true;
    for (long long n : sizes) ok = run_case(n, n == 1000003) && ok;
    printf("%s starts_with\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
