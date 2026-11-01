#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernel.cu"

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                             \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

static int failures = 0;
static const int32_t END_FILL = -7;
static const int64_t VALUE_FILL = 123456789;

// kind: 0 small alphabet, 1 distinct, 2 all equal, 3 long runs, 4 alternating
// extremes, 5 random.
static void run_case(std::mt19937_64& rng, int64_t n, int kind) {
    std::vector<int64_t> values(n);
    int64_t run_left = 0, current = 0;
    for (int64_t i = 0; i < n; i++) {
        switch (kind) {
            case 0: values[i] = (int64_t)(rng() % 3); break;
            case 1: values[i] = i; break;
            case 2: values[i] = INT64_MIN; break;
            case 3:
                if (run_left == 0) { run_left = 1 + (int64_t)(rng() % 5000); current = (int64_t)rng(); }
                values[i] = current;
                run_left--;
                break;
            case 4: values[i] = (i % 2 == 0) ? INT64_MAX : INT64_MIN; break;
            default: values[i] = (int64_t)rng(); break;
        }
    }
    std::vector<int32_t> want_ends(n, END_FILL), got_ends(n, END_FILL);
    std::vector<int64_t> want_vals(n, VALUE_FILL), got_vals(n, VALUE_FILL);
    int32_t want_count = 0, got_count = 0;
    for (int64_t i = 0; i < n; i++) {
        if (i == 0 || values[i] != values[i - 1]) {
            want_vals[want_count] = values[i];
            if (want_count > 0) want_ends[want_count - 1] = (int32_t)i;
            want_count++;
        }
    }
    if (n > 0) want_ends[want_count - 1] = (int32_t)n;
    int64_t* d_values; int32_t *d_ends, *d_count; int64_t* d_vals;
    CHECK(cudaMalloc(&d_values, (n > 0 ? n : 1) * 8));
    CHECK(cudaMalloc(&d_ends, (n > 0 ? n : 1) * 4));
    CHECK(cudaMalloc(&d_vals, (n > 0 ? n : 1) * 8));
    CHECK(cudaMalloc(&d_count, 4));
    if (n > 0) {
        CHECK(cudaMemcpy(d_values, values.data(), n * 8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_ends, got_ends.data(), n * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_vals, got_vals.data(), n * 8, cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemset(d_count, 0, 4));
    kernel<<<1, 256>>>(d_values, d_ends, d_vals, d_count, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    if (n > 0) {
        CHECK(cudaMemcpy(got_ends.data(), d_ends, n * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(got_vals.data(), d_vals, n * 8, cudaMemcpyDeviceToHost));
    }
    CHECK(cudaMemcpy(&got_count, d_count, 4, cudaMemcpyDeviceToHost));
    bool ok = got_ends == want_ends && got_vals == want_vals && got_count == want_count;
    if (!ok) failures++;
    printf("%s run_end_encode cuda n=%lld kind=%d\n", ok ? "PASS" : "FAIL", (long long)n, kind);
    cudaFree(d_values); cudaFree(d_ends); cudaFree(d_vals); cudaFree(d_count);
}

int main() {
    std::mt19937_64 rng(11);
    for (int64_t n : {0, 1, 2, 255, 2048, 2049, 1000, 1000003})
        for (int kind = 0; kind < 6; kind++) run_case(rng, n, kind);
    return failures ? 1 : 0;
}
