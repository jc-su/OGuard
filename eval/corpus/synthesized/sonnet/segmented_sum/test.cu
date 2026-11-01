// Standalone test for segmented_sum. Allocates inputs, launches the
// kernel, and checks the result against a CPU reference, for
// n (total values, split across g segments of varied, including
// zero, length) in {0, 1, 1000, 1000003}.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const int *offsets, const double *values,
                                   double *sums, long long g);

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

const int BLOCK = 256;
const int WARP_SIZE = 32;

static std::vector<int> make_offsets(long long n, long long g,
                                     std::mt19937_64 &rng) {
    std::vector<long long> sizes(g, 0);
    if (n > 0) {
        // Random composition of n into g nonnegative parts: g - 1
        // cut points drawn with replacement from [0, n] and sorted.
        std::vector<long long> cuts(g - 1);
        std::uniform_int_distribution<long long> dist(0, n);
        for (auto &c : cuts) c = dist(rng);
        std::sort(cuts.begin(), cuts.end());
        long long prev = 0;
        for (long long j = 0; j < g - 1; ++j) {
            sizes[j] = cuts[j] - prev;
            prev = cuts[j];
        }
        sizes[g - 1] = n - prev;
    }
    std::vector<int> offsets(g + 1, 0);
    for (long long j = 0; j < g; ++j) offsets[j + 1] = offsets[j] + (int)sizes[j];
    return offsets;
}

static bool run_case(long long n, std::mt19937_64 &rng) {
    long long g = n > 0 ? std::max(1LL, std::min(n, 5000LL)) : 5;
    std::vector<int> offsets = make_offsets(n, g, rng);

    std::vector<double> values(n > 0 ? n : 1);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (long long i = 0; i < n; ++i) values[i] = dist(rng);

    std::vector<double> ref(g, 0.0);
    for (long long j = 0; j < g; ++j)
        for (int i = offsets[j]; i < offsets[j + 1]; ++i) ref[j] += values[i];

    int *d_offsets;
    double *d_values, *d_sums;
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * (g + 1)));
    CUDA_CHECK(cudaMalloc(&d_values, sizeof(double) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_sums, sizeof(double) * g));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * (g + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));

    long long threads_needed = g * WARP_SIZE;
    long long blocks = (threads_needed + BLOCK - 1) / BLOCK;
    kernel<<<blocks, BLOCK>>>(d_offsets, d_values, d_sums, g);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> got(g);
    CUDA_CHECK(cudaMemcpy(got.data(), d_sums, sizeof(double) * g,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (long long j = 0; j < g; ++j) {
        double tol = 1e-8 * std::max(1.0, std::fabs(ref[j]));
        if (std::fabs(got[j] - ref[j]) > tol) ok = false;
    }

    printf("n=%lld: %s (g=%lld)\n", n, ok ? "PASS" : "FAIL", g);

    cudaFree(d_offsets);
    cudaFree(d_values);
    cudaFree(d_sums);
    return ok;
}

int main() {
    std::mt19937_64 rng(0);
    bool all_ok = true;
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        all_ok &= run_case(n, rng);
    }
    return all_ok ? 0 : 1;
}
