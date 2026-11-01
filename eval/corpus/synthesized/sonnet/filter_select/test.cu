// Standalone test for filter_select. Allocates inputs, launches the
// kernel, and checks the result against a CPU reference, for
// n in {0, 1, 1000, 1000003}.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const double *quantity, int *sel,
                                   int *count, double threshold,
                                   long long n);

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

static bool run_case(long long n, std::mt19937_64 &rng) {
    const double threshold = 0.0;

    std::vector<double> quantity(n > 0 ? n : 1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (long long i = 0; i < n; ++i) quantity[i] = dist(rng);

    std::vector<int> ref_ids;
    for (long long i = 0; i < n; ++i)
        if (quantity[i] < threshold) ref_ids.push_back((int)i);

    double *d_quantity;
    int *d_sel, *d_count;
    CUDA_CHECK(cudaMalloc(&d_quantity, sizeof(double) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_sel, sizeof(int) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_quantity, quantity.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_quantity, d_sel, d_count, threshold, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int got_count = 0;
    CUDA_CHECK(cudaMemcpy(&got_count, d_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool ok = (got_count == (int)ref_ids.size());
    if (ok && got_count > 0) {
        std::vector<int> got_ids(got_count);
        CUDA_CHECK(cudaMemcpy(got_ids.data(), d_sel, sizeof(int) * got_count,
                              cudaMemcpyDeviceToHost));
        std::sort(got_ids.begin(), got_ids.end());
        std::vector<int> ref_sorted = ref_ids;
        std::sort(ref_sorted.begin(), ref_sorted.end());
        ok = (got_ids == ref_sorted);
    }

    printf("n=%lld: %s (count=%d, ref=%zu)\n", n, ok ? "PASS" : "FAIL",
           got_count, ref_ids.size());

    cudaFree(d_quantity);
    cudaFree(d_sel);
    cudaFree(d_count);
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
