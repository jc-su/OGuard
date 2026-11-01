// Standalone test for gather. Allocates inputs, launches the kernel,
// and checks the result against a CPU reference, for
// n (and m = n) in {0, 1, 1000, 1000003}.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const double *col, const int *sel,
                                   double *out, long long m, long long n);

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
    long long m = n;

    std::vector<double> col(n > 0 ? n : 1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (long long i = 0; i < n; ++i) col[i] = dist(rng);

    std::vector<int> sel(m > 0 ? m : 1, 0);
    if (n > 0) {
        std::uniform_int_distribution<long long> idx_dist(0, n - 1);
        for (long long i = 0; i < m; ++i) sel[i] = (int)idx_dist(rng);
    }

    std::vector<double> ref(m > 0 ? m : 1);
    for (long long i = 0; i < m; ++i) ref[i] = col[sel[i]];

    double *d_col, *d_out;
    int *d_sel;
    CUDA_CHECK(cudaMalloc(&d_col, sizeof(double) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_sel, sizeof(int) * (m > 0 ? m : 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * (m > 0 ? m : 1)));
    CUDA_CHECK(cudaMemcpy(d_col, col.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sel, sel.data(), sizeof(int) * m,
                          cudaMemcpyHostToDevice));

    long long blocks = (m + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_col, d_sel, d_out, m, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> got(m > 0 ? m : 1);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, sizeof(double) * m,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (long long i = 0; i < m; ++i) {
        if (got[i] != ref[i]) {
            ok = false;
            break;
        }
    }

    printf("n=%lld: %s\n", n, ok ? "PASS" : "FAIL");

    cudaFree(d_col);
    cudaFree(d_sel);
    cudaFree(d_out);
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
