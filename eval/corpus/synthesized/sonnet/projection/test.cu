// Standalone test for projection. Allocates inputs, launches the
// kernel, and checks the result against a CPU reference, for
// n in {0, 1, 1000, 1000003}.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const double *price, const double *discount,
                                   double *out, long long n);

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
    std::vector<double> price(n > 0 ? n : 1), discount(n > 0 ? n : 1);
    std::uniform_real_distribution<double> price_dist(0.0, 100.0);
    std::uniform_real_distribution<double> disc_dist(0.0, 1.0);
    for (long long i = 0; i < n; ++i) {
        price[i] = price_dist(rng);
        discount[i] = disc_dist(rng);
    }

    std::vector<double> ref(n > 0 ? n : 1);
    for (long long i = 0; i < n; ++i) ref[i] = price[i] * (1.0 - discount[i]);

    double *d_price, *d_discount, *d_out;
    CUDA_CHECK(cudaMalloc(&d_price, sizeof(double) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_discount, sizeof(double) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMemcpy(d_price, price.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_discount, discount.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_price, d_discount, d_out, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> got(n > 0 ? n : 1);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, sizeof(double) * n,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (long long i = 0; i < n; ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-9) {
            ok = false;
            break;
        }
    }

    printf("n=%lld: %s\n", n, ok ? "PASS" : "FAIL");

    cudaFree(d_price);
    cudaFree(d_discount);
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
