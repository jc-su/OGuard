// Standalone test for q6_sum. Allocates inputs, launches the kernel,
// and checks the result against a CPU reference, for
// n in {0, 1, 1000, 1000003}.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const int *shipdate, const double *discount,
                                   const double *quantity, const double *price,
                                   double *result, int date_lo, int date_hi,
                                   double disc_lo, double disc_hi,
                                   double qty_max, long long n);

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
const int DATE_LO = 9000, DATE_HI = 9100;
const double DISC_LO = 0.02, DISC_HI = 0.09;
const double QTY_MAX = 24.0;

static bool run_case(long long n, std::mt19937_64 &rng) {
    long long m = n > 0 ? n : 1;
    std::vector<int> shipdate(m);
    std::vector<double> discount(m), quantity(m), price(m);

    std::uniform_int_distribution<int> date_dist(8900, 9199);
    std::uniform_real_distribution<double> disc_dist(0.0, 0.12);
    std::uniform_real_distribution<double> qty_dist(0.0, 50.0);
    std::uniform_real_distribution<double> price_dist(1.0, 1000.0);
    for (long long i = 0; i < n; ++i) {
        shipdate[i] = date_dist(rng);
        discount[i] = disc_dist(rng);
        quantity[i] = qty_dist(rng);
        price[i] = price_dist(rng);
    }

    double ref = 0.0;
    for (long long i = 0; i < n; ++i) {
        if (shipdate[i] >= DATE_LO && shipdate[i] < DATE_HI &&
            discount[i] >= DISC_LO && discount[i] <= DISC_HI &&
            quantity[i] < QTY_MAX) {
            ref += price[i] * discount[i];
        }
    }

    int *d_shipdate;
    double *d_discount, *d_quantity, *d_price, *d_result;
    CUDA_CHECK(cudaMalloc(&d_shipdate, sizeof(int) * m));
    CUDA_CHECK(cudaMalloc(&d_discount, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_quantity, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_price, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_shipdate, shipdate.data(), sizeof(int) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_discount, discount.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quantity, quantity.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_price, price.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(double)));

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_shipdate, d_discount, d_quantity, d_price,
                                   d_result, DATE_LO, DATE_HI, DISC_LO,
                                   DISC_HI, QTY_MAX, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double got = 0.0;
    CUDA_CHECK(cudaMemcpy(&got, d_result, sizeof(double),
                          cudaMemcpyDeviceToHost));

    double tol = 1e-6 * std::max(1.0, std::fabs(ref));
    bool ok = std::fabs(got - ref) <= tol;

    printf("n=%lld: %s (got=%.6f, ref=%.6f)\n", n, ok ? "PASS" : "FAIL", got,
           ref);

    cudaFree(d_shipdate);
    cudaFree(d_discount);
    cudaFree(d_quantity);
    cudaFree(d_price);
    cudaFree(d_result);
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
