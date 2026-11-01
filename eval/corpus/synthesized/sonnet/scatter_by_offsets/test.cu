// Standalone test for scatter_by_offsets. Allocates inputs, launches
// the kernel, and checks the result against a CPU reference, for
// n in {0, 1, 1000, 1000003}.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const signed char *flags, const int *offsets,
                                   int *out, long long n, long long total);

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
    std::vector<signed char> flags(n > 0 ? n : 1, 0);
    std::bernoulli_distribution flag_dist(0.3);
    for (long long i = 0; i < n; ++i) flags[i] = flag_dist(rng) ? 1 : 0;

    std::vector<int> offsets(n > 0 ? n : 1, 0);
    int running = 0;
    for (long long i = 0; i < n; ++i) {
        offsets[i] = running;
        if (flags[i]) running++;
    }
    long long total = running;

    std::vector<int> ref(total > 0 ? total : 1, -1);
    for (long long i = 0; i < n; ++i)
        if (flags[i]) ref[offsets[i]] = (int)i;

    signed char *d_flags;
    int *d_offsets, *d_out;
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(signed char) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * (total > 0 ? total : 1)));
    CUDA_CHECK(cudaMemcpy(d_flags, flags.data(), sizeof(signed char) * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * n,
                          cudaMemcpyHostToDevice));

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_flags, d_offsets, d_out, n, total);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> got(total > 0 ? total : 1);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, sizeof(int) * total,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (long long i = 0; i < total; ++i) {
        if (got[i] != ref[i]) {
            ok = false;
            break;
        }
    }

    printf("n=%lld: %s (total=%lld)\n", n, ok ? "PASS" : "FAIL", total);

    cudaFree(d_flags);
    cudaFree(d_offsets);
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
