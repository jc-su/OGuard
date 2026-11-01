#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include "kernel.cu"

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
            exit(2);                                                       \
        }                                                                  \
    } while (0)

static bool check(int64_t n, const std::string& mode, std::mt19937_64& rng)
{
    std::vector<int64_t> values(n);
    std::vector<char> valid(n);
    for (int64_t i = 0; i < n; i++) {
        values[i] = (int64_t)rng();
        if (mode == "half") valid[i] = (rng() & 1);
        else if (mode == "mostly_null") valid[i] = (rng() % 1000 == 0);
        else if (mode == "all_null") valid[i] = 0;
        else if (mode == "all_valid") valid[i] = 1;
        else if (mode == "first_only") valid[i] = (i == 0);
        else if (mode == "last_only") valid[i] = (i == n - 1);
        else valid[i] = (i == n / 2);
    }
    int64_t nb = (n + 7) / 8;
    std::vector<uint8_t> bitmap(nb, 0xFF);  // padding bits of the last byte are 1
    for (int64_t i = 0; i < n; i++)
        if (!valid[i]) bitmap[i / 8] &= (uint8_t)~(1u << (i % 8));
    std::vector<int64_t> expected(n);
    int64_t state = 0;
    for (int64_t i = 0; i < n; i++) {
        if (valid[i]) state = values[i];
        expected[i] = state;
    }
    int64_t *d_values, *d_out;
    uint8_t* d_valid;
    CK(cudaMalloc(&d_values, (n > 0 ? n : 1) * 8));
    CK(cudaMalloc(&d_out, (n > 0 ? n : 1) * 8));
    CK(cudaMalloc(&d_valid, nb > 0 ? nb : 1));
    CK(cudaMemcpy(d_values, values.data(), n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_valid, bitmap.data(), nb, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0x4D, (n > 0 ? n : 1) * 8));
    kernel<<<1, 256>>>(d_values, d_valid, d_out, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<int64_t> got(n);
    CK(cudaMemcpy(got.data(), d_out, n * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_values);
    cudaFree(d_out);
    cudaFree(d_valid);
    bool ok = true;
    for (int64_t i = 0; i < n; i++)
        if (got[i] != expected[i]) {
            printf("  mismatch row %lld: got %lld expected %lld\n", (long long)i, (long long)got[i],
                   (long long)expected[i]);
            ok = false;
            break;
        }
    printf("n=%lld mode=%s: %s\n", (long long)n, mode.c_str(), ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    std::mt19937_64 rng(18);
    bool ok = true;
    for (int64_t n : {0, 1, 1000, 2048, 2049, 1000003})
        for (const char* mode : {"half", "mostly_null", "all_null", "all_valid", "first_only", "last_only", "middle_only"})
            ok &= check(n, mode, rng);
    return ok ? 0 : 1;
}
