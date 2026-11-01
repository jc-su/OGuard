// GPU test of the CUDA valid_rows kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_valid_rows test.cu && ./test_valid_rows
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernel.cu"

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err_));                              \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static void launch(const uint8_t* valid, int32_t* rows, int32_t* count, int64_t n) {
  if (n == 0) return;
  const int64_t grid = (n + valid_rows::kTile - 1) / valid_rows::kTile;
  kernel<<<static_cast<unsigned>(grid), valid_rows::kBlock>>>(valid, rows, count, n);
  CUDA_CHECK(cudaGetLastError());
}

static bool check(std::mt19937_64& rng, int64_t n, double density) {
  const int64_t nbytes = (n + 7) / 8;
  std::vector<uint8_t> valid(static_cast<size_t>(std::max<int64_t>(nbytes, 1)), 0);
  std::bernoulli_distribution bit(density);
  for (int64_t i = 0; i < nbytes * 8; ++i) {
    // Padding bits past row n are set so the kernel must ignore them.
    if (i >= n || bit(rng)) valid[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
  }
  std::vector<int32_t> expected;
  for (int64_t i = 0; i < n; ++i) {
    if ((valid[i >> 3] >> (i & 7)) & 1) expected.push_back(static_cast<int32_t>(i));
  }

  uint8_t* d_valid;
  int32_t* d_rows;
  int32_t* d_count;
  const size_t cap = static_cast<size_t>(std::max<int64_t>(n, 1));
  CUDA_CHECK(cudaMalloc(&d_valid, valid.size()));
  CUDA_CHECK(cudaMalloc(&d_rows, cap * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_valid, valid.data(), valid.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_rows, 0xFF, cap * sizeof(int32_t)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int32_t)));
  launch(d_valid, d_rows, d_count, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  int32_t count = 0;
  CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost));
  bool ok = count == static_cast<int32_t>(expected.size());
  if (ok) {
    std::vector<int32_t> rows(static_cast<size_t>(count));
    CUDA_CHECK(cudaMemcpy(rows.data(), d_rows, rows.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    std::sort(rows.begin(), rows.end());
    ok = rows == expected;
  }
  if (!ok) {
    std::printf("FAIL n=%lld density=%.2f count=%d expected=%zu\n",
                static_cast<long long>(n), density, count, expected.size());
  }
  CUDA_CHECK(cudaFree(d_valid));
  CUDA_CHECK(cudaFree(d_rows));
  CUDA_CHECK(cudaFree(d_count));
  return ok;
}

int main() {
  std::mt19937_64 rng(1);
  bool ok = true;
  for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
    for (double density : {0.0, 0.03, 0.5, 0.97, 1.0}) ok &= check(rng, n, density);
  }
  for (int64_t n : {7LL, 8LL, 9LL, 255LL, 256LL, 257LL, 2047LL, 2048LL, 2049LL, 65539LL}) {
    ok &= check(rng, n, 0.5);
  }
  std::printf("valid_rows cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
