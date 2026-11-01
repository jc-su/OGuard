// GPU test of the CUDA radix_scatter kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_radix_scatter test.cu && ./test_radix_scatter
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

static void launch(const uint32_t* keys, int32_t* cursors, int32_t* out, int64_t n,
                   int64_t shift) {
  if (n == 0) return;
  const int64_t grid = (n + radix_scatter::kTile - 1) / radix_scatter::kTile;
  kernel<<<static_cast<unsigned>(grid), radix_scatter::kBlock>>>(keys, cursors, out, n, shift);
  CUDA_CHECK(cudaGetLastError());
}

static bool check(const std::vector<uint32_t>& keys, int shift, const char* label) {
  const int64_t n = static_cast<int64_t>(keys.size());
  auto digit = [&](int64_t i) { return static_cast<int>((keys[i] >> shift) & 255u); };
  std::vector<int32_t> counts(256, 0);
  for (int64_t i = 0; i < n; ++i) ++counts[digit(i)];
  std::vector<int32_t> starts(256, 0);
  for (int d = 1; d < 256; ++d) starts[d] = starts[d - 1] + counts[d - 1];

  uint32_t* d_keys;
  int32_t* d_cursors;
  int32_t* d_out;
  CUDA_CHECK(cudaMalloc(&d_keys, (n + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cursors, 256 * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cursors, starts.data(), 256 * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(int32_t)));
  launch(d_keys, d_cursors, d_out, n, shift);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> out(static_cast<size_t>(n));
  std::vector<int32_t> cursors(256);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cursors.data(), d_cursors, 256 * sizeof(int32_t), cudaMemcpyDeviceToHost));
  bool ok = true;
  // Every slot of digit d's range holds a row with digit d, and every row appears once.
  std::vector<char> seen(static_cast<size_t>(n), 0);
  for (int d = 0; d < 256 && ok; ++d) {
    ok = cursors[d] == starts[d] + counts[d];
    for (int32_t j = starts[d]; j < starts[d] + counts[d] && ok; ++j) {
      const int32_t row = out[j];
      ok = row >= 0 && row < n && !seen[row] && digit(row) == d;
      if (ok) seen[row] = 1;
    }
  }
  if (!ok) std::printf("FAIL %s n=%lld shift=%d\n", label, static_cast<long long>(n), shift);
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_cursors));
  CUDA_CHECK(cudaFree(d_out));
  return ok;
}

int main() {
  std::mt19937_64 rng(7);
  bool ok = true;
  for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
    std::vector<uint32_t> keys(static_cast<size_t>(n));
    for (auto& k : keys) k = static_cast<uint32_t>(rng());
    for (int shift : {0, 8, 16, 24}) ok &= check(keys, shift, "random");
    std::fill(keys.begin(), keys.end(), 0xDEADBEEFu);
    ok &= check(keys, 8, "constant");
    for (int64_t i = 0; i < n; ++i) keys[i] = static_cast<uint32_t>(i);
    ok &= check(keys, 0, "sequential");
    for (auto& k : keys) k = static_cast<uint32_t>(rng() % 3) << 16;
    ok &= check(keys, 16, "three digits");
  }
  for (int64_t n : {255LL, 256LL, 257LL, 2047LL, 2048LL, 2049LL}) {
    std::vector<uint32_t> keys(static_cast<size_t>(n));
    for (auto& k : keys) k = static_cast<uint32_t>(rng());
    ok &= check(keys, 24, "tile edges");
  }
  std::printf("radix_scatter cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
