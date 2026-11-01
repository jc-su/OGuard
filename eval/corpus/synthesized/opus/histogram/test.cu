// GPU test of kernel.cu (histogram) against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_histogram
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "kernel.cu"

#define CUDA_CHECK(call)                                                               \
  do {                                                                                 \
    const cudaError_t err_ = (call);                                                   \
    if (err_ != cudaSuccess) {                                                         \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,               \
                   cudaGetErrorString(err_));                                          \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

constexpr int kBlockSize = 256;
constexpr int kBlocksPerSm = 4;
constexpr int kBins = 256;
constexpr int64_t kShifts[] = {0, 3, 8, 13, 16, 24};

static bool check(const std::string& name, const std::vector<uint32_t>& keys, int64_t shift,
                  int sm_count) {
  const int64_t n = static_cast<int64_t>(keys.size());
  std::vector<int32_t> expected(kBins, 0);
  for (const uint32_t key : keys) ++expected[(key >> shift) & 0xFF];

  uint32_t* d_keys = nullptr;
  int32_t* d_counts = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_counts, kBins * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_counts, 0, kBins * sizeof(int32_t)));

  // A launch needs at least one block, so n == 0 launches nothing.
  if (n > 0) {
    const int64_t blocks = std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                             static_cast<int64_t>(sm_count) * kBlocksPerSm);
    kernel<<<static_cast<unsigned>(blocks), kBlockSize>>>(d_keys, d_counts, shift, n);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> got(kBins);
  CUDA_CHECK(cudaMemcpy(got.data(), d_counts, kBins * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_counts));

  for (int b = 0; b < kBins; ++b) {
    if (got[b] != expected[b]) {
      std::printf("histogram cuda n=%lld keys=%s shift=%lld: FAIL at bin %d: got %d, expected %d\n",
                  static_cast<long long>(n), name.c_str(), static_cast<long long>(shift), b,
                  got[b], expected[b]);
      return false;
    }
  }
  return true;
}

static bool run_case(int64_t n, std::mt19937_64& rng, int sm_count) {
  // Uniform keys, and two skewed key sets that put many keys in one bin.
  std::vector<uint32_t> uniform(n);
  std::vector<uint32_t> constant(n, 0xA5C3E17Fu);
  std::vector<uint32_t> four_values(n);
  const uint32_t values[] = {0u, 0xFFFFFFFFu, 0x12345678u, 0x80000000u};
  std::uniform_int_distribution<uint32_t> key_dist;
  std::uniform_int_distribution<int> pick(0, 3);
  for (int64_t i = 0; i < n; ++i) {
    uniform[i] = key_dist(rng);
    four_values[i] = values[pick(rng)];
  }

  bool ok = true;
  for (const int64_t shift : kShifts) {
    ok = check("uniform", uniform, shift, sm_count) && ok;
    ok = check("constant", constant, shift, sm_count) && ok;
    ok = check("four_values", four_values, shift, sm_count) && ok;
  }
  if (ok) {
    std::printf("histogram cuda n=%lld key_sets=3 shifts=6: pass\n", static_cast<long long>(n));
  }
  return ok;
}

int main() {
  int device = 0;
  int sm_count = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  const int64_t sizes[] = {0, 1, 1000, 1000003};
  std::mt19937_64 rng(11);
  bool ok = true;
  for (const int64_t n : sizes) ok = run_case(n, rng, sm_count) && ok;
  return ok ? 0 : 1;
}
