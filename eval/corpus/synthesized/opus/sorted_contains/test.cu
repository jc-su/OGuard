// GPU test of the CUDA sorted_contains kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_sorted_contains test.cu && ./test_sorted_contains
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

static void launch(const int64_t* keys, const int64_t* members, int8_t* match, int64_t n,
                   int64_t s) {
  if (n == 0) return;
  const int64_t grid = (n + sorted_contains::kBlock - 1) / sorted_contains::kBlock;
  kernel<<<static_cast<unsigned>(grid), sorted_contains::kBlock>>>(keys, members, match, n, s);
  CUDA_CHECK(cudaGetLastError());
}

static bool check(const std::vector<int64_t>& keys, const std::vector<int64_t>& members) {
  const int64_t n = static_cast<int64_t>(keys.size());
  const int64_t s = static_cast<int64_t>(members.size());
  int64_t* d_keys;
  int64_t* d_members;
  int8_t* d_match;
  CUDA_CHECK(cudaMalloc(&d_keys, (n + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_members, (s + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_match, n + 1));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_members, members.data(), s * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_match, 0x7F, n + 1));
  launch(d_keys, d_members, d_match, n, s);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int8_t> match(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(match.data(), d_match, n, cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int64_t i = 0; i < n && ok; ++i) {
    const int8_t expected = std::binary_search(members.begin(), members.end(), keys[i]) ? 1 : 0;
    ok = match[i] == expected;
  }
  if (!ok) std::printf("FAIL n=%lld s=%lld\n", static_cast<long long>(n), static_cast<long long>(s));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_members));
  CUDA_CHECK(cudaFree(d_match));
  return ok;
}

int main() {
  std::mt19937_64 rng(10);
  std::uniform_int_distribution<int64_t> value(-(1LL << 40), 1LL << 40);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  bool ok = true;
  for (int64_t s : {0LL, 1LL, 2LL, 3LL, 1000LL, 1000003LL}) {
    std::vector<int64_t> members;
    while (static_cast<int64_t>(members.size()) < s) {
      while (static_cast<int64_t>(members.size()) < s) members.push_back(value(rng));
      std::sort(members.begin(), members.end());
      members.erase(std::unique(members.begin(), members.end()), members.end());
    }
    for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
      std::vector<int64_t> keys(static_cast<size_t>(n));
      for (auto& k : keys) {
        const double u = unit(rng);
        if (s > 0 && u < 0.5) {
          k = members[rng() % s];
        } else if (s > 0 && u < 0.7) {
          k = members[rng() % s] + 1;
        } else {
          k = value(rng);
        }
      }
      if (n >= 2) {
        keys[0] = kMin;
        keys[1] = kMax;
      }
      ok &= check(keys, members);
    }
  }
  // Members at the ends of the int64 range.
  ok &= check({kMin, kMin + 1, -5, -4, 0, 6, 7, 8, kMax - 1, kMax}, {kMin, -5, 0, 7, kMax});
  std::printf("sorted_contains cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
