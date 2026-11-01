// GPU test of the CUDA merge_sorted kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_merge_sorted test.cu && ./test_merge_sorted
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <utility>
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

static void launch(const int64_t* a, const int64_t* b, int64_t* out, int64_t m, int64_t p) {
  const int64_t total = m + p;
  if (total == 0) return;
  const int64_t grid = (total + merge_sorted::kTile - 1) / merge_sorted::kTile;
  kernel<<<static_cast<unsigned>(grid), merge_sorted::kBlock>>>(a, b, out, m, p);
  CUDA_CHECK(cudaGetLastError());
}

static std::vector<int64_t> sorted_values(std::mt19937_64& rng, int64_t size, int64_t spread) {
  std::uniform_int_distribution<int64_t> value(-spread, spread);
  std::vector<int64_t> v(static_cast<size_t>(size));
  for (auto& x : v) x = value(rng);
  std::sort(v.begin(), v.end());
  return v;
}

static bool check(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const int64_t m = static_cast<int64_t>(a.size());
  const int64_t p = static_cast<int64_t>(b.size());
  std::vector<int64_t> expected(static_cast<size_t>(m + p));
  std::merge(a.begin(), a.end(), b.begin(), b.end(), expected.begin());

  int64_t* d_a;
  int64_t* d_b;
  int64_t* d_out;
  CUDA_CHECK(cudaMalloc(&d_a, (m + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_b, (p + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_out, (m + p + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), m * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), p * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, (m + p + 1) * sizeof(int64_t)));
  launch(d_a, d_b, d_out, m, p);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int64_t> out(static_cast<size_t>(m + p));
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, (m + p) * sizeof(int64_t), cudaMemcpyDeviceToHost));
  const bool ok = out == expected;
  if (!ok) std::printf("FAIL m=%lld p=%lld\n", static_cast<long long>(m), static_cast<long long>(p));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));
  return ok;
}

int main() {
  std::mt19937_64 rng(9);
  bool ok = true;
  for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
    const std::pair<int64_t, int64_t> splits[] = {
        {n, 0}, {0, n}, {n / 2, n - n / 2}, {n / 3, n - n / 3}, {n, n}, {n, 3}};
    for (const auto& [m, p] : splits) {
      // Many ties, and a wide range with few ties.
      ok &= check(sorted_values(rng, m, 50), sorted_values(rng, p, 50));
      ok &= check(sorted_values(rng, m, 1LL << 62), sorted_values(rng, p, 1LL << 62));
    }
  }
  // Disjoint ranges: all of a before all of b, and the reverse.
  std::vector<int64_t> low(5000), high(2000);
  for (int i = 0; i < 5000; ++i) low[i] = i;
  for (int i = 0; i < 2000; ++i) high[i] = 5000 + i;
  ok &= check(low, high);
  ok &= check(high, low);
  // Interleaved, all equal, and extreme values.
  std::vector<int64_t> even(10000), odd(10000);
  for (int i = 0; i < 10000; ++i) {
    even[i] = 2 * i;
    odd[i] = 2 * i + 1;
  }
  ok &= check(even, odd);
  ok &= check(std::vector<int64_t>(3000, 7), std::vector<int64_t>(4000, 7));
  const int64_t lo = std::numeric_limits<int64_t>::min();
  const int64_t hi = std::numeric_limits<int64_t>::max();
  ok &= check({lo, lo, 0, hi, hi}, {lo, 0, hi});
  std::printf("merge_sorted cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
