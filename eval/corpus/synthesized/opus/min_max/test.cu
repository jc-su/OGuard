// GPU test of the CUDA min_max kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_min_max test.cu && ./test_min_max
#include <algorithm>
#include <cmath>
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

static int resident_blocks() {
  int device = 0;
  int sms = 0;
  int per_sm = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, min_max::kBlock, 0));
  return sms * per_sm;
}

static void launch(const double* values, double* result, int64_t n) {
  if (n == 0) return;
  const int64_t grid =
      std::min<int64_t>((n + min_max::kBlock - 1) / min_max::kBlock, resident_blocks());
  kernel<<<static_cast<unsigned>(grid), min_max::kBlock>>>(values, result, n);
  CUDA_CHECK(cudaGetLastError());
}

static bool check(const std::vector<double>& values, const char* label) {
  const int64_t n = static_cast<int64_t>(values.size());
  const double inf = std::numeric_limits<double>::infinity();
  double expected[2] = {inf, -inf};
  for (double v : values) {
    expected[0] = std::min(expected[0], v);
    expected[1] = std::max(expected[1], v);
  }
  double* d_values;
  double* d_result;
  CUDA_CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_result, 2 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  const double init[2] = {inf, -inf};
  CUDA_CHECK(cudaMemcpy(d_result, init, sizeof(init), cudaMemcpyHostToDevice));
  launch(d_values, d_result, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  double result[2];
  CUDA_CHECK(cudaMemcpy(result, d_result, sizeof(result), cudaMemcpyDeviceToHost));
  const bool ok = result[0] == expected[0] && result[1] == expected[1];
  if (!ok) {
    std::printf("FAIL %s n=%lld result=[%g, %g] expected=[%g, %g]\n", label,
                static_cast<long long>(n), result[0], result[1], expected[0], expected[1]);
  }
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_result));
  return ok;
}

int main() {
  std::mt19937_64 rng(5);
  std::normal_distribution<double> normal;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  bool ok = true;
  for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
    std::vector<double> v(static_cast<size_t>(n));
    for (auto& x : v) x = normal(rng) * 1e3;
    ok &= check(v, "normal");
    for (auto& x : v) x = -unit(rng) - 1.0;
    ok &= check(v, "negative");
    for (auto& x : v) x = unit(rng) + 1.0;
    ok &= check(v, "positive");
    std::fill(v.begin(), v.end(), 3.25);
    ok &= check(v, "constant");
  }
  std::vector<double> v(1000003);
  for (auto& x : v) x = normal(rng);
  v[123457] = std::numeric_limits<double>::infinity();
  v[999999] = -std::numeric_limits<double>::infinity();
  ok &= check(v, "infinities");
  for (auto& x : v) x = normal(rng);
  v.front() = -1e300;
  v.back() = 1e300;
  ok &= check(v, "extremes at ends");
  ok &= check({-0.0, 0.0, -0.0}, "signed zeros");
  std::printf("min_max cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
