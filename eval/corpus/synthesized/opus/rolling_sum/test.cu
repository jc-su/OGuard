// GPU test of the CUDA rolling_sum kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_rolling_sum test.cu && ./test_rolling_sum
//
// The CPU reference sums every window directly.
#include <cmath>
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

static void launch(const double* values, double* out, int64_t n, int64_t window) {
  if (window < 1 || window > rolling_sum::kMaxWindow) {
    std::fprintf(stderr, "window must be in [1, %d], got %lld\n", rolling_sum::kMaxWindow,
                 static_cast<long long>(window));
    std::exit(1);
  }
  if (n == 0) return;
  const int64_t grid = (n + rolling_sum::kTile - 1) / rolling_sum::kTile;
  kernel<<<static_cast<unsigned>(grid), rolling_sum::kBlock>>>(values, out, n, window);
  CUDA_CHECK(cudaGetLastError());
}

static bool check(const std::vector<double>& values, int64_t window, const char* label) {
  const int64_t n = static_cast<int64_t>(values.size());
  double* d_values;
  double* d_out;
  CUDA_CHECK(cudaMalloc(&d_values, (n + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(double)));
  launch(d_values, d_out, n, window);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> out(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int64_t i = 0; i < n && ok; ++i) {
    double expected = 0.0;
    for (int64_t j = i - window + 1 > 0 ? i - window + 1 : 0; j <= i; ++j) expected += values[j];
    ok = std::fabs(out[i] - expected) <= 1e-9 + 1e-9 * std::fabs(expected);
    if (!ok) {
      std::printf("FAIL %s n=%lld window=%lld i=%lld out=%.17g expected=%.17g\n", label,
                  static_cast<long long>(n), static_cast<long long>(window),
                  static_cast<long long>(i), out[i], expected);
    }
  }
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_out));
  return ok;
}

int main() {
  std::mt19937_64 rng(6);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  auto random_values = [&](int64_t n) {
    std::vector<double> v(static_cast<size_t>(n));
    for (auto& x : v) x = value(rng);
    return v;
  };
  bool ok = true;
  for (int64_t n : {0LL, 1LL, 1000LL}) {
    const auto v = random_values(n);
    for (int64_t w : {1, 2, 3, 7, 32, 255, 256, 257, 1000, 1023, 1024}) ok &= check(v, w, "random");
  }
  {
    const auto v = random_values(1000003);
    for (int64_t w : {1, 7, 256, 1024}) ok &= check(v, w, "random");
  }
  for (int64_t n : {1023LL, 1024LL, 1025LL, 2047LL, 2048LL, 2049LL, 5000LL}) {
    const auto v = random_values(n);
    for (int64_t w : {1, 5, 1000, 1024}) ok &= check(v, w, "tile edges");
  }
  // A huge value early on must not disturb the windows that exclude it:
  // a difference of prefix sums would lose the small values after it.
  auto v = random_values(100000);
  v[5] = 1e15;
  for (int64_t w : {1, 16, 1024}) ok &= check(v, w, "spike");
  std::printf("rolling_sum cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
