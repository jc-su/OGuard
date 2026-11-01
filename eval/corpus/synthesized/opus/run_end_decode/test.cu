// GPU test of the CUDA run_end_decode kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_run_end_decode test.cu && ./test_run_end_decode
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
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

static void launch(const double* run_values, const int32_t* run_ends, double* out, int64_t r,
                   int64_t n) {
  if (n == 0) return;
  const int64_t grid = (n + run_end_decode::kTile - 1) / run_end_decode::kTile;
  kernel<<<static_cast<unsigned>(grid), run_end_decode::kBlock>>>(run_values, run_ends, out, r, n);
  CUDA_CHECK(cudaGetLastError());
}

static std::vector<int32_t> random_run_ends(std::mt19937_64& rng, int64_t n, int64_t r) {
  // r - 1 distinct cut points in [1, n - 1], then n.
  std::vector<int32_t> ends;
  if (r - 1 > (n - 1) / 2) {
    std::vector<int32_t> all(static_cast<size_t>(n - 1));
    for (int64_t i = 0; i < n - 1; ++i) all[i] = static_cast<int32_t>(i + 1);
    std::shuffle(all.begin(), all.end(), rng);
    ends.assign(all.begin(), all.begin() + (r - 1));
  } else {
    std::set<int32_t> cuts;
    while (static_cast<int64_t>(cuts.size()) < r - 1) {
      cuts.insert(static_cast<int32_t>(1 + rng() % (n - 1)));
    }
    ends.assign(cuts.begin(), cuts.end());
  }
  std::sort(ends.begin(), ends.end());
  ends.push_back(static_cast<int32_t>(n));
  return ends;
}

static std::vector<int32_t> skewed_run_ends(std::mt19937_64& rng, int64_t n) {
  // One long run, runs of about a tile, and short runs elsewhere.
  std::vector<int32_t> ends;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  int64_t total = 0;
  while (total < n) {
    const double u = unit(rng);
    int64_t length = u < 0.001 ? n / 3 : u < 0.01 ? 2000 + static_cast<int64_t>(rng() % 100)
                                                  : 1 + static_cast<int64_t>(rng() % 4);
    length = std::max<int64_t>(1, std::min(length, n - total));
    total += length;
    ends.push_back(static_cast<int32_t>(total));
  }
  return ends;
}

static std::vector<int32_t> unit_runs(int64_t n) {
  std::vector<int32_t> ends(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) ends[i] = static_cast<int32_t>(i + 1);
  return ends;
}

static bool check(std::mt19937_64& rng, const std::vector<int32_t>& run_ends, int64_t n,
                  const char* label) {
  const int64_t r = static_cast<int64_t>(run_ends.size());
  std::normal_distribution<double> value;
  std::vector<double> run_values(static_cast<size_t>(r));
  for (auto& v : run_values) v = value(rng);
  std::vector<double> expected;
  expected.reserve(static_cast<size_t>(n));
  for (int64_t j = 0, row = 0; j < r; ++j) {
    for (; row < run_ends[j]; ++row) expected.push_back(run_values[j]);
  }

  double* d_values;
  int32_t* d_ends;
  double* d_out;
  CUDA_CHECK(cudaMalloc(&d_values, (r + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ends, (r + 1) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_values, run_values.data(), r * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ends, run_ends.data(), r * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(double)));
  launch(d_values, d_ends, d_out, r, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> out(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
  const bool ok = out == expected;
  if (!ok) std::printf("FAIL %s n=%lld r=%lld\n", label, static_cast<long long>(n),
                       static_cast<long long>(r));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_ends));
  CUDA_CHECK(cudaFree(d_out));
  return ok;
}

int main() {
  std::mt19937_64 rng(3);
  bool ok = check(rng, {}, 0, "empty");
  for (int64_t n : {1LL, 1000LL, 1000003LL}) {
    ok &= check(rng, {static_cast<int32_t>(n)}, n, "one run");
    ok &= check(rng, unit_runs(n), n, "unit runs");
    for (int64_t r : {std::max<int64_t>(1, n / 1000), std::max<int64_t>(1, n / 10),
                      std::max<int64_t>(1, n / 2)}) {
      ok &= check(rng, random_run_ends(rng, n, r), n, "random runs");
    }
    ok &= check(rng, skewed_run_ends(rng, n), n, "skewed runs");
  }
  for (int64_t n : {1791LL, 1792LL, 1793LL, 2047LL, 2048LL, 2049LL, 4096LL, 300000LL}) {
    ok &= check(rng, random_run_ends(rng, n, std::max<int64_t>(1, n / 3)), n, "tile edges");
    ok &= check(rng, unit_runs(n), n, "tile edges, unit runs");
  }
  std::printf("run_end_decode cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
