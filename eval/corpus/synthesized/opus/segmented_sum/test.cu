// GPU test of kernel.cu (segmented_sum) against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_segmented_sum
//
// The values are multiples of 1/8 with magnitude at most 2**17, so every
// partial sum is exact in double and the check can require equality,
// independent of the order of the additions.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
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

// The grid for the mapping that the kernel selects when offsets[0] == 0. If
// offsets[0] > 0, the kernel can select a mapping for shorter segments; the
// grid-stride loops keep that correct.
static unsigned blocks_for(int64_t n, int64_t g) {
  using namespace segmented_sum;
  constexpr int64_t kWarpsPerBlock = kBlockSize / kWarpSize;
  if (n >= kBlockModeMinAverage * g) return static_cast<unsigned>(g);
  if (n >= kWarpModeMinAverage * g) {
    return static_cast<unsigned>((g + kWarpsPerBlock - 1) / kWarpsPerBlock);
  }
  return static_cast<unsigned>((g + kBlockSize - 1) / kBlockSize);
}

// Nondecreasing offsets with offsets[g] == n. offsets[0] can be above 0, and segments can be empty.
static std::vector<int32_t> random_offsets(int64_t n, int64_t g, std::mt19937_64& rng) {
  std::vector<int32_t> offsets(g + 1);
  std::uniform_int_distribution<int64_t> dist(0, n);
  for (int32_t& o : offsets) o = static_cast<int32_t>(dist(rng));
  std::sort(offsets.begin(), offsets.end());
  offsets[g] = static_cast<int32_t>(n);
  return offsets;
}

// n / 4 short segments (0 to 3 values) and three long segments, one of them at
// a lane 31 position, so that the kernel uses one thread per segment and the
// warp sums the long ones. Requires n >= 1000.
static std::vector<int32_t> skewed_offsets(int64_t n, std::mt19937_64& rng) {
  const int64_t g = n / 4;
  std::vector<int64_t> lengths(g);
  std::uniform_int_distribution<int64_t> dist(0, 3);
  for (int64_t& length : lengths) length = dist(rng);
  const int64_t long_rows[] = {31, g / 2, g - 1};
  for (const int64_t row : long_rows) lengths[row] = 0;
  int64_t remainder = n;
  for (const int64_t length : lengths) remainder -= length;
  lengths[long_rows[0]] = remainder / 3;
  lengths[long_rows[1]] = remainder / 3;
  lengths[long_rows[2]] = remainder - 2 * (remainder / 3);
  std::vector<int32_t> offsets(g + 1);
  offsets[0] = 0;
  for (int64_t j = 0; j < g; ++j) offsets[j + 1] = static_cast<int32_t>(offsets[j] + lengths[j]);
  return offsets;
}

static bool check(const std::string& name, const std::vector<int32_t>& offsets,
                  const std::vector<double>& values) {
  const int64_t g = static_cast<int64_t>(offsets.size()) - 1;
  const int64_t n = static_cast<int64_t>(values.size());
  std::vector<double> expected(g);
  for (int64_t j = 0; j < g; ++j) {
    double acc = 0.0;
    for (int64_t i = offsets[j]; i < offsets[j + 1]; ++i) acc += values[i];
    expected[j] = acc;
  }

  int32_t* d_offsets = nullptr;
  double* d_values = nullptr;
  double* d_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_values, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sums, g * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  // All bytes 0xFF is a NaN, so an unwritten segment fails the check.
  CUDA_CHECK(cudaMemset(d_sums, 0xFF, g * sizeof(double)));

  // A launch needs at least one block, so g == 0 launches nothing.
  if (g > 0) {
    kernel<<<blocks_for(n, g), kBlockSize>>>(d_offsets, d_values, d_sums, g);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> got(g);
  CUDA_CHECK(cudaMemcpy(got.data(), d_sums, g * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_sums));

  for (int64_t j = 0; j < g; ++j) {
    if (!(got[j] == expected[j])) {
      std::printf("segmented_sum cuda n=%lld case=%s g=%lld: FAIL at %lld: got %.17g, expected %.17g\n",
                  static_cast<long long>(n), name.c_str(), static_cast<long long>(g),
                  static_cast<long long>(j), got[j], expected[j]);
      return false;
    }
  }
  return true;
}

static bool run_case(int64_t n, std::mt19937_64& rng) {
  std::vector<double> values(n);
  std::uniform_int_distribution<int64_t> value_dist(-(int64_t{1} << 20), (int64_t{1} << 20) - 1);
  for (double& v : values) v = static_cast<double>(value_dist(rng)) / 8.0;

  std::vector<std::pair<std::string, std::vector<int32_t>>> cases;
  cases.emplace_back("one_segment", std::vector<int32_t>{0, static_cast<int32_t>(n)});
  cases.emplace_back("late_start", std::vector<int32_t>{static_cast<int32_t>(std::max<int64_t>(0, n - 40)),
                                                        static_cast<int32_t>(n)});
  cases.emplace_back("g=n/5000", random_offsets(n, std::max<int64_t>(1, n / 5000), rng));
  cases.emplace_back("g=n/100", random_offsets(n, std::max<int64_t>(1, n / 100), rng));
  cases.emplace_back("g=n/10", random_offsets(n, std::max<int64_t>(1, n / 10), rng));
  cases.emplace_back("g=n+5", random_offsets(n, n + 5, rng));
  if (n >= 1000) cases.emplace_back("skewed", skewed_offsets(n, rng));
  if (n == 0) cases.emplace_back("g=0", std::vector<int32_t>{0});

  bool ok = true;
  std::string names;
  for (const auto& [name, offsets] : cases) {
    ok = check(name, offsets, values) && ok;
    names += " " + name;
  }
  if (ok) {
    std::printf("segmented_sum cuda n=%lld cases=[%s ]: pass\n", static_cast<long long>(n),
                names.c_str());
  }
  return ok;
}

int main() {
  const int64_t sizes[] = {0, 1, 1000, 1000003};
  std::mt19937_64 rng(12);
  bool ok = true;
  for (const int64_t n : sizes) ok = run_case(n, rng) && ok;
  return ok ? 0 : 1;
}
