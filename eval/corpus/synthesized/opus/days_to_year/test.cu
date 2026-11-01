// GPU test of the CUDA days_to_year kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_days_to_year test.cu && ./test_days_to_year
//
// The CPU reference does not share the kernel's algorithm: it counts the days
// from 1970-01-01 to January 1 of a year with the leap-year rule and searches
// the year whose January 1 is the last one on or before the date.
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

constexpr int32_t kFirstDay = -719162;  // 0001-01-01
constexpr int32_t kLastDay = 2932896;   // 9999-12-31

static int64_t leap_years_before(int64_t y) {  // leap years in [1, y), y >= 1
  const int64_t x = y - 1;
  return x / 4 - x / 100 + x / 400;
}

static int64_t jan1(int64_t y) {  // days from 1970-01-01 to January 1 of year y
  return 365 * (y - 1970) + leap_years_before(y) - leap_years_before(1970);
}

static int32_t reference_year(int32_t day) {
  int64_t y = 1970 + static_cast<int64_t>(day / 365.2425);
  while (jan1(y) > day) --y;
  while (jan1(y + 1) <= day) ++y;
  return static_cast<int32_t>(y);
}

static void launch(const int32_t* days, int32_t* year, int64_t n) {
  if (n == 0) return;
  const int64_t grid = (n + days_to_year::kBlock - 1) / days_to_year::kBlock;
  kernel<<<static_cast<unsigned>(grid), days_to_year::kBlock>>>(days, year, n);
  CUDA_CHECK(cudaGetLastError());
}

static bool check(const std::vector<int32_t>& days) {
  const int64_t n = static_cast<int64_t>(days.size());
  int32_t* d_days;
  int32_t* d_year;
  CUDA_CHECK(cudaMalloc(&d_days, (n + 1) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_year, (n + 1) * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_days, days.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_year, 0xFF, (n + 1) * sizeof(int32_t)));
  launch(d_days, d_year, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int32_t> year(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(year.data(), d_year, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int64_t i = 0; i < n; ++i) {
    if (year[i] != reference_year(days[i])) {
      std::printf("FAIL n=%lld day=%d year=%d expected=%d\n", static_cast<long long>(n),
                  days[i], year[i], reference_year(days[i]));
      ok = false;
      break;
    }
  }
  CUDA_CHECK(cudaFree(d_days));
  CUDA_CHECK(cudaFree(d_year));
  return ok;
}

int main() {
  bool ok = reference_year(kFirstDay) == 1 && reference_year(kLastDay) == 9999 &&
            reference_year(0) == 1970 && reference_year(-1) == 1969 &&
            reference_year(11016) == 2000 && reference_year(10956) == 1999;
  if (!ok) std::printf("FAIL reference self-check\n");
  std::mt19937_64 rng(8);
  std::uniform_int_distribution<int32_t> day(kFirstDay, kLastDay);
  for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
    std::vector<int32_t> days(static_cast<size_t>(n));
    for (auto& d : days) d = day(rng);
    ok &= check(days);
  }
  // Every date from 0001-01-01 to 9999-12-31.
  std::vector<int32_t> all;
  for (int32_t d = kFirstDay; d <= kLastDay; ++d) all.push_back(d);
  ok &= check(all);
  std::printf("days_to_year cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
