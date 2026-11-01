// Test for days_to_year against an independent CPU reference, n in
// {0,1,1000,1000003}. The reference uses Hinnant's `days_from_civil`
// (the inverse direction of the kernel's `civil_from_days`) plus a
// binary search over the year, which is a different code path from the
// kernel's O(1) formula and so gives a genuine cross-check.
#include "kernel.cu"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err_ = (expr);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                   cudaGetErrorString(err_), __FILE__, __LINE__);            \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

static long long days_from_civil(long long y, int m, int d) {
  y -= (m <= 2) ? 1 : 0;
  long long era = (y >= 0 ? y : y - 399) / 400;
  long long yoe = y - era * 400;                                      // [0,399]
  long long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;      // [0,365]
  long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;               // [0,146096]
  return era * 146097 + doe - 719468;
}

static int reference_year(long long days) {
  long long lo = 1, hi = 9999;
  while (lo < hi) {
    long long mid = lo + (hi - lo + 1) / 2;
    if (days_from_civil(mid, 1, 1) <= days) lo = mid; else hi = mid - 1;
  }
  return (int)lo;
}

static bool run_case(long long n, long long min_day, long long max_day,
                      std::mt19937_64 &rng) {
  std::uniform_int_distribution<long long> day_dist(min_day, max_day);
  std::vector<int> h_days(n > 0 ? n : 1);
  std::vector<int> h_ref(n > 0 ? n : 1);
  for (long long i = 0; i < n; ++i) {
    long long d = day_dist(rng);
    h_days[i] = (int)d;
    h_ref[i] = reference_year(d);
  }

  int *d_days = nullptr, *d_year = nullptr;
  CUDA_CHECK(cudaMalloc(&d_days, (n > 0 ? n : 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_year, (n > 0 ? n : 1) * sizeof(int)));
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(d_days, h_days.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_days, d_year, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<int> h_year(n > 0 ? n : 1);
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(h_year.data(), d_year, n * sizeof(int),
                          cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_days));
  CUDA_CHECK(cudaFree(d_year));

  bool ok = true;
  for (long long i = 0; i < n; ++i) {
    if (h_year[i] != h_ref[i]) { ok = false; break; }
  }
  std::printf("n=%lld: -> %s\n", n, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  long long min_day = days_from_civil(1, 1, 1);
  long long max_day = days_from_civil(9999, 12, 31);
  std::printf("valid day range: [%lld, %lld]\n", min_day, max_day);

  std::mt19937_64 rng(908070);
  long long sizes[] = {0, 1, 1000, 1000003};
  bool all_ok = true;
  for (long long n : sizes) all_ok &= run_case(n, min_day, max_day, rng);

  // A few known fixed points.
  {
    int known_days[] = {0, -1, 1, (int)min_day, (int)max_day};
    int expect[] = {1970, 1969, 1970, 1, 9999};
    int *d_days = nullptr, *d_year = nullptr;
    CUDA_CHECK(cudaMalloc(&d_days, sizeof(known_days)));
    CUDA_CHECK(cudaMalloc(&d_year, sizeof(known_days)));
    CUDA_CHECK(cudaMemcpy(d_days, known_days, sizeof(known_days),
                          cudaMemcpyHostToDevice));
    kernel<<<1, 256>>>(d_days, d_year, 5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int got[5];
    CUDA_CHECK(cudaMemcpy(got, d_year, sizeof(got), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_days));
    CUDA_CHECK(cudaFree(d_year));
    bool ok = true;
    for (int i = 0; i < 5; ++i) if (got[i] != expect[i]) ok = false;
    std::printf("fixed points: -> %s\n", ok ? "PASS" : "FAIL");
    all_ok &= ok;
  }

  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
