// Test for run_end_decode against a CPU reference, n in {0,1,1000,1000003}.
#include "kernel.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
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

// Builds a random run-end encoding of n rows: strictly increasing
// run_ends with run_ends.back() == n.
static void make_runs(long long n, std::mt19937 &rng,
                       std::vector<int> &run_ends,
                       std::vector<double> &run_values) {
  run_ends.clear();
  run_values.clear();
  if (n == 0) return;
  long long max_runs = std::min<long long>(n, 200);
  std::uniform_int_distribution<long long> count_dist(1, max_runs);
  long long r = count_dist(rng);
  std::set<long long> cuts;
  if (n > 1) {
    std::uniform_int_distribution<long long> cut_dist(1, n - 1);
    while ((long long)cuts.size() < r - 1 && (long long)cuts.size() < n - 1) {
      cuts.insert(cut_dist(rng));
    }
  }
  std::vector<long long> ends(cuts.begin(), cuts.end());
  ends.push_back(n);
  std::uniform_real_distribution<double> val_dist(-1e6, 1e6);
  for (long long e : ends) {
    run_ends.push_back((int)e);
    run_values.push_back(val_dist(rng));
  }
}

static bool run_case(long long n, std::mt19937 &rng) {
  std::vector<int> h_ends;
  std::vector<double> h_values;
  make_runs(n, rng, h_ends, h_values);
  long long r = (long long)h_ends.size();

  std::vector<double> h_ref(n > 0 ? n : 1);
  {
    long long prev = 0;
    for (long long j = 0; j < r; ++j) {
      for (long long i = prev; i < h_ends[j]; ++i) h_ref[i] = h_values[j];
      prev = h_ends[j];
    }
  }

  double *d_values = nullptr, *d_out = nullptr;
  int *d_ends = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, (r > 0 ? r : 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ends, (r > 0 ? r : 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, (n > 0 ? n : 1) * sizeof(double)));
  if (r > 0) {
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), r * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ends, h_ends.data(), r * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_values, d_ends, d_out, r, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<double> h_out(n > 0 ? n : 1);
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(double),
                          cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_ends));
  CUDA_CHECK(cudaFree(d_out));

  bool ok = true;
  for (long long i = 0; i < n; ++i) {
    if (h_out[i] != h_ref[i]) {
      ok = false;
      break;
    }
  }
  std::printf("n=%lld r=%lld: -> %s\n", n, r, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937 rng(777);
  long long sizes[] = {0, 1, 1000, 1000003};
  bool all_ok = true;
  for (long long n : sizes) all_ok &= run_case(n, rng);
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
