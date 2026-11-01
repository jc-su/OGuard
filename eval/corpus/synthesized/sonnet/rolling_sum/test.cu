// Test for rolling_sum against a CPU reference, n in {0,1,1000,1000003},
// with a couple of different window sizes (including window > n).
#include "kernel.cu"

#include <cmath>
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

static bool run_case(long long n, long long window, std::mt19937 &rng) {
  std::uniform_real_distribution<double> val_dist(-100.0, 100.0);
  std::vector<double> h_values(n > 0 ? n : 1);
  for (long long i = 0; i < n; ++i) h_values[i] = val_dist(rng);

  std::vector<double> h_ref(n > 0 ? n : 1);
  // Prefix-sum based reference for speed on large n.
  std::vector<double> prefix(n + 1, 0.0);
  for (long long i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + h_values[i];
  for (long long i = 0; i < n; ++i) {
    long long lo = std::max<long long>(0, i - window + 1);
    h_ref[i] = prefix[i + 1] - prefix[lo];
  }

  double *d_values = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, (n > 0 ? n : 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out, (n > 0 ? n : 1) * sizeof(double)));
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    long long tile_size = threads + window - 1;
    size_t shared_bytes = (size_t)tile_size * sizeof(double);
    kernel<<<blocks, threads, shared_bytes>>>(d_values, d_out, n, window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<double> h_out(n > 0 ? n : 1);
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(double),
                          cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_out));

  bool ok = true;
  for (long long i = 0; i < n; ++i) {
    double diff = std::fabs(h_out[i] - h_ref[i]);
    double tol = 1e-6 * std::max(1.0, std::fabs(h_ref[i]));
    if (diff > tol) { ok = false; break; }
  }
  std::printf("n=%lld window=%lld: -> %s\n", n, window, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937 rng(2024);
  bool all_ok = true;
  all_ok &= run_case(0, 5, rng);
  all_ok &= run_case(1, 1, rng);
  all_ok &= run_case(1, 1024, rng);   // window larger than n
  all_ok &= run_case(1000, 7, rng);
  all_ok &= run_case(1000, 1024, rng); // window larger than n
  all_ok &= run_case(1000003, 1, rng);
  all_ok &= run_case(1000003, 300, rng);
  all_ok &= run_case(1000003, 1024, rng);
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
