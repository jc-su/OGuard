// Test for min_max against a CPU reference, n in {0,1,1000,1000003}.
#include "kernel.cu"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

static bool run_case(long long n, std::mt19937 &rng) {
  std::uniform_real_distribution<double> val_dist(-1e6, 1e6);
  std::vector<double> h_values(n > 0 ? n : 1);
  double ref_min = std::numeric_limits<double>::infinity();
  double ref_max = -std::numeric_limits<double>::infinity();
  for (long long i = 0; i < n; ++i) {
    double v = val_dist(rng);
    h_values[i] = v;
    if (v < ref_min) ref_min = v;
    if (v > ref_max) ref_max = v;
  }

  double *d_values = nullptr, *d_result = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, (n > 0 ? n : 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_result, 2 * sizeof(double)));
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));
  double h_init[2] = {std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()};
  CUDA_CHECK(cudaMemcpy(d_result, h_init, 2 * sizeof(double),
                        cudaMemcpyHostToDevice));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_values, d_result, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  double h_result[2];
  CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_result));

  bool ok = (h_result[0] == ref_min) && (h_result[1] == ref_max);
  std::printf("n=%lld: min=%.6g (ref %.6g) max=%.6g (ref %.6g) -> %s\n", n,
              h_result[0], ref_min, h_result[1], ref_max,
              ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937 rng(55);
  long long sizes[] = {0, 1, 1000, 1000003};
  bool all_ok = true;
  for (long long n : sizes) all_ok &= run_case(n, rng);
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
