// Test for dictionary_decode against a CPU reference, n in {0,1,1000,1000003}.
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

static bool run_case(long long n, std::mt19937 &rng) {
  long long d = 37;
  std::vector<double> h_dict(d);
  std::uniform_real_distribution<double> val_dist(-1e6, 1e6);
  for (auto &v : h_dict) v = val_dist(rng);

  std::uniform_int_distribution<int> code_dist(0, (int)d - 1);
  std::vector<int> h_codes(n > 0 ? n : 1);
  std::vector<double> h_ref(n > 0 ? n : 1);
  for (long long i = 0; i < n; ++i) {
    int c = code_dist(rng);
    h_codes[i] = c;
    h_ref[i] = h_dict[c];
  }

  int *d_codes = nullptr;
  double *d_dict = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_codes, (n > 0 ? n : 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dict, d * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out, (n > 0 ? n : 1) * sizeof(double)));
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(d_codes, h_codes.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dict, h_dict.data(), d * sizeof(double),
                        cudaMemcpyHostToDevice));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_codes, d_dict, d_out, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<double> h_out(n > 0 ? n : 1);
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(double),
                          cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_codes));
  CUDA_CHECK(cudaFree(d_dict));
  CUDA_CHECK(cudaFree(d_out));

  bool ok = true;
  for (long long i = 0; i < n; ++i) {
    if (h_out[i] != h_ref[i]) {
      ok = false;
      break;
    }
  }
  std::printf("n=%lld: -> %s\n", n, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937 rng(4242);
  long long sizes[] = {0, 1, 1000, 1000003};
  bool all_ok = true;
  for (long long n : sizes) all_ok &= run_case(n, rng);
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
