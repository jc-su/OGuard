// GPU test of the CUDA dictionary_decode kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_dictionary_decode test.cu && ./test_dictionary_decode
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

static void launch(const int32_t* codes, const double* dictionary, double* out, int64_t n,
                   int64_t d) {
  if (n == 0) return;
  const int64_t grid = (n + dictionary_decode::kBlock - 1) / dictionary_decode::kBlock;
  kernel<<<static_cast<unsigned>(grid), dictionary_decode::kBlock>>>(codes, dictionary, out,
                                                                     n, d);
  CUDA_CHECK(cudaGetLastError());
}

static bool check(std::mt19937_64& rng, int64_t n, int64_t d) {
  std::uniform_int_distribution<int32_t> code(0, static_cast<int32_t>(d - 1));
  std::normal_distribution<double> value;
  std::vector<int32_t> codes(static_cast<size_t>(n));
  std::vector<double> dictionary(static_cast<size_t>(d));
  for (auto& c : codes) c = code(rng);
  for (auto& v : dictionary) v = value(rng);

  int32_t* d_codes;
  double* d_dictionary;
  double* d_out;
  CUDA_CHECK(cudaMalloc(&d_codes, (n + 1) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_dictionary, d * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out, (n + 1) * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_codes, codes.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dictionary, dictionary.data(), d * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0xFF, (n + 1) * sizeof(double)));
  launch(d_codes, d_dictionary, d_out, n, d);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> out(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int64_t i = 0; i < n && ok; ++i) ok = out[i] == dictionary[codes[i]];
  if (!ok) std::printf("FAIL n=%lld d=%lld\n", static_cast<long long>(n), static_cast<long long>(d));
  CUDA_CHECK(cudaFree(d_codes));
  CUDA_CHECK(cudaFree(d_dictionary));
  CUDA_CHECK(cudaFree(d_out));
  return ok;
}

int main() {
  std::mt19937_64 rng(2);
  bool ok = true;
  for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
    for (int64_t d : {1LL, 7LL, 1000LL, 100000LL}) ok &= check(rng, n, d);
  }
  for (int64_t n : {255LL, 256LL, 257LL, 65537LL}) ok &= check(rng, n, 3);
  std::printf("dictionary_decode cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
