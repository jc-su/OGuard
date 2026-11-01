// Test for valid_rows: compares the GPU kernel against a CPU reference for
// n in {0, 1, 1000, 1000003}. Since the kernel may write the valid row ids
// in any order, both `rows` and the reference are sorted before comparing.
#include "kernel.cu"

#include <algorithm>
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
  long long nbytes = (n + 7) / 8;
  std::vector<unsigned char> h_valid(nbytes > 0 ? nbytes : 1, 0);
  std::uniform_int_distribution<int> bit_dist(0, 1);
  std::vector<int> h_ref;
  h_ref.reserve(n);
  for (long long i = 0; i < n; ++i) {
    int bit = bit_dist(rng);
    if (bit) {
      h_valid[i >> 3] |= (unsigned char)(1u << (i & 7));
      h_ref.push_back((int)i);
    }
  }

  unsigned char *d_valid = nullptr;
  int *d_rows = nullptr;
  int *d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_valid, (nbytes > 0 ? nbytes : 1)));
  CUDA_CHECK(cudaMalloc(&d_rows, (n > 0 ? n : 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_valid, h_valid.data(), (nbytes > 0 ? nbytes : 1),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_valid, d_rows, d_count, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  int h_count = 0;
  CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> h_rows(h_count > 0 ? h_count : 1);
  if (h_count > 0) {
    CUDA_CHECK(cudaMemcpy(h_rows.data(), d_rows, h_count * sizeof(int),
                          cudaMemcpyDeviceToHost));
  }
  h_rows.resize(h_count);

  CUDA_CHECK(cudaFree(d_valid));
  CUDA_CHECK(cudaFree(d_rows));
  CUDA_CHECK(cudaFree(d_count));

  std::sort(h_rows.begin(), h_rows.end());
  bool ok = (h_rows.size() == h_ref.size()) &&
            std::equal(h_rows.begin(), h_rows.end(), h_ref.begin());
  std::printf("n=%lld: count=%d expected=%zu -> %s\n", n, h_count,
              h_ref.size(), ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937 rng(12345);
  long long sizes[] = {0, 1, 1000, 1000003};
  bool all_ok = true;
  for (long long n : sizes) {
    all_ok &= run_case(n, rng);
  }
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
