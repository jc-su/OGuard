// Test for radix_scatter against a CPU reference, n in {0,1,1000,1000003}.
// Checks that each byte bucket's slice of `out` is exactly the set of row
// ids whose byte equals that bucket (order within a bucket is unspecified).
#include "kernel.cu"

#include <algorithm>
#include <cstdint>
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

static bool run_case(long long n, int shift, std::mt19937 &rng) {
  std::uniform_int_distribution<uint32_t> key_dist(
      0, std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> h_keys(n > 0 ? n : 1);
  std::vector<std::vector<int>> buckets(256);
  for (long long i = 0; i < n; ++i) {
    uint32_t k = key_dist(rng);
    h_keys[i] = k;
    unsigned byte = (k >> shift) & 255u;
    buckets[byte].push_back((int)i);
  }

  std::vector<int> h_cursors(256), h_bucket_start(256);
  int running = 0;
  for (int b = 0; b < 256; ++b) {
    h_cursors[b] = running;
    h_bucket_start[b] = running;
    running += (int)buckets[b].size();
  }

  uint32_t *d_keys = nullptr;
  int *d_cursors = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, (n > 0 ? n : 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_cursors, 256 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, (n > 0 ? n : 1) * sizeof(int)));
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cursors, h_cursors.data(), 256 * sizeof(int),
                        cudaMemcpyHostToDevice));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_keys, d_cursors, d_out, n, shift);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<int> h_out(n > 0 ? n : 1);
  std::vector<int> h_final_cursors(256);
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(int),
                          cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_final_cursors.data(), d_cursors, 256 * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_cursors));
  CUDA_CHECK(cudaFree(d_out));

  bool ok = true;
  for (int b = 0; b < 256 && ok; ++b) {
    int expect_end = h_bucket_start[b] + (int)buckets[b].size();
    if (h_final_cursors[b] != expect_end) { ok = false; break; }
    std::vector<int> got(h_out.begin() + h_bucket_start[b],
                          h_out.begin() + expect_end);
    std::vector<int> expect = buckets[b];
    std::sort(got.begin(), got.end());
    std::sort(expect.begin(), expect.end());
    if (got != expect) { ok = false; break; }
  }
  std::printf("n=%lld shift=%d: -> %s\n", n, shift, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937 rng(31337);
  bool all_ok = true;
  all_ok &= run_case(0, 0, rng);
  all_ok &= run_case(1, 8, rng);
  all_ok &= run_case(1000, 0, rng);
  all_ok &= run_case(1000, 16, rng);
  all_ok &= run_case(1000003, 0, rng);
  all_ok &= run_case(1000003, 24, rng);
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
