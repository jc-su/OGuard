// Test for merge_sorted against a CPU reference (a stable merge with
// a-before-equal-b), for total sizes m+p in {0,1,1000,1000003}, split a
// few different ways, including empty a or b and a low-cardinality case
// that stresses the tie-break rule.
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

static bool run_case(long long m, long long p, int value_range,
                      std::mt19937_64 &rng) {
  std::uniform_int_distribution<long long> val_dist(-value_range, value_range);
  std::vector<long long> h_a(m > 0 ? m : 1), h_b(p > 0 ? p : 1);
  for (auto &v : h_a) v = val_dist(rng);
  for (auto &v : h_b) v = val_dist(rng);
  std::sort(h_a.begin(), h_a.begin() + m);
  std::sort(h_b.begin(), h_b.begin() + p);

  // Stable merge with a-before-equal-b as the CPU reference.
  std::vector<long long> h_ref(m + p);
  {
    long long i = 0, j = 0, k = 0;
    while (i < m && j < p) {
      if (h_a[i] <= h_b[j]) h_ref[k++] = h_a[i++];
      else h_ref[k++] = h_b[j++];
    }
    while (i < m) h_ref[k++] = h_a[i++];
    while (j < p) h_ref[k++] = h_b[j++];
  }

  long long *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  long long total = m + p;
  CUDA_CHECK(cudaMalloc(&d_a, (m > 0 ? m : 1) * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_b, (p > 0 ? p : 1) * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_out, (total > 0 ? total : 1) * sizeof(long long)));
  if (m > 0)
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), m * sizeof(long long),
                          cudaMemcpyHostToDevice));
  if (p > 0)
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), p * sizeof(long long),
                          cudaMemcpyHostToDevice));

  if (total > 0) {
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_a, d_b, d_out, m, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<long long> h_out(total > 0 ? total : 1);
  if (total > 0)
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(long long),
                          cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_out));

  bool ok = true;
  for (long long k = 0; k < total; ++k) {
    if (h_out[k] != h_ref[k]) { ok = false; break; }
  }
  std::printf("m=%lld p=%lld range=%d: -> %s\n", m, p, value_range,
              ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937_64 rng(13);
  bool all_ok = true;
  all_ok &= run_case(0, 0, 1000000, rng);
  all_ok &= run_case(1, 0, 1000000, rng);
  all_ok &= run_case(0, 1, 1000000, rng);
  all_ok &= run_case(400, 600, 1000000, rng);
  all_ok &= run_case(500000, 500003, 1000000, rng);
  // Low-cardinality: exercises the a-before-equal-b tie-break heavily.
  all_ok &= run_case(500000, 500003, 5, rng);
  all_ok &= run_case(1000003, 0, 1000000, rng);
  all_ok &= run_case(0, 1000003, 1000000, rng);
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
