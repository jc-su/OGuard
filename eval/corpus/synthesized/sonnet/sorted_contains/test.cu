// Test for sorted_contains against a CPU reference, n in {0,1,1000,1000003}.
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

static bool run_case(long long n, long long s, std::mt19937_64 &rng) {
  // Distinct sorted members drawn from a somewhat larger universe than
  // the keys' range, so both hits and misses occur.
  long long universe = std::max<long long>(2 * (s + n) + 10, 1000);
  std::uniform_int_distribution<long long> val_dist(0, universe);

  std::set<long long> member_set;
  while ((long long)member_set.size() < s) member_set.insert(val_dist(rng));
  std::vector<long long> h_members(member_set.begin(), member_set.end());

  std::vector<long long> h_keys(n > 0 ? n : 1);
  std::vector<signed char> h_ref(n > 0 ? n : 1);
  for (long long i = 0; i < n; ++i) {
    long long k = val_dist(rng);
    h_keys[i] = k;
    h_ref[i] = member_set.count(k) ? 1 : 0;
  }

  long long *d_keys = nullptr, *d_members = nullptr;
  signed char *d_match = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, (n > 0 ? n : 1) * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_members, (s > 0 ? s : 1) * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_match, (n > 0 ? n : 1)));
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), n * sizeof(long long),
                          cudaMemcpyHostToDevice));
  if (s > 0)
    CUDA_CHECK(cudaMemcpy(d_members, h_members.data(), s * sizeof(long long),
                          cudaMemcpyHostToDevice));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_keys, d_members, d_match, n, s);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<signed char> h_match(n > 0 ? n : 1);
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(h_match.data(), d_match, n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_members));
  CUDA_CHECK(cudaFree(d_match));

  bool ok = true;
  long long hits = 0;
  for (long long i = 0; i < n; ++i) {
    if (h_match[i] != h_ref[i]) { ok = false; break; }
    hits += h_ref[i];
  }
  std::printf("n=%lld s=%lld: hits=%lld -> %s\n", n, s, hits,
              ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937_64 rng(2468);
  bool all_ok = true;
  all_ok &= run_case(0, 0, rng);
  all_ok &= run_case(0, 500, rng);
  all_ok &= run_case(1, 0, rng);
  all_ok &= run_case(1, 1, rng);
  all_ok &= run_case(1000, 300, rng);
  all_ok &= run_case(1000003, 0, rng);
  all_ok &= run_case(1000003, 200000, rng);
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
