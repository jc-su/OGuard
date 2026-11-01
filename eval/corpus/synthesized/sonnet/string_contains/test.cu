// Test for string_contains against a CPU reference, n in {0,1,1000,1000003}.
#include "kernel.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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

static bool cpu_contains(const unsigned char *s, int len,
                          const unsigned char *pat, int k) {
  if (k == 0) return true;
  if (k > len) return false;
  for (int s0 = 0; s0 <= len - k; ++s0) {
    bool ok = true;
    for (int j = 0; j < k; ++j) {
      if (s[s0 + j] != pat[j]) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

static bool run_case(long long n, const std::string &pattern,
                      std::mt19937 &rng) {
  const char alphabet[] = "ab c"; // small alphabet, biased to hit "ab"
  std::uniform_int_distribution<int> len_dist(0, 20);
  std::uniform_int_distribution<int> ch_dist(0, 3);

  std::vector<int> h_offsets(n + 1);
  std::vector<unsigned char> h_chars;
  h_offsets[0] = 0;
  for (long long i = 0; i < n; ++i) {
    int len = len_dist(rng);
    for (int c = 0; c < len; ++c) h_chars.push_back((unsigned char)alphabet[ch_dist(rng)]);
    h_offsets[i + 1] = h_offsets[i] + len;
  }
  long long total_chars = h_offsets[n];

  int k = (int)pattern.size();
  std::vector<unsigned char> h_pattern(pattern.begin(), pattern.end());

  std::vector<signed char> h_ref(n > 0 ? n : 1);
  for (long long i = 0; i < n; ++i) {
    int len = h_offsets[i + 1] - h_offsets[i];
    h_ref[i] = cpu_contains(h_chars.data() + h_offsets[i], len,
                             h_pattern.data(), k) ? 1 : 0;
  }

  int *d_offsets = nullptr;
  unsigned char *d_chars = nullptr, *d_pattern = nullptr;
  signed char *d_match = nullptr;
  CUDA_CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_chars, (total_chars > 0 ? total_chars : 1)));
  CUDA_CHECK(cudaMalloc(&d_pattern, (k > 0 ? k : 1)));
  CUDA_CHECK(cudaMalloc(&d_match, (n > 0 ? n : 1)));
  CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (total_chars > 0)
    CUDA_CHECK(cudaMemcpy(d_chars, h_chars.data(), total_chars,
                          cudaMemcpyHostToDevice));
  if (k > 0)
    CUDA_CHECK(cudaMemcpy(d_pattern, h_pattern.data(), k,
                          cudaMemcpyHostToDevice));

  if (n > 0) {
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    size_t shared_bytes = (size_t)k;
    kernel<<<blocks, threads, shared_bytes>>>(d_offsets, d_chars, d_pattern,
                                               d_match, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<signed char> h_match(n > 0 ? n : 1);
  if (n > 0)
    CUDA_CHECK(cudaMemcpy(h_match.data(), d_match, n, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_chars));
  CUDA_CHECK(cudaFree(d_pattern));
  CUDA_CHECK(cudaFree(d_match));

  bool ok = true;
  long long hits = 0;
  for (long long i = 0; i < n; ++i) {
    if (h_match[i] != h_ref[i]) { ok = false; break; }
    hits += h_ref[i];
  }
  std::printf("n=%lld pattern=\"%s\": hits=%lld -> %s\n", n, pattern.c_str(),
              hits, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  std::mt19937 rng(99);
  long long sizes[] = {0, 1, 1000, 1000003};
  bool all_ok = true;
  for (long long n : sizes) all_ok &= run_case(n, "ab", rng);
  all_ok &= run_case(1000, "", rng); // empty pattern matches everything
  std::printf(all_ok ? "ALL PASS\n" : "SOME FAILED\n");
  return all_ok ? 0 : 1;
}
