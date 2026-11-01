// GPU test of the CUDA string_contains kernel against a CPU reference.
// Build and run: nvcc -O3 -arch=sm_86 -o test_string_contains test.cu && ./test_string_contains
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <string_view>
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

struct Column {
  std::vector<int32_t> offsets;
  std::string chars;
};

static void launch(const int32_t* offsets, const uint8_t* chars, const uint8_t* pattern,
                   int8_t* match, int64_t n, int64_t k) {
  if (n == 0) return;
  const int64_t grid = (n + string_contains::kBlock - 1) / string_contains::kBlock;
  kernel<<<static_cast<unsigned>(grid), string_contains::kBlock>>>(offsets, chars, pattern,
                                                                   match, n, k);
  CUDA_CHECK(cudaGetLastError());
}

static Column make_column(std::mt19937_64& rng, int64_t n, int max_short, double long_fraction,
                          int max_long) {
  std::uniform_int_distribution<int> short_len(0, max_short);
  std::uniform_int_distribution<int> long_len(129, max_long);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  Column c;
  c.offsets.push_back(0);
  for (int64_t i = 0; i < n; ++i) {
    const int len = unit(rng) < long_fraction ? long_len(rng) : short_len(rng);
    for (int j = 0; j < len; ++j) c.chars.push_back(static_cast<char>('a' + rng() % 4));
    c.offsets.push_back(static_cast<int32_t>(c.chars.size()));
  }
  return c;
}

// Checks the strings offsets[first:] of the column (first > 0 is a sliced column).
static bool check(const Column& c, const std::string& pattern, size_t first = 0) {
  const std::vector<int32_t> offsets(c.offsets.begin() + first, c.offsets.end());
  const int64_t n = static_cast<int64_t>(offsets.size()) - 1;
  const int64_t k = static_cast<int64_t>(pattern.size());
  const int64_t bytes = static_cast<int64_t>(c.chars.size());
  int32_t* d_offsets;
  uint8_t* d_chars;
  uint8_t* d_pattern;
  int8_t* d_match;
  CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_chars, bytes + 1));
  CUDA_CHECK(cudaMalloc(&d_pattern, k + 1));
  CUDA_CHECK(cudaMalloc(&d_match, n + 1));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_chars, c.chars.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pattern, pattern.data(), k, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_match, 0x7F, n + 1));
  launch(d_offsets, d_chars, d_pattern, d_match, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int8_t> match(static_cast<size_t>(n));
  CUDA_CHECK(cudaMemcpy(match.data(), d_match, n, cudaMemcpyDeviceToHost));
  bool ok = true;
  const std::string_view all(c.chars);
  for (int64_t i = 0; i < n && ok; ++i) {
    const std::string_view str = all.substr(offsets[i], offsets[i + 1] - offsets[i]);
    const int8_t expected = str.find(pattern) != std::string_view::npos ? 1 : 0;
    ok = match[i] == expected;
    if (!ok) {
      std::printf("FAIL n=%lld string %lld (length %zu) pattern '%s' match=%d\n",
                  static_cast<long long>(n), static_cast<long long>(i), str.size(),
                  pattern.c_str(), match[i]);
    }
  }
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_chars));
  CUDA_CHECK(cudaFree(d_pattern));
  CUDA_CHECK(cudaFree(d_match));
  return ok;
}

int main() {
  std::mt19937_64 rng(4);
  bool ok = true;
  const std::string patterns[] = {"", "a", "cab", "abcda", "dddddddd"};
  for (int64_t n : {0LL, 1LL, 1000LL, 1000003LL}) {
    // Short strings only, then a mix with long strings (the warp path).
    for (const auto& [max_short, long_fraction] : {std::pair{20, 0.0}, std::pair{300, 0.02}}) {
      const Column c = make_column(rng, n, max_short, long_fraction, 3000);
      for (const auto& pattern : patterns) ok &= check(c, pattern);
    }
  }
  // Long strings with the pattern only at the start, only at the end, or
  // longer than the short-string limit.
  std::string body;
  for (int i = 0; i < 700; ++i) body += "ab";
  Column c;
  c.offsets.push_back(0);
  for (const std::string& s : {"xyz" + body, body + "xyz", body + "xy", std::string("xyz"),
                               std::string("xy"), std::string(), body + body}) {
    c.chars += s;
    c.offsets.push_back(static_cast<int32_t>(c.chars.size()));
  }
  std::string long_pattern;
  for (int i = 0; i < 100; ++i) long_pattern += "ab";
  ok &= check(c, "xyz");
  ok &= check(c, "");
  ok &= check(c, long_pattern + "x");
  ok &= check(c, "b" + body.substr(0, 1398) + "x");
  // A sliced column: offsets do not start at 0.
  ok &= check(make_column(rng, 5000, 40, 0.05, 1000), "cab", 100);
  std::printf("string_contains cuda: %s\n", ok ? "passed" : "FAILED");
  return ok ? 0 : 1;
}
