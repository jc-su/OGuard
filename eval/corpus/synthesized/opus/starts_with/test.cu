// GPU test of kernel.cu (starts_with) against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_starts_with
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernel.cu"

#define CUDA_CHECK(call)                                                               \
  do {                                                                                 \
    const cudaError_t err_ = (call);                                                   \
    if (err_ != cudaSuccess) {                                                         \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,               \
                   cudaGetErrorString(err_));                                          \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

constexpr int kBlockSize = 256;
constexpr int kMaxSymbols = 8;
// A small alphabet with 1, 2, 3, and 4 byte UTF-8 characters, so that prefixes match often.
const char* const kAlphabet[] = {"a", "b", "\xC3\xA9", "\xE2\x82\xAC", "\xF0\x9F\x98\x80"};
constexpr int kAlphabetSize = 5;

struct Column {
  std::vector<int32_t> offsets;
  std::vector<uint8_t> chars;
};

static Column make_column(int64_t n, std::mt19937_64& rng) {
  Column column;
  column.offsets.resize(n + 1);
  column.offsets[0] = 0;
  std::uniform_int_distribution<int> length_dist(0, kMaxSymbols);
  std::uniform_int_distribution<int> symbol_dist(0, kAlphabetSize - 1);
  for (int64_t i = 0; i < n; ++i) {
    const int symbols = length_dist(rng);
    for (int s = 0; s < symbols; ++s) {
      const char* symbol = kAlphabet[symbol_dist(rng)];
      column.chars.insert(column.chars.end(), symbol, symbol + std::strlen(symbol));
    }
    column.offsets[i + 1] = static_cast<int32_t>(column.chars.size());
  }
  return column;
}

// Returns the number of matches, or -1 on a mismatch.
static int64_t run_pattern(const Column& column, int32_t* d_offsets, uint8_t* d_chars,
                           const std::string& pattern) {
  const int64_t n = static_cast<int64_t>(column.offsets.size()) - 1;
  const int64_t k = static_cast<int64_t>(pattern.size());

  std::vector<int8_t> expected(n);
  int64_t matches = 0;
  for (int64_t i = 0; i < n; ++i) {
    const int32_t begin = column.offsets[i];
    const int32_t length = column.offsets[i + 1] - begin;
    expected[i] = length >= k && std::equal(pattern.begin(), pattern.end(),
                                            column.chars.begin() + begin,
                                            [](char a, uint8_t b) {
                                              return static_cast<uint8_t>(a) == b;
                                            });
    matches += expected[i];
  }

  uint8_t* d_prefix = nullptr;
  int8_t* d_match = nullptr;
  CUDA_CHECK(cudaMalloc(&d_prefix, k));
  CUDA_CHECK(cudaMalloc(&d_match, n));
  CUDA_CHECK(cudaMemcpy(d_prefix, pattern.data(), k, cudaMemcpyHostToDevice));
  // Fill the output with -1 so that an unwritten row fails the check.
  CUDA_CHECK(cudaMemset(d_match, 0xFF, n));

  // A launch needs at least one block, so n == 0 launches nothing.
  if (n > 0) {
    const int64_t blocks = (n + kBlockSize - 1) / kBlockSize;
    kernel<<<static_cast<unsigned>(blocks), kBlockSize>>>(d_offsets, d_chars, d_prefix, d_match,
                                                          n, k);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int8_t> got(n);
  CUDA_CHECK(cudaMemcpy(got.data(), d_match, n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_prefix));
  CUDA_CHECK(cudaFree(d_match));

  for (int64_t i = 0; i < n; ++i) {
    if (got[i] != expected[i]) {
      std::printf("starts_with cuda n=%lld k=%lld: FAIL at %lld: got %d, expected %d\n",
                  static_cast<long long>(n), static_cast<long long>(k),
                  static_cast<long long>(i), got[i], expected[i]);
      return -1;
    }
  }
  return matches;
}

static bool run_case(int64_t n, std::mt19937_64& rng) {
  const Column column = make_column(n, rng);

  std::vector<std::string> patterns = {
      "",
      "a",
      "\xC3\xA9",                          // é
      "a\xC3\xA9",                         // aé
      "\xE2\x82\xAC\xF0\x9F\x98\x80",      // €😀
      "ab\xE2\x82\xAC\xC3\xA9",            // ab€é
      "\xE2\x82",                          // a partial UTF-8 sequence: the test is bytewise
      std::string(40, 'a'),                // longer than every string
  };
  if (n > 0) {
    // A pattern equal to a whole string, so that a string of exactly k bytes matches.
    int64_t longest = 0;
    for (int64_t i = 1; i < n; ++i) {
      if (column.offsets[i + 1] - column.offsets[i] >
          column.offsets[longest + 1] - column.offsets[longest]) {
        longest = i;
      }
    }
    patterns.emplace_back(column.chars.begin() + column.offsets[longest],
                          column.chars.begin() + column.offsets[longest + 1]);
  }

  int32_t* d_offsets = nullptr;
  uint8_t* d_chars = nullptr;
  CUDA_CHECK(cudaMalloc(&d_offsets, column.offsets.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_chars, column.chars.size()));
  CUDA_CHECK(cudaMemcpy(d_offsets, column.offsets.data(),
                        column.offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_chars, column.chars.data(), column.chars.size(),
                        cudaMemcpyHostToDevice));

  bool ok = true;
  std::string summary;
  for (const std::string& pattern : patterns) {
    const int64_t matches = run_pattern(column, d_offsets, d_chars, pattern);
    if (matches < 0) {
      ok = false;
    } else {
      summary += " " + std::to_string(matches);
    }
  }
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_chars));
  if (ok) {
    std::printf("starts_with cuda n=%lld patterns=%zu matches=[%s ]: pass\n",
                static_cast<long long>(n), patterns.size(), summary.c_str());
  }
  return ok;
}

int main() {
  const int64_t sizes[] = {0, 1, 1000, 1000003};
  std::mt19937_64 rng(10);
  bool ok = true;
  for (const int64_t n : sizes) ok = run_case(n, rng) && ok;
  return ok ? 0 : 1;
}
