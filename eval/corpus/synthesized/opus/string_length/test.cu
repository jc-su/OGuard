// GPU test of kernel.cu (string_length) against a CPU reference.
// Build: nvcc -arch=sm_86 -O3 test.cu -o test_string_length
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
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
constexpr int kMaxCodePoints = 16;
constexpr int kLongStringCodePoints = 5000;
constexpr int kLongStrings = 10;

// A valid Unicode scalar value, with an equal share of 1, 2, 3, and 4 byte encodings.
static uint32_t random_code_point(std::mt19937_64& rng) {
  switch (std::uniform_int_distribution<int>(1, 4)(rng)) {
    case 1:
      return std::uniform_int_distribution<uint32_t>(0x0, 0x7F)(rng);
    case 2:
      return std::uniform_int_distribution<uint32_t>(0x80, 0x7FF)(rng);
    case 3: {
      // Skip the surrogate range U+D800 to U+DFFF.
      const uint32_t v = std::uniform_int_distribution<uint32_t>(0x800, 0xFFFF - 0x800)(rng);
      return v >= 0xD800 ? v + 0x800 : v;
    }
    default:
      return std::uniform_int_distribution<uint32_t>(0x10000, 0x10FFFF)(rng);
  }
}

static void append_utf8(std::vector<uint8_t>& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<uint8_t>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<uint8_t>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<uint8_t>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<uint8_t>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<uint8_t>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<uint8_t>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<uint8_t>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<uint8_t>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<uint8_t>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<uint8_t>(0x80 | (cp & 0x3F)));
  }
}

// CPU reference: decode the string strictly and count its code points.
// Exits on a malformed sequence, because the generator must produce valid UTF-8.
static int32_t count_code_points(const uint8_t* s, int32_t len) {
  int32_t count = 0;
  int32_t j = 0;
  while (j < len) {
    const uint8_t b = s[j];
    int width;
    if (b < 0x80) {
      width = 1;
    } else if ((b & 0xE0) == 0xC0) {
      width = 2;
    } else if ((b & 0xF0) == 0xE0) {
      width = 3;
    } else if ((b & 0xF8) == 0xF0) {
      width = 4;
    } else {
      std::fprintf(stderr, "generator error: invalid lead byte 0x%02x\n", b);
      std::exit(1);
    }
    if (j + width > len) {
      std::fprintf(stderr, "generator error: truncated sequence\n");
      std::exit(1);
    }
    for (int k = 1; k < width; ++k) {
      if ((s[j + k] & 0xC0) != 0x80) {
        std::fprintf(stderr, "generator error: missing continuation byte\n");
        std::exit(1);
      }
    }
    j += width;
    ++count;
  }
  return count;
}

static bool run_case(int64_t n, std::mt19937_64& rng) {
  std::vector<int> per_string(n);
  std::uniform_int_distribution<int> length_dist(0, kMaxCodePoints);
  for (int64_t i = 0; i < n; ++i) per_string[i] = length_dist(rng);
  if (n >= 1000) {
    std::uniform_int_distribution<int64_t> row_dist(0, n - 1);
    for (int k = 0; k < kLongStrings; ++k) per_string[row_dist(rng)] = kLongStringCodePoints;
  }

  std::vector<int32_t> offsets(n + 1);
  std::vector<uint8_t> chars;
  std::vector<int32_t> expected(n);
  offsets[0] = 0;
  for (int64_t i = 0; i < n; ++i) {
    for (int c = 0; c < per_string[i]; ++c) append_utf8(chars, random_code_point(rng));
    offsets[i + 1] = static_cast<int32_t>(chars.size());
    expected[i] = count_code_points(chars.data() + offsets[i], offsets[i + 1] - offsets[i]);
    if (expected[i] != per_string[i]) {
      std::fprintf(stderr, "generator error: string %lld\n", static_cast<long long>(i));
      std::exit(1);
    }
  }

  int32_t* d_offsets = nullptr;
  uint8_t* d_chars = nullptr;
  int32_t* d_lengths = nullptr;
  CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_chars, chars.size()));
  CUDA_CHECK(cudaMalloc(&d_lengths, n * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_chars, chars.data(), chars.size(), cudaMemcpyHostToDevice));
  // Fill the output with -1 so that an unwritten row fails the check.
  CUDA_CHECK(cudaMemset(d_lengths, 0xFF, n * sizeof(int32_t)));

  // A launch needs at least one block, so n == 0 launches nothing.
  if (n > 0) {
    const int64_t blocks = (n + kBlockSize - 1) / kBlockSize;
    kernel<<<static_cast<unsigned>(blocks), kBlockSize>>>(d_offsets, d_chars, d_lengths, n);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> got(n);
  CUDA_CHECK(cudaMemcpy(got.data(), d_lengths, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_chars));
  CUDA_CHECK(cudaFree(d_lengths));

  int64_t mismatches = 0;
  int64_t first = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (got[i] != expected[i]) {
      if (first < 0) first = i;
      ++mismatches;
    }
  }
  if (mismatches != 0) {
    std::printf("string_length cuda n=%lld: FAIL, %lld mismatches, first at %lld: got %d, expected %d\n",
                static_cast<long long>(n), static_cast<long long>(mismatches),
                static_cast<long long>(first), got[first], expected[first]);
    return false;
  }
  std::printf("string_length cuda n=%lld bytes=%zu: pass\n", static_cast<long long>(n),
              chars.size());
  return true;
}

int main() {
  const int64_t sizes[] = {0, 1, 1000, 1000003};
  std::mt19937_64 rng(9);
  bool ok = true;
  for (const int64_t n : sizes) ok = run_case(n, rng) && ok;
  return ok ? 0 : 1;
}
