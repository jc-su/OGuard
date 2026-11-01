#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

#include "kernel.cu"

#define CK(x)                                                                 \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(2);                                                                \
    }                                                                         \
  } while (0)

template <class T> static T* to_dev(const std::vector<T>& v) {
  T* p = nullptr;
  CK(cudaMalloc(&p, v.size() * sizeof(T)));
  if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
template <class T> static T* filled(size_t n, T v) { return to_dev(std::vector<T>(n, v)); }
template <class T> static std::vector<T> from_dev(const T* p, size_t n) {
  std::vector<T> v(n);
  if (n) CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
static unsigned blocks(int64_t n) { return (unsigned)((n + 255) / 256); }
static void sync() {
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
}
static bool same_f64(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size()) return false;
  for (size_t i = 0; i < x.size(); i++) {
    if (std::isnan(x[i]) && std::isnan(y[i])) continue;
    if (x[i] != y[i]) return false;
  }
  return true;
}

static const int64_t SIZES[] = {0, 1, 1000, 1024, 1025, 1000003};

static bool run_case(int64_t n) {
  std::mt19937_64 rng(n + 1);
  std::vector<int64_t> keys(n);
  for (auto& k : keys) k = (int64_t)rng();
  int64_t* dk = to_dev(keys);
  bool ok = true;
  for (int64_t words : {1, 64, 4096, 1 << 16}) {
    std::vector<uint32_t> ref(words, 0);
    for (int64_t key : keys) {
      uint64_t k = (uint64_t)key;
      uint64_t p1 = ((k * 0x9E3779B97F4A7C15ULL) >> 32) % (32 * (uint64_t)words);
      uint64_t p2 = ((k * 0xC2B2AE3D27D4EB4FULL) >> 32) % (32 * (uint64_t)words);
      ref[p1 / 32] |= 1u << (p1 % 32);
      ref[p2 / 32] |= 1u << (p2 % 32);
    }
    uint32_t* db = filled<uint32_t>(words, 0u);
    if (n > 0) kernel<<<blocks(n), 256>>>(dk, db, n, words);
    sync();
    ok &= from_dev(db, words) == ref;
    cudaFree(db);
  }
  cudaFree(dk);
  return ok;
}

int main() {
  int bad = 0;
  for (int64_t n : SIZES) {
    bool ok = run_case(n);
    printf("n=%lld %s\n", (long long)n, ok ? "PASS" : "FAIL");
    bad += !ok;
  }
  return bad ? 1 : 0;
}
