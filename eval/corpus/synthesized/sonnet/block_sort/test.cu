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
  bool ok = true;
  for (int variant = 0; variant < 2; variant++) {
    std::vector<int32_t> keys(n);
    for (auto& x : keys) x = variant ? (int32_t)(rng() % 10) : (int32_t)rng();
    if (!variant)
      for (int64_t i = 0; i < n; i++) {
        if (rng() % 50 == 0) keys[i] = INT32_MAX;
        if (rng() % 50 == 0) keys[i] = INT32_MIN;
      }
    std::vector<int32_t> ref = keys;
    for (int64_t t = 0; t < n; t += 1024) std::sort(ref.begin() + t, ref.begin() + std::min<int64_t>(t + 1024, n));
    int32_t *dk = to_dev(keys), *dout = filled<int32_t>(n, 777);
    if (n > 0) kernel<<<(unsigned)((n + 1023) / 1024), 256>>>(dk, dout, n);
    sync();
    ok &= from_dev(dout, n) == ref;
    cudaFree(dk); cudaFree(dout);
  }
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
