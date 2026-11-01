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
  std::uniform_real_distribution<double> d(-1, 1);
  std::vector<double> values(n);
  std::vector<int32_t> first(n);
  for (auto& x : values) x = d(rng);
  for (int64_t i = 0; i < n;) {
    int64_t len = std::min<int64_t>(1 + rng() % 3000, n - i);
    for (int64_t j = 0; j < len; j++) first[i + j] = i;
    i += len;
  }
  double* dv = to_dev(values);
  int32_t* df = to_dev(first);
  bool ok = true;
  for (int64_t k : {1, 3, 1024}) {
    std::vector<double> ref(n);
    for (int64_t i = 0; i < n; i++) ref[i] = i - k >= first[i] ? values[i - k] : NAN;
    double* dout = filled<double>(n, -12345.0);
    if (n > 0) kernel<<<blocks(n), 256>>>(dv, df, dout, n, k);
    sync();
    ok &= same_f64(from_dev(dout, n), ref);
    cudaFree(dout);
  }
  cudaFree(dv); cudaFree(df);
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
