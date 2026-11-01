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
  std::vector<int32_t> off_a(n + 1, 0), off_b(n + 1, 0);
  std::vector<uint8_t> ca, cb;
  std::vector<int8_t> ref(n);
  for (int64_t i = 0; i < n; i++) {
    int len = rng() % 12, mode = rng() % 4;  // 0 equal, 1 last byte differs, 2 longer, 3 independent
    std::vector<uint8_t> s(len), t(len + (mode == 2));
    for (auto& c : s) c = 97 + rng() % 3;
    for (auto& c : t) c = 97 + rng() % 3;
    if (mode != 3) for (int j = 0; j < len; j++) t[j] = s[j];
    if (mode == 1 && len > 0) t[len - 1] ^= 1;
    ca.insert(ca.end(), s.begin(), s.end());
    cb.insert(cb.end(), t.begin(), t.end());
    off_a[i + 1] = ca.size();
    off_b[i + 1] = cb.size();
    ref[i] = s.size() == t.size() && memcmp(s.data(), t.data(), s.size()) == 0;
  }
  int32_t *doa = to_dev(off_a), *dob = to_dev(off_b);
  uint8_t *dca = to_dev(ca), *dcb = to_dev(cb);
  int8_t* dm = filled<int8_t>(n, 5);
  if (n > 0) kernel<<<blocks(n), 256>>>(doa, dca, dob, dcb, dm, n);
  sync();
  bool ok = from_dev(dm, n) == ref;
  cudaFree(doa); cudaFree(dob); cudaFree(dca); cudaFree(dcb); cudaFree(dm);
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
