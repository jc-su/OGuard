// dictionary_decode: out[i] = dictionary[codes[i]].
//
// A coalesced read of the codes, a gather from the dictionary through the
// read-only cache (a small dictionary stays in L1/L2), and a coalesced store.
// The kernel uses a grid-stride loop, so it is correct for any grid; the
// launch uses one thread per row.
//
// Launch: 256 threads per block, ceil(n / 256) blocks.
#include <cstdint>

namespace dictionary_decode {
constexpr int kBlock = 256;
}  // namespace dictionary_decode

extern "C" __global__ void __launch_bounds__(dictionary_decode::kBlock)
kernel(const int32_t* __restrict__ codes, const double* __restrict__ dictionary,
       double* __restrict__ out, int64_t n, int64_t d) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    out[i] = __ldg(dictionary + __ldg(codes + i));
  }
}
