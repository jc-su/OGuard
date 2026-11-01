// min_max: reduce `values` to its minimum and maximum, merged into
// result[0] and result[1] (initialized to +inf / -inf by the caller).
//
// Each block reduces its own tile in shared memory (a tree reduction),
// then only the block's single winning min and single winning max are
// atomically merged into the global result -- ceil(n / 256) atomics
// instead of n. CUDA has no native atomicMin/atomicMax for double, so
// they are implemented with an atomicCAS retry loop on the bit pattern.
#include <cfloat>

__device__ double atomicMinDouble(double *addr, double val) {
  unsigned long long int *addr_u = (unsigned long long int *)addr;
  unsigned long long int old = *addr_u, assumed;
  do {
    if (__longlong_as_double(old) <= val) break;
    assumed = old;
    old = atomicCAS(addr_u, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ double atomicMaxDouble(double *addr, double val) {
  unsigned long long int *addr_u = (unsigned long long int *)addr;
  unsigned long long int old = *addr_u, assumed;
  do {
    if (__longlong_as_double(old) >= val) break;
    assumed = old;
    old = atomicCAS(addr_u, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

extern "C" __global__ void kernel(const double *__restrict__ values,
                                   double *__restrict__ result,
                                   long long n) {
  __shared__ double smin[256];
  __shared__ double smax[256];

  int tid = threadIdx.x;
  long long i = (long long)blockIdx.x * blockDim.x + tid;

  if (i < n) {
    double v = values[i];
    smin[tid] = v;
    smax[tid] = v;
  } else {
    smin[tid] = HUGE_VAL;
    smax[tid] = -HUGE_VAL;
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smin[tid] = min(smin[tid], smin[tid + s]);
      smax[tid] = max(smax[tid], smax[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicMinDouble(&result[0], smin[0]);
    atomicMaxDouble(&result[1], smax[0]);
  }
}
