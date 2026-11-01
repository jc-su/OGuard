#include <stdint.h>

// True when candidate (av, ai) is a better minimum than (bv, bi): a smaller value,
// or an equal value at a smaller row. A row of -1 marks an empty candidate.
__device__ __forceinline__ bool better(int64_t av, int32_t ai, int64_t bv, int32_t bi) {
    if (ai < 0) return false;
    if (bi < 0) return true;
    return av < bv || (av == bv && ai < bi);
}

// row[0] holds the row of the current minimum, so the input array supplies its value
// and no other scratch is needed. Each block reduces its stride of rows and then
// replaces row[0] by compare-and-swap while its candidate beats the stored row.
extern "C" __global__ void kernel(const int64_t* __restrict__ values, int32_t* row, int64_t n) {
    __shared__ int64_t sv[256];
    __shared__ int32_t si[256];
    const int t = threadIdx.x;

    int64_t bv = 0;
    int32_t bi = -1;
    const int64_t stride = (int64_t)gridDim.x * 256;
    for (int64_t i = (int64_t)blockIdx.x * 256 + t; i < n; i += stride) {
        int64_t v = values[i];
        if (bi < 0 || v < bv) {
            bv = v;
            bi = (int32_t)i;
        }
    }
    sv[t] = bv;
    si[t] = bi;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (t < s && better(sv[t + s], si[t + s], sv[t], si[t])) {
            sv[t] = sv[t + s];
            si[t] = si[t + s];
        }
        __syncthreads();
    }

    if (t == 0 && si[0] >= 0) {
        const int64_t cv = sv[0];
        const int32_t ci = si[0];
        int32_t cur = *(volatile int32_t*)row;
        while (true) {
            if (cur >= 0) {
                int64_t v = values[cur];
                if (v < cv || (v == cv && cur < ci)) break;
            }
            int32_t old = atomicCAS(row, cur, ci);
            if (old == cur) break;
            cur = old;
        }
    }
}
