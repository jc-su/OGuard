#include <stdint.h>

// One block scans the whole column: the kernel has no scratch memory for a
// cross-block carry. Thread t owns rows tile + 8 * t .. tile + 8 * t + 7, which
// are exactly the bits of one validity byte.
extern "C" __global__ void kernel(const int64_t* values, const uint8_t* valid,
                                  int64_t* out, int64_t n)
{
    __shared__ int s_has[256];
    __shared__ int64_t s_val[256];
    const int t = threadIdx.x;
    int64_t carry = 0;
    for (int64_t tile = 0; tile < n; tile += 2048) {
        const int64_t base = tile + (int64_t)t * 8;
        unsigned bits = 0;
        if (base < n) bits = valid[base >> 3];
        int has = 0;
        int64_t val = 0;
        for (int k = 0; k < 8; k++) {
            if (base + k < n && ((bits >> k) & 1u)) {
                has = 1;
                val = values[base + k];
            }
        }
        // inclusive scan over (has, val); combine(a, b) = b if b.has else a
        s_has[t] = has;
        s_val[t] = val;
        __syncthreads();
        for (int d = 1; d < 256; d <<= 1) {
            int prev_has = 0;
            int64_t prev_val = 0;
            if (t >= d) {
                prev_has = s_has[t - d];
                prev_val = s_val[t - d];
            }
            __syncthreads();
            if (t >= d && s_has[t] == 0) {
                s_has[t] = prev_has;
                s_val[t] = prev_val;
            }
            __syncthreads();
        }
        int64_t state = carry;
        if (t > 0 && s_has[t - 1]) state = s_val[t - 1];
        const int64_t next_carry = s_has[255] ? s_val[255] : carry;
        __syncthreads();
        for (int k = 0; k < 8; k++) {
            if (base + k < n) {
                if ((bits >> k) & 1u) state = values[base + k];
                out[base + k] = state;
            }
        }
        carry = next_carry;
    }
}
