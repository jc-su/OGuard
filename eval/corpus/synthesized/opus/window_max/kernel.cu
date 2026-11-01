// Window max: out[i] is the largest of values[max(0, i - w + 1) : i + 1].
#include <cstdint>

constexpr int kMaxW = 1024;
// Outputs per block tile. A tile reads kOutTile + w - 1 values.
constexpr int kOutTile = 1024;
constexpr int kSpan = kOutTile + kMaxW;

// Each block computes kOutTile outputs at a time from the kOutTile + w - 1 values they
// cover, held in shared memory. Pass s turns maxima over 2^s values into maxima over
// 2^(s + 1) values, until the length k is the largest power of two not above w. A
// window of w values is then the union of the k values at its start and the k values
// at its end.
extern "C" __global__ void kernel(const int64_t* values, int64_t* out, int64_t n, int64_t w) {
    __shared__ int64_t buf[2][kSpan];
    const int t = threadIdx.x;
    const int step = blockDim.x;
    const int span = kOutTile + static_cast<int>(w) - 1;
    const int64_t ntiles = (n + kOutTile - 1) / kOutTile;
    for (int64_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int64_t base = tile * kOutTile;
        // Local position 0 is row base - w + 1. Rows outside [0, n) hold INT64_MIN,
        // which never changes a maximum.
        for (int j = t; j < span; j += step) {
            const int64_t row = base - w + 1 + j;
            buf[0][j] = row >= 0 && row < n ? values[row] : INT64_MIN;
        }
        __syncthreads();

        int src = 0;
        int k = 1;
        for (; 2 * k <= w; k *= 2) {
            // buf[src][j] is the maximum of local [j, j + k) for j <= span - k.
            for (int j = t; j <= span - 2 * k; j += step) {
                buf[1 - src][j] = max(buf[src][j], buf[src][j + k]);
            }
            __syncthreads();
            src = 1 - src;
        }

        // The window of row base + o is local [o, o + w - 1].
        for (int o = t; o < kOutTile; o += step) {
            const int64_t row = base + o;
            if (row < n) out[row] = max(buf[src][o], buf[src][o + w - k]);
        }
        // buf is written again by the next tile.
        __syncthreads();
    }
}
