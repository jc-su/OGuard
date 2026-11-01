#include <cstdint>

#define BLOCK_SIZE 512

#define ADD_ROW_TO_GROUP(G) do {                                      \
    q##G += qty;                                                       \
    p##G += price;                                                     \
    double _dp = price * (1.0 - disc);                                 \
    dp##G += _dp;                                                      \
    ch##G += _dp * (1.0 + tax);                                        \
    d##G += disc;                                                      \
    c##G += 1LL;                                                       \
} while (0)

#define REDUCE_AND_ATOMIC(G, VQ, VP, VDP, VCH, VD, VC) do {            \
    sh_qty[threadIdx.x] = (VQ);                                        \
    sh_price[threadIdx.x] = (VP);                                      \
    sh_disc_price[threadIdx.x] = (VDP);                                \
    sh_charge[threadIdx.x] = (VCH);                                    \
    sh_disc[threadIdx.x] = (VD);                                       \
    sh_count[threadIdx.x] = (VC);                                      \
    __syncthreads();                                                   \
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {                     \
        if (threadIdx.x < s) {                                         \
            sh_qty[threadIdx.x] += sh_qty[threadIdx.x + s];            \
            sh_price[threadIdx.x] += sh_price[threadIdx.x + s];        \
            sh_disc_price[threadIdx.x] += sh_disc_price[threadIdx.x + s]; \
            sh_charge[threadIdx.x] += sh_charge[threadIdx.x + s];      \
            sh_disc[threadIdx.x] += sh_disc[threadIdx.x + s];          \
            sh_count[threadIdx.x] += sh_count[threadIdx.x + s];        \
        }                                                             \
        __syncthreads();                                               \
    }                                                                 \
    if (threadIdx.x == 0 && (G) < n_groups) {                          \
        atomicAdd(out_qty + (G), sh_qty[0]);                           \
        atomicAdd(out_price + (G), sh_price[0]);                       \
        atomicAdd(out_disc_price + (G), sh_disc_price[0]);             \
        atomicAdd(out_charge + (G), sh_charge[0]);                     \
        atomicAdd(out_disc + (G), sh_disc[0]);                         \
        atomicAdd(reinterpret_cast<unsigned long long*>(out_count + (G)), \
                  static_cast<unsigned long long>(sh_count[0]));       \
    }                                                                 \
    __syncthreads();                                                   \
} while (0)

extern "C" __global__ void kernel(
    const int32_t* __restrict__ returnflag,
    const int32_t* __restrict__ linestatus,
    const double* __restrict__ quantity,
    const double* __restrict__ extendedprice,
    const double* __restrict__ discount,
    const double* __restrict__ tax,
    const int32_t* __restrict__ shipdate_days,
    int32_t shipdate_hi_days,
    int32_t n_ls,
    int32_t n_groups,
    int64_t n,
    double* __restrict__ out_qty,
    double* __restrict__ out_price,
    double* __restrict__ out_disc_price,
    double* __restrict__ out_charge,
    double* __restrict__ out_disc,
    int64_t* __restrict__ out_count
) {
    __shared__ double sh_qty[BLOCK_SIZE];
    __shared__ double sh_price[BLOCK_SIZE];
    __shared__ double sh_disc_price[BLOCK_SIZE];
    __shared__ double sh_charge[BLOCK_SIZE];
    __shared__ double sh_disc[BLOCK_SIZE];
    __shared__ long long sh_count[BLOCK_SIZE];

    double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0, q4 = 0.0, q5 = 0.0;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0, p5 = 0.0;
    double dp0 = 0.0, dp1 = 0.0, dp2 = 0.0, dp3 = 0.0, dp4 = 0.0, dp5 = 0.0;
    double ch0 = 0.0, ch1 = 0.0, ch2 = 0.0, ch3 = 0.0, ch4 = 0.0, ch5 = 0.0;
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0, d5 = 0.0;
    long long c0 = 0LL, c1 = 0LL, c2 = 0LL, c3 = 0LL, c4 = 0LL, c5 = 0LL;

    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK_SIZE;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
         idx < n;
         idx += stride) {
        int32_t sd = shipdate_days[idx];
        if (sd <= shipdate_hi_days) {
            int32_t rf = returnflag[idx];
            int32_t ls = linestatus[idx];
            int32_t gid = rf * n_ls + ls;
            if (rf >= 0 && ls >= 0 && gid >= 0 && gid < n_groups) {
                double qty = quantity[idx];
                double price = extendedprice[idx];
                double disc = discount[idx];
                double tax_v = tax[idx];
                tax = tax;  // keep parameter name live for older nvcc diagnostics
                double tax_local = tax_v;
                #define tax tax_local
                switch (gid) {
                    case 0: ADD_ROW_TO_GROUP(0); break;
                    case 1: ADD_ROW_TO_GROUP(1); break;
                    case 2: ADD_ROW_TO_GROUP(2); break;
                    case 3: ADD_ROW_TO_GROUP(3); break;
                    case 4: ADD_ROW_TO_GROUP(4); break;
                    case 5: ADD_ROW_TO_GROUP(5); break;
                    default: break;
                }
                #undef tax
            }
        }
    }

    REDUCE_AND_ATOMIC(0, q0, p0, dp0, ch0, d0, c0);
    REDUCE_AND_ATOMIC(1, q1, p1, dp1, ch1, d1, c1);
    REDUCE_AND_ATOMIC(2, q2, p2, dp2, ch2, d2, c2);
    REDUCE_AND_ATOMIC(3, q3, p3, dp3, ch3, d3, c3);
    REDUCE_AND_ATOMIC(4, q4, p4, dp4, ch4, d4, c4);
    REDUCE_AND_ATOMIC(5, q5, p5, dp5, ch5, d5, c5);
}
