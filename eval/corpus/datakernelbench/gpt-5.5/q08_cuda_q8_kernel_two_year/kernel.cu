#include <cstdint>

#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int32_t* __restrict__ year_ptr,
    const double* __restrict__ price_ptr,
    const double* __restrict__ discount_ptr,
    const int64_t* __restrict__ suppkey_ptr,
    const uint8_t* __restrict__ target_supp_ptr,
    int64_t target_supp_len,
    int32_t base_year,
    double* __restrict__ out_target_ptr,
    double* __restrict__ out_total_ptr,
    int64_t* __restrict__ out_count_ptr,
    int64_t n_elements
) {
    __shared__ double s_total0[BLOCK_SIZE];
    __shared__ double s_total1[BLOCK_SIZE];
    __shared__ double s_target0[BLOCK_SIZE];
    __shared__ double s_target1[BLOCK_SIZE];
    __shared__ unsigned long long s_count0[BLOCK_SIZE];
    __shared__ unsigned long long s_count1[BLOCK_SIZE];

    int tid = threadIdx.x;
    double total0 = 0.0;
    double total1 = 0.0;
    double target0 = 0.0;
    double target1 = 0.0;
    unsigned long long count0 = 0ULL;
    unsigned long long count1 = 0ULL;

    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid; idx < n_elements; idx += stride) {
        int32_t y = year_ptr[idx];
        int32_t g = y - base_year;
        if (g == 0 || g == 1) {
            double vol = price_ptr[idx] * (1.0 - discount_ptr[idx]);
            int64_t sk = suppkey_ptr[idx];
            bool is_target = (sk >= 0 && sk < target_supp_len && target_supp_ptr[sk] != 0);
            if (g == 0) {
                total0 += vol;
                target0 += is_target ? vol : 0.0;
                count0 += 1ULL;
            } else {
                total1 += vol;
                target1 += is_target ? vol : 0.0;
                count1 += 1ULL;
            }
        }
    }

    s_total0[tid] = total0;
    s_total1[tid] = total1;
    s_target0[tid] = target0;
    s_target1[tid] = target1;
    s_count0[tid] = count0;
    s_count1[tid] = count1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_total0[tid] += s_total0[tid + s];
            s_total1[tid] += s_total1[tid + s];
            s_target0[tid] += s_target0[tid + s];
            s_target1[tid] += s_target1[tid + s];
            s_count0[tid] += s_count0[tid + s];
            s_count1[tid] += s_count1[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out_total_ptr[0], s_total0[0]);
        atomicAdd(&out_total_ptr[1], s_total1[0]);
        atomicAdd(&out_target_ptr[0], s_target0[0]);
        atomicAdd(&out_target_ptr[1], s_target1[0]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&out_count_ptr[0]), s_count0[0]);
        atomicAdd(reinterpret_cast<unsigned long long*>(&out_count_ptr[1]), s_count1[0]);
    }
}
