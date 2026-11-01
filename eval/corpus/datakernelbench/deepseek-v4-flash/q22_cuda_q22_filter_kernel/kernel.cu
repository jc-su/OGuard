#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

extern "C" __global__ void kernel(
    const int64_t* __restrict__ custkey,
    const double* __restrict__ acctbal,
    const int32_t* __restrict__ cntrycode,
    const bool* __restrict__ in_mask,
    const int64_t* __restrict__ sorted_okeys,
    int64_t okeys_size,
    double avg_val,
    int32_t* __restrict__ out_code,
    double* __restrict__ out_bal,
    long long* __restrict__ out_n,
    int64_t n
) {
    __shared__ int64_t pred[BLOCK_SIZE];
    __shared__ int32_t s_code[BLOCK_SIZE];
    __shared__ double s_bal[BLOCK_SIZE];
    __shared__ int64_t prefix[BLOCK_SIZE];
    __shared__ long long block_offset;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int32_t my_code = 0;
    double my_bal = 0.0;
    int64_t my_pred = 0;

    if (idx < n) {
        bool in_set = in_mask[idx];
        double bal = acctbal[idx];
        int64_t ck = custkey[idx];

        // Binary search for NOT EXISTS
        int64_t lo = 0;
        int64_t hi = okeys_size;
        while (lo < hi) {
            int64_t mid = (lo + hi) >> 1;
            if (sorted_okeys[mid] < ck) lo = mid + 1;
            else hi = mid;
        }
        bool has_order = (lo < okeys_size) && (sorted_okeys[lo] == ck);

        bool keep = in_set && (bal > avg_val) && !has_order;
        if (keep) {
            my_code = cntrycode[idx];
            my_bal = bal;
            my_pred = 1;
        }
    }

    pred[threadIdx.x] = my_pred;
    s_code[threadIdx.x] = my_code;
    s_bal[threadIdx.x] = my_bal;
    __syncthreads();

    // Exclusive prefix sum
    prefix[threadIdx.x] = my_pred;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int64_t val = prefix[threadIdx.x];
        if (threadIdx.x >= offset) {
            val += prefix[threadIdx.x - offset];
        }
        __syncthreads();
        prefix[threadIdx.x] = val;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_offset = atomicAdd((unsigned long long*)out_n, (unsigned long long)prefix[blockDim.x - 1]);
    }
    __syncthreads();

    if (my_pred) {
        int64_t write_pos = (prefix[threadIdx.x] - 1) + block_offset;
        out_code[write_pos] = my_code;
        out_bal[write_pos] = my_bal;
    }
}
