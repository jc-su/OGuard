#include <cstdint>

extern "C" __global__ void kernel(
    const int32_t* __restrict__ l_commitdate,
    const int32_t* __restrict__ l_receiptdate,
    const int32_t* __restrict__ l_shipdate,
    const int32_t* __restrict__ l_shipmode,
    const int32_t* __restrict__ o_orderpriority_high,
    int32_t receipt_lo_days,
    int32_t receipt_hi_days,
    int* __restrict__ out,
    int64_t n_elements
) {
    __shared__ int block_counts[4]; // 4 counters per block

    // Initialize shared memory to 0 by thread 0
    if (threadIdx.x == 0) {
        block_counts[0] = 0;
        block_counts[1] = 0;
        block_counts[2] = 0;
        block_counts[3] = 0;
    }
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int32_t commit = l_commitdate[idx];
        int32_t receipt = l_receiptdate[idx];
        int32_t ship = l_shipdate[idx];
        int32_t mode = l_shipmode[idx];
        int32_t high = o_orderpriority_high[idx];

        bool keep = (commit < receipt) && (ship < commit) && (receipt >= receipt_lo_days) && (receipt < receipt_hi_days);

        if (keep) {
            // mode: 0 for MAIL, 1 for SHIP (encoded as in reference: shipmode1=MAIL=0, shipmode2=SHIP=1)
            // high: 1 for high priority, 0 for low
            // high counts come before low counts in output (matching reference: high then low)
            int idx_count = mode * 2 + (1 - high);
            atomicAdd(&block_counts[idx_count], 1);
        }
    }

    __syncthreads();

    // Add block counts to global output
    if (threadIdx.x == 0) {
        atomicAdd(&out[0], block_counts[0]);
        atomicAdd(&out[1], block_counts[1]);
        atomicAdd(&out[2], block_counts[2]);
        atomicAdd(&out[3], block_counts[3]);
    }
}
