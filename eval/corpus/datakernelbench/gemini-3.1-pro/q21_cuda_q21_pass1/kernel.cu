#include <cstdint>

// Pass 1: Filter lineitem by 'F' orders, count per order, and pack suppkey + late flag
extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey,
    const int64_t* __restrict__ l_suppkey,
    const int32_t* __restrict__ l_receiptdate,
    const int32_t* __restrict__ l_commitdate,
    const int32_t* __restrict__ order_map,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ lineitems,
    int64_t n_lineitems,
    int64_t max_orderkey
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitems) return;

    int64_t okey = l_orderkey[idx];
    if (okey > max_orderkey || okey < 0) return;

    int32_t dense_idx = order_map[okey];
    if (dense_idx >= 0) {
        // Read remaining columns only for qualifying orders to save memory bandwidth
        int32_t rdate = l_receiptdate[idx];
        int32_t cdate = l_commitdate[idx];
        bool late = rdate > cdate;
        int32_t suppkey = (int32_t)l_suppkey[idx];
        
        // Pack: suppkey in upper 31 bits, late flag in lowest bit
        uint32_t packed = (suppkey << 1) | (late ? 1 : 0);
        
        int pos = atomicAdd(&counts[dense_idx], 1);
        if (pos < 16) {
            lineitems[dense_idx * 16 + pos] = packed;
        }
    }
}
