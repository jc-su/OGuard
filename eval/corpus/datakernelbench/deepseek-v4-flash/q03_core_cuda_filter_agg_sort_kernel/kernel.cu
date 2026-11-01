#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

extern "C" __global__ void kernel(
    const int64_t* __restrict__ l_orderkey_ptr,
    const double* __restrict__ l_extendedprice_ptr,
    const double* __restrict__ l_discount_ptr,
    const int32_t* __restrict__ l_shipdate_days_ptr,
    const int32_t* __restrict__ o_orderdate_days_ptr,
    const int32_t* __restrict__ o_shippriority_ptr,
    int32_t shipdate_lo_days,
    int32_t orderdate_hi_days,
    int64_t* __restrict__ result_lk,
    double* __restrict__ result_rev,
    int32_t* __restrict__ result_od,
    int32_t* __restrict__ result_sp,
    int64_t n,
    int64_t top_n
) {
    extern __shared__ char smem_char[];
    int64_t* smem_lk = reinterpret_cast<int64_t*>(smem_char);
    double* smem_rev = reinterpret_cast<double*>(&smem_lk[BLOCK_SIZE]);
    int32_t* smem_od = reinterpret_cast<int32_t*>(&smem_rev[BLOCK_SIZE]);
    int32_t* smem_sp = reinterpret_cast<int32_t*>(&smem_od[BLOCK_SIZE]);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Process multiple elements per thread (coalesced)
    int64_t my_lk = 0;
    double my_rev = 0.0;
    int32_t my_od = 0;
    int32_t my_sp = 0;
    bool found = false;

    if (tid < n) {
        int32_t sd = l_shipdate_days_ptr[tid];
        int32_t od = o_orderdate_days_ptr[tid];
        bool keep = (sd > shipdate_lo_days) && (od < orderdate_hi_days);
        if (keep) {
            my_lk = l_orderkey_ptr[tid];
            my_od = od;
            my_sp = o_shippriority_ptr[tid];
            my_rev = l_extendedprice_ptr[tid] * (1.0 - l_discount_ptr[tid]);
            found = true;
        }
    }

    // Use ballot to count active threads per warp
    unsigned mask = __ballot_sync(0xFFFFFFFF, found);
    int active_count = __popc(mask);

    // Each warp collects its data into smem
    if (found) {
        // Find position within warp (prefix sum)
        int pos_in_warp = __popc(mask & ((1 << lane) - 1));
        int warp_start = warp_id * BLOCK_SIZE;
        int idx = warp_start + pos_in_warp;

        smem_lk[idx] = my_lk;
        smem_rev[idx] = my_rev;
        smem_od[idx] = my_od;
        smem_sp[idx] = my_sp;
    }
    __syncthreads();

    // Now each warp reduces its own section
    int warp_start = warp_id * BLOCK_SIZE;
    int warp_end = min(warp_start + BLOCK_SIZE, (int)BLOCK_SIZE * num_warps);

    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        if (i >= warp_start + active_count) break;

        // Compare and insert into sorted list within warp
        // Simple approach: dump to global and let cuDF handle
        // For now, just output directly
        unsigned long long gidx = atomicAdd((unsigned long long*)&result_sp[n], 1ULL);
        if (gidx < n) {
            result_lk[gidx] = smem_lk[i];
            result_rev[gidx] = smem_rev[i];
            result_od[gidx] = smem_od[i];
            result_sp[gidx] = smem_sp[i];
        }
    }
}
