#include <cuda.h>
#include <cuda_runtime.h>

extern "C" __global__ void kernel(
    const int64_t* __restrict__ orderkey,
    const int32_t* __restrict__ supp_id,
    const uint8_t* __restrict__ is_late,
    const int32_t* __restrict__ map,
    const int32_t* __restrict__ distinct,
    const int32_t* __restrict__ late_dist,
    const int64_t n_rows,
    uint32_t* __restrict__ out_counts)     // per‑supplier histogram (uint32)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows) return;

    int32_t sid = supp_id[idx];
    if (sid < 0) return;               // supplier not in selected nation
    if (!is_late[idx]) return;         // l1 must be late

    int64_t key = orderkey[idx];
    int32_t pos = map[key];
    if (pos < 0) return;               // safety guard (should not happen)

    int32_t d  = distinct[pos];
    int32_t ld = late_dist[pos];

    bool exists_other = (d >= 2);
    bool not_exists_other_late = (ld <= 1);

    if (exists_other && not_exists_other_late) {
        atomicAdd(&out_counts[sid], 1u);
    }
}
