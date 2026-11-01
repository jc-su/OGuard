// Pass 2: Evaluate EXISTS and NOT EXISTS conditions per order, then count numwait
extern "C" __global__ void kernel(
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ lineitems,
    const int32_t* __restrict__ supp_map,
    int32_t* __restrict__ out_counts,
    int64_t num_orders_f,
    int64_t max_suppkey
) {
    int64_t dense_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dense_idx >= num_orders_f) return;

    int count = counts[dense_idx];
    if (count == 0) return;
    if (count > 16) count = 16; 

    int32_t items[16];
    for (int i = 0; i < count; i++) {
        items[i] = lineitems[dense_idx * 16 + i];
    }

    int u_supp[16];
    bool u_late[16];
    int u_count = 0;

    // Deduplicate suppliers and compute per-supplier late status
    for (int i = 0; i < count; i++) {
        int s = items[i] >> 1;
        bool l = items[i] & 1;
        bool found = false;
        for (int j = 0; j < u_count; j++) {
            if (u_supp[j] == s) {
                u_late[j] = u_late[j] || l;
                found = true;
                break;
            }
        }
        if (!found) {
            u_supp[u_count] = s;
            u_late[u_count] = l;
            u_count++;
        }
    }

    int late_count = 0;
    for (int j = 0; j < u_count; j++) {
        if (u_late[j]) late_count++;
    }

    // Evaluate for each original lineitem
    for (int i = 0; i < count; i++) {
        int s = items[i] >> 1;
        bool l = items[i] & 1;

        // Conditions: lineitem is late, >1 distinct suppliers, <=1 late distinct suppliers
        if (l && u_count >= 2 && late_count <= 1) {
            if (s >= 0 && s <= max_suppkey) {
                int s_id = supp_map[s];
                if (s_id != -1) {
                    atomicAdd(&out_counts[s_id], 1);
                }
            }
        }
    }
}
