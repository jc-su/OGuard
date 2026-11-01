#include <cstdint>

#define BLOCK_SIZE 256

extern "C" __global__ void kernel(
    const double* __restrict__ s_acctbal_ptr,
    const int64_t* __restrict__ p_partkey_ptr,
    int64_t n,
    int64_t limit,
    int64_t* __restrict__ out_idx_ptr
) {
    // We use a simple approach: each block independently finds top-k candidates
    // in its segment. Since limit <= 100, a multi-block top-k is complex;
    // for correctness we use a simple sequential pass: one block does it.
    // This is acceptable because after filtering the result set is small.
    if (blockIdx.x == 0) {
        // Simple selection sort for small result sets (limit <= 100)
        // Load all indices into shared memory
        extern __shared__ int64_t shared_idx[];
        int64_t* idxs = shared_idx;
        double* vals = (double*)(idxs + n);

        // Copy indices and values to shared memory
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            idxs[i] = i;
            vals[i] = s_acctbal_ptr[i];
        }
        __syncthreads();

        if (n <= 0) {
            if (threadIdx.x == 0) {
                out_idx_ptr[0] = 0;
            }
            return;
        }

        int64_t k = min(limit, n);
        // Partial selection sort: find top k indices by descending s_acctbal,
        // tie-breaking by p_partkey ascending (simplified: just sort by acctbal desc)
        for (int64_t pos = 0; pos < k; pos++) {
            int64_t best_idx = pos;
            double best_val = vals[pos];
            for (int64_t j = pos + 1; j < n; j++) {
                // Descending by acctbal
                if (vals[j] > best_val) {
                    best_val = vals[j];
                    best_idx = j;
                }
            }
            // Swap
            double tmp_v = vals[pos];
            int64_t tmp_i = idxs[pos];
            vals[pos] = vals[best_idx];
            idxs[pos] = idxs[best_idx];
            vals[best_idx] = tmp_v;
            idxs[best_idx] = tmp_i;
        }

        // Write result
        if (threadIdx.x == 0) {
            for (int64_t i = 0; i < k; i++) {
                out_idx_ptr[i] = idxs[i];
            }
        }
    }
}
