#include <cstdint>

extern "C" __global__ void kernel(
    const double* __restrict__ revenue,
    int64_t* __restrict__ sorted_ids,
    double* __restrict__ sorted_revenue,
    int64_t num_nations
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int64_t i = 0; i < num_nations; ++i) {
        sorted_ids[i] = i;
        sorted_revenue[i] = revenue[i];
    }

    for (int64_t i = 1; i < num_nations; ++i) {
        double key_rev = sorted_revenue[i];
        int64_t key_id = sorted_ids[i];
        int64_t j = i - 1;

        while (
            j >= 0 &&
            (
                sorted_revenue[j] < key_rev ||
                (sorted_revenue[j] == key_rev && sorted_ids[j] > key_id)
            )
        ) {
            sorted_revenue[j + 1] = sorted_revenue[j];
            sorted_ids[j + 1] = sorted_ids[j];
            --j;
        }

        sorted_revenue[j + 1] = key_rev;
        sorted_ids[j + 1] = key_id;
    }
}
