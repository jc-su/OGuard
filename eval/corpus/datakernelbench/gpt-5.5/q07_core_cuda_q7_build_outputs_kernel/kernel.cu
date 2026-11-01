#include <cstdint>

#define NUM_GROUPS 8

extern "C" __global__ void kernel(
    const double* __restrict__ revenue8_ptr,
    int32_t* __restrict__ out_supp_ptr,
    int32_t* __restrict__ out_cust_ptr,
    int32_t* __restrict__ out_year_ptr,
    double* __restrict__ out_revenue_ptr,
    int32_t* __restrict__ count_ptr
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t count = 0;
        #pragma unroll
        for (int gid = 0; gid < NUM_GROUPS; ++gid) {
            const int32_t supp = gid / 4;
            const int32_t cust = (gid / 2) % 2;
            const int32_t year = (gid % 2) + 1995;
            const bool valid_pair = ((supp == 0 && cust == 1) || (supp == 1 && cust == 0));
            const double rev = revenue8_ptr[gid];
            if (valid_pair && rev > 0.0) {
                out_supp_ptr[count] = supp;
                out_cust_ptr[count] = cust;
                out_year_ptr[count] = year;
                out_revenue_ptr[count] = rev;
                ++count;
            }
        }
        count_ptr[0] = count;
    }
}
