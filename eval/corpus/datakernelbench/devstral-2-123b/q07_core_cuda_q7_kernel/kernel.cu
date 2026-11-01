#define BLOCK_SIZE 512
#define NUM_GROUPS 8

extern "C" __global__ void kernel(
    const int32_t* __restrict__ supp_nation_code,
    const int32_t* __restrict__ cust_nation_code,
    const int32_t* __restrict__ l_shipdate_days,
    const double* __restrict__ l_extendedprice,
    const double* __restrict__ l_discount,
    int32_t shipdate_lo_days,
    int32_t shipdate_hi_days,
    double* __restrict__ out_revenue,
    int32_t* __restrict__ out_count,
    int64_t n_elements
) {
    __shared__ double block_revenue[NUM_GROUPS][BLOCK_SIZE];
    __shared__ int block_count[NUM_GROUPS][BLOCK_SIZE];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double my_revenue[NUM_GROUPS] = {0.0};
    int my_count[NUM_GROUPS] = {0};

    if (idx < n_elements) {
        int32_t sd = l_shipdate_days[idx];
        bool keep = (sd >= shipdate_lo_days) && (sd <= shipdate_hi_days);
        if (keep) {
            double price = l_extendedprice[idx];
            double disc = l_discount[idx];
            double volume = price * (1.0 - disc);

            // Extract year from shipdate days
            int64_t days = (int64_t)sd + 719468;
            int64_t era = days >= 0 ? days / 146097 : (days - 146096) / 146097;
            int64_t doe = days - era * 146097;
            int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            int32_t calc_y = (int32_t)(yoe + era * 400);
            int32_t doy = (int32_t)doe - (365 * yoe + yoe / 4 - yoe / 100);
            int32_t mp = (5 * doy + 2) / 153;
            int32_t m = mp < 10 ? mp + 3 : mp - 9;
            int32_t year = m <= 2 ? calc_y + 1 : calc_y;

            int32_t supp = supp_nation_code[idx];
            int32_t cust = cust_nation_code[idx];
            int32_t year_offset = year - 1995;

            // Build composite group key
            int64_t key = (int64_t)supp * 4 + (int64_t)cust * 2 + (int64_t)year_offset;

            // Accumulate in registers for each group
            for (int g = 0; g < NUM_GROUPS; ++g) {
                if (key == g) {
                    my_revenue[g] = volume;
                    my_count[g] = 1;
                    break;
                }
            }
        }
    }

    // Store partial results in shared memory
    for (int g = 0; g < NUM_GROUPS; ++g) {
        block_revenue[g][threadIdx.x] = my_revenue[g];
        block_count[g][threadIdx.x] = my_count[g];
    }

    __syncthreads();

    // Block-level reduction for each group
    for (int g = 0; g < NUM_GROUPS; ++g) {
        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                block_revenue[g][threadIdx.x] += block_revenue[g][threadIdx.x + s];
                block_count[g][threadIdx.x] += block_count[g][threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (block_revenue[g][0] > 0.0 || block_count[g][0] > 0) {
                atomicAdd(&out_revenue[g], block_revenue[g][0]);
            }
            if (block_count[g][0] > 0) {
                atomicAdd(&out_count[g], block_count[g][0]);
            }
        }
    }
}
