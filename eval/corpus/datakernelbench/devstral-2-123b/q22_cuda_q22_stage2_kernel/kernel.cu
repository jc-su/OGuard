extern "C" __global__ void kernel(
    const int64_t* __restrict__ c_custkey_ptr,
    const double* __restrict__ c_acctbal_ptr,
    const int32_t* __restrict__ cntrycode_ptr,
    const bool* __restrict__ in_cntrycode_mask_ptr,
    const int64_t* __restrict__ o_custkey_ptr,
    double avg_val,
    double* __restrict__ numcust_buff,
    double* __restrict__ totacctbal_buff,
    int64_t n_customer,
    int64_t n_orders
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_customer) {
        int64_t custkey = c_custkey_ptr[idx];
        double acctbal = c_acctbal_ptr[idx];
        int32_t cntry = cntrycode_ptr[idx];
        bool in_mask = in_cntrycode_mask_ptr[idx];

        int64_t left = 0, right = n_orders;
        bool exists = false;
        while (left < right) {
            int64_t mid = (left + right) / 2;
            int64_t mid_key = o_custkey_ptr[mid];
            if (mid_key < custkey) {
                left = mid + 1;
            } else if (mid_key > custkey) {
                right = mid;
            } else {
                exists = true;
                break;
            }
        }

        if (in_mask && (acctbal > avg_val) && !exists) {
            atomicAdd(numcust_buff + cntry, 1.0);
            atomicAdd(totacctbal_buff + cntry, acctbal);
        }
    }
}
