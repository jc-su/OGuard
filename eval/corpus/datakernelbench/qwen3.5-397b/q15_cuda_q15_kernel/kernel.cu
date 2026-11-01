#include <cuda_runtime.h>

__device__ int binary_search(const long long* arr, int n, long long val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

extern "C" __global__ void kernel(
    const long long* l_suppkey,
    const double* l_price,
    const double* l_disc,
    const int* l_shipdate,
    int shipdate_lo,
    int shipdate_hi,
    const long long* s_suppkey_sorted,
    int n_suppliers,
    double* revenue_out,
    int n_lineitems
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_lineitems) return;

    int sd = l_shipdate[idx];
    if (sd < shipdate_lo || sd >= shipdate_hi) return;

    long long supp = l_suppkey[idx];
    double rev = l_price[idx] * (1.0 - l_disc[idx]);

    int pos = binary_search(s_suppkey_sorted, n_suppliers, supp);
    if (pos < n_suppliers && s_suppkey_sorted[pos] == supp) {
        atomicAdd(&revenue_out[pos], rev);
    }
}
