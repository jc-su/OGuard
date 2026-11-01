// GPU kernel for q6_sum (TPC-H Q6).
//
// Adds price[i] * discount[i] over the rows with
// date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi,
// and quantity[i] < qty_max, into result.
//
// Precondition: the caller zeroes result before each launch.
//
// Each block reduces its rows in shared memory and adds its partial
// sum into result with one atomic add per block, instead of one
// atomic add per matching row.

const int BLOCK = 256;

extern "C" __global__ void kernel(const int *shipdate, const double *discount,
                                   const double *quantity, const double *price,
                                   double *result, int date_lo, int date_hi,
                                   double disc_lo, double disc_hi,
                                   double qty_max, long long n) {
    __shared__ double partial[BLOCK];

    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double val = 0.0;
    if (i < n) {
        int sd = shipdate[i];
        double d = discount[i];
        double q = quantity[i];
        if (sd >= date_lo && sd < date_hi && d >= disc_lo && d <= disc_hi &&
            q < qty_max) {
            val = price[i] * d;
        }
    }
    partial[tid] = val;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, partial[0]);
    }
}
