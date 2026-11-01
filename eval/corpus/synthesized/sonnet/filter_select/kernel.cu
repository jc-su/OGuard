// GPU kernel for filter_select.
//
// Writes the row ids i where quantity[i] < threshold into sel, in any
// order, and the number of matches into count.
//
// Precondition: the caller zeroes count before each launch, the same
// "zero on entry" convention used for the accumulator outputs
// elsewhere in this corpus.

extern "C" __global__ void kernel(const double *quantity, int *sel,
                                   int *count, double threshold,
                                   long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && quantity[i] < threshold) {
        int pos = atomicAdd(count, 1);
        sel[pos] = (int)i;
    }
}
