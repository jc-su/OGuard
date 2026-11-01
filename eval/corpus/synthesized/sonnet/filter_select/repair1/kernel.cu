// GPU kernel for filter_select.
//
// Writes the row ids i where quantity[i] < threshold into sel, in any
// order, and the number of matches into count.
//
// count is an output buffer, so its value on entry is not part of the
// task's contract: every store must be safe for whatever count
// already holds. The atomic counter can therefore return a position
// outside [0, n) if count did not start at zero, so the store into
// sel is guarded by an explicit two-sided bounds check before it
// runs. Under the documented precondition that the caller zeroes
// count before each launch, the guard never rejects a valid position,
// because at most n rows can match and each match gets a distinct
// position in [0, n).

extern "C" __global__ void kernel(const double *quantity, int *sel,
                                   int *count, double threshold,
                                   long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && quantity[i] < threshold) {
        int pos = atomicAdd(count, 1);
        if (pos >= 0 && pos < n) {
            sel[pos] = (int)i;
        }
    }
}
