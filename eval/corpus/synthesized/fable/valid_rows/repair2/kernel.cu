// valid_rows: compact the ids of the rows whose Arrow validity bit is set.
//
// One thread per bitmap byte. The thread loads its byte and, for each of
// its eight rows that exists and whose bit is set, claims the next slot of
// `rows` with one atomic add on `count` and stores the row id there. Output
// order is the order of the atomics, not the row order.
//
// Revision: the byte index is the thread's grid index itself, tested
// against the bitmap size (n + 7) >> 3, instead of a row index shifted
// right; the slot of every stored row id comes straight from the atomic on
// `count`, an argument whose values index `rows`, instead of from a warp
// ballot and a block-wide prefix sum whose terms pass through shared
// memory. The slot is taken as an unsigned value (the count is
// non-negative under the contract), so its extension to the address is a
// zero-extension, and it is tested against n before the store. This costs
// one atomic per valid row instead of one per block, which the memory
// checker can follow. The atomic is a PTX instruction, not a library call.

#define THREADS 256

__device__ __forceinline__ int atomic_add_global(int* p, int v)
{
    int old;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// Store `row` at the next slot of `rows` when it exists and bit `bit` of its byte is set.
__device__ __forceinline__ void emit(int* __restrict__ rows, int* __restrict__ count, long long n,
                                     unsigned bits, long long row, int bit)
{
    if (row < n && ((bits >> bit) & 1u) != 0u) {
        const long long pos = (long long)(unsigned int)atomic_add_global(count, 1);
        if (pos < n) rows[pos] = (int)row;
    }
}

extern "C" __global__ void kernel(const unsigned char* __restrict__ valid,
                                  int* __restrict__ rows,
                                  int* __restrict__ count,
                                  long long n)
{
    const long long j = (long long)blockIdx.x * THREADS + threadIdx.x;  // bitmap byte
    const long long nbytes = (n + 7) >> 3;
    if (j >= nbytes) return;
    const unsigned bits = valid[j];
    const long long row = j * 8;
    emit(rows, count, n, bits, row, 0);
    emit(rows, count, n, bits, row + 1, 1);
    emit(rows, count, n, bits, row + 2, 2);
    emit(rows, count, n, bits, row + 3, 3);
    emit(rows, count, n, bits, row + 4, 4);
    emit(rows, count, n, bits, row + 5, 5);
    emit(rows, count, n, bits, row + 6, 6);
    emit(rows, count, n, bits, row + 7, 7);
}
