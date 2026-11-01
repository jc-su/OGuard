// valid_rows: compact the ids of the rows whose Arrow validity bit is set.
//
// One thread per row. Each warp ballots its valid lanes, warp 0 prefix-sums
// the warp counts with a shuffle scan, lane 31 claims one range of `rows`
// with a single atomic add on `count`, and every valid lane stores its id at
// block base + warp offset + rank within the warp, so the stores of a warp
// are contiguous. Output order is by block claim order, not by row id.
//
// Revision: the byte index into `valid` is tested against the bitmap size
// (n + 7) >> 3 before the load, the store position is tested against n, the
// cross-warp prefix sum is a shuffle scan instead of a loop over shared
// memory, and the atomic add is a PTX instruction instead of a library call
// that takes the pointer. The tests always pass under the task's contract
// (row i < n has its bit inside the bitmap; `count` is zero on entry, so
// every position is below n); they state the bound of each access for the
// memory checker.

#define THREADS 256
#define WARPS (THREADS / 32)

// atomicAdd as one PTX instruction; the library function is a call that takes
// the pointer, which the memory checker cannot follow.
__device__ __forceinline__ int atomic_add_global(int* p, int v)
{
    int old;
    asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

extern "C" __global__ void kernel(const unsigned char* __restrict__ valid,
                                  int* __restrict__ rows,
                                  int* __restrict__ count,
                                  long long n)
{
    __shared__ int warp_count[32];  // one slot per lane of warp 0; WARPS are used
    __shared__ int warp_offset[32];
    __shared__ int block_base;

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31;
    const unsigned wid = tid >> 5;
    const long long i = (long long)blockIdx.x * THREADS + tid;
    const long long nbytes = (n + 7) >> 3;

    bool ok = false;
    if (i < n) {
        const long long byte = i >> 3;
        if (byte < nbytes) ok = (valid[byte] >> (i & 7)) & 1;
    }

    const unsigned ballot = __ballot_sync(0xffffffffu, ok);
    if (lane == 0) warp_count[wid] = __popc(ballot);
    __syncthreads();

    if (tid < 32) {
        const int c = tid < WARPS ? warp_count[tid] : 0;
        int x = c;
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, x, off);
            if (lane >= off) x += y;
        }
        warp_offset[tid] = x - c;
        if (tid == 31) block_base = x > 0 ? atomic_add_global(count, x) : 0;
    }
    __syncthreads();

    if (ok) {
        const int rank = __popc(ballot & ((1u << lane) - 1u));
        const long long pos = (long long)block_base + warp_offset[wid] + rank;
        if (pos >= 0 && pos < n) rows[pos] = (int)i;
    }
}
