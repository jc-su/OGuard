// valid_rows: compact the ids of the rows whose Arrow validity bit is set.
//
// One thread per row. Each warp ballots its valid lanes, the block
// prefix-sums the eight warp counts, claims one range of `rows` with a single
// atomicAdd on `count`, and every valid lane stores its id at
// block base + warp offset + rank within the warp, so the stores of a warp
// are contiguous. Output order is by block claim order, not by row id.

extern "C" __global__ void kernel(const unsigned char* __restrict__ valid,
                                  int* __restrict__ rows,
                                  int* __restrict__ count,
                                  long long n)
{
    __shared__ int warp_offset[8];
    __shared__ int block_base;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const long long i = (long long)blockIdx.x * 256 + tid;

    bool ok = false;
    if (i < n) ok = (valid[i >> 3] >> (i & 7)) & 1;

    const unsigned ballot = __ballot_sync(0xffffffffu, ok);
    if (lane == 0) warp_offset[wid] = __popc(ballot);
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < 8; ++w) {
            const int c = warp_offset[w];
            warp_offset[w] = total;
            total += c;
        }
        block_base = total > 0 ? atomicAdd(count, total) : 0;
    }
    __syncthreads();

    if (ok) {
        const int rank = __popc(ballot & ((1u << lane) - 1u));
        rows[block_base + warp_offset[wid] + rank] = (int)i;
    }
}
