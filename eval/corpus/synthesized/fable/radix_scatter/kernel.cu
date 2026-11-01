// radix_scatter: scatter row ids by one key byte, as one pass of an LSD
// radix sort.
//
// Each block takes a tile of 2048 keys (8 per thread, coalesced loads). It
// ranks every key inside the tile with an atomic add on a shared 256-bin
// histogram, claims one range per non-empty bin with a single global
// atomicAdd on that bin's cursor (at most 256 global atomics per 2048 keys
// instead of one per key), and stores each row id at bin base + rank. On
// exit every cursor has advanced to the end of its bin.

#define THREADS 256
#define ITEMS 8
#define TILE (THREADS * ITEMS)

extern "C" __global__ void kernel(const unsigned int* __restrict__ keys,
                                  int* __restrict__ cursors,
                                  int* __restrict__ out,
                                  long long n,
                                  long long shift)
{
    __shared__ int hist[256];
    __shared__ int base[256];
    const int tid = threadIdx.x;
    const long long t0 = (long long)blockIdx.x * TILE;
    const int sh = (int)shift;

    hist[tid] = 0;
    __syncthreads();

    int ranks[ITEMS];
    int bins[ITEMS];
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const long long i = t0 + k * THREADS + tid;
        if (i < n) {
            const int b = (int)((keys[i] >> sh) & 255u);
            bins[k] = b;
            ranks[k] = atomicAdd(&hist[b], 1);
        }
    }
    __syncthreads();

    const int c = hist[tid];
    if (c > 0) base[tid] = atomicAdd(&cursors[tid], c);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const long long i = t0 + k * THREADS + tid;
        if (i < n) out[base[bins[k]] + ranks[k]] = (int)i;
    }
}
