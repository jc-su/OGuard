// valid_rows: compact the ids of the valid rows of an Arrow validity bitmap.
//
// One thread per row. Rather than one atomicAdd per valid row against a
// single global counter, each warp reserves its output slots with a single
// atomicAdd covering every valid lane in the warp (warp-aggregated atomics):
// __ballot_sync collects which lanes want to write, __popc/lane-mask give
// each lane its rank within the warp's group, the elected leader lane (the
// lowest set bit in the ballot) does the one atomicAdd for the whole warp,
// and __shfl_sync broadcasts the warp's base offset back to every lane.
extern "C" __global__ void kernel(const unsigned char *valid, int *rows,
                                   int *count, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;

  bool wants = false;
  if (i < n) {
    unsigned char byte = valid[i >> 3];
    wants = (byte >> (i & 7)) & 1u;
  }

  const unsigned mask = 0xffffffffu; // every lane reaches this ballot
  unsigned vote = __ballot_sync(mask, wants);
  if (wants) {
    unsigned lane = threadIdx.x & 31u;
    unsigned lane_mask_lt = (1u << lane) - 1u;
    int rank = __popc(vote & lane_mask_lt);
    int leader = __ffs((int)vote) - 1;

    int warp_offset = 0;
    if ((int)lane == leader) {
      warp_offset = atomicAdd(count, __popc(vote));
    }
    warp_offset = __shfl_sync(mask, warp_offset, leader);

    rows[warp_offset + rank] = (int)i;
  }
}
