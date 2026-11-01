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
  // `valid` holds (n + 7) / 8 bytes (signature.json's declared length for
  // this argument). Guard the byte index against that exact expression,
  // in addition to guarding the row index against n: `i < n` alone implies
  // `i / 8 < (n + 7) / 8` mathematically, but a static prover that treats
  // integer division opaquely cannot chain that implication on its own, so
  // it cannot tie the load's address back to the declared buffer length
  // through `i < n` alone. Spelling the byte-index guard out explicitly,
  // using the same "(n + 7) / 8" shape as the declared length, gives it a
  // bound it can match directly.
  long long valid_len = (n + 7) / 8;

  bool wants = false;
  if (i >= 0 && i < n) {
    long long byte_idx = i / 8;
    if (byte_idx >= 0 && byte_idx < valid_len) {
      unsigned char byte = valid[byte_idx];
      wants = (byte >> (i % 8)) & 1u;
    }
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
