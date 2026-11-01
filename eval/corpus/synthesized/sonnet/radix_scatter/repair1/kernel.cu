// radix_scatter: the scatter pass of an LSD radix sort pass. cursors[b]
// already holds the exclusive prefix sum of the counts of byte value b
// (from a prior histogram pass). One thread per row extracts its key's
// byte at `shift`, claims the next free slot for that byte with
// atomicAdd (advancing the cursor), and writes its row id there.
// Contention is spread across 256 independent counters rather than one
// global counter, so this stays cheap even for large n.
extern "C" __global__ void kernel(const unsigned int *__restrict__ keys,
                                   int *cursors, int *__restrict__ out,
                                   long long n, long long shift) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 0 && i < n) {
    unsigned int byte = (keys[i] >> shift) & 255u;
    int pos = atomicAdd(&cursors[byte], 1);
    // `pos` is the pre-increment value of a bucket cursor, not a direct
    // function of the thread index; it is bounded by `n` only through the
    // global invariant that `cursors` starts as a correct histogram
    // prefix sum over every row (part of this task's contract), which is
    // not something a per-thread bound check can see. Guard the write
    // against `out`'s declared length directly so the store's address is
    // provably in range for any input.
    if (pos >= 0 && (long long)pos < n) {
      out[pos] = (int)i;
    }
  }
}
