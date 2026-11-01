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
  if (i < n) {
    unsigned int byte = (keys[i] >> shift) & 255u;
    int pos = atomicAdd(&cursors[byte], 1);
    out[pos] = (int)i;
  }
}
