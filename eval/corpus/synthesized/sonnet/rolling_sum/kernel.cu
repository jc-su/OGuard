// rolling_sum: out[i] = sum(values[max(0, i - window + 1) .. i]).
//
// Each block of 256 threads covers 256 consecutive output rows. It first
// cooperatively loads its input tile -- the block's 256 rows plus the
// `window - 1` rows of "halo" immediately before them -- into dynamic
// shared memory once (zero-padded where the halo runs off the start of
// the array). Every thread then sums its own window directly out of
// shared memory, so each input element is read from global memory
// exactly once no matter how much its window overlaps its neighbors'.
extern "C" __global__ void kernel(const double *__restrict__ values,
                                   double *__restrict__ out, long long n,
                                   long long window) {
  extern __shared__ double tile[];

  long long block_start = (long long)blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  long long tile_start = block_start - (window - 1);
  long long tile_size = (long long)blockDim.x + window - 1;

  for (long long idx = tid; idx < tile_size; idx += blockDim.x) {
    long long g = tile_start + idx;
    tile[idx] = (g >= 0 && g < n) ? values[g] : 0.0;
  }
  __syncthreads();

  long long i = block_start + tid;
  if (i < n) {
    double acc = 0.0;
    for (long long j = 0; j < window; ++j) {
      acc += tile[tid + j];
    }
    out[i] = acc;
  }
}
