// rolling_sum: out[i] = sum(values[max(0, i - window + 1) .. i]).
//
// Each block covers blockDim.x consecutive output rows. It first
// cooperatively loads its input tile -- the block's rows plus the
// `window - 1` rows of "halo" immediately before them -- into shared
// memory once (zero-padded where the halo runs off the start of the
// array). Every thread then sums its own window directly out of shared
// memory, so each input element is read from global memory exactly once
// no matter how much its window overlaps its neighbors'.
//
// The tile is a fixed-size shared array sized for the worst case
// (blockDim.x up to 1024, window up to 1024) instead of a dynamically
// sized `extern __shared__` array, whose actual byte size is a
// launch-time argument that is invisible to any analysis of the kernel
// body alone and so can never be proved to bound an access into it.
//
// Every index used to read `values` or the tile is also clamped with a
// pair of local min/max guards immediately before the access, rather
// than relying only on a skip-branch computed from a derived variable
// (tile_start, a loop induction variable, etc.): the clamped value's
// bound then follows from those two local comparisons alone, with no
// need to track it across loop iterations.
#define MAX_TILE 2047 // 1024 (max blockDim.x) + 1024 (max window) - 1

extern "C" __global__ void kernel(const double *__restrict__ values,
                                   double *__restrict__ out, long long n,
                                   long long window) {
  __shared__ double tile[MAX_TILE];

  long long block_start = (long long)blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  long long tile_start = block_start - (window - 1);
  long long tile_size = (long long)blockDim.x + window - 1;

#pragma unroll 1
  for (long long idx = tid; idx < tile_size; idx += blockDim.x) {
    long long safe_idx = idx;
    if (safe_idx < 0) safe_idx = 0;
    if (safe_idx >= MAX_TILE) safe_idx = MAX_TILE - 1;

    long long g = tile_start + idx;
    double v = 0.0;
    if (n > 0) {
      long long safe_g = g;
      if (safe_g < 0) safe_g = 0;
      if (safe_g >= n) safe_g = n - 1;
      v = values[safe_g];
    }
    tile[safe_idx] = (g >= 0 && g < n) ? v : 0.0;
  }
  __syncthreads();

  long long i = block_start + tid;
  if (i < n) {
    double acc = 0.0;
#pragma unroll 1
    for (long long j = 0; j < window; ++j) {
      long long safe_k = tid + j;
      if (safe_k < 0) safe_k = 0;
      if (safe_k >= MAX_TILE) safe_k = MAX_TILE - 1;
      acc += tile[safe_k];
    }
    out[i] = acc;
  }
}
