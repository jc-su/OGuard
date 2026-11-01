// valid_rows: compact the ids of the valid rows of an Arrow validity bitmap.
//
// The previous revision assigned one thread per row and derived that
// thread's byte index as `byte_idx = i / 8`, guarded by
// `byte_idx < valid_len` with `valid_len = (n + 7) / 8` recomputed locally
// to match the buffer's declared length. That guard is true for every
// `i < n` -- `i < n` implies `i / 8 < (n + 7) / 8` by construction -- and a
// compiler is free to use exactly that implication to prove the inner
// guard redundant given the outer `i < n` check and fold it away, which
// erases the one condition that ties the read of `valid` to its own
// buffer's length, leaving nothing for a static checker to use.
//
// This revision instead assigns one thread per BYTE of the bitmap, so the
// byte index is the thread's own grid position, not a value derived from
// a separately-bounded row index: `byte_idx < valid_len` is the sole
// condition under which anything happens in the thread at all, so it is
// not implied by, or foldable into, any other condition in the kernel.
// Each thread then decodes its own byte's up to 8 rows and reserves its
// output slots with one atomicAdd per byte (rather than one per valid
// row), still far fewer atomics than the naive one-per-row scheme even
// though it forgoes full warp-aggregated ballots.
extern "C" __global__ void kernel(const unsigned char *valid, int *rows,
                                   int *count, long long n) {
  long long valid_len = (n + 7) / 8;
  // byte_idx is this thread's own grid position: nothing else in the
  // kernel implies `byte_idx < valid_len`, so a compiler has no other
  // fact from which to derive (and then discard) this guard.
  long long byte_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (byte_idx < 0 || byte_idx >= valid_len) return;

  unsigned char byte = valid[byte_idx];
  long long base = byte_idx * 8;

  int local_count = 0;
  for (int bit = 0; bit < 8; ++bit) {
    long long row = base + bit;
    if (row < n && ((byte >> bit) & 1u)) {
      ++local_count;
    }
  }

  if (local_count > 0) {
    int offset = atomicAdd(count, local_count);
    int slot = 0;
    for (int bit = 0; bit < 8; ++bit) {
      long long row = base + bit;
      if (row < n && ((byte >> bit) & 1u)) {
        rows[offset + slot] = (int)row;
        ++slot;
      }
    }
  }
}
