# bloom_build notes

- `bits` has kind `inout` in `signature.json`: the kernel sets bits with
  atomic OR, so its result depends on the contents of `bits` on entry, which
  the task requires to be zero.
- Both kernels compute `h % (32 * words)` as `h & (32 * words - 1)`, which
  requires `words` to be a power of two, as the task states. `signature.json`
  records this in the range of `words`.
- The tests run each size with `words` of 1, 1024, and 2^26 and compare the
  whole filter with the reference.
