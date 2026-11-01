# bloom_build

- `bits` is declared `inout` in `signature.json`: the kernel ORs bits into it
  atomically, so the result depends on the zeroed contents on entry, and words
  that no key hashes to keep their entry value.
- `words` is a power of two, so `p % (32 * words)` is `p & (32 * words - 1)`.
  With `words = 2^26` the bit index reaches `2^31 - 1`, so the index math is
  done in unsigned 64-bit and the word index in 64-bit.
