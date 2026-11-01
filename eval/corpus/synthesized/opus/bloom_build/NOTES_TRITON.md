# bloom_build (Triton)

- The Triton launcher passes a Python int scalar as i32 when it fits, while
  `TYPES` declares the scalars as i64. With `words = 2^26`, the product
  `32 * words` is 2^31 and overflows i32. The kernel does not form it: since
  `32 * words` is a power of two, bit `h % (32 * words)` is bit `h & 31` of
  word `(h >> 5) & (words - 1)`, which is correct for both scalar widths. The
  test covers `words = 2^26`.
- The bits are set with relaxed atomic OR, since only the atomicity of each OR
  is needed. Time on GPU 1 (RTX A6000) for 2^25 keys: 2.0 ms with a 4 MB
  filter (2^20 words), 19.4 ms with a 64 MB filter (2^24 words), where each
  atomic goes to a random DRAM sector outside the L2 cache.
