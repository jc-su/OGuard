# bloom_probe (Triton)

- Same hash arithmetic as bloom_build: uint64 bitcast, wrapping multiply by
  uint64 constexprs, logical `>> 32`, and `& (32 * words - 1)` with the mask
  computed in int64 via `tl.cast(words, tl.int64)`.
- Each row loads its two filter words with a gather and tests the two bits;
  `match` is the AND of the two tests cast to int8.
