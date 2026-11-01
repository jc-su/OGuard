# bloom_build (Triton)

- Keys are bitcast to uint64 and multiplied by the two constants as uint64
  constexprs, so the product wraps as the task specifies; `>> 32` on an
  unsigned tensor is a logical shift.
- `words` is a power of two, so `h % (32 * words)` is `h & (32 * words - 1)`.
  The mask is computed as int64 (`tl.cast(words, tl.int64) * 32 - 1`) because
  `32 * words` is 2^31 at the upper limit, which overflows int32. `tl.cast` is
  used instead of `.to` because Triton specializes a scalar argument equal to 1
  into a plain constexpr int, which has no `.to` method.
- Bits are set with `tl.atomic_or` under `sem="relaxed"`; the ordering of the
  ORs does not matter for the result.
