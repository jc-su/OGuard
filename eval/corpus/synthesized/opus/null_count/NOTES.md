# null_count

Both kernels are complete and pass their tests.

Each thread counts the zero bits of whole bytes of the bitmap. The last byte
is masked to the rows below `n`, so set or unset padding bits after row
`n - 1` do not count; the tests set those bits at random. The block sums its
threads with warp shuffles and shared memory and adds the sum to `count[0]`
with one 64-bit atomic. `run` and `test.cu` launch at most 4 blocks per SM
and the threads loop over the bytes.

`count` is marked `inout` in `signature.json` because the kernel adds to it.
The tests start it at 12345 to check that the kernel adds instead of
overwriting.
