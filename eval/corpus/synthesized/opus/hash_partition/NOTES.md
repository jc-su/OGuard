# hash_partition

Both kernels are complete and pass their tests.

Each block counts its rows in a shared-memory histogram of `p` bins and then
adds the nonzero bins to `counts` with global atomics. `run` and `test.cu`
launch at most 4 blocks per SM so that this flush happens a bounded number of
times; the threads loop over the rows.

The hash is computed on the unsigned 64-bit value of the key, so the product
wraps modulo 2^64. Since `h(x) < 2^32` and `p <= 1024`, the CUDA kernel takes
the remainder in 32-bit unsigned arithmetic.

In `signature.json`, `part` indexes into `counts` (its values are bins of
`counts`), and `counts` is `inout` because the kernel adds to it.
