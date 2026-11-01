# window_max

Both kernels are complete and pass their tests.

A block computes 1024 outputs at a time. It loads the `1024 + w - 1` values
that their windows cover into shared memory, with `INT64_MIN` for rows outside
`[0, n)`. Then `log2(K)` passes, where `K` is the largest power of two not
above `w`, turn each position into the maximum of the `K` values starting
there (each pass doubles the length, reading one buffer and writing the
other). The window of `w` values that ends at a row is the union of the `K`
values at its start and the `K` values at its end, so each output is the
maximum of two shared-memory values. This costs `O(log w)` work per value
instead of the `O(w)` of a direct loop. Shared memory is two buffers of 2048
int64 values (32 KiB), enough for `w = 1024`.
