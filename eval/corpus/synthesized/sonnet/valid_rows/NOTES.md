# valid_rows

Status: complete. Numba and CUDA C++ both tested on GPU 1 for n in
{0, 1, 1000, 1000003}, all pass.

Approach: one thread per row, testing its bit in the bitmap. Rows are
compacted into `rows` with warp-aggregated atomics: each warp does a
single `atomicAdd` on `count` covering all of its valid lanes (via
`__ballot_sync`/`popc` for the per-lane rank and `__shfl_sync` to
broadcast the warp's base offset from the elected leader lane), instead
of one atomicAdd per valid row. This cuts atomic traffic on the single
`count` counter by up to 32x versus a naive per-thread atomicAdd.

Caveats:
- `count` is required to be zero on entry; the kernel never zeroes it
  itself, matching the spec.
- For n = 0 the launch is skipped entirely (a 0-block CUDA launch is a
  legal no-op; Numba's launch config rejects a 0 grid dimension outright,
  so `run()` returns early when n <= 0).
