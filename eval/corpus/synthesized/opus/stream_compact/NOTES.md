# stream_compact

Both kernels are complete and pass their tests.

Each warp takes a ballot of its kept rows and lane 0 reserves the output slots
of the whole warp with one `atomicAdd` on `count[0]`. A kept lane writes to the
reserved base plus the number of kept lanes below it. The order of the warps in
`out` depends on the order of the atomics, so it changes from run to run, which
the task allows. The tests sort the first `count[0]` values before comparing
them and check that `out[count[0]:]` is not written.

`count` is marked `inout` in `signature.json` because the kernel adds to it,
so its value on entry (zero) is part of the contract.

The loop condition `i - lane < n` is the same in all 32 lanes of a warp, which
keeps the full-mask warp intrinsics valid in the last, partial warp. This needs
a block size that is a multiple of 32; `run` and `test.cu` use 256.
