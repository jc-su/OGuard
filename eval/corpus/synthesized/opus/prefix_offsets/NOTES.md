# prefix_offsets

Both kernels are complete and pass their tests.

The signature has no scratch buffer, so a multi-block single-pass scan
(decoupled look-back, which needs per-block status words and a block counter
that start at zero) cannot be written against it. The kernel therefore scans
with one block of 256 threads: block 0 walks the column in tiles of 4096
values, scans each tile in shared memory, and carries the tile total to the
next tile. Any other blocks of the grid return at once, so the result does not
depend on the grid size. `run` and `test.cu` launch one block.

Each tile is loaded with coalesced reads into padded shared memory, each thread
scans 16 consecutive values in registers, and the loads of the next tile are
issued before the current tile is scanned. Measured on GPU 1 (RTX A6000) for
n = 1000003: 0.31 ms for the CUDA kernel and 0.42 ms for the Numba kernel. A
multi-block scan with a scratch buffer would be faster for large n.

The task does not bound the sum of `lengths`. When an offset exceeds 2^31 - 1,
both kernels store it modulo 2^32, as an int32 cumulative sum would. The CUDA
kernel sums in uint32 so that this wrap is defined behavior.
