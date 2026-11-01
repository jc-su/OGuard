# block_sort notes

- One block of 256 threads sorts one tile of 1024 keys with a bitonic sorting
  network. Thread t holds rows 4t to 4t + 3 in registers. The stages with
  stride 1 and 2 compare registers of one thread, the stages with stride 4 to
  64 exchange keys between the threads of a warp with `shfl_xor`, and the
  stages with stride 128 to 512 compare keys in shared memory between block
  barriers. A first version that ran every stage in shared memory took 2.6
  times as long in CUDA C++.
- A short last tile is filled with `INT32_MAX` up to 1024 rows. The filler
  sorts after the real keys, and a filler row equals any real `INT32_MAX`
  key, so the first `count` rows of the sorted tile are the sorted keys. The
  tests include keys equal to `INT32_MIN` and `INT32_MAX`.
- `kernel.cu` moves each thread's four rows between registers and shared
  memory as one `int4`. `numba.py` stores row e at shared index
  e + e / 32, so that the four rows of each thread in a warp fall in distinct
  banks.
- compute-sanitizer memcheck, racecheck, synccheck, and initcheck report no
  errors for both versions on the test inputs.
- Measured on the RTX A6000 with 64M keys: `kernel.cu` 1.32 ms, `numba.py`
  2.57 ms. Numba does not unroll the loops over the stages, so the Numba
  kernel runs more index and branch instructions per stage.
