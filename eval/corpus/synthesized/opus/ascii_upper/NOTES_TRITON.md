# ascii_upper (Triton)

- The byte count `offsets[n]` is in device memory, so the kernel loads it and
  masks every byte position against it. `run` sizes the grid from
  `chars.numel()`, which is `offsets[n]` by the task's contract, so the
  launch needs no device-to-host copy.
- A mask against a runtime value that the compiler cannot prove to be a
  multiple of 16 makes it split the 16-byte vector accesses into single-byte
  accesses. The kernel therefore masks only the block that contains the end of
  the data and processes every other block without a mask. On 2^27 bytes
  (GPU 1, RTX A6000) this took the kernel from 937 us (287 GB/s) to 397 us
  (677 GB/s).
