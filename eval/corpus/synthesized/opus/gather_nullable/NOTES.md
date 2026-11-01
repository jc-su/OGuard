# gather_nullable notes

- The kernels trust the task's contract that every `idx[i]` is -1 or a row id
  of `col`, and read `col[idx[i]]` for every `idx[i] >= 0`. They do not use
  `n`; `signature.json` records the contract as `idx` indexing into `col`,
  whose length is `n`.
- The NaN of an unmatched row has the bit pattern `0x7ff8000000000000`, the
  quiet NaN that NumPy and Arrow write, so the tests compare the output bytes
  with the reference directly.
