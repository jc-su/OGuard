# Opus: kernels of HELDOUT4.md left out of the corpus

- `string_prefix`, Numba, CUDA C++, and Triton: the corpus adapter cannot
  read the `signature.json`. It gives `out` the length `n * k`, a product of
  two sizes, which is not a sum of size names and constants with constant
  factors, nor a length such as `offsets[n]` or a quotient that the adapter
  turns into a size symbol of its own (`out: a length 'n * k' the adapter
  does not read`). `eval.corpus.sources` wrote no PTX for them.
- `transpose`, Numba, CUDA C++, and Triton: the corpus adapter cannot read
  the `signature.json`, which gives `source` and `out` the length
  `rows * cols` (`source: a length 'rows * cols' the adapter does not
  read`). `eval.corpus.sources` wrote no PTX for them.

The roles of both tasks are in `roles.tsv`, as for the other tasks.
