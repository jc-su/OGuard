# Sonnet: kernels of HELDOUT4.md left out of the corpus

- `string_prefix`, Numba, CUDA C++, and Triton: the adapter cannot read the
  `signature.json`. The length of `out` is `n * k`, a product of two size
  names, which is neither a sum of size names and constants with constant
  factors nor a length such as `offsets[n]` that gives the buffer its own
  size symbol. `eval.corpus.sources` wrote no PTX for them
  (`out: a length 'n * k' the adapter does not read`).
- `transpose`, Numba, CUDA C++, and Triton: the adapter cannot read the
  `signature.json`. The lengths of `source` and `out` are `rows * cols`, a
  product of two size names, for the same reason. `eval.corpus.sources`
  wrote no PTX for them
  (`source: a length 'rows * cols' the adapter does not read`).

The roles of both tasks are in `roles.tsv`; the kernels have no rows in
`verdicts.tsv`.
