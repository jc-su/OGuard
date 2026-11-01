# Haiku: kernels of HELDOUT3.md left out of the corpus

- `lower_bound`, Numba, CUDA C++, and Triton: the kernels' arguments differ
  from the `signature.json`. All three name the second argument
  `sorted_arr`, so their arguments are `(keys, sorted_arr, out, n, s)`,
  while the task and the `signature.json` name it `sorted`:
  `(keys, sorted, out, n, s)`. `eval.corpus.sources` wrote no PTX for them.
