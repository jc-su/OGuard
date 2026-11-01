# Haiku: kernels of HELDOUT2.md left out of the corpus

- `lag`, Triton: the kernel's arguments differ from its `signature.json`.
  `kernel(values, first, out, n, k, BLOCK_SIZE: tl.constexpr)` takes
  `BLOCK_SIZE` as a `tl.constexpr`, but the module's `CONSTANTS` is empty
  (`run` passes `BLOCK_SIZE=256` at the launch), so the kernel's arguments
  are `(values, first, out, n, k, BLOCK_SIZE)`, not the signature's
  `(values, first, out, n, k)`. `eval.corpus.sources triton` wrote no PTX
  for it.
