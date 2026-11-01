# Opus: kernels of HELDOUT5.md left out of the corpus

- `run_end_encode`, `segmented_cummax`, `range_window_sum`, and
  `fill_forward`, Numba: the source does not compile in the review's
  environment, where the GPUs are hidden (`CUDA_VISIBLE_DEVICES=`). Each
  kernel is a cooperative launch that calls `cuda.cg.this_grid()`, and
  numba-cuda compiles that overload as a device function, which asks the
  driver for the current device's compute capability: `cuInit` fails with
  `CUDA_ERROR_NO_DEVICE`, and `eval.corpus.sources` wrote no PTX for them.
  The grid barrier lowers to calls of `cudaCGGetIntrinsicHandle` and
  `cudaCGSynchronize`, functions of CUDA's device runtime library that the
  kernel's PTX would declare but not define.

The roles of the four tasks are in `roles.tsv`, and their CUDA C++ and Triton
kernels are in the corpus. `range_window_sum/launch.json` states the grid
claim of the Numba kernel, at most 1024 blocks, the size of its shared array
of block prefix sums; the claim applies if the kernel is compiled later.
