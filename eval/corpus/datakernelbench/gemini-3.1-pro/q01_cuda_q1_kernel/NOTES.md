# q01_cuda_q1_kernel

## Source

`logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/cuda/full/queries/q01/round_04.py`, copied as `plan.py`: `q1_kernel` in `_CUDA_SOURCE`, lines 12-92 of the plan file.

## Extraction edits

- `__global__ void q1_kernel(` is declared `extern "C" __global__ void kernel(`.
- The host code (the launcher and `PYBIND11_MODULE`) is left out, and so are the host includes (`<torch/extension.h>` and the C++ standard headers); the CUDA headers that nvcc includes declare the signed fixed-width integer types that the kernel uses.
- Every access, guard, bound, index, and type is as in the plan.

## Launch

`q1_launcher` launches `q1_kernel<<<blocks, block_size, smem_size>>>` with `block_size = 512`, `blocks = (n + 511) / 512`, `num_sets = 512 / 32 = 16`, reduced to `49152 / (num_groups * 6 * 8)` (at least 1) when 16 sets of `num_groups * 6` doubles exceed 48 KiB, and `smem_size = num_sets * num_groups * 6 * 8` bytes of dynamic shared memory.

`launch.json` states the block (512, 1, 1) and the dynamic shared memory 768 * num_groups bytes, with the size maximum `num_groups <= 64`.

The dynamic shared memory depends on `num_groups`. This corpus kernel is the launch with `num_groups <= 64`, where the plan keeps `num_sets = 16` and passes `768 * num_groups` bytes. The plan's TPC-H launch has `num_groups = 3 * 2 = 6` (l_returnflag A, N, R; l_linestatus F, O). With more than 64 groups the plan passes `num_sets = 49152 / (48 * num_groups)` sets, a shared memory that is not affine in `num_groups`; that launch is not in the corpus.

## Lengths and ranges

- `rf_ptr`, `ls_ptr`, `qty_ptr`, `price_ptr`, `disc_ptr`, `tax_ptr`, `shipdate_ptr`: columns of `lineitem`, `n_elements = qty.numel()` rows each.
- `sum_qty`, `sum_base_price`, `sum_disc_price`, `sum_charge`, `sum_disc`, `count_order`: `torch::zeros({num_groups})`, zeroed by the plan before the launch and accumulated with `atomicAdd`: readwrite.
- `num_groups = len(rf_cats) * num_ls`, the outputs' length (size), at most 64 in this configuration; `num_sets = 16` in this configuration; `num_ls = len(ls_cats)`, at most `num_groups`, so in [0, 64]; `shipdate_hi` is any date.
- `rf_ptr` and `ls_ptr` hold category codes in `[0, len(rf_cats))` and `[0, num_ls)`, but the kernel's buffers have other lengths, so they are `read`, and the relation `num_groups = len(rf_cats) * num_ls` is not stated.
