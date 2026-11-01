# Triton versions of the GPU query operator tasks

You wrote each task of [TASKS.md](TASKS.md) as a Numba kernel (`numba.py`) and a CUDA C++ kernel (`kernel.cu`). Write a third version of each task in Triton (Triton 3.8, PyTorch 2.14, CUDA 13.0, RTX A6000, sm_86), in the same task directory. Write each kernel the way you would for a production engine: correct on every input size, including 0 and sizes that are not a multiple of the block size, and fast.

- `triton.py`: a Triton kernel `@triton.jit def kernel(...)` whose arguments are those of your Numba kernel, in the same order and with the same names (your `signature.json` lists them), followed by any `tl.constexpr` arguments, such as a block size. Define at module level:
  - `TYPES`: a dict from each argument that is not a `tl.constexpr` to its Triton type: `"*fp64"`, `"*fp32"`, `"*i64"`, `"*i32"`, `"*i8"`, `"*u8"`, or `"*u32"` for a pointer to that element type, and `"i64"`, `"i32"`, or `"fp64"` for a scalar.
  - `CONSTANTS`: a dict from each `tl.constexpr` argument to its value.
  - `NUM_WARPS`: the number of warps of each program.
  - `run(...)`: launches `kernel[grid](...)` with `num_warps=NUM_WARPS` on PyTorch CUDA tensors, with the arguments of your Numba `run`, in the same order.
- `test_triton.py`: a test of `run` on the GPU against a NumPy reference, with n in {0, 1, 1000, 1000003}.

Record in `NOTES.md` any task you could not complete in Triton and why.
