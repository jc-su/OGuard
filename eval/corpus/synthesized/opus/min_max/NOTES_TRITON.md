# min_max (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

A persistent grid of at most 4 programs per SM (8 warps each). Each program
walks the column with a grid stride, keeps 2048 running minima and maxima in
registers, reduces them once and merges its result with one float64 atomic
min on result[0] and one atomic max on result[1]. With n = 0 no program runs
and the result keeps +inf and -inf.

## Tests

n in {0, 1, 1000, 1000003} with normal values, all-negative values, positive
values and a constant column, compared exactly with NumPy (min and max are
exact). The test also compiles the kernel with the declared `TYPES`.

## Performance (RTX A6000, n = 10^8)

1.14 ms, 699 GB/s; `torch.aminmax` takes 1.69 ms.
