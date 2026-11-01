# merge_sorted (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

Merge path. Each program writes 2048 outputs [k0, k1). The split of a
diagonal k is the smallest i with a[i] > b[k - i - 1], which puts an element
of a before an equal element of b; both splits of the program are found
together by an 8-ary search. The tile then holds a[i0:i1] ascending followed
by b[j0:j1] descending, a bitonic sequence (the last tile is padded in the
middle with the largest int64), and a bitonic merge network of 11
compare-exchange stages, written with `tl.reshape` to a [2] * 11 view and
xor reductions, sorts it in registers. The tile is stored with contiguous,
coalesced writes.

## Tests

m and p each in {0, 1, 1000, 1000003} (all 16 pairs) with values in [-3, 3)
(long runs of equal keys across a and b) and in [-10^15, 10^15), with the
int64 minimum and maximum included; also two disjoint ranges in both orders.
The output is compared with `np.sort` of the concatenation. The test also
compiles the kernel with the declared `TYPES`.

## Performance (RTX A6000, m = p = 2.5 * 10^7)

1.30 ms, 616 GB/s of input plus output. The first version (a per-element
binary search in the other slice and 256-ary split search) took 9.4 ms.
