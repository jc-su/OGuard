# sorted_contains (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

One lane per key runs a branchless binary search over `members`: the interval
length is the same in every lane, so the loop count ceil(log2(s)) is uniform
and each step is one gather and one select. The first steps read the same few
members in every lane and hit in cache. With s = 0 the kernel reads no member
and writes 0.

## Tests

n in {0, 1, 1000, 1000003} and s in {0, 1, 2, 7, 1000, 1000003}, with about
half of the keys drawn from the members, and the int64 minimum and maximum as
keys and possibly as members; compared with `np.isin`. The test also compiles
the kernel with the declared `TYPES`.

## Performance (RTX A6000, n = 2.5 * 10^7)

s = 1000: 0.34 ms. s = 10^5: 2.4 ms (`torch.searchsorted`: 2.37 ms). s = 10^6:
4.7 ms (`torch.searchsorted`: 5.4 ms). s = 10^7 (80 MB): 36 ms, bound by
random DRAM reads in the last levels of the search (`torch.searchsorted`:
31 ms). A faster search for sets larger than L2 would need a cache-friendly
copy of the set or sorted keys, which a single kernel over these arguments
cannot build.
