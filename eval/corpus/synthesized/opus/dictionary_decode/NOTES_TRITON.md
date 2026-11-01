# dictionary_decode (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

One lane per row, 2048 rows per program: load the code, gather the float64
dictionary entry, store it. The codes and the output are streamed with
evict-first, which keeps the L2 cache for the dictionary; an evict-last hint on
the dictionary itself was measured slower (up to 1.4 times at 400 000 entries).
Codes are in [0, d) by contract, so `d` is not read by the kernel.

## Tests

n in {0, 1, 1000, 1000003} with d in {1, 257, 100 000}, compared bit for bit
with `dictionary[codes]`. The test also compiles the kernel with the declared
`TYPES`.

## Performance (RTX A6000, n = 10^8)

d = 1000: 1.76 ms, 683 GB/s (same as a device copy of the traffic). d = 10^6
(8 MB, larger than L2): 5.2 ms, bound by the random gather
(`torch.index_select`: 5.4 ms).
