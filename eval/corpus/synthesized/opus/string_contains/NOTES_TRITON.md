# string_contains (Triton)

## Status

Complete. `test_triton.py` passed.

## Approach

Each program owns 64 strings, one lane per string, in two phases.

Lane phase: a lane walks its string one byte per step and keeps the last bytes
it read in a shift register, so a step is one byte load and a few integer
operations whatever the pattern length is. A match ends at byte p when the
register equals the last pattern bytes. The register is one uint32 for k <= 4,
two uint32 for k <= 8 and two uint64 above; patterns longer than 16 bytes
check their leading bytes with extra loads, only in steps where some lane of
the program matched the last 16 bytes. The walk stops after 256 bytes.

Long-string phase: every string of the tile longer than 256 bytes is scanned
by all lanes together, 256 consecutive start positions per step (coalesced),
for the matches that end at byte 256 or later. A position is a candidate when
the first and last pattern bytes match; the middle bytes are checked only when
the chunk has a candidate, and the scan stops at the first match. This keeps
one long string from holding all lanes of a program.

The empty pattern matches every string.

Other designs measured on the same data: character-parallel chunks (one lane
per byte, a scan to map matches to strings) were 1.1 to 2.6 times slower in 7
of 9 cases and 1.1 to 1.3 times faster for the 11-byte pattern on random and
skewed text; lane per string without the long-string phase was up to 7.6 times
slower on the skewed layout.

## Tests

n in {0, 1, 1000, 1000003}; short strings (0 to 11 bytes) and a skewed layout
(0 to 39 bytes, 0.1% of 2000 to 20 000 bytes); text over the alphabet `abc`
with pattern lengths 0, 1, 2, 4, 5, 8, 9, 16, 17, 18 and 300, and binary data
over bytes 0 to 2 with patterns of zero bytes (the register starts at zero).
Also: patterns planted at every start position around byte 256 in strings of
255 to 5000 bytes, each also cut one byte short, and matches that would cross
a string boundary. The test also compiles the kernel with the declared
`TYPES`.

## Performance (RTX A6000)

4 million TPC-H-like part names (131 MB): `green` 354 us, `e` 230 us,
`special requests` 444 us. 10 million random strings of 0 to 40 bytes
(200 MB): `a` 491 us, `abc` 674 us, `hello world` 1.0 ms. 5 million strings
of the skewed layout: `a` 337 us, `abc` 612 us, `hello world` 1.0 ms. The lane
phase is bound by L1 throughput (the lanes of a warp read 32 different
strings), so it runs at 180 to 660 GB/s rather than at the DRAM bandwidth.
