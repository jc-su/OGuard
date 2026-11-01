# string_contains: Triton

Status: complete. `test_triton.py` passes for n in {0, 1, 1000, 1000003} with patterns of length 0, 1, 2, 10, 21 (including a byte that never occurs), strings up to 400 bytes with patterns up to 40 bytes, and a column of empty strings.

Approach: lanes over (string, start position) pairs. A program owns 64 strings and tests 8 candidate positions of each at once as a [64, 8] tile, walking positions in chunks of 8 up to the longest string of the program. For each pattern byte q the tile loads `chars[begin + p + q]`; along the position axis those are consecutive bytes of one string, so the loads coalesce, and the per-string result is a max over the position axis. Lanes whose string is too short for the position, or whose string already matched, load nothing. The empty pattern stores 1 for every row.

A first version with one string per lane and the pattern bytes as the second tile axis was 4 to 6 times slower (about 620 us) because each lane walked its string serially with uncoalesced byte loads.

Time at n = 1000003 (lengths 0 to 24): 102 us for a 2-byte pattern, 132 us for a 10-byte pattern.
