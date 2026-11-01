# string_equal (Triton)

- Each program compares ROWS rows as a [ROWS, CHUNK] byte tile and loops up to
  the longest row among the rows whose two lengths are equal. Rows with
  different lengths load no bytes, and a row stops loading bytes after its
  first mismatching chunk.
- The tile shape trades short strings against long ones. Time on GPU 1
  (RTX A6000) for 2^21 rows with lengths uniform in [0, L) and about a quarter
  of the rows differing in one byte:

  | ROWS, CHUNK, warps | L = 16 | L = 40 | L = 200 |
  |--------------------|--------|--------|---------|
  | 128, 8, 4 (chosen) | 103 us | 205 us | 1679 us |
  | 128, 16, 4         | 152 us | 316 us | 1297 us |
  | 64, 32, 4          | 261 us | 387 us | 1096 us |
  | 512, 8, 8          | 129 us | 228 us | 1440 us |

  The chosen shape favors the short strings that are common in query columns;
  on strings of about 100 bytes it is about 1.5 times slower than the best
  shape for them.
