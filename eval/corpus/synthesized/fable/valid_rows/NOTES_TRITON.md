# valid_rows: Triton

Status: complete. `test_triton.py` passes for n in {0, 1, 1000, 1000003} with random, all-valid, all-null, and sparse bitmaps.

Approach: each program owns 2048 rows and loads its 256 bitmap bytes, expands them to one bit per row with a broadcast shift and a row-major reshape, reserves a segment of `rows` with one relaxed atomic add on `count`, and scatters its valid row ids at the offsets of a block-local exclusive prefix sum (`tl.cumsum`). Output order is by program, then by row within the program.

Time at n = 1000003 (random bitmap, RTX A6000): 9 us.
