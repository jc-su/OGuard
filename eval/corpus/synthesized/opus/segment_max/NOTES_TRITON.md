# segment_max (Triton)

- The grid has one program per segment (g programs). Each program reads its segment in chunks of BLOCK = 256 rows with 2 warps, so an empty segment stores the smallest int64 value without any loads.
- A segment of any length is correct. A very long segment is reduced by a single program, so a column with a few huge segments uses only a small part of the GPU.
- Tests: `python -P test_triton.py` with g in {0, 1, 1000, 1000003}. Segment lengths are 0 to 8 with about 20% empty, and one segment of 5000 rows to exercise the chunk loop. Values span the full int64 range.
