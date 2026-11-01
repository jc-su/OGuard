# string_equal (Triton)

- One row per thread: BLOCK=256 rows per program with 8 warps. Each program
  loads both offset pairs, marks rows with equal lengths, and loops `j` from 0
  to the longest such string in the program, loading byte `j` of both strings
  under the mask `j < length`. Rows that already differ keep loading until the
  loop ends; the loop bound is a scalar tensor (`tl.max` of the lengths), which
  Triton lowers to a dynamic `scf.for`.
- Benchmark at n = 1000003, lengths 0 to 24, on the RTX A6000: BLOCK=256 with
  4 or 8 warps runs in about 60 us; BLOCK=512 with 4 warps (4 rows per thread)
  takes 240 us and BLOCK=1024 with 4 warps takes 570 us. More rows per thread
  serializes the byte loop, so the mapping stays at one row per thread.
