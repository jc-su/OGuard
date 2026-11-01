# radix_scatter: Triton

Status: complete. `test_triton.py` passes for n in {0, 1, 1000, 1000003} with uniform keys at shifts 0, 8, 16, 24, all keys equal, three distinct digits, and digit 255; it checks that `out` is a permutation of the row ids grouped by digit in cursor order and that the cursors end at their segment ends.

Approach: block-level ranking so the 256 cursors see one atomic add per (program, non-empty digit) instead of one per key. A program owns 2048 keys: `tl.histogram` of the digits with an exclusive `tl.cumsum` gives each digit's local segment start; one masked relaxed `atomic_add` per digit reserves the program's segments in `out` and returns their global starts; the keys are sorted by (digit, local index) packed into one int32 with `tl.sort`, which groups equal digits, and the output position of the key at sorted index j is j + (global start - local start) of its digit, gathered per key from the 256-entry table with `tl.gather`. Rows past n get an INT32_MAX sentinel that sorts last and is never stored; its digit field is clamped so the gather stays in the table (the first version did not clamp it and hit an illegal address).

Time at n = 1000003: 45 us with uniform keys, 125 us with all keys equal. Plain per-key atomics were 419 us and 1273 us.
