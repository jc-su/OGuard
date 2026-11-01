# window_max (Triton)

The kernel is complete and passes `test_triton.py`.

## Algorithm

The kernel uses the van Herk / Gil-Werman algorithm, so the work per row does
not depend on `w`. Rows are split into segments of `w` rows starting at
multiples of `w`. For row `i` and `j = i - w + 1`, the window `[j, i]` is the
end of the segment of `j` followed by the start of the segment of `i`, so
`out[i]` is the maximum of the suffix maximum at `j` and the prefix maximum at
`i`. Each program computes both with segmented `tl.associative_scan` calls
(the suffix with `reverse=True`) on lanes indexed by the output row, and adds
the parts of the segments that lie outside its tile with two reductions over
`HALO` lanes.

`HALO` is 1024, the largest `w` in the task. `run` raises `ValueError` for `w`
outside `[1, 1024]`, since a larger window would need a larger halo.

With `reverse=True`, Triton calls the combine function with the accumulated
value from the right as the first argument and the new element as the second,
so one combine function serves both scan directions. The test checks this
through windows that cross segment and tile boundaries.

Measured on the RTX A6000 at n = 1000003: about 36 us for every `w` from 1 to
1024.
