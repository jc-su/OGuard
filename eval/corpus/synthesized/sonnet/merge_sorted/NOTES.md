# merge_sorted

Status: complete. Numba and CUDA C++ both tested on GPU 1 for total
sizes m+p in {0, 1, 1000, 1000003}, split several ways (including
m = 0 or p = 0), plus a low-cardinality case to stress ties, all pass.

Approach: merge path / co-rank. For each output position k, a binary
search over the anti-diagonal finds the split (i, j) with i + j = k such
that the first k merged elements are exactly a[0:i) and b[0:j); each of
the m + p positions is then produced independently by one thread, with
no cross-thread synchronization. Total work is O((m + p) log(min(m, p))),
the standard way to parallelize a stable two-way merge on a GPU (the
result is a value sequence, so the a-before-equal-b tie-break has no
effect observable in `out` here, but the co-rank comparison still
implements it: `a[i-1] > b[j]` uses strict `>`, so equal values are
resolved toward taking more from `a`).
