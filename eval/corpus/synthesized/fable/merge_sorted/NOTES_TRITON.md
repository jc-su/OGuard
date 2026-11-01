# merge_sorted: Triton

Status: complete. `test_triton.py` passes for m + p in {0, 1, 1000, 1000003} with splits (n, 0), (0, n), (n/2, n/2), (n/10, 9n/10), values that are mostly distinct, values in [0, 50) with many ties, disjoint ranges in both orders, and 1000 + 1000003 equal values.

Approach: merge path. A program owns 1024 output positions starting at diagonal d0. It finds the partition at d0 (the number of a-elements among the first d0 outputs) with a k-ary search: all 1024 lanes probe evenly spaced positions of the candidate interval, a block reduction counts the probes where a[i] <= b[d0 - i - 1], and the interval shrinks by a factor of 1024 per round (two rounds for a million rows), so the partition costs two dependent memory rounds instead of twenty. Then every lane finds its own partition inside the program's window of a and b with an 11-step binary search of masked gathers and emits min(a[i], b[j]), a first on ties. All searches use the same predicate, which is what makes the merge stable. The scalar tail beyond m + p is masked.

Time at 500001 + 500002 rows: 63 us.
