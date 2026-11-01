# scatter_add (Triton)

- One float64 atomic add per row. For rows with the same `idx`, the order of the adds varies between runs, so the result can differ in the last bits. The test compares against `np.bincount` with weights, with a tolerance of 1e-12 times the sum of absolute values per output row.
