# masked_sum (Triton)

- Each program reduces 4096 rows (8 warps) and adds its partial sum to `result[0]` with one float64 atomic add. The order of the atomic adds varies between runs, so the last bits of the sum can differ from run to run. The test compares against NumPy with a tolerance of 1e-12 times the sum of absolute values.
- Rows with `mask[i] == 0` are dropped with `tl.where`, so a NaN in a masked-out row does not reach the sum. The test places NaNs there.
