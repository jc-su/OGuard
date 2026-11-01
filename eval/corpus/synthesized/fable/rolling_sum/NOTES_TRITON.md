# rolling_sum: Triton

Status: complete. `test_triton.py` passes for n in {0, 1, 1000, 1000003} with windows 1, 2, 7, 100, 1023, 1024, values in [-1, 1), values of magnitude 1e6, and all ones; the largest error against the NumPy reference is 7.5e-8 absolute on window sums of magnitude up to about 3e7 (relative 2e-15).

Approach: a program owns 3072 output rows and loads a tile of 4096 values that starts 1024 rows earlier (the halo for the largest window). With z[t] = x[t] - x[t - window] (zero for t < window), the inclusive scan of z at t is exactly the sum of the window ending at t, so one `tl.cumsum` per tile gives all outputs; every partial sum is itself a window sum, so no large prefix sums are subtracted. Rows before the first row read as zero, which gives the shorter windows at the start of the column. `run` rejects a window outside [1, 1024].

Time at n = 1000003, window = 1024: 46 us. A direct per-row loop over the window was 30 us for window 2 but 3.9 ms for window 1000.
