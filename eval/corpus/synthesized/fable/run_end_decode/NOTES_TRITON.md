# run_end_decode: Triton

Status: complete. `test_triton.py` passes for n in {0, 1, 1000, 1000003} with r = 1, r = n, and intermediate run counts.

Approach: one output row per lane, 1024 rows per program. Each lane finds its run j = |{t : run_ends[t] <= i}| by binary lifting from the largest power of two not above r; the step sequence is the same for all lanes, so the loop is uniform and every probe is a masked gather. The rows of one program map to a contiguous run range, so the probes stay in cache. The value is then one gather from `run_values`.

Time at n = 1000003, r = 200000: 30 us (r = 1000003: about 80 us).
