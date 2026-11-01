# sorted_contains: Triton

Status: complete. `test_triton.py` passes for (n, s) in {(0, 0), (0, 5), (1, 1), (1, 0), (1000, 500), (1000003, 300000), (1000003, 1)} with about half the keys drawn from the members, plus the int64 limits as members and keys.

Approach: one key per lane, 512 keys per program (8 warps). Each lane finds j = |{t : members[t] < key}| by binary lifting from the largest power of two not above s (uniform steps, masked gathers, INT64_MAX for probes past the end), then tests members[j] == key. The first search levels touch few cache lines and stay in L1; the lower levels are L2 gathers, which is why smaller programs with more resident warps won the sweep (512/8 warps: 141 us, 1024/4 warps: 166 us).

Time at n = 1000003, s = 300000: 142 us.
