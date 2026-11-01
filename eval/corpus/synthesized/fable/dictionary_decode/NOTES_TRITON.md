# dictionary_decode: Triton

Status: complete. `test_triton.py` passes for (n, d) in {(0, 0), (1, 1), (1000, 37), (1000003, 100000)}.

Approach: one row per lane, 1024 rows per program (4 warps): load the code, gather the dictionary value, store. `d` is accepted for the signature and not used by the kernel.

Time at n = 1000003, d = 100000: 30 us.
