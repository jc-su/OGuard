# string_reverse (Triton)

- Each program handles BLOCK = 1024 bytes of `chars`. Two scalar binary searches over `offsets` find the strings that hold the program's first and last byte. Each byte then searches only between those two strings, which is a few steps when strings are short. The byte at position p of string [start, end) is written to `out[start + end - 1 - p]`.
- `run` sizes the grid from `chars.numel()`, which the task defines as `offsets[n]`, so the host does not read `offsets[n]` from the device. The kernel loads `offsets[n]` and masks bytes past it.
- Tests: `python -P test_triton.py` with n in {0, 1, 1000, 1000003}. Strings are 0 to 20 bytes with about 20% empty, plus one 5000-byte string.
