# Notes

## Grid claims

Recorded 2026-09-29: a blind review under the grid claim of design decision 33
(`"grid": [x, y, z]` in `triton.json` or `launch.json`), made before the gate
decided the kernels. Sonnet's development Triton kernels compute the program
offset `pid * BLOCK` in 32 bits (`shl.b32` of `%ctaid.x`); its held-out
kernels use `tl.program_id(0).to(tl.int64)`; Numba and CUDA C++ use 64 bits.

- Kernels given a claim: 21. Triton, in `triton.json`, at 2^31 / BLOCK
  programs for the `constexpr` that multiplies `pid`: [8388608, 1, 1] (256)
  for `days_to_year`, `dictionary_decode`, `min_max`, `rolling_sum`,
  `radix_scatter`, and `valid_rows`; [2097152, 1, 1] (1024) for `gather`,
  `projection`, `q6_sum`, `filter_select`, and `scatter_by_offsets`;
  [16777216, 1, 1] (128) for `groupby_sum`, `hash_build`, `hash_probe`,
  `starts_with`, and `string_length`; [524288, 1, 1] (4096) for `histogram`.
  Numba and CUDA C++, in `launch.json`: `radix_scatter` and `valid_rows` with
  [2147483647, 1, 1], whose C witness was a grid with a second row of blocks
  in y.
- Verdicts changed from C to safe: 12, all from `decisions.tsv`, appended to
  `verdicts.tsv` for the maintainer to carry into the table: `days_to_year`,
  `dictionary_decode`, `gather`, `groupby_sum`, `hash_build`, `histogram`,
  `min_max`, `projection`, `q6_sum`, `rolling_sum`, `starts_with`, and
  `string_length` (Triton). The checker limits that `classification.md`
  records for some of them (PL, WS, SB, MT, PK, FL, DL) are unchanged.
- Verdicts changed from C to E: 9, all from `decisions.tsv`, appended to
  `verdicts.tsv`: `filter_select` Triton (`sel[pos]` with pos from `count` on
  entry), `hash_probe` Triton (`out_build[pos]` and `out_probe[pos]` under
  pos < out_cap, but pos is negative when `out_count` is negative on entry),
  `scatter_by_offsets` Triton (`out[offsets[i]]` with offsets[i] = total
  inside the range [0, total] of an offsets buffer), and `radix_scatter` and
  `valid_rows` in Numba, CUDA C++, and Triton (`out[pos]` or `rows[pos]` with
  the position from `cursors` or `count` on entry; the `valid_rows` kernels
  also read `valid[i >> 3]` under i < n, and the bitmap's length is a size
  symbol of its own).
- Left C with a grid claim pending a symbolic bound: `string_equal` Triton
  (`verdicts.tsv`; one program per row with no `r < n` test, so the bound is n
  programs); `tile_scan` (Numba, CUDA C++, Triton; `totals[blockIdx.x]` with no
  bound, (n + 1023) / 1024 blocks); from `decisions.tsv`, `segmented_sum`
  Triton (one program per segment with no `pid < g` test, g programs; also
  `pos += 1024` in int32 wraps for a segment ending at 2^31 - 1023 or later,
  which no grid bounds) and `string_contains` Triton (one program per string
  with no `row < n` test, n programs).
- Stay C without a claim: `segmented_sum` CUDA C++ (an int32 `idx += 32`
  inside a segment).
