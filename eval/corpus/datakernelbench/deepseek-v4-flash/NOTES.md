# deepseek-v4-flash

The corpus kernels of the DataKernelBench family `deepseek-ai__deepseek-v4-flash`, extracted and reviewed by the protocol
of `../README.md` from DataKernelBench at commit `43c361d`.

## Plans taken

For each toolchain and query, the `full` plan `round_<k>.py` with `k` the `output_round` of the query's
`results.json`, for the queries whose `results.json` has `"correct": true`:

| Toolchain | Query | Plan | Kernels launched |
| --- | --- | --- | --- |
| cuda | q01 | `round_09.py` | `q1_per_group_kernel` |
| cuda | q02 | `round_01.py` | none |
| cuda | q05 | `round_02.py` | `q5_scatter_kernel` |
| cuda | q06 | `round_01.py` | `q6_kernel` |
| cuda | q07 | `round_01.py` | `q7_kernel` |
| cuda | q08 | `round_01.py` | `q8_volume_kernel`, `q8_year_kernel` |
| cuda | q09 | `round_05.py` | none |
| cuda | q11 | `round_03.py` | `q11_agg_kernel` |
| cuda | q12 | `round_05.py` | `q12_kernel` |
| cuda | q13 | `round_02.py` | `q13_gather_kernel`, `q13_groupby_kernel`, `q13_histogram_kernel` |
| cuda | q14 | `round_01.py` | `promo_revenue_kernel` |
| cuda | q15 | `round_04.py` | `q15_kernel` |
| cuda | q16 | `round_06.py` | `build_composite_keys_kernel`, `filter_partsupp_kernel` |
| cuda | q17 | `round_01.py` | `compute_part_stats_kernel`, `q17_kernel` |
| cuda | q18 | `round_09.py` | none |
| cuda | q19 | `round_02.py` | `q19_kernel` |
| cuda | q21 | `round_09.py` | `q21_full_kernel` |
| cuda | q22 | `round_03.py` | `q22_avg_kernel`, `q22_filter_kernel` |
| triton | q02 | `round_04.py` | none |
| triton | q03 | `round_07.py` | `_q3_kernel` |
| triton | q04 | `round_04.py` | none |
| triton | q05 | `round_02.py` | `_q5_kernel` (8 configurations) |
| triton | q06 | `round_01.py` | `_q6_kernel` (8 configurations) |
| triton | q07 | `round_02.py` | `_q7_kernel` (8 configurations) |
| triton | q08 | `round_02.py` | `_q8_kernel_fused` (8 configurations) |
| triton | q09 | `round_02.py` | `_compute_amount_kernel` |
| triton | q10 | `round_04.py` | none |
| triton | q11 | `round_04.py` | `_q11_global_total_kernel` |
| triton | q12 | `round_02.py` | `_q12_kernel` (8 configurations) |
| triton | q13 | `round_07.py` | none |
| triton | q14 | `round_02.py` | `_promo_kernel` (8 configurations) |
| triton | q16 | `round_01.py` | none |
| triton | q17 | `round_05.py` | none |
| triton | q19 | `round_02.py` | `_q19_kernel` (8 configurations) |
| triton | q20 | `round_08.py` | none |
| triton | q21 | `round_03.py` | `_q21_mask_kernel` |
| triton | q22 | `round_07.py` | `_q22_avg_kernel` |

That is 37 plans. The plans of the other queries failed DataKernelBench's correctness check: cuda q03, q04, q10, q20; triton q01, q15, q18.

## Kernels

82 kernels are extracted: 21 CUDA kernels and 61 Triton kernels. Seven Triton kernels
run through an `@triton.autotune` decorator with 16 configurations (`BLOCK_SIZE` 1024, 2048, 4096, 8192;
`num_warps` 4 and 8; `num_stages` 2 and 4). A Triton corpus kernel states its `constexpr` values and its
warps, and none of these kernels has a loop that Triton pipelines, so `num_stages` does not change the
compiled code (the front end's PTX is the same for `num_stages` 2, 4, and Triton's default); each such
kernel therefore gives eight corpus kernels, one per `BLOCK_SIZE` and `num_warps`, in the autotuner's
order, the first without a suffix and the others `_2` to `_8`.

No launched kernel is left out: every one compiles through the corpus front ends within the README's edits,
and its binding builds.

The plans also compile or define kernels that they never launch; these are not corpus kernels:

- cuda q02: `q2_topk_kernel` is compiled, but `run_query` never calls `_launch_q2`, its only launcher.
- cuda q08: `q8_kernel` is compiled but never launched; `q8_launcher` launches `q8_volume_kernel` and `q8_year_kernel`.
- cuda q09: `q9_amount_kernel` is compiled, but `run_query` never calls `q9_amount_launcher`; it computes the amount with cuDF.
- cuda q16: `hash_group_counts_kernel` is compiled, but `run_query` never calls `hash_counts`, its launcher.
- cuda q18: `noop_kernel` is compiled, but `run_query` never calls `verify_cuda`, its launcher.
- triton q02: `_min_cost_kernel` is defined, but `run_query` never calls `_launch_min_cost`.
- triton q04: `_q4_lineitem_filter_kernel` is defined but never launched; `run_query` uses PyTorch operations.
- triton q10: `_q10_kernel` is defined but never launched; `run_query` uses `scatter_add_`.
- triton q13: `_q13_histogram_kernel` is defined but never launched; `_query_core` uses `torch.bincount`.
- triton q16: `_q16_filter_join_agg_kernel`, `_count_distinct_kernel`, and `_count_distinct_groups_kernel` are defined but never launched; `_query_core` uses `torch.unique`.
- triton q17: `_q17_fused_kernel` is defined but never launched; `run_query` uses PyTorch operations.
- triton q20: `_q20_sum_for_key` is defined but never launched; `run_query` uses `scatter_add_`.
- triton q22: `_q22_filter_kernel` is defined but never launched; only `_q22_avg_kernel` is.

## Review

`verdicts.tsv` classifies each kernel against its plan (C, E, or safe, as the README defines them), from the
source and the plan alone, without the gate's decisions. Before the grid claims
below: C: 65, E: 4, safe: 13; after them: C: 12, E: 13, safe: 57.

## Grid claims

Reviewed blind, before the gate decided the kernels, after the claim language
gained `grid`: the largest grid of any launch, per dimension. All 65 `C`
kernels needed a large grid for the access outside the buffer, and all 65
got a claim, in `triton.json` (`"grid": [2^31 / BLOCK_SIZE, 1, 1]`) or in
`launch.json`:

- 61 Triton kernels, whose 32-bit `pid * BLOCK_SIZE + tl.arange(0,
  BLOCK_SIZE)` wraps once the grid exceeds `2^31 / BLOCK_SIZE` programs:
  131072 programs for `BLOCK_SIZE = 16384` (q03), 2097152 for 1024, 1048576
  for 2048, 524288 for 4096, and 262144 for 8192.
- `q11_cuda_q11_agg_kernel`: 4194304 blocks in x (`2^32 / 1024`, where the
  unsigned 32-bit row id wraps) and 1 in y and z.
- `q16_cuda_build_composite_keys_kernel`, `q16_cuda_filter_partsupp_kernel`,
  and `q22_cuda_q22_filter_kernel`: 1 block in y and z with x up to the GPU's
  limit; their 64-bit row ids need no bound in x, and the claim rules out the
  second block in y that ran every row twice.

Verdicts changed from `C` to `safe` for 44 kernels: the Triton kernels of
q03 (1), q06 (8), q07 (8), q09 (1), q11 (1), q12 (8), q14 (8), q19 (8), and
q22 (1), whose every access is at the masked offsets or at a constant index
of a one-, two-, or eight-element output.

Verdicts changed from `C` to `E` for 9 kernels, whose remaining access
needs a relation that the gate cannot establish: the 8 q08 Triton kernels
(the atomics at `year - year_offset`, inside `num_groups` elements only for
the plan's two contiguous years) and `q21_triton__q21_mask_kernel` (the
loads at the searchsorted index, which its offsets role allows to equal
`n_lo`; the plan's joins keep every l1 order key inside `agg`).

12 kernels stay `C` with a claim, for a defect that does not depend on the
grid: the 8 q05 Triton kernels add to `out_ptr + i` for `i` in 0..24 from
every program, past the 5 elements of the plan's own launch; the four CUDA
kernels above take their write positions from an inout counter
(`key_count_out`, `out_count`, `out_n`) whose contents on entry the gate
does not establish, and `q11_cuda_q11_agg_kernel` also hashes a negative
partkey to an unsigned bucket index past its shared arrays.

No kernel needs a grid bound that depends on a size symbol.

## Core plans

The held-out set: for each toolchain and query, the `core` plan `round_<k>.py` with `k` the `output_round` of
the query's `results.json`, for the queries whose `results.json` has `"correct": true`. The kernels are extracted
and reviewed by the same protocol as the `full` plans above, after the gate's rules froze, into the task
directories `<query>_core_<toolchain>_<kernel>/`.

| Toolchain | Query | Plan | Kernels launched |
| --- | --- | --- | --- |
| cuda | q01 | `round_09.py` | none |
| cuda | q02 | `round_01.py` | `tpch_q2_sort_sel` |
| cuda | q03 | `round_06.py` | `filter_agg_sort_kernel` |
| cuda | q05 | `round_01.py` | `q5_kernel` |
| cuda | q06 | `round_01.py` | `q6_kernel` |
| cuda | q07 | `round_02.py` | `q7_kernel` |
| cuda | q08 | `round_01.py` | `q8_scatter_kernel`, `q8_compact_kernel` |
| cuda | q09 | `round_01.py` | none |
| cuda | q10 | `round_02.py` | `filter_agg_kernel` |
| cuda | q11 | `round_01.py` | none |
| cuda | q12 | `round_03.py` | `q12_kernel_mail`, `q12_kernel_ship` |
| cuda | q13 | `round_04.py` | `q13_hash_kernel`, `q13_gather_kernel` |
| cuda | q14 | `round_07.py` | `promo_revenue_kernel` |
| cuda | q15 | `round_01.py` | `filter_kernel` |
| cuda | q16 | `round_05.py` | `fill_hash_buckets_kernel`, `count_runs_kernel` |
| cuda | q17 | `round_10.py` | `q17_final_kernel` |
| cuda | q18 | `round_07.py` | `q18_threshold_kernel` |
| cuda | q19 | `round_01.py` | `q19_kernel` |
| cuda | q20 | `round_03.py` | `q20_filter_kernel`, `q20_isin_kernel` |
| cuda | q21 | `round_04.py` | `q21_kernel` |
| cuda | q22 | `round_03.py` | `q22_avg_kernel`, `q22_filter_kernel`, `q22_groupby_kernel` |
| triton | q01 | `round_09.py` | `_q1_filter_and_reduce` |
| triton | q02 | `round_01.py` | none |
| triton | q03 | `round_02.py` | `_filter_proj_kernel` |
| triton | q04 | `round_09.py` | none |
| triton | q05 | `round_01.py` | `_q5_kernel` (8 configurations) |
| triton | q06 | `round_01.py` | `_q6_kernel` (8 configurations) |
| triton | q07 | `round_05.py` | `_q7_kernel` (8 configurations) |
| triton | q08 | `round_05.py` | none |
| triton | q09 | `round_04.py` | `_q9_amount_kernel` |
| triton | q10 | `round_02.py` | none |
| triton | q11 | `round_06.py` | none |
| triton | q12 | `round_01.py` | `_q12_kernel` (8 configurations) |
| triton | q13 | `round_03.py` | `_q13_scatter_add_kernel`, `_q13_gather_counts_kernel` |
| triton | q14 | `round_01.py` | `_q14_kernel` (8 configurations) |
| triton | q17 | `round_05.py` | `_q17_filter_sum_kernel` (6 configurations) |
| triton | q18 | `round_02.py` | none |
| triton | q19 | `round_02.py` | `_q19_kernel` (8 configurations) |
| triton | q20 | `round_01.py` | none |
| triton | q21 | `round_06.py` | `_q21_filter_kernel` |
| triton | q22 | `round_05.py` | `_q22_avg_kernel`, `_q22_filter_group_kernel` |

That is 41 plans, 21 CUDA and 20 Triton. The plans of the other queries failed DataKernelBench's correctness
check: cuda q04; triton q15, q16.

### Kernels

87 kernels are extracted: 25 CUDA kernels and 62 Triton kernels. Six Triton kernels run through an
`@triton.autotune` decorator. Five list 16 configurations (`BLOCK_SIZE` 1024, 2048, 4096, 8192; `num_warps` 4
and 8; `num_stages` 2 and 4), `_q5_kernel` lists 24 (the same with `num_stages` 2, 3, and 4), and
`_q17_filter_sum_kernel` lists 12 (`BLOCK_SIZE` 4096, 8192, 16384; `num_warps` 4 and 8; `num_stages` 2 and 4).
As for the `full` plans, none of these kernels has a loop that Triton pipelines, so `num_stages` does not
change the compiled code (checked: the front end's PTX of every configuration is the same for each `num_stages`
the autotuner lists and for Triton's default); each such kernel gives one corpus kernel per `BLOCK_SIZE` and
`num_warps`, in the autotuner's order, the first without a suffix and the others `_2` onward: eight kernels
each for q05, q06, q07, q12, q14, and q19, and six for q17.

Two `tl.constexpr` arguments take a value at the launch: `num_nations` of `_q5_kernel` is `len(nations_unique)`,
5 for the plan's region ASIA (TPC-H has five nations per region at every scale factor), and `BLOCK_SIZE` is the
configuration's. The CUDA plan of q02 launches `tpch_q2_sort_sel` with `2 * n * 8` bytes of dynamic shared
memory clamped to [1 KiB, 128 KiB]; `launch.json` states `16 * n` bytes with `n` at most 3072, the 48 KiB limit
of a launch without an opt-in, above which the plan's launch does not run. The CUDA plans of q08 and q15 assign
their source string more than once; the kernels are those of the string that `_load_cuda` compiles (q08's
`_CUDA_SOURCE_FIXED`, q15's third `_CUDA_SOURCE`).

No launched kernel is left out: every one compiles through the corpus front ends within the README's edits, and
its binding builds.

The plans also compile or define kernels that they never launch; these are not corpus kernels:

- cuda q01: `q1_kernel` is compiled, but `_query_core` never calls `q1_launcher`; it aggregates with `torch.unique` and `scatter_add_`.
- cuda q09: `q9_kernel` is compiled, but `_query_core` never calls `q9_launcher`; it computes the amount with a PyTorch expression.
- cuda q11: `q11_value_kernel`, `q11_scatter_reduce_kernel`, `q11_sum_kernel`, and `q11_filter_keys_kernel` are compiled, but `q11_launcher` launches none of them and `_launch_q11` never calls it; the plan uses PyTorch operations.
- cuda q16: the sort between its two kernels is CUB's `DeviceRadixSort`, not a kernel of the plan.
- triton q02: `_q2_sort_kernel` is defined but never launched; `_query_core` returns `torch.arange`.
- triton q04: the plan defines no kernel; `_query_core` uses PyTorch operations.
- triton q08: `_q8_reduce` is defined but never launched; `_query_core` uses `scatter_add_`.
- triton q10: `_q10_sum_kernel` is defined but never launched; `_query_core` uses `scatter_add_`.
- triton q11: `_q11_kernel_group_agg` is defined but never launched; `_query_core` uses `scatter_add_`.
- triton q18: `_q18_groupby_sum_kernel` and `_q18_count_filter_kernel` are defined but never launched; `_query_core` uses `scatter_add_`.
- triton q20: `_tpc_h_20_kernel` is defined but never launched; `_query_core` uses `torch.unique`, `scatter_add_`, and `torch.isin`.

### Review

`verdicts.tsv` classifies each core kernel against its plan (C, E, or safe, as the README defines them), from
the source and the plan alone, without the gate's decisions, under the kernel's grid claim. CUDA: C 8, E 4,
safe 13. Triton: C 0, E 11, safe 51. In all: C 8, E 15, safe 64.

### Grid claims

The claims are stated as for the `full` plans: the largest grid of any launch, per dimension, where an access
stays inside its buffer only up to a grid.

- All 62 Triton kernels state `"grid": [2^31 / BLOCK_SIZE, 1, 1]` in `triton.json`, past which the 32-bit
  `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` wraps: 2097152 programs for `BLOCK_SIZE = 1024`, 1048576 for
  2048, 524288 for 4096, 262144 for 8192, and 131072 for 16384.
- `q03_core_cuda_filter_agg_sort_kernel` and `q08_core_cuda_q8_compact_kernel`: 8388608 blocks in x
  (`2^31 / 256`, where the `int` row index wraps) and 1 in y and z.
- `q18_core_cuda_q18_threshold_kernel`: 16777216 blocks in x (`2^32 / 256`, where the unsigned 32-bit row
  index wraps) and 1 in y and z; `q20_core_cuda_q20_filter_kernel` and `q20_core_cuda_q20_isin_kernel`:
  4194304 blocks in x (`2^32 / 1024`) and 1 in y and z.
- `q15_core_cuda_filter_kernel`: 1 block in y and z with x up to the GPU's limit; its 64-bit row index needs
  no bound in x, and the claim rules out the second block in y that ran every row twice.

The 18 other CUDA kernels state no grid: their accesses stay inside their buffers under every grid (a large
grid repeats rows behind an `idx < n` guard, or the grid-stride loop of `promo_revenue_kernel` never passes
it), or the defect does not depend on the grid (`q10_core_cuda_filter_agg_kernel` writes a per-block output
at `blockIdx.x`, which no constant grid bounds, and `q22_core_cuda_q22_filter_kernel` guards its position
above but not below).

The 8 `C` kernels, all CUDA: six take their write positions from an inout counter whose contents on entry the
gate does not establish (`q08_core_cuda_q8_compact_kernel`, whose 20-element outputs are also fewer than the
100 bins it compacts; `q15_core_cuda_filter_kernel`; `q18_core_cuda_q18_threshold_kernel`; the two q20
kernels; and `q22_core_cuda_q22_filter_kernel`, whose `pos < n` guard admits a negative counter);
`q03_core_cuda_filter_agg_sort_kernel` counts with `atomicAdd` on `result_sp[n]`, one past its `n` elements,
and writes its shared arrays past the 6144 bytes of the launch; and `q10_core_cuda_filter_agg_kernel` writes
`out_keys_ptr[blockIdx.x]` into a buffer of the plan's block count, which any larger grid passes.

The 15 `E` kernels need a relation that the gate cannot establish: the 8 q07 Triton kernels add at
`supp * 4 + cust * 2 + (year - 1995)` under `key < 8` only, so a kept shipdate before 1995 gives a negative
key (the plan's date parameters keep 1995 and 1996); `q07_core_cuda_q7_kernel` adds to an 8-element shared
array at `supp * 4 + cust * 2 + year_offset` without a bound on the 0/1 codes; `q08_core_cuda_q8_scatter_kernel`
adds at `year - 1990` into 100 bins (the plan's years are 1995 and 1996); `q02_core_cuda_tpch_q2_sort_sel`
writes `min(limit, n)` rows into an output of exactly that length, which the signature can name only as a size
symbol of its own; three q13 kernels index the count array by a customer key, the two gathers relying on
TPC-H keys being positive and the Triton scatter-add on every `o_custkey` being a `c_custkey` (the CUDA hash
kernel bounds the key and is safe); and `q21_core_triton__q21_filter_kernel` loads
`lo_orderkeys_ptr[n_lo]` when an l1 order key is above every key of `agg`, which the plan's joins rule out.

No kernel needs a grid bound that depends on a size symbol.
