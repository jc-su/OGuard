# minimax-m2.5

DataKernelBench family `minimaxai__minimax-m2.5`, commit `43c361d`. Each plan
is `logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/<toolchain>/full/queries/<query>/round_<k>.py`,
with `k` the `output_round` of the query's `results.json`.

## Plans taken

The queries whose `results.json` has `"correct": true`: 36 plans.

| Toolchain | Query | Round | Kernels launched | Corpus kernels |
| --- | --- | --- | --- | --- |
| cuda | q01 | 08 | none (PyTorch only) | 0 |
| cuda | q02 | 04 | Thrust library kernels only | 0 |
| cuda | q03 | 05 | `count_filter_kernel`, `filter_project_kernel` | 2 |
| cuda | q04 | 06 | `count_by_priority_kernel` | 1 |
| cuda | q05 | 02 | `q5_revenue_kernel` | 1 |
| cuda | q06 | 01 | `q6_fused_kernel` | 1 |
| cuda | q08 | 01 | `hash_aggregate_kernel` | 1 |
| cuda | q09 | 02 | `q9_kernel` | 1 |
| cuda | q10 | 02 | `agg_revenue_kernel` | 1 |
| cuda | q11 | 02 | `compute_value_kernel` | 1 |
| cuda | q12 | 10 | `q12_kernel` | 1 |
| cuda | q13 | 04 | `count_orders_kernel`, `gather_counts_kernel` | 2 |
| cuda | q14 | 01 | `promo_revenue_kernel` | 1 |
| cuda | q15 | 01 | `filter_aggregate_kernel` | 1 |
| cuda | q16 | 03 | `pack_pairs_kernel`, `count_unique_atomic_kernel` | 2 |
| cuda | q18 | 04 | `filter_big_groups` (and Thrust library kernels) | 1 |
| cuda | q19 | 02 | `q19_kernel` | 1 |
| cuda | q21 | 03 | `q21_fused_kernel` | 1 |
| cuda | q22 | 08 | none (PyTorch only) | 0 |
| triton | q01 | 02 | `_q1_kernel` | 8 |
| triton | q02 | 02 | `_topk_kernel` | 8 |
| triton | q04 | 04 | none (`_q4_filter_count_kernel` is defined, not launched) | 0 |
| triton | q05 | 02 | `_agg_kernel` | 8 |
| triton | q06 | 01 | `_q6_kernel` | 8 |
| triton | q07 | 03 | `_q7_kernel` | 8 |
| triton | q08 | 07 | `_q8_aggregate_kernel` | 8 |
| triton | q09 | 01 | `_q9_compute_kernel` of `_KERNEL_SOURCE_V2` | 12 |
| triton | q10 | 01 | `_revenue_kernel` | 8 |
| triton | q11 | 04 | none (`_q11_sort_reduce_kernel` is defined, and its launcher is never called) | 0 |
| triton | q12 | 01 | `_q12_kernel` | 8 |
| triton | q14 | 01 | `_promo_revenue_kernel` | 8 |
| triton | q15 | 08 | `_q15_agg_kernel` | 8 |
| triton | q16 | 01 | `_count_unique_kernel` | 8 |
| triton | q18 | 06 | `_filter_threshold_kernel` | 6 |
| triton | q19 | 05 | `_q19_kernel` | 8 |
| triton | q22 | 03 | `_avg_kernel`, `_filter_groupby_kernel` | 16 |

The queries with `"correct": false` are not taken: cuda q07, q17, q20, and
triton q03, q13, q17, q21.

## Kernels extracted

35 kernels of the plans' sources are launched: 19 CUDA kernels and 16 Triton
kernels. They give 149 corpus kernels: 19 for nvcc and 130 for Triton, one per
autotuner configuration. Every one compiles through its front end and its
binding builds.

Every Triton kernel is launched through `@triton.autotune`. Its configurations
vary `BLOCK_SIZE`, `num_warps`, and `num_stages`. The front end takes the
`constexpr` values and `num_warps` but not `num_stages`, so two configurations
that differ only in `num_stages` give the same corpus kernel; each corpus
kernel is one (`BLOCK_SIZE`, `num_warps`) pair, numbered in the order of the
plan's configuration list (`BLOCK_SIZE` outer, `num_warps` inner).

## Kernels left out

No kernel of a plan's source is left out: none fails extraction or
compilation within the README's edits.

The plans also launch library kernels, which have no source in the plan and
so cannot be extracted within the README's edits:

- cuda q02: Thrust's `copy`, `sort_by_key`, and `reduce_by_key` (in
  `q2_min_cost_launcher`), and the fills of `thrust::device_vector`; the CUDA
  source defines no `__global__` kernel.
- cuda q18: Thrust's `sort_by_key`, `reduce_by_key`, and `copy`, and the fills
  of `thrust::device_vector`, around `filter_big_groups`.
- every plan: the PyTorch and cuDF operators it calls.

Kernels that a plan defines but does not launch are not corpus kernels:
cuda q08 `q8_aggregate_kernel`, `count_years_kernel`,
`aggregate_by_year_kernel`, `simple_aggregate_kernel`; triton q02
`_min_supplycost_kernel`, `_q2_filter_kernel`; triton q04
`_q4_filter_count_kernel`; triton q09 `_q9_fused_kernel` and the first
`_q9_compute_kernel` (both in `_KERNEL_SOURCE`, which the plan does not load)
and `_q9_fused_agg_kernel`; triton q11 `_q11_sort_reduce_kernel`; triton q20
`_aggregate_lineitem_kernel`, `_scan_and_compare_kernel`,
`_collect_qualifying_suppliers_kernel`.

## Contracts

- A size symbol's least value is 1 where the plan launches only with rows
  (an early return on an empty input, or a `max()` or `min()` of the input
  before the launch), and 0 otherwise.
- TPC-H facts that hold at every scale factor are stated where a kernel
  needs them: the nation table has 25 rows (cuda q05 `num_nations` at most 25,
  cuda q09 `num_nations` equal to 25), and order dates lie in 1992 to 1998
  (cuda q08 `num_years` at most 7).
- An index role is stated where the plan produces the index values in the
  target's range (ids from `range(len(...))`, `searchsorted` positions kept
  below the length), or sizes the target from the greatest of the same key
  column, whose keys are positive in TPC-H. It is not stated where the range
  rests on a foreign key between two tables (cuda q13 `count_orders_kernel`)
  or on values packed with other data (cuda q16 `count_unique_atomic_kernel`,
  triton q16, triton q01).
- A buffer that the plan zeroes and the kernel updates with atomics is
  `readwrite`; a buffer that the kernel only writes is `write`.

## Review

`verdicts.tsv` holds the review of each corpus kernel: 8 `C`, 22 `E`, and
119 `safe` with the grid claims below (133 `C`, 4 `E`, and 12 `safe` before
them).

- Every Triton kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0,
  BLOCK_SIZE)` in 32 bits. A grid of more than `2^31 / BLOCK_SIZE` programs
  wraps the offset to a negative value that passes `offs < n_elements`, so
  the masked loads read before their buffers: all 130 were `C` before the
  claims. Triton q07
  also reads past its inputs with the plan's own grid, since its `keep` mask
  does not include the bound test.
- cuda q03 `filter_project_kernel` and cuda q18 `filter_big_groups` take
  output positions from an atomic counter; their row index wraps modulo 2^32,
  so a large grid repeats a row and writes one position past the outputs:
  `C` before the claims. cuda q08 `hash_aggregate_kernel` steps its
  grid-stride loop by a signed 32-bit stride that turns negative on a grid of
  2^21 blocks: `C` before the claims.
- cuda q09, q12, q13 `count_orders_kernel`, and q16
  `count_unique_atomic_kernel` index by values of an input that the plan keeps
  in range but the gate cannot bound: `E`.
- The other 12 CUDA kernels are `safe`.

## Grid claims

A kernel's claim now states the largest grid of any launch (`grid` in
`triton.json`, or in the `nvcc` declaration of `launch.json`), and the catalog
refuses a larger launch. All 133 `C` kernels got a claim: the 130 Triton
kernels at `2^31 / BLOCK_SIZE` programs (2097152 for `BLOCK_SIZE` 1024,
1048576 for 2048, 524288 for 4096, 262144 for 8192); cuda q03 and q18 at
`2^32 / blockDim` blocks (4194304 and 16777216), the largest grid whose
unsigned row index does not wrap; and cuda q08 at 2097151 blocks, the largest
grid whose `int` stride `gridDim.x * blockDim.x` stays positive; all with one
block in y and z.

Verdicts changed: 107 from `C` to `safe` (106 Triton kernels and cuda q08),
and 18 from `C` to `E`, where the claim removes the grid defect and a plan
precondition that the gate cannot establish remains: Triton q01 (8 kernels,
`rf * 256 + ls` against 65536 slots), Triton q16 (8 kernels, `key // 100001`
against `num_groups`), and cuda q03 and q18 (the counter zeroed on entry,
and for q03 `filtered_count` as the count of matching rows).

8 kernels stay `C`: Triton q07, whose `keep` mask omits the bound test, so
the loads of `price_ptr` and `discount_ptr` read past their inputs on the
plan's own grid. No kernel of this model needs a bound that depends on a
size symbol.

## Core plans

The `core` plans are the held-out set: each is
`logs/kernel_plans/tpch/sf10/h100/minimaxai__minimax-m2.5/<toolchain>/core/queries/<query>/round_<k>.py`,
with `k` the `output_round` of the query's `results.json`, taken where it
has `"correct": true`: 39 plans, 18 cuda and 21 triton. Their kernels are the
tasks `<query>_core_<toolchain>_<kernel>/`, extracted and reviewed under the
rules above after they froze, without running the gate.

### Plans taken

| Toolchain | Query | Round | Kernels launched | Corpus kernels |
| --- | --- | --- | --- | --- |
| cuda | q02 | 01 | `generate_indices_kernel` | 1 |
| cuda | q05 | 01 | `q5_kernel` | 1 |
| cuda | q06 | 01 | `q6_kernel` | 1 |
| cuda | q07 | 02 | `q7_kernel` | 1 |
| cuda | q08 | 03 | `agg_kernel` | 1 |
| cuda | q09 | 01 | `compute_amount_kernel` | 1 |
| cuda | q10 | 01 | Thrust library kernels only | 0 |
| cuda | q11 | 01 | `reduce_by_key_kernel`, `filter_kernel` | 2 |
| cuda | q12 | 02 | none (the extension does not build; PyTorch fallback) | 0 |
| cuda | q13 | 03 | `histogram_kernel` | 1 |
| cuda | q14 | 04 | `promo_kernel` | 1 |
| cuda | q15 | 03 | `q15_filter_aggregate_kernel` | 1 |
| cuda | q17 | 03 | `compute_part_sums_kernel`, `compute_filtered_sum_kernel` | 2 |
| cuda | q18 | 01 | `group_sum_kernel` | 1 |
| cuda | q19 | 01 | `q19_kernel` | 1 |
| cuda | q20 | 01 | `sum_by_group_kernel`, `filter_and_collect_suppkeys_kernel` | 2 |
| cuda | q21 | 04 | `k` | 1 |
| cuda | q22 | 03 | `ker_avg`, `ker_grp` | 2 |
| triton | q02 | 02 | none (PyTorch only) | 0 |
| triton | q03 | 03 | `_filter_predicate_kernel` | 6 |
| triton | q04 | 02 | `_count_priority_kernel` | 8 |
| triton | q05 | 01 | `_agg_revenue_kernel` | 8 |
| triton | q06 | 01 | `_q6_kernel` | 8 |
| triton | q07 | 01 | `_q7_kernel` | 8 |
| triton | q08 | 02 | `_q8_accum_kernel` | 8 |
| triton | q09 | 02 | `_amount_kernel` | 8 |
| triton | q10 | 02 | `_agg_kernel` | 8 |
| triton | q11 | 01 | `_grouped_sum_kernel` | 8 |
| triton | q12 | 01 | `_q12_kernel` | 8 |
| triton | q13 | 06 | `_count_orders_kernel` | 8 |
| triton | q14 | 01 | `_promo_revenue_kernel` | 8 |
| triton | q15 | 07 | none (PyTorch only) | 0 |
| triton | q16 | 06 | `_count_unique_kernel` | 6 |
| triton | q17 | 05 | `_q17_phase1_kernel`, `_q17_phase2_kernel` | 16 |
| triton | q18 | 09 | none (`_simple_reduce_kernel` is defined, not launched) | 0 |
| triton | q19 | 02 | `_q19_kernel` | 8 |
| triton | q20 | 05 | none (three kernels are defined, not launched) | 0 |
| triton | q21 | 01 | `_count_kernel` | 8 |
| triton | q22 | 01 | `_group_agg_kernel` | 8 |

The queries with `"correct": false` are not taken: cuda q01, q03, q04, q16,
and triton q01.

### Kernels extracted

38 kernels of the plans' sources are launched: 20 CUDA kernels of 16 plans
and 18 Triton kernels of 17 plans. They give 160 corpus kernels: 20 for nvcc
and 140 for Triton, one per autotuner configuration. Every one compiles
through its front end and its binding builds.

Every Triton kernel is launched through `@triton.autotune`. Most list
`BLOCK_SIZE` in {1024, 2048, 4096, 8192}, `num_warps` in {4, 8}, and
`num_stages` in {2, 4}, which give 8 corpus kernels per plan kernel, numbered
as for the `full` plans (`BLOCK_SIZE` outer, `num_warps` inner). Triton q03
lists `BLOCK_SIZE` in {1024, 2048, 4096} and `num_warps` in {4, 8} with no
`num_stages`, and triton q16 `BLOCK_SIZE` in {256, 512, 1024} and `num_warps`
in {4, 8}: 6 corpus kernels each.

Two CUDA kernels take `uint8_t` from `<torch/extension.h>`, which nvcc cannot
include without PyTorch's headers: cuda q14 `promo_kernel` and cuda q21 `k`
include `<cstdint>` in its place, the case the collection's README records.
The triton q17 kernels annotate `n_elements: int`, which the extraction
keeps.

### Kernels left out

No kernel of a plan's source is left out: none fails extraction or
compilation within the README's edits.

Plans that launch no kernel of their own:

- cuda q10: Thrust's `sort_by_key`, `reduce_by_key`, and `copy`, and the
  fills of `thrust::device_vector`; the CUDA source defines no `__global__`
  kernel.
- cuda q12: the plan loads `_CUDA_SOURCE_V2`, whose launcher names
  `torch::k64`, which PyTorch does not define, so the extension does not
  build; the plan catches the failure and runs its PyTorch fallback
  (`_USE_CUDA = False`). Its kernel `q12_fused_kernel` is not launched. The
  first source string `_CUDA_SOURCE` (`q12_kernel`, `reduce_block_counts`,
  `q12_kernel_separate`) is never loaded.
- triton q02, q15: PyTorch only; the plans define no kernel.
- triton q18: `_simple_reduce_kernel` is defined, and `_query_core` uses
  `torch.unique` and `scatter_add_` instead.
- triton q20: `_sort_by_key_kernel`, `_segmented_reduce_kernel`, and
  `_eval_pred_kernel` are defined, and `_query_core` uses PyTorch instead.
- every plan: the PyTorch and cuDF operators it calls.

### Contracts

- The conventions of the `full` plans hold: a size symbol's least value is 1
  where the plan launches only with rows (cuda q02 `k`, q09 `n`, q13 `n`;
  triton q10 `n_elements`, q16 `n_elements`, q17 `n_parts`, and q08
  `n_rows` and `num_groups`, since `unique_years[0]` raises on an empty
  input), and 0 otherwise. An accumulator that masked lanes address at 0
  has least value 1 where a program runs only with rows and a row implies
  an element (triton q05 `num_nations`, q11 `num_groups`, q21
  `num_suppliers`), as for the `full` triton q05.
- TPC-H facts: cuda q05 `num_nations` at most 25 (`launch.json`); triton q13
  `num_customers_plus_one` at least 2 (keys start at 1).
- An index role is stated where the plan produces the values in the target's
  range: ids from `range(len(...))` (cuda q05 `nation_ids`, cuda q21 `s`,
  triton q05 `nation_id_ptr`, triton q16 `unique_group_ptr`, triton q21
  `supp_ids_ptr`), the inverse of `torch.unique` (cuda q08 `inv`, cuda q20
  `group_ids` of both kernels, triton q11 `inverse_ptr`), `searchsorted`
  positions clamped below the length (cuda q17 and triton q17 `part_idx`),
  and the running count of group starts (cuda q18 `group_idx`). It is not
  stated where the range rests on a foreign key (cuda q13 `keys`, triton q13
  `o_custkey_ptr`), on flag values (cuda and triton q07 `supp`, `cust`), on
  a phone prefix (triton q22 `cntrycode_ptr`), on an offset from the least
  key (triton q10 `custkey_ptr`), or on an unclamped `searchsorted` (cuda q21
  `idx`).
- A buffer that the plan zeroes and the kernel updates with atomics is
  `readwrite`; a buffer that the kernel only writes is `write` (cuda q20
  `output`, filled with -1 and written at every row).

### Review

`verdicts.tsv` holds the review of each corpus kernel: 6 `C`, 37 `E`, and
117 `safe` with the grid claims below; nvcc 15 `safe` and 5 `E`, Triton
6 `C`, 32 `E`, and 102 `safe`.

- Every Triton kernel computes its offsets in 32 bits (`pid * BLOCK_SIZE +
  tl.arange(...)`, or `pid * BLOCK_SIZE + i` in the scalar loops of triton
  q16 and q17), so each has the claim `2^31 / BLOCK_SIZE` programs (2097152
  for `BLOCK_SIZE` 1024, 1048576 for 2048, 524288 for 4096, 262144 for
  8192, and for triton q16 8388608 for 256 and 4194304 for 512); all 140
  were `C` before the claims.
- Triton q03 stays `C`: `tl.store(mask_ptr + offs, ...)` has no mask, so the
  last program writes past `mask_ptr` on the plan's own grid.
- Triton q07 (8), q10 (8), q13 (8), and q22 (8) are `E`: an unmasked atomic
  add at a value of an input that the plan keeps in range (nation flags and
  the 1995 to 1996 window, the key offset from the least key, the customer
  foreign key, the two-digit phone prefix) and the gate cannot bound.
- cuda q11 `reduce_by_key_kernel` steps its grid-stride loop by the unsigned
  32-bit `blockDim.x * gridDim.x` and takes output positions from an atomic
  counter: `C` before its claim of 4194303 blocks, the largest grid whose
  stride stays below 2^32 and whose rows are visited once; `E` under it
  (the counter zeroed on entry). cuda q11 `filter_kernel` keeps its row
  index in an `int32_t`: `C` before its claim of 2097152 blocks, `E` under
  it (the counter zeroed on entry).
- cuda q07 (nation flags), q13 (customer foreign key), and q21 (an unclamped
  `searchsorted` position) index by values the plan keeps in range: `E`.
- The other 15 CUDA kernels are `safe`: an unsigned 32-bit row index
  widened to int64 and guarded by the row count, shared arrays of the block
  size or of a constant that a guard bounds, and index roles where a value
  addresses an accumulator.

### Grid claims

All 140 Triton kernels state `2^31 / BLOCK_SIZE` programs in `triton.json`,
and cuda q11 `reduce_by_key_kernel` (4194303 blocks) and `filter_kernel`
(2097152 blocks) state their grids in `launch.json`, all with one block in
y and z. Verdicts changed by the claims: 102 Triton kernels from `C` to
`safe`, 32 Triton kernels and the two cuda q11 kernels from `C` to `E`; the
6 triton q03 kernels stay `C`. No other CUDA kernel needs a claim.
