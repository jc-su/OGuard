# qwen3.5-397b

The kernels of the DataKernelBench family `qwen__qwen3.5-397b-a17b-fp8`, from DataKernelBench at
commit `43c361d`, extracted and reviewed as `../README.md` states.

## Plans taken

For each query whose `results.json` has `"correct": true`, the `full` plan
`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/<toolchain>/full/queries/<query>/round_<k>.py`
with `k` the query's `output_round`: 19 CUDA plans (q07, q16, and q18 are not
correct) and 22 Triton plans, 41 in all.

| Plan | Round | Corpus kernels |
| --- | --- | --- |
| cuda q01 | 2 | `q01_cuda_q1_kernel` |
| cuda q02 | 9 | `q02_cuda_filter_min_cost_kernel` |
| cuda q03 | 3 | `q03_cuda_compute_block_prefix_kernel`, `q03_cuda_q3_compact_kernel` |
| cuda q04 | 3 | `q04_cuda_mark_valid_keys_kernel`, `q04_cuda_count_priorities_kernel` |
| cuda q05 | 1 | `q05_cuda_q5_kernel` |
| cuda q06 | 1 | `q06_cuda_q6_kernel` |
| cuda q08 | 3 | `q08_cuda_q8_kernel` |
| cuda q09 | 5 | `q09_cuda_q9_kernel` |
| cuda q10 | 1 | `q10_cuda_q10_revenue_kernel` |
| cuda q11 | 3 | none (below) |
| cuda q12 | 6 | `q12_cuda_q12_kernel` |
| cuda q13 | 1 | none (below) |
| cuda q14 | 1 | `q14_cuda_q14_kernel` |
| cuda q15 | 4 | `q15_cuda_q15_kernel` |
| cuda q17 | 1 | `q17_cuda_kernel_agg`, `q17_cuda_kernel_final` |
| cuda q19 | 1 | `q19_cuda_q19_kernel` |
| cuda q20 | 2 | `q20_cuda_q20_kernel` |
| cuda q21 | 2 | none (below) |
| cuda q22 | 3 | `q22_cuda_q22_kernel`, `q22_cuda_q22_avg_kernel` |
| triton q01 | 7 | `q01_triton__q1_kernel`, `q01_triton__q1_kernel_2`, `q01_triton__q1_kernel_3`, `q01_triton__q1_kernel_4`, `q01_triton__q1_kernel_5`, `q01_triton__q1_kernel_6` |
| triton q02 | 5 | `q02_triton__min_cost_per_partkey`, `q02_triton__filter_min_cost` |
| triton q03 | 3 | `q03_triton__count_valid_kernel`, `q03_triton__count_valid_kernel_2`, `q03_triton__filter_pack_kernel`, `q03_triton__filter_pack_kernel_2` |
| triton q04 | 7 | `q04_triton__q4_kernel` |
| triton q05 | 4 | `q05_triton__q5_kernel` |
| triton q06 | 1 | `q06_triton__q6_kernel`, `q06_triton__q6_kernel_2`, `q06_triton__q6_kernel_3`, `q06_triton__q6_kernel_4`, `q06_triton__q6_kernel_5`, `q06_triton__q6_kernel_6`, `q06_triton__q6_kernel_7`, `q06_triton__q6_kernel_8` |
| triton q07 | 3 | `q07_triton__q7_kernel`, `q07_triton__q7_kernel_2`, `q07_triton__q7_kernel_3`, `q07_triton__q7_kernel_4`, `q07_triton__q7_kernel_5`, `q07_triton__q7_kernel_6` |
| triton q08 | 1 | `q08_triton__q8_kernel`, `q08_triton__q8_kernel_2`, `q08_triton__q8_kernel_3`, `q08_triton__q8_kernel_4`, `q08_triton__q8_kernel_5`, `q08_triton__q8_kernel_6`, `q08_triton__q8_kernel_7`, `q08_triton__q8_kernel_8` |
| triton q09 | 6 | `q09_triton__q9_amount_kernel` |
| triton q10 | 1 | `q10_triton__q10_aggregate_kernel`, `q10_triton__q10_aggregate_kernel_2`, `q10_triton__q10_aggregate_kernel_3`, `q10_triton__q10_aggregate_kernel_4`, `q10_triton__q10_aggregate_kernel_5`, `q10_triton__q10_aggregate_kernel_6`, `q10_triton__q10_aggregate_kernel_7`, `q10_triton__q10_aggregate_kernel_8` |
| triton q11 | 1 | `q11_triton__q11_accumulate_kernel`, `q11_triton__q11_accumulate_kernel_2`, `q11_triton__q11_accumulate_kernel_3`, `q11_triton__q11_accumulate_kernel_4`, `q11_triton__q11_accumulate_kernel_5`, `q11_triton__q11_accumulate_kernel_6`, `q11_triton__q11_accumulate_kernel_7`, `q11_triton__q11_accumulate_kernel_8` |
| triton q12 | 3 | `q12_triton__q12_kernel`, `q12_triton__q12_kernel_2`, `q12_triton__q12_kernel_3`, `q12_triton__q12_kernel_4`, `q12_triton__q12_kernel_5`, `q12_triton__q12_kernel_6` |
| triton q13 | 6 | none (below) |
| triton q14 | 1 | `q14_triton__promo_kernel`, `q14_triton__promo_kernel_2`, `q14_triton__promo_kernel_3`, `q14_triton__promo_kernel_4`, `q14_triton__promo_kernel_5`, `q14_triton__promo_kernel_6`, `q14_triton__promo_kernel_7`, `q14_triton__promo_kernel_8` |
| triton q15 | 8 | `q15_triton__q15_kernel` |
| triton q16 | 1 | `q16_triton__count_distinct_per_group_kernel`, `q16_triton__count_distinct_per_group_kernel_2`, `q16_triton__count_distinct_per_group_kernel_3`, `q16_triton__count_distinct_per_group_kernel_4`, `q16_triton__count_distinct_per_group_kernel_5`, `q16_triton__count_distinct_per_group_kernel_6`, `q16_triton__count_distinct_per_group_kernel_7`, `q16_triton__count_distinct_per_group_kernel_8` |
| triton q17 | 1 | `q17_triton__accumulate_kernel`, `q17_triton__accumulate_kernel_2`, `q17_triton__accumulate_kernel_3`, `q17_triton__accumulate_kernel_4`, `q17_triton__accumulate_kernel_5`, `q17_triton__accumulate_kernel_6`, `q17_triton__accumulate_kernel_7`, `q17_triton__accumulate_kernel_8`, `q17_triton__filter_sum_kernel`, `q17_triton__filter_sum_kernel_2`, `q17_triton__filter_sum_kernel_3`, `q17_triton__filter_sum_kernel_4`, `q17_triton__filter_sum_kernel_5`, `q17_triton__filter_sum_kernel_6`, `q17_triton__filter_sum_kernel_7`, `q17_triton__filter_sum_kernel_8` |
| triton q18 | 4 | `q18_triton__orderkey_sum_kernel`, `q18_triton__orderkey_sum_kernel_2`, `q18_triton__orderkey_sum_kernel_3`, `q18_triton__orderkey_sum_kernel_4`, `q18_triton__orderkey_sum_kernel_5`, `q18_triton__orderkey_sum_kernel_6`, `q18_triton__orderkey_sum_kernel_7`, `q18_triton__orderkey_sum_kernel_8`, `q18_triton__orderkey_sum_kernel_9`, `q18_triton__orderkey_sum_kernel_10`, `q18_triton__orderkey_sum_kernel_11`, `q18_triton__orderkey_sum_kernel_12` |
| triton q19 | 2 | `q19_triton__q19_kernel`, `q19_triton__q19_kernel_2`, `q19_triton__q19_kernel_3`, `q19_triton__q19_kernel_4`, `q19_triton__q19_kernel_5`, `q19_triton__q19_kernel_6`, `q19_triton__q19_kernel_7`, `q19_triton__q19_kernel_8` |
| triton q20 | 3 | `q20_triton__q20_compare_kernel`, `q20_triton__q20_compare_kernel_2`, `q20_triton__q20_compare_kernel_3`, `q20_triton__q20_compare_kernel_4`, `q20_triton__q20_compare_kernel_5`, `q20_triton__q20_compare_kernel_6`, `q20_triton__q20_compare_kernel_7`, `q20_triton__q20_compare_kernel_8` |
| triton q21 | 1 | `q21_triton__q21_kernel`, `q21_triton__q21_kernel_2`, `q21_triton__q21_kernel_3`, `q21_triton__q21_kernel_4`, `q21_triton__q21_kernel_5`, `q21_triton__q21_kernel_6` |
| triton q22 | 1 | `q22_triton__q22_avg_kernel`, `q22_triton__q22_avg_kernel_2`, `q22_triton__q22_avg_kernel_3`, `q22_triton__q22_avg_kernel_4`, `q22_triton__q22_avg_kernel_5`, `q22_triton__q22_avg_kernel_6`, `q22_triton__q22_avg_kernel_7`, `q22_triton__q22_avg_kernel_8`, `q22_triton__q22_agg_kernel`, `q22_triton__q22_agg_kernel_2`, `q22_triton__q22_agg_kernel_3`, `q22_triton__q22_agg_kernel_4`, `q22_triton__q22_agg_kernel_5`, `q22_triton__q22_agg_kernel_6`, `q22_triton__q22_agg_kernel_7`, `q22_triton__q22_agg_kernel_8` |

## Kernels

162 corpus kernels: 20 CUDA kernels and 142 Triton kernels.
The Triton kernels come from 25 kernels of the plans; a plan that autotunes a
kernel gives one corpus kernel per configuration, suffixed `_2`, `_3`, and so on in
the order of the autotuner's list. The autotuner configurations of these plans also
vary `num_stages`; the Triton front end does not take `num_stages`, and none of these
kernels has a loop that the software pipeliner would transform (`tl.static_range`
loops are unrolled), so configurations that differ only in `num_stages` give one
corpus kernel. Each kernel's `NOTES.md` names the `num_stages` values it covers.

Task names keep the kernel's name in the plan, so a Triton kernel named `_q1_kernel`
gives `q01_triton__q1_kernel`.

## Left out

- `q21_cuda_q21_count_kernel` and `q21_cuda_q21_count_kernel_shared` (CUDA q21,
  round 2): both call `atomicAdd(&out_counts[...], 1LL)` or `atomicAdd(&s_counts[...],
  1LL)` on an `int64_t` pointer, and CUDA declares no `atomicAdd` for a signed 64-bit
  integer, so nvcc refuses the plan's own source ("no instance of overloaded function
  atomicAdd matches the argument list, argument types are: (int64_t *, long long)").
  Making it compile would change the type of an access, which the README does not
  allow. The plan wraps the build in `try: _EXT = _load_cuda() except Exception:
  _EXT = None` and runs a PyTorch fallback when `_EXT is None`, so DataKernelBench's
  correct run most likely never launched these kernels: the same nvcc error ended
  round 2 of this model's CUDA q13 plan in DataKernelBench's own log.
- CUDA q11 (round 3) and CUDA q13 (round 1): the plans write no kernel. They launch
  only Thrust algorithms (q11: `thrust::transform`, `thrust::sort_by_key` twice,
  `thrust::reduce_by_key`, `thrust::reduce`, and `thrust::copy_if` with a device
  lambda; q13: `thrust::sort` and `thrust::reduce_by_key`), whose kernels are
  Thrust's and CUB's templates rather than code of the plan, and none of them is an
  `extern "C"` kernel that the README's edits can produce.

Kernels that a plan defines but never launches are not corpus kernels: Triton q13's
`_histogram_kernel` (the plan uses `torch.bincount` instead) and Triton q18's
`_filter_kernel` (the plan filters with a PyTorch mask).

## Other observations

- CUDA q22 allocates `count_out` with `torch::zeros({7}, torch::kInt64)`, a host
  tensor, and passes its pointer to `q22_kernel`; the kernel's accesses to it stay
  within its 7 elements.
- Several Triton kernels add into a fixed number of slots of an output that the plan
  sizes from the data (Triton q01: 36 slots of `6 * n_rf * n_ls`; Triton q04: 5 slots
  of `num_priorities`), or index an accumulator by a key whose range the plan takes
  from TPC-H's data without establishing it (Triton q15: `l_suppkey` into `n_sup + 1`
  totals). Each of these kernels also has a 32-bit program offset, which decides its
  verdict.

## Review

`verdicts.tsv` reviews each kernel from its source and its plan, before and
without the gate's decisions: 13 `C`, 17 `E`, 132 `safe` with the grid claims
below (147 `C`, 1 `E`, 14 `safe` before them).

- Every Triton kernel computes its offsets as `pid * BLOCK_SIZE + tl.arange(0,
  BLOCK_SIZE)` (or `pid * BLOCK_SIZE + i`) in 32 bits; a grid of more than
  `2^31 / BLOCK_SIZE` programs makes them wrap negative, the mask `offs < n_elements`
  holds for them, and the first masked load reads before its buffer. A launch can
  have any grid up to the kernel's claim, so these kernels were `C` before the
  claims. Triton q02's `_filter_min_cost` also stores
  past `n` without a mask on the plan's own grid, and Triton q03's kernels access
  the per-program buffers at `pid` without a bound.
- CUDA, before the claims: `C` for the compaction kernels whose output position
  comes from a counter
  (q02, q20: a grid with two blocks in y runs every row twice), for the per-block
  buffers indexed by `blockIdx.x` (q03), and for a signed 32-bit thread index (q15);
  `E` for q12, whose local array is indexed by 0/1 flags that the gate cannot
  bound; the others are `safe`.

## Grid claims

A kernel's claim now states the largest grid of any launch (`grid` in
`triton.json`, or in the `nvcc` declaration of `launch.json`), and the catalog
refuses a larger launch. 140 kernels got a claim: the 137 Triton kernels whose
only grid defect is the 32-bit offset, claimed at `2^31 / BLOCK_SIZE` programs
(2097152 for `BLOCK_SIZE` 1024, 1048576 for 2048, 524288 for 4096, 262144 for
8192, 131072 for 16384), and CUDA q02, q15, and q20, claimed at the largest
grid whose thread index does not wrap (`2^32 / blockDim` blocks for the
unsigned index widened to `int64_t` in q02 and q20, that is 4194304 and
16777216; `2^31 / 1024` = 2097152 for the `int` index of q15), with one block
in y and z.

Verdicts changed: 118 from `C` to `safe` (117 Triton kernels and CUDA q15),
and 16 from `C` to `E`, where the claim removes the grid defect and a plan
precondition that the gate cannot establish remains: Triton q01 (6 kernels,
36 output slots against `6 * n_groups`), Triton q04 (5 slots against
`num_priorities`), Triton q07 (6 kernels, `supp` and `cust` as 0/1 flags),
Triton q15 (`l_suppkey` within `n_suppliers`), and CUDA q02 and q20 (the
counter zeroed on entry).

13 kernels stay `C`:

- Triton q21 (6 kernels): the claim is stated, but the binary search's
  midpoint `(low + high) // 2` is 32-bit and wraps negative once `n_agg`
  exceeds `2^30`, so a load of `agg_orderkey_ptr` reads before its buffer.
  The first review did not list this defect.
- Triton q02 `_filter_min_cost`: no claim; its unmasked store past `n` is on
  the plan's own grid.
- Triton q03 `_count_valid_kernel` (2 kernels) and `_filter_pack_kernel`
  (2 kernels), and CUDA q03 `compute_block_prefix_kernel` and
  `q3_compact_kernel`: the bound would depend on a size symbol, which the
  claim language does not take: `(n_elements + 4095) / 4096` programs for
  the Triton kernels and `(n + 1023) / 1024` blocks for the CUDA kernels, one
  per element of the per-block buffer. `_filter_pack_kernel` and
  `q3_compact_kernel` would also still rest on the prefix-sum relation of
  their block offsets, which the gate does not establish.

## Core plans

The `core` plans of the same family, extracted after the gate's rules froze
(the collection's held-out set), as `../README.md` states. Their task
directories are named `<query>_core_<toolchain>_<kernel>/`.

### Plans taken

For each query whose `results.json` under `core` has `"correct": true`, the plan
`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/<toolchain>/core/queries/<query>/round_<k>.py`
with `k` the query's `output_round`: 18 CUDA plans (q03, q10, and q20 are not
correct, and q07 has no `results.json`) and 19 Triton plans (q16, q17, and q21
are not correct), 37 in all.

| Plan | Round | Corpus kernels |
| --- | --- | --- |
| cuda q01 | 1 | `q01_core_cuda_q1_kernel` |
| cuda q02 | 1 | `q02_core_cuda_generate_indices_kernel` |
| cuda q04 | 2 | `q04_core_cuda_q4_exists_bincount_kernel` |
| cuda q05 | 1 | `q05_core_cuda_q5_kernel` |
| cuda q06 | 1 | `q06_core_cuda_q6_kernel` |
| cuda q08 | 2 | `q08_core_cuda_q8_aggregate_kernel` |
| cuda q09 | 1 | `q09_core_cuda_q9_amount_kernel` |
| cuda q11 | 1 | `q11_core_cuda_q11_fused_aggregate_kernel`, `q11_core_cuda_q11_simple_aggregate_kernel`, `q11_core_cuda_q11_total_sum_kernel` |
| cuda q12 | 4 | `q12_core_cuda_q12_kernel` |
| cuda q13 | 2 | `q13_core_cuda_count_orders_per_customer`, `q13_core_cuda_gather_customer_counts` |
| cuda q14 | 1 | `q14_core_cuda_q14_kernel` |
| cuda q15 | 2 | `q15_core_cuda_q15_kernel` |
| cuda q16 | 3 | `q16_core_cuda_count_unique_per_group_kernel` |
| cuda q17 | 2 | `q17_core_cuda_q17_compute_aggs`, `q17_core_cuda_q17_compute_final` |
| cuda q18 | 1 | `q18_core_cuda_accumulate_kernel`, `q18_core_cuda_extract_kernel` |
| cuda q19 | 1 | `q19_core_cuda_q19_kernel` |
| cuda q21 | 2 | `q21_core_cuda_q21_kernel` |
| cuda q22 | 2 | `q22_core_cuda_q22_phase1_kernel`, `q22_core_cuda_q22_phase2_kernel` |
| triton q01 | 3 | `q01_core_triton__q1_kernel`, `q01_core_triton__q1_kernel_2`, `q01_core_triton__q1_kernel_3`, `q01_core_triton__q1_kernel_4`, `q01_core_triton__q1_kernel_5`, `q01_core_triton__q1_kernel_6`, `q01_core_triton__q1_kernel_7`, `q01_core_triton__q1_kernel_8` |
| triton q02 | 2 | `q02_core_triton__topk_index_kernel`, `q02_core_triton__topk_index_kernel_2`, `q02_core_triton__topk_index_kernel_3`, `q02_core_triton__topk_index_kernel_4`, `q02_core_triton__topk_index_kernel_5`, `q02_core_triton__topk_index_kernel_6`, `q02_core_triton__topk_index_kernel_7`, `q02_core_triton__topk_index_kernel_8` |
| triton q03 | 2 | `q03_core_triton__filter_kernel` |
| triton q04 | 1 | `q04_core_triton__filter_orders_kernel`, `q04_core_triton__filter_orders_kernel_2`, `q04_core_triton__filter_orders_kernel_3`, `q04_core_triton__filter_orders_kernel_4`, `q04_core_triton__filter_orders_kernel_5`, `q04_core_triton__filter_orders_kernel_6`, `q04_core_triton__filter_orders_kernel_7`, `q04_core_triton__filter_orders_kernel_8`, `q04_core_triton__filter_lineitems_kernel`, `q04_core_triton__filter_lineitems_kernel_2`, `q04_core_triton__filter_lineitems_kernel_3`, `q04_core_triton__filter_lineitems_kernel_4`, `q04_core_triton__filter_lineitems_kernel_5`, `q04_core_triton__filter_lineitems_kernel_6`, `q04_core_triton__filter_lineitems_kernel_7`, `q04_core_triton__filter_lineitems_kernel_8` |
| triton q05 | 1 | `q05_core_triton__q5_kernel`, `q05_core_triton__q5_kernel_2`, `q05_core_triton__q5_kernel_3`, `q05_core_triton__q5_kernel_4`, `q05_core_triton__q5_kernel_5`, `q05_core_triton__q5_kernel_6`, `q05_core_triton__q5_kernel_7`, `q05_core_triton__q5_kernel_8` |
| triton q06 | 1 | `q06_core_triton__q6_kernel`, `q06_core_triton__q6_kernel_2`, `q06_core_triton__q6_kernel_3`, `q06_core_triton__q6_kernel_4`, `q06_core_triton__q6_kernel_5`, `q06_core_triton__q6_kernel_6`, `q06_core_triton__q6_kernel_7`, `q06_core_triton__q6_kernel_8` |
| triton q07 | 3 | `q07_core_triton__q7_kernel` |
| triton q08 | 1 | `q08_core_triton__market_share_kernel`, `q08_core_triton__market_share_kernel_2`, `q08_core_triton__market_share_kernel_3`, `q08_core_triton__market_share_kernel_4`, `q08_core_triton__market_share_kernel_5`, `q08_core_triton__market_share_kernel_6`, `q08_core_triton__market_share_kernel_7`, `q08_core_triton__market_share_kernel_8` |
| triton q09 | 3 | `q09_core_triton__profit_kernel` |
| triton q10 | 1 | `q10_core_triton__q10_kernel`, `q10_core_triton__q10_kernel_2`, `q10_core_triton__q10_kernel_3`, `q10_core_triton__q10_kernel_4`, `q10_core_triton__q10_kernel_5`, `q10_core_triton__q10_kernel_6`, `q10_core_triton__q10_kernel_7`, `q10_core_triton__q10_kernel_8` |
| triton q11 | 1 | `q11_core_triton_fused_scatter_add_kernel`, `q11_core_triton_fused_scatter_add_kernel_2`, `q11_core_triton_fused_scatter_add_kernel_3`, `q11_core_triton_fused_scatter_add_kernel_4`, `q11_core_triton_fused_scatter_add_kernel_5`, `q11_core_triton_fused_scatter_add_kernel_6`, `q11_core_triton_fused_scatter_add_kernel_7`, `q11_core_triton_fused_scatter_add_kernel_8` |
| triton q12 | 4 | none (below) |
| triton q13 | 4 | `q13_core_triton__q13_count_orders_kernel`, `q13_core_triton__q13_gather_counts_kernel` |
| triton q14 | 2 | `q14_core_triton__promo_revenue_kernel`, `q14_core_triton__promo_revenue_kernel_2`, `q14_core_triton__promo_revenue_kernel_3`, `q14_core_triton__promo_revenue_kernel_4`, `q14_core_triton__promo_revenue_kernel_5`, `q14_core_triton__promo_revenue_kernel_6`, `q14_core_triton__promo_revenue_kernel_7`, `q14_core_triton__promo_revenue_kernel_8` |
| triton q15 | 4 | `q15_core_triton__q15_kernel`, `q15_core_triton__q15_kernel_2`, `q15_core_triton__q15_kernel_3`, `q15_core_triton__q15_kernel_4`, `q15_core_triton__q15_kernel_5`, `q15_core_triton__q15_kernel_6`, `q15_core_triton__q15_kernel_7`, `q15_core_triton__q15_kernel_8` |
| triton q18 | 4 | none (below) |
| triton q19 | 2 | `q19_core_triton__q19_kernel`, `q19_core_triton__q19_kernel_2` |
| triton q20 | 1 | `q20_core_triton__scatter_add_kernel`, `q20_core_triton__scatter_add_kernel_2`, `q20_core_triton__scatter_add_kernel_3`, `q20_core_triton__scatter_add_kernel_4`, `q20_core_triton__scatter_add_kernel_5`, `q20_core_triton__scatter_add_kernel_6`, `q20_core_triton__scatter_add_kernel_7`, `q20_core_triton__scatter_add_kernel_8` |
| triton q22 | 1 | `q22_core_triton__avg_kernel`, `q22_core_triton__avg_kernel_2`, `q22_core_triton__avg_kernel_3`, `q22_core_triton__avg_kernel_4`, `q22_core_triton__avg_kernel_5`, `q22_core_triton__avg_kernel_6`, `q22_core_triton__agg_kernel`, `q22_core_triton__agg_kernel_2`, `q22_core_triton__agg_kernel_3`, `q22_core_triton__agg_kernel_4`, `q22_core_triton__agg_kernel_5`, `q22_core_triton__agg_kernel_6` |

### Kernels

139 corpus kernels: 24 CUDA kernels and 115 Triton kernels. The Triton kernels
come from 19 kernels of the plans; as for the `full` plans, an autotuned kernel
gives one corpus kernel per `BLOCK_SIZE` and `num_warps`, in the order of the
autotuner's list, and the configurations that differ only in `num_stages` give
one corpus kernel (Triton q22's `_agg_kernel` has a `tl.static_range` loop,
which is unrolled; the others have no loop). Triton q03, q07, q09, and q13 fix
their `BLOCK_SIZE` without an autotuner and take the default `num_warps = 4`;
Triton q19 lists two configurations.

Task names keep the kernel's name in the plan, so Triton q11's
`fused_scatter_add_kernel` gives `q11_core_triton_fused_scatter_add_kernel`
and the kernels named with a leading underscore give `..._triton__...`.

Extraction cases beyond the README's list, each recorded in the kernel's
`NOTES.md`:

- CUDA q05 and q12 include only `<torch/extension.h>`; `<cstdint>` is included
  in its place for `int32_t` and `int64_t`. CUDA q17, q18, and q22 include
  `<cstdio>`, the Thrust headers, and `<algorithm>`, which the kernels do not
  use; they are left out with the host code.
- CUDA q01 and q11 launch with dynamic shared memory: 49152 bytes for
  `q1_kernel`, 8192 bytes for `q11_total_sum_kernel`, and
  `8 * n_unique_keys` bytes for `q11_fused_aggregate_kernel`, whose
  `launch.json` states the term and the size maximum `n_unique_keys <= 1024`
  of the branch that launches it (with more distinct keys the plan launches
  `q11_simple_aggregate_kernel`, whose `n_unique_keys` symbol therefore has
  the least value 1025). Both branches are corpus kernels, since the plan
  launches one or the other by the data.
- CUDA q04's `priority_counts` is declared `unsigned long long int*`, a
  spelling the corpus's reader of C element types (`C_ELEMENTS` in
  `eval/corpus/sources.py`) does not list; `sources nvcc` and `binding_of`
  take the names only and compile it, and the source is kept as the plan
  states it.
- CUDA q02's `out_ptr` has `min(n, limit)` elements, a length that is not
  affine in the two scalars; it is its own size symbol `n_out`.
- CUDA q15 launches a fixed grid of 1024 blocks and indexes a per-block buffer
  by `blockIdx.x`; its claim states that grid (below).

### Left out

- Triton q12 (round 4): `_q12_kernel` declares `n_elements: tl.constexpr` and
  the plan passes the row count of the joined table, a value taken from the
  data that is not known at the launch (the README's fourth case), so its 8
  configurations are left out.
- Triton q18 (round 4): `_fused_group_sum_filter_kernel` is defined but never
  launched; `_query_core` computes the sums with `torch.unique_consecutive`
  and `scatter_add_`. The plan gives no corpus kernel.
- The plans that DataKernelBench's check did not pass (CUDA q03, q10, q20;
  Triton q16, q17, q21) and CUDA q07, which has no `results.json`, are not
  in the selection.

Every extracted kernel compiles through its front end; none is left out for
a compile failure.

### Grid claims

Every Triton kernel computes its offsets as `pid * BLOCK_SIZE + tl.arange(0,
BLOCK_SIZE)` in 32 bits, so each of the 115 Triton kernels claims `2^31 /
BLOCK_SIZE` programs in `triton.json` (2097152 for `BLOCK_SIZE` 1024, 1048576
for 2048, 524288 for 4096, 262144 for 8192). Two CUDA kernels claim a grid in
`launch.json`: `q15_core_cuda_q15_kernel` claims `[1024, 1, 1]`, the fixed
grid of its only launch (the plan's `num_blocks = 1024`), whose per-block rows
of `block_totals` a larger grid would overrun; `q18_core_cuda_extract_kernel`
claims `[4194304, 1, 1]` (2^32 / 1024 blocks in x, one in y and z), the
largest grid whose thread index does not wrap, since its output positions
come from counters. The other 22 CUDA kernels index the rows by a thread
index that is unsigned and widened, or by a grid-stride loop, and need no
claim.

### Review

`verdicts.tsv` reviews each kernel under its claim, from its source and its
plan, without the gate's decisions: 0 `C`, 30 `E`, 109 `safe`; by toolchain,
CUDA 0 `C`, 6 `E`, 18 `safe`, and Triton 0 `C`, 24 `E`, 91 `safe`.

The `E` kernels rest on a precondition of the plan that the gate cannot
establish:

- A range of the data that the plan does not check: CUDA q01 and Triton q01
  (8 kernels) index a fixed accumulator by the category codes of
  `l_returnflag` and `l_linestatus` (CUDA q01 checks the key from above only,
  so a negative code reaches its dynamic shared memory; Triton q01's key
  `rf * 256 + ls` assumes both below 256); CUDA q04 zeroes and flushes a
  static array of `MAX_PRIORITIES = 10` entries up to `num_priorities`, the
  number of distinct order priorities (5 at TPC-H's data); Triton q07 adds at
  `supp * 4 + cust * 2 + year` for 0/1 flags, unmasked; Triton q22's
  `_agg_kernel` (6 kernels) looks a two-digit country code up in a table of
  100 entries.
- A foreign key of TPC-H: CUDA and Triton q13's counting kernels index a
  buffer sized by the largest `c_custkey` with `o_custkey`; Triton q15
  (8 kernels) indexes a table sized by the largest `s_suppkey` with
  `l_suppkey`. The gathering kernels of q13, which index the same buffer with
  `c_custkey`, take `index-in` and are `safe`, as the `full` extraction did
  for a buffer sized from the largest value of the column that indexes it.
- A relation the signature cannot state: CUDA q02's `out_ptr` has
  `min(n, limit)` elements; CUDA q15's `block_stride` equals `n_sup`, and its
  `supp_pos` values are below `n_sup`; CUDA q18's `extract_kernel` appends at
  positions from counters that the plan zeroes on entry.

The `safe` kernels bound every access by a masked or guarded row index, by a
binary search that stays within `[0, n)`, by a fixed slot below a fixed
length, or by an `index-in` argument whose range the plan establishes
(`torch.unique` inverses, `range(len(...))` ids, or a buffer sized from the
column's largest value).
