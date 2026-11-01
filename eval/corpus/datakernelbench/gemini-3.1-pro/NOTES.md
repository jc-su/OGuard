# gemini-3.1-pro

DataKernelBench family `gcp__gemini-3.1-pro-preview`, commit `43c361d`.

## Plans taken

22 `full` plans whose `results.json` has `"correct": true`, each at its `output_round`, under `logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/<toolchain>/full/queries/<query>/`:

| Toolchain | Query | Plan file |
| --- | --- | --- |
| cuda | q01 | `round_04.py` |
| cuda | q02 | `round_02.py` |
| cuda | q06 | `round_01.py` |
| cuda | q10 | `round_02.py` |
| cuda | q12 | `round_02.py` |
| cuda | q13 | `round_04.py` |
| cuda | q14 | `round_02.py` |
| cuda | q15 | `round_02.py` |
| cuda | q16 | `round_02.py` |
| cuda | q17 | `round_02.py` |
| cuda | q19 | `round_01.py` |
| cuda | q21 | `round_02.py` |
| triton | q01 | `round_03.py` |
| triton | q06 | `round_01.py` |
| triton | q08 | `round_03.py` |
| triton | q11 | `round_02.py` |
| triton | q15 | `round_01.py` |
| triton | q16 | `round_03.py` |
| triton | q17 | `round_04.py` |
| triton | q18 | `round_03.py` |
| triton | q19 | `round_02.py` |
| triton | q21 | `round_02.py` |

The other queries' plans failed DataKernelBench's correctness check and are not taken: cuda q03, cuda q04, cuda q05, cuda q07, cuda q08, cuda q09, cuda q11, cuda q18, cuda q20, cuda q22, triton q02, triton q03, triton q04, triton q05, triton q07, triton q09, triton q10, triton q12, triton q13, triton q14, triton q20, triton q22.

## Kernels

The 22 plans launch 29 kernels: 17 CUDA kernels (one launch configuration each) and 12 Triton kernels. Eleven of the Triton kernels are autotuned: ten list the `BLOCK_SIZE` values 1024, 2048, 4096, 8192 with `num_warps` 4 and 8 (and `num_stages` 2 and 4 for nine of them), and `_q11_kernel` lists four configurations (`num_stages` 2 or 4); `_q8_kernel` has one launch with `BLOCK_SIZE = 4096`. Each (`BLOCK_SIZE`, `num_warps`) pair is one corpus kernel, numbered in the autotuner's order. The corpus module states `CONSTANTS` and `NUM_WARPS` and leaves `num_stages` at the front end's default; for every Triton corpus kernel the PTX is the same for `num_stages` 2, 3, and 4, so configurations that differ in `num_stages` alone are one corpus kernel.

Extracted: 102 corpus kernels, 17 CUDA (nvcc) and 85 Triton. Left out: 0. Every kernel compiles through the corpus front ends and its binding builds.

Two extraction notes:

- `q01_cuda_q1_kernel`: the plan's dynamic shared memory is `num_sets * num_groups * 48` bytes with `num_sets = 16` for `num_groups <= 64` and fewer sets above. The corpus kernel is the configuration `num_groups <= 64` (`768 * num_groups` bytes), which holds the plan's TPC-H launch (6 groups); the launches with more than 64 groups have a shared memory that is not affine in `num_groups` and are not in the corpus.
- `q16_cuda_set_bits_kernel`, `q16_cuda_count_bits_kernel`, `q21_cuda_q21_pass1`: the kernels use `uint32_t`, which the plan takes from `<torch/extension.h>`; the sources include `<cstdint>` in its place (recorded in each `NOTES.md`).

## Review

`verdicts.tsv` classifies each corpus kernel: 68 safe, 34 E, 0 C, after the grid claims of the next section. Before the claims it was 86 C, 9 E, 7 safe, for these reasons:

- Every Triton kernel computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` in int32 (the PTX shifts `%ctaid.x` in 32 bits and sign-extends the offset): a grid of more than `2^31 / BLOCK_SIZE` programs makes `offs` negative, `offs < n` holds, and the first load reads before its buffer (C before the claims). `_q15_kernel`, `_count_unique_sorted_kernel`, and `filter_and_count` also index a buffer by values whose range the gate cannot establish.
- `q16_cuda_count_bits_kernel` computes `int32_t g` from `blockIdx.x * blockDim.x`, which is negative for a grid of more than `2^23` blocks (C before the claims).
- The E kernels index a buffer by values whose range the plan establishes through another table or kernel (a foreign key, a dense map with -1 entries, category codes, non-negative counts, the product length of a bitmap), relations that the gate cannot establish.
- The other CUDA kernels compute `int64_t idx = blockIdx.x * blockDim.x + threadIdx.x` in unsigned 32-bit arithmetic, which stays non-negative for any grid.

## Grid claims

A kernel's `triton.json` or `launch.json` can state `grid`, the largest grid
of any launch as maxima per dimension. The catalog refuses a larger launch,
and the review assumes the claim (`0 <= program_id(0) < x`, and so on). All
86 `C` kernels received a claim and were re-reviewed under it with the same
rules otherwise (any grid up to the claim, an output's contents on entry
unknown, no relations between buffers). Result: 86 claims, 61 verdicts changed
from `C` to `safe`, 25 from `C` to `E`, 0 stay `C`; no kernel needs a bound in
a size symbol.

- The 85 Triton kernels claim `2^31 / BLOCK_SIZE` programs in x and 1 in y and
  z (the plans launch 1-D grids): 2097152 for `BLOCK_SIZE` 1024, 1048576 for
  2048, 524288 for 4096, 262144 for 8192. Under the claim `offs` stays below
  2^31 and does not wrap.
  - 61 became `safe`: the 8 `q01_triton_q1_kernel`, 8 `q06_triton__q6_kernel`,
    `q08_triton__q8_kernel`, 4 `q11_triton__q11_kernel`, 8 `q17_triton_pass1_kernel`,
    8 `q17_triton_pass2_kernel`, 8 `q18_triton__group_sum_kernel`, 8
    `q19_triton__q19_kernel`, and 8 `q21_triton_build_order_stats` configurations;
    their other accesses were already inside (masks, constant indices, or
    index-in columns).
  - 24 became `E`: the 8 `q15_triton__q15_kernel` configurations index
    `revenue_ptr` and `present_ptr` by the unchecked `l_suppkey` (the
    lineitem-supplier foreign key); the 8 `q16_triton__count_unique_sorted_kernel`
    configurations index `counts_ptr` by `curr // multiplier` (the composite key's
    construction); the 8 `q21_triton_filter_and_count` configurations index
    `counts` by a `target_supp_map` entry checked only for `>= 0` (the map's
    contents).
- `q16_cuda_count_bits_kernel` claims (8388608, 1, 1), `2^31 / 256` blocks with
  its 256-thread block, so `int32_t g` stays non-negative. It became `E`: the
  read `bitmap[w * num_groups + g]` for `w < num_words` is inside only when
  `bitmap` holds `num_words * num_groups` words, a relation between the scalar
  and the buffer's length that the gate cannot establish.

## Core plans

The held-out set: the `core` plans of the same family, extracted after the gate's rules froze, under `logs/kernel_plans/tpch/sf10/h100/gcp__gemini-3.1-pro-preview/<toolchain>/core/queries/<query>/`. 25 plans have `"correct": true`, each taken at its `output_round`:

| Toolchain | Query | Plan file |
| --- | --- | --- |
| cuda | q02 | `round_01.py` |
| cuda | q05 | `round_01.py` |
| cuda | q06 | `round_01.py` |
| cuda | q09 | `round_01.py` |
| cuda | q10 | `round_02.py` |
| cuda | q11 | `round_01.py` |
| cuda | q13 | `round_01.py` |
| cuda | q14 | `round_01.py` |
| cuda | q18 | `round_02.py` |
| cuda | q19 | `round_01.py` |
| cuda | q20 | `round_03.py` |
| cuda | q21 | `round_01.py` |
| triton | q02 | `round_01.py` |
| triton | q04 | `round_02.py` |
| triton | q06 | `round_01.py` |
| triton | q08 | `round_02.py` |
| triton | q09 | `round_02.py` |
| triton | q10 | `round_01.py` |
| triton | q11 | `round_01.py` |
| triton | q12 | `round_01.py` |
| triton | q14 | `round_05.py` |
| triton | q15 | `round_02.py` |
| triton | q16 | `round_02.py` |
| triton | q18 | `round_02.py` |
| triton | q19 | `round_02.py` |

The other core plans failed DataKernelBench's correctness check and are not taken: cuda q01, cuda q03, cuda q04, cuda q07, cuda q08, cuda q12, cuda q15, cuda q16, cuda q17, cuda q22, triton q01, triton q03, triton q05, triton q07, triton q13, triton q17, triton q20, triton q21, triton q22.

### Kernels

The 25 plans launch 33 kernels: 16 CUDA kernels (one launch configuration each; q13 launches two and q20 four) and 17 Triton kernels. The task directories are `<query>_core_<toolchain>_<kernel>/`. Thirteen of the Triton kernels are autotuned: eight (`mark_exists_kernel` and `count_priorities_kernel` of q04, `_q6_kernel`, `_q8_kernel`, `_q10_kernel`, `_q11_scatter_kernel`, `_q12_kernel`, `_q15_kernel`) list the `BLOCK_SIZE` values 1024, 2048, 4096, 8192 with `num_warps` 4 and 8 and `num_stages` 2 and 4, giving 8 corpus kernels each; `_arange_kernel` of q02 lists 1024, 2048, 4096 with `num_warps` 4 and 8 (6 corpus kernels); `_q14_kernel` lists four configurations, one per `BLOCK_SIZE` with its own `num_warps` and `num_stages` (4); `_pack_kernel` and `_unpack_count_kernel` of q16 list four configurations, one per `BLOCK_SIZE`, without `num_stages` (4 each). `_q9_kernel`, `_group_sum_kernel` of q18, and `_q19_kernel` have one launch each with `BLOCK_SIZE = 4096` and `num_warps = 8`. Each (`BLOCK_SIZE`, `num_warps`) pair is one corpus kernel, numbered in the autotuner's order; the corpus module states `CONSTANTS` and `NUM_WARPS` and leaves `num_stages` at the front end's default, and the PTX of each autotuned kernel is the same for `num_stages` 2, 3, and 4 (checked for one configuration of each), so configurations that differ in `num_stages` alone are one corpus kernel.

Extracted: 101 corpus kernels, 16 CUDA (nvcc) and 85 Triton. Left out: 0. Every kernel compiles through the corpus front ends and its binding builds.

Extraction notes:

- `q05_core_cuda_q5_kernel`: the plan's dynamic shared memory is `num_copies * num_nations * 8` bytes with `num_copies = 8` for `num_nations <= 768` and fewer copies above. The corpus kernel is the configuration `num_nations <= 768` (`64 * num_nations` bytes, `num_copies` stated as `[8, 8]`), which holds the plan's TPC-H launch (5 nations); the launches with more than 768 nations have a shared memory that is not affine in `num_nations` and are not in the corpus.
- `q18_core_cuda_q18_agg_kernel` and the four q20 kernels use `uint8_t` or `uint64_t`, which the plan takes from `<torch/extension.h>`; the sources include `<cstdint>` in its place. The q20 kernels copy the plan's device functions `hash_pair` or `hash_single` unchanged and leave out `<cuda_runtime.h>`, which nvcc includes for every source.
- `q04_core_triton_count_priorities_kernel`: the `tl.constexpr` `NUM_PRIORITIES` is the count of distinct `o_orderpriority` values, which TPC-H fixes at 5; `CONSTANTS` states 5.
- `q06_core_triton__q6_kernel` is the same source as the `full` plan's `_q6_kernel`; `q10_core_cuda_q10_kernel`, `q18_core_cuda_q18_agg_kernel`, `q10_core_triton__q10_kernel`, `q11_core_triton__q11_scatter_kernel`, and `q18_core_triton__group_sum_kernel` accumulate into a buffer of `max(key) + 1` elements that the plan sizes by the maximum of the key column it passes, so that column is `index-in` into the buffer, as in the `full` extraction.

### Grid claims

86 kernels claim a grid, all written before the review:

- The 85 Triton kernels claim `2^31 / BLOCK_SIZE` programs in x and 1 in y and z (the plans launch 1-D grids): 2097152 for `BLOCK_SIZE` 1024, 1048576 for 2048, 524288 for 4096, 262144 for 8192. Under the claim `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` stays below 2^31 and does not wrap.
- `q20_core_cuda_filter_supplier_kernel` claims (16777216, 1, 1), `2^32 / 256` blocks with its 256-thread block, where the unsigned 32-bit row index wraps and a row would run twice, and 1 in y and z; the kernel appends at `atomicAdd(out_count, 1)`.

The other 15 CUDA kernels compute `int64_t idx = blockIdx.x * blockDim.x + threadIdx.x` in unsigned 32-bit arithmetic, which stays non-negative for any grid, and a repeated row changes no access, so they claim no grid.

### Review

`verdicts.tsv` classifies each core corpus kernel under its claim, without the gate's decisions: 85 safe, 16 E, 0 C. By toolchain: nvcc 12 safe, 4 E; Triton 73 safe, 12 E.

- nvcc E: `q13_core_cuda_count_orders_kernel` adds at `order_counts[o_custkey]` with the unchecked orders-customer foreign key; `q13_core_cuda_cust_dist_kernel_optimized` indexes `sh_hist` and `count_frequencies` by a count read from `order_counts`, whose sign and bound the previous kernel's contents establish; `q20_core_cuda_probe_and_agg_kernel` indexes `ps_partkey`, `ps_suppkey`, and `ps_sum_qty` by a hash table entry checked only against -1; `q20_core_cuda_filter_supplier_kernel` writes `out_idx` at positions from the inout counter `out_count`, whose contents on entry the gate does not establish.
- Triton E: the 8 `q08_core_triton__q8_kernel` configurations add at `year - min_year`, inside `num_groups` elements only for the plan's `min_year` and `num_groups` taken from the column; the 4 `q16_core_triton__unpack_count_kernel` configurations add at the high 32 bits of a packed key, a group id below `num_groups` only by the packing.
- The safe kernels index by the masked or guarded row offset, by a constant index of a small output, by an `index-in` column (the plan sizes the output by the column's maximum, or builds the column as dense row numbers: `supp_ids` of q21, `l_okey_ptr` of q04), by a value the kernel checks against a size argument (`max_key` of q04, `max_s` of q15, `num_nations` of q05), by a hash masked with `H - 1` for a size `H` of at least 1024 (q20), or by a binary search kept inside `[0, num_lo)` (q21).
