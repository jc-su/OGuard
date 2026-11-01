# gpt-5.5 (DataKernelBench family `azure__gpt-5.5`)

## Plans taken

From DataKernelBench at commit `43c361d`, all 44 `full` plans of the family
passed the benchmark's correctness check (`"correct": true` in every
`results.json`). Each plan is
`logs/kernel_plans/tpch/sf10/h100/azure__gpt-5.5/<toolchain>/full/queries/<query>/round_<k>.py`
with `k` the query's `output_round`:

| Toolchain | Round 1 | Round 2 | Round 3 |
| --- | --- | --- | --- |
| `cuda` | q01, q02, q05, q06, q07, q08, q10, q11, q12, q16, q17, q18, q19, q20, q22 | q03, q09, q13, q14, q15, q21 | q04 |
| `triton` | q01 to q22 | | |

## Kernels

The 44 plans launch 103 kernels: 57 CUDA kernels in the 22 CUDA plans and
46 Triton kernels in the 22 Triton plans. Each kernel has one launch
configuration. The plans launch two kernels twice with the same
configuration, CUDA `q16 fill_i64_kernel` and Triton `q02 _mark_keys_kernel`,
and each is one corpus kernel. Of the 103 kernels, 99 are in the corpus
(57 CUDA, 42 Triton) and 4 Triton kernels are left out.

The corpus also includes kernels that the plan launches only on a branch
that the benchmark's run at scale factor 10 probably did not take:
- CUDA `q01_cuda_q1_kernel_generic`, for more than 6 groups.
- CUDA `q08_cuda_q8_kernel_shared_years` and
  `q08_cuda_q8_kernel_global_years`, for a year count other than 2 (at most
  128 and more than 128 years); the query's default range has 2 years.
- The three CUDA dense kernels of q18, for a lineitem table that is not
  sorted by `l_orderkey`.
- Triton `q09_triton__q9_row_atomic_kernel`, for more than 16 years.
- Triton `q13_triton__hist_full_kernel`, for a customer with 256 or more
  orders.

It also includes `q12_triton__q12_filter_agg_kernel`, which the plan launches
from its `_query_core`, a function that `run_query` does not call. Each
kernel's `NOTES.md` states the branch, and its signature states the bound
that the branch condition gives.

## Kernels left out

- `q05` Triton `_q5_grouped_revenue_kernel`: the plan passes the row count
  `n_elements` as a `tl.constexpr`. Its value is the row count of the
  plan's cuDF join (the 1994 lineitem rows of ASIA customers and suppliers of
  the same nation). The plan does not fix that count, and TPC-H does not state
  it, so the configuration of the launch at scale factor 10 is unknown.
- `q08` Triton `_q8_agg_kernel`: the same reason. `n_elements` and `base_year`
  are `tl.constexpr` values taken from the plan's cuDF join result `li`.
- `q11` Triton `_q11_compact_kernel`: Triton 3.8.0, the corpus front end,
  does not compile it. `tl.atomic_add(write_count_ptr, one, sem="relaxed", mask=keep)`
  adds a 1024-element value through the scalar pointer `write_count_ptr`, and
  the `tt.atomic_rmw` operation fails verification ("ptr type matches value
  type"). Compiling it would require changing the atomic's pointer, and the
  README does not allow that edit.
- `q16` Triton `_count_sorted_unique_pairs_kernel`: the plan passes the row
  count `n_elements` as a `tl.constexpr`. Its value is the number of
  (group, supplier) pairs left after the plan's cuDF filters and joins. The
  plan does not fix it, and the benchmark's records do not state it, so the
  configuration of the launch is unknown.

## Extraction and review conventions

- A Triton `tl.constexpr` whose value TPC-H fixes at scale factor 10 takes that
  value: table row counts (lineitem 59,986,052, orders 15,000,000, customer
  1,500,000, supplier 100,000), key maxima, category counts (3 return flags and
  2 line statuses in q01, 5 order priorities in q04), and the 7 country codes of
  q22. Row counts were checked against the benchmark's
  `benchmarks/tpch/meta_data/sf10/summarize/` tables.
- `TYPES` gives a Python `float` argument the type `fp32`, the type that
  Triton's JIT gives it, and a Python `int` `i32` below 2^31. The front end
  takes no `num_stages`.
- A size symbol has least value 1 where the plan returns before the launch
  when the size is 0, and 0 otherwise. `launch.json` states a size maximum
  where the plan bounds the size at the launch, for example with a
  `TORCH_CHECK` or a branch condition.
- A buffer has the role `index-in` only where the plan guarantees that its
  values lie in the target's range. For example, the plan sizes the target
  from the maximum of the same keys, or makes the values by construction.
  Foreign-key columns probed against another table's lookup, and buffers that
  hold -1 sentinels, are `read`.
- The review assumes that a launch can have any grid up to the claim of
  `triton.json` or `launch.json` (see "Grid claims" below) and, without a
  claim, (2^31 - 1, 65535, 65535), with the block of `launch.json` or
  `32 * num_warps`. Before the claims, 41 of the 42 Triton kernels were `C`. Most
  compute `pid * BLOCK + tl.arange(...)` in 32 bits, and each `triton.ptx`
  shows the `shl.b32` of `ctaid.x` and the sign-extended address. A large grid
  wraps the offset negative, the negative offset passes the `offs < n` mask,
  and the kernel accesses memory before the buffer. The CUDA kernels all build
  64-bit thread indices, so this wrap does not occur in them. The CUDA `C`
  kernels have other causes: they write `block_max[blockIdx.x]` into a buffer
  sized by the plan's block count, or they append at `atomicAdd(counter, 1)`
  without checking the position against the output length. A kernel that
  ignores `blockIdx.y` repeats every append when `gridDim.y` is 2.
- `E` marks an access that stays inside its buffer only because of contents
  that the gate cannot establish. Examples are row ids that an earlier kernel
  stored in a hash table or map (q15, q17, q20), or a lookup offset by the
  minimum of the same column (q14).

## Grid claims

A kernel's `triton.json` or `launch.json` can state `grid`, the largest grid
of any launch as maxima per dimension. The catalog refuses a larger launch,
and the review assumes the claim (`0 <= program_id(0) < x`, and so on). Of the
50 `C` kernels, 46 received a claim and were re-reviewed under it with the
same rules otherwise (any grid up to the claim, an output's contents on entry
unknown, no relations between buffers). Result: 46 claims, 36 verdicts changed
from `C` to `safe`, 10 from `C` to `E`, 4 stay `C` (none with a claim).

- 38 Triton kernels compute `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
  in 32 bits and claim `2^31 / BLOCK_SIZE` programs in x and 1 in y and z (the
  plans launch 1-D grids): 134217728 for `BLOCK_SIZE` 16 (1 kernel), 8388608
  for 256 (10), 4194304 for 512 (1), 2097152 for 1024 (21), 1048576 for 2048
  (2), 524288 for 4096 (3). Under the claim `offs` stays below 2^31 and does not
  wrap.
  - 34 became `safe`: their other accesses were already inside.
  - 3 became `E`: `q17_triton__q17_pass1_sum_count_kernel`,
    `q17_triton__q17_pass2_revenue_kernel`, and
    `q20_triton__q20_aggregate_lineitem_kernel` index `sum_qty_ptr` (and
    `cnt_qty_ptr`) by a value read from a map that an earlier kernel filled,
    checked only for `>= 0`.
  - 1 became `E`: `q15_triton__q15_select_kernel` appends at `count_ptr`'s old
    value plus a prefix sum without a bound; the claim's y = z = 1 removes the
    repeated selection, and the position stays inside only because the plan
    zeroes `count_ptr`, a precondition on an output's contents on entry that the
    catalog cannot check.
- 2 Triton kernels index a buffer of constant length by a program id and claim
  that length: `q13_triton__reduce_partial_hist_kernel` (256, 1, 1), the length
  of `hist_ptr`, and `q15_triton__q15_block_max_kernel` (98, 1, 1), the length
  of `block_max_ptr`. Both became `safe`.
- 6 CUDA kernels append at `atomicAdd(counter, 1)` without a bound and ignore
  `blockIdx.y`: `q11_cuda_fill_passing_parts_kernel`,
  `q13_cuda_compact_histogram_kernel`, `q15_cuda_q15_fill_selected_kernel`,
  `q18_cuda_q18_fill_big_dense_kernel`, `q18_cuda_q18_fill_big_sorted_runs_kernel`,
  and `q20_cuda_filter_supplier_kernel`. Their thread indices are 64-bit, so
  each claims (2147483647, 1, 1): every 1-D grid, and no launch that repeats
  the appends through `gridDim.y`. All 6 became `E`: the position stays inside
  only because the plan zeroes the counter (and sizes the outputs with a count
  kernel over the same inputs), preconditions on an output's contents on entry
  and on relations between buffers that the catalog cannot check.
- 4 `C` kernels have no claim, since their bound is in a size symbol, which the
  claim language does not take: `q13_cuda_reduce_max_array_kernel`
  (`gridDim.x <= (n_values + 1023) / 1024`, the length of `block_max`),
  `q13_cuda_reduce_max_customer_counts_kernel`
  (`gridDim.x <= (n_customers + 1023) / 1024`), `q15_cuda_q15_reduce_max_kernel`
  (`gridDim.x <= (n_sup + 1023) / 1024`), and `q20_triton__build_ps_slots_kernel`
  (`program_id(0) < n_ps`, the length of the two partsupp columns; its second
  defect, the unbounded old value of `counts_ptr[p]`, would then make it `E`).

## Core plans

The `core` plans are the held-out set of the collection, extracted after the
gate's rules froze. From DataKernelBench at commit `43c361d`, all 44 `core`
plans of the family passed the benchmark's correctness check. Each plan is
`logs/kernel_plans/tpch/sf10/h100/azure__gpt-5.5/<toolchain>/core/queries/<query>/round_<k>.py`
with `k` the query's `output_round`:

| Toolchain | Round 1 | Round 2 | Round 3 | Round 4 | Round 6 |
| --- | --- | --- | --- | --- | --- |
| `cuda` | q01, q02, q04 to q10, q12, q14 to q22 | q03, q11 | q13 | | |
| `triton` | q01 to q07, q09 to q12, q15 to q19, q21, q22 | q08, q20 | | q14 | q13 |

The task directories are `<query>_core_<toolchain>_<kernel>/`, and the rows of
`roles.tsv` and `verdicts.tsv` follow the `full` rows.

### Kernels

The 44 plans launch 104 kernels: 59 CUDA kernels in the 22 CUDA plans and 45
Triton kernels in the 22 Triton plans. Each kernel has one launch
configuration. Of the 104 kernels, 101 are in the corpus (57 CUDA, 44 Triton)
and 3 are left out (below).

Kernels that a plan launches more than once with the same configuration are one
corpus kernel each: the reduction loops CUDA `q13 q13_reduce_max_i32_kernel`,
CUDA `q15 q15_max_plain_kernel`, and Triton `q04 _max_i64_kernel`; the scan
helpers CUDA `q18 scan_counts_kernel`, `scan_small_kernel`, and
`add_block_offsets_kernel` (two calls on each path) and `q18 total_counts_kernel`
(one call on each path); and Triton `q16 _fill_i64_kernel` (the counts with 0,
the table with -1).

The corpus also includes kernels that the plan launches only on a branch that
the benchmark's run at scale factor 10 probably did not take:
- CUDA `q18`: the dense path (`max_key_kernel`, `dense_accumulate_kernel`,
  `dense_count_blocks_kernel`, `dense_compact_kernel`), taken only when
  `l_orderkey` is not nondecreasing.
- Triton `q13 _hist_customer_counts_i32_kernel`, taken only when the customer
  row count differs from the largest customer key.
- Triton `q15 _q15_hash_agg_kernel`, the hash path, taken only when `s_suppkey`
  is not `1..n_sup` (its insert kernel is left out, below).
Each kernel's `NOTES.md` states the branch.

### Kernels left out

- `q03` CUDA `q3_filter_project_kernel` and `q3_write_top_kernel`: the plan
  passes buffers of the structs `Q3Key` (16 bytes: `int64_t`, three `int32_t`)
  and `Q3Top` (24 bytes: `double`, `int64_t`, two `int32_t`). The signature's
  element types are 1, 2, 4, or 8 bytes wide, and stating the buffers with
  another element type would change the type of the kernel's accesses, which
  the README does not allow. nvcc compiles both kernels.
- `q15` Triton `_q15_hash_insert_kernel`: Triton 3.8.0, the corpus front end,
  does not compile it (`unsupported AST node type: Break`, the `break` inside
  its `while True` probe loop). The plan launches it only on the hash path,
  which the benchmark's run did not take, so the plan's own run never compiled
  it either.

### Extraction and review conventions

The conventions of the `full` extraction apply. Points specific to the `core`
plans:
- `tl.constexpr` values fixed by TPC-H at scale factor 10 or by the plan:
  `q02 n_elements = 100` (`min(limit, rows)` with the plan's `limit` of 100; the
  Q2 answer has 100 rows), `q04 NUM_PRIORITIES = 5`, `q05 NUM_NATIONS = 5` (the
  nations of ASIA), `q13 HIST_BINS = 64`, `q14 _q14_finalize_kernel
  BLOCK_SIZE = 8192` (the next power of two of the 7,323 scan programs over the
  59,986,052 joined rows), `q21 SEARCH_ITERS = 23` (the bit length of the count
  of orders with status `F`, about 48.6% of 15,000,000 orders, which lies
  between 2^22 and 2^23), and the plans' own `MAX_PROBE` (64), `MAX_PROBES`
  (128), and block sizes.
- A size maximum from the launch itself: CUDA `q04 q4_count_priorities_kernel`
  requests `8 * num_priorities` bytes of dynamic shared memory, which the launch
  refuses above 48 KiB, so `launch.json` states `num_priorities <= 6144`.
- A length that a branch establishes: Triton `q13
  _hist_dense_customer_counts_i32_kernel` runs only when the customer row count
  equals the largest key, so `counts_ptr` has `n_customers + 1` elements.
- Roles: a key column is `index-in` where the plan sizes the target from the
  maximum of the same column (`q04`, `q10`, `q11`, `q13`, `q17`, `q18`, `q22`)
  or builds the values as ordinals (`q01`, `q04` priority codes, `q05`, `q16`,
  `q21`); foreign keys probed against another table's buffer (`q04` order keys
  of orders, `q13` and `q22` customer keys of orders, `q17` and `q20` lineitem
  keys), keys offset by a minimum (`q10`, `q11` Triton), and keys clamped before
  the maximum (`q18` CUDA dense path) are `read`.
- A size symbol that the kernel does not receive (an output sized by an earlier
  kernel's count, a table sized by the host) is a `symbol` row without an
  argument row, as in the `full` rows.

### Grid claims

- 34 Triton kernels compute `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
  in 32 bits and claim `2^31 / BLOCK_SIZE` programs in x and 1 in y and z (the
  plans launch 1-D grids): 524288 for `BLOCK_SIZE` 4096 (1 kernel), 1048576 for
  2048 (3), 2097152 for 1024 (23), 4194304 for 512 (1), 8388608 for 256 (6).
  Under the claim 24 are `safe` and 10 are `E`: the value read from a map or
  hash table that an earlier kernel filled (`q15 _q15_hash_agg_kernel`, `q16
  _count_groups_from_table_kernel`, `q17 _q17_sum_count_kernel` and
  `_q17_final_sum_kernel`, `q20 _accum_lineitem_kernel`), a key offset by the
  minimum of the same column (`q10`, `q11`), a foreign key bounded only by the
  other table's maximum (`q13 _scatter_order_counts_i32_kernel`), and an append
  at an atomic cursor that the plan zeroes (`q20 _select_supplier_kernel`).
- 3 CUDA kernels append at `atomicAdd(counter, 1)` without a bound and ignore
  `blockIdx.y`: `q11 q11_compact_filtered_kernel`, `q15 q15_fill_selected_kernel`,
  and `q20 select_supplier_kernel`. Their thread indices are 64-bit, so each
  claims (2147483647, 1, 1); all three are `E`.
- 6 Triton `C` kernels have no claim, since their bound is in a size symbol:
  `q03 _q3_count_kernel` and `_q3_scatter_kernel`, `q04 _max_i64_kernel`, and
  `q14 _q14_scan_kernel` (a store or load at `program_id(0)` into a buffer of
  the plan's program count), and `q20 _insert_ps_kernel` and
  `_insert_qual_supp_kernel` (`program_id(0)` indexes the partsupp columns).
- 8 CUDA `C` kernels have no claim for the same reason: `q10
  q10_max_reduce_kernel`, `q13 q13_reduce_max_i32_kernel`, `q15
  q15_max_present_kernel` and `q15_max_plain_kernel`, `q18
  sorted_count_segments_kernel`, `scan_counts_kernel`, and
  `dense_count_blocks_kernel`, and `q22 max_i64_kernel` write
  `block[blockIdx.x]` into a buffer of the plan's block count.
- 4 Triton kernels use no program id (`q08 _q8_finalize_2y_kernel`, `q13
  _sort_fixed_hist_i32_kernel`, `q14 _q14_finalize_kernel`, `q22
  _q22_extract_kernel`) and have no claim; `q14 _q14_finalize_kernel` is `E`
  (its load bound `n_blocks` is related to the buffer's length only by the
  plan's `partial_stride = max(n_blocks, 1)`), the other three are `safe`.

### Verdicts

- CUDA (`nvcc`), 57 kernels: 40 `safe`, 9 `E`, 8 `C`.
- Triton, 44 kernels: 28 `safe`, 10 `E`, 6 `C`.

The 9 CUDA `E` kernels: the three append kernels above; `q10
q10_compact_sorted_kernel`, `q18 sorted_compact_segments_kernel`, and `q18
dense_compact_kernel` (positions from a prefix sum over the same data); `q17
q17_accumulate_avg_kernel`, `q17 q17_final_sum_kernel`, and `q20
aggregate_lineitem_kernel` (row ids read from a hash table that an earlier
kernel filled).
