# devstral-2-123b

DataKernelBench family `mistralai__devstral-2-123b-instruct-2512`, commit
`43c361d`. The plans are
`logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/<toolchain>/full/queries/<query>/round_<k>.py`
for each query whose `results.json` has `"correct": true`, with `k` its
`output_round`.

## Plans taken

20 plans, 10 per toolchain.

| Toolchain | Query | Round | Kernels launched | Corpus kernels |
| --- | --- | --- | --- | --- |
| cuda | q05 | 6 | `q5_kernel` | 1 |
| cuda | q06 | 1 | `q6_kernel` | 1 |
| cuda | q10 | 10 | `q10_kernel` | 1 |
| cuda | q12 | 3 | `q12_kernel` | 1 |
| cuda | q14 | 1 | `promo_revenue_kernel` | 1 |
| cuda | q15 | 5 | `q15_kernel` | 1 |
| cuda | q17 | 10 | `q17_first_pass`, `q17_second_pass` | 2 |
| cuda | q18 | 10 | (the plan runs the query with cuDF operations) | 0 |
| cuda | q19 | 1 | `q19_kernel` | 1 |
| cuda | q22 | 3 | `q22_stage1_kernel`, `q22_stage2_kernel` | 2 |
| triton | q01 | 8 | `_q1_fused_kernel`, 3 autotuner configurations | 3 |
| triton | q02 | 3 | (the plan defines `_triton_topk` and never launches it) | 0 |
| triton | q03 | 7 | `fused_filter_revenue_kernel` | 1 |
| triton | q04 | 2 | `_count_by_priority_kernel`, 16 autotuner configurations | 8 |
| triton | q06 | 1 | `_q6_kernel`, 16 autotuner configurations | 8 |
| triton | q09 | 2 | `_profit_kernel`, 16 autotuner configurations | 8 |
| triton | q14 | 2 | `_promo_revenue_kernel`, 8 autotuner configurations | 8 |
| triton | q15 | 8 | `_q15_kernel`, 8 autotuner configurations | 8 |
| triton | q18 | 7 | (the plan defines `_group_sum_kernel` and never launches it) | 0 |
| triton | q19 | 2 | `_q19_kernel`, 12 autotuner configurations | 6 |

The CUDA plans for q01-q04, q07-q09, q11, q13, q16, q20 and q21, and the
Triton plans for q05, q07, q08, q10-q13, q16, q17 and q20-q22, have
`"correct": false` and are not taken.

## Kernels

61 kernels extracted: 11 CUDA and 50 Triton. All 61 compile through the
corpus front ends (`kernel.O1.ptx` and `triton.ptx`) and their bindings build.

0 kernels left out: every kernel that a taken plan launches is extracted and compiles.

The autotuners of q04, q06, q09 and q19 list each pair of `BLOCK_SIZE` and
`num_warps` twice, with `num_stages=2` and `num_stages=4`. The corpus module
states `CONSTANTS` and `NUM_WARPS` and does not state `num_stages`, and these
kernels are loop-free, so pipelining leaves them unchanged and the two
configurations of a pair are one corpus kernel. Each kernel's `NOTES.md` names
its configuration; the suffixes `_2`, `_3`, ... follow the order of the
configurations in the plan.

## Review

`verdicts.tsv` holds the review of each kernel: 51 `C`, 2 `E`, 8 `safe`
before the grid claims below; 9 `C`, 10 `E`, 42 `safe` after them.

- `C`: `q15_cuda_q15_kernel` indexes arrays of `len(supplier)` elements by the
  raw supplier key, which reaches `len(supplier)`; every Triton kernel computes
  `pid * BLOCK_SIZE` in 32 bits, which wraps negative on a large grid; the
  q04 Triton kernel also adds atomically at `counts_ptr[-1]` from the masked
  lanes of its last program.
- `E`: `q10_cuda_q10_kernel` (customer keys of orders above `max(c_custkey)`)
  and `q22_cuda_q22_stage2_kernel` (a country code outside `[0, 100)` whose
  mask bit is set).
- `safe`: the remaining CUDA kernels of q05, q06, q12, q14, q17 (both), q19
  and q22 (stage 1).

## Grid claims

Reviewed blind, before the gate decided the kernels, after the claim language
gained `grid`: the largest grid of any launch, per dimension. The 50 Triton
`C` kernels needed a grid of more than `2^31 / BLOCK_SIZE` programs for their
32-bit `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` to wrap, and each got
that bound as its claim in `triton.json` (`"grid": [2^31 / BLOCK_SIZE, 1,
1]`): 2097152 programs for `BLOCK_SIZE = 1024`, 1048576 for 2048, 524288 for
4096, 262144 for 8192, 131072 for 16384, 65536 for 32768, and 32768 for
65536. `q15_cuda_q15_kernel` is `C` for a raw supplier key that reaches the
end of its buffers on the plan's own launch, not for the grid, and got no
claim.

Verdicts changed from `C` to `safe` for 34 kernels: the Triton kernels of
q01 (3), q03 (1), q06 (8), q09 (8), q14 (8), and q19 (6), whose every access
is at the masked offsets (q03's stores at `base + tl.arange` use the same
offsets) or at a constant index of a one- or three-element buffer.

Verdicts changed from `C` to `E` for the 8 q15 Triton kernels: the
remaining reads of `suppkey_map_ptr` at a raw `l_suppkey` and the atomics on
`totals_ptr` and `present_ptr` at the loaded map value stay inside their
buffers only under the lineitem-to-supplier foreign key and the map's
`[-1, n_supp)` values, which the plan gives and the gate cannot establish.

The 8 q04 Triton kernels stay `C` with a claim: their unmasked
`tl.atomic_add(counts_ptr + priority, ...)` adds at `counts_ptr[-1]` from
the masked lanes (`other = -1`) of the last program of the plan's own grid
whenever `n_elements` is not a multiple of `BLOCK_SIZE`, whatever the grid.

No kernel needs a grid bound that depends on a size symbol.

## Core plans

The held-out set: the `core` plans,
`logs/kernel_plans/tpch/sf10/h100/mistralai__devstral-2-123b-instruct-2512/<toolchain>/core/queries/<query>/round_<k>.py`
for each query whose `results.json` has `"correct": true`, with `k` its
`output_round`, extracted after the gate's rules froze. Their kernels are the
task directories `<query>_core_<toolchain>_<kernel>/`.

### Plans taken

18 plans: 10 CUDA and 8 Triton.

| Toolchain | Query | Round | Kernels launched | Corpus kernels |
| --- | --- | --- | --- | --- |
| cuda | q02 | 3 | (the plan defines `topk_kernel` and never launches it; `_query_core` calls `torch.topk`) | 0 |
| cuda | q05 | 10 | `q5_kernel` | 1 |
| cuda | q06 | 4 | `q6_kernel` | 1 |
| cuda | q07 | 3 | `q7_kernel` | 1 |
| cuda | q08 | 9 | `market_share_kernel` | 1 |
| cuda | q09 | 6 | `q9_kernel` | 1 |
| cuda | q11 | 8 | `groupby_sum_kernel` | 1 |
| cuda | q12 | 8 | (the plan defines `q12_kernel` and never launches it; `_query_core` uses `scatter_add_`) | 0 |
| cuda | q14 | 1 | `promo_fuse_kernel` | 1 |
| cuda | q19 | 1 | `q19_kernel` | 1 |
| triton | q02 | 6 | `_topk_kernel`, 16 autotuner configurations | 8 |
| triton | q05 | 4 | (the plan defines `_core_kernel` and never launches it; `_query_core` uses `scatter_add_`) | 0 |
| triton | q06 | 3 | `_q6_kernel`, 16 autotuner configurations | 8 |
| triton | q09 | 3 | `_q9_kernel`, 16 autotuner configurations | 8 |
| triton | q10 | 1 | (the plan defines `_q10_kernel` and `_q10_agg_kernel` and never launches them; `_query_core` uses `torch.unique` and `scatter_add_`) | 0 |
| triton | q12 | 5 | `_q12_kernel`, 16 autotuner configurations | 8 |
| triton | q19 | 2 | (the plan defines no kernel; `_query_core` is eager PyTorch) | 0 |
| triton | q22 | 2 | (the plan defines `_q22_atomic_scatter_add` and never launches it; `_query_core` is eager PyTorch) | 0 |

The CUDA plans for q01, q03, q04, q10, q13, q15-q18 and q20-q22, and the
Triton plans for q01, q03, q04, q07, q08, q11, q13-q18, q20 and q21, have
`"correct": false` and are not taken.

### Kernels

40 kernels extracted: 8 CUDA and 32 Triton. All 40 compile through the corpus
front ends (`kernel.O1.ptx` and `triton.ptx`) and their bindings build.

0 kernels left out: every kernel that a taken plan launches is extracted and
compiles.

- The four Triton autotuners list `BLOCK_SIZE` in {1024, 2048, 4096, 8192}
  and `num_warps` in {4, 8}, each pair twice, with `num_stages=2` and
  `num_stages=4`; each pair is one corpus kernel, with the suffixes `_2` to
  `_8` in the plan's order. q06, q09 and q12 are loop-free or unrolled
  (`tl.static_range(2)`). q02's `for i in range(min(limit, n_elements))`
  holds only scalar stores, so pipelining leaves it unchanged; the PTX for
  `num_stages=2` and `num_stages=4` was compiled outside the corpus and is the
  same for all eight configurations.
- Triton q02's `tl.constexpr` `limit` is the query parameter `limit`, 100 by
  default, which the plan passes unchanged; `CONSTANTS` states 100. The
  kernel's two loads have unused values, so its PTX has no load. Its output
  `out_ptr` has `min(limit, n_elements)` elements, which is not affine in the
  sizes, so `signature.json` gives it its own size symbol `n_out`.
- CUDA q08's per-block outputs `sum_brazil` and `sum_total` have
  `(n_elements + 1023) / 1024` elements, a quotient, which gives them their
  own size symbol `sum_brazil`.
- CUDA q11's kernel keeps the plan's `size_t` parameters (`unique_count`,
  `max_unique`, `n`); `signature.json` states them as `uint64`. Its device
  function `atomicAdd_ull` is copied unchanged.
- CUDA q06 and q09 leave out `<cuda_runtime.h>` (and q06 also
  `<device_launch_parameters.h>` and `<cstdio>`), which only their host code
  uses; the other CUDA kernels leave out `<torch/extension.h>` with the host
  code, as the `full` kernels do. CUDA q07's static shared memory is 48 KiB
  (`8 x 512` doubles and ints), the limit, which nvcc accepts.

### Review

`verdicts.tsv` holds the review of each core kernel: 33 `C`, 0 `E`, 7 `safe`
before the grid claims below; 9 `C`, 8 `E`, 23 `safe` after them. Per
toolchain after the claims: CUDA 1 `C`, 0 `E`, 7 `safe`; Triton 8 `C`, 8
`E`, 16 `safe`.

- `C`: `q08_core_cuda_market_share_kernel` adds into `sum_brazil[blockIdx.x]`
  and `sum_total[blockIdx.x]`, buffers of the plan's block count
  `(n_elements + 1023) / 1024`, without a bound, so a launch with more blocks
  than that writes past their end; the 8 q09 Triton kernels store
  `tl.store(out_ptr + offs, amount)` without a mask, so the last program of
  the plan's own grid stores past `n_elements` whenever it is not a multiple
  of `BLOCK_SIZE`.
- `E`: the 8 q02 Triton kernels store at `out_ptr + i` for
  `i < min(100, n_elements)` and stay inside `out_ptr` only because the plan
  allocates it with `n_out = min(100, n_elements)` elements, a relation
  between a buffer's length and a size that the gate cannot establish.
- `safe`: the CUDA kernels of q05, q06, q07, q09, q11, q14 and q19 (unsigned
  32-bit thread indices widened to 64 bits, row accesses under the row count,
  shared arrays sized by the block, constant indices into the one-, two-, or
  eight-element outputs, and q11's appends guarded by `pos < max_unique`, the
  outputs' length), and the Triton kernels of q06 and q12 (masked row loads,
  constant indices into the three-, one-, or two-element buffers).

### Grid claims

The 32 Triton kernels compute `offs = pid * BLOCK_SIZE + tl.arange(0,
BLOCK_SIZE)` in 32 bits, which wraps negative on a grid of more than
`2^31 / BLOCK_SIZE` programs, and each got that bound as its claim in
`triton.json` (`"grid": [2^31 / BLOCK_SIZE, 1, 1]`): 2097152 programs for
`BLOCK_SIZE = 1024`, 1048576 for 2048, 524288 for 4096, and 262144 for 8192.
The q02 kernels' wrapping loads are dead in the PTX, but the claim follows the
source, as for the `full` kernels.

Verdicts changed from `C` to `safe` for the 16 Triton kernels of q06 and q12,
whose every other access is at the masked offsets or at a constant index.
Verdicts changed from `C` to `E` for the 8 q02 Triton kernels (the `out_ptr`
length above). The 8 q09 Triton kernels stay `C` with a claim, for the
unmasked store.

`q08_core_cuda_market_share_kernel` is `C` for a grid bound in a size symbol
(`gridDim.x <= (n_elements + 1023) / 1024`), which the claim language does
not take, and got no claim. No other kernel needs a grid bound that depends
on a size symbol; the CUDA kernels' thread indices are unsigned 32-bit values
widened to 64 bits and need no bound.
