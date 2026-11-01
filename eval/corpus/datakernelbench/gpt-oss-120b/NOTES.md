# gpt-oss-120b

The kernels of the DataKernelBench family `openai__gpt-oss-120b`, extracted
from DataKernelBench at commit `43c361d` by the protocol of
[../README.md](../README.md).

## Plans taken

The `full` plan of each query whose `results.json` has `"correct": true`, at
its `output_round`, under
`logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/<toolchain>/full/queries/<query>/`:

| Toolchain | Plans (query: round) |
| --- | --- |
| CUDA (19) | q01: 1, q05: 2, q06: 1, q07: 1, q08: 3, q09: 3, q10: 1, q11: 2, q12: 2, q13: 9, q14: 3, q15: 1, q16: 5, q17: 1, q18: 1, q19: 1, q20: 7, q21: 5, q22: 4 |
| Triton (17) | q01: 6, q02: 4, q03: 3, q04: 5, q05: 1, q06: 1, q07: 3, q09: 2, q10: 2, q12: 5, q13: 3, q14: 2, q15: 6, q17: 2, q19: 2, q21: 3, q22: 2 |

The CUDA plans of q02, q03, and q04 and the Triton plans of q08, q11, q16,
q18, and q20 did not pass DataKernelBench's correctness check and are not
taken.

## Plans that launch no kernel of their own

Five of the 36 plans launch no kernel that the plan's source defines, so they
give no corpus kernel:

- CUDA q13: `torch.bincount` and `gather` only; the plan has no CUDA source.
- CUDA q16: cuDF only; the plan has no CUDA source.
- CUDA q18: its `_CUDA_SOURCE` has no `__global__` function; the launcher calls
  Thrust's `sort_by_key` and `reduce_by_key`, library kernels.
- CUDA q20: its `_CUDA_SOURCE` is an empty `PYBIND11_MODULE`; the query runs
  in PyTorch.
- Triton q02: its `_KERNEL_SOURCE` defines `_unused_range_kernel`, which the
  plan never launches.

## Kernels extracted

130 kernels from the other 31 plans: 17 CUDA kernels (q17 and q22 launch two
each) and 113 Triton kernels. No kernel is left out: every kernel compiles
through the corpus front ends and its binding builds.

A Triton kernel gives one corpus kernel per autotuner configuration, the
second and later with the suffixes `_2`, `_3`, and so on, in the order of the
plan's configuration list. Six plans list each `BLOCK_SIZE` and `num_warps`
twice, with `num_stages` 2 and 4 (q06, q07, q12, q14, q19, and both kernels of
q22); the two give one corpus kernel, since a corpus module states no
`num_stages` and none of these kernels has a loop that software pipelining
would change. Each task's `NOTES.md` names its configuration.

Three extraction decisions go beyond the literal list of edits and are
recorded in the kernels' `NOTES.md`:

- CUDA declarations. Six CUDA kernels (q01, q07, q12, q14, q21, and q22
  `group_kernel`) have comments in their parameter lists, some with commas or
  parentheses. The corpus reads a kernel's parameters from its declaration
  (`eval/corpus/sources.py`, `c_parameters`), which does not skip comments, so
  the `extern "C"` declaration states the parameters without those comments.
  The parameters, their order, types, and qualifiers are the plan's.
- q08 `q8_agg_kernel` uses `uint8_t`, which the plan declares through
  `<torch/extension.h>`. The nvcc front end has no torch include path, so
  `kernel.cu` includes `<cstdint>` in its place.
- q08's launch passes `16 * num_years` bytes of dynamic shared memory. A
  launch above the default limit of 48 KiB does not run, so `launch.json`
  states the size maximum `num_years <= 3072`.

Every CUDA launch has 1024 threads per block (`BLOCK_SIZE = 1024`).

## Review

`verdicts.tsv` classifies each kernel from its source and plan, with the
facts that the gate can establish (buffer lengths, scalar ranges, and the
value range of an index input), any grid, and unknown output contents on
entry.

Before the grid claims below:

| Toolchain | C | E | safe |
| --- | --- | --- | --- |
| CUDA | 2 | 7 | 8 |
| Triton | 112 | 0 | 1 |
| All | 114 | 7 | 9 |

After them: CUDA 0 `C`, 7 `E`, 10 `safe`; Triton 12 `C`, 24 `E`, 77 `safe`;
all 12 `C`, 31 `E`, 87 `safe`.

- C, 32-bit Triton program offsets (112): every Triton kernel except
  `_noop_kernel` computes `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
  in 32 bits (the PTX shifts `%ctaid.x` with `shl.b32`, compares with
  `setp.lt.s32`, and sign-extends the offset), so a grid of more than
  `2^31 / BLOCK_SIZE` programs gives a negative offset that passes the mask
  and reads before the first buffer.
- C, 32-bit CUDA row ids (2): q08 `q8_agg_kernel` and q22 `group_kernel` hold
  `blockIdx.x * blockDim.x + threadIdx.x` in an `int`, which is negative on a
  grid of more than `2^21` blocks and passes the bound test.
- E (7): the out-of-bounds access needs a data precondition that the gate
  cannot establish: category codes below 256 (q01), at most 64 nations (q05),
  0/1 nation codes (q07), keys bounded by a buffer sized from their maximum
  and not negative (q10, q11), referential integrity of lineitem supplier
  keys (q15), and an order-key map whose values are -1 or an index (q21).
- safe (9): the CUDA kernels of q06, q09, q12, q14, q17 (both), q19, and q22
  `avg_kernel`, whose row ids are 64-bit and whose shared arrays match the
  1024-thread block, and the Triton `_noop_kernel` of q15.

## Grid claims

Reviewed blind, before the gate decided the kernels, after the claim language
gained `grid`: the largest grid of any launch, per dimension. All 114 `C`
kernels needed a large grid for the access outside the buffer, and all 114
got a claim:

- 112 Triton kernels, whose 32-bit `pid * BLOCK_SIZE + tl.arange(0,
  BLOCK_SIZE)` wraps once the grid exceeds `2^31 / BLOCK_SIZE` programs, got
  that bound in `triton.json` (`"grid": [2^31 / BLOCK_SIZE, 1, 1]`): 2097152
  programs for `BLOCK_SIZE = 1024`, 1048576 for 2048, 524288 for 4096, and
  262144 for 8192.
- `q22_cuda_group_kernel`: 2097152 blocks in x (`2^31 / 1024`, where the
  `int` row id turns negative) and 1 in y and z, in `launch.json`.
- `q08_cuda_q8_agg_kernel`: 2097151 blocks in x, one below `2^31 / 1024`,
  because its grid-stride loop also holds `stride = blockDim.x * gridDim.x`
  in an `int`, which is `2^31` (negative) at 2097152 blocks and would step
  the 64-bit row index below zero; and 1 in y and z.

Verdicts changed from `C` to `safe` for 78 kernels: both CUDA kernels and
the Triton kernels of q03 (8), q06 (8), q09 (8), q10 (8), q12 (8), q14 (8),
q17 (4), q19 (8), q21 (8), and q22 `avg_kernel` (8), whose every access is
at the masked offsets, at an index with an index role, or at a constant
index of a small output.

Verdicts changed from `C` to `E` for 24 kernels, whose remaining unmasked
atomic needs a data precondition that the gate cannot establish: q01 (4,
category codes below 256), q07 (8, 0/1 nation codes), q13 (4, order
customer keys inside `[0, max(c_custkey)]`), and q22 `agg_kernel` (8, kept
rows' country codes below 100).

12 Triton kernels stay `C` with a claim, for a defect that does not depend
on the grid: the masked lanes of q04 `agg_kernel` (4), q04 `bitmap_kernel`
(4), and q05 `agg_q5_kernel` (4) perform an unmasked atomic at index 0 of an
output whose size symbol has the least value 0 (`num_priorities`,
`bitmap_size`, `num_nations`), outside it when that size is 0 and a launch
has a program, for example `n_elements = 0` with one program, which the
claim allows; the q04 kernels also index `bitmap_ptr` by a raw order key.
The earlier review named this masked-lane write for q04 `agg_kernel` and
q05 but not for q04 `bitmap_kernel`, whose `tl.atomic_max(bitmap_ptr +
orderkey, 1)` is unmasked in the same way.

No kernel needs a grid bound that depends on a size symbol.

## Core plans

The held-out set: the `core` plan of each query whose `results.json` has
`"correct": true`, at its `output_round`, under
`logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/<toolchain>/core/queries/<query>/`,
extracted after the gate's rules froze, into task directories named
`<query>_core_<toolchain>_<kernel>/`. The `full` plans above are untouched.

| Toolchain | Plans (query: round) |
| --- | --- |
| CUDA (21) | q01: 1, q02: 1, q03: 1, q04: 2, q05: 1, q06: 1, q07: 1, q08: 1, q09: 1, q10: 2, q11: 1, q12: 2, q13: 1, q15: 1, q16: 2, q17: 1, q18: 2, q19: 1, q20: 1, q21: 3, q22: 3 |
| Triton (18) | q01: 7, q02: 4, q03: 6, q04: 1, q05: 2, q06: 2, q07: 6, q09: 4, q10: 2, q11: 3, q12: 10, q13: 4, q14: 3, q15: 10, q17: 6, q18: 1, q19: 2, q21: 2 |

The CUDA plan of q14 and the Triton plans of q08, q16, q20, and q22 did not
pass DataKernelBench's correctness check and are not taken.

### Plans that launch no kernel of their own

- CUDA q18: `torch.unique` and `scatter_add_` only; the plan has no CUDA
  source.

### Kernels extracted

106 kernels from the other 38 plans: 24 CUDA kernels (q04, q17, q20, and q22
launch two each) and 82 Triton kernels. No kernel is left out: every kernel
compiles through the corpus front ends and its binding builds.

A Triton kernel gives one corpus kernel per autotuner configuration, the
second and later with the suffixes `_2`, `_3`, and so on, in the order of the
plan's configuration list: q01, q04, q05, q14, q17 (both kernels), and q18 have
4 configurations (`BLOCK_SIZE` in 1024, 2048, 4096, 8192, or 2048 to 16384
for q14); q10, q11, and q21 have 8 (four `BLOCK_SIZE` values and `num_warps`
in 4, 8); q03, q06, and q12 list each `BLOCK_SIZE` and `num_warps` twice, with
`num_stages` 2 and 4, and the two give one corpus kernel (8 each), since a
corpus module states no `num_stages` and none of these kernels has a loop that
software pipelining would change. q02, q07, q09, q13, q15, and q19 launch one
configuration: a `BLOCK_SIZE` passed at the launch (256, 8192, 128, 1024) or a
`constexpr` default in the kernel's parameter list (8192 for q07 and q13,
kept as written), with the default `num_warps = 4`.

Extraction decisions beyond the literal list of edits, each recorded in the
kernels' `NOTES.md`:

- `<cstdint>`. Eight CUDA plans (q02, q08, q09, q11, q16, q20 both kernels,
  and q22 both kernels) declare their fixed-width integer types only through
  `<torch/extension.h>` (q08 also through `<c10/cuda/CUDAStream.h>`), so
  `kernel.cu` includes `<cstdint>` in its place. q07 and q21 include
  `<cstdint>` themselves, and it is copied.
- CUDA declarations keep the comments of the plans' parameter lists (q03,
  q07, q12, q16, q17 `q17_pass2`): the corpus's reader skips them now.
- q12 `q12_kernel` is already declared `extern "C" __global__` by the plan;
  the declaration keeps that and renames it.
- q20 `pack_key_kernel` is launched twice, on the filtered lineitem and on
  the filtered partsupp, in the same configuration: one corpus kernel, whose
  `N` is the rows of either launch.
- Two Triton plans annotate parameters (q05, pointer parameters with
  `tl.pointer_type(...)`; q12, scalars with `tl.int32`); the annotations are
  kept, and Triton compiles them with the corpus's `TYPES`.
- q15 `_noop_kernel` takes one scalar and no buffer, and makes no memory
  access; its signature has one argument and no size other than that scalar.
- q19 `_revenue_kernel` writes one entry per program into an output of
  `(n_elements + 1023) / 1024` entries, a quotient the adapter maps to the
  size symbol `out_ptr`.

Every CUDA launch has 1024 threads per block (`BLOCK_SIZE = 1024`); q22
`avg_kernel` passes 16384 bytes of dynamic shared memory, stated in its
`launch.json`.

### Grid claims

Stated with the extraction, before the review, as the README requires: the
largest grid for which every access stays inside its buffer.

- 81 Triton kernels compute `offs = pid * BLOCK_SIZE + tl.arange(0,
  BLOCK_SIZE)` in 32 bits (the PTX shifts `%ctaid.x` and compares with
  `setp.lt.s32`) and got `"grid": [2^31 / BLOCK_SIZE, 1, 1]` in
  `triton.json`: 8388608 programs for `BLOCK_SIZE = 256` (q02), 2097152 for 1024, 1048576 for 2048, 524288 for 4096, 262144 for
  8192, and 131072 for 16384 (q14 `_4`). The 1 in y and z also keeps the
  compaction of q03 from repeating rows. q15 `_noop_kernel` makes no access
  and states no claim.
- Three CUDA kernels hold `blockIdx.x * blockDim.x + threadIdx.x` in an
  `int`: q01 `q1_kernel`, q22 `avg_kernel`, and q22 `agg_kernel` got
  `"grid": [2097152, 1, 1]` (`2^31 / 1024`, where the row id turns negative)
  in `launch.json`. The other 21 CUDA kernels hold the row id in an
  `int64_t` and need no claim.
- q19 `_revenue_kernel` stores at `out_ptr + pid`, inside its output only
  for a grid of at most `cdiv(n_elements, 1024)` programs, a bound that
  depends on a size symbol and that the claim language cannot state; its
  claim is the offset bound above, and the store is reviewed under it.

### Review

`verdicts.tsv` classifies each kernel under its claim, from its source and
plan, with the facts that the gate can establish (buffer lengths, scalar
ranges, and the value range of an index input), any grid the claim allows,
and unknown output contents on entry.

| Toolchain | C | E | safe |
| --- | --- | --- | --- |
| CUDA | 0 | 9 | 15 |
| Triton | 13 | 18 | 51 |
| All | 13 | 27 | 66 |

- C (13 Triton kernels). Masked lanes of an unmasked atomic write index 0 of
  an output whose size symbol has the least value 0: q04 `_bincount_kernel`
  (4, `num_priorities`), q05 `_q5_agg_kernel` (4, `num_nations`), and q17
  `_agg_kernel` (4, `n_parts`), reachable with `n = 0` and one program, a
  launch the claim allows. q19 `_revenue_kernel` (1) stores at `out_ptr +
  pid` into an output of `cdiv(n_elements, 1024)` entries, and the gate does
  not establish the grid.
- E (9 CUDA, 18 Triton). The access outside the buffer needs a precondition
  of the plan that the gate cannot establish: category codes below 256 (q01,
  both toolchains, 5 kernels), 0/1 nation codes with no 2-entry buffer (q07,
  both toolchains, 2), a raw key bounded only by an output sized from the
  keys' maximum (q04 CUDA both kernels, q10 CUDA, q13 both toolchains, q18
  Triton 4: 9), the year range of a column against `min_year` and
  `num_years` (q08 CUDA, 1), the value range of `lower_idx` and the relation
  `n_groups <= li_n` (q20 `ps_mask_kernel`, 1), an exact match of every l1
  order key in the aggregated table (q21 CUDA, 1), and the compaction
  counter's contents on entry (q03 Triton, 8).
- safe (15 CUDA, 51 Triton): every access is at a masked or bounded row
  offset, at an index with an index role (q04 `order_agg_kernel` priorities,
  q05, q12 CUDA and Triton shipmode codes into 2-entry outputs, q16 group
  ids, q21 supplier ids, q10 inverse indices, q17 `_filter_sum_kernel` part
  indices), at a constant index of a small output, inside a binary search
  that keeps its bounds whatever the keys' order (q15, q17 CUDA, q22
  `agg_kernel`), or inside a shared array sized for the 1024-thread block.

No kernel is left out.
