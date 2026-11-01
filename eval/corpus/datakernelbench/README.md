# DataKernelBench operator corpus

The kernels of this collection were written by seven model families from
outside Anthropic for [DataKernelBench](https://github.com/kerneldf/datakernelbench),
a benchmark that asks models to rewrite the 22 TPC-H queries as GPU kernel
plans. The models knew nothing of OpGate. The plan files and the kernels
taken from them are DataKernelBench's, under the Apache License 2.0
([LICENSE](LICENSE)). The collection tests the gate on
kernels that neither the gate's authors nor the models of the synthesized
collection wrote, for tasks that the benchmark chose.

## Selection

From DataKernelBench at commit `43c361d`, for each model family below and
each query `q01` to `q22`, the plans of each toolchain, `cuda` and `triton`,
that DataKernelBench's correctness check passed: the file
`logs/kernel_plans/tpch/sf10/h100/<family>/<toolchain>/<level>/queries/<query>/round_<k>.py`,
where the query's `results.json` has `"correct": true` and `k` is its
`output_round`. Every kernel that such a plan launches is a corpus kernel.
DataKernelBench asks each model for two plans of a query: `full`, the whole
query as a kernel plan, and `core`, only its tensor-intensive core. The
`full` plans were extracted first, and the gate's rules were revised with
them in view (design decision 33 and later); the `core` plans, written by
the same models for the same queries, were extracted after those rules
froze and were held out when added. Later rules were tuned with both plan
levels in view, so the reports count the whole collection in the development
set.

| Model | DataKernelBench family |
| --- | --- |
| `deepseek-v4-flash` | `deepseek-ai__deepseek-v4-flash` |
| `devstral-2-123b` | `mistralai__devstral-2-123b-instruct-2512` |
| `gemini-3.1-pro` | `gcp__gemini-3.1-pro-preview` |
| `gpt-5.5` | `azure__gpt-5.5` |
| `gpt-oss-120b` | `openai__gpt-oss-120b` |
| `minimax-m2.5` | `minimaxai__minimax-m2.5` |
| `qwen3.5-397b` | `qwen__qwen3.5-397b-a17b-fp8` |

## Layout

Each kernel is a task directory `<model>/<query>_<toolchain>_<kernel>/` for
a `full` plan and `<model>/<query>_core_<toolchain>_<kernel>/` for a `core`
plan, for example `gpt-5.5/q06_cuda_q6_kernel/` and
`gpt-5.5/q06_core_triton__q6_kernel/`, where `<kernel>` is the kernel's name
in the plan. A plan that launches one kernel in more than one configuration
(template arguments, block, dynamic shared memory, Triton `constexpr` values
or `num_warps`, or autotuner configurations) gives one corpus kernel per
configuration, the second and later with the suffix `_2`, `_3`, and so on.

- `kernel.cu` or `triton.py`: the kernel as the plan states it (below).
- `plan.py`: a copy of the plan file the kernel comes from.
- `signature.json`: each kernel argument, in order, in the format of the
  synthesized collection (`name`, `kind`, `dtype`, `length`,
  `indexes_into`), stating what the plan establishes when it launches the
  kernel.
- `launch.json` (CUDA only): the block, the dynamic shared memory, and the
  grid maxima (`"grid": [x, y, z]`, when the kernel is correct only up to a
  grid, such as `2^31 / blockDim` blocks for a 32-bit row index) of the
  plan's launch, `{"nvcc": {"block": [x, y, z], "shared": {"constant": c,
  "terms": {size: coefficient}}, "size_maxima": {}}}`.
- `NOTES.md`: the extraction edits, where the plan launches the kernel, and
  how the plan establishes each length and range.
- `kernel.O1.ptx`, or `triton.ptx` and `triton.json`: the PTX of the front
  end, written by `uv run python -m eval.corpus.sources nvcc|triton <model>`;
  `triton.json` states the dynamic shared memory of the launch and, when the
  kernel is correct only up to a grid, its grid maxima (`"grid": [x, y, z]`,
  `2^31 / BLOCK` programs for an offset `pid * BLOCK` computed in 32 bits).

`<model>/roles.tsv` holds the reviewed roles of each argument, in the format
of the synthesized collection, and `<model>/verdicts.tsv` the review of each
kernel (below). `decisions.tsv` holds the gate's decisions.

## Extraction

The kernel's source is the plan's, with only these edits, each recorded in
the kernel's `NOTES.md`:

- CUDA: the entry kernel is named `kernel` and declared
  `extern "C" __global__ void kernel(...)`; a template is instantiated with
  the plan's arguments; the includes, macros, constants, types, and device
  functions that the kernel uses are copied unchanged; host code is left out.
- Triton: the entry kernel is named `kernel`; the `@triton.jit` helpers and
  module constants that it uses are copied unchanged; an autotuner decorator
  is replaced by the configuration's values. The module defines `TYPES` (each
  argument that is not a `tl.constexpr`, with the Triton type that the plan's
  value has: the tensor's element type for a pointer, `i32` for an integer
  below 2^31 at scale factor 10 and `i64` otherwise, `fp64` or `fp32` for a
  float), `CONSTANTS` (each `tl.constexpr` argument's value at the plan's
  launch), and `NUM_WARPS` (the launch's `num_warps`, 4 when it states none).

The kernel's accesses are not edited: no guard, bound, index, or type of a
memory access changes.

A kernel that is correct only up to a grid states that grid in its claim
(`"grid"` in `triton.json` or `launch.json`, design decision 33): the
largest grid for which every access stays inside its buffer, such as
`2^31 / BLOCK` programs for a Triton offset `pid * BLOCK + tl.arange(...)`
computed in 32 bits, `2^31 / blockDim` blocks for a row index kept in an
`int`, or 1 in the dimensions the kernel ignores when an atomic append
would repeat rows. The review of the kernel is made under its claim.

The extraction met four cases that the list above does not name, each
recorded in the kernel's `NOTES.md`:

- A CUDA plan that takes `uint8_t` or `uint32_t` from `<torch/extension.h>`,
  which nvcc cannot include without PyTorch's headers, includes `<cstdint>`
  in its place (4 kernels); a CUDA source leaves out the includes that only
  its host code uses.
- The corpus's reader of a CUDA prototype did not skip comments when the
  gpt-oss-120b kernels were extracted, so six of them state their parameter
  lists without the comments inside them; the parameters, their order,
  types, and qualifiers are the plan's. The reader skips comments now.
- An autotuner that lists each block size and warp count with several
  `num_stages` gives one corpus kernel per block size and warp count: the
  front end states no `num_stages`, and none of these kernels has a loop that
  software pipelining changes, so their PTX is the same (checked for
  deepseek-v4-flash and gemini-3.1-pro).
- A Triton `tl.constexpr` whose value the plan takes from the data, such as a
  join's row count, is not known at the launch, so its kernel is left out (4
  gpt-5.5 kernels); one that TPC-H fixes at scale factor 10, such as the
  number of nations, takes that value.

A plan can launch no kernel of its own (it calls only cuDF, PyTorch, or
Thrust), or define kernels that it never launches; neither gives a corpus
kernel. Two qwen3.5-397b CUDA kernels that nvcc refuses (a signed 64-bit
`atomicAdd`, which the plan's own build also failed on) and one gpt-5.5
Triton kernel that Triton 3.8 refuses are left out. Each model's `NOTES.md`
lists its plans, its kernels, and every kernel left out, with the reason.

A plan's kernel that has several configurations is several corpus kernels,
one per configuration, each decided by the gate; `make export` reports both
the corpus kernels and the plans' kernels, a plan's kernel being accepted
when the gate accepts every configuration of it.

## Review

The review classifies each kernel against the plan's contract before the
gate decides it, with the facts that the gate can establish: the buffers'
lengths, the scalars' ranges, and the value range of an index or offsets
input. The gate does not establish the contents of an output on entry,
relations between buffers (sorted, distinct, or increasing values, or a
prefix sum), or the grid: a launch can have any grid. The verdicts are those
of the synthesized collection:

- `C`: a memory access outside its buffer is reachable with inputs that meet
  the plan's contract and a launch that the signature allows.
- `E`: a memory access outside its buffer is reachable only when a
  precondition of the plan that the gate cannot establish is violated.
- `safe`: every memory access stays inside its buffer under every fact that
  the gate can establish.

`<model>/verdicts.tsv` has the columns `model`, `task`, `toolchain`,
`verdict`, and `evidence`: for `C` and `E`, the input or launch that reaches
the access; for `safe`, why each access stays inside its buffer.

## Label corrections of 2026-10-01

A second review of 60 sampled labels
(`eval/records/label_agreement_2026-10-01.json`) found two patterns that the
verdicts above label differently: deepseek-v4-flash labeled `C` a write that
needs an output counter to be nonzero on entry, and gpt-oss-120b labeled `C`
an access outside an output of size 0 where the plan never launches with
that size. Each row of both patterns was reviewed again from the kernel's
source and plan, and these `verdicts.tsv` rows changed (the models'
`NOTES.md` keep the counts of the first review):

| Model | Task | Toolchain | Old | New | Reason |
| --- | --- | --- | --- | --- | --- |
| `deepseek-v4-flash` | `q15_core_cuda_filter_kernel` | nvcc | `C` | `E` | The one access outside a buffer needs the output counter to be nonzero on entry; the plan zeroes it just before the launch, a precondition that the gate cannot establish. |
| `deepseek-v4-flash` | `q16_cuda_build_composite_keys_kernel` | nvcc | `C` | `E` | The one access outside a buffer needs the output counter to be nonzero on entry; the plan zeroes it just before the launch, a precondition that the gate cannot establish. |
| `deepseek-v4-flash` | `q16_cuda_filter_partsupp_kernel` | nvcc | `C` | `E` | The one access outside a buffer needs the output counter to be nonzero on entry; the plan zeroes it just before the launch, a precondition that the gate cannot establish. |
| `deepseek-v4-flash` | `q18_core_cuda_q18_threshold_kernel` | nvcc | `C` | `E` | The one access outside a buffer needs the output counter to be nonzero on entry; the plan zeroes it just before the launch, a precondition that the gate cannot establish. |
| `deepseek-v4-flash` | `q20_core_cuda_q20_filter_kernel` | nvcc | `C` | `E` | The one access outside a buffer needs the output counter to be nonzero on entry; the plan zeroes it just before the launch, a precondition that the gate cannot establish. |
| `deepseek-v4-flash` | `q20_core_cuda_q20_isin_kernel` | nvcc | `C` | `E` | The one access outside a buffer needs the output counter to be nonzero on entry; the plan zeroes it just before the launch, a precondition that the gate cannot establish. |
| `deepseek-v4-flash` | `q22_cuda_q22_filter_kernel` | nvcc | `C` | `E` | The one access outside a buffer needs the output counter to be nonzero on entry; the plan zeroes it just before the launch, a precondition that the gate cannot establish. |
| `deepseek-v4-flash` | `q22_core_cuda_q22_filter_kernel` | nvcc | `C` | `C` | Label kept, evidence corrected: the counter on entry is a precondition, but the kernel lacks a grid claim and ignores `blockIdx.y` and `blockIdx.z`, so the grid (1, 65535, 65535) repeats one kept row more than 2^31 times, the 32-bit counter wraps, and a negative position passes the `pos < n` guard. |
| `gpt-oss-120b` | `q04_triton__bitmap_kernel` | triton | `C` | `E` | `bitmap_size = 0` needs a largest order key of -1 (the plan takes `torch.max` of the keys, which raises on an empty table), against the non-negative keys that the kept rows' atomic also needs. |
| `gpt-oss-120b` | `q04_triton__bitmap_kernel_2` | triton | `C` | `E` | `bitmap_size = 0` needs a largest order key of -1 (the plan takes `torch.max` of the keys, which raises on an empty table), against the non-negative keys that the kept rows' atomic also needs. |
| `gpt-oss-120b` | `q04_triton__bitmap_kernel_3` | triton | `C` | `E` | `bitmap_size = 0` needs a largest order key of -1 (the plan takes `torch.max` of the keys, which raises on an empty table), against the non-negative keys that the kept rows' atomic also needs. |
| `gpt-oss-120b` | `q04_triton__bitmap_kernel_4` | triton | `C` | `E` | `bitmap_size = 0` needs a largest order key of -1 (the plan takes `torch.max` of the keys, which raises on an empty table), against the non-negative keys that the kept rows' atomic also needs. |
| `gpt-oss-120b` | `q04_triton__agg_kernel` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan never launches on an empty orders table; the bitmap load also needs the non-negative keys. |
| `gpt-oss-120b` | `q04_triton__agg_kernel_2` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan never launches on an empty orders table; the bitmap load also needs the non-negative keys. |
| `gpt-oss-120b` | `q04_triton__agg_kernel_3` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan never launches on an empty orders table; the bitmap load also needs the non-negative keys. |
| `gpt-oss-120b` | `q04_triton__agg_kernel_4` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan never launches on an empty orders table; the bitmap load also needs the non-negative keys. |
| `gpt-oss-120b` | `q04_core_triton__bincount_kernel` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan returns before the launch when `n == 0`. |
| `gpt-oss-120b` | `q04_core_triton__bincount_kernel_2` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan returns before the launch when `n == 0`. |
| `gpt-oss-120b` | `q04_core_triton__bincount_kernel_3` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan returns before the launch when `n == 0`. |
| `gpt-oss-120b` | `q04_core_triton__bincount_kernel_4` | triton | `C` | `E` | `num_priorities = 0` needs `n_elements = 0` by the index role, and the plan returns before the launch when `n == 0`. |
| `gpt-oss-120b` | `q17_core_triton__agg_kernel` | triton | `C` | `E` | `n_parts = 0` needs `n_rows = 0` by the index role, and the plan returns before the launch when `sel_idx` is empty. |
| `gpt-oss-120b` | `q17_core_triton__agg_kernel_2` | triton | `C` | `E` | `n_parts = 0` needs `n_rows = 0` by the index role, and the plan returns before the launch when `sel_idx` is empty. |
| `gpt-oss-120b` | `q17_core_triton__agg_kernel_3` | triton | `C` | `E` | `n_parts = 0` needs `n_rows = 0` by the index role, and the plan returns before the launch when `sel_idx` is empty. |
| `gpt-oss-120b` | `q17_core_triton__agg_kernel_4` | triton | `C` | `E` | `n_parts = 0` needs `n_rows = 0` by the index role, and the plan returns before the launch when `sel_idx` is empty. |

The 8 gpt-oss-120b Triton kernels of q05 (`q05_triton__agg_q5_kernel` and
`q05_core_triton__q5_agg_kernel`, 4 configurations each) stay `C`: their
plans call the kernel even when the region has zero nations, with
`num_nations = 0`, and a launch can have any grid up to the claim, so one
program adds at `out_ptr[0]` of the empty output.

## Contract corrections of 2026-10-02

A classification of the safe kernels that the frozen gate refuses with "the
access is not proved inside its buffer" found two contracts that state less
than their plans establish. Each was checked against the plan and corrected
in the kernel's contract files; the verdicts stay `safe`:

| Model | Task | Files | Old | New | Reason |
| --- | --- | --- | --- | --- | --- |
| `gpt-5.5` | `q17_core_cuda_q17_init_kernel` | `signature.json`, `NOTES.md` | `n_init` a scalar of any value | `n_init` in `[0, 2147483647]` | The plan sets `n_init = max(hash_size, n_parts)`, the maximum of two size symbols that the contract bounds by 2^31 - 1, so the grid-stride index `i < n_init` cannot pass 2^63 and wrap to a negative index that also passes `i < hash_size`. |
| `gpt-5.5` | `q18_core_cuda_add_block_offsets_kernel` | `roles.tsv`, `NOTES.md` | `block_offsets` an independent size symbol | `block_offsets = (n + 1023) // 1024`, a derived size | The plan allocates `block_offsets` with `scan_blocks = (n + 1023) / 1024` elements and launches `scan_blocks` blocks of 1024 threads, so `blockIdx.x < block_offsets` under `gid < n`. The `derived` target of the symbol row states the relation (`eval/corpus/adapter.py`); `eval/records/derived_size_contract_2026-09-30.json` decided the same PTX with this relation. |

## Contract corrections of 2026-10-03

A classification of the safe kernels that the gate refuses for a reason
other than termination, participation, or the time limit found one contract
that states less than the kernel's source does. It was checked against the
source and corrected in the kernel's contract files; the verdict stays
`safe`:

| Model | Task | Files | Old | New | Reason |
| --- | --- | --- | --- | --- | --- |
| `gpt-5.5` | `q05_core_cuda_q5_sort_kernel` | `roles.tsv`, `signature.json`, `NOTES.md` | `sorted_ids` written only | `sorted_ids` read and written | The insertion sort reads the identifiers back, `int64_t key_id = sorted_ids[i];` and `sorted_ids[j] > key_id` in `kernel.cu`, after the copy loop writes them, so the gate refused the read `ld.global.u64` for the buffer's direction. |
