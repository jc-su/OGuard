# A generated database kernel faults a shared worker

Recorded 2026-09-27. This experiment connects the Dask worker experiment to a public, independently
generated database kernel. It does not replace or modify either Dask worker experiment.

The source is the public [DataKernelBench
artifact](https://github.com/kerneldf/datakernelbench) at commit
`43c361d4ed789d1feaffbcee750e5b256590571a`. DataKernelBench asks models to
turn TPC-H SQL and a validated cuDF/PyTorch TorchPlan into optimized CUDA or
Triton modules, then executes the result. Its paper appeared in August 2026
and evaluates ten proprietary and open-weight models over all 22 TPC-H queries
at SF10 on an H100. This is the same emerging workload class as OpGate:
query-specific GPU code generated from an ordinary database request.

The executable audit and replay harness is
[`generated_kernel.py`](generated_kernel.py). The successful complete result
is in
`records/e6_datakernelbench_2026-09-27.json` (`../records/e6_datakernelbench_2026-09-27.json`, not included in this archive),
and the standalone GPU-independent audit is in
`records/e6_datakernelbench_audit_2026-09-27.json` (`../records/e6_datakernelbench_audit_2026-09-27.json`, not included in this archive).

## What is independent

The selected module is DataKernelBench's first answer from
`aws/claude-opus-4-7` for TPC-H Q3 in its Triton/core setting. The upstream
run began on 2026-05-24. Its input was a normal Q3 SQL optimization request,
the validated Q3 TorchPlan, the data and H100 descriptions, and the framework
instructions. There was no attack prompt or request to access unrelated data.

The harness verifies all of the following before reading or executing the
candidate:

- the upstream Git commit;
- the base, environment, SF10, H100, Triton, and core-level prompt components;
- the Q3 SQL and reference TorchPlan;
- the run metadata naming Claude Opus 4.7, Triton, core, TPC-H, and SF10;
- the generated `round_01.py`, its per-query log, and its `results.json`.

Every reviewed file is compared byte for byte with its contents at the pinned
Git revision.
The harness imports this file directly and never copies or edits its kernel.

## Artifact-wide audit

The audit enumerates the Cartesian product of ten models, CUDA and Triton,
core and full-query specialization, and 22 TPC-H queries: 880 runs. It fails
if a model, setting, query, result, source contents, or expected missing record changes.
The upstream artifact contains 878 `results.json` files. Its own analysis
imputes the two absent files as worker crashes, producing the complete 880-run
matrix.

| First-round outcome | Count | Denominator |
| --- | ---: | ---: |
| Correct module | 385 | 880 |
| Runtime error or worker crash | 213 | 880 |
| Compile or import error | 154 | 880 |
| Wrong values or row count | 36 | 880 |
| LLM or code-extraction failure | 92 | 880 |
| Explicit `illegal memory access` record, spanning six model IDs | 10 | 880 |

The categories in the first five rows are disjoint and total 880. The ten
illegal-address records are a subset of the 213 runtime errors and crashes.
Across every repair round, the artifact contains 34 explicit illegal-address
records, 26 out-of-memory records, and three device-assert records. Eventually
682 of 880 runs pass the correctness and speed threshold, and 729 produce a
correct module. Those eventual successes do not make first execution safe: a
faulting first answer can destroy the context before a repair request runs.

DataKernelBench handles that consequence operationally. Its default
`isolate_queries=true` launches every query in a new subprocess, described in
the repository as giving each query "fresh CUDA after bad kernels." Its
generation loop stops repair after detecting a CUDA error because the context
is poisoned. These choices are evidence from another project that generated
kernels need process isolation today; they are not an OpGate assumption.

## Independent scale-change corroboration

The July 2026 paper [*From Custom-Fit to Portable: Bridging the Gap Between
Synthesized and Engineered GPU Query
Execution*](https://arxiv.org/abs/2607.07632) supplies a different normal
trigger. Its SHADB system uses Claude Opus 4.8 in an autonomous,
profile-guided loop to generate self-contained CUDA or HIP database programs.
The authors report that one generated Q4.2 candidate compiled but hit a CUDA
illegal-memory-access error during ordinary SF100 execution. More broadly,
when they ran code generated at SF10 unchanged at SF100, only 2 of 13 queries
remained correct: five crashed on out-of-bounds accesses and six silently
returned wrong results.

This matters because the trigger is data growth, not an attack prompt. A
generated operator can pass its original validation and become unsafe when a
tenant later runs the same query at another supported scale. That is a direct
motivation for OpGate's symbolic, all-size proof obligation.

We inspected the public [SHADB
repository](https://gitlab.eurecom.fr/dislab/SHADB) at commit
`20438bda62cdd74dead2cf33dfdfe1d3a34b49c2`. Its default tree contains the
synthesis framework but not the paper's 13 generated candidates or their run
records. Therefore these numbers remain external paper evidence; they are not
reported as an OpGate reproduction. DataKernelBench is the reproducible,
file-level independent audit in E6a.

## The selected defect

The Q3 module creates four output tensors of length `n` and a scalar counter.
Each block reserves an output range and uses it for stream compaction:

```python
n_keep = tl.sum(keep_i, axis=0)
base = tl.atomic_add(counter_ptr, n_keep)
write_idx = base + pos
tl.store(out_lk_ptr + write_idx, lk, mask=keep)
```

The model also decorated this stateful kernel with 12 autotuning
configurations. It supplied neither `reset_to_zero=["counter_ptr"]` nor a
restore hook. [Triton's `autotune`
documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
states that all configurations run, values updated by the kernel are therefore
updated repeatedly, and `reset_to_zero` is the mechanism for avoiding that
behavior.

For a launch in which `n` rows pass the predicate, the first evaluated
configuration advances the counter to `n`. The next configuration begins at
`n` and stores beyond the four `n`-element outputs. Repeated benchmarking
advances it farther. For the replay's 1,000,003 valid rows, even the conservative
assumption of one launch per configuration gives the second configuration a
write range of 1,000,003 through 2,000,005, outside the allocated range 0
through 1,000,002. Twelve single launches reach index 12,000,035; Triton's
benchmarking can execute each configuration more than once. DataKernelBench's
unchanged H100 run reports:

```text
Triton Error [CUDA]: an illegal memory access was encountered
```

Its traceback places the failure inside Triton's autotuner while it launches
`_q3_filter_kernel`. This is a realistic synthesis defect: the individual
kernel's bounds mask looks correct, but mutable launch state makes later
autotuning executions unsafe. It is exactly the kind of multi-launch reasoning
that result comparison after execution cannot provide.

## Shared-worker replay

The `blast-radius` case makes the consequence concrete in the same real stack
as the Dask worker experiment:

1. One `dask-cuda.LocalCUDACluster` worker owns one GPU, CUDA context, and RMM
   pool.
2. Two separately named Dask clients represent service-layer principals A and
   B. B persists a live cuDF column and successfully reduces it.
3. A submits the commit-pinned upstream Q3 module after its source and static
   defect witnesses are checked directly. The task receives no Dask
   future and has no dependency on B's column.
4. The module receives 1,000,003 well-typed rows satisfying Q3's two date
   predicates. It calls its public `_query_core`; the generated source is
   unchanged.
5. After A's task, B repeats the same cuDF reduction and synchronizes the
   worker context.

The replay ran successfully on an NVIDIA RTX A6000 (compute capability 8.6)
with driver 610.57.04. The worker used Python 3.12.9, cuDF 26.08.01, RMM and
dask-cuda 26.08.00, Dask and Distributed 2026.7.1, PyTorch 2.14.0+cu130, and
Triton 3.8.0. This also reproduces on a different GPU and Triton version from
the upstream H100/Triton 3.7 run while keeping the generated module unchanged.

B's initial `Series.sum()` returned 1,048,576. The scheduler recorded an empty
dependency set for A's generated-kernel task. Both clients' pre-fault work ran
in the same worker and PID. A's task then raised `Triton Error [CUDA]: an
illegal memory access was encountered`. B's repeated reduction failed inside
RMM with `cudaErrorIllegalAddress`, and B's direct context synchronization
failed with CUDA error 700. The complete record reports `success: true`.

NVIDIA defines an illegal address as leaving the process in an inconsistent
state in which later CUDA work returns the same error until the process is
relaunched. E6b observes precisely that documented consequence across the two
client workloads.

Run the audit without a GPU:

```console
.venv/bin/python eval/motivation/generated_kernel.py audit \
  --upstream /path/to/datakernelbench \
  --output eval/records/e6_datakernelbench_audit_2026-09-27.json
```

Run the complete experiment on a GPU:

```console
.venv/bin/python eval/motivation/generated_kernel.py all \
  --upstream /path/to/datakernelbench \
  --output eval/records/e6_datakernelbench_2026-09-27.json
```

## Claim boundary

E6a supports the claim that LLM-generated GPU database code is real and
timely, and that independently generated modules contain concrete memory faults
under ordinary SQL requests. E6b supports the additional claim that one such
unchanged module prevents a concurrent benign cuDF workload from continuing in
a shared worker context.

This experiment does not show that this Q3 defect deliberately selects or returns a victim's
secret. The Dask worker experiment establishes the confidentiality mechanism separately: an unchecked
in-pool access from A returns B's changing value. The unchecked corpus run shows that synthesized
defects can cross neighboring pooled buffers without an allocation-level
sanitizer error. The evidence composes as follows:

| Question | Evidence |
| --- | --- |
| Do systems really generate database GPU kernels from normal requests? | DataKernelBench's public SQL-to-CUDA/Triton pipeline, prompts, model IDs, and outputs |
| Do non-adversarial generated kernels actually access memory illegally? | The 880-run E6a audit and the selected upstream Q3 result |
| Does a fault affect another workload sharing the context? | The Dask worker experiment availability and the successful E6b replay with an independently generated module |
| Can one workload read another live pooled buffer? | The Dask worker experiment confidentiality, including a differential secret update and zero memcheck findings |
| Are all current Dask deployments exposed as hostile multi-tenant systems? | No. Dask documents arbitrary code execution and requires deployments to isolate and govern access; Dask Gateway normally provisions separate clusters |

The defensible paper scenario is a GPU analytics service that wants to
consolidate service-layer tenants or LLM-generated operators in one worker to
avoid the process-per-tenant cost measured by the alternatives measurement. Dask client names instantiate
that proposed layout; they are not security principals supplied by Dask. The
status quo avoids the risk through trust or separate processes and clusters.
OpGate's purpose is to make the consolidated layout safe before the generated
kernel's first execution.

## Paper-ready motivation

The following claim can be used now, based on the public artifact, E6a, and
E6b:

> A 2026 independent study generated CUDA and Triton implementations of all 22
> TPC-H queries from ordinary SQL requests using ten models. Our audit of its
> 880-run artifact finds ten first-round modules explicitly recorded as illegal
> GPU accesses. One Claude Opus Q3 module omitted the state reset required for
> its autotuned stream-compaction counter, so later configurations wrote beyond
> `n`-element outputs. The study stops repair after CUDA faults and runs each
> query in a fresh subprocess, preventing those candidates from sharing a live
> analytics context.

The introduction can then cite SHADB separately for the deployment-time
trigger: changing only scale factor caused five of 13 generated queries to
crash out of bounds and six more to return wrong answers.

> When we submitted that unchanged, commit-pinned module beside a benign cuDF
> workload in one dask-cuda worker, its task had no dependency on the benign
> column, yet the resulting illegal address made the benign reduction fail.

Do not say that DataKernelBench is multi-tenant, that Dask promises hostile
client isolation, or that the selected Q3 defect steals a chosen secret. Those
claims are unnecessary and are not established by this experiment.

## Primary sources

- [DataKernelBench paper](https://arxiv.org/abs/2608.25061)
- [DataKernelBench artifact](https://github.com/kerneldf/datakernelbench)
- [SHADB/custom-fit-to-portable paper](https://arxiv.org/abs/2607.07632)
- [SHADB synthesis framework](https://gitlab.eurecom.fr/dislab/SHADB)
- [Triton autotuning state-reset semantics](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- [cuDF custom Numba kernels and undefined behavior without a bounds guard](https://docs.rapids.ai/api/cudf/latest/user_guide/guide-to-udfs/)
- [Dask's arbitrary-code and isolation guidance](https://docs.dask.org/en/stable/faq.html#is-dask-secure)
- [Dask Gateway's per-user cluster architecture](https://gateway.dask.org/)
- [`LocalCUDACluster` one-GPU-per-process and RMM-pool API](https://docs.rapids.ai/api/dask-cuda/legacy/api/)
- [CUDA illegal-address process state](https://docs.nvidia.com/cuda/archive/10.0/cuda-driver-api/group__CUDA__TYPES.html)
