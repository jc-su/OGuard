# OpGate

OpGate lets the tenants of a GPU analytics engine share one CUDA context. A
checker proved sound in Rocq verifies each operator's PTX once for every data
size; an untrusted guard pass adds bounds checks and iteration limits that the
checker verifies; and a catalog checks every reservation, allocation, grant,
and launch. Two theorems prove that what a tenant observes depends only on its
steps and on the buffers granted to it.

This repository is the artifact of the paper *OpGate: Verified Isolation for
Untrusted Operators in a Shared GPU Analytics Engine* (PVLDB, Vol. 20).

## Contents

| Path | Contents |
| --- | --- |
| `opgate/` | the Python package: `signature`, `binding`, `catalog`, `gate`, `executor`, `frontends` (Numba, CUDA C++, Triton), and `hosts` (cuDF, PyTorch, and the multi-tenant service) |
| `formal/` | the Rocq development: the gate and its soundness proofs, the step check, the launch model, and the isolation theorems; `formal/README.md` describes each file |
| `eval/` | the code of every experiment, the two operator corpora, the baselines, and `eval/export/`, which turns experiment records into the paper's figures, tables, and numbers |
| `tests/` | unit, proof-regression, host, and GPU tests |
| `docs/` | `ARCHITECTURE.md` (code design), `LOWERING.md` (PTX idioms and the checker rules that cover them), `GLOSSARY.md` (vocabulary) |

## Requirements

Linux with an NVIDIA GPU, CUDA 13 with `nvcc`, `make`, `uv`, Rocq (Coq 8.15)
with `coqc`, and `libatomic.so.1`. The paper's measurements use one RTX A6000
(sm_86) with driver 610.57.04, CUDA 13.0, and Python 3.12.

## Quick start

```sh
uv sync --all-extras
make -C formal build      # build the proof library; it is published at formal/build/current
make test                 # the fast CPU tests
```

Collecting the tests needs a visible CUDA device, because the service's launch
path looks up driver functions when it is imported. `make test-cpu-full` adds
the whole-kernel proof regressions, and `make test-all` adds the GPU tests.
`make help` lists every target.

## Results and their Rocq names

| Paper | Rocq | File |
| --- | --- | --- |
| Theorem 4.1, tenant isolation | `tenant_isolation` | `formal/PipelineIsolation.v` |
| Theorem 4.2, declassification | `tenant_declassification` | `formal/PipelineGrants.v` |
| the gate's soundness theorem (Section 4.3) | `verify_certificate_sound` | `formal/GateParse.v` |
| the participation theorem (Section 4.3) | `verify_certificate_participation` | `formal/GateParticipation.v` |
| soundness of the termination check | `term_check_sound` | `formal/GateTerm.v` |
| soundness of the shared-value route | `shared_check_sound` | `formal/GateShared.v` |

`formal/README.md` lists the remaining lemmas and the assumptions of each
statement.

## Experiments

Each experiment is one `make` target that writes a record to `eval/records/`.
`make export` reads the records and writes the figures' data, the tables, and
`numbers.tex`. Timed targets run on an idle GPU pinned to one core, for
example `make timed GPU=0 CORE=8`. A target that reads an earlier record takes
it through a `make` variable, such as `REGISTRATION=eval/records/<record>.json`.

| Paper | Target |
| --- | --- |
| Section 1, motivation | `motivation-cudf-udf`, `motivation-dask-worker`, `motivation-generated-kernel`, `motivation-index-intermediates` |
| Section 6.2, Table 1: leakage attacks | `rq1-leakage` |
| Section 6.2, Table 1: model-written defects | `rq1-corpus-ptx`, `rq1-guarded-defects` |
| Section 6.2: service handles and the generated Q3 module | `rq1-service-boundary`, `rq1-generated-ptx` |
| Section 6.2, Figure 5: fault isolation | `rq4-fault-isolation` |
| Section 6.3, Figure 6: throughput and device memory | `rq4-consolidation` |
| Section 6.3: mixed operators | `rq4-mixed` |
| Section 6.4, Figure 7: acceptance as written and with checks | `rq4-registration`, `rq2-guarded` |
| Section 6.4: cuDF's UDF guide, changed kernels, rule ablation | `rq2-cudf-udfs`, `rq2-differential`, `rq2-rule-ablation` |
| Section 6.5, Figure 9: device time | `rq4-guard-cost` |
| Section 6.5, Table 2: registration and cold queries | `rq4-registration`, `rq4-generation`, `rq4-cold-query` |
| Section 6.5, Figure 8 and steps and queries | `rq4-step-check`, `rq4-clearing`, `rq4-query-overhead`, `rq3-index-path`, `rq3-tpch-queries`, `rq3-torch-queries` |
| Section 5: the Python step check against Rocq | `rq1-trace-replay` |

## Corpora

`eval/corpus/synthesized/` holds the operators that four Claude models wrote in
Numba, CUDA C++, and Triton for the task lists `TASKS.md`, `TRITON.md`, and
`HELDOUT.md` to `HELDOUT5.md`. The paper's new kernels are the task lists
`HELDOUT4.md` and `HELDOUT5.md`, written after a freeze of the checker. Each
model's `roles.tsv` and `verdicts.tsv` hold the reviewed argument roles and
labels. `eval/corpus/datakernelbench/` holds the kernels of DataKernelBench's
TPC-H plans from seven model families, under the benchmark's Apache License
2.0.

The corpus files named the paths of the machine where the models wrote and
compiled them. In the files of `eval/corpus/`, the archive replaces the
repository's path by `/opgate`, the models' scratch directories by `/scratch`,
and the home directory by `$HOME`. The PTX's `.file` lines are debug
information; the gate reads no path from them. A Markdown link to a file that
the artifact leaves out, such as an experiment record, is rendered as its path.
