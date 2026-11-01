# Tenants on one GPU through the service

This is engineering evidence for the RQ4 design, not a paper result. The
complete raw record is
`records/service_consolidation_2026-09-29.json` (`../records/service_consolidation_2026-09-29.json`, not included in this archive)
(38,104,454 bytes). It contains the source bytes used by the run, the submitted PTX,
every latency sample, per-tenant completions, service registration evidence,
and final cleanup telemetry.

A later primary-source audit found that driver r610 adds static MPS SM
partitions with partial containment for SM-triggered faults. This machine runs
r610.57 and exposes 21 four-SM chunks, so the benchmark now includes that mode.
Its separate complete raw pilot is
`records/service_consolidation_static_mps_2026-09-29.json` (`../records/service_consolidation_static_mps_2026-09-29.json`, not included in this archive)
(16,172,595 bytes).
Keeping it separate preserves the original 24-cell record and makes clear that
these are not randomized paired repetitions from one frozen revision.

The experiment runs the same recorded Sonnet TPC-H Q6 PTX and deterministic
input in three configurations. `service` sends requests from independent OS
clients through the restricted OpGate Unix-socket boundary to one context and
one shared table. `process` gives each tenant a process, context, executor, and
table. `mps` runs the process configuration under CUDA MPS. Every completed
request is compared with the NumPy reference before it is counted.

| configuration | SF | tenants | queries/s | p99 (us) | peak device memory (MiB) | Jain fairness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| service | 0.1 | 1 | 448.3 | 2409.9 | 357 | 1.0000 |
| service | 0.1 | 2 | 1416.6 | 2534.6 | 357 | 0.9780 |
| service | 0.1 | 4 | 1162.1 | 7789.9 | 357 | 0.9995 |
| service | 0.1 | 8 | 2067.8 | 4893.1 | 357 | 1.0000 |
| service | 1 | 1 | 452.2 | 2710.3 | 541 | 1.0000 |
| service | 1 | 2 | 1657.3 | 1492.9 | 541 | 1.0000 |
| service | 1 | 4 | 1595.4 | 3063.6 | 541 | 1.0000 |
| service | 1 | 8 | 1451.1 | 6309.3 | 541 | 1.0000 |
| process | 0.1 | 1 | 4436.2 | 261.2 | 357 | 1.0000 |
| process | 0.1 | 2 | 7565.2 | 340.9 | 715 | 1.0000 |
| process | 0.1 | 4 | 8984.7 | 447.6 | 1430 | 1.0000 |
| process | 0.1 | 8 | 8968.9 | 895.0 | 2860 | 1.0000 |
| process | 1 | 1 | 2079.9 | 561.8 | 541 | 1.0000 |
| process | 1 | 2 | 2962.7 | 678.3 | 1083 | 1.0000 |
| process | 1 | 4 | 2985.7 | 1348.4 | 2166 | 1.0000 |
| process | 1 | 8 | 2997.0 | 2694.2 | 4332 | 1.0000 |
| MPS | 0.1 | 1 | 4466.1 | 260.3 | 385 | 1.0000 |
| MPS | 0.1 | 2 | 7959.8 | 350.8 | 736 | 0.9961 |
| MPS | 0.1 | 4 | 16784.4 | 337.3 | 1438 | 0.9992 |
| MPS | 0.1 | 8 | 32658.1 | 349.3 | 2842 | 0.9992 |
| MPS | 1 | 1 | 2228.7 | 486.8 | 569 | 1.0000 |
| MPS | 1 | 2 | 3773.7 | 568.2 | 1104 | 1.0000 |
| MPS | 1 | 4 | 4002.6 | 1502.7 | 2174 | 0.9837 |
| MPS | 1 | 8 | 4006.5 | 3497.3 | 4314 | 0.9998 |

| configuration | SF | tenants | queries/s | p99 (us) | peak device memory (MiB) | SMs per tenant | unassigned SMs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static MPS | 0.1 | 1 | 4006.9 | 340.0 | 385 | 84 | 0 |
| static MPS | 0.1 | 2 | 7587.1 | 362.0 | 736 | 40 | 4 |
| static MPS | 0.1 | 4 | 13137.4 | 406.3 | 1438 | 20 | 4 |
| static MPS | 0.1 | 8 | 17356.4 | 563.0 | 2842 | 8 | 20 |
| static MPS | 1 | 1 | 2190.2 | 553.7 | 569 | 84 | 0 |
| static MPS | 1 | 2 | 2745.5 | 826.4 | 1104 | 40 | 4 |
| static MPS | 1 | 4 | 3222.4 | 1276.5 | 2174 | 20 | 4 |
| static MPS | 1 | 8 | 2841.9 | 2913.5 | 4314 | 8 | 20 |

The useful signal is the memory slope. Service memory stays at 357 MiB for
SF0.1 and 541 MiB for SF1 from one through eight tenants. Process memory rises
to 2,860 and 4,332 MiB; MPS rises to 2,842 and 4,314 MiB. This is consistent
with one service table and context versus one table and context per process.

The throughput points diagnose work still needed. The Python socket and single
owner-thread dispatch path completes far fewer Q6 requests than direct process
launches, while MPS overlaps the short Q6 kernels effectively. The single
repeat is irregular enough that it cannot support an overhead claim. The
service path needs dispatch profiling and optimization before the final run;
the final experiment also needs the full seeded query mix so tiny Q6 kernels do
not dominate the conclusion.

Static MPS keeps default MPS's per-process memory slope. Its fixed partitions
reduce Q6 throughput increasingly at higher tenant counts: at eight tenants,
each client receives 8 SMs and 20 of the GPU's 84 SMs remain unassigned because
the equal partition size must be an integer number of four-SM chunks. This is
the relevant present-day trade-off for the isolation comparison: static MPS
adds partial SM-fault containment but neither shares the table nor lets another
tenant borrow an idle partition. A repeated final experiment must include both
default and static MPS and must inject the same fault cases to measure the
documented containment boundary on this driver.

All 24 cells completed. The record retains its source and workload bytes, and
the exporter compares the two pilots' PTX text and environment directly. In
each of the eight service cells, exactly one accepted Q6
module was loaded despite one registration per tenant; every registration
audit contains the submitted PTX and server-derived binding; and final
telemetry has no live buffers, service handles, input errors, or fatal errors.

The final RQ4 result must follow
`../docs/EVALUATION_BLUEPRINT.md` (`../../docs/EVALUATION_BLUEPRINT.md`, not included in this archive): freeze a
commit, randomize configuration order, run five five-minute repeats, obtain
offered-load curves, add process plus separately allocated exported inputs,
measure MIG on supported hardware, and report confidence intervals. The pilot
CSV files are regenerated by `uv run python -m eval.export`; they retain each
configuration's QPS, p99, memory, fairness, and request count at every tenant
count and scale, plus the static-MPS SM allocation and unused chunks.
