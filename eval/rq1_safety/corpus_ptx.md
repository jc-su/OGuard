# The gate on the PTX of the triggered corpus defects

Recorded 2026-09-27 and run again with the final gate on 2026-09-28, with the
same decisions. The executable check is
[`corpus_ptx.py`](corpus_ptx.py), and the complete record is
`records/e3_gate_corresponding_2026-09-28.json` (`../records/e3_gate_corresponding_2026-09-28.json`, not included in this archive)
(the first run's is
`records/e3_gate_corresponding_2026-09-27.json` (`../records/e3_gate_corresponding_2026-09-27.json`, not included in this archive)).
The archive of the unchecked corpus run remains unchanged.

The unchecked corpus run records the effect of 27 synthesized defects in a pooled GPU layout. Its
archived harness imported each `numba.py` and JIT-compiled it, or compiled each
`kernel.cu` to a temporary cubin with `nvcc -cubin -arch=sm_86 -O3`. The
archive did not retain the executed binary. It therefore cannot support an
exact-byte comparison after the fact.

The corpus separately retains level-3 `numba.ptx` and `kernel.ptx` artifacts
generated from the same source files. This check reads the unchecked corpus run's `results.jsonl`,
selects the corresponding artifact for each recorded source, records its PTX
path, and invokes the gate with the binding derived from the retained task
signature.

| Recorded effect of the unchecked corpus run | Corresponding PTX artifacts | Gate result |
| --- | ---: | --- |
| silent cross-buffer write | 19 | 19 refused |
| silent cross-buffer read | 5 | 5 refused |
| context fault | 3 | 3 refused |
| **Total** | **27** | **27 refused** |

No effect is inferred from a filename or gate result. Each classification is
copied from the execution record of the unchecked corpus run and checked against the expected 19/5/3
split. The check fails if the unchecked corpus run contains a duplicate or other number of kernels,
if a recorded case has no effect, or if the gate accepts any of the 27
corresponding artifacts.

This establishes gate coverage for compiler artifacts from all 27 source files
whose effects the unchecked corpus run measured. It does not establish that these PTX bytes are
identical to the unchecked corpus run's JIT output or temporary cubin. The paired replay supplies the paired
exact-byte result for the concrete disclosure, fault, and safe control. The
executor, which launches only the verified text of a module the gate
accepts, is tested separately (`tests/executor`).

Run the check with:

```console
uv run python -m eval.rq1_safety.corpus_ptx \
  --output eval/records/e3_gate_corresponding_<date>.json
```
