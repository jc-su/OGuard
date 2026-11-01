# The gate on the PTX of the generated Q3 module

Recorded 2026-09-27 and run again with the final gate on 2026-09-28. The
executable check is [`generated_ptx.py`](generated_ptx.py), its complete result
is `records/e6_gate_exact_2026-09-28.json` (`../records/e6_gate_exact_2026-09-28.json`, not included in this archive)
(the first run's is
`records/e6_gate_exact_2026-09-27.json` (`../records/e6_gate_exact_2026-09-27.json`, not included in this archive)),
and the preserved compiler artifacts are
`records/e6_q3_filter_sm86_2026-09-27.ptx.gz` (`../records/e6_q3_filter_sm86_2026-09-27.ptx.gz`, not included in this archive)
and
`records/e6_q3_filter_sm86_2026-09-27.metadata.json` (`../records/e6_q3_filter_sm86_2026-09-27.metadata.json`, not included in this archive).

The generated-kernel replay executes an unchanged, independently generated TPC-H Q3 Triton module and
observes its illegal address poison another client's cuDF work. This experiment gates one
compiler-emitted `_q3_filter_kernel` variant captured from that replay's
Triton cache. Before the decision, the check verifies:

- the pinned DataKernelBench commit and every the generated-kernel replay source artifact;
- the successful execution record of the generated-kernel replay;
- Triton 3.8.0 metadata for sm_86, 4 warps, and 2 stages;
- the PTX entry, all 16 parameters, launch width, atomic, and output stores.

The manual binding is deliberately permissive and does not assume the unsafe
counter starts at zero. It declares six read spans, four `n`-row write spans,
a 4-byte read-write counter, the size, and both implicit Triton pointers with
full-width integer bounds. OpGate refused the 75,936-byte PTX both times. The
first run's gate reported only that Rocq did not prove every global access in
its region. The final gate names the first failure: the translation does not
model `ldmatrix.sync.aligned.m8n8.x4.shared.b16`, a warp's matrix load from
shared memory, so the refusal comes before any access of the unreset counter is
checked. The decision is sound, but it does not isolate the counter defect.

This shows that the implemented gate reaches a real compiler output from the
independent generation path. It covers one sm_86 autotune variant with a
manually constructed binding. Verification of every compiled variant and
complete mediation of the autotuner remain implementation work.

Run it against the pinned upstream checkout:

```console
.venv/bin/python -m eval.rq1_safety.generated_ptx \
  --upstream /path/to/datakernelbench \
  --output eval/records/e6_gate_exact_2026-09-28.json
```
