# The same PTX, unchecked and through the gate

Recorded 2026-09-27. The executable experiment is
[`paired_replay.py`](paired_replay.py), and the complete record is
`records/e7_gate_replay_2026-09-27.json` (`../records/e7_gate_replay_2026-09-27.json`, not included in this archive).
The Dask worker experiment remains unchanged.

This experiment is the paired causal experiment missing from the motivation results. The
parent compiles three Numba kernels once for sm_86, writes each PTX module once,
and gives those exact files to three fresh child processes. The current harness compares
the complete PTX text between the unguarded and protected cases; the linked
historical record retains its original metadata.

| Module | Unguarded | Protected |
| --- | --- | --- |
| `source[source.size]` | reads B's changing secret | refused before task submission |
| `source[2**32]` | CUDA error 700; B then fails | refused before task submission |
| `source[0]` | safe control | accepted and launched as the verified bytes |

The deployment is the configuration of the Dask worker experiment: two named Dask clients, one
dask-cuda worker process, one 256 MiB RMM pool, and adjacent 32-element cuDF
columns. The exact-PTX loader uses the Numba array ABI and CUDA driver rather
than recompiling the accepted module at launch.

In the unguarded confidentiality child, A's task had only
`tenant-a-live-column` as a scheduler dependency. Its first result was B's
`0x11223344556677`; after B updated the same live column, its second result was
`0x77665544332211`. In the unguarded availability child, B's baseline sum
succeeded, A's wild read raised error 700, and B's next cuDF sum and direct
synchronization failed.

In the protected child, OpGate refused both unsafe modules. The gate
branch did not create a Dask task for either module and did not call the CUDA
loader. OpGate accepted the safe control, the loader executed exactly the
accepted module using only A's column, and returned A's first value. B then
updated its secret; B's sum returned the expected value and direct CUDA
synchronization succeeded.

This shows the implemented gate makes the intended decision early enough to
prevent both observed effects, while accepting a nearby safe kernel. It does
not establish complete mediation. The current repository still lacks the
catalog, executor, allocation wrapper, cuDF host hook, and restricted service
boundary. Ordinary `Client.submit` remains available in the experiment, so a
host that exposed it to an untrusted tenant could bypass this gate branch.

Run all three fresh-process cases:

```console
.venv/bin/python eval/rq1_safety/paired_replay.py all \
  --output eval/records/e7_gate_replay_2026-09-27.json
```
