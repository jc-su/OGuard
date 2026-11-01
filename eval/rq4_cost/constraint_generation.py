"""Measure bounded constraint generation against its equivalent eager expression.

    uv run python -m eval.projection_cost --output eval/records/projection_cost_<date>.json

This is a CPU microbenchmark of the elimination step, not a kernel-coverage
or end-to-end speed claim. GateSoundAccess proves the prefix equivalence.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from opgate.gate import formal_dir

SOURCE = """From Coq Require Import List ZArith.
Require Import OpGate.GateIsa OpGate.GateForm OpGate.GateDomain.
Import ListNotations.
Open Scope Z_scope.

Definition eager (x : atom) (cs : list constraint) : list constraint :=
  let has := fun k : constraint => let '(ts, _, _) := k in negb (Z.eqb (tget ts x) 0) in
  let with_x := filter has cs in
  filter (fun k => negb (has k)) cs ++
  match find (fun k : constraint => let '(_, lo, hi) := k in Z.eqb lo hi) with_x with
  | Some ke =>
      let '(te, _, _) := ke in
      flat_map (fun k => let '(t, _, _) := k in if terms_eqb t te then [] else [eliminate x k ke]) with_x
  | None =>
      firstn FM_CAP (flat_map (fun k1 => flat_map (fun k2 =>
                       let '(t1, _, _) := k1 in let '(t2, _, _) := k2 in
                       if terms_eqb t1 t2 then [] else [eliminate x k1 k2]) with_x) with_x)
  end.

Definition constraints : list constraint :=
  map (fun k => ([((1%positive, EX), 1); ((Pos.of_nat (k + 3), EX), 1)], 0, 100))
      (seq 0 COUNT).
Eval vm_compute in List.length (METHOD (1%positive, EX) constraints).
"""


def measure(directory: Path, count: int, method: str, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        query = Path(tmp) / "Query.v"
        query.write_text(SOURCE.replace("COUNT", str(count)).replace("METHOD", method))
        started = time.monotonic()
        try:
            result = subprocess.run(
                ["coqc", "-q", "-Q", str(directory), "OpGate", str(query)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "constraints": count,
                "method": method,
                "seconds": time.monotonic() - started,
                "timeout": True,
            }
    if result.returncode or "= 64%nat" not in result.stdout:
        raise RuntimeError(result.stderr or result.stdout)
    return {
        "constraints": count,
        "method": method,
        "seconds": time.monotonic() - started,
        "timeout": False,
        "retained": 64,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = formal_dir()
    results = []
    for count in (128, 256, 512):
        for method in ("eager", "fm_step"):
            row = measure(directory, count, method, 60)
            results.append(row)
            print(row, flush=True)
    record = {
        "scope": "synthetic constraint-generation microbenchmark, including coqc startup",
        "query": SOURCE,
        "library": str(directory),
        "coqc": subprocess.run(
            ["coqc", "--version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "results": results,
    }
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
