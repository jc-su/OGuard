"""The launch model recognizes the zeroing prologue that `opgate.gate.zeroed`
writes (design decision 50); CPU only.

`prologue_of` in `GatePrologue.v` reads the lowered prologue of a kernel with
shared variables off its program: the thread's index, the loop that clears
dynamic shared memory, the rounds of stores of the other shared variables, and
the barrier. `launch_ni` in `StoreLaunch.v` then proves that two runs from any
two shared memories agree, and `check_entry_certificate` requires the
recognition, so the gate accepts a kernel with shared variables only with a
recognized prologue.
"""

import re
import subprocess
import tempfile
from pathlib import Path

from opgate.binding import Binding
from opgate.gate import export_terms, formal_dir, zeroed
from tests.gate.test_zeroing import BINDING, DYNAMIC, TWO_WORDS, kernel

REPORT = """
Require Import OpGate.GateParse OpGate.GatePrologue.
Definition report := Eval vm_compute in
  match parse_text text size with
  | Ok ks =>
      map (fun e => match lookup mfs (fst e) with
                    | Some mf => match front (snd e) mf with
                                 | Ok (prog, F) =>
                                     (List.length (lf_shared F), prologue_ok F mf prog,
                                      match prologue_of F mf prog with
                                      | Some ps => Some (List.length (pr_static ps),
                                                         match pr_loops ps with Some _ => true | None => false end)
                                      | None => None
                                      end)
                                 | Err _ _ => (0%nat, false, None)
                                 end
                    | None => (0%nat, false, None)
                    end) (ks_entries ks)
  | Err _ _ => []
  end.
Print report.
"""


def report(text: str, binding: Binding) -> str:
    """What Rocq prints of the recognized prologue of each entry kernel."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Prologue.v"
        path.write_text(export_terms(text, [binding]) + REPORT)
        result = subprocess.run(
            ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=600,
        )
    assert result.returncode == 0, result.stderr[-2000:]
    found = re.search(r"report = (.*?)\n\s*: ", result.stdout, re.S)
    assert found is not None, result.stdout
    return re.sub(r"\s+", " ", found.group(1)).strip()


def test_the_rounds_of_a_static_shared_variable_are_recognized() -> None:
    # one shared variable of 512 bytes, cleared by 128 threads in one round of 4-byte stores
    assert report(zeroed(kernel(), BINDING), BINDING) == "[(1%nat, true, Some (1%nat, false))]"


def test_the_loop_of_dynamic_shared_memory_is_recognized() -> None:
    assert report(zeroed(DYNAMIC, TWO_WORDS), TWO_WORDS) == "[(1%nat, true, Some (0%nat, true))]"


def test_a_kernel_without_shared_variables_needs_no_recognized_prologue() -> None:
    text = (
        kernel()
        .replace(".shared .align 4 .b8 buf[512];\n", "")
        .replace(
            "mov.u32 %r3, buf;\nadd.s32 %r4, %r3, %r2;\nst.shared.u32 [%r4], %r1;\nbar.sync 0;\nld.shared.u32 %r5, [%r4];\n",
            "mov.u32 %r5, %r1;\n",
        )
    )
    assert report(zeroed(text, BINDING), BINDING) == "[(0%nat, true, None)]"
