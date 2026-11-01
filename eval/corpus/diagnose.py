"""Rocq's view of why the gate refuses a corpus kernel.

    uv run python -m eval.corpus.diagnose haiku/gather numba [accesses|edges|translation|state B]

`accesses`, the default, prints each access the check refuses in the first
kernel, with the name, fact, and range of each register of its address form,
and the constraints in force. `edges` prints each control-flow edge whose
state does not imply its target's invariant, with the facts and constraints
of the invariant that the edge's state does not imply. `state B` prints the
body and the invariant of block B. `translation` prints the translated
program and the bounds and roles of its fixed registers. A development tool
of the lowering catalog; not part of a test.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.gate import export_terms, formal_dir, zeroed

COMMON = """
Definition kernel_of : option (pkernel * binding) :=
  match parse_text text size with
  | Ok ks => match ks_entries ks, mfs with (_, k) :: _, (_, mf) :: _ => Some (k, mf) | _, _ => None end
  | Err _ _ => None
  end.
Definition names : list (string * reg) :=
  match kernel_of with
  | Some (k, mf) =>
      match inline k with
      | Ok (K', org) =>
          match (p <- translate K' mf org ;; F <- facts K' mf ;; ret (p, F)) init with
          | Ok (_, s) => ts_names s
          | Err _ _ => []
          end
      | Err _ _ => []
      end
  | None => []
  end.
Definition name_of (r : reg) : string :=
  match find (fun p => Pos.eqb (snd p) r) names with Some (n, _) => n | None => "(fresh)" end.
Definition named (ts : terms) := map (fun t => (name_of (fst (fst t)), snd (fst t), snd t)) ts.
"""

ACCESSES = """
Definition address0 (i : instr) : list reg :=
  match i with
  | ILoad _ _ _ _ a _ | IMem _ _ _ a _ _ _ _ _ | ILdL _ _ _ a _ _ | IStL _ a _ _ _ _
  | ILdS _ _ _ a _ | IShA _ _ a _ _ _ => [a]
  | _ => []
  end.
Definition address (st : astate) (i : instr) : list reg :=
  flat_map (fun a => a :: match get st a with
                          | Some (FForm f _ _) => map (fun t => fst (fst t)) (ft f)
                          | _ => [] end) (address0 i).
Definition diagnosis :=
  match kernel_of with
  | Some (k, mf) =>
      match front k mf with
      | Ok (prog, F) =>
          inl (lf_ok F, List.length prog,
               match analyze F prog with
               | Some inv =>
                   Some (map (fun bi => let '(b, i) := bi in
                               let st := walk F (inv_at F inv b) (firstn i (block_body prog b)) in
                               (b, i, fetch prog b i,
                                match fetch prog b i with
                                | Some ins => map (fun r => (r, name_of r, get st r, reg_range F st r)) (address st ins)
                                | None => [] end,
                                map (fun k => let '(ts, lo, hi) := k in (named ts, lo, hi)) (acons st)))
                          (flat_map (fun b => if is_bottom (inv_at F inv b) then []
                                              else walk_fail F (inv_at F inv b) b 0 (block_body prog b))
                                    (seq 0 (List.length prog))))
               | None => None
               end)
      | Err e i => inr (e, i)
      end
  | None => inr ("no kernel or binding", None)
  end.
Eval vm_compute in diagnosis.
"""

EDGES = """
Definition bad_edges :=
  match kernel_of with
  | Some (k, mf) =>
      match front k mf with
      | Ok (prog, F) =>
          match analyze F prog with
          | Some inv =>
              inl (flat_map (fun b =>
                     let out := walk F (inv_at F inv b) (block_body prog b) in
                     if is_bottom (inv_at F inv b) then [] else
                     flat_map (fun e => let '(v, s) := e in
                       match s with
                       | None => []
                       | Some s =>
                           if orb (negb (Nat.ltb v (List.length prog))) (state_le F s (inv_at F inv v)) then [] else
                           [(b, v,
                             PM.fold (fun r g acc =>
                                        if fact_le_o F s (eff_fact F s r) (Some g) then acc
                                        else (r, name_of r, eff_fact F s r, g) :: acc) (amap (inv_at F inv v)) [],
                             map (fun k => let '(ts, lo, hi) := k in (named ts, lo, hi))
                                 (filter (fun k => negb (cons_le F s [k])) (acons (inv_at F inv v))))]
                       end) (edge_states F prog b out))
                   (seq 0 (List.length prog)))
          | None => inr "the analysis reaches no fixpoint"
          end
      | Err e _ => inr e
      end
  | None => inr "no kernel or binding"
  end.
Eval vm_compute in bad_edges.
"""

STATE = """
Definition block_state :=
  match kernel_of with
  | Some (k, mf) =>
      match front k mf with
      | Ok (prog, F) =>
          match analyze F prog with
          | Some inv =>
              let st := inv_at F inv BLOCK in
              inl (block_body prog BLOCK,
                   map (fun p => (fst p, name_of (fst p), snd p)) (PM.elements (amap st)),
                   map (fun k => let '(ts, lo, hi) := k in (named ts, lo, hi)) (acons st))
          | None => inr "the analysis reaches no fixpoint"
          end
      | Err e _ => inr e
      end
  | None => inr "no kernel or binding"
  end.
Eval vm_compute in block_state.
"""

TRANSLATION = """
Definition program_of :=
  match kernel_of with
  | Some (k, mf) =>
      match front k mf with
      | Ok (prog, F) => Some (prog, map (fun x => (fst x, name_of (fst x), snd x)) (lf_bounds F), lf_roles F)
      | Err _ _ => None
      end
  | None => None
  end.
Eval vm_compute in program_of.
"""

MODES = {"accesses": ACCESSES, "edges": EDGES, "translation": TRANSLATION}


def query(text: str, source: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Query.v"
        path.write_text(text + COMMON + source)
        result = subprocess.run(
            ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=900,
        )
    return result.stdout + result.stderr[-3000:]


def main(argv: list[str]) -> int:
    model, task = argv[0].split("/")
    toolchain = argv[1]
    mode = argv[2] if len(argv) > 2 else "accesses"
    if mode == "state":
        source = STATE.replace("BLOCK", f"{int(argv[3])}%nat")
    elif mode in MODES:
        source = MODES[mode]
    else:
        raise SystemExit(f"mode {mode!r} is not one of {sorted(MODES)} or state")
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    terms = export_terms(zeroed(text, binding), [binding])
    print(query(terms, source))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
