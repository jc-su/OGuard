"""The checked walk trims its state after every TRIM_EVERY-th instruction of a block, keeping
the registers live after it, the registers that the successors' cases name, and the fixed
registers, and keeps every fact after a block's last instruction."""

import subprocess

from opgate.gate import formal_dir

SOURCE = r"""
From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning.
Import ListNotations.
Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.empty nat; lf_fixed := [99%positive];
 lf_bounds := []; lf_aligns := []; lf_roles := [];
 lf_shared_window := 100%positive; lf_local_window := 101%positive;
 lf_shared := []; lf_locals := [(291%positive,(8,[]))];
 lf_cells := [(290%positive,291%positive,0)]; lf_payloads := PM.empty unit;
 lf_derived := []; lf_names := [] |}.
(* Block 0 writes the registers 10 to 139, one at each of its first 130
   instructions; block 1 stores register 20 at the address in register 5. *)
Definition long : program :=
 [map (fun k => IImm None (Pos.of_nat (10 + k)) 0) (seq 0 130) ++ [IBra None 1];
  [IMem None KStore false 5%positive 0 8 (Some (OReg 20%positive)) [] []; IExit None]].
Definition named : astate :=
 {| amap := PM.add 7%positive (FForm (fatom 64 (6%positive,EX) 1) 0 None) (PM.empty fact);
    acons := [] |}.
Definition cases : list (list astate) := [[empty_state facts]; [named]].
Definition plan := case_plan facts long cases 0.
Definition trims (plan : walk_plan) : list nat :=
 map fst (filter (fun x => match snd (snd x) with Some _ => true | None => false end)
   (combine (seq 0 (List.length plan)) plan)).
Example plan_is_the_body : map fst plan = block_body long 0.
Proof. vm_compute. reflexivity. Qed.
Example every_16th_but_not_after_the_last :
 trims plan = map (fun k => (16 * k + 15)%nat) (seq 0 8).
Proof. vm_compute. reflexivity. Qed.
Example block_of_128_keeps_its_last :
 trims (case_plan facts [map (fun k => IImm None (Pos.of_nat (10 + k)) 0) (seq 0 128)] [[]] 0)
 = map (fun k => (16 * k + 15)%nat) (seq 0 7).
Proof. vm_compute. reflexivity. Qed.
(* Kept: live in block 1 (20, 5), named by its case (7, and 6 that the fact
   reads), fixed (99), and the local cell (290, 291); dropped: written and
   never read (30, 15). *)
Example seed_keeps_live_named_and_fixed :
 match snd (nth 63 plan (IExit None, None)) with
 | Some keep => map (fun r => PM.mem r keep)
     [20%positive; 5%positive; 7%positive; 6%positive; 99%positive; 290%positive; 291%positive;
      30%positive; 15%positive]
 | None => []
 end = [true; true; true; true; true; true; true; false; false].
Proof. vm_compute. reflexivity. Qed.
(* The walk drops the fact of a register at the trim after it is written
   (30 at 31, 135 at 127), keeps a live one (20), and keeps every fact
   written after the last trim (138). *)
Example walk_drops_dead_facts :
 let out := plan_walk facts (empty_state facts) plan in
 map (fun r => match get out r with Some _ => true | None => false end)
   [20%positive; 30%positive; 135%positive; 138%positive] = [true; false; false; true].
Proof. vm_compute. reflexivity. Qed.
Print Assumptions trim_after_sound.
"""


def test_trimmed_walk_plan(tmp_path):
    query = tmp_path / "TrimmedWalk.v"
    query.write_text(SOURCE)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Closed under the global context" in result.stdout, result.stdout
