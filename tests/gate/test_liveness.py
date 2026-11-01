"""Whole-entry proposal trimming retains control, payload and local dependencies."""

import subprocess

from opgate.gate import formal_dir

SOURCE = r"""
From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning.
Import ListNotations.
Open Scope Z_scope.
Definition marks (rs : list reg) := mark_regs rs (PM.empty unit).
Definition facts : lfacts := {|
 lf_widths := PM.empty nat; lf_fixed := [99%positive];
 lf_bounds := []; lf_aligns := []; lf_roles := [];
 lf_shared_window := 100%positive; lf_local_window := 101%positive;
 lf_shared := []; lf_locals := [(91%positive,(8,[]))];
 lf_cells := [(90%positive,91%positive,0)]; lf_payloads := PM.empty unit;
 lf_derived := []; lf_names := [] |}.
Example guard_and_payload_read :
 instruction_inputs (IMem (Some (false,7%positive)) KStore false 2%positive 0 8
  (Some (OReg 3%positive)) [] []) = [7%positive;2%positive;3%positive].
Proof. reflexivity. Qed.
Example guarded_definition_does_not_kill :
 instruction_definite_writes (IDef (Some (true,7%positive)) [1%positive;2%positive]) = [].
Proof. reflexivity. Qed.
Example multiple_unconditional_destinations :
 instruction_definite_writes (IMem None KLoad false 3%positive 0 16 None [] [1%positive;2%positive])
 = [1%positive;2%positive].
Proof. reflexivity. Qed.
Example local_load_defines_its_destination :
 instruction_definite_writes (ILdL None 64 1%positive 2%positive 0 []) = [1%positive].
Proof. reflexivity. Qed.
Example hidden_local_write_is_conservative :
 instruction_definite_writes (IStL None 2%positive 0 8 (Some(OReg 3%positive)) []) = [].
Proof. reflexivity. Qed.
Example guarded_write_keeps_old_destination :
 map (fun r => PM.mem r (live_before (IMov (Some(true,7%positive)) 1%positive 2%positive)
  (marks [1%positive]))) [1%positive;2%positive;7%positive] = [true;true;true].
Proof. vm_compute. reflexivity. Qed.
Example overwritten_destination_is_dead :
 map (fun r => PM.mem r (live_before (IMov None 1%positive 2%positive)
  (marks [1%positive]))) [1%positive;2%positive] = [false;true].
Proof. vm_compute. reflexivity. Qed.
Definition diamond : program :=
 [[IBra (Some(true,4%positive)) 2];
  [IMem None KStore false 2%positive 0 8 (Some(OReg 6%positive)) [] []; IExit None];
  [IMem None KStore false 3%positive 0 8 (Some(OReg 8%positive)) [] []; IExit None]].
Example both_branch_successors : live_successors diamond 0 = [2%nat;1%nat].
Proof. reflexivity. Qed.
Example both_branch_input_sets :
 map (fun r => PM.mem r (nth 0 (live_program facts diamond) (PM.empty unit)))
 [2%positive;3%positive;4%positive;6%positive;8%positive] = repeat true 5.
Proof. vm_compute. reflexivity. Qed.
Example local_cell_and_base_are_roots :
 map (fun m => (PM.mem 90%positive m, PM.mem 91%positive m)) (live_program facts diamond)
 = repeat (true,true) 3.
Proof. vm_compute. reflexivity. Qed.
Example guarded_exit_keeps_fallthrough : live_successors [[IExit(Some(true,4%positive))]] 0 = [1%nat].
Proof. reflexivity. Qed.
Example unconditional_exit_has_no_successor : live_successors [[IExit None]] 0 = [].
Proof. reflexivity. Qed.
Definition cyclic : program :=
 [[IImm None 1%positive 0;IBra None 1];
  [IBra (Some(true,2%positive)) 3];
  [IBin None BAdd 64 1%positive 1%positive (OReg 3%positive);IBra None 1];
  [IMem None KStore false 5%positive 0 8 (Some(OReg 1%positive)) [] [];IExit None]].
Example cycle_reads_reach_loop_head :
 map (fun r => PM.mem r (nth 1 (live_program facts cyclic) (PM.empty unit)))
 [1%positive;2%positive;3%positive;5%positive] = repeat true 4.
Proof. vm_compute. reflexivity. Qed.
Definition c1 := {|cc:=CLe;cw:=64;csg:=false;ca:=2%positive;cb:=OReg 3%positive|}.
Definition c2 := {|cc:=CEq;cw:=64;csg:=true;ca:=4%positive;cb:=OReg 5%positive|}.
Definition k37 : constraint := ([((3%positive,EX),1);((7%positive,EX),-1)],0,0).
Definition k78 : constraint := ([((7%positive,EX),1);((8%positive,EX),-1)],0,0).
Definition k77 : constraint := ([((77%positive,EX),1)],0,1).
Definition k99 : constraint := ([((99%positive,EX),1)],0,1).
Definition st : astate := {|
 amap := PM.add 1%positive (FPred [c1;c2] true)
 (PM.add 2%positive (FForm (fatom 64 (6%positive,EX) 1) 0 None)
 (PM.add 6%positive (FAlign 3)
 (PM.add 77%positive (FAlign 4) (PM.empty fact))));
 acons := [k37;k78;k77;k99] |}.
Definition trimmed := trim_live_state facts [marks [1%positive]] 0 st.
Example predicate_fact_is_whole : get trimmed 1%positive = Some(FPred [c1;c2] true).
Proof. vm_compute. reflexivity. Qed.
Example fact_dependency_is_kept : get trimmed 6%positive = Some(FAlign 3).
Proof. vm_compute. reflexivity. Qed.
Example transitive_constraint_dependency_is_kept : acons trimmed = [k37;k78;k99].
Proof. vm_compute. reflexivity. Qed.
Example independent_fact_is_removed : get trimmed 77%positive = None.
Proof. vm_compute. reflexivity. Qed.
Example constant_contradiction_is_not_dropped :
 acons(trim_live_state facts [] 0 {|amap:=PM.empty fact;acons:=[([],1,0)]|}) = [([],1,0)].
Proof. vm_compute. reflexivity. Qed.
Example negative_predicate_is_whole :
 get (whole_state_filter (marks [1%positive]) (fun _=>false)
 {|amap:=PM.add 1%positive(FNeg [c1;c2] true)(PM.empty fact);acons:=[k77]|}) 1%positive
 = Some(FNeg [c1;c2] true).
Proof. vm_compute. reflexivity. Qed.
Example arbitrary_empty_keep_remains_sound : forall t rs,
 state_ok facts t st rs -> state_ok facts t (whole_state_filter (PM.empty unit) (fun _=>false) st) rs.
Proof. intros. apply whole_state_filter_sound. assumption. Qed.
Example boundary_is_applied_only_to_incoming_states :
 iterate_with_boundary facts (fun _ _ => bottom_state) 1 [[IExit None]]
 (revisit_limits [[IExit None]]) [Some st] [0%nat] [] = Some [Some st].
Proof. reflexivity. Qed.
Print Assumptions whole_state_filter_sound.
Print Assumptions trim_live_state_sound.
Print Assumptions arbitrary_empty_keep_remains_sound.
Definition history_program : program :=
 [[IImm None 1%positive 0; IBin None BAdd 64 3%positive 1%positive (OImm 1); IBra None 1];
  [IMem None KStore false 2%positive 0 8 (Some(OReg 3%positive)) [] []; IExit None]].
Definition history_state : astate :=
 {|amap:=PM.add 1%positive(FForm(fconst 64 0)0 None)
   (PM.add 3%positive(FForm(fconst 64 1)0 None)(PM.empty fact));acons:=[]|}.
Definition raw_history:=trim_live_state facts (cfg_live_program facts history_program) 1 history_state.
Definition supported_history:=trim_live_state facts (live_program facts history_program) 1 history_state.
Example normalized_constant_has_no_current_dependency : fact_inputs(FForm(fconst 64 1)0 None)=[].
Proof. reflexivity. Qed.
Example runtime_dead_previous_operand : get raw_history 1%positive=None.
Proof. vm_compute. reflexivity. Qed.
Example static_support_retains_previous_operand :
 get supported_history 1%positive=Some(FForm(fconst 64 0)0 None).
Proof. vm_compute. reflexivity. Qed.
Definition historical_relation:=FForm(fadd(fatom 64 (1%positive,EX)1)(fconst 64 1))0 None.
Example support_recovers_relation_after_constant_normalization :
 register_fact_le facts supported_history 3%positive historical_relation=true.
Proof. vm_compute. reflexivity. Qed.
Example current_dependencies_alone_lose_relation :
 register_fact_le facts raw_history 3%positive historical_relation=false.
Proof. vm_compute. reflexivity. Qed.
Definition fixed_refined_state : astate :=
 {|amap:=PM.add 99%positive(FForm(fconst 64 1)0 None)(PM.empty fact);acons:=[]|}.
Example dead_refined_fixed_fact_retained :
 get(trim_live_state facts(live_program facts [[IExit None]])0 fixed_refined_state)99%positive
 =Some(FForm(fconst 64 1)0 None).
Proof. vm_compute. reflexivity. Qed.
Example conditional_definition_support_has_guard :
 PM.find 1%positive (instruction_support_graph facts [[IMov(Some(false,7%positive))1%positive 2%positive]])
 =Some[7%positive;2%positive].
Proof. vm_compute. reflexivity. Qed.
Example each_explicit_destination_has_inputs :
 map(fun r=>PM.find r(instruction_support_graph facts
 [[IMem None KLoad false 3%positive 0 16 None [] [1%positive;2%positive]]]))
 [1%positive;2%positive]=[Some[3%positive];Some[3%positive]].
Proof. vm_compute. reflexivity. Qed.
Example cyclic_support_closes_once :
 let graph:=PM.add 1%positive[2%positive;2%positive]
 (PM.add 2%positive[1%positive;3%positive](PM.empty(list reg)))in
 map(fun r=>PM.mem r(register_closure graph [1%positive]))[1%positive;2%positive;3%positive;4%positive]
 =[true;true;true;false].
Proof. vm_compute. reflexivity. Qed.

Example hidden_local_store_supports_cell_history :
 PM.find 90%positive (instruction_support_graph facts
 [[IStL None 91%positive 0 8 (Some(OReg 3%positive)) []; IFrame [91%positive]]])
 = Some [91%positive;3%positive].
Proof. vm_compute. reflexivity. Qed.
Example generic_store_supports_cell_history :
 PM.find 90%positive (instruction_support_graph facts
 [[IMem (Some(false,7%positive)) KStore true 2%positive 0 8 (Some(OReg 3%positive)) [] []]])
 = Some [7%positive;2%positive;3%positive].
Proof. vm_compute. reflexivity. Qed.
Example local_cell_support_keeps_normalized_source_history :
 let prog := [[IImm None 1%positive 0;
 IBin None BAdd 64 3%positive 1%positive (OImm 1);
 IStL None 91%positive 0 8 (Some(OReg 3%positive)) []; IBra None 1]; [IExit None]] in
 PM.mem 1%positive (nth 1 (live_program facts prog) (PM.empty unit)) = true.
Proof. vm_compute. reflexivity. Qed.

"""


def test_liveness_dependencies_and_whole_state_weakening(tmp_path):
    query = tmp_path / "LivenessControls.v"
    query.write_text(SOURCE)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 3, result.stdout
