(*
  What the checker's facts mean.

  An environment fixes one thread's run: the memory at each step, the
  results of the instructions the checker does not model, the thread's view
  of shared memory at each step, and the registers' launch values, among
  them the size registers. `env_ok` states the launch facts the checker
  assumes, each of which the runtime establishes: the bounds and alignments
  of fixed registers, and the ranges of the elements of index inputs.

  `fact_ok` gives the meaning of one fact about a register's value at step t,
  and `state_ok` the meaning of an abstract state.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import PArith.PArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
From Coq Require Import FSets.FMapPositive.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.
Require Import OpGate.GateDomain.

Local Open Scope Z_scope.

Record env := {
  e_mem : nat -> Z -> Z;          (* memory at each step *)
  e_arb : nat -> reg -> Z;        (* results of the unmodeled instructions *)
  e_shm : nat -> Z -> Z;          (* the thread's view of shared memory at each step *)
  e_init : reg -> Z               (* registers at launch *)
}.

Section Meaning.

Variable F : lfacts.
Variable prog : program.
Variable E : env.

Definition wdz (r : reg) : Z := wd F r.
Definition run' :=
  run prog (wdn F) (e_mem E) (e_arb E) (e_shm E) (e_init E) (lf_cells F) (lf_locals F) (lf_local_window F).

(* The value of a bound form: its terms read at the launch values and its
   constant. *)
Definition bval (f : form) : Z := tsum (wdn F) (e_init E) (ft f) + flo f.

(* The element of index input P at byte offset e, read as its integer type. *)
Definition ix_read (ix : index) (P : reg) (t : nat) (e : Z) : Z :=
  extend (Z.to_nat (8 * ix_elem ix)) (ix_signed ix)
         (trunc (Z.to_nat (8 * ix_elem ix)) (e_mem E t (trunc 64 (e_init E P + e)))).

Record env_ok : Prop := {
  (* registers hold values of their width; fixed registers their bounds *)
  ok_init : forall r, 0 <= e_init E r < 2 ^ wdz r;
  ok_widths : forall r, 1 <= wdz r <= 64;
  ok_bounds : forall r lo hi, is_fixed F r = true -> lookup (lf_bounds F) r = Some (lo, hi) ->
              lo <= e_init E r <= hi;
  ok_aligns : forall r k, is_fixed F r = true -> lookup (lf_aligns F) r = Some k ->
              (2 ^ Z.max 0 k | e_init E r);
  (* base parameters are fixed, and their spans are well formed *)
  ok_roles_fixed : forall P sp, role_of F P = Some sp -> is_fixed F P = true;
  ok_spans : forall P sp, role_of F P = Some sp -> span_ok F sp = true;
  (* every whole element of an index input lies in its range, at every step:
     no thread writes an index input during the launch *)
  ok_index : forall P sp ix t e, role_of F P = Some sp -> sp_index sp = Some ix ->
             0 <= e -> e mod ix_elem ix = 0 -> e + ix_elem ix <= bval (sp_ext sp) ->
             bval (ix_lo ix) <= ix_read ix P t e <= bval (ix_hi ix);
  (* the windows' registers are fixed; each local variable's register is
     fixed and 32 bits wide, and the variable lies in the window's first
     window_size bytes; distinct variables of one frame, or of a frame and a
     frame it calls, do not overlap; each cell is a register that is not
     fixed, of 4 or 8 bytes, inside its variable *)
  ok_windows : is_fixed F (lf_shared_window F) = true /\ is_fixed F (lf_local_window F) = true;
  ok_locals : forall u s, In (u, s) (lf_locals F) ->
              is_fixed F u = true /\ wd F u = 32 /\ 0 <= e_init E u /\ e_init E u + fst s <= window_size;
  ok_disjoint : forall u1 s1 f1 u2 s2 f2, In (u1, (s1, f1)) (lf_locals F) -> In (u2, (s2, f2)) (lf_locals F) ->
                u1 <> u2 -> prefix f1 f2 = true ->
                e_init E u1 + s1 <= e_init E u2 \/ e_init E u2 + s2 <= e_init E u1;
  ok_cells : forall k, In k (lf_cells F) -> cell_ok F k = true;
  (* no buffer lies in a window: the driver maps no allocation there *)
  ok_apart : forall P sp w, role_of F P = Some sp -> (w = lf_shared_window F \/ w = lf_local_window F) ->
             e_init E P + bval (sp_ext sp) <= e_init E w \/
             (e_init E w + window_size <= e_init E P /\ e_init E P + bval (sp_ext sp) <= 2 ^ 64)
}.

Definition cmp_holds (rs : reg -> Z) (k : cmpf) : Prop :=
  cmp_eval (cc k) (extend (cw k) (csg k) (rs (ca k))) (extend (cw k) (csg k) (opv rs (cb k))) = true.

Definition aligned (k v : Z) : Prop := (2 ^ Z.max 0 k | v).

(* x holds of value v at step t, with the atoms of its forms read in rs. *)
Definition fact_okv (t : nat) (rs : reg -> Z) (x : fact) (v : Z) : Prop :=
  match x with
  | FForm f al vr =>
      wf f /\ sat (wdn F) rs f v /\ aligned al v /\ (forall lo hi, vr = Some (lo, hi) -> lo <= v <= hi)
  | FAlign k => aligned k v
  | FPred cs exact =>
      (v <> 0 -> Forall (cmp_holds rs) cs) /\ (exact = true -> Forall (cmp_holds rs) cs -> v <> 0)
  | FNeg cs exact =>
      (v = 0 -> Forall (cmp_holds rs) cs) /\ (exact = true -> Forall (cmp_holds rs) cs -> v = 0)
  end.

Definition fact_ok (t : nat) (rs : reg -> Z) (r : reg) (x : fact) : Prop := fact_okv t rs x (rs r).

Definition cons_ok (rs : reg -> Z) (k : constraint) : Prop :=
  let '(ts, lo, hi) := k in lo <= tsum (wdn F) rs ts <= hi.

Definition state_ok (t : nat) (st : astate) (rs : reg -> Z) : Prop :=
  (forall r x, get st r = Some x -> fact_ok t rs r x) /\ Forall (cons_ok rs) (acons st).

(* ------------------------------------------------------------------ *)
(* The regions an access may reach.                                    *)

(* A write of the width bytes at byte offset e of a buffer with index range
   ix: a store of one whole element whose value, read as the element type,
   lies in the range. *)
Definition index_write (ix : index) (k : mkind) (e width : Z) (v : option Z) : Prop :=
  k = KStore /\ width = ix_elem ix /\ e mod ix_elem ix = 0 /\
  exists z, v = Some z /\
    bval (ix_lo ix) <= extend (Z.to_nat (8 * ix_elem ix)) (ix_signed ix) z <= bval (ix_hi ix).

(* A write of the width bytes at byte offset e of a flag span: a store of
   the number 1 into one whole word of 4 bytes. *)
Definition flag_write (k : mkind) (e width : Z) (v : option Z) : Prop :=
  k = KStore /\ width = 4 /\ e mod 4 = 0 /\ v = Some 1.

(* region t k addr width v: an access of kind k to the width bytes at addr,
   made at step t, stays inside a buffer the launch may reach that way; an
   access to a buffer with an index range, an index input or output, reads
   or writes one whole element, and a write writes a value in the range, v;
   a write into a flag span stores 1 into one whole word. *)
Inductive region (t : nat) : mkind -> Z -> Z -> option Z -> Prop :=
| RgBuffer : forall k P sp e width v,
    role_of F P = Some sp -> allows (sp_dir sp) k = true ->
    0 <= e -> e + width <= bval (sp_ext sp) ->
    (forall ix, sp_index sp = Some ix -> width = ix_elem ix /\ e mod ix_elem ix = 0) ->
    (forall ix, sp_index sp = Some ix -> k <> KLoad -> index_write ix k e width v) ->
    (sp_flag sp = true -> flag_write k e width v) ->
    region t k (trunc 64 (e_init E P + e)) width v.

(* Register values fit their widths, and fixed registers keep their launch
   values. *)
Definition regs_ok (rs : reg -> Z) : Prop :=
  (forall r, 0 <= rs r < 2 ^ wdz r) /\ (forall r, is_fixed F r = true -> rs r = e_init E r).

(* The width bytes at generic address x lie in the window of shared or local
   memory whose base a window register holds at launch. The hardware maps
   them to that memory, so the access touches no global memory. *)
Definition in_window (x width : Z) : bool :=
  existsb (fun w => (e_init E w <=? x) && (x + width <=? e_init E w + window_size))
          [lf_shared_window F; lf_local_window F].

(* The width bytes at address x of a state space lie inside a variable of
   vars, which lies in the space's first window_size bytes. The memory of a
   block or a thread holds every variable it declares; an access past it
   faults. *)
Definition in_var (vars : list (reg * form)) (x width : Z) : Prop :=
  exists u bytes, In (u, bytes) vars /\ 0 <= e_init E u /\ e_init E u + bval bytes <= window_size /\
    e_init E u <= x /\ x + width <= e_init E u + bval bytes.

(* A generic address inside a shared variable in the shared window, or
   inside a local variable in the local window. *)
Definition in_private (x width : Z) : Prop :=
  in_var (lf_shared F) (x - e_init E (lf_shared_window F)) width \/
  in_var (local_vars F) (x - e_init E (lf_local_window F)) width.

Definition private_in (p : private) : Prop :=
  match p with
  | PShared x width => in_var (lf_shared F) x width
  | PLocal x width => in_var (local_vars F) x width
  | PGeneric x width => in_private x width
  end.

End Meaning.

(* Arbitrary whole-entry filtering is a weakening, independently of how
   a proposal selects its keep map. *)
Lemma filter_fact_map_lookup : forall keep m r f,
  PM.find r (filter_fact_map keep m) = Some f -> PM.find r m = Some f.
Proof.
  intros keep m r f H. unfold filter_fact_map in H. rewrite PM.fold_1 in H.
  assert (Hgen : forall xs acc,
    (forall e, In e xs -> In e (PM.elements m)) ->
    (forall r f, PM.find r acc = Some f -> PM.find r m = Some f) ->
    forall r f, PM.find r (fold_left (fun acc e =>
      if PM.mem (fst e) keep then PM.add (fst e) (snd e) acc else acc) xs acc) = Some f ->
      PM.find r m = Some f).
  { induction xs as [|[r0 f0] xs IH]; intros acc Hin Ha r1 f1 Hf; cbn [fold_left] in Hf.
    - exact (Ha _ _ Hf).
    - eapply IH; [intros e He; apply Hin; right; exact He| |exact Hf].
      intros r2 f2 H2. cbn [fst snd] in H2.
      destruct (PM.mem r0 keep); [|exact (Ha _ _ H2)].
      destruct (Pos.eq_dec r2 r0) as [->|Hne].
      + rewrite PM.gss in H2. injection H2 as <-.
        apply PM.elements_complete. apply Hin. left. reflexivity.
      + rewrite PM.gso in H2 by exact Hne. exact (Ha _ _ H2). }
  eapply Hgen; [intros e He; exact He| |exact H].
  intros r1 f1 Hf. rewrite PM.gempty in Hf. discriminate.
Qed.

Theorem state_subset_weakens : forall F t (original retained : astate) rs,
  (forall r f, get retained r = Some f -> get original r = Some f) ->
  (forall k, In k (acons retained) -> In k (acons original)) ->
  state_ok F t original rs -> state_ok F t retained rs.
Proof.
  intros F t original retained rs Hmap Hcons [Hfacts Hconstraints]. split.
  - intros r f H. apply Hfacts. now apply Hmap.
  - rewrite Forall_forall in Hconstraints |- *.
    intros k H. apply Hconstraints. now apply Hcons.
Qed.

Theorem whole_state_filter_sound : forall F t keep keep_constraint st rs,
  state_ok F t st rs -> state_ok F t (whole_state_filter keep keep_constraint st) rs.
Proof.
  intros F t keep keep_constraint st rs H.
  eapply state_subset_weakens; [| |exact H].
  - intros r f Hget. unfold get, whole_state_filter in Hget; cbn [amap] in Hget.
    exact (filter_fact_map_lookup keep (amap st) r f Hget).
  - intros k Hk. cbn [whole_state_filter acons] in Hk.
    apply filter_In in Hk. exact (proj1 Hk).
Qed.

Theorem trim_live_seed_sound : forall F t seed st rs,
  state_ok F t st rs -> state_ok F t (trim_live_seed F seed st) rs.
Proof.
  intros F t seed st rs H. unfold trim_live_seed.
  apply whole_state_filter_sound. exact H.
Qed.

Theorem trim_after_sound : forall F t keep st rs,
  state_ok F t st rs -> state_ok F t (trim_after F keep st) rs.
Proof.
  intros F t [keep|] st rs H; [apply trim_live_seed_sound|]; exact H.
Qed.

(* A trim keeps a bottom state bottom: its contradictory constraint reads
   no register. *)
Lemma trim_after_bottom : forall F keep st,
  is_bottom st = true -> is_bottom (trim_after F keep st) = true.
Proof.
  intros F [keep|] st H; [|exact H].
  unfold trim_after, trim_live_seed, whole_state_filter, is_bottom in *. cbn [acons].
  apply existsb_exists in H. destruct H as [[[ts lo] hi] [Hin Hk]].
  apply existsb_exists. exists (ts, lo, hi). split; [|exact Hk].
  apply filter_In. split; [exact Hin|].
  destruct ts as [|x ts]; [reflexivity|discriminate].
Qed.

Theorem trim_live_state_sound : forall F t table b st rs,
  state_ok F t st rs -> state_ok F t (trim_live_state F table b st) rs.
Proof.
  intros F t table b st rs H. unfold trim_live_state.
  apply trim_live_seed_sound. exact H.
Qed.

Theorem retain_affine_bridges_sound : forall F t st retained rs,
 state_ok F t st rs -> state_ok F t retained rs ->
 state_ok F t (retain_affine_bridges F st retained) rs.
Proof.
 intros F t st retained rs Hst Hr. unfold retain_affine_bridges.
 destruct (filter (affine_bridge_candidate F st retained) (map fst (PM.elements (amap st)))).
 - exact Hr.
 - apply whole_state_filter_sound. exact Hst.
Qed.

Theorem trim_join_live_state_sound : forall F t table incoming b st rs,
  state_ok F t st rs -> state_ok F t (trim_join_live_state F table incoming b st) rs.
Proof.
  intros F t table incoming b st rs H. unfold trim_join_live_state.
  apply retain_affine_bridges_sound; [exact H|].
  apply trim_live_seed_sound. exact H.
Qed.

(* No register values satisfy a bottom state. *)
Lemma bottom_not_ok : forall F t st rs, is_bottom st = true -> state_ok F t st rs -> False.
Proof.
  intros F t st rs Hb [_ Hc]. unfold is_bottom in Hb. apply existsb_exists in Hb.
  destruct Hb as [[[ts lo] hi] [Hin Hk]]. destruct ts as [| x ts]; cbn [null andb] in Hk; [| discriminate].
  pose proof (proj1 (Forall_forall _ _) Hc _ Hin) as Hok. cbn [cons_ok tsum] in Hok.
  apply orb_true_iff in Hk. destruct Hk as [Hk | Hk]; apply Z.ltb_lt in Hk; lia.
Qed.
