(*
  The effect of the loops that clear dynamic shared memory.

  The zeroing prologue (GateFront.v, design decision 46) clears a dynamic
  shared variable whose bytes depend on the launch's sizes with a loop: thread
  t of the N threads of a block stores zero at the elements t, t + N,
  t + 2N, ... below the count of elements, which the loop computes from the
  parameters that hold the sizes. The gate requires the lowered loops to
  have the form of their plan (`zero_program_check`), which every accepted
  kernel passes (`accepted_zero_plan`). This file proves, in the one-thread
  semantics of GateIsa.v, what the loops do:

  - `zero_loops_thread`: a thread's run stores zero at element e of each
    dynamic variable, at the variable's address plus e times the width of a
    store, for each e below the count with e mod N its linear index, before
    it leaves the loops; the count is the constant plus each coefficient
    times its size, which the 32-bit arithmetic of the loop computes exactly
    since the launch keeps it below 2^32;
  - `zero_plan_count`: the count times the width is the binding's bytes of
    dynamic shared memory;
  - `zero_element_thread`: every element e has a thread of the block whose
    linear index is e mod N, so the threads of a block store every element;
  - `accepted_zeroing`: for every accepted kernel with such a variable, every
    byte below the binding's bytes is in a word that the thread of its
    element stores zero to before it leaves the loops.

  The loops end where the stores of the other shared variables and the
  barrier that ends the prologue follow, and no branch enters the prologue
  (zeroing_check). Trusted, as for the rest of the prologue: that the
  barrier makes these stores visible to the block's later loads. Nothing
  here is an axiom.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import PArith.PArith.
From Coq Require Import Arith.PeanoNat.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
From Coq Require Import ZArith.Znumtheory.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.
Require Import OpGate.GateDomain.
Require Import OpGate.GateMeaning.
Require Import OpGate.GateFront.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Equality of the lowered instructions.                               *)

Lemma guard_eqb_eq : forall g h, guard_eqb g h = true -> g = h.
Proof.
  intros [[a p]|] [[b q]|]; cbn; try discriminate; [|reflexivity].
  intros H. apply andb_true_iff in H. destruct H as [Ha Hp].
  apply eqb_prop in Ha. apply Pos.eqb_eq in Hp. subst. reflexivity.
Qed.

Lemma opnd_eqb_eq : forall a b, opnd_eqb a b = true -> a = b.
Proof.
  intros [r|x] [s|y]; cbn; try discriminate; intros H.
  - apply Pos.eqb_eq in H. subst. reflexivity.
  - apply Z.eqb_eq in H. subst. reflexivity.
Qed.

Lemma option_opnd_eqb_eq : forall a b, option_opnd_eqb a b = true -> a = b.
Proof.
  intros [a|] [b|]; cbn; try discriminate; [|reflexivity].
  intros H. apply opnd_eqb_eq in H. subst. reflexivity.
Qed.

Ltac eqb_facts :=
  repeat match goal with
  | H : _ && _ = true |- _ => apply andb_true_iff in H; destruct H
  | H : Pos.eqb _ _ = true |- _ => apply Pos.eqb_eq in H; subst
  | H : Z.eqb _ _ = true |- _ => apply Z.eqb_eq in H; subst
  | H : Nat.eqb _ _ = true |- _ => apply Nat.eqb_eq in H; subst
  | H : Bool.eqb _ _ = true |- _ => apply eqb_prop in H; subst
  | H : guard_eqb _ _ = true |- _ => apply guard_eqb_eq in H; subst
  | H : opnd_eqb _ _ = true |- _ => apply opnd_eqb_eq in H; subst
  | H : option_opnd_eqb _ _ = true |- _ => apply option_opnd_eqb_eq in H; subst
  end.

Lemma zero_instr_eqb_eq : forall a b, zero_instr_eqb a b = true -> a = b.
Proof.
  intros a b. destruct a, b; cbn [zero_instr_eqb]; try (intros H; discriminate H);
    repeat match goal with
           | |- (match ?c with _ => _ end) = true -> _ => destruct c
           end;
    try (intros H; discriminate H); intros H; eqb_facts; reflexivity.
Qed.

Lemma zero_instrs_eqb_eq : forall l m, zero_instrs_eqb l m = true -> l = m.
Proof.
  induction l as [|a l IH]; intros [|b m]; cbn; try discriminate; [reflexivity|].
  intros H. apply andb_true_iff in H. destruct H as [Ha Hl].
  rewrite (zero_instr_eqb_eq a b Ha), (IH m Hl). reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* Runs of straight-line instructions.                                 *)

Section Run.

Variable prog : program.
Variable wd : reg -> nat.
Variable mem : nat -> Z -> Z.
Variable arb : nat -> reg -> Z.
Variable shm : nat -> Z -> Z.
Variable init : reg -> Z.
Variable cells : list (reg * reg * Z).
Variable locals : list (reg * (Z * frame)).
Variable lwin : reg.

Notation run_ := (run prog wd mem arb shm init cells locals lwin).
Notation step_ := (step prog wd mem arb shm init cells locals lwin).
Notation at_ b i rs := ({| blk := b; idx := i; regs := rs |}).

(* An instruction of the lowered loops other than a branch: unguarded, and
   one whose result is a function of the registers. *)
Definition plain (i : instr) : bool :=
  match i with
  | IImm None _ _ | IMov None _ _ | IMad None _ _ _ _ _ | ICvt None _ _ _ _ _
  | IBin None _ _ _ _ _ | ISetp None _ _ _ _ _ _ | IShA None _ _ _ _ _ => true
  | _ => false
  end.

(* What a plain instruction does to the registers. *)
Definition exec1 (rs : reg -> Z) (i : instr) : reg -> Z :=
  match dest i with
  | Some d => upd wd rs d (result mem arb shm O rs i)
  | None => rs
  end.

Lemma step_plain : forall t b k rs i,
  fetch prog b k = Some i -> plain i = true ->
  step_ t (at_ b k rs) = Some (goto (next_pos prog b k) (exec1 rs i)).
Proof.
  intros t b k rs i Hf Hp. unfold step. cbn [blk idx regs]. rewrite Hf.
  destruct i; cbn in Hp; try discriminate;
    repeat match goal with g : guard |- _ => destruct g; [discriminate|] end;
    reflexivity.
Qed.

Lemma step_branch : forall t b k rs g tg,
  fetch prog b k = Some (IBra g tg) ->
  step_ t (at_ b k rs) =
  Some (if holds g rs then goto (tg, O) rs else goto (next_pos prog b k) rs).
Proof.
  intros t b k rs g tg Hf. unfold step. cbn [blk idx regs]. rewrite Hf. cbn [guard_of].
  destruct (holds g rs); reflexivity.
Qed.

Lemma run_succ : forall n s, run_ n = Some s -> run_ (S n) = step_ n s.
Proof. intros n s H. cbn [run]. rewrite H. reflexivity. Qed.

Lemma nth_error_skipn_head : forall {A} k (l : list A) x rest,
  skipn k l = x :: rest -> nth_error l k = Some x.
Proof.
  intros A k. induction k as [|k IH]; intros [|y l] x rest H; cbn in *; try discriminate.
  - injection H as -> ->. reflexivity.
  - exact (IH l x rest H).
Qed.

Lemma fetch_skipn : forall b k i rest,
  skipn k (block_body prog b) = i :: rest -> fetch prog b k = Some i.
Proof. intros b k i rest H. unfold fetch. exact (nth_error_skipn_head _ _ _ _ H). Qed.

Lemma skipn_succ : forall {A} k (l : list A) x rest,
  skipn k l = x :: rest -> skipn (S k) l = rest.
Proof.
  intros A k. induction k as [|k IH]; intros [|y l] x rest H; cbn in *; try discriminate.
  - injection H as -> ->. reflexivity.
  - exact (IH l x rest H).
Qed.

Lemma skipn_app_rest : forall {A} (l : list A) k (xs r : list A),
  skipn k xs = app l r -> skipn (k + length l) xs = r.
Proof.
  intros A l. induction l as [|y l IH]; intros k xs r H.
  - cbn in *. rewrite Nat.add_0_r. exact H.
  - cbn [app] in H. pose proof (skipn_succ _ _ _ _ H) as H1.
    cbn [length]. replace (k + S (length l))%nat with (S k + length l)%nat by lia.
    exact (IH (S k) xs r H1).
Qed.

Lemma skipn_cons_length : forall {A} k (l : list A) x rest,
  skipn k l = x :: rest -> (k + S (length rest) = length l)%nat.
Proof.
  intros A k. induction k as [|k IH]; intros [|y l] x rest H; cbn in *; try discriminate.
  - injection H as -> ->. reflexivity.
  - rewrite <- (IH l x rest H). lia.
Qed.

Lemma block_len_skipn : forall b k x rest,
  skipn k (block_body prog b) = x :: rest -> (k + S (length rest) = block_len prog b)%nat.
Proof. intros b k x rest H. exact (skipn_cons_length _ _ _ _ H). Qed.

(* A run through plain instructions inside a block. *)
Lemma run_plain_inside : forall l b k n rs rest,
  run_ n = Some (at_ b k rs) -> skipn k (block_body prog b) = app l rest -> rest <> [] ->
  forallb plain l = true ->
  run_ (n + length l)%nat = Some (at_ b (k + length l) (fold_left exec1 l rs)).
Proof.
  induction l as [|i l IH]; intros b k n rs rest Hrun Hskip Hrest Hplain.
  - cbn. rewrite Nat.add_0_r, Nat.add_0_r. exact Hrun.
  - cbn [forallb] in Hplain. apply andb_true_iff in Hplain. destruct Hplain as [Hi Hl].
    cbn [app] in Hskip.
    pose proof (fetch_skipn _ _ _ _ Hskip) as Hf.
    pose proof (block_len_skipn _ _ _ _ Hskip) as Hlen.
    rewrite app_length in Hlen. destruct rest as [|x rest']; [contradiction|]. cbn [length] in Hlen.
    assert (Hnext : next_pos prog b k = (b, S k)).
    { unfold next_pos. destruct (Nat.ltb_spec (S k) (block_len prog b)); [reflexivity|lia]. }
    assert (Hrun1 : run_ (S n) = Some (at_ b (S k) (exec1 rs i))).
    { rewrite (run_succ _ _ Hrun), (step_plain _ _ _ _ _ Hf Hi), Hnext. reflexivity. }
    pose proof (skipn_succ _ _ _ _ Hskip) as Hskip1.
    specialize (IH b (S k) (S n) (exec1 rs i) (x :: rest') Hrun1 Hskip1 ltac:(discriminate) Hl).
    cbn [length fold_left]. replace (n + S (length l))%nat with (S n + length l)%nat by lia.
    replace (k + S (length l))%nat with (S k + length l)%nat by lia. exact IH.
Qed.

(* A run through the plain instructions that end a block. *)
Lemma run_plain_end : forall l i b k n rs,
  run_ n = Some (at_ b k rs) -> skipn k (block_body prog b) = app l [i] ->
  forallb plain (app l [i]) = true ->
  run_ (S (n + length l)) = Some (at_ (S b) O (fold_left exec1 (app l [i]) rs)).
Proof.
  intros l i b k n rs Hrun Hskip Hplain.
  rewrite forallb_app in Hplain. apply andb_true_iff in Hplain. destruct Hplain as [Hl Hi].
  cbn in Hi. rewrite andb_true_r in Hi.
  pose proof (run_plain_inside l b k n rs [i] Hrun Hskip ltac:(discriminate) Hl) as Hin.
  assert (Hskip1 : skipn (k + length l) (block_body prog b) = [i]).
  { exact (skipn_app_rest _ _ _ _ Hskip). }
  pose proof (fetch_skipn _ _ _ _ Hskip1) as Hf.
  pose proof (block_len_skipn _ _ _ _ Hskip1) as Hlen. cbn [length] in Hlen.
  assert (Hnext : next_pos prog b (k + length l) = (S b, O)).
  { unfold next_pos. destruct (Nat.ltb_spec (S (k + length l)) (block_len prog b)); [lia|reflexivity]. }
  rewrite (run_succ _ _ Hin), (step_plain _ _ _ _ _ Hf Hi), Hnext.
  rewrite fold_left_app. reflexivity.
Qed.

End Run.

(* ------------------------------------------------------------------ *)
(* The plan's form.                                                    *)

(* The registers that the lowered loops write, and those that they only
   read: the thread's index, the sizes, and the variables. *)
Definition zero_written (zp : zero_plan) : list reg := zero_regs_list (zp_regs zp).
Definition zero_read (zp : zero_plan) : list reg :=
  app [zp_tx zp; zp_ty zp; zp_tz zp] (app (map fst (zp_terms zp)) (zp_bases zp)).
Definition zero_temps_of (prog : program) (zp : zero_plan) (j : nat) : list (reg * nat) :=
  zero_temps (zero_setup_block prog j) (length (zp_terms zp)).

Lemma pos_nodup_spec : forall l, pos_nodup l = true -> NoDup l.
Proof.
  assert (H : forall l seen ok, fst (fold_left (fun acc r => let '(ok, seen) := acc in
                    (ok && negb (existsb (Pos.eqb r) seen), r :: seen)) l (ok, seen)) = true ->
                ok = true /\ NoDup l /\ forall r, In r l -> ~ In r seen).
  { induction l as [|x l IH]; intros seen ok Hf; cbn in Hf.
    - split; [exact Hf|]. split; [constructor|]. intros r [].
    - destruct (IH (x :: seen) _ Hf) as [Hok [Hnd Hout]].
      apply andb_true_iff in Hok. destruct Hok as [Hok Hx]. apply negb_true_iff in Hx.
      split; [exact Hok|]. split.
      + constructor; [|exact Hnd]. intros Hin. exact (Hout x Hin (or_introl eq_refl)).
      + intros r [<-|Hr] Hs.
        * assert (existsb (Pos.eqb x) seen = true)
            by (apply existsb_exists; exists x; split; [exact Hs|apply Pos.eqb_refl]).
          congruence.
        * exact (Hout r Hr (or_intror Hs)). }
  intros l Hl. exact (proj1 (proj2 (H l [] true Hl))).
Qed.

Lemma pos_apart_spec : forall l m r, pos_apart l m = true -> In r l -> ~ In r m.
Proof.
  intros l m r H Hl Hm. unfold pos_apart in H. rewrite forallb_forall in H.
  specialize (H r Hl). apply negb_true_iff in H.
  assert (existsb (Pos.eqb r) m = true)
    by (apply existsb_exists; exists r; split; [exact Hm|apply Pos.eqb_refl]).
  congruence.
Qed.

Lemma in_combine_seq : forall {A} (l : list A) s j x,
  nth_error l j = Some x -> In ((s + j)%nat, x) (combine (seq s (length l)) l).
Proof.
  intros A l. induction l as [|y l IH]; intros s j x H; [destruct j; discriminate|].
  destruct j as [|j]; cbn in H |- *.
  - injection H as ->. left. rewrite Nat.add_0_r. reflexivity.
  - right. replace (s + S j)%nat with (S s + j)%nat by lia. exact (IH (S s) j x H).
Qed.

Record zero_form (wd : reg -> nat) (zp : zero_plan) (prog : program) : Prop := {
  zf_blocks : (3 * length (zp_bases zp) < length prog)%nat;
  zf_index : firstn 6 (block_body prog 0) = zero_index_ir zp;
  zf_loops : forall j base, nth_error (zp_bases zp) j = Some base ->
    zero_setup_block prog j = zero_setup_ir zp (zero_temps_of prog zp j) /\
    block_body prog (3 * j + 1) = zero_head_ir (zp_regs zp) (3 * j + 3) /\
    block_body prog (3 * j + 2) = zero_body_ir zp base (3 * j + 1) /\
    forall v from, In (v, from) (zero_temps_of prog zp j) ->
      wd v = 64%nat /\ (32 <= from)%nat /\ ~ In v (zero_written zp) /\ ~ In v (zero_read zp);
  zf_distinct : NoDup (zero_written zp);
  zf_apart : forall r, In r (zero_written zp) -> ~ In r (zero_read zp);
  zf_widths : forall r, In r [zr_zero (zp_regs zp); zr_index (zp_regs zp); zr_address (zp_regs zp);
                              zr_count (zp_regs zp); zr_element (zp_regs zp)] -> wd r = 32%nat;
  zf_pred : wd (zr_pred (zp_regs zp)) = 1%nat;
  zf_bx : 0 < zp_bx zp; zf_by : 0 < zp_by zp; zf_bz : 0 < zp_bz zp;
  zf_threads : zp_threads zp < 2 ^ 31;
  zf_width : 0 < zp_width zp < 2 ^ 31
}.

Lemma zero_program_ok_form : forall wd zp prog,
  zero_program_ok wd zp prog = true -> zero_form wd zp prog.
Proof.
  intros wd zp prog H. unfold zero_program_ok in H. rewrite !andb_true_iff in H.
  destruct H as [[[[[[[[[[[[Hb Hi] Hl] Hnd] Hap] Hw] Hp] Hx] Hy] Hz] Ht] Hw0] Hw1].
  split.
  - apply Nat.ltb_lt. exact Hb.
  - exact (zero_instrs_eqb_eq _ _ Hi).
  - intros j base Hj. rewrite forallb_forall in Hl.
    pose proof (Hl _ (in_combine_seq (zp_bases zp) 0 j base Hj)) as H. cbn beta iota in H.
    rewrite !andb_true_iff in H. destruct H as [[[Hs Hh] Hbd] Ht0].
    split; [exact (zero_instrs_eqb_eq _ _ Hs)|]. split; [exact (zero_instrs_eqb_eq _ _ Hh)|].
    split; [exact (zero_instrs_eqb_eq _ _ Hbd)|].
    intros v from Hin. rewrite forallb_forall in Ht0. specialize (Ht0 _ Hin). cbn beta iota in Ht0.
    rewrite !andb_true_iff in Ht0. destruct Ht0 as [[Hv Hf] Hn]. apply negb_true_iff in Hn.
    split; [apply Nat.eqb_eq; exact Hv|]. split; [apply Nat.leb_le; exact Hf|].
    assert (Hno : ~ In v (app (zero_written zp) (zero_read zp))).
    { intros Hin'. unfold zero_written, zero_read in Hin'.
      assert (existsb (Pos.eqb v) (app (zero_regs_list (zp_regs zp))
                (app [zp_tx zp; zp_ty zp; zp_tz zp] (app (map fst (zp_terms zp)) (zp_bases zp)))) = true)
        by (apply existsb_exists; exists v; split; [exact Hin'|apply Pos.eqb_refl]).
      congruence. }
    split; intros Hin'; apply Hno; apply in_or_app; [left|right]; exact Hin'.
  - exact (pos_nodup_spec _ Hnd).
  - intros r Hr. exact (pos_apart_spec _ _ r Hap Hr).
  - intros r Hr. rewrite forallb_forall in Hw. apply Nat.eqb_eq. exact (Hw r Hr).
  - apply Nat.eqb_eq. exact Hp.
  - apply Z.ltb_lt. exact Hx.
  - apply Z.ltb_lt. exact Hy.
  - apply Z.ltb_lt. exact Hz.
  - apply Z.ltb_lt. exact Ht.
  - split; apply Z.ltb_lt; assumption.
Qed.

(* ------------------------------------------------------------------ *)
(* What the loops compute.                                             *)

(* The sum of each coefficient times its size's register. *)
Fixpoint zero_sum (rs : reg -> Z) (ts : list (reg * Z)) : Z :=
  match ts with
  | [] => 0
  | (s, c) :: ts' => c * rs s + zero_sum rs ts'
  end.

(* The count of elements of each loop, at the launch values rs. *)
Definition zero_elements (zp : zero_plan) (rs : reg -> Z) : Z := zp_const zp + zero_sum rs (zp_terms zp).

(* The thread's linear index in its block. *)
Definition zero_thread (zp : zero_plan) (rs : reg -> Z) : Z :=
  (rs (zp_tz zp) * zp_by zp + rs (zp_ty zp)) * zp_bx zp + rs (zp_tx zp).

(* Where a thread is before the count of loop j: in block 0 after the
   thread's index, or at the start of block 3j. *)
Definition zero_start (j : nat) : nat * nat :=
  match j with O => (O, 6%nat) | S j' => ((3 * j' + 3)%nat, O) end.

Lemma trunc_small : forall n v, 0 <= v < 2 ^ Z.of_nat n -> trunc n v = v.
Proof. intros n v H. unfold trunc. apply Z.mod_small. exact H. Qed.

Lemma trunc_trunc_32 : forall from v, (32 <= from)%nat -> trunc 32 (trunc from v) = trunc 32 v.
Proof.
  intros from v H. unfold trunc. symmetry. apply Zmod_div_mod.
  - apply Z.pow_pos_nonneg; lia.
  - apply Z.pow_pos_nonneg; lia.
  - exists (2 ^ (Z.of_nat from - 32)). rewrite <- Z.pow_add_r by lia. f_equal. lia.
Qed.

Lemma nodup6 : forall a b c d e f : reg, NoDup [a; b; c; d; e; f] ->
  a <> b /\ a <> c /\ a <> d /\ a <> e /\ a <> f /\ b <> c /\ b <> d /\ b <> e /\ b <> f /\
  c <> d /\ c <> e /\ c <> f /\ d <> e /\ d <> f /\ e <> f.
Proof.
  intros a b c d e f H. rewrite !NoDup_cons_iff in H. cbn [In] in H.
  repeat split; intros Heq; subst; tauto.
Qed.

(* The bounds of a thread's linear index and of its partial sum. *)
Lemma index_arith : forall bx by_ bz tx ty tz,
  0 < bx -> 0 < by_ -> 0 < bz -> 0 <= tx < bx -> 0 <= ty < by_ -> 0 <= tz < bz ->
  (tx < bx * by_ * bz /\ ty < bx * by_ * bz /\ tz < bx * by_ * bz) /\
  0 <= tz * by_ + ty < bx * by_ * bz /\
  0 <= (tz * by_ + ty) * bx + tx < bx * by_ * bz.
Proof.
  intros bx by_ bz tx ty tz Hx Hy Hz Htx Hty Htz.
  assert (K1 : 0 <= tz * by_ + ty <= (bz - 1) * by_ + (by_ - 1)) by nia.
  assert (K2 : 0 <= (tz * by_ + ty) * bx <= ((bz - 1) * by_ + (by_ - 1)) * bx)
    by (split; [apply Z.mul_nonneg_nonneg; lia|apply Z.mul_le_mono_nonneg_r; lia]).
  assert (E : ((bz - 1) * by_ + (by_ - 1)) * bx + (bx - 1) = bx * by_ * bz - 1) by ring.
  assert (K3 : (bz - 1) * by_ + (by_ - 1) <= ((bz - 1) * by_ + (by_ - 1)) * bx) by nia.
  split; [|split; lia].
  split; [|split]; nia.
Qed.

Lemma mod_accumulate : forall x c y M, 0 < M -> ((x mod M) * c + y mod M) mod M = (y + c * x) mod M.
Proof.
  intros x c y M HM. rewrite Zplus_mod_idemp_r. rewrite <- Zplus_mod_idemp_l.
  rewrite Zmult_mod_idemp_l. rewrite Zplus_mod_idemp_l. f_equal. ring.
Qed.

Lemma mod_step : forall d y, 0 < d -> 0 < y -> y mod d = 0 -> d <= y.
Proof.
  intros d y Hd Hy H. apply Z.mod_divide in H; [|lia]. destruct H as [q Hq]. subst y.
  assert (Hq : 0 < q).
  { destruct (Z.lt_ge_cases 0 q) as [Hq|Hq]; [exact Hq|].
    assert (q * d <= 0) by (apply Z.mul_nonpos_nonneg; lia). lia. }
  assert (d * 1 <= d * q) by (apply Z.mul_le_mono_nonneg_l; lia). lia.
Qed.

Lemma address_arith : forall B C E w, 0 <= B -> B + C * w < 2 ^ 32 -> 0 <= E < C -> 0 < w ->
  trunc 32 (E * w + trunc 32 B) = B + E * w.
Proof.
  intros B C E w HB HBC HE Hw.
  assert (Ew : 0 <= E * w <= C * w)
    by (split; [apply Z.mul_nonneg_nonneg|apply Z.mul_le_mono_nonneg_r]; lia).
  unfold trunc. change (2 ^ Z.of_nat 32) with (2 ^ 32).
  rewrite (Z.mod_small B) by lia. rewrite Z.mod_small by lia. ring.
Qed.

Lemma next_arith : forall E C N, 0 <= E < C -> C + N < 2 ^ 32 -> 0 < N -> trunc 32 (E + N) = E + N.
Proof.
  intros E C N HE HC HN. unfold trunc. change (2 ^ Z.of_nat 32) with (2 ^ 32).
  apply Z.mod_small. lia.
Qed.

Lemma map_fst_combine : forall {A B} (l : list A) (l' : list B),
  (length l <= length l')%nat -> map fst (combine l l') = l.
Proof.
  intros A B l. induction l as [|x l IH]; intros [|y l'] H; cbn in *; try lia; try reflexivity.
  rewrite IH by lia. reflexivity.
Qed.

Ltac upd_simpl :=
  unfold upd;
  repeat match goal with
  | |- context [Pos.eqb ?x ?x] => rewrite Pos.eqb_refl
  | H : ?x <> ?y |- context [Pos.eqb ?x ?y] => rewrite (proj2 (Pos.eqb_neq x y) H)
  | H : ?y <> ?x |- context [Pos.eqb ?x ?y] => rewrite (proj2 (Pos.eqb_neq x y) (not_eq_sym H))
  end.

Section Loops.

Variable prog : program.
Variable wd : reg -> nat.
Variable mem : nat -> Z -> Z.
Variable arb : nat -> reg -> Z.
Variable shm : nat -> Z -> Z.
Variable init : reg -> Z.
Variable cells : list (reg * reg * Z).
Variable locals : list (reg * (Z * frame)).
Variable lwin : reg.
Variable zp : zero_plan.

Notation run_ := (run prog wd mem arb shm init cells locals lwin).
Notation exec := (exec1 wd mem arb shm).
Notation at_ b i rs := ({| blk := b; idx := i; regs := rs |}).
Local Notation z := (zr_zero (zp_regs zp)).
Local Notation t := (zr_index (zp_regs zp)).
Local Notation a := (zr_address (zp_regs zp)).
Local Notation n := (zr_count (zp_regs zp)).
Local Notation e := (zr_element (zp_regs zp)).
Local Notation p := (zr_pred (zp_regs zp)).
Local Notation N := (zp_threads zp).
Local Notation w := (zp_width zp).
Local Notation T := (zero_thread zp init).
Local Notation C := (zero_elements zp init).

Hypothesis Hform : zero_form wd zp prog.
Hypothesis Htx : 0 <= init (zp_tx zp) < zp_bx zp.
Hypothesis Hty : 0 <= init (zp_ty zp) < zp_by zp.
Hypothesis Htz : 0 <= init (zp_tz zp) < zp_bz zp.
Hypothesis Hsizes : forall s c, In (s, c) (zp_terms zp) -> 0 <= init s < 2 ^ 64.
Hypothesis Hcount : 0 <= C /\ C + N < 2 ^ 32.
Hypothesis Hbases : forall B, In B (zp_bases zp) -> 0 <= init B /\ init B + C * w < 2 ^ 32.
Hypothesis Hnonempty : zp_bases zp <> [].

Lemma written_distinct :
  z <> t /\ z <> a /\ z <> n /\ z <> e /\ z <> p /\ t <> a /\ t <> n /\ t <> e /\ t <> p /\
  a <> n /\ a <> e /\ a <> p /\ n <> e /\ n <> p /\ e <> p.
Proof. exact (nodup6 _ _ _ _ _ _ (zf_distinct _ _ _ Hform)). Qed.

Lemma read_apart : forall r, In r (zero_read zp) ->
  r <> z /\ r <> t /\ r <> a /\ r <> n /\ r <> e /\ r <> p.
Proof.
  intros r Hr. pose proof (zf_apart _ _ _ Hform) as H.
  repeat split; intros ->; refine (H _ _ Hr); unfold zero_written, zero_regs_list; cbn; tauto.
Qed.

Lemma width32 : wd z = 32%nat /\ wd t = 32%nat /\ wd a = 32%nat /\ wd n = 32%nat /\ wd e = 32%nat.
Proof.
  pose proof (zf_widths _ _ _ Hform) as H.
  repeat split; apply H; cbn; tauto.
Qed.

Lemma threads_pos : 0 < N.
Proof.
  unfold zp_threads. pose proof (zf_bx _ _ _ Hform). pose proof (zf_by _ _ _ Hform).
  pose proof (zf_bz _ _ _ Hform). apply Z.mul_pos_pos; [apply Z.mul_pos_pos|]; lia.
Qed.

Lemma thread_bounds : 0 <= T < N.
Proof.
  unfold zero_thread, zp_threads.
  destruct (index_arith (zp_bx zp) (zp_by zp) (zp_bz zp) (init (zp_tx zp)) (init (zp_ty zp)) (init (zp_tz zp))
              (zf_bx _ _ _ Hform) (zf_by _ _ _ Hform) (zf_bz _ _ _ Hform) Htx Hty Htz) as (_ & _ & H).
  exact H.
Qed.

Lemma threads_small : N < 2 ^ 31.
Proof. exact (zf_threads _ _ _ Hform). Qed.

(* After the thread's index: zero in z, the index in t, and the other
   registers but the address as at launch. *)
Lemma index_regs : forall rs, rs = fold_left exec (zero_index_ir zp) init ->
  rs z = 0 /\ rs t = T /\ forall r, r <> z -> r <> t -> r <> a -> rs r = init r.
Proof.
  intros rs ->. destruct written_distinct as (Hzt & Hza & _ & _ & _ & Hta & _).
  destruct width32 as (Wz & Wt & Wa & _).
  assert (Rx := read_apart (zp_tx zp) ltac:(unfold zero_read; cbn; tauto)).
  assert (Ry := read_apart (zp_ty zp) ltac:(unfold zero_read; cbn; tauto)).
  assert (Rz := read_apart (zp_tz zp) ltac:(unfold zero_read; cbn; tauto)).
  destruct Rx as (Hxz & Hxt & Hxa & _). destruct Ry as (Hyz & Hyt & Hya & _).
  destruct Rz as (Hzz & Hzt' & Hza' & _).
  destruct (index_arith (zp_bx zp) (zp_by zp) (zp_bz zp) (init (zp_tx zp)) (init (zp_ty zp)) (init (zp_tz zp))
              (zf_bx _ _ _ Hform) (zf_by _ _ _ Hform) (zf_bz _ _ _ Hform) Htx Hty Htz) as ((Ix & Iy & Iz) & Ia & Ib).
  pose proof threads_small as Hsmall. unfold zp_threads in Hsmall.
  assert (P : 2 ^ Z.of_nat 32 = 2 ^ 32) by reflexivity.
  unfold zero_index_ir. cbn [fold_left exec1 dest result opv].
  split; [|split].
  - upd_simpl. rewrite Wz. reflexivity.
  - upd_simpl. rewrite Wt, Wa. unfold zero_thread.
    rewrite (trunc_small 32 (init (zp_tz zp))) by (rewrite P; lia).
    rewrite (trunc_small 32 (init (zp_ty zp))) by (rewrite P; lia).
    rewrite (trunc_small 32 (init (zp_tx zp))) by (rewrite P; lia).
    rewrite (trunc_small 32 (init (zp_tz zp) * zp_by zp + init (zp_ty zp))) by (rewrite P; lia).
    rewrite trunc_small by (rewrite P; lia). reflexivity.
  - intros r Hrz Hrt Hra. upd_simpl. reflexivity.
Qed.

Definition zero_terms_ir (xs : list ((reg * Z) * (reg * nat))) : list instr :=
  flat_map (fun x => let '((s, c), (v, from)) := x in zero_term_ir (zp_regs zp) s c v from) xs.

(* The loads of the sizes add each coefficient times its size to the count,
   modulo 2^32, and change no register but the count, the element, and their
   own 64-bit registers. *)
Lemma terms_regs : forall xs rs acc,
  (forall s c v from, In ((s, c), (v, from)) xs ->
     In (s, c) (zp_terms zp) /\ wd v = 64%nat /\ (32 <= from)%nat /\
     ~ In v (zero_written zp) /\ ~ In v (zero_read zp)) ->
  rs n = acc mod 2 ^ 32 -> (forall r, In r (zero_read zp) -> rs r = init r) ->
  fold_left exec (zero_terms_ir xs) rs n = (acc + zero_sum init (map fst xs)) mod 2 ^ 32 /\
  forall r, r <> n -> r <> e -> (forall s c v from, In ((s, c), (v, from)) xs -> r <> v) ->
    fold_left exec (zero_terms_ir xs) rs r = rs r.
Proof.
  induction xs as [|[[s c] [v from]] xs IH]; intros rs acc Hxs Hn Hread.
  - cbn. rewrite Z.add_0_r. split; [exact Hn|]. intros. reflexivity.
  - unfold zero_terms_ir. cbn [flat_map]. rewrite fold_left_app. fold (zero_terms_ir xs).
    destruct (Hxs s c v from (or_introl eq_refl)) as (Hsc & Wv & Hfrom & Hvw & Hvr).
    destruct written_distinct as (_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hne & _).
    destruct width32 as (_ & _ & _ & Wn & We).
    assert (Hsr : In s (zero_read zp)).
    { unfold zero_read. apply in_or_app. right. apply in_or_app. left.
      apply (in_map fst _ (s, c)). exact Hsc. }
    destruct (read_apart s Hsr) as (_ & _ & _ & Hsn & Hse & _).
    assert (Hvn : v <> n) by (intros ->; apply Hvw; unfold zero_written, zero_regs_list; cbn; tauto).
    assert (Hve : v <> e) by (intros ->; apply Hvw; unfold zero_written, zero_regs_list; cbn; tauto).
    assert (Hsv : s <> v) by (intros ->; exact (Hvr Hsr)).
    set (rs4 := fold_left exec (zero_term_ir (zp_regs zp) s c v from) rs).
    assert (H4n : rs4 n = (acc + c * init s) mod 2 ^ 32).
    { unfold rs4, zero_term_ir. cbn [fold_left exec1 dest result opv extend]. upd_simpl.
      rewrite Wn, We, Wv, (Hread s Hsr), Hn.
      rewrite (trunc_small 64 0) by (cbn -[Z.pow]; lia).
      rewrite Z.mul_1_r, Z.add_0_r, (trunc_small 64 (init s)) by (cbn -[Z.pow]; exact (Hsizes s c Hsc)).
      rewrite trunc_trunc_32 by exact Hfrom.
      unfold trunc. change (2 ^ Z.of_nat 32) with (2 ^ 32).
      apply mod_accumulate. lia. }
    assert (H4r : forall r, r <> n -> r <> e -> r <> v -> rs4 r = rs r).
    { intros r Hrn Hre Hrv. unfold rs4, zero_term_ir. cbn [fold_left exec1 dest result opv extend].
      upd_simpl. reflexivity. }
    assert (Hread4 : forall r, In r (zero_read zp) -> rs4 r = init r).
    { intros r Hr. destruct (read_apart r Hr) as (_ & _ & _ & Hrn & Hre & _).
      rewrite H4r; [exact (Hread r Hr)|exact Hrn|exact Hre|]. intros ->. exact (Hvr Hr). }
    destruct (IH rs4 (acc + c * init s) (fun s' c' v' f' H => Hxs s' c' v' f' (or_intror H)) H4n Hread4)
      as [IHn IHr].
    split.
    + rewrite IHn. cbn [map zero_sum fst]. f_equal. ring.
    + intros r Hrn Hre Hrv. rewrite IHr; [| exact Hrn | exact Hre |].
      * apply H4r; [exact Hrn|exact Hre|]. exact (Hrv s c v from (or_introl eq_refl)).
      * intros s' c' v' f' H. exact (Hrv s' c' v' f' (or_intror H)).
Qed.

Lemma temps_length : forall j, length (zero_temps_of prog zp j) = length (zp_terms zp).
Proof. intros j. unfold zero_temps_of, zero_temps. rewrite map_length, seq_length. reflexivity. Qed.

Lemma plain_terms : forall xs, forallb plain (zero_terms_ir xs) = true.
Proof.
  intros xs. apply forallb_forall. intros i Hi. unfold zero_terms_ir in Hi.
  apply in_flat_map in Hi. destruct Hi as [[[s c] [v from]] [_ Hi]].
  cbn in Hi. destruct Hi as [<-|[<-|[<-|[<-|[]]]]]; reflexivity.
Qed.

(* The count of loop j: from its start, a thread reaches the loop's head
   with the count in n and its index as the first element. *)
Lemma setup_run : forall j base b k m rs,
  nth_error (zp_bases zp) j = Some base ->
  skipn k (block_body prog b) = zero_setup_ir zp (zero_temps_of prog zp j) ->
  run_ m = Some (at_ b k rs) -> rs z = 0 -> rs t = T ->
  (forall r, In r (zero_read zp) -> rs r = init r) ->
  exists m' rs', (m <= m')%nat /\ run_ m' = Some (at_ (S b) O rs') /\
    rs' z = 0 /\ rs' t = T /\ rs' n = C /\ rs' e = T /\
    (forall r, In r (zero_read zp) -> rs' r = init r).
Proof.
  intros j base b k m rs Hj Hskip Hrun Hz Ht Hread.
  destruct (zf_loops _ _ _ Hform j base Hj) as (_ & _ & _ & Htemps).
  set (xs := combine (zp_terms zp) (zero_temps_of prog zp j)).
  assert (Hsetup : zero_setup_ir zp (zero_temps_of prog zp j) =
                   app (IImm None n (zp_const zp) :: zero_terms_ir xs) [IMov None e t]) by reflexivity.
  rewrite Hsetup in Hskip.
  assert (Hplain : forallb plain (app (IImm None n (zp_const zp) :: zero_terms_ir xs) [IMov None e t]) = true).
  { rewrite forallb_app. cbn [forallb plain]. rewrite plain_terms. reflexivity. }
  pose proof (run_plain_end prog wd mem arb shm init cells locals lwin _ _ b k m rs Hrun Hskip Hplain) as Hend.
  eexists. eexists. split; [|split; [exact Hend|]]; [lia|].
  rewrite fold_left_app. cbn [fold_left].
  destruct written_distinct as (Hzt & Hza & Hzn & Hze & Hzp & Hta & Htn & Hte & Htp & Han & Hae & Hap & Hne & Hnp & Hep).
  destruct width32 as (Wz & Wt & Wa & Wn & We).
  set (rs1 := exec rs (IImm None n (zp_const zp))).
  assert (Hxs : forall s c v from, In ((s, c), (v, from)) xs ->
     In (s, c) (zp_terms zp) /\ wd v = 64%nat /\ (32 <= from)%nat /\
     ~ In v (zero_written zp) /\ ~ In v (zero_read zp)).
  { intros s c v from Hin. split; [exact (in_combine_l _ _ _ _ Hin)|].
    exact (Htemps v from (in_combine_r _ _ _ _ Hin)). }
  assert (H1n : rs1 n = zp_const zp mod 2 ^ 32).
  { unfold rs1. cbn [exec1 dest result]. upd_simpl. rewrite Wn. reflexivity. }
  assert (H1r : forall r, r <> n -> rs1 r = rs r).
  { intros r Hr. unfold rs1. cbn [exec1 dest result]. upd_simpl. reflexivity. }
  assert (Hread1 : forall r, In r (zero_read zp) -> rs1 r = init r).
  { intros r Hr. destruct (read_apart r Hr) as (_ & _ & _ & Hrn & _). rewrite H1r by exact Hrn.
    exact (Hread r Hr). }
  destruct (terms_regs xs rs1 (zp_const zp) Hxs H1n Hread1) as [H2n H2r].
  set (rs2 := fold_left exec (zero_terms_ir xs) rs1) in *.
  assert (Hv : forall r, In r (zero_written zp) \/ In r (zero_read zp) ->
                 forall s c v from, In ((s, c), (v, from)) xs -> r <> v).
  { intros r Hr s c v from Hin ->. destruct (Hxs s c v from Hin) as (_ & _ & _ & Hw & Hrd).
    destruct Hr; contradiction. }
  assert (Hwz : In z (zero_written zp)) by (unfold zero_written, zero_regs_list; cbn; tauto).
  assert (Hwt : In t (zero_written zp)) by (unfold zero_written, zero_regs_list; cbn; tauto).
  assert (HC : (zp_const zp + zero_sum init (map fst xs)) mod 2 ^ 32 = C).
  { unfold xs. rewrite map_fst_combine by (rewrite temps_length; lia).
    apply Z.mod_small. fold (zero_elements zp init). pose proof threads_pos. lia. }
  pose proof thread_bounds as HT. pose proof threads_small as Hsmall.
  cbn [exec1 dest result]. upd_simpl. rewrite We.
  split; [|split; [|split; [|split]]].
  - rewrite H2r by (try (exact (Hv z (or_introl Hwz))); assumption). rewrite H1r by exact Hzn. exact Hz.
  - rewrite H2r by (try (exact (Hv t (or_introl Hwt))); assumption). rewrite H1r by exact Htn. exact Ht.
  - rewrite H2n. exact HC.
  - rewrite H2r by (try (exact (Hv t (or_introl Hwt))); assumption). rewrite H1r by exact Htn. rewrite Ht.
    apply trunc_small. cbn -[Z.pow]. lia.
  - intros r Hr. destruct (read_apart r Hr) as (_ & _ & _ & Hrn & Hre & _).
    rewrite (proj2 (Pos.eqb_neq r e) Hre). rewrite (H2r r Hrn Hre (Hv r (or_intror Hr))).
    exact (Hread1 r Hr).
Qed.

Lemma fetch_body : forall b body k i,
  block_body prog b = body -> nth_error body k = Some i -> fetch prog b k = Some i.
Proof. intros b body k i Hb Hi. unfold fetch. rewrite Hb. exact Hi. Qed.

(* The head of loop j: a thread whose element E is not below the count
   leaves the loop, changing only the predicate; one whose element is below
   it goes to the loop's body. *)
Lemma loop_head : forall j base, nth_error (zp_bases zp) j = Some base ->
  forall E m rs, run_ m = Some (at_ (3 * j + 1)%nat O rs) -> rs e = E -> 0 <= E < 2 ^ 32 -> rs n = C ->
  exists rs1, (forall r, r <> p -> rs1 r = rs r) /\
    run_ (S (S m)) = Some (if C <=? E then at_ (3 * j + 3)%nat O rs1 else at_ (3 * j + 2)%nat O rs1).
Proof.
  intros j base Hj E m rs Hrun HE HE0 Hn.
  destruct (zf_loops _ _ _ Hform j base Hj) as (_ & Hhead & _ & _).
  destruct width32 as (_ & _ & _ & _ & _). pose proof (zf_pred _ _ _ Hform) as Wp.
  assert (P : 2 ^ Z.of_nat 32 = 2 ^ 32) by reflexivity.
  assert (Hf0 : fetch prog (3 * j + 1) 0 = Some (ISetp None CGe 32 false p e (OReg n)))
    by (apply (fetch_body _ _ _ _ Hhead); reflexivity).
  assert (Hf1 : fetch prog (3 * j + 1) 1 = Some (IBra (Some (false, p)) (3 * j + 3)))
    by (apply (fetch_body _ _ _ _ Hhead); reflexivity).
  assert (Hlen : block_len prog (3 * j + 1) = 2%nat) by (unfold block_len; rewrite Hhead; reflexivity).
  set (rs1 := exec rs (ISetp None CGe 32 false p e (OReg n))).
  exists rs1. split.
  { intros r Hr. unfold rs1. cbn [exec1 dest result]. upd_simpl. reflexivity. }
  assert (H1p : rs1 p = if C <=? E then 1 else 0).
  { unfold rs1. cbn [exec1 dest result opv extend cmp_eval]. upd_simpl. rewrite Wp, HE, Hn.
    pose proof threads_pos.
    rewrite (trunc_small 32 C) by (rewrite P; lia). rewrite (trunc_small 32 E) by (rewrite P; lia).
    destruct (C <=? E); reflexivity. }
  assert (Hrun1 : run_ (S m) = Some (at_ (3 * j + 1)%nat 1%nat rs1)).
  { rewrite (run_succ _ _ _ _ _ _ _ _ _ _ _ Hrun), (step_plain _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hf0) by reflexivity.
    unfold next_pos. rewrite Hlen. reflexivity. }
  rewrite (run_succ _ _ _ _ _ _ _ _ _ _ _ Hrun1), (step_branch _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hf1).
  cbn [holds xorb]. rewrite H1p. unfold next_pos. rewrite Hlen.
  replace (S (3 * j + 1)) with (3 * j + 2)%nat by lia.
  destruct (C <=? E); reflexivity.
Qed.

(* The body of loop j: from it with the element E below the count, a thread
   stores zero at the variable's address plus E times the width, and
   returns to the head with the element E + N, changing only the address,
   the element, and the predicate. *)
Lemma loop_body : forall j base, nth_error (zp_bases zp) j = Some base ->
  forall E m rs, run_ m = Some (at_ (3 * j + 2)%nat O rs) -> rs e = E -> 0 <= E < C ->
  rs z = 0 -> rs base = init base ->
  exists rs3 rs5,
    run_ (m + 2)%nat = Some (at_ (3 * j + 2)%nat 2%nat rs3) /\ rs3 a = init base + E * w /\ rs3 z = 0 /\
    run_ (m + 5)%nat = Some (at_ (3 * j + 1)%nat O rs5) /\ rs5 e = E + N /\
    (forall r, r <> a -> r <> e -> rs5 r = rs r).
Proof.
  intros j base Hj E m rs Hrun HE HEC Hz Hbase.
  destruct (zf_loops _ _ _ Hform j base Hj) as (_ & _ & Hbody & _).
  assert (Hbr : In base (zero_read zp)).
  { unfold zero_read. apply in_or_app. right. apply in_or_app. right. exact (nth_error_In _ _ Hj). }
  destruct (Hbases base (nth_error_In _ _ Hj)) as [HB0 HB1].
  destruct (read_apart base Hbr) as (_ & _ & Hba & _ & Hbe & _).
  destruct written_distinct as (Hzt & Hza & Hzn & Hze & Hzp & Hta & Htn & Hte & Htp & Han & Hae & Hap & Hne & Hnp & Hep).
  destruct width32 as (Wz & Wt & Wa & Wn & We).
  pose proof threads_pos as HN. pose proof (zf_width _ _ _ Hform) as Hw.
  assert (P : 2 ^ Z.of_nat 32 = 2 ^ 32) by reflexivity.
  assert (Hb0 : skipn 0 (block_body prog (3 * j + 2)) =
                app [IMov None a base; IMad None 32 a e (OImm w) (OReg a)]
                    [IShA None KStore a 0 w (Some (OReg z)); IBin None BAdd 32 e e (OImm N);
                     IBra None (3 * j + 1)]) by (rewrite Hbody; reflexivity).
  pose proof (run_plain_inside prog wd mem arb shm init cells locals lwin _ _ _ _ _ _ Hrun Hb0
                ltac:(discriminate) eq_refl) as Hrun3.
  cbn [length fold_left Nat.add] in Hrun3.
  set (rs3 := exec (exec rs (IMov None a base)) (IMad None 32 a e (OImm w) (OReg a))) in Hrun3.
  assert (Hb2 : skipn 2 (block_body prog (3 * j + 2)) =
                app [IShA None KStore a 0 w (Some (OReg z)); IBin None BAdd 32 e e (OImm N)]
                    [IBra None (3 * j + 1)]) by (rewrite Hbody; reflexivity).
  pose proof (run_plain_inside prog wd mem arb shm init cells locals lwin _ _ _ _ _ _ Hrun3 Hb2
                ltac:(discriminate) eq_refl) as Hrun5.
  cbn [length fold_left Nat.add] in Hrun5.
  set (rs5 := exec (exec rs3 (IShA None KStore a 0 w (Some (OReg z)))) (IBin None BAdd 32 e e (OImm N)))
    in Hrun5.
  assert (Hf4 : fetch prog (3 * j + 2) 4 = Some (IBra None (3 * j + 1)))
    by (apply (fetch_body _ _ _ _ Hbody); reflexivity).
  exists rs3, rs5. split; [exact Hrun3|]. split; [|split; [|split; [|split]]].
  - unfold rs3. cbn [exec1 dest result opv]. upd_simpl. rewrite Wa, Hbase, HE.
    exact (address_arith (init base) C E w HB0 HB1 HEC (proj1 Hw)).
  - unfold rs3. cbn [exec1 dest result opv]. upd_simpl. exact Hz.
  - replace (m + 5)%nat with (S (m + 2 + 2)) by lia.
    rewrite (run_succ _ _ _ _ _ _ _ _ _ _ _ Hrun5), (step_branch _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hf4).
    reflexivity.
  - unfold rs5, rs3. cbn [exec1 dest result opv binop_eval]. upd_simpl. rewrite We, HE.
    exact (next_arith E C N HEC (proj2 Hcount) HN).
  - intros r Hra Hre. unfold rs5, rs3. cbn [exec1 dest result opv]. upd_simpl. reflexivity.
Qed.

(* Loop j: from its head with the element E, a thread stores zero at each
   element E, E + N, ... below the count, at the variable's address plus
   the element times the width, and leaves the loop, changing no register
   but the address, the element, and the predicate. *)
Lemma loop_run : forall j base, nth_error (zp_bases zp) j = Some base ->
  forall k E m rs, run_ m = Some (at_ (3 * j + 1)%nat O rs) -> rs e = E -> 0 <= E < 2 ^ 32 ->
  C - E <= Z.of_nat k * N -> rs n = C -> rs z = 0 ->
  (forall r, In r (zero_read zp) -> rs r = init r) ->
  exists m' rs', (m <= m')%nat /\ run_ m' = Some (at_ (3 * j + 3)%nat O rs') /\
    (forall r, r <> a -> r <> e -> r <> p -> rs' r = rs r) /\
    forall x, E <= x < C -> (x - E) mod N = 0 ->
      exists m'' s, (m <= m'' < m')%nat /\ run_ m'' = Some s /\
        fetch prog (blk s) (idx s) = Some (IShA None KStore a 0 w (Some (OReg z))) /\
        blk s = (3 * j + 2)%nat /\ regs s a = init base + x * w /\ regs s z = 0.
Proof.
  intros j base Hj. destruct (zf_loops _ _ _ Hform j base Hj) as (_ & _ & Hbody & _).
  assert (Hbr : In base (zero_read zp)).
  { unfold zero_read. apply in_or_app. right. apply in_or_app. right. exact (nth_error_In _ _ Hj). }
  destruct written_distinct as (Hzt & Hza & Hzn & Hze & Hzp & Hta & Htn & Hte & Htp & Han & Hae & Hap & Hne & Hnp & Hep).
  pose proof threads_pos as HN.
  induction k as [|k IH]; intros E m rs Hrun HE HE0 Hk Hn Hz Hread;
    destruct (loop_head j base Hj E m rs Hrun HE HE0 Hn) as (rs1 & H1r & Hrun2);
    destruct (Z.leb_spec C E) as [HCE|HCE].
  1, 3: exists (S (S m)), rs1; split; [lia|]; split; [exact Hrun2|]; split;
        [intros r _ _ Hr; exact (H1r r Hr)|intros x Hx; exfalso; clear - Hx HCE; lia].
  - (* no iteration is left, yet an element is *) cbn in Hk. clear - Hk HCE. lia.
  - destruct (read_apart base Hbr) as (_ & _ & _ & _ & _ & Hbp).
    destruct (loop_body j base Hj E (S (S m)) rs1 Hrun2
                ltac:(rewrite H1r by (intros Hh; apply Hep; exact Hh); exact HE) ltac:(lia)
                ltac:(rewrite H1r by exact Hzp; exact Hz)
                ltac:(rewrite H1r by exact Hbp; exact (Hread base Hbr)))
      as (rs3 & rs5 & Hrun3 & H3a & H3z & Hrun5 & H5e & H5r).
    assert (H5 : forall r, r <> a -> r <> e -> r <> p -> rs5 r = rs r)
      by (intros r Hra Hre Hrp; rewrite H5r by assumption; exact (H1r r Hrp)).
    destruct (IH (E + N) _ rs5 Hrun5 H5e ltac:(clear - HE0 HCE Hcount HN; lia)
                 ltac:(rewrite Nat2Z.inj_succ, Z.mul_succ_l in Hk; clear - Hk; lia)
                 ltac:(rewrite H5 by (exact (not_eq_sym Han) || exact Hne || exact Hnp); exact Hn)
                 ltac:(rewrite H5 by (exact Hza || exact Hze || exact Hzp); exact Hz)
                 ltac:(intros r Hr; destruct (read_apart r Hr) as (_ & _ & Hra & _ & Hre & Hrp);
                       rewrite H5 by assumption; exact (Hread r Hr)))
      as (m' & rs' & Hm' & Hrun' & Hframe & Hstores).
    exists m', rs'. split; [lia|]. split; [exact Hrun'|]. split.
    + intros r Hra Hre Hrp. rewrite Hframe by assumption. exact (H5 r Hra Hre Hrp).
    + intros x Hx Hmod. destruct (Z.eq_dec x E) as [->|HxE].
      * exists (S (S m) + 2)%nat, (at_ (3 * j + 2)%nat 2%nat rs3).
        split; [lia|]. split; [exact Hrun3|]. split; [apply (fetch_body _ _ _ _ Hbody); reflexivity|].
        split; [reflexivity|]. split; [exact H3a|exact H3z].
      * assert (HxN : E + N <= x) by (pose proof (mod_step N (x - E) HN ltac:(lia) Hmod); lia).
        assert (Hmod' : (x - (E + N)) mod N = 0).
        { replace (x - (E + N)) with (x - E + (-1) * N) by ring. rewrite Z.mod_add by lia. exact Hmod. }
        destruct (Hstores x ltac:(lia) Hmod') as (m'' & s & Hm'' & Hs).
        exists m'', s. split; [lia|exact Hs].
Qed.

Lemma start_setup : forall j base, nth_error (zp_bases zp) j = Some base ->
  skipn (snd (zero_start j)) (block_body prog (fst (zero_start j))) =
    zero_setup_ir zp (zero_temps_of prog zp j) /\
  S (fst (zero_start j)) = (3 * j + 1)%nat.
Proof.
  intros j base Hj. destruct (zf_loops _ _ _ Hform j base Hj) as (Hs & _).
  destruct j as [|j]; cbn [zero_start fst snd]; (split; [|lia]).
  - exact Hs.
  - cbn [skipn]. replace (3 * j + 3)%nat with (3 * S j)%nat by lia. exact Hs.
Qed.

Lemma element_of_thread : forall x, 0 <= x -> x mod N = T -> T <= x /\ (x - T) mod N = 0.
Proof.
  intros x Hx Hmod. pose proof threads_pos as HN. rewrite <- Hmod. split.
  - apply Z.mod_le; lia.
  - rewrite (Z.mod_eq x N) by lia. replace (x - (x - N * (x / N))) with ((x / N) * N) by ring.
    apply Z.mod_mul. lia.
Qed.

(* The loops before loop j: a thread reaches the count of loop j with zero
   in z, its index in t, and the registers that the loops read as at
   launch, having stored each element of each earlier loop that is its own. *)
Lemma loops_run : forall j, (j <= length (zp_bases zp))%nat ->
  exists m rs, run_ m = Some (at_ (fst (zero_start j)) (snd (zero_start j)) rs) /\
    rs z = 0 /\ rs t = T /\ (forall r, In r (zero_read zp) -> rs r = init r) /\
    forall j' base x, (j' < j)%nat -> nth_error (zp_bases zp) j' = Some base -> 0 <= x < C -> x mod N = T ->
      exists m'' s, (m'' < m)%nat /\ run_ m'' = Some s /\
        fetch prog (blk s) (idx s) = Some (IShA None KStore a 0 w (Some (OReg z))) /\
        blk s = (3 * j' + 2)%nat /\ regs s a = init base + x * w /\ regs s z = 0.
Proof.
  induction j as [|j IH]; intros Hj.
  - destruct (zp_bases zp) as [|b0 bs] eqn:Hb; [contradiction|].
    destruct (zf_loops _ _ _ Hform 0 b0 ltac:(rewrite Hb; reflexivity)) as (Hs0 & _).
    assert (Hb0 : skipn 0 (block_body prog 0) =
                  app (zero_index_ir zp) (zero_setup_ir zp (zero_temps_of prog zp 0))).
    { cbn [skipn]. rewrite <- (zf_index _ _ _ Hform), <- Hs0. cbn [zero_setup_block].
      symmetry. apply firstn_skipn. }
    assert (Hrun0 : run_ 0 = Some (at_ O O init)) by reflexivity.
    assert (Hne : zero_setup_ir zp (zero_temps_of prog zp 0) <> []) by (unfold zero_setup_ir; discriminate).
    pose proof (run_plain_inside prog wd mem arb shm init cells locals lwin _ _ _ _ _ _ Hrun0 Hb0 Hne eq_refl)
      as Hrun6.
    destruct (index_regs _ eq_refl) as (Hz & Ht & Hr).
    exists 6%nat, (fold_left exec (zero_index_ir zp) init). split; [exact Hrun6|].
    split; [exact Hz|]. split; [exact Ht|]. split.
    + intros r Hrd. destruct (read_apart r Hrd) as (Hrz & Hrt & Hra & _). exact (Hr r Hrz Hrt Hra).
    + intros j' base x Hj'. lia.
  - destruct (nth_error (zp_bases zp) j) as [base|] eqn:Hjb.
    2: { apply nth_error_None in Hjb. lia. }
    destruct (IH ltac:(lia)) as (m & rs & Hrun & Hz & Ht & Hread & Hstores).
    destruct (start_setup j base Hjb) as [Hskip Hhead].
    destruct (setup_run j base _ _ m rs Hjb Hskip Hrun Hz Ht Hread)
      as (m1 & rs1 & Hm1 & Hrun1 & H1z & H1t & H1n & H1e & H1read).
    rewrite Hhead in Hrun1.
    pose proof thread_bounds as HT. pose proof threads_pos as HN. pose proof threads_small as Hsmall.
    assert (HCN : C - T <= Z.of_nat (Z.to_nat C) * N).
    { rewrite Z2Nat.id by (clear - Hcount; lia).
      assert (C * 1 <= C * N) by (apply Z.mul_le_mono_nonneg_l; clear - Hcount HN; lia). lia. }
    destruct (loop_run j base Hjb (Z.to_nat C) T m1 rs1 Hrun1 H1e ltac:(clear - HT Hsmall; lia) HCN
                H1n H1z H1read)
      as (m2 & rs2 & Hm2 & Hrun2 & Hframe & Hst).
    destruct written_distinct as (Hzt & Hza & Hzn & Hze & Hzp & Hta & Htn & Hte & Htp & _).
    exists m2, rs2. cbn [zero_start fst snd]. split; [exact Hrun2|].
    split; [rewrite Hframe by assumption; exact H1z|].
    split; [rewrite Hframe by assumption; exact H1t|]. split.
    + intros r Hr. destruct (read_apart r Hr) as (_ & _ & Hra & _ & Hre & Hrp).
      rewrite Hframe by assumption. exact (H1read r Hr).
    + intros j' base' x Hj' Hb' Hx Hmod. destruct (Nat.lt_ge_cases j' j) as [Hlt|Hge].
      * destruct (Hstores j' base' x Hlt Hb' Hx Hmod) as (m'' & s & Hm'' & Hs).
        exists m'', s. split; [lia|exact Hs].
      * assert (j' = j) as -> by lia. rewrite Hjb in Hb'. injection Hb' as <-.
        destruct (element_of_thread x (proj1 Hx) Hmod) as [HTx Hmod'].
        destruct (Hst x ltac:(lia) Hmod') as (m'' & s & Hm'' & Hs).
        exists m'', s. split; [lia|exact Hs].
Qed.

(* A thread's run: it leaves the last loop, at the start of block 3m for m
   dynamic variables, where the stores of the other shared variables and the
   barrier follow, having stored zero at each element of each variable that
   is its own: below the count, with the thread's linear index as its
   remainder modulo N. *)
Theorem zero_loops_thread :
  exists m rs, run_ m = Some (at_ (3 * length (zp_bases zp))%nat O rs) /\
    forall j base x, nth_error (zp_bases zp) j = Some base -> 0 <= x < C -> x mod N = T ->
      exists m'' s, (m'' < m)%nat /\ run_ m'' = Some s /\
        fetch prog (blk s) (idx s) = Some (IShA None KStore a 0 w (Some (OReg z))) /\
        blk s = (3 * j + 2)%nat /\ regs s a = init base + x * w /\ regs s z = 0.
Proof.
  destruct (length (zp_bases zp)) as [|k] eqn:Hlen.
  { apply length_zero_iff_nil in Hlen. contradiction. }
  destruct (loops_run (S k) ltac:(lia)) as (m & rs & Hrun & _ & _ & _ & Hstores).
  exists m, rs. cbn [zero_start fst snd] in Hrun. split.
  - replace (3 * S k)%nat with (3 * k + 3)%nat by lia. exact Hrun.
  - intros j base x Hj Hx Hmod.
    assert (Hjl : (j < length (zp_bases zp))%nat) by (apply nth_error_Some; rewrite Hj; discriminate).
    rewrite Hlen in Hjl. exact (Hstores j base x Hjl Hj Hx Hmod).
Qed.

End Loops.

(* ------------------------------------------------------------------ *)
(* Every element has a thread of the block.                            *)

(* Element x of a loop belongs to the thread whose linear index is
   x mod N; that thread's index in each dimension lies in the block. *)
Lemma zero_element_thread : forall bx by_ bz x, 0 < bx -> 0 < by_ -> 0 < bz -> 0 <= x ->
  let t := x mod (bx * by_ * bz) in
  0 <= t mod bx < bx /\ 0 <= (t / bx) mod by_ < by_ /\ 0 <= t / (bx * by_) < bz /\
  (t / (bx * by_) * by_ + (t / bx) mod by_) * bx + t mod bx = t.
Proof.
  intros bx by_ bz x Hx Hy Hz H0 t.
  assert (Hxy : 0 < bx * by_) by (apply Z.mul_pos_pos; lia).
  assert (Ht : 0 <= t < bx * by_ * bz) by (apply Z.mod_pos_bound; apply Z.mul_pos_pos; lia).
  split; [apply Z.mod_pos_bound; lia|]. split; [apply Z.mod_pos_bound; lia|]. split.
  - split; [apply Z.div_pos; lia|]. apply Z.div_lt_upper_bound; lia.
  - rewrite <- Z.div_div by lia.
    assert (E1 : t / bx / by_ * by_ + (t / bx) mod by_ = t / bx)
      by (rewrite Z.mul_comm; symmetry; apply Z.div_mod; lia).
    rewrite E1, Z.mul_comm. symmetry. apply Z.div_mod. lia.
Qed.

(* ------------------------------------------------------------------ *)
(* The count is the binding's bytes.                                   *)

From Coq Require Import Strings.String.
Local Open Scope string_scope.


(* The bytes of dynamic shared memory that the binding states, at the
   values rs of the registers that the launch facts name for the sizes. *)
Definition shared_bytes (F : lfacts) (Mf : binding) (rs : reg -> Z) : Z :=
  af_const (b_shared Mf) +
  fold_right (fun x acc => match reg_named F (size_name (fst x)) with
                           | Some s => snd x * rs s
                           | None => 0
                           end + acc) 0 (af_terms (b_shared Mf)).

Lemma shared_width_divides : forall vs nm al b, In (nm, al, b) vs -> b mod shared_width vs = 0.
Proof.
  intros vs nm al b Hin.
  assert (G : forall w, forallb (fun v : String.string * Z * Z => let '(_, align, bytes) := v in
                                   Z.eqb (align mod w) 0 && Z.eqb (bytes mod w) 0) vs = true -> b mod w = 0).
  { intros w Hw. rewrite forallb_forall in Hw. specialize (Hw _ Hin). cbn in Hw.
    apply andb_true_iff in Hw. apply Z.eqb_eq. exact (proj2 Hw). }
  unfold shared_width. cbv zeta.
  destruct (forallb (fun v : String.string * Z * Z => let '(_, align, bytes) := v in
                       Z.eqb (align mod 4) 0 && Z.eqb (bytes mod 4) 0) vs) eqn:H4; [exact (G 4 H4)|].
  destruct (forallb (fun v : String.string * Z * Z => let '(_, align, bytes) := v in
                       Z.eqb (align mod 2) 0 && Z.eqb (bytes mod 2) 0) vs) eqn:H2; [exact (G 2 H2)|].
  apply Z.mod_1_r.
Qed.

Lemma shared_width_pos : forall vs, 0 < shared_width vs.
Proof.
  intros vs. unfold shared_width. cbv zeta.
  destruct (forallb _ vs); [lia|]. destruct (forallb _ vs); lia.
Qed.

Lemma zero_terms_count : forall l F w terms rs,
  option_all (map (fun x => match reg_named F (size_name (fst x)) with
                            | Some s => Some (s, snd x / w)
                            | None => None
                            end) l) = Some terms ->
  (forall x, In x l -> snd x mod w = 0) -> 0 < w ->
  zero_sum rs terms * w =
  fold_right (fun x acc => match reg_named F (size_name (fst x)) with
                           | Some s => snd x * rs s
                           | None => 0
                           end + acc) 0 l.
Proof.
  induction l as [|[k c] l IH]; intros F w terms rs Hall Hdiv Hw.
  - cbn in Hall. injection Hall as <-. reflexivity.
  - cbn [map option_all fst snd] in Hall. cbn [fold_right fst snd].
    destruct (reg_named F (size_name k)) as [s|]; [|discriminate].
    destruct (option_all _) as [ts|] eqn:Hts; [|discriminate]. injection Hall as <-.
    cbn [zero_sum]. rewrite Z.mul_add_distr_r, (IH F w ts rs Hts (fun x H => Hdiv x (or_intror H)) Hw).
    f_equal. specialize (Hdiv (k, c) (or_introl eq_refl)). cbn [snd] in Hdiv.
    assert (Ec : c / w * w = c) by (rewrite Z.mul_comm; symmetry; apply Z.div_exact; lia).
    transitivity (c / w * w * rs s); [ring|]. rewrite Ec. reflexivity.
Qed.

(* The count of each loop, in elements, times the width of a store is the
   binding's bytes of dynamic shared memory. *)
Theorem zero_plan_count : forall K Mf F prog zp,
  zero_plan_of K Mf F prog = Some zp -> zeroed_dynamic K Mf <> [] ->
  forall rs, zero_elements zp rs * zp_width zp = shared_bytes F Mf rs.
Proof.
  intros K Mf F prog zp H Hdyn rs. unfold zero_plan_of in H.
  destruct (b_block Mf) as [[bx by_] bz].
  destruct (reg_named F "%tid.x"); [|discriminate]. destruct (reg_named F "%tid.y"); [|discriminate].
  destruct (reg_named F "%tid.z"); [|discriminate].
  destruct (zero_terms_of Mf F (shared_width (width_entries K Mf))) as [terms|] eqn:Hterms; [|discriminate].
  destruct (option_all _); [|discriminate]. destruct (zero_regs_of prog); [|discriminate].
  injection H as <-. unfold zero_elements, shared_bytes. cbn [zp_const zp_terms zp_width].
  set (w := shared_width (width_entries K Mf)).
  assert (Hw : 0 < w) by apply shared_width_pos.
  destruct (zeroed_dynamic K Mf) as [|[name align] rest] eqn:Hd; [contradiction|].
  assert (Hdiv : forall c, In c (af_const (b_shared Mf) :: map snd (af_terms (b_shared Mf))) -> c mod w = 0).
  { intros c Hc. apply (shared_width_divides _ name align c). unfold width_entries. rewrite Hd.
    apply in_or_app. right. cbn [flat_map]. apply in_or_app. left. apply in_map_iff.
    exists c. split; [reflexivity|exact Hc]. }
  rewrite Z.mul_add_distr_r.
  rewrite (zero_terms_count _ F w terms rs Hterms) by
    (try exact Hw; intros x Hx; apply Hdiv; right; apply in_map; exact Hx).
  f_equal. rewrite Z.mul_comm. symmetry. apply Z.div_exact; [lia|].
  exact (Hdiv _ (or_introl eq_refl)).
Qed.

(* ------------------------------------------------------------------ *)
(* Every accepted kernel.                                              *)

Local Transparent front.

(* A kernel that the gate accepts, whose prologue clears dynamic shared
   memory with loops, has lowered loops of their plan's form. *)
Theorem accepted_zero_plan : forall K Mf prog F,
  front K Mf = Ok (prog, F) -> zeroed_dynamic K Mf <> [] ->
  exists zp, zero_plan_of K Mf F prog = Some zp /\ zero_form (wdn F) zp prog.
Proof.
  intros K Mf prog F Hfront Hdyn.
  unfold front, front_core, front_lower in Hfront.
  destruct (zeroing_check K Mf); [|discriminate].
  destruct (inline K) as [[K' org]|e i]; [|discriminate].
  destruct (defined_check K' org); [|discriminate].
  destruct ((p <- translate_placed K' Mf org ;; F <- facts K' Mf ;; ret (p,F)) init)
    as [[[pp raw] ts]|msg where_]; [|discriminate].
  destruct (zero_program_check K Mf (payload_facts raw (map (map fst) pp)) (map (map fst) pp)) as [[]|e i]
    eqn:Hcheck; [|discriminate].
  destruct (barrier_boundary _ _ _ _ _ _); [|discriminate].
  injection Hfront as <- <-.
  unfold zero_program_check in Hcheck.
  destruct (null (zeroed_dynamic K Mf)) eqn:Hnull.
  { destruct (zeroed_dynamic K Mf); [contradiction|discriminate]. }
  destruct (zero_plan_of _ _ _ _) as [zp|]; [|discriminate].
  destruct (zero_program_ok _ _ _) eqn:Hok; [|discriminate].
  exists zp. split; [reflexivity|]. exact (zero_program_ok_form _ _ _ Hok).
Qed.

Lemma zero_plan_bases : forall K Mf F prog zp,
  zero_plan_of K Mf F prog = Some zp -> zeroed_dynamic K Mf <> [] -> zp_bases zp <> [].
Proof.
  intros K Mf F prog zp H Hdyn. unfold zero_plan_of in H.
  destruct (b_block Mf) as [[bx by_] bz].
  destruct (reg_named F "%tid.x"); [|discriminate]. destruct (reg_named F "%tid.y"); [|discriminate].
  destruct (reg_named F "%tid.z"); [|discriminate].
  destruct (zero_terms_of _ _ _); [|discriminate].
  destruct (option_all _) as [bases|] eqn:Hb; [|discriminate]. destruct (zero_regs_of prog); [|discriminate].
  injection H as <-. cbn [zp_bases]. intros ->.
  destruct (zeroed_dynamic K Mf); [contradiction|]. cbn in Hb.
  destruct (reg_named _ _); [|discriminate]. destruct (option_all _); discriminate.
Qed.

(* The effect of the loops in every accepted kernel: in every run of a
   thread whose launch values put its index in the block, the sizes in 64
   bits, the binding's bytes D of dynamic shared memory in [0, shared_max]
   as the launch check requires, and each dynamic variable's address with D
   below 2^32, the thread reaches the start of block 3m after its loops,
   where the stores of the other shared variables and the barrier follow,
   and before that it stored zero, in w bytes at the variable's address
   plus (x / w) * w, for each byte x below D whose element x / w has the
   thread's linear index as its remainder modulo N. By
   `zero_element_thread`, each element has such a thread in the block, so
   the block's threads store zero at every byte of each dynamic variable
   before the barrier. *)
Theorem accepted_zeroing : forall K Mf prog F,
  front K Mf = Ok (prog, F) -> zeroed_dynamic K Mf <> [] ->
  exists zp, zero_plan_of K Mf F prog = Some zp /\
  forall E,
    0 <= e_init E (zp_tx zp) < zp_bx zp -> 0 <= e_init E (zp_ty zp) < zp_by zp ->
    0 <= e_init E (zp_tz zp) < zp_bz zp ->
    (forall s c, In (s, c) (zp_terms zp) -> 0 <= e_init E s < 2 ^ 64) ->
    0 <= shared_bytes F Mf (e_init E) <= shared_max ->
    (forall B, In B (zp_bases zp) -> 0 <= e_init E B /\ e_init E B + shared_bytes F Mf (e_init E) < 2 ^ 32) ->
    exists m s, run' F prog E m = Some s /\ blk s = (3 * List.length (zp_bases zp))%nat /\ idx s = O /\
    forall j B x, nth_error (zp_bases zp) j = Some B -> 0 <= x < shared_bytes F Mf (e_init E) ->
      (x / zp_width zp) mod zp_threads zp = zero_thread zp (e_init E) ->
      exists m' s', (m' < m)%nat /\ run' F prog E m' = Some s' /\
        fetch prog (blk s') (idx s') =
          Some (IShA None KStore (zr_address (zp_regs zp)) 0 (zp_width zp) (Some (OReg (zr_zero (zp_regs zp))))) /\
        blk s' = (3 * j + 2)%nat /\
        regs s' (zr_address (zp_regs zp)) = e_init E B + x / zp_width zp * zp_width zp /\
        e_init E B + x / zp_width zp * zp_width zp <= e_init E B + x <
          e_init E B + x / zp_width zp * zp_width zp + zp_width zp /\
        regs s' (zr_zero (zp_regs zp)) = 0.
Proof.
  intros K Mf prog F Hfront Hdyn.
  destruct (accepted_zero_plan K Mf prog F Hfront Hdyn) as (zp & Hplan & Hform).
  exists zp. split; [exact Hplan|].
  intros E Htx Hty Htz Hsizes HD Hbases.
  pose proof (zero_plan_count K Mf F prog zp Hplan Hdyn (e_init E)) as Hcount.
  set (D := shared_bytes F Mf (e_init E)) in *.
  pose proof (zf_width _ _ _ Hform) as Hw. pose proof (zf_threads _ _ _ Hform) as HN.
  pose proof (zf_bx _ _ _ Hform). pose proof (zf_by _ _ _ Hform). pose proof (zf_bz _ _ _ Hform).
  assert (HN0 : 0 < zp_threads zp) by (unfold zp_threads; apply Z.mul_pos_pos; [apply Z.mul_pos_pos|]; lia).
  set (C := zero_elements zp (e_init E)) in *.
  assert (HC : 0 <= C <= D).
  { split.
    - destruct (Z.lt_ge_cases C 0) as [Hneg|]; [|assumption].
      assert (C * zp_width zp < 0) by (apply Z.mul_neg_pos; lia). unfold shared_max in HD. lia.
    - rewrite <- Hcount. assert (C * 1 <= C * zp_width zp) by (apply Z.mul_le_mono_nonneg_l; lia). lia. }
  assert (Hcount' : 0 <= C /\ C + zp_threads zp < 2 ^ 32) by (unfold shared_max in HD; lia).
  assert (Hbases' : forall B, In B (zp_bases zp) -> 0 <= e_init E B /\ e_init E B + C * zp_width zp < 2 ^ 32)
    by (intros B HB; rewrite Hcount; exact (Hbases B HB)).
  destruct (zero_loops_thread prog (wdn F) (e_mem E) (e_arb E) (e_shm E) (e_init E) (lf_cells F)
              (lf_locals F) (lf_local_window F) zp Hform Htx Hty Htz Hsizes Hcount' Hbases'
              (zero_plan_bases K Mf F prog zp Hplan Hdyn))
    as (m & rs & Hrun & Hstores).
  exists m, {| blk := (3 * List.length (zp_bases zp))%nat; idx := O; regs := rs |}.
  split; [exact Hrun|]. split; [reflexivity|]. split; [reflexivity|].
  intros j B x Hj Hx Hmod.
  assert (Hxw : 0 <= x / zp_width zp < C).
  { split; [apply Z.div_pos; lia|]. apply Z.div_lt_upper_bound; [lia|]. rewrite Z.mul_comm, Hcount. lia. }
  destruct (Hstores j B (x / zp_width zp) Hj Hxw Hmod) as (m' & s' & Hm' & Hrun' & Hf & Hb & Ha & Hz).
  exists m', s'. split; [exact Hm'|]. split; [exact Hrun'|]. split; [exact Hf|]. split; [exact Hb|].
  split; [exact Ha|]. split; [|exact Hz].
  pose proof (Z.mul_div_le x (zp_width zp) ltac:(lia)).
  pose proof (Z.mod_pos_bound x (zp_width zp) ltac:(lia)).
  pose proof (Z.div_mod x (zp_width zp) ltac:(lia)). lia.
Qed.
