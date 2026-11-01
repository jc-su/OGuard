(* StoreLaunch.v: a launch as every thread of a grid over byte-addressed
   memory.

   Global memory holds a byte at each address. Each block of the launch has
   a shared memory and each thread a local memory, which start with
   whatever the launches before it left there. A thread runs the one-thread
   semantics of GateIsa. At its t-th step it reads each memory as the
   memory is then: a load of b bits at address x reads the b / 8 bytes from
   x, least significant first. A shared memory load reads its block's
   shared memory, and a load at a generic address in the window of shared
   or local memory reads that memory. A store of known value writes that
   value's bytes; an atomic, a store whose value the model does not know,
   and a shared memory access that the model does not decode write the
   bytes that the thread's oracle gives. The results of the instructions
   that the model does not decode come from the oracle too. An oracle is a
   function of the step, of the bytes of the block's shared variables, and
   of the bytes of the thread's local variables at that step: what a thread
   can read of those memories. A schedule, a list of threads, orders the
   steps. The prologue's barrier (GatePrologue.v) holds a thread that has
   reached it until every thread of its block has reached it.

   The run of each thread is then a run of the one-thread semantics in an
   environment whose memories at step t are the memories when the thread
   took its t-th step (`thread_ok`). So the gate's theorem, `kernel_safe`,
   holds of every step of every thread, once each thread's environment has
   the launch facts; those depend on global memory only through an
   invariant that the runtime establishes before the launch and that each
   write the gate allows keeps (`launch_inv`). A launch keeps every byte of
   global memory outside those it may write (`launch_frame`). Two runs of
   one launch with the same schedule and oracles, from global memories that
   agree on the bytes it may read, from local memories that agree on the
   bytes of each thread's local variables, and from shared memories that
   may differ in every byte, agree after it on the bytes it may read
   (`launch_ni`): a load of local memory reads bytes of the thread's local
   variables, which the two runs write alike; before any thread of a block
   passes the prologue's barrier, the block's threads have stored zero over
   every byte of its shared variables, and until then no thread of the
   block reads shared memory.

   Trusted: that the hardware runs a launch as some interleaving of its
   threads' steps, in which an access touches exactly its bytes; that the
   values the oracles give include what the hardware computes from the
   thread's registers and the memories the model gives; and that a
   `bar.sync` holds a thread until every thread of its block arrives, and
   makes the stores before it visible to the loads after it. *)

From Coq Require Import ZArith.ZArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Lia.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.
Require Import OpGate.GateDomain.
Require Import OpGate.GateMeaning.
Require Import OpGate.GateThread.
Require Import OpGate.GateFront.
Require Import OpGate.GateZero.
Require Import OpGate.GatePrologue.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Memory.                                                             *)

Definition memory := Z -> Z.

Fixpoint word_of (m : memory) (a : Z) (n : nat) : Z :=
  match n with O => 0 | S n' => m a mod 256 + 256 * word_of m (a + 1) n' end.

(* The 64-bit word at address a, least significant byte first. *)
Definition read_word (m : memory) (a : Z) : Z := word_of m a 8.

(* Memory after the width bytes at a take the bytes of v. *)
Definition write_bytes (m : memory) (a width v : Z) : memory :=
  fun x => if (a <=? x) && (x <? a + width) then (v / 2 ^ (8 * (x - a))) mod 256 else m x.

Lemma pow_bytes : forall n, 2 ^ (8 * Z.of_nat (S n)) = 256 * 2 ^ (8 * Z.of_nat n).
Proof.
  intros n. rewrite Nat2Z.inj_succ, <- Z.add_1_r, Z.mul_add_distr_l, Z.pow_add_r by lia.
  change (2 ^ (8 * 1)) with 256. lia.
Qed.

Lemma word_bound : forall m a n, 0 <= word_of m a n < 2 ^ (8 * Z.of_nat n).
Proof.
  intros m a n. revert a. induction n as [| n IH]; intros a; cbn [word_of]; [cbn; lia |].
  pose proof (Z.mod_pos_bound (m a) 256 ltac:(lia)). specialize (IH (a + 1)).
  rewrite pow_bytes. lia.
Qed.

Lemma word_split : forall m a w k,
    word_of m a (w + k) = word_of m a w + 2 ^ (8 * Z.of_nat w) * word_of m (a + Z.of_nat w) k.
Proof.
  intros m a w. revert a. induction w as [| w IH]; intros a k.
  - cbn [plus]. replace (a + Z.of_nat 0) with a by (cbn; lia). cbn [word_of].
    change (2 ^ (8 * Z.of_nat 0)) with 1. lia.
  - cbn [word_of plus]. rewrite IH, pow_bytes.
    replace (a + 1 + Z.of_nat w) with (a + Z.of_nat (S w)) by lia. ring.
Qed.

Lemma word_agree : forall m1 m2 a w,
    (forall y, a <= y < a + Z.of_nat w -> m1 y = m2 y) -> word_of m1 a w = word_of m2 a w.
Proof.
  intros m1 m2 a w. revert a. induction w as [| w IH]; intros a H; [reflexivity |].
  cbn [word_of]. rewrite (H a ltac:(lia)), (IH (a + 1)); [reflexivity |]. intros y Hy. apply H. lia.
Qed.

(* The low w bytes of the word at a are the bytes from a. *)
Lemma read_word_low : forall m a w, (w <= 8)%nat -> read_word m a mod 2 ^ (8 * Z.of_nat w) = word_of m a w.
Proof.
  intros m a w Hw. unfold read_word.
  assert (E : word_of m a 8 = word_of m a (w + (8 - w))) by (f_equal; lia).
  rewrite E, word_split.
  rewrite (Z.mul_comm (2 ^ (8 * Z.of_nat w)) (word_of m (a + Z.of_nat w) (8 - w))).
  rewrite Z.mod_add by (apply Z.pow_nonzero; lia).
  apply Z.mod_small. apply word_bound.
Qed.

(* A load of w whole bytes reads only those bytes. *)
Lemma trunc_read_agree : forall m1 m2 a w,
    (forall y, a <= y < a + Z.of_nat w -> m1 y = m2 y) ->
    trunc (8 * w) (read_word m1 a) = trunc (8 * w) (read_word m2 a).
Proof.
  intros m1 m2 a w H. unfold trunc. rewrite Nat2Z.inj_mul. change (Z.of_nat 8) with 8.
  destruct (Nat.le_gt_cases w 8) as [Hw | Hw].
  - rewrite !read_word_low by exact Hw. apply word_agree. exact H.
  - unfold read_word. rewrite !Z.mod_small.
    + apply word_agree. intros y Hy. apply H. lia.
    + pose proof (word_bound m2 a 8). split; [lia |].
      apply (Z.lt_le_trans _ (2 ^ (8 * Z.of_nat 8))); [lia | apply Z.pow_le_mono_r; lia].
    + pose proof (word_bound m1 a 8). split; [lia |].
      apply (Z.lt_le_trans _ (2 ^ (8 * Z.of_nat 8))); [lia | apply Z.pow_le_mono_r; lia].
Qed.



(* Memory that holds zero at every address. *)
Definition zero : memory := fun _ => 0.

(* What the model leaves open in one thread, as functions of the step, of
   the bytes of the block's shared variables, and of the bytes of the
   thread's local variables at the step: the results of the instructions
   that the model does not decode, and the bytes that an atomic, a store of
   unknown value, or a shared memory access that the model does not decode
   writes. The bytes of the variables are lists, so that two memories that
   hold the same bytes there give the oracle the same input. *)
Record oracle := {
  o_arb : nat -> reg -> list Z -> list Z -> Z;
  o_wr : nat -> list Z -> list Z -> Z
}.

(* A thread: its state, the count of its steps, global memory, its block's
   shared memory, and its local memory at each of them, its local memory
   now, and whether it has executed the prologue's barrier. *)
Record thread := {
  th_state : state;
  th_count : nat;
  th_hist : nat -> memory;
  th_shist : nat -> memory;
  th_lhist : nat -> memory;
  th_local : memory;
  th_passed : bool
}.

(* Global memory, the shared memory of each block, and the threads. *)
Record grid := { g_mem : memory; g_shared : nat -> memory; g_threads : list thread }.

Lemma flat_map_ext_in : forall {A B : Type} (f g : A -> list B) (l : list A),
    (forall a, In a l -> f a = g a) -> flat_map f l = flat_map g l.
Proof.
  intros A B f g l. induction l as [| a l IH]; intros H; [reflexivity |].
  cbn [flat_map]. rewrite (H a (or_introl eq_refl)), (IH (fun a' Ha' => H a' (or_intror Ha'))). reflexivity.
Qed.

Fixpoint set_nth {A : Type} (l : list A) (n : nat) (x : A) : list A :=
  match l, n with
  | [], _ => []
  | _ :: t, O => x :: t
  | y :: t, S n' => y :: set_nth t n' x
  end.

Section Launch.

Variable F : lfacts.
Variable prog : program.
(* The position of the prologue's barrier, for a kernel with shared
   variables. *)
Variable bar : option (nat * nat).

(* The bytes of the shared variables, at the launch values init: what a
   thread of the block can read of its shared memory, in the order of the
   variables. *)
Definition shared_bytes (init : reg -> Z) (m : memory) : list Z :=
  flat_map (fun v => map (fun k => m (init (fst v) + Z.of_nat k)) (seq 0 (Z.to_nat (fval F init (snd v)))))
           (lf_shared F).

Lemma shared_bytes_agree : forall init m1 m2,
    (forall x, declared F init x -> m1 x = m2 x) -> shared_bytes init m1 = shared_bytes init m2.
Proof.
  intros init m1 m2 H. unfold shared_bytes. apply flat_map_ext_in.
  intros [u f] Hin. apply map_ext_in. intros k Hk. apply in_seq in Hk. destruct Hk as [_ Hk].
  cbn [fst snd] in *. apply H. exists u, f. split; [exact Hin |].
  destruct (Z.le_gt_cases 0 (fval F init f)) as [Hf | Hf].
  - apply Nat2Z.inj_lt in Hk. rewrite Nat2Z.inj_add, Z2Nat.id in Hk by exact Hf. cbn in Hk. lia.
  - destruct (fval F init f) as [| p | p] eqn:E; [lia | lia |]. cbn in Hk. lia.
Qed.

(* A byte of the thread's local variables, at the launch values init. *)
Definition declared_local (init : reg -> Z) (x : Z) : Prop :=
  exists u f, In (u, f) (local_vars F) /\ init u <= x < init u + fval F init f.

(* The bytes of the local variables, at the launch values init: what a
   thread can read of its local memory, in the order of the variables. *)
Definition local_bytes (init : reg -> Z) (m : memory) : list Z :=
  flat_map (fun v => map (fun k => m (init (fst v) + Z.of_nat k)) (seq 0 (Z.to_nat (fval F init (snd v)))))
           (local_vars F).

Lemma local_bytes_agree : forall init m1 m2,
    (forall x, declared_local init x -> m1 x = m2 x) -> local_bytes init m1 = local_bytes init m2.
Proof.
  intros init m1 m2 H. unfold local_bytes. apply flat_map_ext_in.
  intros [u f] Hin. apply map_ext_in. intros k Hk. apply in_seq in Hk. destruct Hk as [_ Hk].
  cbn [fst snd] in *. apply H. exists u, f. split; [exact Hin |].
  destruct (Z.le_gt_cases 0 (fval F init f)) as [Hf | Hf].
  - apply Nat2Z.inj_lt in Hk. rewrite Nat2Z.inj_add, Z2Nat.id in Hk by exact Hf. cbn in Hk. lia.
  - destruct (fval F init f) as [| p | p] eqn:E; [lia | lia |]. cbn in Hk. lia.
Qed.

(* The generic addresses of the window whose base register w holds. *)
Definition in_space (init : reg -> Z) (w : reg) (x : Z) : bool :=
  (init w <=? x) && (x <? init w + window_size).

(* The generic addresses of a window, with the windows' bases at launch. *)
Definition in_windows (init : reg -> Z) (x : Z) : bool :=
  existsb (fun w => in_space init w x) [lf_shared_window F; lf_local_window F].

(* The environment of a thread with launch values init and oracle o, with
   global memory hist t, its block's shared memory sh t, and its local
   memory lh t at its step t. *)
Definition env_of (init : reg -> Z) (o : oracle) (hist sh lh : nat -> memory) : env :=
  {| e_mem := fun t a =>
       if in_space init (lf_shared_window F) a then read_word (sh t) (a - init (lf_shared_window F))
       else if in_space init (lf_local_window F) a then read_word (lh t) (a - init (lf_local_window F))
       else read_word (hist t) a;
     e_arb := fun t r => o_arb o t r (shared_bytes init (sh t)) (local_bytes init (lh t));
     e_shm := fun t a => read_word (sh t) a;
     e_init := init |}.

Definition thread_run (init : reg -> Z) (o : oracle) (hist sh lh : nat -> memory) (n : nat) : option state :=
  run' F prog (env_of init o hist sh lh) n.

(* The width bytes at generic address x lie in a window: in_window of an
   environment with launch values init. *)
Definition win_of (init : reg -> Z) (x width : Z) : bool :=
  existsb (fun w => (init w <=? x) && (x + width <=? init w + window_size)) [lf_shared_window F; lf_local_window F].

Lemma win_of_env : forall E, in_window F E = win_of (e_init E).
Proof. intros E. reflexivity. Qed.

(* The value that a store or an atomic of kind k writes: a store's known
   value, or else the oracle's bytes wr. *)
Definition stored (rs : reg -> Z) (k : mkind) (v : option opnd) (wr : Z) : Z :=
  match k, v with KStore, Some o => opv rs o | _, _ => wr end.

(* Global memory after instruction i, in register state rs, writes: a store
   or an atomic in global memory, with wr for a value the model does not
   know. *)
Definition mem_write (init : reg -> Z) (wr : Z) (rs : reg -> Z) (i : instr) (m : memory) : memory :=
  match access_at (win_of init) rs i with
  | Some (KLoad, _, _, _) | None => m
  | Some (_, x, width, v) => write_bytes m x width (match v with Some z => z | None => wr end)
  end.

(* The width bytes at generic address x lie in the window whose base
   register w holds. *)
Definition within (init : reg -> Z) (w : reg) (x width : Z) : bool :=
  (init w <=? x) && (x + width <=? init w + window_size).

(* The block's shared memory after instruction i writes: a shared memory
   store or atomic writes its value, or the bytes wr when the model does
   not know it, and a generic store or atomic in the window of shared
   memory writes there. *)
Definition shared_write (init : reg -> Z) (wr : Z) (rs : reg -> Z) (i : instr) (sm : memory) : memory :=
  if negb (holds (guard_of i) rs) then sm else
  match i with
  | IShA _ KLoad _ _ _ _ => sm
  | IShA _ k a off width v => write_bytes sm (trunc 32 (rs a + off)) width (stored rs k v wr)
  | IMem _ KLoad _ _ _ _ _ _ _ => sm
  | IMem _ k true a off width v _ _ =>
      let x := trunc 64 (rs a + off) in
      if within init (lf_shared_window F) x width
      then write_bytes sm (x - init (lf_shared_window F)) width (stored rs k v wr) else sm
  | _ => sm
  end.

(* The thread's local memory after instruction i writes: a local store, and
   a generic store or atomic in the window of local memory. *)
Definition local_write (init : reg -> Z) (wr : Z) (rs : reg -> Z) (i : instr) (lm : memory) : memory :=
  if negb (holds (guard_of i) rs) then lm else
  match i with
  | IStL _ a off width v _ =>
      write_bytes lm (trunc 64 (rs a + off)) width (match v with Some o => opv rs o | None => wr end)
  | IMem _ KLoad _ _ _ _ _ _ _ => lm
  | IMem _ k true a off width v _ _ =>
      let x := trunc 64 (rs a + off) in
      if within init (lf_local_window F) x width
      then write_bytes lm (x - init (lf_local_window F)) width (stored rs k v wr) else lm
  | _ => lm
  end.

(* Whether a state is at the prologue's barrier. *)
Definition at_bar (s : state) : bool :=
  match bar with Some (b, i) => Nat.eqb (blk s) b && Nat.eqb (idx s) i | None => false end.

Lemma at_bar_spec : forall s q, bar = Some q -> at_bar s = true <-> (blk s, idx s) = q.
Proof.
  intros s [b i] Hb. unfold at_bar. rewrite Hb. rewrite andb_true_iff, !Nat.eqb_eq. split.
  - intros [-> ->]. reflexivity.
  - intros E. injection E as -> ->. split; reflexivity.
Qed.

(* A thread with launch values init before its first step, in global
   memory m0 and the local memory lm0 that the launches before it left. *)
Definition start (init : reg -> Z) (m0 lm0 : memory) : thread :=
  {| th_state := {| blk := 0; idx := 0; regs := init |}; th_count := 0; th_hist := fun _ => m0;
     th_shist := fun _ => zero; th_lhist := fun _ => lm0; th_local := lm0; th_passed := false |}.

(* One step of a thread in global memory m and its block's shared memory
   sm: None once it has left the program. *)
Definition tstep (init : reg -> Z) (o : oracle) (m sm : memory) (th : thread)
    : option (thread * memory * memory) :=
  let s := th_state th in
  let t := th_count th in
  let hist := fun t' => if Nat.eqb t' t then m else th_hist th t' in
  let sh := fun t' => if Nat.eqb t' t then sm else th_shist th t' in
  let lh := fun t' => if Nat.eqb t' t then th_local th else th_lhist th t' in
  let E := env_of init o hist sh lh in
  match fetch prog (blk s) (idx s) with
  | None => None
  | Some i =>
      match step prog (wdn F) (e_mem E) (e_arb E) (e_shm E) (e_init E)
                 (lf_cells F) (lf_locals F) (lf_local_window F) t s with
      | Some s' =>
          let wr := o_wr o t (shared_bytes init sm) (local_bytes init (th_local th)) in
          Some ({| th_state := s'; th_count := S t; th_hist := hist; th_shist := sh; th_lhist := lh;
                   th_local := local_write init wr (regs s) i (th_local th);
                   th_passed := th_passed th || at_bar s |},
                mem_write init wr (regs s) i m, shared_write init wr (regs s) i sm)
      | None => None
      end
  end.

(* A thread at the barrier or past it. *)
Definition arrived (th : thread) : bool := at_bar (th_state th) || th_passed th.

(* Every thread of thread n's block has arrived at the barrier. *)
Definition block_arrived (cta : nat -> nat) (G : grid) (n : nat) : bool :=
  forallb (fun k => match nth_error (g_threads G) k with
                    | Some th' => if Nat.eqb (cta k) (cta n) then arrived th' else true
                    | None => true
                    end) (seq 0 (List.length (g_threads G))).

(* A step of thread n, whose block is cta n: a thread at the barrier waits
   until its block has arrived. *)
Definition gstep (inits : nat -> reg -> Z) (os : nat -> oracle) (cta : nat -> nat) (G : grid) (n : nat) : grid :=
  match nth_error (g_threads G) n with
  | None => G
  | Some th =>
      if at_bar (th_state th) && negb (block_arrived cta G n) then G else
      match tstep (inits n) (os n) (g_mem G) (g_shared G (cta n)) th with
      | None => G
      | Some (th', m', sm') =>
          {| g_mem := m'; g_shared := fun b => if Nat.eqb b (cta n) then sm' else g_shared G b;
             g_threads := set_nth (g_threads G) n th' |}
      end
  end.

(* A launch of threads 0 to count - 1 from global memory m0, the shared
   memories sh0 of the blocks and the local memories lm0 of the threads
   that the launches before it left, in the order of sched. *)
Definition grid0 (inits : nat -> reg -> Z) (count : nat) (m0 : memory) (sh0 lm0 : nat -> memory) : grid :=
  {| g_mem := m0; g_shared := sh0; g_threads := map (fun n => start (inits n) m0 (lm0 n)) (seq 0 count) |}.

Definition launch (inits : nat -> reg -> Z) (os : nat -> oracle) (cta : nat -> nat) (count : nat) (m0 : memory)
    (sh0 lm0 : nat -> memory) (sched : list nat) : grid :=
  fold_left (gstep inits os cta) sched (grid0 inits count m0 sh0 lm0).

(* ------------------------------------------------------------------ *)
(* A thread runs the one-thread semantics.                            *)

(* A step reads the environment at the step that it takes. *)
Lemma write_all_at : forall wd arb1 arb2 t, arb1 t = arb2 t ->
    forall rs ds, write_all wd arb1 rs t ds = write_all wd arb2 rs t ds.
Proof.
  intros wd arb1 arb2 t H rs ds. revert rs. induction ds as [| d ds IH]; intros rs; [reflexivity |].
  cbn [write_all]. rewrite H. apply IH.
Qed.

Lemma step_at : forall wd mem1 mem2 arb1 arb2 shm1 shm2 init cells locals lw t s,
    mem1 t = mem2 t -> arb1 t = arb2 t -> shm1 t = shm2 t ->
    step prog wd mem1 arb1 shm1 init cells locals lw t s = step prog wd mem2 arb2 shm2 init cells locals lw t s.
Proof.
  intros wd mem1 mem2 arb1 arb2 shm1 shm2 init cells locals lw t s Hm Ha Hs.
  unfold step. destruct (fetch prog (blk s) (idx s)) as [i |]; [| reflexivity].
  destruct (negb (holds (guard_of i) (regs s))); [reflexivity |]. cbn beta iota.
  destruct i; cbn beta iota; unfold gstore, lstore, lstore_cell, lload, lframe, lframe_cell, result;
    rewrite ?(write_all_at wd arb1 arb2 t Ha), ?Hm, ?Ha, ?Hs; reflexivity.
Qed.

Lemma run_ext : forall mem1 mem2 arb1 arb2 shm1 shm2 init cells locals lw n,
    (forall t, (t < n)%nat -> mem1 t = mem2 t /\ arb1 t = arb2 t /\ shm1 t = shm2 t) ->
    run prog (wdn F) mem1 arb1 shm1 init cells locals lw n = run prog (wdn F) mem2 arb2 shm2 init cells locals lw n.
Proof.
  intros mem1 mem2 arb1 arb2 shm1 shm2 init cells locals lw n. induction n as [| n IH]; intros H; [reflexivity |].
  cbn [run]. rewrite IH by (intros t Ht; apply H; lia).
  destruct (run prog (wdn F) mem2 arb2 shm2 init cells locals lw n) as [s |]; [| reflexivity].
  destruct (H n ltac:(lia)) as [Hm [Ha Hs]]. apply step_at; assumption.
Qed.

(* The run of a thread's own steps gives its state. *)
Definition thread_ok (init : reg -> Z) (o : oracle) (th : thread) : Prop :=
  thread_run init o (th_hist th) (th_shist th) (th_lhist th) (th_count th) = Some (th_state th).

Lemma start_ok : forall init o m0 lm0, thread_ok init o (start init m0 lm0).
Proof. intros init o m0 lm0. reflexivity. Qed.

(* Below its step t, the environment that a thread's step t records agrees
   with the environment of its history. *)
Lemma env_before : forall init o th m sm t,
    (t < th_count th)%nat ->
    let E := env_of init o (fun t' => if Nat.eqb t' (th_count th) then m else th_hist th t')
                           (fun t' => if Nat.eqb t' (th_count th) then sm else th_shist th t')
                           (fun t' => if Nat.eqb t' (th_count th) then th_local th else th_lhist th t') in
    let E0 := env_of init o (th_hist th) (th_shist th) (th_lhist th) in
    e_mem E0 t = e_mem E t /\ e_arb E0 t = e_arb E t /\ e_shm E0 t = e_shm E t.
Proof.
  intros init o th m sm t Ht E E0. subst E E0. cbn [env_of e_mem e_arb e_shm].
  destruct (Nat.eqb_spec t (th_count th)); [lia |]. split; [| split]; reflexivity.
Qed.

(* The parts of a thread's step. *)
Lemma tstep_spec : forall init o m sm th th' m' sm',
    tstep init o m sm th = Some (th', m', sm') ->
    exists i, fetch prog (blk (th_state th)) (idx (th_state th)) = Some i /\
      th_count th' = S (th_count th) /\
      th_hist th' = (fun t' => if Nat.eqb t' (th_count th) then m else th_hist th t') /\
      th_shist th' = (fun t' => if Nat.eqb t' (th_count th) then sm else th_shist th t') /\
      th_lhist th' = (fun t' => if Nat.eqb t' (th_count th) then th_local th else th_lhist th t') /\
      step prog (wdn F) (e_mem (env_of init o (th_hist th') (th_shist th') (th_lhist th')))
           (e_arb (env_of init o (th_hist th') (th_shist th') (th_lhist th')))
           (e_shm (env_of init o (th_hist th') (th_shist th') (th_lhist th'))) init
           (lf_cells F) (lf_locals F) (lf_local_window F) (th_count th) (th_state th) = Some (th_state th') /\
      th_local th' = local_write init (o_wr o (th_count th) (shared_bytes init sm) (local_bytes init (th_local th)))
                                 (regs (th_state th)) i (th_local th) /\
      th_passed th' = th_passed th || at_bar (th_state th) /\
      m' = mem_write init (o_wr o (th_count th) (shared_bytes init sm) (local_bytes init (th_local th)))
                     (regs (th_state th)) i m /\
      sm' = shared_write init (o_wr o (th_count th) (shared_bytes init sm) (local_bytes init (th_local th)))
                         (regs (th_state th)) i sm.
Proof.
  intros init o m sm th th' m' sm' H. unfold tstep in H.
  destruct (fetch prog (blk (th_state th)) (idx (th_state th))) as [i |] eqn:Ef; [| discriminate].
  destruct (step _ _ _ _ _ _ _ _ _ _ _) as [s' |] eqn:Es; [| discriminate].
  injection H as <- <- <-. exists i. cbn [th_count th_hist th_shist th_lhist th_state th_local th_passed].
  split; [reflexivity |]. split; [reflexivity |]. split; [reflexivity |]. split; [reflexivity |].
  split; [reflexivity |]. split; [exact Es |]. split; [reflexivity |]. split; [reflexivity |]. split; reflexivity.
Qed.

(* The run of a thread up to its step reads the memories of its history. *)
Lemma tstep_run : forall init o m sm th th' m' sm',
    thread_ok init o th -> tstep init o m sm th = Some (th', m', sm') ->
    run' F prog (env_of init o (th_hist th') (th_shist th') (th_lhist th')) (th_count th) = Some (th_state th).
Proof.
  intros init o m sm th th' m' sm' Hok H.
  destruct (tstep_spec init o m sm th th' m' sm' H) as [i [_ [_ [Hh [Hs [Hl _]]]]]].
  rewrite <- Hok. unfold thread_run, run'. rewrite Hh, Hs, Hl. symmetry. apply run_ext.
  exact (env_before init o th m sm).
Qed.

Lemma tstep_ok : forall init o m sm th th' m' sm',
    thread_ok init o th -> tstep init o m sm th = Some (th', m', sm') -> thread_ok init o th'.
Proof.
  intros init o m sm th th' m' sm' Hok H.
  pose proof (tstep_run init o m sm th th' m' sm' Hok H) as Hrun.
  destruct (tstep_spec init o m sm th th' m' sm' H) as [i [_ [Hc [_ [_ [_ [Hs _]]]]]]].
  unfold thread_ok, thread_run, run'. rewrite Hc. cbn [run].
  unfold run' in Hrun. rewrite Hrun. exact Hs.
Qed.

(* A step reads global memory only through the value of a load, shared
   memory only through the value of a shared load, and the oracle at its
   own step. *)
Lemma step_env_eq : forall mem1 mem2 arb1 arb2 shm1 shm2 init t s i,
    fetch prog (blk s) (idx s) = Some i ->
    (forall g gen bits d a off, i = ILoad g gen bits d a off -> holds g (regs s) = true ->
       trunc bits (mem1 t (trunc 64 (regs s a + off))) = trunc bits (mem2 t (trunc 64 (regs s a + off)))) ->
    (forall g bits d a off, i = ILdS g bits d a off -> holds g (regs s) = true ->
       trunc bits (shm1 t (trunc 32 (regs s a + off))) = trunc bits (shm2 t (trunc 32 (regs s a + off)))) ->
    arb1 t = arb2 t ->
    step prog (wdn F) mem1 arb1 shm1 init (lf_cells F) (lf_locals F) (lf_local_window F) t s =
    step prog (wdn F) mem2 arb2 shm2 init (lf_cells F) (lf_locals F) (lf_local_window F) t s.
Proof.
  intros mem1 mem2 arb1 arb2 shm1 shm2 init t s i Ef Hm Hs Ha. unfold step. rewrite Ef.
  destruct (holds (guard_of i) (regs s)) eqn:Hg; cbn [negb]; [| reflexivity].
  destruct i; cbn beta iota; unfold gstore, lstore, lstore_cell, lload, lframe, lframe_cell, result;
    rewrite ?(write_all_at (wdn F) arb1 arb2 t Ha), ?Ha; try reflexivity.
  - cbn [dest result]. rewrite (Hm g gen bits d a off eq_refl Hg). reflexivity.
  - cbn [dest result]. rewrite (Hs g bits d a off eq_refl Hg). reflexivity.
Qed.

End Launch.

Lemma nth_set_nth : forall (A : Type) (l : list A) n k x,
    nth_error (set_nth l n x) k =
    if Nat.eqb k n then match nth_error l n with Some _ => Some x | None => None end else nth_error l k.
Proof.
  intros A l. induction l as [| y l IH]; intros n k x; cbn [set_nth].
  - destruct n, k; cbn [nth_error Nat.eqb]; try reflexivity. destruct (Nat.eqb k n); reflexivity.
  - destruct n as [| n], k as [| k]; cbn [nth_error Nat.eqb]; try reflexivity. exact (IH n k x).
Qed.

Lemma length_set_nth : forall (A : Type) (l : list A) n x, List.length (set_nth l n x) = List.length l.
Proof.
  intros A l. induction l as [| y l IH]; intros n x; [reflexivity |].
  destruct n as [| n]; cbn [set_nth List.length]; [reflexivity | rewrite IH; reflexivity].
Qed.

Lemma nth_error_seq : forall s l n x, nth_error (seq s l) n = Some x -> x = (s + n)%nat.
Proof.
  intros s l n x H. destruct (Nat.lt_ge_cases n l) as [Hl | Hl].
  - rewrite (nth_error_nth' _ 0%nat) in H by (rewrite seq_length; exact Hl).
    injection H as <-. apply seq_nth. exact Hl.
  - rewrite (proj2 (nth_error_None _ _)) in H by (rewrite seq_length; exact Hl). discriminate.
Qed.

(* ------------------------------------------------------------------ *)
(* The safety of a launch.                                             *)

Section Safe.

Variable F : lfacts.
Variable prog : program.
Variable bar : option (nat * nat).
Hypothesis Hsafe : kernel_safe F prog.
Variable inits : nat -> reg -> Z.
Variable os : nat -> oracle.
(* The block of each thread. *)
Variable cta : nat -> nat.

(* An invariant of global memory that the runtime establishes before the
   launch, from which each thread's environment has the launch facts, and
   that each write the gate allows keeps. *)
Variable P : memory -> Prop.
Hypothesis Hlink : forall n hist sh lh, (forall t, P (hist t)) -> launch_ok F (env_of F (inits n) (os n) hist sh lh).
Hypothesis Hkeep : forall n hist sh lh t k x width v wr m,
    (forall t, P (hist t)) -> region F (env_of F (inits n) (os n) hist sh lh) t k x width v -> k <> KLoad -> P m ->
    P (write_bytes m x width (match v with Some z => z | None => wr end)).

(* A step of a thread the invariant holds of keeps it, and its access lies
   in its region. *)
Lemma tstep_safe : forall n m sm th th' m' sm',
    thread_ok F prog (inits n) (os n) th -> (forall t, P (th_hist th t)) -> P m ->
    tstep F prog bar (inits n) (os n) m sm th = Some (th', m', sm') ->
    (forall t, P (th_hist th' t)) /\
    exists i, fetch prog (blk (th_state th)) (idx (th_state th)) = Some i /\ wf_instr F i = true /\
      (forall k x w v, access_at (win_of F (inits n)) (regs (th_state th)) i = Some (k, x, w, v) ->
         region F (env_of F (inits n) (os n) (th_hist th') (th_shist th') (th_lhist th')) (th_count th) k x w v) /\
      (forall p, private_access (win_of F (inits n)) (regs (th_state th)) i = Some p ->
         private_in F (env_of F (inits n) (os n) (th_hist th') (th_shist th') (th_lhist th')) p) /\
      m' = mem_write F (inits n)
             (o_wr (os n) (th_count th) (shared_bytes F (inits n) sm) (local_bytes F (inits n) (th_local th)))
             (regs (th_state th)) i m /\ P m'.
Proof.
  intros n m sm th th' m' sm' Hok Hh Hm H.
  pose proof (tstep_run F prog bar _ _ _ _ _ _ _ _ Hok H) as Hrun.
  destruct (tstep_spec F prog bar _ _ _ _ _ _ _ _ H) as [i [Ef [_ [Hhist [_ [_ [_ [_ [_ [Hm' _]]]]]]]]]].
  assert (Hh' : forall t, P (th_hist th' t)).
  { intros t. rewrite Hhist. destruct (Nat.eqb t (th_count th)); [exact Hm | apply Hh]. }
  pose proof (Hlink n (th_hist th') (th_shist th') (th_lhist th') Hh') as Hk.
  destruct (Hsafe _ Hk (th_count th) (th_state th) i Hrun Ef) as [Hreg [_ [Hpriv Hwf]]].
  rewrite win_of_env in Hreg, Hpriv. cbn [env_of e_init] in Hreg, Hpriv.
  split; [exact Hh' |]. exists i. split; [exact Ef |]. split; [exact Hwf |]. split; [exact Hreg |].
  split; [exact Hpriv |]. split; [exact Hm' |]. rewrite Hm'. unfold mem_write.
  destruct (access_at (win_of F (inits n)) (regs (th_state th)) i) as [[[[k x] w] v] |] eqn:Ea; [| exact Hm].
  destruct k; [exact Hm | |];
    exact (Hkeep n (th_hist th') (th_shist th') (th_lhist th') (th_count th) _ x w v _ m Hh'
             (Hreg _ _ _ _ eq_refl) ltac:(discriminate) Hm).
Qed.

Definition grid_inv (G : grid) : Prop :=
  P (g_mem G) /\
  forall n th, nth_error (g_threads G) n = Some th ->
    thread_ok F prog (inits n) (os n) th /\ forall t, P (th_hist th t).

Lemma grid0_inv : forall count m0 sh0 lm0, P m0 -> grid_inv (grid0 inits count m0 sh0 lm0).
Proof.
  intros count m0 sh0 lm0 Hm. split; [exact Hm |]. intros n th H. cbn [grid0 g_threads] in H.
  rewrite nth_error_map in H. destruct (nth_error (seq 0 count) n) as [k |] eqn:Es; [| discriminate].
  apply nth_error_seq in Es. cbn in Es. subst k.
  injection H as <-. split; [apply start_ok | intros t; exact Hm].
Qed.

(* The parts of a step of the grid that moves a thread. *)
Lemma gstep_cases : forall G n,
    gstep F prog bar inits os cta G n = G \/
    exists th th' m' sm', nth_error (g_threads G) n = Some th /\
      tstep F prog bar (inits n) (os n) (g_mem G) (g_shared G (cta n)) th = Some (th', m', sm') /\
      gstep F prog bar inits os cta G n =
        {| g_mem := m'; g_shared := fun b => if Nat.eqb b (cta n) then sm' else g_shared G b;
           g_threads := set_nth (g_threads G) n th' |}.
Proof.
  intros G n. unfold gstep.
  destruct (nth_error (g_threads G) n) as [th |] eqn:En; [| left; reflexivity].
  destruct (at_bar bar (th_state th) && negb (block_arrived bar cta G n)); [left; reflexivity |].
  destruct (tstep F prog bar (inits n) (os n) (g_mem G) (g_shared G (cta n)) th) as [[[th' m'] sm'] |] eqn:Et;
    [| left; reflexivity].
  right. exists th, th', m', sm'. split; [reflexivity |]. split; [exact Et | reflexivity].
Qed.

Lemma gstep_inv : forall G n, grid_inv G -> grid_inv (gstep F prog bar inits os cta G n).
Proof.
  intros G n [Hm Hth].
  destruct (gstep_cases G n) as [-> | [th [th' [m' [sm' [En [Et ->]]]]]]]; [split; assumption |].
  destruct (Hth n th En) as [Hok Hh].
  destruct (tstep_safe n (g_mem G) _ th th' m' sm' Hok Hh Hm Et) as [Hh' [i [_ [_ [_ [_ [_ Hm']]]]]]].
  split; [exact Hm' |]. intros k thk Hk. cbn [g_threads] in Hk. rewrite nth_set_nth in Hk.
  destruct (Nat.eqb_spec k n) as [-> | Hne].
  - rewrite En in Hk. injection Hk as <-. split; [exact (tstep_ok F prog bar _ _ _ _ _ _ _ _ Hok Et) | exact Hh'].
  - exact (Hth k thk Hk).
Qed.

Lemma launch_inv : forall count m0 sh0 lm0 sched, P m0 ->
    grid_inv (launch F prog bar inits os cta count m0 sh0 lm0 sched).
Proof.
  intros count m0 sh0 lm0 sched Hm. unfold launch. generalize (grid0_inv count m0 sh0 lm0 Hm).
  generalize (grid0 inits count m0 sh0 lm0). induction sched as [| n sched IH]; intros G HG; [exact HG |].
  cbn [fold_left]. apply IH. apply gstep_inv. exact HG.
Qed.

(* A launch changes no byte of global memory outside those its writes may
   reach. *)
Section Frame.

Variable W : Z -> Prop.
Hypothesis Hwrites : forall n hist sh lh t k x w v,
    (forall t, P (hist t)) -> region F (env_of F (inits n) (os n) hist sh lh) t k x w v -> k <> KLoad ->
    forall y, x <= y < x + w -> W y.

Theorem launch_frame : forall count m0 sh0 lm0 sched, P m0 ->
    forall y, ~ W y -> g_mem (launch F prog bar inits os cta count m0 sh0 lm0 sched) y = m0 y.
Proof.
  intros count m0 sh0 lm0 sched Hm y Hy. unfold launch.
  assert (Hgen : forall G, grid_inv G -> g_mem G y = m0 y ->
            g_mem (fold_left (gstep F prog bar inits os cta) sched G) y = m0 y).
  { induction sched as [| n sched IH]; intros G HG Hgy; [exact Hgy |].
    cbn [fold_left]. apply IH; [apply gstep_inv; exact HG |].
    destruct HG as [HmG Hth].
    destruct (gstep_cases G n) as [-> | [th [th' [m' [sm' [En [Et ->]]]]]]]; [exact Hgy |].
    destruct (Hth n th En) as [Hok Hh].
    destruct (tstep_safe n (g_mem G) _ th th' m' sm' Hok Hh HmG Et) as [Hh' [i [_ [_ [Hreg [_ [Hm' _]]]]]]].
    cbn [g_mem]. rewrite Hm'. unfold mem_write.
    destruct (access_at (win_of F (inits n)) (regs (th_state th)) i) as [[[[k x] w] v] |] eqn:Ea; [| exact Hgy].
    destruct k; [exact Hgy | |];
      (unfold write_bytes; destruct ((x <=? y) && (y <? x + w)) eqn:Ey; [| exact Hgy]);
      exfalso; apply Hy; apply andb_true_iff in Ey; destruct Ey as [E1 E2];
      apply Z.leb_le in E1; apply Z.ltb_lt in E2;
      exact (Hwrites n (th_hist th') (th_shist th') (th_lhist th') _ _ x w v Hh' (Hreg _ _ _ _ eq_refl)
               ltac:(discriminate) y ltac:(lia)). }
  apply Hgen; [exact (grid0_inv count m0 sh0 lm0 Hm) | reflexivity].
Qed.

End Frame.

End Safe.

(* ------------------------------------------------------------------ *)
(* Noninterference.                                                    *)

(* Two runs of a launch with the same schedule and oracles, from global
   memories that agree on the bytes its loads may read, from local memories
   that agree on the bytes of each thread's local variables, and from
   shared memories that may differ everywhere, agree on those bytes after
   it. A load of local memory reads bytes of the thread's local variables,
   and the two runs write the same bytes there, so a thread's two local
   memories keep agreeing on those bytes. Before any thread of a block
   passes the prologue's barrier, the threads of the block read no shared
   memory and store zero over every byte of the block's shared variables,
   and once a thread has passed it, the block's two shared memories agree
   on those bytes; so each thread reads the same values in both runs, takes
   the same steps, and writes the same bytes. *)
Section Noninterference.

Variable F : lfacts.
Variable Mf : binding.
Variable prog : program.
Hypothesis Hsafe : kernel_safe F prog.
Variable inits : nat -> reg -> Z.
Variable os : nat -> oracle.
Variable cta : nat -> nat.
Variable count : nat.

Variable P : memory -> Prop.
Hypothesis Hlink : forall n hist sh lh, (forall t, P (hist t)) -> launch_ok F (env_of F (inits n) (os n) hist sh lh).
Hypothesis Hkeep : forall n hist sh lh t k x width v wr m,
    (forall t, P (hist t)) -> region F (env_of F (inits n) (os n) hist sh lh) t k x width v -> k <> KLoad -> P m ->
    P (write_bytes m x width (match v with Some z => z | None => wr end)).

Variable R : Z -> Prop.
Hypothesis Hreads : forall n hist sh lh t x w v,
    (forall t, P (hist t)) -> region F (env_of F (inits n) (os n) hist sh lh) t KLoad x w v ->
    forall y, x <= y < x + w -> R y.
Hypothesis Hlf : lf_ok F = true.
(* The windows of shared and local memory lie apart. *)
Hypothesis Hwindows : forall n, inits n (lf_shared_window F) + window_size <= inits n (lf_local_window F) \/
                                inits n (lf_local_window F) + window_size <= inits n (lf_shared_window F).

(* The barrier at which the threads of a block wait, if any. *)
Variable bar : option (nat * nat).
Local Notation replay n th := (thread_run F prog (inits n) (os n) (th_hist th) (th_shist th) (th_lhist th)).

Definition agree (m1 m2 : memory) : Prop := forall y, R y -> m1 y = m2 y.

(* Two shared memories of a block agree on the bytes of its variables. *)
Definition sm_agree (init : reg -> Z) (sm1 sm2 : memory) : Prop := forall x, declared F init x -> sm1 x = sm2 x.

(* Two local memories of a thread agree on the bytes of its variables. *)
Definition lm_agree (init : reg -> Z) (lm1 lm2 : memory) : Prop := forall x, declared_local F init x -> lm1 x = lm2 x.

(* What two runs keep the same of a thread: its state, the count of its
   steps, and whether it has executed the barrier. Its two local memories
   agree on the bytes of its variables (`lm_agree`). *)
Definition view (th : thread) : state * nat * bool := (th_state th, th_count th, th_passed th).

(* A thread of block b has passed the barrier. *)
Definition passed_block (G : grid) (b : nat) : Prop :=
  exists n th, nth_error (g_threads G) n = Some th /\ cta n = b /\ th_passed th = true.


(* ---- the replay of a thread's own steps after one more step ---- *)

Lemma replay_before : forall n m sm th th' m' sm', thread_ok F prog (inits n) (os n) th ->
    tstep F prog bar (inits n) (os n) m sm th = Some (th', m', sm') ->
    forall t, (t <= th_count th)%nat -> replay n th' t = replay n th t.
Proof.
  intros n m sm th th' m' sm' Hok H t Ht.
  destruct (tstep_spec F prog bar _ _ _ _ _ _ _ _ H) as [i [_ [_ [Hh [Hs [Hl _]]]]]].
  unfold thread_run, run'. rewrite Hh, Hs, Hl. symmetry. apply run_ext.
  intros t' Ht'. exact (env_before F (inits n) (os n) th m sm t' ltac:(lia)).
Qed.

Lemma write_zero : forall sm x w y, write_bytes sm x w 0 y = 0 \/ write_bytes sm x w 0 y = sm y.
Proof.
  intros sm x w y. unfold write_bytes. destruct ((x <=? y) && (y <? x + w)) eqn:E; [left | right; reflexivity].
  apply andb_true_iff in E. rewrite Z.leb_le, Z.ltb_lt in E.
  rewrite Z.div_0_l by (apply Z.pow_nonzero; lia). reflexivity.
Qed.

Lemma write_outside' : forall sm x w v y, ~ (x <= y < x + w) -> write_bytes sm x w v y = sm y.
Proof.
  intros sm x w v y H. unfold write_bytes. destruct ((x <=? y) && (y <? x + w)) eqn:E; [| reflexivity].
  exfalso. apply H. apply andb_true_iff in E. rewrite Z.leb_le, Z.ltb_lt in E. exact E.
Qed.

Lemma write_agree : forall m1 m2 x w v, agree m1 m2 -> agree (write_bytes m1 x w v) (write_bytes m2 x w v).
Proof.
  intros m1 m2 x w v H y Hy. unfold write_bytes. destruct ((x <=? y) && (y <? x + w)); [reflexivity | exact (H y Hy)].
Qed.

Lemma sm_write_agree : forall init sm1 sm2 x w v, sm_agree init sm1 sm2 ->
    sm_agree init (write_bytes sm1 x w v) (write_bytes sm2 x w v).
Proof.
  intros init sm1 sm2 x w v H y Hy. unfold write_bytes. destruct ((x <=? y) && (y <? x + w)); [reflexivity | exact (H y Hy)].
Qed.

Lemma lm_write_agree : forall init lm1 lm2 x w v, lm_agree init lm1 lm2 ->
    lm_agree init (write_bytes lm1 x w v) (write_bytes lm2 x w v).
Proof.
  intros init lm1 lm2 x w v H y Hy. unfold write_bytes. destruct ((x <=? y) && (y <? x + w)); [reflexivity | exact (H y Hy)].
Qed.

Lemma write_zero_in : forall sm x w y, x <= y < x + w -> write_bytes sm x w 0 y = 0.
Proof.
  intros sm x w y Hy. unfold write_bytes.
  rewrite (proj2 (Z.leb_le _ _) (proj1 Hy)), (proj2 (Z.ltb_lt _ _) (proj2 Hy)). cbn [andb].
  rewrite Z.div_0_l by (apply Z.pow_nonzero; lia). reflexivity.
Qed.

Lemma agree_sym : forall m1 m2, agree m1 m2 -> agree m2 m1.
Proof. intros m1 m2 H y Hy. symmetry. exact (H y Hy). Qed.

Lemma sm_agree_sym : forall init sm1 sm2, sm_agree init sm1 sm2 -> sm_agree init sm2 sm1.
Proof. intros init sm1 sm2 H x Hx. symmetry. exact (H x Hx). Qed.

Lemma lm_agree_sym : forall init lm1 lm2, lm_agree init lm1 lm2 -> lm_agree init lm2 lm1.
Proof. intros init lm1 lm2 H x Hx. symmetry. exact (H x Hx). Qed.

(* A step writes shared memory the same way in two memories that agree. *)
Lemma shared_write_agree : forall init wr rs i sm1 sm2, sm_agree init sm1 sm2 ->
    sm_agree init (shared_write F init wr rs i sm1) (shared_write F init wr rs i sm2).
Proof.
  intros init wr rs i sm1 sm2 H. unfold shared_write. destruct (negb (holds (guard_of i) rs)); [exact H |].
  destruct i; try exact H.
  - destruct k; try exact H; destruct gen; try exact H; cbv zeta;
      (destruct (within _ _ _ _); [apply sm_write_agree; exact H | exact H]).
  - destruct k; try exact H; apply sm_write_agree; exact H.
Qed.

(* A step writes local memory the same way in two memories that agree. *)
Lemma local_write_agree : forall init wr rs i lm1 lm2, lm_agree init lm1 lm2 ->
    lm_agree init (local_write F init wr rs i lm1) (local_write F init wr rs i lm2).
Proof.
  intros init wr rs i lm1 lm2 H. unfold local_write. destruct (negb (holds (guard_of i) rs)); [exact H |].
  destruct i; try exact H.
  - destruct k; try exact H; destruct gen; try exact H; cbv zeta;
      (destruct (within _ _ _ _); [apply lm_write_agree; exact H | exact H]).
  - apply lm_write_agree; exact H.
Qed.

(* An address outside the windows is not in a window with any width of at
   least one byte. *)
Lemma win_of_outside : forall init x w, 1 <= w -> in_windows F init x = false -> win_of F init x w = false.
Proof.
  intros init x w Hw H. unfold in_windows, in_space, win_of in *. cbn [existsb] in *.
  rewrite !orb_false_iff, !andb_false_iff, !Z.leb_gt, !Z.ltb_ge in H.
  rewrite !orb_false_iff, !andb_false_iff, !Z.leb_gt. lia.
Qed.

(* The bytes of a shared variable are declared. *)
Lemma in_var_declared : forall init x w, GateMeaning.in_var F {| e_mem := fun _ _ => 0; e_arb := fun _ _ => 0; e_shm := fun _ _ => 0; e_init := init |}
      (lf_shared F) x w -> forall y, x <= y < x + w -> declared F init y.
Proof.
  intros init x w [u [bytes [Hin [_ [_ [Hlo Hhi]]]]]] y Hy. exists u, bytes. split; [exact Hin |].
  unfold bval in Hhi. cbn [e_init] in Hlo, Hhi. unfold fval. lia.
Qed.

(* The bytes of a local variable are declared local bytes (`declared_local`). *)
Lemma in_var_declared_local : forall init x w,
    GateMeaning.in_var F {| e_mem := fun _ _ => 0; e_arb := fun _ _ => 0; e_shm := fun _ _ => 0; e_init := init |}
      (local_vars F) x w -> forall y, x <= y < x + w -> declared_local F init y.
Proof.
  intros init x w [u [bytes [Hin [_ [_ [Hlo Hhi]]]]]] y Hy. exists u, bytes. split; [exact Hin |].
  unfold bval in Hhi. cbn [e_init] in Hlo, Hhi. unfold fval. lia.
Qed.

(* A global access lies apart from the window of shared memory. *)
Lemma global_not_shared : forall E t x w v, launch_ok F E -> region F E t KLoad x w v -> 1 <= w ->
    in_space (e_init E) (lf_shared_window F) x = false.
Proof.
  intros E t x w v Hk Hr Hw. pose proof (env_of_launch F prog E Hlf Hk) as Hok.
  inversion Hr as [k' Pr sp e w' v' Hrole Hallow He Hle _ _ _ Ex]; subst.
  pose proof (ok_init F E Hok Pr) as HP. pose proof (ok_init F E Hok (lf_shared_window F)) as HS.
  pose proof (ok_widths F E Hok Pr) as WP. pose proof (ok_widths F E Hok (lf_shared_window F)) as WS.
  assert (H64 : forall r, 2 ^ wdz F r <= 2 ^ 64).
  { intros r. apply Z.pow_le_mono_r; [lia |]. pose proof (ok_widths F E Hok r). lia. }
  pose proof (H64 Pr). pose proof (H64 (lf_shared_window F)).
  destruct (lk_apart F E Hk Pr sp (lf_shared_window F) Hrole (or_introl eq_refl)) as [Hap | [Hap Hbound]];
    unfold in_space; apply andb_false_iff; unfold trunc; change (Z.of_nat 64) with 64; unfold bval in *.
  - (* the buffer ends before the window *)
    rewrite Z.mod_small by lia. left. apply Z.leb_gt. lia.
  - (* the window ends before the buffer *)
    rewrite Z.mod_small by lia. right. apply Z.ltb_ge. lia.
Qed.

(* A global access lies apart from the window of local memory. *)
Lemma global_not_local : forall E t x w v, launch_ok F E -> region F E t KLoad x w v -> 1 <= w ->
    in_space (e_init E) (lf_local_window F) x = false.
Proof.
  intros E t x w v Hk Hr Hw. pose proof (env_of_launch F prog E Hlf Hk) as Hok.
  inversion Hr as [k' Pr sp e w' v' Hrole Hallow He Hle _ _ _ Ex]; subst.
  pose proof (ok_init F E Hok Pr) as HP. pose proof (ok_init F E Hok (lf_local_window F)) as HL.
  pose proof (ok_widths F E Hok Pr) as WP. pose proof (ok_widths F E Hok (lf_local_window F)) as WL.
  assert (H64 : forall r, 2 ^ wdz F r <= 2 ^ 64).
  { intros r. apply Z.pow_le_mono_r; [lia |]. pose proof (ok_widths F E Hok r). lia. }
  pose proof (H64 Pr). pose proof (H64 (lf_local_window F)).
  destruct (lk_apart F E Hk Pr sp (lf_local_window F) Hrole (or_intror eq_refl)) as [Hap | [Hap Hbound]];
    unfold in_space; apply andb_false_iff; unfold trunc; change (Z.of_nat 64) with 64; unfold bval in *.
  - (* the buffer ends before the window *)
    rewrite Z.mod_small by lia. left. apply Z.leb_gt. lia.
  - (* the window ends before the buffer *)
    rewrite Z.mod_small by lia. right. apply Z.ltb_ge. lia.
Qed.

(* The bytes of shared and local memory that a step reads lie in the
   block's shared variables and the thread's local variables. *)
Lemma step_shared_reads : forall n th m sm th' m' sm' hist,
    (forall t, P (th_hist th t)) -> P m ->
    thread_ok F prog (inits n) (os n) th ->
    tstep F prog bar (inits n) (os n) m sm th = Some (th', m', sm') ->
    hist = (fun t' => if Nat.eqb t' (th_count th) then m else th_hist th t') ->
    exists i, fetch prog (blk (th_state th)) (idx (th_state th)) = Some i /\
      (* a shared load reads declared bytes *)
      (forall g bits d a off, i = ILdS g bits d a off -> holds g (regs (th_state th)) = true ->
         (bits mod 8 = 0)%nat /\
         forall y, trunc 32 (regs (th_state th) a + off) <= y < trunc 32 (regs (th_state th) a + off) + Z.of_nat bits / 8 ->
           declared F (inits n) y) /\
      (* a load at a generic address in the shared window reads declared bytes, one in the local
         window reads bytes of the local variables, and one outside both windows reads bytes of R *)
      (forall g gen bits d a off, i = ILoad g gen bits d a off -> holds g (regs (th_state th)) = true ->
         (bits mod 8 = 0)%nat /\
         let x := trunc 64 (regs (th_state th) a + off) in
         (in_space (inits n) (lf_shared_window F) x = true ->
            forall y, x - inits n (lf_shared_window F) <= y < x - inits n (lf_shared_window F) + Z.of_nat bits / 8 ->
              declared F (inits n) y) /\
         (in_space (inits n) (lf_local_window F) x = true ->
            forall y, x - inits n (lf_local_window F) <= y < x - inits n (lf_local_window F) + Z.of_nat bits / 8 ->
              declared_local F (inits n) y) /\
         (in_space (inits n) (lf_shared_window F) x = false -> in_space (inits n) (lf_local_window F) x = false ->
            forall y, x <= y < x + Z.of_nat bits / 8 -> R y)).
Proof.
  intros n th m sm th' m' sm' hist Hh Hm Hok H ->.
  destruct (tstep_safe F prog bar Hsafe inits os P Hlink Hkeep n m sm th th' m' sm' Hok Hh Hm H)
    as [Hh' [i [Ef [Hwf [Hreg [Hpriv _]]]]]].
  exists i. split; [exact Ef |].
  set (E' := env_of F (inits n) (os n) (th_hist th') (th_shist th') (th_lhist th')) in *.
  pose proof (Hlink n (th_hist th') (th_shist th') (th_lhist th') Hh') as Hk.
  split.
  - intros g bits d a off -> Hg. unfold wf_instr in Hwf. rewrite !andb_true_iff in Hwf.
    destruct Hwf as [[_ [[[_ _] _] Hmod]] _]. apply Z.eqb_eq in Hmod.
    split; [apply Nat2Z.inj; rewrite Nat2Z.inj_mod; exact Hmod |].
    intros y Hy. specialize (Hpriv (PShared (trunc 32 (regs (th_state th) a + off)) (Z.of_nat bits / 8))).
    unfold private_access in Hpriv. cbn [guard_of] in Hpriv. rewrite Hg in Hpriv. cbn [negb] in Hpriv.
    specialize (Hpriv eq_refl). cbn [private_in] in Hpriv.
    destruct Hpriv as [u [bytes [Hin [_ [_ [Hlo Hhi]]]]]]. exists u, bytes. split; [exact Hin |].
    unfold bval in Hhi. subst E'. cbn [env_of e_init] in Hlo, Hhi. unfold fval. lia.
  - intros g gen bits d a off -> Hg. unfold wf_instr in Hwf. rewrite !andb_true_iff in Hwf.
    destruct Hwf as [[_ [[[_ _] H8] Hmod]] _]. apply Z.leb_le in H8. apply Z.eqb_eq in Hmod.
    assert (Hmod' : (bits mod 8 = 0)%nat) by (apply Nat2Z.inj; rewrite Nat2Z.inj_mod; exact Hmod).
    split; [exact Hmod' |]. cbv zeta.
    set (x := trunc 64 (regs (th_state th) a + off)).
    set (w := Z.of_nat bits / 8).
    assert (Hw1 : 1 <= w) by (unfold w; apply Z.div_le_lower_bound; lia).
    split; [| split].
    + (* in the shared window *)
      intros Hsp y Hy.
      destruct (win_of F (inits n) x w) eqn:Hwin.
      * (* the access is private: inside a shared variable, since the windows lie apart *)
        specialize (Hpriv (PGeneric x w)).
        assert (Hpa : private_access (win_of F (inits n)) (regs (th_state th)) (ILoad g gen bits d a off) = Some (PGeneric x w)).
        { unfold private_access. cbn [guard_of]. rewrite Hg. cbn [negb]. fold x w.
          destruct gen; [rewrite Hwin; reflexivity |].
          (* an address that is not generic: the load reads global memory, which lies apart
             from the windows *)
          exfalso.
          assert (Hacc : access_at (win_of F (inits n)) (regs (th_state th)) (ILoad g false bits d a off) =
                         Some (KLoad, x, w, None)).
          { unfold access_at. cbn [guard_of]. rewrite Hg. cbn [negb]. reflexivity. }
          pose proof (global_not_shared E' (th_count th) x w None Hk (Hreg _ _ _ _ Hacc) Hw1) as Hns.
          subst E'. cbn [env_of e_init] in Hns. rewrite Hsp in Hns. discriminate. }
        specialize (Hpriv Hpa). cbn [private_in] in Hpriv. destruct Hpriv as [Hpriv | Hpriv].
        -- subst E'. cbn [env_of e_init] in Hpriv. exact (in_var_declared (inits n) _ _ Hpriv y Hy).
        -- (* a local variable at a shared address: the windows lie apart *)
           exfalso. subst E'. cbn [env_of e_init] in Hpriv.
           destruct Hpriv as [u [bytes [Hin [Hu0 [Hub [Hlo Hhi]]]]]].
           unfold in_space in Hsp. rewrite andb_true_iff, Z.leb_le, Z.ltb_lt in Hsp.
           destruct (Hwindows n) as [Hap | Hap]; lia.
      * (* the whole access is not in a window: it is a global access, which lies apart from the window *)
        exfalso.
        assert (Hacc : access_at (win_of F (inits n)) (regs (th_state th)) (ILoad g gen bits d a off) =
                       Some (KLoad, x, w, None)).
        { unfold access_at. cbn [guard_of]. rewrite Hg. cbn [negb]. fold x w. rewrite Hwin, andb_false_r. reflexivity. }
        pose proof (global_not_shared E' (th_count th) x w None Hk (Hreg _ _ _ _ Hacc) Hw1) as Hns.
        subst E'. cbn [env_of e_init] in Hns. rewrite Hsp in Hns. discriminate.
    + (* in the local window *)
      intros Hlp y Hy.
      destruct (win_of F (inits n) x w) eqn:Hwin.
      * (* the access is private: inside a local variable, since the windows lie apart *)
        specialize (Hpriv (PGeneric x w)).
        assert (Hpa : private_access (win_of F (inits n)) (regs (th_state th)) (ILoad g gen bits d a off) = Some (PGeneric x w)).
        { unfold private_access. cbn [guard_of]. rewrite Hg. cbn [negb]. fold x w.
          destruct gen; [rewrite Hwin; reflexivity |].
          (* an address that is not generic: the load reads global memory, which lies apart
             from the windows *)
          exfalso.
          assert (Hacc : access_at (win_of F (inits n)) (regs (th_state th)) (ILoad g false bits d a off) =
                         Some (KLoad, x, w, None)).
          { unfold access_at. cbn [guard_of]. rewrite Hg. cbn [negb]. reflexivity. }
          pose proof (global_not_local E' (th_count th) x w None Hk (Hreg _ _ _ _ Hacc) Hw1) as Hnl.
          subst E'. cbn [env_of e_init] in Hnl. rewrite Hlp in Hnl. discriminate. }
        specialize (Hpriv Hpa). cbn [private_in] in Hpriv. destruct Hpriv as [Hpriv | Hpriv].
        -- (* a shared variable at a local address: the windows lie apart *)
           exfalso. subst E'. cbn [env_of e_init] in Hpriv.
           destruct Hpriv as [u [bytes [Hin [Hu0 [Hub [Hlo Hhi]]]]]].
           unfold in_space in Hlp. rewrite andb_true_iff, Z.leb_le, Z.ltb_lt in Hlp.
           destruct (Hwindows n) as [Hap | Hap]; lia.
        -- subst E'. cbn [env_of e_init] in Hpriv. exact (in_var_declared_local (inits n) _ _ Hpriv y Hy).
      * (* the whole access is not in a window: it is a global access, which lies apart from the window *)
        exfalso.
        assert (Hacc : access_at (win_of F (inits n)) (regs (th_state th)) (ILoad g gen bits d a off) =
                       Some (KLoad, x, w, None)).
        { unfold access_at. cbn [guard_of]. rewrite Hg. cbn [negb]. fold x w. rewrite Hwin, andb_false_r. reflexivity. }
        pose proof (global_not_local E' (th_count th) x w None Hk (Hreg _ _ _ _ Hacc) Hw1) as Hnl.
        subst E'. cbn [env_of e_init] in Hnl. rewrite Hlp in Hnl. discriminate.
    + intros Hsp Hlp y Hy.
      assert (Hwin : win_of F (inits n) x w = false).
      { apply win_of_outside; [exact Hw1 |]. unfold in_windows. cbn [existsb]. rewrite Hsp, Hlp. reflexivity. }
      assert (Hacc : access_at (win_of F (inits n)) (regs (th_state th)) (ILoad g gen bits d a off) =
                     Some (KLoad, x, w, None)).
      { unfold access_at. cbn [guard_of]. rewrite Hg. cbn [negb]. fold x w. rewrite Hwin, andb_false_r. reflexivity. }
      exact (Hreads n (th_hist th') (th_shist th') (th_lhist th') (th_count th) x w None Hh' (Hreg _ _ _ _ Hacc) y Hy).
Qed.


(* ---- the two runs take the same step ---- *)

Lemma view_parts : forall th1 th2, view th1 = view th2 ->
    th_state th1 = th_state th2 /\ th_count th1 = th_count th2 /\ th_passed th1 = th_passed th2.
Proof. intros th1 th2 H. unfold view in H. injection H as H1 H2 H3. tauto. Qed.


(* ---- the invariant of one run ---- *)

Lemma set_nth_at : forall (A : Type) (l : list A) k x th, nth_error l k = Some th -> nth_error (set_nth l k x) k = Some x.
Proof. intros A l k x th H. rewrite nth_set_nth, Nat.eqb_refl, H. reflexivity. Qed.

Lemma set_nth_other : forall (A : Type) (l : list A) k x n, n <> k -> nth_error (set_nth l k x) n = nth_error l n.
Proof. intros A l k x n H. rewrite nth_set_nth. apply Nat.eqb_neq in H. rewrite H. reflexivity. Qed.

Lemma forallb_same : forall (A : Type) (f g : A -> bool) (l : list A),
    (forall x, In x l -> f x = g x) -> forallb f l = forallb g l.
Proof.
  intros A f g l H. induction l as [| x l IH]; [reflexivity |]. cbn [forallb].
  rewrite (H x (or_introl eq_refl)), IH; [reflexivity |]. intros y Hy. exact (H y (or_intror Hy)).
Qed.

Lemma arrived_view : forall th1 th2, view th1 = view th2 -> arrived bar th1 = arrived bar th2.
Proof.
  intros th1 th2 H. destruct (view_parts th1 th2 H) as (Hst & _ & Hpd). unfold arrived.
  rewrite Hst, Hpd. reflexivity.
Qed.

Lemma block_arrived_view : forall G1 G2 k, List.length (g_threads G1) = List.length (g_threads G2) ->
    (forall n th1 th2, nth_error (g_threads G1) n = Some th1 -> nth_error (g_threads G2) n = Some th2 -> view th1 = view th2) ->
    block_arrived bar cta G1 k = block_arrived bar cta G2 k.
Proof.
  intros G1 G2 k Hlen Hview. unfold block_arrived. rewrite Hlen. apply forallb_same. intros j Hj.
  apply in_seq in Hj. destruct Hj as [_ Hj].
  destruct (nth_error (g_threads G1) j) as [th1 |] eqn:E1; [| exfalso; apply nth_error_None in E1; lia].
  destruct (nth_error (g_threads G2) j) as [th2 |] eqn:E2; [| exfalso; apply nth_error_None in E2; lia].
  destruct (Nat.eqb (cta j) (cta k)); [exact (arrived_view th1 th2 (Hview j th1 th2 E1 E2)) | reflexivity].
Qed.

Lemma passed_block_view : forall G1 G2 b, List.length (g_threads G1) = List.length (g_threads G2) ->
    (forall n th1 th2, nth_error (g_threads G1) n = Some th1 -> nth_error (g_threads G2) n = Some th2 -> view th1 = view th2) ->
    passed_block G2 b -> passed_block G1 b.
Proof.
  intros G1 G2 b Hlen Hview (n & th2 & Hn & Hc & Hp).
  assert (Hlt : (n < List.length (g_threads G2))%nat) by (apply nth_error_Some; rewrite Hn; discriminate).
  destruct (nth_error (g_threads G1) n) as [th1 |] eqn:E1; [| exfalso; apply nth_error_None in E1; lia].
  exists n, th1. split; [exact E1 |]. split; [exact Hc |].
  destruct (view_parts th1 th2 (Hview n th1 th2 E1 Hn)) as (_ & _ & Hpd). rewrite Hpd. exact Hp.
Qed.

(* ---- the launch ---- *)

Lemma grid0_thread : forall m0 sh0 lm0 n th, nth_error (g_threads (grid0 inits count m0 sh0 lm0)) n = Some th ->
    th = start (inits n) m0 (lm0 n).
Proof.
  intros m0 sh0 lm0 n th H. cbn [g_threads grid0] in H.
  assert (Hlt : (n < List.length (map (fun n => start (inits n) m0 (lm0 n)) (seq 0 count)))%nat)
    by (apply nth_error_Some; rewrite H; discriminate).
  rewrite map_length, seq_length in Hlt.
  rewrite (map_nth_error _ _ _ (nth_error_seq0 count n Hlt)) in H. injection H as E. subst th. reflexivity.
Qed.

(* The step of the second run of thread n, given that it reaches the same
   state as the first run's, in terms of the second run's memories. *)
Lemma tstep_same : forall n th1 th2 m2 sm2 i s',
    view th1 = view th2 ->
    fetch prog (blk (th_state th1)) (idx (th_state th1)) = Some i ->
    step prog (wdn F)
         (e_mem (env_of F (inits n) (os n) (fun t' => if Nat.eqb t' (th_count th2) then m2 else th_hist th2 t')
                        (fun t' => if Nat.eqb t' (th_count th2) then sm2 else th_shist th2 t')
                        (fun t' => if Nat.eqb t' (th_count th2) then th_local th2 else th_lhist th2 t')))
         (e_arb (env_of F (inits n) (os n) (fun t' => if Nat.eqb t' (th_count th2) then m2 else th_hist th2 t')
                        (fun t' => if Nat.eqb t' (th_count th2) then sm2 else th_shist th2 t')
                        (fun t' => if Nat.eqb t' (th_count th2) then th_local th2 else th_lhist th2 t')))
         (e_shm (env_of F (inits n) (os n) (fun t' => if Nat.eqb t' (th_count th2) then m2 else th_hist th2 t')
                        (fun t' => if Nat.eqb t' (th_count th2) then sm2 else th_shist th2 t')
                        (fun t' => if Nat.eqb t' (th_count th2) then th_local th2 else th_lhist th2 t')))
         (inits n) (lf_cells F) (lf_locals F) (lf_local_window F) (th_count th2) (th_state th2) = Some s' ->
    exists th2' m2' sm2', tstep F prog bar (inits n) (os n) m2 sm2 th2 = Some (th2', m2', sm2') /\
      th_state th2' = s' /\ th_count th2' = S (th_count th1) /\
      th_lhist th2' = (fun t' => if Nat.eqb t' (th_count th1) then th_local th2 else th_lhist th2 t') /\
      th_local th2' = local_write F (inits n)
                        (o_wr (os n) (th_count th1) (shared_bytes F (inits n) sm2) (local_bytes F (inits n) (th_local th2)))
                        (regs (th_state th1)) i (th_local th2) /\
      th_passed th2' = th_passed th1 || at_bar bar (th_state th1) /\
      m2' = mem_write F (inits n)
              (o_wr (os n) (th_count th1) (shared_bytes F (inits n) sm2) (local_bytes F (inits n) (th_local th2)))
              (regs (th_state th1)) i m2 /\
      sm2' = shared_write F (inits n)
               (o_wr (os n) (th_count th1) (shared_bytes F (inits n) sm2) (local_bytes F (inits n) (th_local th2)))
               (regs (th_state th1)) i sm2.
Proof.
  intros n th1 th2 m2 sm2 i s' Hview Ef Hstep2.
  destruct (view_parts th1 th2 Hview) as (Hst & Hct & Hpd).
  assert (Ef2 : fetch prog (blk (th_state th2)) (idx (th_state th2)) = Some i) by (rewrite <- Hst; exact Ef).
  unfold tstep. cbv zeta. rewrite Ef2.
  match goal with
  | |- context [step ?p1 ?p2 ?p3 ?p4 ?p5 ?p6 ?p7 ?p8 ?p9 ?p10 ?p11] =>
      destruct (step p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11) as [s'' |] eqn:Es2
  end; [| discriminate (eq_trans (eq_sym Es2) Hstep2)].
  injection (eq_trans (eq_sym Es2) Hstep2) as ->.
  eexists _, _, _. split; [reflexivity |]. cbn [th_state th_count th_lhist th_local th_passed].
  rewrite <- Hst, <- Hct, <- Hpd. repeat split; reflexivity.
Qed.

(* When the two shared memories agree on the bytes of the block's variables
   and the two local memories on the bytes of the thread's variables, the
   two runs take the same step of thread n, and the memories still
   agree. *)
Lemma tstep_agree_read : forall n th1 th2 m1 m2 sm1 sm2 th1' m1' sm1',
    thread_ok F prog (inits n) (os n) th1 -> thread_ok F prog (inits n) (os n) th2 ->
    (forall t, P (th_hist th1 t)) -> (forall t, P (th_hist th2 t)) -> P m1 -> P m2 ->
    view th1 = view th2 -> agree m1 m2 -> sm_agree (inits n) sm1 sm2 ->
    lm_agree (inits n) (th_local th1) (th_local th2) ->
    tstep F prog bar (inits n) (os n) m1 sm1 th1 = Some (th1', m1', sm1') ->
    exists th2' m2' sm2', tstep F prog bar (inits n) (os n) m2 sm2 th2 = Some (th2', m2', sm2') /\
      view th1' = view th2' /\ agree m1' m2' /\ sm_agree (inits n) sm1' sm2' /\
      lm_agree (inits n) (th_local th1') (th_local th2').
Proof.
  intros n th1 th2 m1 m2 sm1 sm2 th1' m1' sm1' Hok1 Hok2 Hh1 Hh2 Hm1 Hm2 Hview Hag Hsm Hlm H1.
  destruct (view_parts th1 th2 Hview) as (Hst & Hct & Hpd).
  destruct (tstep_spec F prog bar _ _ _ _ _ _ _ _ H1)
    as [i [Ef [Hc1 [Hh1' [Hs1' [Hl1' [Hstep1 [Hloc1 [Hpass1 [Hm1' Hsm1']]]]]]]]]].
  set (hist1 := fun t' => if Nat.eqb t' (th_count th1) then m1 else th_hist th1 t') in *.
  set (sh1 := fun t' => if Nat.eqb t' (th_count th1) then sm1 else th_shist th1 t') in *.
  set (lh1 := fun t' => if Nat.eqb t' (th_count th1) then th_local th1 else th_lhist th1 t') in *.
  set (hist2 := fun t' => if Nat.eqb t' (th_count th2) then m2 else th_hist th2 t').
  set (sh2 := fun t' => if Nat.eqb t' (th_count th2) then sm2 else th_shist th2 t').
  set (lh2 := fun t' => if Nat.eqb t' (th_count th2) then th_local th2 else th_lhist th2 t').
  destruct (step_shared_reads n th1 m1 sm1 th1' m1' sm1' hist1 Hh1 Hm1 Hok1 H1 eq_refl) as [i' [Ef' [Hlds Hload]]].
  rewrite Ef in Ef'. injection Ef' as <-.
  assert (Hstep2 : step prog (wdn F) (e_mem (env_of F (inits n) (os n) hist2 sh2 lh2)) (e_arb (env_of F (inits n) (os n) hist2 sh2 lh2))
               (e_shm (env_of F (inits n) (os n) hist2 sh2 lh2)) (inits n) (lf_cells F) (lf_locals F) (lf_local_window F)
               (th_count th2) (th_state th2) = Some (th_state th1')).
  { rewrite <- Hct, <- Hst, <- Hstep1, Hh1', Hs1', Hl1'. symmetry.
    apply (step_env_eq F prog _ _ _ _ _ _ (inits n) (th_count th1) (th_state th1) i Ef).
    - (* a load *)
      intros g gen bits d a off -> Hg. destruct (Hload g gen bits d a off eq_refl Hg) as [Hmod [Hshw [Hlcw Hglob]]].
      cbn [env_of e_mem]. unfold hist1, hist2, sh1, sh2, lh1, lh2. rewrite <- Hct, !Nat.eqb_refl. cbv iota.
      cbv zeta in Hshw, Hlcw, Hglob.
      set (x := trunc 64 (regs (th_state th1) a + off)) in *.
      assert (Hb : bits = (8 * (bits / 8))%nat) by (pose proof (Nat.div_mod bits 8 ltac:(lia)); lia).
      destruct (in_space (inits n) (lf_shared_window F) x) eqn:Hsp.
      + rewrite Hb. apply trunc_read_agree. intros y Hy. apply Hsm.
        apply Hshw; [first [exact Hsp | reflexivity] | rewrite Nat2Z.inj_div in Hy; change (Z.of_nat 8) with 8 in Hy; lia].
      + destruct (in_space (inits n) (lf_local_window F) x) eqn:Hlp.
        * rewrite Hb. apply trunc_read_agree. intros y Hy. apply Hlm.
          apply Hlcw; [first [exact Hlp | reflexivity] | rewrite Nat2Z.inj_div in Hy; change (Z.of_nat 8) with 8 in Hy; lia].
        * rewrite Hb. apply trunc_read_agree. intros y Hy. apply Hag.
          apply Hglob; [first [exact Hsp | reflexivity] | first [exact Hlp | reflexivity] |
                        rewrite Nat2Z.inj_div in Hy; change (Z.of_nat 8) with 8 in Hy; lia].
    - (* a shared load *)
      intros g bits d a off -> Hg. destruct (Hlds g bits d a off eq_refl Hg) as [Hmod Hdecl].
      cbn [env_of e_shm]. unfold sh1, sh2. rewrite <- Hct, !Nat.eqb_refl. cbv iota.
      assert (Hb : bits = (8 * (bits / 8))%nat) by (pose proof (Nat.div_mod bits 8 ltac:(lia)); lia).
      rewrite Hb. apply trunc_read_agree. intros y Hy. apply Hsm. apply Hdecl.
      rewrite Nat2Z.inj_div in Hy. change (Z.of_nat 8) with 8 in Hy. lia.
    - (* the oracle *)
      cbn [env_of e_arb]. unfold sh1, sh2, lh1, lh2. rewrite <- Hct, !Nat.eqb_refl. cbv iota.
      rewrite (shared_bytes_agree F (inits n) sm1 sm2 Hsm).
      rewrite (local_bytes_agree F (inits n) (th_local th1) (th_local th2) Hlm). reflexivity. }
  destruct (tstep_same n th1 th2 m2 sm2 i (th_state th1') Hview Ef Hstep2)
    as (th2' & m2' & sm2' & H2 & Hst' & Hct' & Hlh' & Hloc' & Hpass' & Hm2' & Hsm2').
  rewrite (shared_bytes_agree F (inits n) sm1 sm2 Hsm) in Hloc1, Hm1', Hsm1'.
  rewrite (local_bytes_agree F (inits n) (th_local th1) (th_local th2) Hlm) in Hloc1, Hm1', Hsm1'.
  exists th2', m2', sm2'. split; [exact H2 |]. split.
  { unfold view. rewrite Hst', Hct', Hpass', Hc1, Hpass1. reflexivity. }
  split.
  { rewrite Hm1', Hm2'. unfold mem_write.
    destruct (access_at (win_of F (inits n)) (regs (th_state th1)) i) as [[[[k x] w] v] |]; [| exact Hag].
    destruct k; [exact Hag | apply write_agree; exact Hag | apply write_agree; exact Hag]. }
  split.
  { rewrite Hsm1', Hsm2'. apply shared_write_agree. exact Hsm. }
  rewrite Hloc1, Hloc'. apply local_write_agree. exact Hlm.
Qed.


(* ---- a kernel with no shared variables ---- *)

(* With no shared variables, the gate refuses every access to shared
   memory (`private_in` in GateMeaning.v) and the oracles see no bytes of
   it. So two runs agree from any two shared memories, with or without a
   barrier, when their local memories agree on the bytes of the local
   variables. *)
Section NoShared.

Hypothesis Hnone : lf_shared F = [].

Lemma sm_agree_none : forall init sm1 sm2, sm_agree init sm1 sm2.
Proof. intros init sm1 sm2 x [u [f [Hin _]]]. rewrite Hnone in Hin. destruct Hin. Qed.

Record ni0_inv (G1 G2 : grid) : Prop := {
  ni0_mem : agree (g_mem G1) (g_mem G2);
  ni0_count : List.length (g_threads G1) = count;
  ni0_len : List.length (g_threads G2) = count;
  ni0_view : forall n th1 th2, nth_error (g_threads G1) n = Some th1 -> nth_error (g_threads G2) n = Some th2 ->
             view th1 = view th2;
  ni0_local : forall n th1 th2, nth_error (g_threads G1) n = Some th1 -> nth_error (g_threads G2) n = Some th2 ->
              lm_agree (inits n) (th_local th1) (th_local th2)
}.

Lemma ni0_step : forall G1 G2 k, grid_inv F prog inits os P G1 -> grid_inv F prog inits os P G2 -> ni0_inv G1 G2 ->
    ni0_inv (gstep F prog bar inits os cta G1 k) (gstep F prog bar inits os cta G2 k).
Proof.
  intros G1 G2 k [HP1 Hth1] [HP2 Hth2] [Hmem Hc1 Hc2 Hview Hlocal].
  destruct (nth_error (g_threads G1) k) as [th1 |] eqn:E1.
  2: { assert (E2 : nth_error (g_threads G2) k = None) by (apply nth_error_None; apply nth_error_None in E1; lia).
       assert (HG1 : gstep F prog bar inits os cta G1 k = G1) by (unfold gstep; rewrite E1; reflexivity).
       assert (HG2 : gstep F prog bar inits os cta G2 k = G2) by (unfold gstep; rewrite E2; reflexivity).
       rewrite HG1, HG2. constructor; assumption. }
  destruct (nth_error (g_threads G2) k) as [th2 |] eqn:E2.
  2: { exfalso. apply nth_error_None in E2.
       assert (Hk : (k < List.length (g_threads G1))%nat) by (apply nth_error_Some; rewrite E1; discriminate). lia. }
  pose proof (Hview k th1 th2 E1 E2) as Hv. destruct (view_parts th1 th2 Hv) as (Hst & _ & _).
  pose proof (Hlocal k th1 th2 E1 E2) as Hlm.
  destruct (Hth1 k th1 E1) as [Hok1 Hh1]. destruct (Hth2 k th2 E2) as [Hok2 Hh2].
  pose proof (block_arrived_view G1 G2 k ltac:(lia) Hview) as Hba.
  destruct (at_bar bar (th_state th1) && negb (block_arrived bar cta G1 k)) eqn:Hwait.
  - assert (HG1 : gstep F prog bar inits os cta G1 k = G1)
      by (unfold gstep; rewrite E1; cbv beta iota; rewrite Hwait; reflexivity).
    assert (HG2 : gstep F prog bar inits os cta G2 k = G2)
      by (unfold gstep; rewrite E2; cbv beta iota; rewrite <- Hst, <- Hba, Hwait; reflexivity).
    rewrite HG1, HG2. constructor; assumption.
  - destruct (tstep F prog bar (inits k) (os k) (g_mem G1) (g_shared G1 (cta k)) th1) as [[[th1' m1'] sm1'] |] eqn:T1.
    + destruct (tstep_agree_read k th1 th2 (g_mem G1) (g_mem G2) (g_shared G1 (cta k)) (g_shared G2 (cta k)) th1' m1' sm1'
                  Hok1 Hok2 Hh1 Hh2 HP1 HP2 Hv Hmem (sm_agree_none _ _ _) Hlm T1)
        as (th2' & m2' & sm2' & T2 & Hv' & Hag' & _ & Hlm').
      assert (HG1 : gstep F prog bar inits os cta G1 k =
                    {| g_mem := m1'; g_shared := fun b => if Nat.eqb b (cta k) then sm1' else g_shared G1 b;
                       g_threads := set_nth (g_threads G1) k th1' |})
        by (unfold gstep; rewrite E1; cbv beta iota; rewrite Hwait; cbv iota; rewrite T1; reflexivity).
      assert (HG2 : gstep F prog bar inits os cta G2 k =
                    {| g_mem := m2'; g_shared := fun b => if Nat.eqb b (cta k) then sm2' else g_shared G2 b;
                       g_threads := set_nth (g_threads G2) k th2' |})
        by (unfold gstep; rewrite E2; cbv beta iota; rewrite <- Hst, <- Hba, Hwait; cbv iota; rewrite T2; reflexivity).
      rewrite HG1, HG2. constructor.
      * exact Hag'.
      * cbn [g_threads]. rewrite length_set_nth. exact Hc1.
      * cbn [g_threads]. rewrite length_set_nth. exact Hc2.
      * intros n thn1 thn2 Hn1 Hn2. cbn [g_threads] in Hn1, Hn2. destruct (Nat.eq_dec n k) as [-> | Hnk].
        -- rewrite (set_nth_at _ _ _ _ _ E1) in Hn1. rewrite (set_nth_at _ _ _ _ _ E2) in Hn2.
           injection Hn1 as <-. injection Hn2 as <-. exact Hv'.
        -- rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn1. rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn2.
           exact (Hview n thn1 thn2 Hn1 Hn2).
      * intros n thn1 thn2 Hn1 Hn2. cbn [g_threads] in Hn1, Hn2. destruct (Nat.eq_dec n k) as [-> | Hnk].
        -- rewrite (set_nth_at _ _ _ _ _ E1) in Hn1. rewrite (set_nth_at _ _ _ _ _ E2) in Hn2.
           injection Hn1 as <-. injection Hn2 as <-. exact Hlm'.
        -- rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn1. rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn2.
           exact (Hlocal n thn1 thn2 Hn1 Hn2).
    + destruct (tstep F prog bar (inits k) (os k) (g_mem G2) (g_shared G2 (cta k)) th2) as [[[th2' m2'] sm2'] |] eqn:T2.
      * exfalso.
        destruct (tstep_agree_read k th2 th1 (g_mem G2) (g_mem G1) (g_shared G2 (cta k)) (g_shared G1 (cta k)) th2' m2' sm2'
                    Hok2 Hok1 Hh2 Hh1 HP2 HP1 (eq_sym Hv) (agree_sym _ _ Hmem) (sm_agree_none _ _ _)
                    (lm_agree_sym _ _ _ Hlm) T2)
          as (th1' & m1' & sm1' & T1' & _).
        rewrite T1' in T1. discriminate T1.
      * assert (HG1 : gstep F prog bar inits os cta G1 k = G1)
          by (unfold gstep; rewrite E1; cbv beta iota; rewrite Hwait; cbv iota; rewrite T1; reflexivity).
        assert (HG2 : gstep F prog bar inits os cta G2 k = G2)
          by (unfold gstep; rewrite E2; cbv beta iota; rewrite <- Hst, <- Hba, Hwait; cbv iota; rewrite T2; reflexivity).
        rewrite HG1, HG2. constructor; assumption.
Qed.

Lemma ni0_init : forall m0 m0' sh0 sh0' lm0 lm0', agree m0 m0' -> (forall n, lm_agree (inits n) (lm0 n) (lm0' n)) ->
    ni0_inv (grid0 inits count m0 sh0 lm0) (grid0 inits count m0' sh0' lm0').
Proof.
  intros m0 m0' sh0 sh0' lm0 lm0' Hag Hlm.
  assert (Hlen : forall m sh lm, List.length (g_threads (grid0 inits count m sh lm)) = count)
    by (intros; cbn [g_threads grid0]; rewrite map_length, seq_length; reflexivity).
  constructor; [exact Hag | apply Hlen | apply Hlen | |].
  - intros n th1 th2 H1 H2. rewrite (grid0_thread _ _ _ _ _ H1), (grid0_thread _ _ _ _ _ H2). reflexivity.
  - intros n th1 th2 H1 H2. rewrite (grid0_thread _ _ _ _ _ H1), (grid0_thread _ _ _ _ _ H2). exact (Hlm n).
Qed.

Lemma ni0_fold : forall sched G1 G2, grid_inv F prog inits os P G1 -> grid_inv F prog inits os P G2 -> ni0_inv G1 G2 ->
    ni0_inv (fold_left (gstep F prog bar inits os cta) sched G1) (fold_left (gstep F prog bar inits os cta) sched G2).
Proof.
  induction sched as [| k sched IH]; intros G1 G2 Hg1 Hg2 Hni; [exact Hni |].
  cbn [fold_left]. apply IH.
  - exact (gstep_inv F prog bar Hsafe inits os cta P Hlink Hkeep G1 k Hg1).
  - exact (gstep_inv F prog bar Hsafe inits os cta P Hlink Hkeep G2 k Hg2).
  - exact (ni0_step G1 G2 k Hg1 Hg2 Hni).
Qed.

Theorem launch_ni0 : forall m0 m0' sh0 sh0' lm0 lm0' sched, P m0 -> P m0' -> agree m0 m0' ->
    (forall n, lm_agree (inits n) (lm0 n) (lm0' n)) ->
    ni0_inv (launch F prog bar inits os cta count m0 sh0 lm0 sched) (launch F prog bar inits os cta count m0' sh0' lm0' sched).
Proof.
  intros m0 m0' sh0 sh0' lm0 lm0' sched H0 H0' Hag Hlm. unfold launch. apply ni0_fold.
  - exact (grid0_inv F prog inits os P count m0 sh0 lm0 H0).
  - exact (grid0_inv F prog inits os P count m0' sh0' lm0' H0').
  - exact (ni0_init m0 m0' sh0 sh0' lm0 lm0' Hag Hlm).
Qed.

End NoShared.


(* ---- a kernel with shared variables: the zeroing prologue ---- *)

Section Prologue.


(* The prologue, for a kernel with shared variables. *)
Variable ps : prologue.
Hypothesis Hps : prologue_of F Mf prog = Some ps.
Hypothesis Hbar : bar = Some (bar_of ps).

(* The launch values of each thread. *)
Hypothesis Hbounds : forall n r lo hi, is_fixed F r = true -> lookup (lf_bounds F) r = Some (lo, hi) ->
                     lo <= inits n r <= hi.
Hypothesis Hsizes : forall n zp s c, pr_loops ps = Some zp -> In (s, c) (zp_terms zp) -> 0 <= inits n s < 2 ^ 64.
Hypothesis Hshared : forall n zp, pr_loops ps = Some zp -> 0 <= zero_elements zp (inits n) * zp_width zp <= shared_max.
(* The threads of a block share the addresses and bytes of its shared
   variables and the sizes. *)
Hypothesis Huniform : forall n m, cta n = cta m ->
    (forall u f, In (u, f) (lf_shared F) -> inits n u = inits m u /\ fval F (inits n) f = fval F (inits m) f) /\
    (forall zp, pr_loops ps = Some zp -> zero_elements zp (inits n) = zero_elements zp (inits m)).
(* Every thread index of a block is a thread of the launch. *)
Hypothesis Hcomplete : forall n T, (n < count)%nat -> 0 <= T < pr_threads ps ->
    exists m, (m < count)%nat /\ cta m = cta n /\ thread_of ps (inits m) = T.

Let Hsh := prologue_shape F Mf prog ps Hps.
Local Notation z := (pr_zero ps).

(* Thread n has stored zero to byte x of its block's shared memory, by a
   store of the prologue. *)
Definition stored_zero (n : nat) (th : thread) (x : Z) : Prop :=
  exists t s g a off w, (t < th_count th)%nat /\ replay n th t = Some s /\
    fetch prog (blk s) (idx s) = Some (IShA g KStore a off w (Some (OReg z))) /\
    holds g (regs s) = true /\ regs s z = 0 /\
    trunc 32 (regs s a + off) <= x < trunc 32 (regs s a + off) + w.

(* Whether a thread has passed the barrier records whether it executed it. *)
Definition passed_spec (n : nat) (th : thread) : Prop :=
  th_passed th = true <-> exists t s, (t < th_count th)%nat /\ replay n th t = Some s /\ (blk s, idx s) = bar_of ps.

(* ---- a thread before the barrier ---- *)

(* The arrival step of thread n, and what its run does before it. *)
Lemma arrival : forall n hist sh lh,
    let Ev := env_of F (inits n) (os n) hist sh lh in
    exists M rs, run' F prog Ev M = Some {| blk := fst (bar_of ps); idx := snd (bar_of ps); regs := rs |} /\
      (forall t s, (t < M)%nat -> run' F prog Ev t = Some s -> before_bar ps (blk s, idx s)) /\
      (forall x, share ps (inits n) x -> exists t s, (t < M)%nat /\ run' F prog Ev t = Some s /\
         exists g a off w, fetch prog (blk s) (idx s) = Some (IShA g KStore a off w (Some (OReg z))) /\
           holds g (regs s) = true /\ regs s z = 0 /\
           trunc 32 (regs s a + off) <= x < trunc 32 (regs s a + off) + w).
Proof.
  intros n hist sh lh Ev.
  destruct (prologue_run F Mf prog ps Hsh Ev (Hbounds n) (Hsizes n) (Hshared n))
    as (M & rs & HM & _ & Hbefore & Hwit).
  exists M, rs. split; [exact HM |]. split; [exact Hbefore |].
  intros x Hx. destruct (Hwit x Hx) as (t & s & Ht & Hrun & g & off & Hf & Hg & Hz & Hrange).
  exists t, s. split; [exact Ht |]. split; [exact Hrun |]. exists g, (pr_address ps), off, (pr_width ps).
  split; [exact Hf |]. split; [exact Hg |]. split; [exact Hz | exact Hrange].
Qed.

(* A thread that has not passed the barrier is at or before it, and every
   step it took was before it. *)
Lemma not_passed : forall n th, thread_ok F prog (inits n) (os n) th -> passed_spec n th -> th_passed th = false ->
    (forall t s, (t < th_count th)%nat -> replay n th t = Some s -> before_bar ps (blk s, idx s)) /\
    (before_bar ps (blk (th_state th), idx (th_state th)) \/ (blk (th_state th), idx (th_state th)) = bar_of ps).
Proof.
  intros n th Hok Hspec Hp.
  destruct (arrival n (th_hist th) (th_shist th) (th_lhist th)) as (M & rs & HM & Hbefore & _).
  assert (Hnb : forall t s, (t < th_count th)%nat -> replay n th t = Some s -> (blk s, idx s) <> bar_of ps).
  { intros t s Ht Hr E. assert (th_passed th = true) by (apply Hspec; exists t, s; auto). congruence. }
  assert (Hle : (th_count th <= M)%nat).
  { destruct (Nat.le_gt_cases (th_count th) M) as [Hle | Hgt]; [exact Hle |].
    exfalso. apply (Hnb M _ Hgt HM). reflexivity. }
  split.
  - intros t s Ht Hr. exact (Hbefore t s ltac:(lia) Hr).
  - unfold thread_ok in Hok.
    exact (run_region F Mf prog ps Hsh (env_of F (inits n) (os n) (th_hist th) (th_shist th) (th_lhist th))
             (th_count th) (th_state th) Hok Hnb).
Qed.

(* A thread at the barrier that has not passed it has run its whole
   prologue: it has stored zero over its share. *)
Lemma at_bar_stored : forall n th, thread_ok F prog (inits n) (os n) th -> passed_spec n th -> th_passed th = false ->
    at_bar bar (th_state th) = true -> forall x, share ps (inits n) x -> stored_zero n th x.
Proof.
  intros n th Hok Hspec Hp Hat x Hx.
  destruct (arrival n (th_hist th) (th_shist th) (th_lhist th)) as (M & rs & HM & Hbefore & Hwit).
  destruct (not_passed n th Hok Hspec Hp) as [Hall _].
  apply (at_bar_spec bar (th_state th) (bar_of ps) Hbar) in Hat.
  (* the count is the arrival step *)
  assert (HcM : th_count th = M).
  { destruct (Nat.lt_trichotomy (th_count th) M) as [Hlt | [Heq | Hgt]]; [| exact Heq |].
    - exfalso. pose proof (Hbefore (th_count th) (th_state th) Hlt Hok) as Hb. rewrite Hat in Hb.
      exact (not_before_bar ps Hb).
    - exfalso. pose proof (Hall M _ Hgt HM) as Hb. cbn [blk idx] in Hb.
      rewrite <- (surjective_pairing (bar_of ps)) in Hb. exact (not_before_bar ps Hb). }
  destruct (Hwit x Hx) as (t & s & Ht & Hrun & g & a & off & w & Hf & Hg & Hz & Hrange).
  exists t, s, g, a, off, w. split; [lia |]. split; [exact Hrun |]. split; [exact Hf |].
  split; [exact Hg |]. split; [exact Hz | exact Hrange].
Qed.

(* ---- the effect of a step ---- *)

(* A quiet step writes no global memory and no local memory, and writes
   shared memory only by a store of the zero register. *)
Lemma quiet_mem_write : forall init wr rs i m, quiet z i = true -> mem_write F init wr rs i m = m.
Proof. intros init wr rs i m Hq. unfold mem_write. rewrite (quiet_access z i (win_of F init) rs Hq). reflexivity. Qed.

Lemma quiet_local_write : forall init wr rs i lm, quiet z i = true -> local_write F init wr rs i lm = lm.
Proof.
  intros init wr rs i lm Hq. unfold local_write. destruct (negb (holds (guard_of i) rs)); [reflexivity |].
  destruct i; cbn [quiet] in Hq; try discriminate; reflexivity.
Qed.

Definition quiet_store (rs : reg -> Z) (i : instr) (sm : memory) : memory :=
  match i with
  | IShA g KStore a off w (Some (OReg r)) => if holds g rs then write_bytes sm (trunc 32 (rs a + off)) w (rs r) else sm
  | _ => sm
  end.

Lemma quiet_shared_write : forall init wr rs i sm, quiet z i = true ->
    shared_write F init wr rs i sm = quiet_store rs i sm.
Proof.
  intros init wr rs i sm Hq. unfold shared_write, quiet_store.
  destruct i; cbn [quiet] in Hq; try discriminate; cbn [guard_of]; try (destruct (holds g rs); reflexivity).
  destruct k; try discriminate. destruct v as [[r | zz] |]; try discriminate.
  destruct (holds g rs); [| reflexivity]. unfold stored. cbn [opv]. reflexivity.
Qed.

(* The store of a quiet instruction stores the zero register, at the same
   address and width in any two shared memories. *)
Lemma quiet_store_pair : forall rs i sm1 sm2, quiet z i = true ->
    (quiet_store rs i sm1 = sm1 /\ quiet_store rs i sm2 = sm2) \/
    (exists g a off w, i = IShA g KStore a off w (Some (OReg z)) /\
       quiet_store rs i sm1 = write_bytes sm1 (trunc 32 (rs a + off)) w (rs z) /\
       quiet_store rs i sm2 = write_bytes sm2 (trunc 32 (rs a + off)) w (rs z)).
Proof.
  intros rs i sm1 sm2 Hq. unfold quiet_store. destruct i; try (left; split; reflexivity).
  destruct k; try (left; split; reflexivity). destruct v as [[r | zz] |]; try (left; split; reflexivity).
  cbn [quiet] in Hq. apply Pos.eqb_eq in Hq. subst r. destruct (holds g rs); [| left; split; reflexivity].
  right. exists g, a, off, width. split; [reflexivity |]. split; reflexivity.
Qed.

(* The thread's share is the same seen from any thread of its block. *)
Lemma share_at_uniform : forall n m T x, cta n = cta m -> share_at ps (inits n) T x -> share_at ps (inits m) T x.
Proof.
  intros n m T x Hc H. destruct (Huniform n m Hc) as [Hvars Hcount].
  assert (Hbase : forall u, In u (map fst (lf_shared F)) -> inits n u = inits m u).
  { intros u Hu. apply in_map_iff in Hu. destruct Hu as [[u' f] [<- Hin]]. exact (proj1 (Hvars _ _ Hin)). }
  destruct H as [(v & Hv & Hx & Hmod) | (zp & B & Hl & HB & Hx & Hmod)].
  - left. exists v. split; [exact Hv |].
    destruct (sh_static_vars _ _ _ _ Hsh v Hv) as (f & Hin & _).
    rewrite <- (Hbase (sv_base v) ltac:(apply in_map_iff; exists (sv_base v, f); split; [reflexivity | exact Hin])).
    split; [exact Hx | exact Hmod].
  - right. exists zp, B. split; [exact Hl |]. split; [exact HB |].
    destruct (sh_loops _ _ _ _ Hsh zp Hl) as (_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hbases).
    destruct (Hbases B HB) as (f & Hin & _).
    rewrite <- (Hbase B ltac:(apply in_map_iff; exists (B, f); split; [reflexivity | exact Hin])).
    assert (Hcount' : dyn_count ps (inits n) = dyn_count ps (inits m))
      by (unfold dyn_count; rewrite Hl; exact (Hcount zp Hl)).
    rewrite <- Hcount'. split; [exact Hx | exact Hmod].
Qed.

Lemma declared_uniform : forall n m x, cta n = cta m -> declared F (inits n) x -> declared F (inits m) x.
Proof.
  intros n m x Hc [u [f [Hin Hx]]]. destruct (Huniform n m Hc) as [Hvars _].
  destruct (Hvars u f Hin) as [Hu Hf]. exists u, f. split; [exact Hin |]. rewrite <- Hu, <- Hf. exact Hx.
Qed.

(* A thread's own environment, for the lemmas of GatePrologue.v. *)
Let Ev n th := env_of F (inits n) (os n) (th_hist th) (th_shist th) (th_lhist th).

(* The zero register holds zero when a thread before the barrier stores it. *)
Lemma zero_reg : forall n th g a off w, thread_ok F prog (inits n) (os n) th ->
    (forall t s, (t < th_count th)%nat -> replay n th t = Some s -> before_bar ps (blk s, idx s)) ->
    fetch prog (blk (th_state th)) (idx (th_state th)) = Some (IShA g KStore a off w (Some (OReg z))) ->
    regs (th_state th) z = 0.
Proof.
  intros n th g a off w Hok Hbefore Ef. destruct (Nat.eq_dec (th_count th) 0) as [Hc0 | Hc0].
  - (* the first instruction is no store *)
    exfalso. unfold thread_ok, thread_run, run' in Hok. rewrite Hc0 in Hok. cbn [run] in Hok.
    injection Hok as Hs0. rewrite <- Hs0 in Ef. cbn [blk idx] in Ef.
    pose proof (sh_index F Mf prog ps Hsh) as Hidx.
    assert (Hf0 : fetch prog 0 0 = Some (IImm None z 0)).
    { unfold fetch. rewrite <- (firstn_skipn 6 (block_body prog 0)), Hidx. reflexivity. }
    rewrite Hf0 in Ef. discriminate Ef.
  - exact (proj2 (pre_region_zero F Mf prog ps Hsh (Ev n th) (th_count th) (th_state th) Hok Hbefore ltac:(lia))).
Qed.

(* The two runs take the same step of thread n. *)
Lemma tstep_agree : forall n th1 th2 m1 m2 sm1 sm2 th1' m1' sm1',
    thread_ok F prog (inits n) (os n) th1 -> thread_ok F prog (inits n) (os n) th2 ->
    (forall t, P (th_hist th1 t)) -> (forall t, P (th_hist th2 t)) -> P m1 -> P m2 ->
    view th1 = view th2 -> agree m1 m2 -> passed_spec n th1 ->
    (th_passed th1 = true -> sm_agree (inits n) sm1 sm2) ->
    lm_agree (inits n) (th_local th1) (th_local th2) ->
    tstep F prog bar (inits n) (os n) m1 sm1 th1 = Some (th1', m1', sm1') ->
    exists th2' m2' sm2', tstep F prog bar (inits n) (os n) m2 sm2 th2 = Some (th2', m2', sm2') /\
      view th1' = view th2' /\ agree m1' m2' /\
      (th_passed th1 = true -> sm_agree (inits n) sm1' sm2') /\
      (th_passed th1 = false ->
         (sm1' = sm1 /\ sm2' = sm2) \/ (exists x w, sm1' = write_bytes sm1 x w 0 /\ sm2' = write_bytes sm2 x w 0)) /\
      lm_agree (inits n) (th_local th1') (th_local th2').
Proof.
  intros n th1 th2 m1 m2 sm1 sm2 th1' m1' sm1' Hok1 Hok2 Hh1 Hh2 Hm1 Hm2 Hview Hag Hspec Hsm Hlm H1.
  destruct (view_parts th1 th2 Hview) as (Hst & Hct & Hpd).
  destruct (tstep_spec F prog bar _ _ _ _ _ _ _ _ H1)
    as [i [Ef [Hc1 [Hh1' [Hs1' [Hl1' [Hstep1 [Hloc1 [Hpass1 [Hm1' Hsm1']]]]]]]]]].
  set (hist2 := fun t' => if Nat.eqb t' (th_count th2) then m2 else th_hist th2 t').
  set (sh2 := fun t' => if Nat.eqb t' (th_count th2) then sm2 else th_shist th2 t').
  set (lh2 := fun t' => if Nat.eqb t' (th_count th2) then th_local th2 else th_lhist th2 t').
  (* the two cases: before or at the barrier, or past it *)
  destruct (th_passed th1) eqn:Hp.
  - (* past the barrier: every read agrees *)
    destruct (tstep_agree_read n th1 th2 m1 m2 sm1 sm2 th1' m1' sm1' Hok1 Hok2 Hh1 Hh2 Hm1 Hm2 Hview Hag
                (Hsm ltac:(first [exact Hp | reflexivity])) Hlm H1) as (th2' & m2' & sm2' & H2 & Hv' & Hag' & Hsm' & Hlm').
    exists th2', m2', sm2'. split; [exact H2 |]. split; [exact Hv' |]. split; [exact Hag' |].
    split; [intros _; exact Hsm' |].
    split; [intros E; first [discriminate E | rewrite E in Hp; discriminate Hp] | exact Hlm'].
  - (* before or at the barrier: the step reads no memory *)
    destruct (not_passed n th1 Hok1 Hspec Hp) as [Hbefore Hpos].
    assert (Hquiet : quiet z i = true \/ i = INop).
    { destruct Hpos as [Hpos | Hpos].
      - left. exact (region_quiet F Mf prog ps Hsh (blk (th_state th1)) (idx (th_state th1)) i Hpos Ef).
      - right. pose proof (fetch_bar F Mf prog ps Hsh) as Hf. rewrite <- Hpos in Hf. cbn [fst snd] in Hf.
        rewrite Ef in Hf. injection Hf as E. exact E. }
    assert (Hstep2 : step prog (wdn F) (e_mem (env_of F (inits n) (os n) hist2 sh2 lh2)) (e_arb (env_of F (inits n) (os n) hist2 sh2 lh2))
                 (e_shm (env_of F (inits n) (os n) hist2 sh2 lh2)) (inits n) (lf_cells F) (lf_locals F) (lf_local_window F)
                 (th_count th2) (th_state th2) = Some (th_state th1')).
    { rewrite <- Hct, <- Hst, <- Hstep1, Hh1', Hs1', Hl1'. destruct Hquiet as [Hq | ->].
      - symmetry. exact (quiet_step prog z _ _ _ _ _ _ _ (inits n) (lf_cells F) (lf_locals F) (lf_local_window F)
                           (th_count th1) (th_state th1) i Ef Hq).
      - unfold step. rewrite Ef. reflexivity. }
    destruct (tstep_same n th1 th2 m2 sm2 i (th_state th1') Hview Ef Hstep2)
      as (th2' & m2' & sm2' & H2 & Hst' & Hct' & Hlh' & Hloc' & Hpass' & Hm2' & Hsm2').
    try rewrite Hp in Hpass'.
    (* the zero register holds zero when the step stores *)
    assert (Hz : forall g a off w, i = IShA g KStore a off w (Some (OReg z)) -> regs (th_state th1) z = 0).
    { intros g a off w Hi. exact (zero_reg n th1 g a off w Hok1 Hbefore ltac:(rewrite Ef, Hi; reflexivity)). }
    exists th2', m2', sm2'. split; [exact H2 |].
    destruct Hquiet as [Hq | Hnop].
    + (* a quiet instruction *)
      rewrite (quiet_local_write _ _ _ _ _ Hq) in Hloc1. rewrite (quiet_local_write _ _ _ _ _ Hq) in Hloc'.
      rewrite (quiet_mem_write _ _ _ _ _ Hq) in Hm1'. rewrite (quiet_mem_write _ _ _ _ _ Hq) in Hm2'.
      rewrite (quiet_shared_write _ _ _ _ _ Hq) in Hsm1'. rewrite (quiet_shared_write _ _ _ _ _ Hq) in Hsm2'.
      split.
      { unfold view. rewrite Hst', Hct', Hpass', Hc1, Hpass1. reflexivity. }
      split; [rewrite Hm1', Hm2'; exact Hag |].
      split; [intros E; first [discriminate E | rewrite E in Hp; discriminate Hp] |].
      split; [| rewrite Hloc1, Hloc'; exact Hlm].
      intros _. destruct (quiet_store_pair (regs (th_state th1)) i sm1 sm2 Hq) as [[Hu1 Hu2] | (g & a & off & w & Hi & Hw1 & Hw2)].
      * left. split; [rewrite Hsm1'; exact Hu1 | rewrite Hsm2'; exact Hu2].
      * right. exists (trunc 32 (regs (th_state th1) a + off)), w.
        rewrite (Hz g a off w Hi) in Hw1, Hw2.
        split; [rewrite Hsm1'; exact Hw1 | rewrite Hsm2'; exact Hw2].
    + (* the barrier *)
      subst i.
      assert (Hnl : forall init wr rs lm, local_write F init wr rs INop lm = lm) by (intros; reflexivity).
      assert (Hnm : forall init wr rs m, mem_write F init wr rs INop m = m) by (intros; reflexivity).
      assert (Hns : forall init wr rs sm, shared_write F init wr rs INop sm = sm) by (intros; reflexivity).
      rewrite Hnl in Hloc1. rewrite Hnl in Hloc'. rewrite Hnm in Hm1'. rewrite Hnm in Hm2'.
      rewrite Hns in Hsm1'. rewrite Hns in Hsm2'.
      split.
      { unfold view. rewrite Hst', Hct', Hpass', Hc1, Hpass1. reflexivity. }
      split; [rewrite Hm1', Hm2'; exact Hag |].
      split; [intros E; first [discriminate E | rewrite E in Hp; discriminate Hp] |].
      split; [intros _; left; split; [exact Hsm1' | exact Hsm2'] |].
      rewrite Hloc1, Hloc'. exact Hlm.
Qed.

(* Each thread's record of passing the barrier matches its replay, and in a
   block where no thread has passed the barrier, every byte that a thread
   stored zero to holds zero. *)
Definition run_inv (G : grid) : Prop :=
  (forall n th, nth_error (g_threads G) n = Some th -> passed_spec n th) /\
  (forall n th, nth_error (g_threads G) n = Some th -> ~ passed_block G (cta n) ->
     forall x, stored_zero n th x -> g_shared G (cta n) x = 0).

Lemma passed_step : forall n th m sm th' m' sm', thread_ok F prog (inits n) (os n) th -> passed_spec n th ->
    tstep F prog bar (inits n) (os n) m sm th = Some (th', m', sm') -> passed_spec n th'.
Proof.
  intros n th m sm th' m' sm' Hok Hspec H.
  destruct (tstep_spec F prog bar _ _ _ _ _ _ _ _ H) as [i [Ef [Hc [_ [_ [_ [_ [_ [Hpass _]]]]]]]]].
  pose proof (tstep_run F prog bar _ _ _ _ _ _ _ _ Hok H) as Hrun.
  pose proof (replay_before n m sm th th' m' sm' Hok H) as Hrep.
  unfold passed_spec in *. rewrite Hpass, Hc, orb_true_iff. split.
  - intros [Hp | Hb].
    + destruct (proj1 Hspec Hp) as (t & s & Ht & Hr & Hq). exists t, s. split; [lia |]. split; [| exact Hq].
      rewrite Hrep by lia. exact Hr.
    + exists (th_count th), (th_state th). split; [lia |]. split; [exact Hrun |].
      exact (proj1 (at_bar_spec bar (th_state th) (bar_of ps) Hbar) Hb).
  - intros (t & s & Ht & Hr & Hq). destruct (Nat.lt_ge_cases t (th_count th)) as [Hlt | Hge].
    + left. apply (proj2 Hspec). exists t, s. split; [exact Hlt |]. split; [| exact Hq]. rewrite <- Hrep by lia. exact Hr.
    + right. assert (t = th_count th) by lia. subst t. injection (eq_trans (eq_sym Hrun) Hr) as <-.
      exact (proj2 (at_bar_spec bar (th_state th) (bar_of ps) Hbar) Hq).
Qed.

Lemma zero_step : forall G k, grid_inv F prog inits os P G -> run_inv G -> run_inv (gstep F prog bar inits os cta G k).
Proof.
  intros G k [_ Hth] [Hpassed Hzero].
  destruct (gstep_cases F prog bar inits os cta G k) as [HG | (th & th' & m' & sm' & Hnth & Hts & HG)]; rewrite HG.
  { split; assumption. }
  destruct (Hth k th Hnth) as [Hok _].
  split.
  - (* the record of passing *)
    intros n thn Hn. cbn [g_threads] in Hn. destruct (Nat.eq_dec n k) as [-> | Hnk].
    + rewrite (set_nth_at _ _ _ _ _ Hnth) in Hn. injection Hn as <-.
      exact (passed_step k th _ _ th' m' sm' Hok (Hpassed k th Hnth) Hts).
    + rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn. exact (Hpassed n thn Hn).
  - (* zero in a block that has not passed *)
    intros n thn Hn Hnp x Hx. cbn [g_threads g_shared] in Hn |- *.
    destruct (Nat.eqb (cta n) (cta k)) eqn:Eb; cbv iota.
    + apply Nat.eqb_eq in Eb.
      (* the stepping thread has not passed the barrier *)
      assert (Hk' : th_passed th' = false).
      { destruct (th_passed th') eqn:E; [| reflexivity]. exfalso. apply Hnp. exists k, th'.
        split; [exact (set_nth_at _ _ _ _ _ Hnth) |]. split; [symmetry; exact Eb | exact E]. }
      destruct (tstep_spec F prog bar _ _ _ _ _ _ _ _ Hts) as [i [Ef [Hc [_ [_ [_ [_ [_ [Hpass [_ Hsm']]]]]]]]]].
      rewrite Hk' in Hpass. symmetry in Hpass. apply orb_false_iff in Hpass. destruct Hpass as [Hp Hb].
      assert (Hnp0 : ~ passed_block G (cta k)).
      { intros (j & thj & Hj & Hcj & Hpj). apply Hnp. destruct (Nat.eq_dec j k) as [-> | Hjk].
        - rewrite Hnth in Hj. injection Hj as <-. rewrite Hpj in Hp. discriminate Hp.
        - exists j, thj. split; [cbn [g_threads]; rewrite (set_nth_other _ _ _ _ _ Hjk); exact Hj |].
          split; [rewrite Hcj; symmetry; exact Eb | exact Hpj]. }
      destruct (not_passed k th Hok (Hpassed k th Hnth) Hp) as [Hbefore Hpos].
      assert (Hq : quiet z i = true).
      { destruct Hpos as [Hpos | Hpos]; [exact (region_quiet F Mf prog ps Hsh _ _ i Hpos Ef) |].
        exfalso. rewrite (proj2 (at_bar_spec bar (th_state th) (bar_of ps) Hbar) Hpos) in Hb. discriminate Hb. }
      rewrite (quiet_shared_write _ _ _ _ _ Hq) in Hsm'.
      (* a byte that held zero still holds zero after the store *)
      assert (Hstay : forall y, g_shared G (cta k) y = 0 -> sm' y = 0).
      { intros y Hy. rewrite Hsm'.
        destruct (quiet_store_pair (regs (th_state th)) i (g_shared G (cta k)) (g_shared G (cta k)) Hq)
          as [[Hu _] | (g & a & off & w & Hi & Hw & _)]; [rewrite Hu; exact Hy |].
        rewrite Hw, (zero_reg k th g a off w Hok Hbefore ltac:(rewrite Ef, Hi; reflexivity)).
        destruct (write_zero (g_shared G (cta k)) (trunc 32 (regs (th_state th) a + off)) w y) as [H0 | H0];
          rewrite H0; [reflexivity | exact Hy]. }
      destruct (Nat.eq_dec n k) as [-> | Hnk].
      * rewrite (set_nth_at _ _ _ _ _ Hnth) in Hn. injection Hn as <-.
        destruct Hx as (t & s & g & a & off & w & Ht & Hr & Hf & Hg & Hz & Hrange).
        rewrite Hc in Ht. destruct (Nat.lt_ge_cases t (th_count th)) as [Hlt | Hge].
        -- apply Hstay. apply (Hzero k th Hnth Hnp0).
           exists t, s, g, a, off, w. rewrite <- (replay_before k _ _ th th' _ _ Hok Hts t ltac:(lia)). tauto.
        -- assert (t = th_count th) by lia. subst t.
           injection (eq_trans (eq_sym (tstep_run F prog bar _ _ _ _ _ _ _ _ Hok Hts)) Hr) as <-.
           rewrite Ef in Hf. injection Hf as ->.
           rewrite Hsm'. unfold quiet_store. rewrite Hg, Hz. apply write_zero_in. exact Hrange.
      * rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn. apply Hstay. rewrite <- Eb.
        apply (Hzero n thn Hn); [rewrite Eb; exact Hnp0 | exact Hx].
    + apply Nat.eqb_neq in Eb.
      destruct (Nat.eq_dec n k) as [-> | Hnk]; [exfalso; exact (Eb eq_refl) |].
      rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn. apply (Hzero n thn Hn); [| exact Hx].
      intros (j & thj & Hj & Hcj & Hpj). apply Hnp.
      destruct (Nat.eq_dec j k) as [-> | Hjk]; [exfalso; apply Eb; rewrite <- Hcj; reflexivity |].
      exists j, thj. split; [cbn [g_threads]; rewrite (set_nth_other _ _ _ _ _ Hjk); exact Hj |]. split; [exact Hcj | exact Hpj].
Qed.

(* ---- the invariant of two runs ---- *)

Record ni_inv (G1 G2 : grid) : Prop := {
  ni_mem : agree (g_mem G1) (g_mem G2);
  ni_count : List.length (g_threads G1) = count;
  ni_len : List.length (g_threads G2) = count;
  ni_view : forall n th1 th2, nth_error (g_threads G1) n = Some th1 -> nth_error (g_threads G2) n = Some th2 ->
            view th1 = view th2;
  ni_local : forall n th1 th2, nth_error (g_threads G1) n = Some th1 -> nth_error (g_threads G2) n = Some th2 ->
             lm_agree (inits n) (th_local th1) (th_local th2);
  ni_replay : forall n th1 th2, nth_error (g_threads G1) n = Some th1 -> nth_error (g_threads G2) n = Some th2 ->
              forall t, (t <= th_count th1)%nat -> replay n th1 t = replay n th2 t;
  ni_run1 : run_inv G1;
  ni_run2 : run_inv G2;
  ni_shared : forall n th1, nth_error (g_threads G1) n = Some th1 -> th_passed th1 = true ->
              sm_agree (inits n) (g_shared G1 (cta n)) (g_shared G2 (cta n))
}.

Lemma stored_zero_view : forall n th1 th2 x, view th1 = view th2 ->
    (forall t, (t <= th_count th1)%nat -> replay n th1 t = replay n th2 t) -> stored_zero n th1 x -> stored_zero n th2 x.
Proof.
  intros n th1 th2 x Hv Hrep (t & s & g & a & off & w & Ht & Hr & Hrest).
  destruct (view_parts th1 th2 Hv) as (_ & Hct & _).
  exists t, s, g, a, off, w. split; [lia |]. split; [| exact Hrest]. rewrite <- (Hrep t ltac:(lia)). exact Hr.
Qed.

Lemma sm_agree_uniform : forall n m sm1 sm2, cta n = cta m -> sm_agree (inits m) sm1 sm2 -> sm_agree (inits n) sm1 sm2.
Proof. intros n m sm1 sm2 Hc H x Hx. apply H. exact (declared_uniform n m x Hc Hx). Qed.

(* A step of the schedule keeps the invariant. *)
Lemma ni_step : forall G1 G2 k, grid_inv F prog inits os P G1 -> grid_inv F prog inits os P G2 -> ni_inv G1 G2 ->
    ni_inv (gstep F prog bar inits os cta G1 k) (gstep F prog bar inits os cta G2 k).
Proof.
  intros G1 G2 k Hg1 Hg2 Hni.
  pose proof (zero_step G1 k Hg1 (ni_run1 _ _ Hni)) as Hr1'.
  pose proof (zero_step G2 k Hg2 (ni_run2 _ _ Hni)) as Hr2'.
  destruct Hni as [Hmem Hc1 Hc2 Hview Hlocal Hrep Hr1 Hr2 Hshm].
  destruct Hg1 as [HP1 Hth1]. destruct Hg2 as [HP2 Hth2].
  destruct (nth_error (g_threads G1) k) as [th1 |] eqn:E1.
  2: { assert (E2 : nth_error (g_threads G2) k = None) by (apply nth_error_None; apply nth_error_None in E1; lia).
       assert (HG1 : gstep F prog bar inits os cta G1 k = G1) by (unfold gstep; rewrite E1; reflexivity).
       assert (HG2 : gstep F prog bar inits os cta G2 k = G2) by (unfold gstep; rewrite E2; reflexivity).
       rewrite HG1, HG2. constructor; assumption. }
  destruct (nth_error (g_threads G2) k) as [th2 |] eqn:E2.
  2: { exfalso. apply nth_error_None in E2.
       assert (Hk : (k < List.length (g_threads G1))%nat) by (apply nth_error_Some; rewrite E1; discriminate). lia. }
  assert (Hk : (k < count)%nat) by (rewrite <- Hc1; apply nth_error_Some; rewrite E1; discriminate).
  pose proof (Hview k th1 th2 E1 E2) as Hv. destruct (view_parts th1 th2 Hv) as (Hst & Hct & Hpd).
  pose proof (Hlocal k th1 th2 E1 E2) as Hlm.
  destruct (Hth1 k th1 E1) as [Hok1 Hh1]. destruct (Hth2 k th2 E2) as [Hok2 Hh2].
  pose proof (block_arrived_view G1 G2 k ltac:(lia) Hview) as Hba.
  destruct (at_bar bar (th_state th1) && negb (block_arrived bar cta G1 k)) eqn:Hwait.
  - (* the thread waits at the barrier in both runs *)
    assert (HG1 : gstep F prog bar inits os cta G1 k = G1)
      by (unfold gstep; rewrite E1; cbv beta iota; rewrite Hwait; reflexivity).
    assert (HG2 : gstep F prog bar inits os cta G2 k = G2)
      by (unfold gstep; rewrite E2; cbv beta iota; rewrite <- Hst, <- Hba, Hwait; reflexivity).
    rewrite HG1, HG2. constructor; assumption.
  - destruct (tstep F prog bar (inits k) (os k) (g_mem G1) (g_shared G1 (cta k)) th1) as [[[th1' m1'] sm1'] |] eqn:T1.
    + (* both runs step *)
      destruct (tstep_agree k th1 th2 (g_mem G1) (g_mem G2) (g_shared G1 (cta k)) (g_shared G2 (cta k)) th1' m1' sm1'
                  Hok1 Hok2 Hh1 Hh2 HP1 HP2 Hv Hmem (proj1 Hr1 k th1 E1) (Hshm k th1 E1) Hlm T1)
        as (th2' & m2' & sm2' & T2 & Hv' & Hag' & Hpass_agree & Hpre & Hlm').
      assert (HG1 : gstep F prog bar inits os cta G1 k =
                    {| g_mem := m1'; g_shared := fun b => if Nat.eqb b (cta k) then sm1' else g_shared G1 b;
                       g_threads := set_nth (g_threads G1) k th1' |})
        by (unfold gstep; rewrite E1; cbv beta iota; rewrite Hwait; cbv iota; rewrite T1; reflexivity).
      assert (HG2 : gstep F prog bar inits os cta G2 k =
                    {| g_mem := m2'; g_shared := fun b => if Nat.eqb b (cta k) then sm2' else g_shared G2 b;
                       g_threads := set_nth (g_threads G2) k th2' |})
        by (unfold gstep; rewrite E2; cbv beta iota; rewrite <- Hst, <- Hba, Hwait; cbv iota; rewrite T2; reflexivity).
      rewrite HG1 in Hr1' |- *. rewrite HG2 in Hr2' |- *.
      destruct (tstep_spec F prog bar _ _ _ _ _ _ _ _ T1) as [i [Ef [Hc1' [_ [_ [_ [_ [_ [Hpass1 _]]]]]]]]].
      pose proof (tstep_ok F prog bar _ _ _ _ _ _ _ _ Hok1 T1) as Hok1'.
      pose proof (tstep_ok F prog bar _ _ _ _ _ _ _ _ Hok2 T2) as Hok2'.
      destruct (view_parts th1' th2' Hv') as (Hst' & Hct' & Hpd').
      constructor.
      * exact Hag'.
      * cbn [g_threads]. rewrite length_set_nth. exact Hc1.
      * cbn [g_threads]. rewrite length_set_nth. exact Hc2.
      * intros n thn1 thn2 Hn1 Hn2. cbn [g_threads] in Hn1, Hn2. destruct (Nat.eq_dec n k) as [-> | Hnk].
        -- rewrite (set_nth_at _ _ _ _ _ E1) in Hn1. rewrite (set_nth_at _ _ _ _ _ E2) in Hn2.
           injection Hn1 as <-. injection Hn2 as <-. exact Hv'.
        -- rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn1. rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn2. exact (Hview n thn1 thn2 Hn1 Hn2).
      * intros n thn1 thn2 Hn1 Hn2. cbn [g_threads] in Hn1, Hn2. destruct (Nat.eq_dec n k) as [-> | Hnk].
        -- rewrite (set_nth_at _ _ _ _ _ E1) in Hn1. rewrite (set_nth_at _ _ _ _ _ E2) in Hn2.
           injection Hn1 as <-. injection Hn2 as <-. exact Hlm'.
        -- rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn1. rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn2. exact (Hlocal n thn1 thn2 Hn1 Hn2).
      * intros n thn1 thn2 Hn1 Hn2 t Ht. cbn [g_threads] in Hn1, Hn2. destruct (Nat.eq_dec n k) as [-> | Hnk].
        -- rewrite (set_nth_at _ _ _ _ _ E1) in Hn1. rewrite (set_nth_at _ _ _ _ _ E2) in Hn2.
           injection Hn1 as <-. injection Hn2 as <-.
           destruct (Nat.le_gt_cases t (th_count th1)) as [Hle | Hgt].
           ++ rewrite (replay_before k _ _ th1 th1' _ _ Hok1 T1 t Hle), (replay_before k _ _ th2 th2' _ _ Hok2 T2 t ltac:(lia)).
              exact (Hrep k th1 th2 E1 E2 t Hle).
           ++ assert (t = th_count th1') by lia. subst t. unfold thread_ok in Hok1', Hok2'.
              rewrite Hok1', Hct', Hok2', Hst'. reflexivity.
        -- rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn1. rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn2. exact (Hrep n thn1 thn2 Hn1 Hn2 t Ht).
      * exact Hr1'.
      * exact Hr2'.
      * (* the shared memories of a block with a passed thread agree *)
        intros n thn1 Hn1 Hp. cbn [g_threads g_shared] in Hn1 |- *.
        destruct (Nat.eqb (cta n) (cta k)) eqn:Eb; cbv iota.
        -- apply Nat.eqb_eq in Eb. destruct (th_passed th1) eqn:Hp1.
           ++ (* the thread had passed: the memories agreed and still agree *)
              apply (sm_agree_uniform n k _ _ Eb). exact (Hpass_agree ltac:(first [exact Hp1 | reflexivity])).
           ++ (* the thread had not passed: it stores zero or nothing *)
              assert (Hbase : sm_agree (inits n) (g_shared G1 (cta k)) (g_shared G2 (cta k))).
              { destruct (Nat.eq_dec n k) as [-> | Hnk].
                - rewrite (set_nth_at _ _ _ _ _ E1) in Hn1. injection Hn1 as <-.
                  rewrite Hpass1 in Hp. try rewrite Hp1 in Hp. cbn [orb] in Hp.
                  rewrite Hp in Hwait. cbn [andb] in Hwait. apply negb_false_iff in Hwait.
                  destruct (existsb (fun j => match nth_error (g_threads G1) j with
                                              | Some th => Nat.eqb (cta j) (cta k) && th_passed th
                                              | None => false end) (seq 0 count)) eqn:Hex.
                  + (* a thread of the block has passed: the memories agree *)
                    apply existsb_exists in Hex. destruct Hex as (j & _ & Hj). cbv beta in Hj.
                    destruct (nth_error (g_threads G1) j) as [thj |] eqn:Ej; [| discriminate Hj].
                    apply andb_true_iff in Hj. destruct Hj as [Hcj Hpj]. apply Nat.eqb_eq in Hcj.
                    rewrite <- Hcj. apply (sm_agree_uniform k j _ _ (eq_sym Hcj)). exact (Hshm j thj Ej Hpj).
                  + (* no thread of the block has passed: every byte of its variables holds zero *)
                    assert (Hnp1 : ~ passed_block G1 (cta k)).
                    { intros (j & thj & Hj & Hcj & Hpj).
                      assert (Hjc : (j < count)%nat) by (rewrite <- Hc1; apply nth_error_Some; rewrite Hj; discriminate).
                      assert (Hin : existsb (fun j => match nth_error (g_threads G1) j with
                                                      | Some th => Nat.eqb (cta j) (cta k) && th_passed th
                                                      | None => false end) (seq 0 count) = true).
                      { apply existsb_exists. exists j. split; [apply in_seq; lia |]. cbv beta.
                        rewrite Hj, Hpj, (proj2 (Nat.eqb_eq _ _) Hcj). reflexivity. }
                      rewrite Hin in Hex. discriminate Hex. }
                    assert (Hnp2 : ~ passed_block G2 (cta k))
                      by (intros H2; exact (Hnp1 (passed_block_view G1 G2 (cta k) ltac:(lia) Hview H2))).
                    intros x Hx.
                    destruct (share_covers F Mf prog ps Hsh (Ev k th1) x Hx) as (T' & HT' & Hshare).
                    destruct (Hcomplete k T' Hk HT') as (m & Hm & Hcm & HTm).
                    pose proof (share_at_uniform k m T' x (eq_sym Hcm) Hshare) as Hshare_m.
                    assert (Hsm : share ps (inits m) x) by (unfold share; rewrite HTm; exact Hshare_m).
                    destruct (nth_error (g_threads G1) m) as [thm1 |] eqn:Em1; [| exfalso; apply nth_error_None in Em1; lia].
                    destruct (nth_error (g_threads G2) m) as [thm2 |] eqn:Em2; [| exfalso; apply nth_error_None in Em2; lia].
                    destruct (Hth1 m thm1 Em1) as [Hokm _].
                    assert (Hpm : th_passed thm1 = false).
                    { destruct (th_passed thm1) eqn:E; [| reflexivity]. exfalso. apply Hnp1. exists m, thm1. auto. }
                    assert (Hbm : at_bar bar (th_state thm1) = true).
                    { unfold block_arrived in Hwait. rewrite forallb_forall in Hwait.
                      specialize (Hwait m ltac:(apply in_seq; lia)). cbv beta in Hwait. rewrite Em1 in Hwait.
                      rewrite (proj2 (Nat.eqb_eq _ _) Hcm) in Hwait. unfold arrived in Hwait.
                      rewrite Hpm, orb_false_r in Hwait. exact Hwait. }
                    pose proof (at_bar_stored m thm1 Hokm (proj1 Hr1 m thm1 Em1) Hpm Hbm x Hsm) as Hz1.
                    pose proof (stored_zero_view m thm1 thm2 x (Hview m thm1 thm2 Em1 Em2) (Hrep m thm1 thm2 Em1 Em2) Hz1) as Hz2.
                    rewrite <- Hcm.
                    rewrite (proj2 Hr1 m thm1 Em1 ltac:(rewrite Hcm; exact Hnp1) x Hz1).
                    rewrite (proj2 Hr2 m thm2 Em2 ltac:(rewrite Hcm; exact Hnp2) x Hz2). reflexivity.
                - rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn1. rewrite <- Eb. exact (Hshm n thn1 Hn1 Hp). }
              destruct (Hpre ltac:(first [exact Hp1 | reflexivity])) as [[-> ->] | (x0 & w0 & -> & ->)];
                [exact Hbase | apply sm_write_agree; exact Hbase].
        -- apply Nat.eqb_neq in Eb.
           destruct (Nat.eq_dec n k) as [-> | Hnk]; [exfalso; exact (Eb eq_refl) |].
           rewrite (set_nth_other _ _ _ _ _ Hnk) in Hn1. exact (Hshm n thn1 Hn1 Hp).
    + (* the thread does not step in either run *)
      destruct (tstep F prog bar (inits k) (os k) (g_mem G2) (g_shared G2 (cta k)) th2) as [[[th2' m2'] sm2'] |] eqn:T2.
      * exfalso.
        destruct (tstep_agree k th2 th1 (g_mem G2) (g_mem G1) (g_shared G2 (cta k)) (g_shared G1 (cta k)) th2' m2' sm2'
                    Hok2 Hok1 Hh2 Hh1 HP2 HP1 (eq_sym Hv) (agree_sym _ _ Hmem) (proj1 Hr2 k th2 E2)
                    (fun Hp => sm_agree_sym _ _ _ (Hshm k th1 E1 (eq_trans Hpd Hp))) (lm_agree_sym _ _ _ Hlm) T2)
          as (th1' & m1' & sm1' & T1' & _).
        rewrite T1' in T1. discriminate T1.
      * assert (HG1 : gstep F prog bar inits os cta G1 k = G1)
          by (unfold gstep; rewrite E1; cbv beta iota; rewrite Hwait; cbv iota; rewrite T1; reflexivity).
        assert (HG2 : gstep F prog bar inits os cta G2 k = G2)
          by (unfold gstep; rewrite E2; cbv beta iota; rewrite <- Hst, <- Hba, Hwait; cbv iota; rewrite T2; reflexivity).
        rewrite HG1, HG2. constructor; assumption.
Qed.

Lemma run_inv0 : forall m0 sh0 lm0, run_inv (grid0 inits count m0 sh0 lm0).
Proof.
  intros m0 sh0 lm0. split.
  - intros n th H. rewrite (grid0_thread _ _ _ _ _ H). unfold passed_spec. cbn [th_passed th_count start].
    split; [discriminate |]. intros (t & s & Ht & _). lia.
  - intros n th H _ x (t & s & g & a & off & w & Ht & _). rewrite (grid0_thread _ _ _ _ _ H) in Ht.
    cbn [th_count start] in Ht. lia.
Qed.

Lemma ni_init : forall m0 m0' sh0 sh0' lm0 lm0', agree m0 m0' -> (forall n, lm_agree (inits n) (lm0 n) (lm0' n)) ->
    ni_inv (grid0 inits count m0 sh0 lm0) (grid0 inits count m0' sh0' lm0').
Proof.
  intros m0 m0' sh0 sh0' lm0 lm0' Hag Hlm.
  assert (Hlen : forall m sh lm, List.length (g_threads (grid0 inits count m sh lm)) = count)
    by (intros; cbn [g_threads grid0]; rewrite map_length, seq_length; reflexivity).
  constructor.
  - exact Hag.
  - apply Hlen.
  - apply Hlen.
  - intros n th1 th2 H1 H2. rewrite (grid0_thread _ _ _ _ _ H1), (grid0_thread _ _ _ _ _ H2). reflexivity.
  - intros n th1 th2 H1 H2. rewrite (grid0_thread _ _ _ _ _ H1), (grid0_thread _ _ _ _ _ H2). exact (Hlm n).
  - intros n th1 th2 H1 H2 t Ht. rewrite (grid0_thread _ _ _ _ _ H1) in Ht |- *. rewrite (grid0_thread _ _ _ _ _ H2).
    cbn [th_count start] in Ht. assert (t = 0)%nat by lia. subst t. reflexivity.
  - apply run_inv0.
  - apply run_inv0.
  - intros n th1 H1 Hp. rewrite (grid0_thread _ _ _ _ _ H1) in Hp. discriminate Hp.
Qed.

Lemma ni_fold : forall sched G1 G2, grid_inv F prog inits os P G1 -> grid_inv F prog inits os P G2 -> ni_inv G1 G2 ->
    ni_inv (fold_left (gstep F prog bar inits os cta) sched G1) (fold_left (gstep F prog bar inits os cta) sched G2).
Proof.
  induction sched as [| k sched IH]; intros G1 G2 Hg1 Hg2 Hni; [exact Hni |].
  cbn [fold_left]. apply IH.
  - exact (gstep_inv F prog bar Hsafe inits os cta P Hlink Hkeep G1 k Hg1).
  - exact (gstep_inv F prog bar Hsafe inits os cta P Hlink Hkeep G2 k Hg2).
  - exact (ni_step G1 G2 k Hg1 Hg2 Hni).
Qed.

(* Two runs of the launch with the same schedule, from global memories that
   agree on the bytes its loads may read, from initial local memories that
   agree on the bytes of each thread's local variables, and from any two
   initial shared memories, keep the invariant: in particular their global
   memories agree on those bytes, and each thread reaches the same state. *)
Theorem launch_ni : forall m0 m0' sh0 sh0' lm0 lm0' sched, P m0 -> P m0' -> agree m0 m0' ->
    (forall n, lm_agree (inits n) (lm0 n) (lm0' n)) ->
    ni_inv (launch F prog bar inits os cta count m0 sh0 lm0 sched) (launch F prog bar inits os cta count m0' sh0' lm0' sched).
Proof.
  intros m0 m0' sh0 sh0' lm0 lm0' sched H0 H0' Hag Hlm. unfold launch. apply ni_fold.
  - exact (grid0_inv F prog inits os P count m0 sh0 lm0 H0).
  - exact (grid0_inv F prog inits os P count m0' sh0' lm0' H0').
  - exact (ni_init m0 m0' sh0 sh0' lm0 lm0' Hag Hlm).
Qed.

End Prologue.

End Noninterference.
