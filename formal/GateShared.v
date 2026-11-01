(* GateShared.v: barrier participation of a block whose branches depend on
   values that the threads of the block share.

   The rule proves that every thread of a block executes the same sequence
   of waiting sites: CTA barriers and warp collectives (`same_waits`). A
   branch may depend on a value that a block reduction computes: a value
   that each thread reads from shared memory at an address that depends
   only on some bits of its thread index, combined by integer operations
   with the values that butterfly shuffles read from other lanes.

   The model gives each thread arbitrary values for shared memory loads and
   collective results. The rule assumes three facts of the PTX semantics
   about the threads of a block that have executed the same waiting sites
   (`cta_family`):
   - two threads that load the same shared address in the same barrier
     phase read the same value, when no thread of the block writes the
     loaded bytes in that phase and any two threads that write one of
     them in one earlier phase both make atomic read-modify-writes of one
     width or both store one value of one width (`shared_load_agrees`);
   - every lane of a warp gets the same result from a full-mask `redux`
     (`redux_agrees`);
   - a full-mask `shfl.sync.bfly` with lane mask m returns the source
     operand of lane l xor m to lane l (`bfly_reads_partner`).
   They are premises of the theorems, not axioms.

   The analysis keeps, for each register, the set of thread index bits on
   which its value depends (`ss_dep`): two threads whose indices agree on
   those bits hold the same value. A value that does not depend on any bit
   is the same in every thread of the block. It also keeps relations
   between registers (`sfact`): a register that holds the value of a
   register of the lane whose index differs by a mask, possibly through a
   fixed function; a register that holds bits of the thread index; a
   register that holds a base plus bits of the thread index; a register
   that holds a constant. The relations hold within a block; those of the
   thread index bits and the bases that the first block computes and no
   later block writes hold at the start of every later block, which the
   check verifies on every edge. A commutative integer operation
   of a value and the same value of the lane that differs in one bit does
   not depend on that bit. Floating-point operations are not modeled and
   give values that may depend on every bit.

   A conditional branch or exit whose predicate depends on no bit keeps the
   threads at the same position. A conditional branch at the end of a block
   whose predicate may depend on the thread must lead into a set of blocks
   without waiting sites or exits that control leaves only to one join
   block; the blocks may hold loops. Every thread's run ends (kernel_ends,
   which term_check proves), so the threads meet again at the start of the
   join block, where the registers that the blocks may write may differ
   between threads. Threads that load the same shared address read the same
   value when no instruction that may write shared memory can run in the
   barrier phase of the load (`phase_dirty`) and the program's shared
   writes are ordered (`writes_ordered_at`): every write is an atomic, a
   store of the number 0, or a store whose value depends only on the
   thread index bits that its address determines; writes of different
   positions that may run in one phase are atomics of one width or stores
   of 0 of one width; and a thread runs a store of the last kind at most
   once in one phase. If the writes are not ordered, every shared load
   gives a value that may differ in every thread. *)

From Coq Require Import List ZArith Bool Arith Lia PArith.
Require Import OpGate.GateIsa OpGate.GateForm OpGate.GateDomain OpGate.GateMeaning
  OpGate.GateThread OpGate.GateBarrier OpGate.GateLoop OpGate.GateTerm.
Import ListNotations.
Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Collectives.                                                        *)

(* A warp collective at a position of the program, as the lowering of its
   source instruction gives it: the register the instruction defines, its
   source register, the lane mask of a butterfly shuffle, and the register
   that holds its member mask if the mask is not a literal full mask. *)
Inductive collective :=
| CRedux (d src : reg) (mask : option reg)
| CBfly (d src : reg) (lane : Z) (mask : option reg).

Definition pc_eqb (a b : pc) : bool := Nat.eqb (fst a) (fst b) && Nat.eqb (snd a) (snd b).

Fixpoint collective_at (cols : list (pc * collective)) (p : pc) : option collective :=
  match cols with
  | [] => None
  | (q, c) :: rest => if pc_eqb q p then Some c else collective_at rest p
  end.

Definition full_mask : Z := 4294967295.

(* ------------------------------------------------------------------ *)
(* Barrier phases.                                                     *)

(* A CTA barrier is a position of the list cta. Phase starts are numbered
   as bits: bit 0 is the start of the kernel, bit k + 1 the k-th barrier of
   the list. *)
Definition is_cta (cta : list pc) (p : pc) : bool := existsb (pc_eqb p) cta.

Fixpoint cta_index (cta : list pc) (p : pc) : nat :=
  match cta with
  | [] => O
  | q :: rest => if pc_eqb q p then O else S (cta_index rest p)
  end.

Definition phase_bit (cta : list pc) (p : pc) : Z := 2 ^ Z.of_nat (S (cta_index cta p)).

Definition phase_next (cta : list pc) (p : pc) (s : Z) : Z := if is_cta cta p then phase_bit cta p else s.

(* An instruction that may write shared memory: a shared store or atomic,
   or a generic store or atomic, whose address may lie in shared memory. *)
Definition shared_write (i : instr) : bool :=
  match i with
  | IShA _ KLoad _ _ _ _ => false
  | IShA _ _ _ _ _ _ => true
  | IMem _ KLoad _ _ _ _ _ _ _ => false
  | IMem _ _ true _ _ _ _ _ _ => true
  | _ => false
  end.

(* The edges that leave position (b, i) of instruction ins to the start of
   a block, besides falling through within the block. *)
Definition jump_target (ins : instr) : option nat :=
  match ins with IBra _ tg => Some tg | _ => None end.

(* Whether control may fall through past the last instruction of a block. *)
Definition falls_through (body : list instr) : bool :=
  match last_instr body with
  | Some (IBra None _) | Some (IExit None) => false
  | _ => true
  end.

(* The phase starts at the positions of block b from s, with the edges the
   block's instructions take and the phase starts at its shared writes. *)
Fixpoint phase_walk (cta : list pc) (b i : nat) (s : Z) (body : list instr)
    (edges : list (nat * Z)) (dirty : Z) : Z * list (nat * Z) * Z :=
  match body with
  | [] => (s, edges, dirty)
  | ins :: rest =>
      let dirty' := if shared_write ins then Z.lor dirty s else dirty in
      let s' := phase_next cta (b, i) s in
      let edges' := match jump_target ins with Some tg => (tg, s') :: edges | None => edges end in
      phase_walk cta b (S i) s' rest edges' dirty'
  end.

Definition phase_block (cta : list pc) (prog : program) (P : list Z) (b : nat) : Z * list (nat * Z) * Z :=
  let body := block_body prog b in
  let '(s, edges, dirty) := phase_walk cta b 0 (nth b P 0) body [] 0 in
  (s, (if falls_through body then (S b, s) :: edges else edges), dirty).

Definition zsub (a b : Z) : bool := Z.eqb (Z.land a (Z.lnot b)) 0.

Definition phase_edges_ok (P : list Z) (edges : list (nat * Z)) : bool :=
  forallb (fun e => zsub (snd e) (nth (fst e) P 0)) edges.

Definition phase_verify (cta : list pc) (prog : program) (P : list Z) : bool :=
  Z.testbit (nth 0 P 0) 0 &&
  forallb (fun b => let '(_, edges, _) := phase_block cta prog P b in
                    phase_edges_ok P (filter (fun e => Nat.ltb (fst e) (length prog)) edges))
          (seq 0 (length prog)).

Definition phase_dirty (cta : list pc) (prog : program) (P : list Z) : Z :=
  fold_left (fun acc b => let '(_, _, d) := phase_block cta prog P b in Z.lor acc d) (seq 0 (length prog)) 0.

Fixpoint list_update {A} (l : list A) (k : nat) (f : A -> A) : list A :=
  match l, k with
  | [], _ => []
  | x :: t, O => f x :: t
  | x :: t, S k' => x :: list_update t k' f
  end.


(* The check looks the waiting sites up in maps of positions: the key of a
   position's block, and in its map the key of the position. A lookup costs
   two map searches rather than a comparison with every site, whose
   positions are unary numbers. The walks with the maps are equal to the
   walks with the lists (`phase_verify_at_same`, `dep_verify_at_same`). *)
Definition pc_table {A : Type} (entries : list (pc * A)) : PM.t (PM.t A) :=
  fold_right (fun e m =>
    let kb := Pos.of_succ_nat (fst (fst e)) in
    let inner := match PM.find kb m with Some t => t | None => PM.empty A end in
    PM.add kb (PM.add (Pos.of_succ_nat (snd (fst e))) (snd e) inner) m) (PM.empty (PM.t A)) entries.
Definition pc_find {A : Type} (t : PM.t (PM.t A)) (p : pc) : option A :=
  match PM.find (Pos.of_succ_nat (fst p)) t with
  | Some inner => PM.find (Pos.of_succ_nat (snd p)) inner
  | None => None
  end.

(* Each CTA barrier with its phase bit, numbered from k + 1. *)
Fixpoint cta_bits (cta : list pc) (k : nat) : list (pc * Z) :=
  match cta with
  | [] => []
  | p :: rest => (p, 2 ^ Z.of_nat (S k)) :: cta_bits rest (S k)
  end.

Definition phase_next_at (T : PM.t (PM.t Z)) (p : pc) (s : Z) : Z :=
  match pc_find T p with Some bit => bit | None => s end.

Fixpoint phase_walk_at (T : PM.t (PM.t Z)) (b i : nat) (s : Z) (body : list instr)
    (edges : list (nat * Z)) (dirty : Z) : Z * list (nat * Z) * Z :=
  match body with
  | [] => (s, edges, dirty)
  | ins :: rest =>
      let dirty' := if shared_write ins then Z.lor dirty s else dirty in
      let s' := phase_next_at T (b, i) s in
      let edges' := match jump_target ins with Some tg => (tg, s') :: edges | None => edges end in
      phase_walk_at T b (S i) s' rest edges' dirty'
  end.

Definition phase_block_at (T : PM.t (PM.t Z)) (prog : program) (P : list Z) (b : nat) : Z * list (nat * Z) * Z :=
  let body := block_body prog b in
  let '(s, edges, dirty) := phase_walk_at T b 0 (nth b P 0) body [] 0 in
  (s, (if falls_through body then (S b, s) :: edges else edges), dirty).

Definition phase_verify_at (T : PM.t (PM.t Z)) (prog : program) (P : list Z) : bool :=
  Z.testbit (nth 0 P 0) 0 &&
  forallb (fun b => let '(_, edges, _) := phase_block_at T prog P b in
                    phase_edges_ok P (filter (fun e => Nat.ltb (fst e) (length prog)) edges))
          (seq 0 (length prog)).

Definition phase_dirty_at (T : PM.t (PM.t Z)) (prog : program) (P : list Z) : Z :=
  fold_left (fun acc b => let '(_, _, d) := phase_block_at T prog P b in Z.lor acc d) (seq 0 (length prog)) 0.

(* The phase starts that the check verifies, proposed by iteration. *)
Definition phase_round_at (T : PM.t (PM.t Z)) (prog : program) (P : list Z) : list Z :=
  fold_left (fun acc b => let '(_, edges, _) := phase_block_at T prog acc b in
               fold_left (fun acc e => list_update acc (fst e) (Z.lor (snd e))) edges acc)
            (seq 0 (length prog)) P.

Fixpoint phase_iterate_at (fuel : nat) (T : PM.t (PM.t Z)) (prog : program) (P : list Z) : list Z :=
  match fuel with
  | O => P
  | S k => let P' := phase_round_at T prog P in
           if forallb (fun x => Z.eqb (fst x) (snd x)) (combine P P') then P
           else phase_iterate_at k T prog P'
  end.

Definition phase_propose_at (T : PM.t (PM.t Z)) (prog : program) : list Z :=
  phase_iterate_at 1000 T prog (1 :: repeat 0 (pred (length prog))).

Lemma pc_eqb_sym : forall p q, pc_eqb p q = pc_eqb q p.
Proof. intros [a b] [c d]. unfold pc_eqb. cbn. rewrite (Nat.eqb_sym a c), (Nat.eqb_sym b d). reflexivity. Qed.

Lemma pc_find_table : forall {A : Type} (entries : list (pc * A)) p,
  pc_find (pc_table entries) p =
  match find (fun e => pc_eqb p (fst e)) entries with Some e => Some (snd e) | None => None end.
Proof.
  intros A entries [b i]. induction entries as [|[[b' i'] x] rest IH].
  - unfold pc_find. cbn. rewrite PM.gempty. reflexivity.
  - unfold pc_find in *. cbn [pc_table fold_right fst snd find]. cbn [fst snd] in IH. fold (pc_table rest).
    unfold pc_eqb at 1. cbn [fst snd].
    destruct (Nat.eqb b b') eqn:Hb.
    + apply Nat.eqb_eq in Hb. subst b'. rewrite PM.gss. cbn [andb].
      destruct (Nat.eqb i i') eqn:Hi.
      * apply Nat.eqb_eq in Hi. subst i'. rewrite PM.gss. reflexivity.
      * rewrite PM.gso.
        -- rewrite <- IH. destruct (PM.find (Pos.of_succ_nat b) (pc_table rest)); [reflexivity|].
           rewrite PM.gempty. reflexivity.
        -- intros He. apply SuccNat2Pos.inj in He. subst i'. rewrite Nat.eqb_refl in Hi. discriminate.
    + rewrite PM.gso.
      * cbn [andb]. exact IH.
      * intros He. apply SuccNat2Pos.inj in He. subst b'. rewrite Nat.eqb_refl in Hb. discriminate.
Qed.

Lemma cta_bits_find : forall cta k p,
  match find (fun e => pc_eqb p (fst e)) (cta_bits cta k) with Some e => Some (snd e) | None => None end =
  if is_cta cta p then Some (2 ^ Z.of_nat (S (k + cta_index cta p))) else None.
Proof.
  intros cta. induction cta as [|q rest IH]; intros k p; [reflexivity|].
  cbn [cta_bits find fst snd]. unfold is_cta. cbn [existsb cta_index]. rewrite (pc_eqb_sym q p).
  destruct (pc_eqb p q); cbn [orb].
  - rewrite Nat.add_0_r. reflexivity.
  - rewrite IH. fold (is_cta rest p). destruct (is_cta rest p); [|reflexivity].
    rewrite Nat.add_succ_r. reflexivity.
Qed.

Lemma phase_next_at_same : forall cta p s,
  phase_next_at (pc_table (cta_bits cta 0)) p s = phase_next cta p s.
Proof.
  intros cta p s. unfold phase_next_at, phase_next, phase_bit.
  rewrite pc_find_table, cta_bits_find. destruct (is_cta cta p); reflexivity.
Qed.

Lemma phase_walk_at_same : forall cta body b i s edges dirty,
  phase_walk_at (pc_table (cta_bits cta 0)) b i s body edges dirty = phase_walk cta b i s body edges dirty.
Proof.
  intros cta body. induction body as [|ins rest IH]; intros b i s edges dirty; [reflexivity|].
  cbn [phase_walk_at phase_walk]. rewrite phase_next_at_same. apply IH.
Qed.

Lemma phase_block_at_same : forall cta prog P b,
  phase_block_at (pc_table (cta_bits cta 0)) prog P b = phase_block cta prog P b.
Proof. intros. unfold phase_block_at, phase_block. rewrite phase_walk_at_same. reflexivity. Qed.

Lemma phase_verify_at_same : forall cta prog P,
  phase_verify_at (pc_table (cta_bits cta 0)) prog P = phase_verify cta prog P.
Proof.
  intros cta prog P. unfold phase_verify_at, phase_verify. f_equal.
  induction (seq 0 (length prog)) as [|b bs IH]; [reflexivity|].
  cbn [forallb]. rewrite phase_block_at_same, IH. reflexivity.
Qed.

Lemma phase_dirty_at_same : forall cta prog P,
  phase_dirty_at (pc_table (cta_bits cta 0)) prog P = phase_dirty cta prog P.
Proof.
  intros cta prog P. unfold phase_dirty_at, phase_dirty. generalize 0.
  induction (seq 0 (length prog)) as [|b bs IH]; intros acc; [reflexivity|].
  cbn [fold_left]. rewrite phase_block_at_same. apply IH.
Qed.

(* ------------------------------------------------------------------ *)
(* The dependence analysis.                                            *)

(* Functions a relation applies to the related register's value. *)
Inductive sfun := GId | GLo | GHi | GHiShift | GNeg.

Definition gval (g : sfun) (w : nat) (v : Z) : Z :=
  match g with
  | GId => v
  | GLo => trunc 32 v
  | GHi => trunc 32 (v / 2 ^ 32)
  | GHiShift => trunc 64 (trunc 32 (v / 2 ^ 32) * 2 ^ 32)
  | GNeg => trunc w (- v)
  end.

Definition sfun_eqb (a b : sfun) : bool :=
  match a, b with
  | GId, GId | GLo, GLo | GHi, GHi | GHiShift, GHiShift | GNeg, GNeg => true
  | _, _ => false
  end.

(* SRel d m s g: in every thread, d holds g of the value of s in the thread
   whose index differs by xor m. SField d sh mask: d holds the thread index,
   shifted left by sh (right if negative), and masked. SConst d z: d holds z
   in every thread. SBase d u sh mask: d holds the sum of u and that field
   of the thread index, at d's width, as an address in a shared variable
   whose base u holds. *)
Inductive sfact :=
| SRel (d : reg) (m : Z) (s : reg) (g : sfun)
| SField (d : reg) (sh mask : Z)
| SConst (d : reg) (z : Z)
| SBase (d u : reg) (sh mask : Z).

Definition fact_dst (f : sfact) : reg :=
  match f with SRel d _ _ _ | SField d _ _ | SConst d _ | SBase d _ _ _ => d end.

Definition mentions (r : reg) (f : sfact) : bool :=
  match f with
  | SRel d _ s _ | SBase d s _ _ => Pos.eqb d r || Pos.eqb s r
  | SField d _ _ | SConst d _ => Pos.eqb d r
  end.

(* The dependence of each register: absent means that the value may differ
   in any two threads. *)
Record sstate := { ss_dep : PM.t Z; ss_facts : list sfact }.

Definition dep_of (st : sstate) (r : reg) : option Z := PM.find r (ss_dep st).

Definition dunion (a b : option Z) : option Z :=
  match a, b with Some x, Some y => Some (Z.lor x y) | _, _ => None end.

Definition dep_regs (st : sstate) (rs : list reg) : option Z :=
  fold_left (fun acc r => dunion acc (dep_of st r)) rs (Some 0).

(* New facts about d may not refer to d's previous value. *)
Definition fresh_fact (d : reg) (f : sfact) : bool :=
  Pos.eqb (fact_dst f) d && match f with SRel _ _ s _ | SBase _ s _ _ => negb (Pos.eqb s d) | _ => true end.

(* The facts kept: every field of the thread index and every address of a
   base plus a field, which addresses use far from where they are
   computed, and the latest relations and constants, since the relations
   of a reduction are used within a few instructions of the shuffles that
   make them. *)
Definition fact_limit : nat := 64.

Definition is_field (f : sfact) : bool := match f with SField _ _ _ | SBase _ _ _ _ => true | _ => false end.

Definition ss_write (st : sstate) (d : reg) (dep : option Z) (fs : list sfact) : sstate :=
  {| ss_dep := match dep with Some z => PM.add d z (ss_dep st) | None => PM.remove d (ss_dep st) end;
     ss_facts := let kept := filter (fun f => negb (mentions d f)) (ss_facts st) in
                 filter (fresh_fact d) fs ++ filter is_field kept ++
                 firstn fact_limit (filter (fun f => negb (is_field f)) kept) |}.

Definition ss_top (st : sstate) (rs : list reg) : sstate :=
  fold_left (fun st r => ss_write st r None []) rs st.

Definition sfact_eqb (f g : sfact) : bool :=
  match f, g with
  | SRel d m s h, SRel d' m' s' h' =>
      Pos.eqb d d' && Z.eqb m m' && Pos.eqb s s' &&
      match h, h' with
      | GId, GId | GLo, GLo | GHi, GHi | GHiShift, GHiShift | GNeg, GNeg => true
      | _, _ => false
      end
  | SField d sh mk, SField d' sh' mk' => Pos.eqb d d' && Z.eqb sh sh' && Z.eqb mk mk'
  | SConst d z, SConst d' z' => Pos.eqb d d' && Z.eqb z z'
  | SBase d u sh mk, SBase d' u' sh' mk' => Pos.eqb d d' && Pos.eqb u u' && Z.eqb sh sh' && Z.eqb mk mk'
  | _, _ => false
  end.

(* Every fact of G is a fact of fs. *)
Definition facts_incl (G fs : list sfact) : bool := forallb (fun g => existsb (sfact_eqb g) fs) G.

(* The state at the start of block b with the dependences e: no fact at
   the start of the kernel, and the facts G at the start of every other
   block, which the check verifies on every edge into it. *)
Definition entry_state (G : list sfact) (b : nat) (e : PM.t Z) : sstate :=
  {| ss_dep := e; ss_facts := if Nat.eqb b 0 then [] else G |}.

Section Analysis.

Variable F : lfacts.
Variable tidr : reg.
Variable cols : list (pc * collective).
(* The facts at the start of every block but the first. *)
Variable G : list sfact.

Definition field_of (st : sstate) (r : reg) : option (Z * Z) :=
  if Pos.eqb r tidr then Some (0, Z.ones (wd F r)) else
  fold_right (fun f acc => match f with SField d sh mk => if Pos.eqb d r then Some (sh, mk) else acc | _ => acc end)
             None (ss_facts st).

Definition const_of (st : sstate) (r : reg) : option Z :=
  fold_right (fun f acc => match f with SConst d z => if Pos.eqb d r then Some z else acc | _ => acc end)
             None (ss_facts st).

Definition rels_of (st : sstate) (r : reg) : list (Z * reg * sfun) :=
  fold_right (fun f acc => match f with SRel d m s g => if Pos.eqb d r then (m, s, g) :: acc else acc | _ => acc end)
             [] (ss_facts st).

(* The base and the field of the thread index whose sum r holds. *)
Definition base_of (st : sstate) (r : reg) : option (reg * Z * Z) :=
  fold_right (fun f acc => match f with SBase d u sh mk => if Pos.eqb d r then Some (u, sh, mk) else acc | _ => acc end)
             None (ss_facts st).

Definition has_rel (st : sstate) (d : reg) (m : Z) (s : reg) (g : sfun) : bool :=
  existsb (fun x => let '(m', s', g') := x in Z.eqb m' m && Pos.eqb s' s && sfun_eqb g' g) (rels_of st d).

Definition single_lane (m : Z) : bool := existsb (Z.eqb m) [1; 2; 4; 8; 16].

(* The lane masks m under which op(a, b) does not change, for a commutative
   op: b holds a of the lane xor m, or a holds b of the lane xor m. *)
Definition swap_masks (st : sstate) (a b : reg) : Z :=
  fold_left (fun acc x => let '(m, s, g) := x in
               if sfun_eqb g GId && single_lane m && Pos.eqb s a then Z.lor acc m else acc)
            (rels_of st b)
  (fold_left (fun acc x => let '(m, s, g) := x in
                if sfun_eqb g GId && single_lane m && Pos.eqb s b then Z.lor acc m else acc)
             (rels_of st a) 0).

(* The lane masks m for which an equality of a and b does not change: b
   holds the negation of a of the lane xor m, or a of b. *)
Definition neg_masks (st : sstate) (a b : reg) : Z :=
  fold_left (fun acc x => let '(m, s, g) := x in
               if sfun_eqb g GNeg && single_lane m && Pos.eqb s a then Z.lor acc m else acc)
            (rels_of st b)
  (fold_left (fun acc x => let '(m, s, g) := x in
                if sfun_eqb g GNeg && single_lane m && Pos.eqb s b then Z.lor acc m else acc)
             (rels_of st a) 0).

Definition reduce (dep : option Z) (masks : Z) : option Z :=
  option_map (fun z => Z.land z (Z.lnot masks)) dep.

Definition commutative (o : binop) : bool :=
  match o with BAdd | BMul | BAnd | BOr | BXor => true | _ => false end.

(* The thread index bits that a field of the index depends on. *)
Definition field_dep (f : Z * Z) : Z := Z.shiftr (snd f) (fst f).

Definition ones (w : nat) : Z := Z.ones (Z.of_nat w).

(* The field of the index that d holds after instruction i, if any. *)
Definition field_step (st : sstate) (i : instr) : option (Z * Z) :=
  match i with
  | IMov None d a => option_map (fun f => (fst f, Z.land (snd f) (ones (wdn F d)))) (field_of st a)
  | IBin None BAnd w d a (OImm c) =>
      option_map (fun f => (fst f, Z.land (Z.land (Z.land (snd f) (ones w)) (trunc w c)) (ones (wdn F d))))
                 (field_of st a)
  | IBin None BShr w d a (OImm c) =>
      let k := shamt w c in
      option_map (fun f => (fst f - k, Z.land (Z.shiftr (Z.land (snd f) (ones w)) k) (ones (wdn F d))))
                 (field_of st a)
  | IBin None BShl w d a (OImm c) =>
      let k := shamt w c in
      option_map (fun f => (fst f + k, Z.land (Z.shiftl (snd f) k) (ones (wdn F d)))) (field_of st a)
  | ICvt None _ from false d a =>
      option_map (fun f => (fst f, Z.land (Z.land (snd f) (ones from)) (ones (wdn F d)))) (field_of st a)
  | _ => None
  end.

(* The relations that d gets from instruction i, which writes d. *)
Definition rel_step (st : sstate) (i : instr) : list sfact :=
  match i with
  | IMov None d a =>
      (if Nat.eqb (wdn F d) (wdn F a)
       then SRel d 0 a GId :: map (fun x => let '(m, s, g) := x in SRel d m s g) (rels_of st a)
       else []) ++
      match const_of st a with Some z => [SConst d (trunc (wdn F d) z)] | None => [] end
  | IImm None d z => [SConst d (trunc (wdn F d) z)]
  | ICvt None _ from false d a =>
      if Nat.eqb from 64 && Nat.eqb (wdn F d) 32 && Nat.eqb (wdn F a) 64 then
        SRel d 0 a GLo ::
        flat_map (fun x => let '(m, s, g) := x in
                    if sfun_eqb g GId then [SRel d m s GLo]
                    else if sfun_eqb g GHi then [SRel d m s GHi] else []) (rels_of st a)
      else if Nat.eqb from 32 && Nat.leb 32 (wdn F d) && Nat.eqb (wdn F a) 32 then
        flat_map (fun x => let '(m, s, g) := x in
                    if sfun_eqb g GLo || sfun_eqb g GHi then [SRel d m s g] else []) (rels_of st a)
      else []
  | IBfe None w d a p ln =>
      if Nat.eqb w 64 && Nat.eqb p 32 && Nat.eqb ln 32 && Nat.leb 32 (wdn F d) && Nat.eqb (wdn F a) 64 then
        SRel d 0 a GHi ::
        flat_map (fun x => let '(m, s, g) := x in if sfun_eqb g GId then [SRel d m s GHi] else [])
                 (rels_of st a)
      else []
  | IBin None BShl w d a (OImm c) =>
      if Nat.eqb w 64 && Z.eqb c 32 && Nat.eqb (wdn F d) 64 then
        flat_map (fun x => let '(m, s, g) := x in if sfun_eqb g GHi then [SRel d m s GHiShift] else [])
                 (rels_of st a)
      else []
  | IBin None BOr w d a (OReg b) =>
      if Nat.eqb w 64 && Nat.eqb (wdn F d) 64 then
        flat_map (fun x => let '(m, s, g) := x in
                    if Nat.eqb (wdn F s) 64 &&
                       ((sfun_eqb g GLo && has_rel st b m s GHiShift) ||
                        (sfun_eqb g GHiShift && has_rel st b m s GLo))
                    then [SRel d m s GId] else []) (rels_of st a)
      else []
  | IBin None BAdd _ d a (OReg b) =>
      (* a base plus a field of the thread index, in either order *)
      match field_of st b with Some (sh, mk) => [SBase d a sh mk] | None => [] end ++
      match field_of st a with Some (sh, mk) => [SBase d b sh mk] | None => [] end
  | IMad None w d a (OImm k) (OImm c) =>
      if Z.eqb k (-1) && Z.eqb c 0 then
        SRel d 0 a GNeg ::
        flat_map (fun x => let '(m, s, g) := x in if sfun_eqb g GId then [SRel d m s GNeg] else [])
                 (rels_of st a)
      else []
  | _ => []
  end.

(* An instruction whose result is a function of the registers it reads. *)
Definition pure_exact (i : instr) : bool :=
  match i with
  | IMov _ _ _ | IImm _ _ _ | IBin _ _ _ _ _ _ | IMad _ _ _ _ _ _ | IWide _ _ _ _ _ _
  | ICvt _ _ _ _ _ _ | IBfe _ _ _ _ _ _ | ISetp _ _ _ _ _ _ _ | IAndp _ _ _ _ | IOrp _ _ _ _
  | IXorp _ _ _ _ | IMinMax _ _ _ _ _ _ _ | ISelp _ _ _ _ _ _ => true
  | _ => false
  end.

(* The dependence of the value that an exact instruction writes to d. *)
Definition pure_dep (st : sstate) (i : instr) (d : reg) : option Z :=
  let base := dep_regs st (instruction_inputs i) in
  let base := match guard_of i with Some _ => dunion base (dep_of st d) | None => base end in
  match field_step st i with
  | Some f => Some (field_dep f)
  | None =>
      match i with
      | IBin None o _ _ a (OReg b) => if commutative o then reduce base (swap_masks st a b) else base
      | IMinMax None _ _ _ _ a (OReg b) => reduce base (swap_masks st a b)
      | ISetp None c w _ _ a (OReg b) =>
          match c with
          | CEq | CNe => if Nat.eqb (wdn F a) w && Nat.eqb (wdn F b) w
                         then reduce base (neg_masks st a b) else base
          | _ => base
          end
      | _ => base
      end
  end.

Definition mask_ok (st : sstate) (mask : option reg) : bool :=
  match mask with
  | None => true
  | Some r => match const_of st r with Some z => Z.eqb z full_mask | None => false end
  end.

(* The cells of local memory, which local stores and frames may write. *)
Definition cell_regs : list reg := map (fun k => let '(p, _, _) := k in p) (lf_cells F).

Definition writes_cells (i : instr) : bool :=
  match i with IMem _ _ _ _ _ _ _ _ _ | IStL _ _ _ _ _ _ | IFrame _ => true | _ => false end.

(* The state after instruction i at position p, whose phase starts are s;
   dirty holds the phase starts in which a thread may write shared
   memory. *)
Definition ss_transfer (dirty : Z) (p : pc) (s : Z) (st : sstate) (i : instr) : sstate :=
  if pure_exact i then
    match dest i with
    | Some d =>
        let fs := match guard_of i with
                  | None => match field_step st i with
                            | Some f => SField d (fst f) (snd f) :: rel_step st i
                            | None => rel_step st i
                            end
                  | Some _ => []
                  end in
        ss_write st d (pure_dep st i d) fs
    | None => st
    end
  else
  match i with
  | IDef None [d] =>
      match collective_at cols p with
      | Some (CBfly d' src m mask) =>
          if Pos.eqb d' d && negb (Pos.eqb src d) && (0 <=? m) && (m <? 32) &&
             Nat.eqb (wdn F d) 32 && Nat.eqb (wdn F src) 32 && mask_ok st mask
          then ss_write st d (dep_of st src)
                 (SRel d m src GId ::
                  map (fun x => let '(m0, q, g) := x in SRel d (Z.lxor m m0) q g) (rels_of st src) ++
                  match const_of st src with Some z => [SConst d (trunc 32 z)] | None => [] end)
          else ss_write st d None []
      | Some (CRedux d' src mask) =>
          if Pos.eqb d' d && Nat.eqb (wdn F d) 32 && mask_ok st mask
          then ss_write st d (Some (Z.lnot 31)) []
          else ss_write st d None []
      | None => ss_write st d None []
      end
  | ILdS None bits d a off =>
      if Z.eqb (Z.land s dirty) 0 then ss_write st d (dep_of st a) [] else ss_write st d None []
  | _ => ss_top st (loop_writes i ++ (if writes_cells i then cell_regs else []))
  end.

Definition uniform_guard (st : sstate) (g : guard) : bool :=
  match g with
  | None => true
  | Some (_, p) => match dep_of st p with Some z => Z.eqb z 0 | None => false end
  end.

Definition branch_ok (st : sstate) (i : instr) : bool :=
  match i with IBra g _ | IExit g => uniform_guard st g | _ => true end.

(* The walk of block b: the state at its end, the edges it takes to the
   start of a block with the dependences there, or None if a branch's
   predicate may differ between threads. *)
Fixpoint ss_walk (cta : list pc) (dirty : Z) (b i : nat) (s : Z) (st : sstate) (body : list instr)
    (edges : list (nat * sstate)) : option (sstate * list (nat * sstate)) :=
  match body with
  | [] => Some (st, edges)
  | ins :: rest =>
      if negb (branch_ok st ins) then None else
      let st' := ss_transfer dirty (b, i) s st ins in
      let edges' := match jump_target ins with Some tg => (tg, st') :: edges | None => edges end in
      ss_walk cta dirty b (S i) (phase_next cta (b, i) s) st' rest edges'
  end.

(* The blocks that control may enter from block c: the next block, when the
   last instruction of c may fall through, and the targets of its jumps. *)
Definition block_succs (prog : program) (c : nat) : list nat :=
  (if falls_through (block_body prog c) then [S c] else []) ++
  flat_map (fun i => match jump_target i with Some tg => [tg] | None => [] end) (block_body prog c).

Definition mem_nat (d : nat) (R : list nat) : bool := existsb (Nat.eqb d) R.

(* The blocks that a thread may pass after the branch at the end of block b
   before it reaches J: the blocks reachable from the successors of b
   without passing J, found from a list of blocks whose successors are not
   yet added. The search proposes them; region_ok checks them. *)
Definition region_add (J : nat) (wR : list nat * list nat) (d : nat) : list nat * list nat :=
  let '(work, R) := wR in if Nat.eqb d J || mem_nat d R then (work, R) else (d :: work, d :: R).

Fixpoint region_grow (prog : program) (J fuel : nat) (work R : list nat) : list nat :=
  match fuel, work with
  | S f, c :: rest =>
      let '(work', R') := fold_left (region_add J) (block_succs prog c) (rest, R) in
      region_grow prog J f work' R'
  | _, _ => R
  end.

Definition region_blocks (prog : program) (b J : nat) : list nat :=
  let '(work, R) := fold_left (region_add J) (block_succs prog b) ([], []) in
  region_grow prog J (S (length prog)) work R.

(* An instruction in such a block: not a waiting site, a CTA barrier, or an
   exit. *)
Definition region_instr_ok (cta events : list pc) (c k : nat) (i : instr) : bool :=
  negb (is_cta cta (c, k)) && negb (is_cta events (c, k)) &&
  match i with IExit _ => false | _ => true end.

(* The blocks R hold the successors of b other than J, each has an
   instruction, and control leaves them only to J. A thread may pass a block
   of R more than once: it reaches J because its run ends (kernel_ends). *)
Definition region_ok (cta events : list pc) (prog : program) (b J : nat) (R : list nat) : bool :=
  forallb (fun d => Nat.eqb d J || mem_nat d R) (block_succs prog b) &&
  forallb (fun c => negb (Nat.eqb (block_len prog c) 0) &&
    forallb (fun k => match fetch prog c k with Some i => region_instr_ok cta events c k i | None => false end)
            (seq 0 (block_len prog c)) &&
    forallb (fun d => Nat.eqb d J || mem_nat d R) (block_succs prog c)) R.

(* The registers that the instructions of the blocks may write. *)
Definition region_writes (prog : program) (R : list nat) : list reg :=
  flat_map (fun c => flat_map (fun i => loop_writes i ++ (if writes_cells i then cell_regs else []))
                              (block_body prog c)) R.

(* The block at which the threads of a branch at the end of block b to tg
   meet again: the first J from tg such that the jumps of the blocks
   between b and J lead at most to J. *)
Fixpoint region_join (prog : program) (b fuel J : nat) : nat :=
  match fuel with
  | O => J
  | S f =>
      let J' := fold_left Nat.max
                  (flat_map (fun c => flat_map (fun i => match jump_target i with Some tg => [tg] | None => [] end)
                                               (block_body prog c)) (seq (S b) (J - S b))) J in
      if Nat.eqb J' J then J else region_join prog b f J'
  end.

(* A conditional branch at the end of block b whose predicate may differ
   between threads: the threads that branch and those that do not pass
   blocks without waiting sites and meet at the join block, where every
   register that the blocks may write may differ between threads. *)
Definition diverge_block (cta events : list pc) (prog : program) (P : list Z) (dirty : Z) (b : nat) (e : PM.t Z)
    : option (list (nat * sstate)) :=
  let body := block_body prog b in
  match last_instr body with
  | Some (IBra (Some _) tg) =>
      let J := region_join prog b (length prog) tg in
      let R := region_blocks prog b J in
      if region_ok cta events prog b J R &&
         negb (is_cta cta (b, pred (length body))) && negb (is_cta events (b, pred (length body))) then
        match ss_walk cta dirty b 0 (nth b P 0) (entry_state G b e) (removelast body) [] with
        | Some (st, edges) => Some ((J, ss_top st (region_writes prog R)) :: edges)
        | None => None
        end
      else None
  | _ => None
  end.

Definition dep_block (cta events : list pc) (prog : program) (P : list Z) (dirty : Z) (D : list (option (PM.t Z)))
    (b : nat) : option (list (nat * sstate)) :=
  match nth b D None with
  | None => Some []
  | Some e =>
      let body := block_body prog b in
      match ss_walk cta dirty b 0 (nth b P 0) (entry_state G b e) body [] with
      | Some (st, edges) => Some (if falls_through body then (S b, st) :: edges else edges)
      | None => diverge_block cta events prog P dirty b e
      end
  end.

(* m1 is at least as precise as m2: every register that m2 constrains m1
   constrains with fewer bits. *)
Definition dep_le (m1 m2 : PM.t Z) : bool :=
  PM.fold (fun r z acc => acc && match PM.find r m1 with Some z1 => zsub z1 z | None => false end) m2 true.

Definition dep_join (m1 m2 : PM.t Z) : PM.t Z :=
  PM.fold (fun r z acc => match PM.find r m2 with Some z2 => PM.add r (Z.lor z z2) acc | None => acc end)
          m1 (PM.empty Z).

Definition dep_initial (uniform : reg -> bool) : PM.t Z :=
  fold_left (fun m x => if uniform (snd x) then PM.add (snd x) 0 m else m) (lf_names F) (PM.empty Z).

Definition dep_verify (uniform : reg -> bool) (cta events : list pc) (prog : program) (P : list Z) (dirty : Z)
    (D : list (option (PM.t Z))) : bool :=
  match nth 0 D None with
  | Some e0 => dep_le (dep_initial uniform) e0
  | None => false
  end &&
  forallb (fun b => match dep_block cta events prog P dirty D b with
                    | None => false
                    | Some edges =>
                        forallb (fun e => match nth (fst e) D None with
                                          | Some m => dep_le (ss_dep (snd e)) m && (Nat.eqb (fst e) 0 || facts_incl G (ss_facts (snd e)))
                                          | None => false
                                          end)
                                (filter (fun e => Nat.ltb (fst e) (length prog)) edges)
                    end) (seq 0 (length prog)).

Definition dep_merge (x : option (PM.t Z)) (m : PM.t Z) : option (PM.t Z) :=
  match x with None => Some m | Some m0 => Some (dep_join m0 m) end.

Definition dep_same (x y : option (PM.t Z)) : bool :=
  match x, y with
  | Some a, Some b => dep_le a b && dep_le b a
  | None, None => true
  | _, _ => false
  end.

(* The dependence walks with the waiting sites in maps: T the phase bit of
   each CTA barrier, C the collectives. The region test of a divergent
   branch reads the lists, as diverge_block does. *)
Definition ss_transfer_at (C : PM.t (PM.t collective)) (dirty : Z) (p : pc) (s : Z) (st : sstate) (i : instr)
    : sstate :=
  if pure_exact i then
    match dest i with
    | Some d =>
        let fs := match guard_of i with
                  | None => match field_step st i with
                            | Some f => SField d (fst f) (snd f) :: rel_step st i
                            | None => rel_step st i
                            end
                  | Some _ => []
                  end in
        ss_write st d (pure_dep st i d) fs
    | None => st
    end
  else
  match i with
  | IDef None [d] =>
      match pc_find C p with
      | Some (CBfly d' src m mask) =>
          if Pos.eqb d' d && negb (Pos.eqb src d) && (0 <=? m) && (m <? 32) &&
             Nat.eqb (wdn F d) 32 && Nat.eqb (wdn F src) 32 && mask_ok st mask
          then ss_write st d (dep_of st src)
                 (SRel d m src GId ::
                  map (fun x => let '(m0, q, g) := x in SRel d (Z.lxor m m0) q g) (rels_of st src) ++
                  match const_of st src with Some z => [SConst d (trunc 32 z)] | None => [] end)
          else ss_write st d None []
      | Some (CRedux d' src mask) =>
          if Pos.eqb d' d && Nat.eqb (wdn F d) 32 && mask_ok st mask
          then ss_write st d (Some (Z.lnot 31)) []
          else ss_write st d None []
      | None => ss_write st d None []
      end
  | ILdS None bits d a off =>
      if Z.eqb (Z.land s dirty) 0 then ss_write st d (dep_of st a) [] else ss_write st d None []
  | _ => ss_top st (loop_writes i ++ (if writes_cells i then cell_regs else []))
  end.

Fixpoint ss_walk_at (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective)) (dirty : Z) (b i : nat) (s : Z)
    (st : sstate) (body : list instr) (edges : list (nat * sstate)) : option (sstate * list (nat * sstate)) :=
  match body with
  | [] => Some (st, edges)
  | ins :: rest =>
      if negb (branch_ok st ins) then None else
      let st' := ss_transfer_at C dirty (b, i) s st ins in
      let edges' := match jump_target ins with Some tg => (tg, st') :: edges | None => edges end in
      ss_walk_at T C dirty b (S i) (phase_next_at T (b, i) s) st' rest edges'
  end.

Definition diverge_block_at (cta events : list pc) (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective))
    (prog : program) (P : list Z) (dirty : Z) (b : nat) (e : PM.t Z) : option (list (nat * sstate)) :=
  let body := block_body prog b in
  match last_instr body with
  | Some (IBra (Some _) tg) =>
      let J := region_join prog b (length prog) tg in
      let R := region_blocks prog b J in
      if region_ok cta events prog b J R &&
         negb (is_cta cta (b, pred (length body))) && negb (is_cta events (b, pred (length body))) then
        match ss_walk_at T C dirty b 0 (nth b P 0) (entry_state G b e) (removelast body) [] with
        | Some (st, edges) => Some ((J, ss_top st (region_writes prog R)) :: edges)
        | None => None
        end
      else None
  | _ => None
  end.

Definition dep_block_at (cta events : list pc) (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective))
    (prog : program) (P : list Z) (dirty : Z) (D : list (option (PM.t Z))) (b : nat)
    : option (list (nat * sstate)) :=
  match nth b D None with
  | None => Some []
  | Some e =>
      let body := block_body prog b in
      match ss_walk_at T C dirty b 0 (nth b P 0) (entry_state G b e) body [] with
      | Some (st, edges) => Some (if falls_through body then (S b, st) :: edges else edges)
      | None => diverge_block_at cta events T C prog P dirty b e
      end
  end.

Definition dep_verify_at (uniform : reg -> bool) (cta events : list pc) (T : PM.t (PM.t Z))
    (C : PM.t (PM.t collective)) (prog : program) (P : list Z) (dirty : Z) (D : list (option (PM.t Z))) : bool :=
  match nth 0 D None with
  | Some e0 => dep_le (dep_initial uniform) e0
  | None => false
  end &&
  forallb (fun b => match dep_block_at cta events T C prog P dirty D b with
                    | None => false
                    | Some edges =>
                        forallb (fun e => match nth (fst e) D None with
                                          | Some m => dep_le (ss_dep (snd e)) m && (Nat.eqb (fst e) 0 || facts_incl G (ss_facts (snd e)))
                                          | None => false
                                          end)
                                (filter (fun e => Nat.ltb (fst e) (length prog)) edges)
                    end) (seq 0 (length prog)).

Definition dep_round_at (cta events : list pc) (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective))
    (prog : program) (P : list Z) (dirty : Z) (D : list (option (PM.t Z))) : option (list (option (PM.t Z))) :=
  fold_left (fun acc b => match acc with
               | None => None
               | Some acc =>
               match dep_block_at cta events T C prog P dirty acc b with
               | Some edges =>
                   Some (fold_left (fun acc e => list_update acc (fst e) (fun x => dep_merge x (ss_dep (snd e)))) edges acc)
               | None => None
               end end) (seq 0 (length prog)) (Some D).

Fixpoint dep_iterate_at (fuel : nat) (cta events : list pc) (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective))
    (prog : program) (P : list Z) (dirty : Z) (D : list (option (PM.t Z))) : list (option (PM.t Z)) :=
  match fuel with
  | O => D
  | S k => match dep_round_at cta events T C prog P dirty D with
           | Some D' => if forallb (fun x => dep_same (fst x) (snd x)) (combine D D') then D
                        else dep_iterate_at k cta events T C prog P dirty D'
           | None => D
           end
  end.

Definition dep_propose_at (uniform : reg -> bool) (cta events : list pc) (T : PM.t (PM.t Z))
    (C : PM.t (PM.t collective)) (prog : program) (P : list Z) (dirty : Z) : list (option (PM.t Z)) :=
  dep_iterate_at 1000 cta events T C prog P dirty (Some (dep_initial uniform) :: repeat None (pred (length prog))).

(* The facts that the check proposes at the start of the blocks after the
   first: the fields of the thread index and the addresses of a base plus
   a field at the end of the first block, of registers that no later block
   writes. The proposal is untrusted: dep_verify checks the facts on every
   edge. *)
Definition entry_facts_at (uniform : reg -> bool) (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective))
    (prog : program) (P : list Z) (dirty : Z) : list sfact :=
  let '(st, _, _) :=
    fold_left (fun acc ins => let '(st, j, s) := acc in
                 (ss_transfer_at C dirty (0%nat, j) s st ins, S j, phase_next_at T (0%nat, j) s))
              (block_body prog 0) ({| ss_dep := dep_initial uniform; ss_facts := [] |}, 0%nat, nth 0 P 0) in
  let later := flat_map (fun b => flat_map (fun i => loop_writes i ++ (if writes_cells i then cell_regs else []))
                                           (block_body prog b)) (seq 1 (pred (length prog))) in
  filter (fun f => is_field f && forallb (fun r => negb (mentions r f)) later) (ss_facts st).

(* ------------------------------------------------------------------ *)
(* The shared writes that the route orders.                            *)

(* The threads of the block, 0 to n - 1, agree on every thread index bit
   from log2 n up: index_bits holds the bits below. *)
Variable n : nat.

Definition index_bits : Z := Z.ones (Z.log2_up (Z.of_nat n)).

(* The class of an instruction that may write shared memory. WAtomic w: an
   atomic read-modify-write of w bytes. WZero w: a store of the number 0
   in w bytes. WStore w: a store of w bytes whose value is the same in any
   two threads that store it at one address in one step, since the
   address determines the thread index bits on which the value depends.
   WOther: any other write, a generic store or atomic, a vector store, or
   a store whose value may differ between such threads. WNone: no write. *)
Inductive wclass := WNone | WAtomic (w : Z) | WZero (w : Z) | WStore (w : Z) | WOther.

(* A store of w bytes of o stores the number 0: an immediate, or, with the
   state st at the store, a register that holds a constant. *)
Definition zero_store (st : option sstate) (w : Z) (o : opnd) : bool :=
  match o with
  | OImm z => Z.eqb (trunc (Z.to_nat (8 * w)) z) 0
  | OReg v =>
      match st with
      | Some st => match const_of st v with Some z => Z.eqb (trunc (Z.to_nat (8 * w)) z) 0 | None => false end
      | None => false
      end
  end.

(* A store of w bytes of o at the address in a: any two threads that store
   at one address store one value. The address is the same in every
   thread and the value as well; or the address is a base that is the same
   in every thread plus a field of the thread index whose values are
   multiples of w below 2^32, so two threads that store at overlapping
   addresses agree on the field's bits, on which the value depends below
   log2 n. *)
Definition store_orderly (st : sstate) (a : reg) (w : Z) (o : opnd) : bool :=
  match match o with OImm _ => Some 0 | OReg v => dep_of st v end with
  | None => false
  | Some dv =>
      match dep_of st a with Some da => Z.eqb da 0 && Z.eqb (Z.land dv index_bits) 0 | None => false end ||
      match base_of st a with
      | Some (u, sh, mk) =>
          match dep_of st u with
          | Some du =>
              Z.eqb du 0 && Nat.leb 32 (wdn F a) && (0 <=? mk) && (mk <? 2 ^ 32) &&
              existsb (Z.eqb w) [1; 2; 4; 8] && Z.eqb (Z.land mk (w - 1)) 0 &&
              zsub (Z.land dv index_bits) (field_dep (sh, mk))
          | None => false
          end
      | None => false
      end
  end.

(* The class of instruction i, with the state st at it if the analysis
   holds there at every execution. *)
Definition write_class (st : option sstate) (i : instr) : wclass :=
  match i with
  | IShA _ KLoad _ _ _ _ => WNone
  | IShA _ KRmw _ _ w _ => WAtomic w
  | IShA _ KStore a _ w (Some o) =>
      if zero_store st w o then WZero w
      else match st with
           | Some st => if store_orderly st a w o then WStore w else WOther
           | None => WOther
           end
  | IShA _ KStore _ _ _ None => WOther
  | IMem _ KLoad _ _ _ _ _ _ _ => WNone
  | IMem _ _ true _ _ _ _ _ _ => WOther
  | _ => WNone
  end.

(* The writes of the positions of block b from j, each with its position,
   its phase starts, and its class. *)
Fixpoint class_walk (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective)) (dirty : Z) (b j : nat) (s : Z)
    (st : option sstate) (body : list instr) : list (pc * Z * wclass) :=
  match body with
  | [] => []
  | ins :: rest =>
      let later := class_walk T C dirty b (S j) (phase_next_at T (b, j) s)
                     (option_map (fun st => ss_transfer_at C dirty (b, j) s st ins) st) rest in
      match write_class st ins with WNone => later | c => ((b, j), s, c) :: later end
  end.

(* The blocks that the threads of a divergent branch pass: the blocks of
   the region of each block whose walk stops at its last branch. *)
Definition diverged_blocks (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective)) (prog : program) (P : list Z)
    (dirty : Z) (D : list (option (PM.t Z))) (b : nat) : list nat :=
  match nth b D None with
  | None => []
  | Some e =>
      match ss_walk_at T C dirty b 0 (nth b P 0) (entry_state G b e) (block_body prog b) [] with
      | Some _ => []
      | None =>
          match last_instr (block_body prog b) with
          | Some (IBra (Some _) tg) => region_blocks prog b (region_join prog b (length prog) tg)
          | _ => []
          end
      end
  end.

(* The writes of the program: a block that a divergent branch's threads
   may pass has no state that holds at every execution. *)
Definition program_writes (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective)) (prog : program) (P : list Z)
    (dirty : Z) (D : list (option (PM.t Z))) : list (pc * Z * wclass) :=
  let regions := flat_map (diverged_blocks T C prog P dirty D) (seq 0 (length prog)) in
  flat_map (fun b => class_walk T C dirty b 0 (nth b P 0)
              (if mem_nat b regions then None
               else option_map (fun e => (entry_state G b e)) (nth b D None))
              (block_body prog b))
           (seq 0 (length prog)).

(* Two writes of different positions in one phase that the route orders:
   atomics of one width, or stores of the number 0 of one width. *)
Definition compatible (c1 c2 : wclass) : bool :=
  match c1, c2 with
  | WAtomic w1, WAtomic w2 | WZero w1, WZero w2 => Z.eqb w1 w2
  | _, _ => false
  end.

Fixpoint phase_iter_at (T : PM.t (PM.t Z)) (b j : nat) (s : Z) (k : nat) : Z :=
  match k with O => s | S k' => phase_iter_at T b (S j) (phase_next_at T (b, j) s) k' end.

(* Every two threads that write one byte in one phase make atomics of one
   width or store one value of one width: every write is of a class other
   than WOther; two writes of different positions whose phase starts meet
   are compatible; and no thread stores at a WStore position twice in one
   phase: with the WStore positions added to the phase starts, none is a
   phase start at its own position. *)
Definition writes_ordered_at (cta : list pc) (T : PM.t (PM.t Z)) (C : PM.t (PM.t collective)) (prog : program)
    (P : list Z) (dirty : Z) (D : list (option (PM.t Z))) : bool :=
  let W := program_writes T C prog P dirty D in
  forallb (fun x => match snd x with WOther => false | _ => true end) W &&
  forallb (fun x => forallb (fun y =>
     pc_eqb (fst (fst x)) (fst (fst y)) || Z.eqb (Z.land (snd (fst x)) (snd (fst y))) 0 ||
     compatible (snd x) (snd y)) W) W &&
  let stores := flat_map (fun x => match snd x with WStore _ => [fst (fst x)] | _ => [] end) W in
  let cta' := cta ++ stores in
  let T' := pc_table (cta_bits cta' 0) in
  let P' := phase_propose_at T' prog in
  phase_verify_at T' prog P' &&
  forallb (fun q => Z.eqb (Z.land (phase_iter_at T' (fst q) 0 (nth (fst q) P' 0) (snd q)) (phase_bit cta' q)) 0)
          stores.

End Analysis.

Lemma collective_table_same : forall cols p, pc_find (pc_table cols) p = collective_at cols p.
Proof.
  intros cols p. rewrite pc_find_table.
  induction cols as [|[q c] rest IH]; [reflexivity|]. cbn [find fst snd collective_at].
  rewrite (pc_eqb_sym p q). destruct (pc_eqb q p); [reflexivity|exact IH].
Qed.

Lemma ss_transfer_at_same : forall F tidr cols dirty p s st i,
  ss_transfer_at F tidr (pc_table cols) dirty p s st i = ss_transfer F tidr cols dirty p s st i.
Proof. intros. unfold ss_transfer_at, ss_transfer. rewrite collective_table_same. reflexivity. Qed.

Lemma ss_walk_at_same : forall F tidr cols cta dirty body b i s st edges,
  ss_walk_at F tidr (pc_table (cta_bits cta 0)) (pc_table cols) dirty b i s st body edges =
  ss_walk F tidr cols cta dirty b i s st body edges.
Proof.
  intros F tidr cols cta dirty body. induction body as [|ins rest IH]; intros b i s st edges; [reflexivity|].
  cbn [ss_walk_at ss_walk]. rewrite ss_transfer_at_same, phase_next_at_same.
  destruct (negb (branch_ok st ins)); [reflexivity|]. apply IH.
Qed.

Lemma diverge_block_at_same : forall F tidr cols G cta events prog P dirty b e,
  diverge_block_at F tidr G cta events (pc_table (cta_bits cta 0)) (pc_table cols) prog P dirty b e =
  diverge_block F tidr cols G cta events prog P dirty b e.
Proof.
  intros. unfold diverge_block_at, diverge_block. cbv zeta.
  destruct (last_instr (block_body prog b)) as [i|]; [|reflexivity].
  destruct i; try reflexivity. destruct g; [|reflexivity].
  rewrite ss_walk_at_same. reflexivity.
Qed.

Lemma dep_block_at_same : forall F tidr cols G cta events prog P dirty D b,
  dep_block_at F tidr G cta events (pc_table (cta_bits cta 0)) (pc_table cols) prog P dirty D b =
  dep_block F tidr cols G cta events prog P dirty D b.
Proof.
  intros. unfold dep_block_at, dep_block. destruct (nth b D None) as [e|]; [|reflexivity]. cbv zeta.
  rewrite ss_walk_at_same. destruct (ss_walk F tidr cols cta dirty b 0 (nth b P 0) _ _ _) as [[st edges]|];
    [reflexivity|apply diverge_block_at_same].
Qed.

Lemma dep_verify_at_same : forall F tidr cols G uniform cta events prog P dirty D,
  dep_verify_at F tidr G uniform cta events (pc_table (cta_bits cta 0)) (pc_table cols) prog P dirty D =
  dep_verify F tidr cols G uniform cta events prog P dirty D.
Proof.
  intros. unfold dep_verify_at, dep_verify. f_equal.
  induction (seq 0 (length prog)) as [|b bs IH]; [reflexivity|].
  cbn [forallb]. rewrite dep_block_at_same, IH. reflexivity.
Qed.

(* The check: the block has whole warps; the thread index register is
   never written; the phase starts and dependences are closed under the
   program's edges; every conditional branch and exit has a predicate that
   is the same in every thread, or leads around blocks without waiting
   sites to a join block. A shared load reads a value that depends only on
   its address's bits when no thread may write shared memory in its phase
   and the program's shared writes are ordered (writes_ordered_at); if they
   are not, every shared load is checked as a value that may differ in
   every thread (the phases dirty, all of them). *)
Definition all_dirty : Z := -1.

Definition shared_check (F : lfacts) (uniform : reg -> bool) (tidr : reg) (n : nat)
    (prog : program) (cta events : list pc) (cols : list (pc * collective)) : bool :=
  Nat.ltb 0 n && Nat.leb n 1024 && Nat.eqb (n mod 32) 0 &&
  negb (is_cell F tidr) && forallb (loop_preserves tidr) (concat prog) &&
  let T := pc_table (cta_bits cta 0) in
  let C := pc_table cols in
  let P := phase_propose_at T prog in
  phase_verify_at T prog P &&
  let dirty := phase_dirty_at T prog P in
  let G := entry_facts_at F tidr uniform T C prog P dirty in
  let D := dep_propose_at F tidr G uniform cta events T C prog P dirty in
  if dep_verify_at F tidr G uniform cta events T C prog P dirty D &&
     writes_ordered_at F tidr G n cta T C prog P dirty D
  then true
  else dep_verify_at F tidr G uniform cta events T C prog P all_dirty
         (dep_propose_at F tidr G uniform cta events T C prog P all_dirty).

(* ------------------------------------------------------------------ *)
(* Thread index bits.                                                  *)

(* Threads x and y agree on the index bits of z. *)
Definition agree (z : Z) (x y : nat) : Prop := Z.land z (Z.lxor (Z.of_nat x) (Z.of_nat y)) = 0.

(* The thread whose index differs from x by xor m: the lane l xor m of x's
   warp when m is below 32. *)
Definition partner (x : nat) (m : Z) : nat := Z.to_nat (Z.lxor (Z.of_nat x) m).

Lemma agree_bits : forall z x y,
  agree z x y <-> forall i, 0 <= i -> Z.testbit z i = true ->
                   Z.testbit (Z.of_nat x) i = Z.testbit (Z.of_nat y) i.
Proof.
  intros z x y. unfold agree. split.
  - intros H i Hi Hz. assert (Hb := f_equal (fun v => Z.testbit v i) H). cbn beta in Hb.
    rewrite Z.land_spec, Z.lxor_spec, Hz, Z.testbit_0_l in Hb. cbn in Hb.
    destruct (Z.testbit (Z.of_nat x) i), (Z.testbit (Z.of_nat y) i); cbn in Hb; congruence.
  - intros H. apply Z.bits_inj'. intros i Hi. rewrite Z.land_spec, Z.lxor_spec, Z.testbit_0_l.
    destruct (Z.testbit z i) eqn:Hz; [|reflexivity]. rewrite (H i Hi Hz). apply Bool.xorb_nilpotent.
Qed.

Lemma agree_zero : forall x y, agree 0 x y.
Proof. intros x y. reflexivity. Qed.

Lemma agree_lor : forall a b x y, agree (Z.lor a b) x y -> agree a x y /\ agree b x y.
Proof.
  intros a b x y H. rewrite agree_bits in H. split; apply agree_bits; intros i Hi Hz; apply H; auto;
  rewrite Z.lor_spec, Hz; [reflexivity|apply orb_true_r].
Qed.

Lemma agree_sub : forall a b x y, zsub a b = true -> agree b x y -> agree a x y.
Proof.
  intros a b x y Hs H. unfold zsub in Hs. apply Z.eqb_eq in Hs. rewrite agree_bits in *.
  intros i Hi Ha. apply H; [exact Hi|].
  assert (Hb := f_equal (fun v => Z.testbit v i) Hs). cbn beta in Hb.
  rewrite Z.land_spec, Z.lnot_spec, Ha, Z.testbit_0_l in Hb by exact Hi.
  destruct (Z.testbit b i); [reflexivity|discriminate].
Qed.

Lemma lxor_nonneg : forall a b, 0 <= a -> 0 <= b -> 0 <= Z.lxor a b.
Proof. intros a b Ha Hb. apply Z.lxor_nonneg. split; intros; assumption. Qed.

Lemma partner_value : forall x m, 0 <= m -> Z.of_nat (partner x m) = Z.lxor (Z.of_nat x) m.
Proof. intros x m Hm. unfold partner. rewrite Z2Nat.id; [reflexivity|]. apply lxor_nonneg; lia. Qed.

Lemma partner_zero : forall x, partner x 0 = x.
Proof. intros x. unfold partner. rewrite Z.lxor_0_r. apply Nat2Z.id. Qed.

Lemma partner_partner : forall x m k, 0 <= m -> 0 <= k ->
  partner (partner x m) k = partner x (Z.lxor m k).
Proof.
  intros x m k Hm Hk. unfold partner at 1. rewrite partner_value by exact Hm.
  rewrite Z.lxor_assoc. reflexivity.
Qed.

Lemma partner_self : forall x m, 0 <= m -> partner (partner x m) m = x.
Proof. intros x m Hm. rewrite partner_partner by lia. rewrite Z.lxor_nilpotent. apply partner_zero. Qed.

(* xor with a mask below 32 changes only the five low bits. *)
Lemma lxor_low : forall v m, 0 <= v -> 0 <= m < 32 -> Z.lxor v m / 32 = v / 32.
Proof.
  intros v m Hv Hm. change 32 with (2 ^ 5). rewrite <- !Z.shiftr_div_pow2 by lia.
  rewrite Z.shiftr_lxor. replace (Z.shiftr m 5) with 0; [apply Z.lxor_0_r|].
  rewrite Z.shiftr_div_pow2 by lia. symmetry. apply Z.div_small. change (2 ^ 5) with 32. lia.
Qed.

Lemma partner_warp : forall x m, 0 <= m < 32 -> (partner x m / 32 = x / 32)%nat.
Proof.
  intros x m Hm. apply Nat2Z.inj. rewrite !Nat2Z.inj_div, partner_value by lia.
  apply lxor_low; lia.
Qed.

Lemma partner_lt : forall n x m, (x < n)%nat -> (n mod 32 = 0)%nat -> 0 <= m < 32 -> (partner x m < n)%nat.
Proof.
  intros n x m Hx Hn Hm. pose proof (partner_warp x m Hm) as Hw.
  pose proof (Nat.div_mod (partner x m) 32 ltac:(lia)) as Hp.
  pose proof (Nat.mod_upper_bound (partner x m) 32 ltac:(lia)).
  pose proof (Nat.div_mod n 32 ltac:(lia)) as Hn'. rewrite Hn, Nat.add_0_r in Hn'.
  assert (x / 32 < n / 32)%nat.
  { apply Nat.div_lt_upper_bound; [lia|]. rewrite <- Hn'. exact Hx. }
  rewrite Hw in Hp. nia.
Qed.

Lemma agree_partner : forall z x y m, 0 <= m -> agree z x y -> agree z (partner x m) (partner y m).
Proof.
  intros z x y m Hm H. unfold agree in *. rewrite !partner_value by exact Hm.
  rewrite (Z.lxor_comm (Z.of_nat y) m), Z.lxor_assoc, <- (Z.lxor_assoc m m), Z.lxor_nilpotent,
    Z.lxor_0_l. exact H.
Qed.

Lemma agree_warp : forall x y, agree (Z.lnot 31) x y -> (x / 32 = y / 32)%nat.
Proof.
  intros x y H. rewrite agree_bits in H. apply Nat2Z.inj. rewrite !Nat2Z.inj_div.
  change (Z.of_nat 32) with (2 ^ 5). rewrite <- !Z.shiftr_div_pow2 by lia.
  apply Z.bits_inj'. intros i Hi. rewrite !Z.shiftr_spec by exact Hi. apply H; [lia|].
  rewrite Z.lnot_spec by lia. change 31 with (Z.ones 5). rewrite Z.ones_spec_high by lia. reflexivity.
Qed.

Lemma single_lane_cases : forall m, single_lane m = true -> m = 1 \/ m = 2 \/ m = 4 \/ m = 8 \/ m = 16.
Proof.
  intros m H. unfold single_lane in H. cbn [existsb] in H.
  repeat (apply orb_true_iff in H; destruct H as [H|H]; [apply Z.eqb_eq in H; lia|]). discriminate.
Qed.

(* A value that depends on the bits D and does not change when one bit m
   of the index changes depends only on the bits of D other than m. *)
Lemma reduce_single : forall n (f : nat -> Z) D m, single_lane m = true -> (n mod 32 = 0)%nat ->
  (forall x y, (x < n)%nat -> (y < n)%nat -> agree D x y -> f x = f y) ->
  (forall x, (x < n)%nat -> f (partner x m) = f x) ->
  forall x y, (x < n)%nat -> (y < n)%nat -> agree (Z.land D (Z.lnot m)) x y -> f x = f y.
Proof.
  intros n f D m Hm Hn Hdep Hinv x y Hx Hy H.
  assert (Hm32 : 0 <= m < 32) by (destruct (single_lane_cases m Hm) as [->|[->|[->|[->| ->]]]]; lia).
  assert (Hk : exists k, 0 <= k < 5 /\ m = 2 ^ k).
  { destruct (single_lane_cases m Hm) as [->|[->|[->|[->| ->]]]];
    [exists 0|exists 1|exists 2|exists 3|exists 4]; split; (lia || reflexivity). }
  destruct Hk as [k [Hk ->]].
  rewrite agree_bits in H.
  destruct (Bool.bool_dec (Z.testbit (Z.of_nat x) k) (Z.testbit (Z.of_nat y) k)) as [Heq|Hne].
  - apply Hdep; [exact Hx|exact Hy|]. apply agree_bits. intros i Hi Hz.
    destruct (Z.eq_dec i k) as [->|Hik]; [exact Heq|].
    apply H; [exact Hi|]. rewrite Z.land_spec, Z.lnot_spec, Hz, Z.pow2_bits_false by lia. reflexivity.
  - rewrite <- (Hinv y Hy). apply Hdep; [exact Hx|apply partner_lt; assumption|].
    apply agree_bits. intros i Hi Hz. rewrite partner_value by lia. rewrite Z.lxor_spec.
    destruct (Z.eq_dec i k) as [->|Hik].
    + rewrite Z.pow2_bits_true by lia.
      destruct (Z.testbit (Z.of_nat x) k), (Z.testbit (Z.of_nat y) k); cbn; congruence.
    + rewrite Z.pow2_bits_false by lia. rewrite xorb_false_r. apply H; [exact Hi|].
      rewrite Z.land_spec, Z.lnot_spec, Hz, Z.pow2_bits_false by lia. reflexivity.
Qed.

(* A value of the index's field. *)
Definition field_value (x : nat) (f : Z * Z) : Z := Z.land (Z.shiftl (Z.of_nat x) (fst f)) (snd f).

Lemma field_agree : forall f x y, agree (field_dep f) x y -> field_value x f = field_value y f.
Proof.
  intros [sh mk] x y H. rewrite agree_bits in H. unfold field_value, field_dep in *. cbn [fst snd] in *.
  apply Z.bits_inj'. intros i Hi. rewrite !Z.land_spec.
  destruct (Z.testbit mk i) eqn:Hmk; [|rewrite !andb_false_r; reflexivity].
  rewrite !andb_true_r. destruct (Z_lt_le_dec (i - sh) 0) as [Hl|Hl].
  - destruct (Z_le_gt_dec 0 sh) as [Hs|Hs].
    + rewrite !Z.shiftl_spec by exact Hi. rewrite !Z.testbit_neg_r by exact Hl. reflexivity.
    + lia.
  - assert (Hsh : Z.testbit (Z.shiftl (Z.of_nat x) sh) i = Z.testbit (Z.of_nat x) (i - sh) /\
                  Z.testbit (Z.shiftl (Z.of_nat y) sh) i = Z.testbit (Z.of_nat y) (i - sh)).
    { split; apply Z.shiftl_spec; exact Hi. }
    destruct Hsh as [-> ->]. apply H; [exact Hl|].
    rewrite Z.shiftr_spec by exact Hl. replace (i - sh + sh) with i by lia. exact Hmk.
Qed.

(* ------------------------------------------------------------------ *)
(* Steps.                                                              *)

Definition in_range (F : lfacts) (rs : reg -> Z) : Prop := forall r, 0 <= rs r < 2 ^ wd F r.

Lemma result_pure : forall i m1 a1 s1 t1 m2 a2 s2 t2 rs, pure_exact i = true ->
  result m1 a1 s1 t1 rs i = result m2 a2 s2 t2 rs i.
Proof. intros i m1 a1 s1 t1 m2 a2 s2 t2 rs H. destruct i; try discriminate H; reflexivity. Qed.

Section Steps.

Variable F : lfacts.
Variable prog : program.
Variable E : env.

Lemma upd_range : forall rs d v, in_range F rs -> in_range F (upd (wdn F) rs d v).
Proof.
  intros rs d v H r. unfold upd. destruct (Pos.eqb r d) eqn:Hr; [|apply H].
  apply Pos.eqb_eq in Hr. subst r. unfold trunc, wd. apply Z.mod_pos_bound. apply Z.pow_pos_nonneg; lia.
Qed.

Lemma write_all_range : forall rs t ds, in_range F rs -> in_range F (write_all (wdn F) (e_arb E) rs t ds).
Proof.
  intros rs t ds. revert rs. induction ds as [|d ds IH]; intros rs H; cbn [write_all]; [exact H|].
  apply IH. apply upd_range. exact H.
Qed.

Lemma fold_cells_range : forall (f : (reg -> Z) -> reg * reg * Z -> reg -> Z),
  (forall rs k, in_range F rs -> in_range F (f rs k)) ->
  forall cells rs, in_range F rs -> in_range F (fold_left f cells rs).
Proof.
  intros f Hf cells. induction cells as [|k cells IH]; intros rs H; cbn [fold_left]; [exact H|].
  apply IH. apply Hf. exact H.
Qed.

Lemma lstore_range : forall t rs x width v, in_range F rs ->
  in_range F (lstore (wdn F) (e_arb E) (e_init E) (lf_cells F) t rs x width v).
Proof.
  intros t rs x width v H. unfold lstore. apply fold_cells_range; [|exact H].
  intros rs0 [[p u] c] H0. unfold lstore_cell.
  destruct (_ && _)%bool; [apply upd_range; exact H0|].
  destruct (_ && _)%bool; [apply upd_range; exact H0|exact H0].
Qed.

Lemma lframe_range : forall t rs vs, in_range F rs ->
  in_range F (lframe (wdn F) (e_arb E) (lf_cells F) t rs vs).
Proof.
  intros t rs vs H. unfold lframe. apply fold_cells_range; [|exact H].
  intros rs0 [[p u] c] H0. unfold lframe_cell.
  destruct (existsb (Pos.eqb u) vs); [apply upd_range; exact H0|exact H0].
Qed.

Lemma step_range : forall t s s', in_range F (regs s) -> phase_step F prog E t s = Some s' ->
  in_range F (regs s').
Proof.
  intros t s s' H Hs. unfold phase_step, step in Hs.
  destruct (fetch prog (blk s) (idx s)) as [i|]; [|discriminate]. injection Hs as <-.
  destruct (holds (guard_of i) (regs s)); cbn [negb]; [|exact H].
  destruct i; cbn [regs goto dest]; try exact H; try (apply upd_range; exact H).
  - destruct k; [apply write_all_range; exact H| |];
    (destruct gen; [unfold gstore; destruct (_ && _)%bool;
                    [apply lstore_range; apply write_all_range; exact H|apply write_all_range; exact H]
                   |apply write_all_range; exact H]).
  - apply write_all_range. exact H.
  - apply lstore_range. exact H.
  - apply lframe_range. exact H.
Qed.

Lemma step_pure_regs : forall t s s' ins d,
  fetch prog (blk s) (idx s) = Some ins -> pure_exact ins = true -> dest ins = Some d ->
  phase_step F prog E t s = Some s' ->
  regs s' = if holds (guard_of ins) (regs s)
            then upd (wdn F) (regs s) d (result (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => 0) 0 (regs s) ins)
            else regs s.
Proof.
  intros t s s' ins d Hf Hp Hd Hs. unfold phase_step, step in Hs. rewrite Hf in Hs. injection Hs as <-.
  rewrite (result_pure ins (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => 0) 0 (e_mem E) (e_arb E) (e_shm E) t)
    by exact Hp.
  destruct (holds (guard_of ins) (regs s)); cbn [negb]; [|reflexivity].
  destruct ins; try discriminate Hp; cbn [dest] in Hd; injection Hd as <-; reflexivity.
Qed.

Lemma step_def_regs : forall t s s' d,
  fetch prog (blk s) (idx s) = Some (IDef None [d]) -> phase_step F prog E t s = Some s' ->
  regs s' = upd (wdn F) (regs s) d (e_arb E t d).
Proof.
  intros t s s' d Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs. injection Hs as <-. reflexivity.
Qed.

Lemma step_lds_regs : forall t s s' bits d a off,
  fetch prog (blk s) (idx s) = Some (ILdS None bits d a off) -> phase_step F prog E t s = Some s' ->
  regs s' = upd (wdn F) (regs s) d (trunc bits (e_shm E t (trunc 32 (regs s a + off)))).
Proof.
  intros t s s' bits d a off Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs. injection Hs as <-.
  reflexivity.
Qed.

Lemma step_keep : forall t s s' ins r,
  fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
  existsb (Pos.eqb r) (loop_writes ins) = false -> (writes_cells ins = false \/ is_cell F r = false) ->
  regs s' r = regs s r.
Proof.
  intros t s s' ins r Hf Hs Hw Hc. destruct Hc as [Hc|Hc].
  - unfold phase_step, step in Hs. rewrite Hf in Hs. injection Hs as <-.
    destruct (holds (guard_of ins) (regs s)); cbn [negb]; [|reflexivity].
    destruct ins; try discriminate Hc; cbn [loop_writes dest existsb] in Hw; cbn [regs goto dest];
      try reflexivity; try (rewrite orb_false_r in Hw; unfold upd; rewrite Hw; reflexivity).
    apply loop_write_all_preserves. exact Hw.
  - apply (loop_step_preserves F prog E t s s' ins r Hc Hf Hs). unfold loop_preserves. rewrite Hw. reflexivity.
Qed.

End Steps.

Lemma firstn_In_sub : forall {A} k (l : list A) a, In a (firstn k l) -> In a l.
Proof.
  intros A k. induction k as [|k IH]; intros l a H; [contradiction|].
  destruct l as [|b l]; [contradiction|]. destruct H as [H|H]; [left; exact H|right; exact (IH l a H)].
Qed.

(* ------------------------------------------------------------------ *)
(* What the abstract state states about the registers of every thread. *)

Section Relation.

Variable F : lfacts.
Variable n : nat.

(* Two threads whose indices agree on a register's bits hold the same
   value in it. *)
Definition deps_ok (m : PM.t Z) (R : nat -> reg -> Z) : Prop :=
  forall r z x y, PM.find r m = Some z -> (x < n)%nat -> (y < n)%nat -> agree z x y -> R x r = R y r.

Definition fact_holds (R : nat -> reg -> Z) (f : sfact) : Prop :=
  match f with
  | SRel d m s g => 0 <= m < 32 /\ forall x, (x < n)%nat -> R x d = gval g (wdn F d) (R (partner x m) s)
  | SField d sh mk => forall x, (x < n)%nat -> R x d = field_value x (sh, mk)
  | SConst d z => forall x, (x < n)%nat -> R x d = z
  | SBase d u sh mk => forall x, (x < n)%nat -> R x d = trunc (wdn F d) (R x u + field_value x (sh, mk))
  end.

Definition sstate_ok (st : sstate) (R : nat -> reg -> Z) : Prop :=
  deps_ok (ss_dep st) R /\ forall f, In f (ss_facts st) -> fact_holds R f.

Definition dep_spec (dep : option Z) (R : nat -> reg -> Z) (d : reg) : Prop :=
  forall z, dep = Some z -> forall x y, (x < n)%nat -> (y < n)%nat -> agree z x y -> R x d = R y d.

Lemma fact_holds_ext : forall R R' f, fact_holds R f ->
  (forall x r, (x < n)%nat -> mentions r f = true -> R' x r = R x r) ->
  (n mod 32 = 0)%nat -> fact_holds R' f.
Proof.
  intros R R' f H He Hn. destruct f as [d m s g|d sh mk|d z|d u sh mk]; cbn [fact_holds mentions] in *.
  - destruct H as [Hm H]. split; [exact Hm|]. intros x Hx.
    rewrite (He x d Hx (orb_true_intro _ _ (or_introl (Pos.eqb_refl d)))).
    rewrite (He (partner x m) s (partner_lt n x m Hx Hn Hm) (orb_true_intro _ _ (or_intror (Pos.eqb_refl s)))).
    apply H. exact Hx.
  - intros x Hx. rewrite (He x d Hx (Pos.eqb_refl d)). apply H. exact Hx.
  - intros x Hx. rewrite (He x d Hx (Pos.eqb_refl d)). apply H. exact Hx.
  - intros x Hx. rewrite (He x d Hx (orb_true_intro _ _ (or_introl (Pos.eqb_refl d)))).
    rewrite (He x u Hx (orb_true_intro _ _ (or_intror (Pos.eqb_refl u)))). apply H. exact Hx.
Qed.

Lemma ss_write_ok : forall st d dep fs R R', (n mod 32 = 0)%nat ->
  sstate_ok st R -> (forall x r, (x < n)%nat -> r <> d -> R' x r = R x r) -> dep_spec dep R' d ->
  (forall f, In f fs -> fresh_fact d f = true -> fact_holds R' f) ->
  sstate_ok (ss_write st d dep fs) R'.
Proof.
  intros st d dep fs R R' Hn [Hdeps Hfacts] Hother Hdep Hfs. split.
  - intros r z x y Hr Hx Hy Ha. unfold ss_write in Hr. cbn [ss_dep] in Hr.
    destruct (Pos.eq_dec r d) as [->|Hrd].
    + destruct dep as [z'|]; [rewrite PM.gss in Hr; injection Hr as ->; exact (Hdep z eq_refl x y Hx Hy Ha)|].
      rewrite PM.grs in Hr. discriminate.
    + rewrite (Hother x r Hx Hrd), (Hother y r Hy Hrd).
      destruct dep as [z'|]; [rewrite PM.gso in Hr by exact Hrd|rewrite PM.gro in Hr by exact Hrd];
      exact (Hdeps r z x y Hr Hx Hy Ha).
  - intros f Hf. unfold ss_write in Hf. cbn [ss_facts] in Hf. apply in_app_or in Hf. destruct Hf as [Hf|Hf].
    + apply filter_In in Hf. destruct Hf as [Hf Hfresh]. exact (Hfs f Hf Hfresh).
    + assert (Hk : In f (filter (fun f => negb (mentions d f)) (ss_facts st))).
      { apply in_app_or in Hf. destruct Hf as [Hf|Hf]; [apply filter_In in Hf; exact (proj1 Hf)|].
        apply firstn_In_sub in Hf. apply filter_In in Hf. exact (proj1 Hf). }
      clear Hf. rename Hk into Hf.
      apply filter_In in Hf. destruct Hf as [Hf Hm]. apply negb_true_iff in Hm.
      apply (fact_holds_ext R); [exact (Hfacts f Hf)| |exact Hn].
      intros x r Hx Hr. apply Hother; [exact Hx|]. intros ->. congruence.
Qed.

Lemma ss_top_shape : forall rs st,
  (forall r, PM.find r (ss_dep (ss_top st rs)) = if existsb (Pos.eqb r) rs then None else PM.find r (ss_dep st)) /\
  (forall f, In f (ss_facts (ss_top st rs)) -> In f (ss_facts st) /\ forall r, In r rs -> mentions r f = false).
Proof.
  intros rs. induction rs as [|r0 rs IH]; intros st; cbn [ss_top fold_left existsb].
  - split; [reflexivity|]. intros f Hf. split; [exact Hf|]. intros r [].
  - change (fold_left (fun st r => ss_write st r None []) rs (ss_write st r0 None [])) with
      (ss_top (ss_write st r0 None []) rs).
    destruct (IH (ss_write st r0 None [])) as [Hd Hf]. split.
    + intros r. rewrite Hd. unfold ss_write. cbn [ss_dep].
      destruct (Pos.eqb_spec r r0) as [->|Hne]; cbn [orb].
      * rewrite PM.grs. destruct (existsb (Pos.eqb r0) rs); reflexivity.
      * rewrite PM.gro by exact Hne. reflexivity.
    + intros f Hin. destruct (Hf f Hin) as [Hin' Hm]. unfold ss_write in Hin'. cbn [ss_facts filter app] in Hin'.
      assert (Hk : In f (filter (fun f => negb (mentions r0 f)) (ss_facts st))).
      { apply in_app_or in Hin'. destruct Hin' as [Hin'|Hin']; [apply filter_In in Hin'; exact (proj1 Hin')|].
        apply firstn_In_sub in Hin'. apply filter_In in Hin'. exact (proj1 Hin'). }
      apply filter_In in Hk. destruct Hk as [Hin'' Hm0]. clear Hin'. rename Hin'' into Hin'.
      split; [exact Hin'|]. intros r [<-|Hr]; [apply negb_true_iff; exact Hm0|exact (Hm r Hr)].
Qed.

Lemma ss_top_ok : forall st rs R R', (n mod 32 = 0)%nat -> sstate_ok st R ->
  (forall x r, (x < n)%nat -> existsb (Pos.eqb r) rs = false -> R' x r = R x r) ->
  sstate_ok (ss_top st rs) R'.
Proof.
  intros st rs R R' Hn [Hdeps Hfacts] Hother. destruct (ss_top_shape rs st) as [Hd Hf]. split.
  - intros r z x y Hr Hx Hy Ha. rewrite Hd in Hr. destruct (existsb (Pos.eqb r) rs) eqn:Hrs; [discriminate|].
    rewrite (Hother x r Hx Hrs), (Hother y r Hy Hrs). exact (Hdeps r z x y Hr Hx Hy Ha).
  - intros f Hin. destruct (Hf f Hin) as [Hin' Hm].
    apply (fact_holds_ext R); [exact (Hfacts f Hin')| |exact Hn].
    intros x r Hx Hr. apply Hother; [exact Hx|].
    destruct (existsb (Pos.eqb r) rs) eqn:Hrs; [|reflexivity].
    apply existsb_exists in Hrs. destruct Hrs as [r' [Hr' Heq]]. apply Pos.eqb_eq in Heq. subst r'.
    rewrite (Hm r Hr') in Hr. discriminate.
Qed.

Lemma const_of_ok : forall st R r z, sstate_ok st R -> const_of st r = Some z ->
  forall x, (x < n)%nat -> R x r = z.
Proof.
  intros st R r z [_ Hfacts]. unfold const_of. induction (ss_facts st) as [|f fs IH]; cbn [fold_right];
    [discriminate|].
  intros H. destruct f as [d m s g|d sh mk|d z0|d u sh mk]; try (apply IH; [intros f' Hf'; apply Hfacts; right; exact Hf'|exact H]).
  destruct (Pos.eqb_spec d r) as [->|Hne].
  - injection H as <-. exact (Hfacts (SConst r z0) (or_introl eq_refl)).
  - apply IH; [intros f' Hf'; apply Hfacts; right; exact Hf'|exact H].
Qed.

Lemma rels_of_ok : forall st R r m s g, sstate_ok st R -> In (m, s, g) (rels_of st r) ->
  fact_holds R (SRel r m s g).
Proof.
  intros st R r m s g [_ Hfacts]. unfold rels_of. induction (ss_facts st) as [|f fs IH]; cbn [fold_right];
    [intros []|].
  intros H. assert (Hrest : forall f', In f' fs -> fact_holds R f') by (intros f' Hf'; apply Hfacts; right; exact Hf').
  destruct f as [d m0 s0 g0|d sh mk|d z0|d u sh mk]; try exact (IH Hrest H).
  destruct (Pos.eqb_spec d r) as [->|Hne]; [|exact (IH Hrest H)].
  destruct H as [H|H]; [|exact (IH Hrest H)].
  injection H as <- <- <-. exact (Hfacts _ (or_introl eq_refl)).
Qed.

Lemma base_of_ok : forall st R r u sh mk, sstate_ok st R -> base_of st r = Some (u, sh, mk) ->
  fact_holds R (SBase r u sh mk).
Proof.
  intros st R r u sh mk [_ Hfacts]. unfold base_of. induction (ss_facts st) as [|f fs IH]; cbn [fold_right];
    [discriminate|].
  intros H. assert (Hrest : forall f', In f' fs -> fact_holds R f') by (intros f' Hf'; apply Hfacts; right; exact Hf').
  destruct f as [d m0 s0 g0|d sh0 mk0|d z0|d u0 sh0 mk0]; try exact (IH Hrest H).
  destruct (Pos.eqb_spec d r) as [->|Hne]; [|exact (IH Hrest H)].
  injection H as <- <- <-. exact (Hfacts _ (or_introl eq_refl)).
Qed.

Lemma sfun_eqb_eq : forall a b, sfun_eqb a b = true -> a = b.
Proof. intros [] []; cbn; congruence. Qed.

Lemma has_rel_ok : forall st R d m s g, sstate_ok st R -> has_rel st d m s g = true ->
  fact_holds R (SRel d m s g).
Proof.
  intros st R d m s g Hok H. unfold has_rel in H. apply existsb_exists in H.
  destruct H as [[[m' s'] g'] [Hin Heq]]. rewrite !andb_true_iff in Heq.
  destruct Heq as [[Hm Hs] Hg]. apply Z.eqb_eq in Hm. apply Pos.eqb_eq in Hs. apply sfun_eqb_eq in Hg.
  subst. exact (rels_of_ok st R d m s g Hok Hin).
Qed.

Lemma dep_regs_ok : forall st R rs z, sstate_ok st R -> dep_regs st rs = Some z ->
  forall x y, (x < n)%nat -> (y < n)%nat -> agree z x y -> forall r, In r rs -> R x r = R y r.
Proof.
  intros st R rs z [Hdeps _]. unfold dep_regs.
  assert (Hgen : forall acc, fold_left (fun acc r => dunion acc (dep_of st r)) rs acc = Some z ->
    exists z0, acc = Some z0 /\ (forall x y, agree z x y -> agree z0 x y) /\
    forall r, In r rs -> exists zr, dep_of st r = Some zr /\ forall x y, agree z x y -> agree zr x y).
  { induction rs as [|r rs IH]; intros acc H; cbn [fold_left] in H.
    - exists z. split; [exact H|]. split; [auto|]. intros r [].
    - destruct (IH _ H) as [z1 [Hz1 [Ha1 Hr1]]]. unfold dunion in Hz1.
      destruct acc as [z0|]; [|discriminate]. destruct (dep_of st r) as [zr|] eqn:Hzr; [|discriminate].
      injection Hz1 as <-. exists z0. split; [reflexivity|]. split.
      + intros x y Ha. exact (proj1 (agree_lor _ _ _ _ (Ha1 x y Ha))).
      + intros r' [<-|Hr']; [exists zr; split; [exact Hzr|]|exact (Hr1 r' Hr')].
        intros x y Ha. exact (proj2 (agree_lor _ _ _ _ (Ha1 x y Ha))). }
  intros H x y Hx Hy Ha r Hr. destruct (Hgen _ H) as [_ [_ [_ Hrs]]].
  destruct (Hrs r Hr) as [zr [Hzr Hag]]. exact (Hdeps r zr x y Hzr Hx Hy (Hag x y Ha)).
Qed.

End Relation.

(* ------------------------------------------------------------------ *)
(* Arithmetic.                                                         *)

Definition result0 (rs : reg -> Z) (i : instr) : Z :=
  result (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => 0) 0 rs i.

Lemma opv_agree : forall rs1 rs2 o, (forall r, In r (opnd_inputs o) -> rs1 r = rs2 r) -> opv rs1 o = opv rs2 o.
Proof. intros rs1 rs2 [r|z] H; cbn; [apply H; left; reflexivity|reflexivity]. Qed.

Lemma result_inputs : forall i rs1 rs2, pure_exact i = true ->
  (forall r, In r (instruction_inputs i) -> rs1 r = rs2 r) -> result0 rs1 i = result0 rs2 i.
Proof.
  intros i rs1 rs2 Hp H.
  assert (Hin : forall r, In r (match i with
    | IMov _ _ s => [s]
    | IBin _ _ _ _ a b | ISetp _ _ _ _ _ a b | IMinMax _ _ _ _ _ a b | IDiv _ _ _ _ _ a b => a :: opnd_inputs b
    | IMad _ _ _ a b c => a :: opnd_inputs b ++ opnd_inputs c
    | IWide _ _ _ a b c => a :: opnd_inputs b ++ match c with Some r => [r] | _ => [] end
    | ICvt _ _ _ _ _ r | IBfe _ _ _ r _ _ => [r]
    | IAndp _ _ a b | IOrp _ _ a b | IXorp _ _ a b => [a; b]
    | ISelp _ _ _ a b p => p :: opnd_inputs a ++ opnd_inputs b
    | ILoad _ _ _ _ a _ | ILdS _ _ _ a _ | ILdL _ _ _ a _ _ | IShA _ _ a _ _ _ => [a]
    | IMem _ _ _ a _ _ v _ _ | IStL _ a _ _ v _ => a :: match v with Some o => opnd_inputs o | _ => [] end
    | _ => [] end) -> rs1 r = rs2 r).
  { intros r Hr. apply H. unfold instruction_inputs. destruct (guard_of i) as [[neg p]|]; [right|]; exact Hr. }
  clear H. unfold result0.
  destruct i; try discriminate Hp; cbn [result];
    repeat match goal with
    | |- context [rs1 ?r] =>
        let E := fresh in assert (E : rs1 r = rs2 r) by (apply Hin; cbn; rewrite ?in_app_iff; tauto);
        rewrite E; clear E
    | |- context [opv rs1 ?o] =>
        let E := fresh in assert (E : opv rs1 o = opv rs2 o)
          by (apply opv_agree; intros r Hr; apply Hin; cbn; rewrite ?in_app_iff; tauto);
        rewrite E; clear E
    end; try reflexivity.
  destruct c as [r|]; [|reflexivity]. rewrite (Hin r); [reflexivity|]. cbn. rewrite in_app_iff. cbn. tauto.
Qed.

Lemma holds_inputs : forall i rs1 rs2,
  (forall r, In r (instruction_inputs i) -> rs1 r = rs2 r) -> holds (guard_of i) rs1 = holds (guard_of i) rs2.
Proof.
  intros i rs1 rs2 H. unfold instruction_inputs in H. destruct (guard_of i) as [[neg p]|]; [|reflexivity].
  unfold holds. rewrite (H p (or_introl eq_refl)). reflexivity.
Qed.

Lemma binop_comm : forall o w a b, commutative o = true -> binop_eval o w a b = binop_eval o w b a.
Proof.
  intros o w a b H. destruct o; try discriminate H; cbn [binop_eval];
    [apply Z.add_comm|apply Z.mul_comm|apply Z.land_comm|apply Z.lor_comm|apply Z.lxor_comm].
Qed.

Lemma trunc_small : forall w v, 0 <= v < 2 ^ Z.of_nat w -> trunc w v = v.
Proof. intros w v H. unfold trunc. apply Z.mod_small. exact H. Qed.

Lemma trunc_range : forall w v, 0 <= trunc w v < 2 ^ Z.of_nat w.
Proof. intros w v. unfold trunc. apply Z.mod_pos_bound. apply Z.pow_pos_nonneg; lia. Qed.

Lemma extend_inj : forall w sg u v, 0 <= u < 2 ^ Z.of_nat w -> 0 <= v < 2 ^ Z.of_nat w ->
  extend w sg u = extend w sg v -> u = v.
Proof.
  intros w sg u v Hu Hv H. unfold extend in H. destruct sg.
  - unfold sext in H. rewrite !trunc_small in H by assumption.
    destruct w as [|w].
    + cbn in Hu, Hv. lia.
    + rewrite Nat2Z.inj_succ in *. rewrite Z.pow_succ_r in Hu, Hv by lia.
      replace (Z.succ (Z.of_nat w) - 1) with (Z.of_nat w) in H by lia.
      rewrite Z.pow_succ_r in H by lia.
      destruct (u <? 2 ^ Z.of_nat w) eqn:Hu', (v <? 2 ^ Z.of_nat w) eqn:Hv';
        rewrite ?Z.ltb_lt, ?Z.ltb_ge in *; lia.
  - rewrite !trunc_small in H by assumption. exact H.
Qed.

(* The two halves of a 64-bit value give the value back. *)
Lemma halves_recompose : forall v, 0 <= v < 2 ^ 64 ->
  Z.lor (trunc 64 (trunc 32 v)) (trunc 64 (trunc 64 (trunc 32 (v / 2 ^ 32) * 2 ^ 32))) = v.
Proof.
  intros v Hv. assert (Hlo := trunc_range 32 v). assert (Hhi := trunc_range 32 (v / 2 ^ 32)).
  cbn in Hlo, Hhi.
  assert (Hm : 0 <= trunc 32 (v / 2 ^ 32) * 2 ^ 32 < 2 ^ 64) by (cbn; lia).
  rewrite (trunc_small 64 (trunc 32 v)) by (cbn; lia).
  rewrite (trunc_small 64 (trunc 32 (v / 2 ^ 32) * 2 ^ 32)) by exact Hm.
  rewrite (trunc_small 64 (trunc 32 (v / 2 ^ 32) * 2 ^ 32)) by exact Hm.
  unfold trunc. cbn [Z.of_nat Pos.of_succ_nat].
  rewrite (Z.mod_small (v / 2 ^ 32)) by (split; [apply Z.div_pos; lia|apply Z.div_lt_upper_bound; lia]).
  rewrite <- Z.shiftl_mul_pow2, <- Z.shiftr_div_pow2 by lia.
  change (2 ^ Z.of_nat 32) with (2 ^ 32).
  rewrite <- Z.land_ones by lia.
  apply Z.bits_inj'. intros i Hi. rewrite Z.lor_spec, Z.land_spec.
  destruct (Z_lt_le_dec i 32) as [Hl|Hl].
  - rewrite Z.ones_spec_low by lia. rewrite Z.shiftl_spec_low by lia.
    rewrite andb_true_r, orb_false_r. reflexivity.
  - rewrite Z.ones_spec_high by lia. rewrite andb_false_r, orb_false_l.
    rewrite Z.shiftl_spec by lia. rewrite Z.shiftr_spec by lia.
    replace (i - 32 + 32) with i by lia. reflexivity.
Qed.

Lemma neg_mod_twice : forall M c, 0 < M -> 0 <= c < M -> (- ((- c) mod M)) mod M = c.
Proof.
  intros M c HM Hc. replace (- ((- c) mod M)) with ((-1) * ((- c) mod M)) by ring.
  rewrite Z.mul_mod_idemp_r by lia. replace (-1 * - c) with c by ring. apply Z.mod_small. exact Hc.
Qed.

Lemma neg_sym : forall w a c, 0 <= a < 2 ^ Z.of_nat w -> 0 <= c < 2 ^ Z.of_nat w ->
  (a = trunc w (- c) <-> c = trunc w (- a)).
Proof.
  intros w a c Ha Hc. unfold trunc. assert (Hp : 0 < 2 ^ Z.of_nat w) by lia.
  split; intros H; subst; symmetry; apply neg_mod_twice; assumption.
Qed.

Lemma shiftr_shiftl_any : forall X sh k, 0 <= k -> Z.shiftr (Z.shiftl X sh) k = Z.shiftl X (sh - k).
Proof.
  intros X sh k Hk. apply Z.bits_inj'. intros i Hi.
  rewrite Z.shiftr_spec by exact Hi. rewrite !Z.shiftl_spec by lia. f_equal. lia.
Qed.

Lemma shamt_nonneg : forall w b, 0 <= shamt w b.
Proof. intros w b. unfold shamt. pose proof (trunc_range 32 b). lia. Qed.

Lemma field_shl : forall X sh mk k M, 0 <= k ->
  Z.land (Z.shiftl (Z.land (Z.shiftl X sh) mk) k) M = Z.land (Z.shiftl X (sh + k)) (Z.land (Z.shiftl mk k) M).
Proof.
  intros X sh mk k M Hk. apply Z.bits_inj'. intros i Hi. rewrite !Z.land_spec.
  destruct (Z_lt_le_dec i k) as [Hl|Hl].
  - rewrite (Z.shiftl_spec_low _ k i Hl), (Z.shiftl_spec_low mk k i Hl).
    destruct (Z.testbit (Z.shiftl X (sh + k)) i); reflexivity.
  - rewrite !Z.shiftl_spec by lia. rewrite Z.land_spec. rewrite !Z.shiftl_spec by lia.
    replace (i - k - sh) with (i - (sh + k)) by lia. destruct (Z.testbit X (i - (sh + k)));
    destruct (Z.testbit mk (i - k)); destruct (Z.testbit M i); reflexivity.
Qed.

Lemma lxor_lt32 : forall a b, 0 <= a < 32 -> 0 <= b < 32 -> 0 <= Z.lxor a b < 32.
Proof.
  intros a b Ha Hb. split; [apply lxor_nonneg; lia|].
  pose proof (lxor_low a b ltac:(lia) Hb) as H. rewrite (Z.div_small a 32) in H by lia.
  assert (0 <= Z.lxor a b) by (apply lxor_nonneg; lia).
  destruct (Z_lt_le_dec (Z.lxor a b) 32) as [Hl|Hl]; [exact Hl|].
  assert (1 <= Z.lxor a b / 32) by (apply Z.div_le_lower_bound; lia). lia.
Qed.

(* ------------------------------------------------------------------ *)
(* Soundness of the transfer of one instruction.                      *)

Section Transfer.

Variable F : lfacts.
Variable tidr : reg.
Variable cols : list (pc * collective).
Variable n : nat.
Hypothesis Hn : (n mod 32 = 0)%nat.

Variables R R' : nat -> reg -> Z.
Hypothesis Hrange : forall x, (x < n)%nat -> in_range F (R x).
Hypothesis Htid : forall x, (x < n)%nat -> R x tidr = Z.of_nat x.

Lemma range_of : forall x r, (x < n)%nat -> 0 <= R x r < 2 ^ Z.of_nat (wdn F r).
Proof. intros x r Hx. exact (Hrange x Hx r). Qed.

Lemma field_of_ok : forall st r f, sstate_ok F n st R -> field_of F tidr st r = Some f ->
  forall x, (x < n)%nat -> R x r = field_value x f.
Proof.
  intros st r f Hok H x Hx. unfold field_of in H. destruct (Pos.eqb_spec r tidr) as [->|Hne].
  - injection H as <-. unfold field_value. cbn [fst snd]. rewrite Z.shiftl_0_r, Htid by exact Hx.
    rewrite Z.land_ones by (unfold wd; lia). symmetry. apply Z.mod_small.
    rewrite <- (Htid x Hx). exact (Hrange x Hx tidr).
  - destruct Hok as [_ Hfacts]. revert H. induction (ss_facts st) as [|g gs IH]; cbn [fold_right]; [discriminate|].
    intros H. assert (Hrest : forall f', In f' gs -> fact_holds F n R f') by (intros f' Hf'; apply Hfacts; right; exact Hf').
    destruct g as [d m s gg|d sh mk|d z|d u sh mk]; try exact (IH Hrest H).
    destruct (Pos.eqb_spec d r) as [->|Hd]; [|exact (IH Hrest H)].
    injection H as <-. exact (Hfacts (SField r sh mk) (or_introl eq_refl) x Hx).
Qed.

Lemma land_ones_trunc : forall w v, trunc w v = Z.land v (ones w).
Proof. intros w v. unfold trunc, ones. rewrite Z.land_ones by lia. reflexivity. Qed.

Lemma field_step_ok : forall st ins d f, sstate_ok F n st R -> dest ins = Some d ->
  field_step F tidr st ins = Some f ->
  (forall x, (x < n)%nat -> R' x d = trunc (wdn F d) (result0 (R x) ins)) ->
  forall x, (x < n)%nat -> R' x d = field_value x f.
Proof.
  intros st ins d f Hok Hd Hf Hval x Hx. rewrite (Hval x Hx). clear Hval.
  destruct ins; cbn [field_step] in Hf; try discriminate Hf;
    destruct g; try discriminate Hf; cbn [dest] in Hd; injection Hd as <-.
  - destruct (field_of F tidr st s) as [[sh mk]|] eqn:Hfo; [|discriminate]. injection Hf as <-.
    unfold result0. cbn [result]. rewrite (field_of_ok st s (sh, mk) Hok Hfo x Hx).
    unfold field_value. cbn [fst snd]. rewrite land_ones_trunc, Z.land_assoc. reflexivity.
  - destruct o; try discriminate Hf; destruct b as [r|c]; try discriminate Hf;
      destruct (field_of F tidr st a) as [[sh mk]|] eqn:Hfo; try discriminate Hf; injection Hf as <-;
      unfold result0; cbn [result binop_eval opv]; rewrite (field_of_ok st a (sh, mk) Hok Hfo x Hx);
      unfold field_value; cbn [fst snd].
    + rewrite <- Z.shiftl_mul_pow2 by (apply shamt_nonneg).
      rewrite ?land_ones_trunc. apply field_shl. apply shamt_nonneg.
    + rewrite <- Z.shiftr_div_pow2 by (apply shamt_nonneg).
      rewrite !land_ones_trunc. rewrite <- Z.land_assoc. rewrite Z.shiftr_land.
      rewrite shiftr_shiftl_any by (apply shamt_nonneg). rewrite <- Z.land_assoc. reflexivity.
    + rewrite !land_ones_trunc. rewrite <- !Z.land_assoc. reflexivity.
  - destruct signed; [discriminate|].
    destruct (field_of F tidr st s) as [[sh mk]|] eqn:Hfo; [|discriminate]. injection Hf as <-.
    unfold result0. cbn [result extend]. rewrite (field_of_ok st s (sh, mk) Hok Hfo x Hx).
    unfold field_value. cbn [fst snd]. rewrite !land_ones_trunc, <- !Z.land_assoc. reflexivity.
Qed.

Lemma fresh_rel : forall d d' m s g, fresh_fact d (SRel d' m s g) = true -> d' = d /\ s <> d.
Proof.
  intros d d' m s g H. unfold fresh_fact in H. cbn [fact_dst] in H. apply andb_true_iff in H.
  destruct H as [H1 H2]. apply Pos.eqb_eq in H1. apply negb_true_iff in H2. split; [exact H1|].
  intros ->. rewrite Pos.eqb_refl in H2. discriminate.
Qed.

Lemma fresh_base : forall d d' u sh mk, fresh_fact d (SBase d' u sh mk) = true -> d' = d /\ u <> d.
Proof.
  intros d d' u sh mk H. unfold fresh_fact in H. cbn [fact_dst] in H. apply andb_true_iff in H.
  destruct H as [H1 H2]. apply Pos.eqb_eq in H1. apply negb_true_iff in H2. split; [exact H1|].
  intros ->. rewrite Pos.eqb_refl in H2. discriminate.
Qed.

Lemma rel_step_ok : forall st ins d, sstate_ok F n st R -> guard_of ins = None -> dest ins = Some d ->
  (forall x, (x < n)%nat -> R' x d = trunc (wdn F d) (result0 (R x) ins)) ->
  (forall x r, (x < n)%nat -> r <> d -> R' x r = R x r) ->
  forall f, In f (rel_step F tidr st ins) -> fresh_fact d f = true -> fact_holds F n R' f.
Proof.
  intros st ins d Hok Hg Hd Hval Hother f Hin Hfresh.
  destruct ins; cbn [rel_step] in Hin; try contradiction; cbn [guard_of] in Hg; subst g;
    cbn [dest] in Hd; injection Hd as <-.
  - (* IMov *)
    assert (Hv : forall x, (x < n)%nat -> R' x d0 = trunc (wdn F d0) (R x s)) by
      (intros x Hx; rewrite (Hval x Hx); reflexivity).
    apply in_app_or in Hin. destruct Hin as [Hin|Hin].
    + destruct (Nat.eqb_spec (wdn F d0) (wdn F s)) as [Hw|Hw]; [|contradiction].
      assert (Hv' : forall x, (x < n)%nat -> R' x d0 = R x s).
      { intros x Hx. rewrite (Hv x Hx). apply trunc_small. rewrite Hw. apply range_of. exact Hx. }
      destruct Hin as [<-|Hin].
      * destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs]. split; [lia|]. intros x Hx.
        rewrite partner_zero, (Hv' x Hx). cbn [gval]. symmetry. apply Hother; assumption.
      * apply in_map_iff in Hin. destruct Hin as [[[m s0] g0] [<- Hin]].
        destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs].
        destruct (rels_of_ok F n st R s m s0 g0 Hok Hin) as [Hm Hr]. split; [exact Hm|].
        intros x Hx. rewrite (Hv' x Hx), (Hr x Hx), Hw.
        rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs). reflexivity.
    + destruct (const_of st s) as [z|] eqn:Hc; [|contradiction]. destruct Hin as [<-|[]].
      intros x Hx. rewrite (Hv x Hx), (const_of_ok F n st R s z Hok Hc x Hx). reflexivity.
  - (* IImm *)
    destruct Hin as [<-|[]]. intros x Hx. rewrite (Hval x Hx). reflexivity.
  - (* IBin *)
    destruct o; try contradiction; destruct b as [b|c]; try contradiction.
    + (* BAdd: a base plus a field of the thread index *)
      assert (Hv : forall x, (x < n)%nat -> R' x d0 = trunc (wdn F d0) (R x a + R x b))
        by (intros x Hx; rewrite (Hval x Hx); reflexivity).
      apply in_app_or in Hin. destruct Hin as [Hin|Hin].
      * destruct (field_of F tidr st b) as [[sh mk]|] eqn:Hfo; [|contradiction]. destruct Hin as [<-|[]].
        destruct (fresh_base _ _ _ _ _ Hfresh) as [_ Hu]. intros x Hx.
        rewrite (Hv x Hx), (Hother x a Hx Hu), (field_of_ok st b (sh, mk) Hok Hfo x Hx). reflexivity.
      * destruct (field_of F tidr st a) as [[sh mk]|] eqn:Hfo; [|contradiction]. destruct Hin as [<-|[]].
        destruct (fresh_base _ _ _ _ _ Hfresh) as [_ Hu]. intros x Hx.
        rewrite (Hv x Hx), (Hother x b Hx Hu), (field_of_ok st a (sh, mk) Hok Hfo x Hx), Z.add_comm.
        reflexivity.
    + (* BShl *)
      destruct (Nat.eqb w 64 && Z.eqb c 32 && Nat.eqb (wdn F d0) 64) eqn:Hw; [|contradiction].
      rewrite !andb_true_iff in Hw. destruct Hw as [[Hw Hc] Hwd]. apply Nat.eqb_eq in Hw, Hwd.
      apply Z.eqb_eq in Hc. subst w c.
      apply in_flat_map in Hin. destruct Hin as [[[m s0] g0] [Hr Hin]].
      destruct (sfun_eqb g0 GHi) eqn:Hg; [|contradiction]. apply sfun_eqb_eq in Hg. subst g0.
      destruct Hin as [<-|[]]. destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs].
      destruct (rels_of_ok F n st R a m s0 GHi Hok Hr) as [Hm Ha]. split; [exact Hm|]. intros x Hx.
      rewrite (Hval x Hx), Hwd. unfold result0. cbn [result binop_eval opv gval].
      rewrite (Ha x Hx). cbn [gval]. change (shamt 64 32) with 32.
      rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs). reflexivity.
    + (* BOr *)
      destruct (Nat.eqb w 64 && Nat.eqb (wdn F d0) 64) eqn:Hw; [|contradiction].
      apply andb_true_iff in Hw. destruct Hw as [Hw Hwd]. apply Nat.eqb_eq in Hw, Hwd. subst w.
      apply in_flat_map in Hin. destruct Hin as [[[m s0] g0] [Hr Hin]].
      destruct (Nat.eqb (wdn F s0) 64 && _) eqn:Hc; [|contradiction]. destruct Hin as [<-|[]].
      destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs].
      apply andb_true_iff in Hc. destruct Hc as [Hws Hc]. apply Nat.eqb_eq in Hws.
      destruct (rels_of_ok F n st R a m s0 g0 Hok Hr) as [Hm Ha]. split; [exact Hm|]. intros x Hx.
      rewrite (Hval x Hx), Hwd. unfold result0. cbn [result binop_eval opv gval].
      rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs).
      assert (HV := range_of (partner x m) s0 (partner_lt n x m Hx Hn Hm)). rewrite Hws in HV.
      apply orb_true_iff in Hc. destruct Hc as [Hc|Hc]; apply andb_true_iff in Hc; destruct Hc as [Hg Hb];
        apply sfun_eqb_eq in Hg; subst g0; destruct (has_rel_ok F n st R b m s0 _ Hok Hb) as [_ Hbv];
        rewrite (Ha x Hx), (Hbv x Hx); cbn [gval].
      * rewrite halves_recompose by exact HV. apply trunc_small. exact HV.
      * rewrite Z.lor_comm. rewrite halves_recompose by exact HV. apply trunc_small. exact HV.
  - (* IMad *)
    destruct b as [b|k]; [contradiction|]. destruct c as [c|c]; [contradiction|].
    destruct (Z.eqb k (-1) && Z.eqb c 0) eqn:Hk; [|contradiction].
    apply andb_true_iff in Hk. destruct Hk as [Hk Hc]. apply Z.eqb_eq in Hk, Hc. subst k c.
    assert (Hv : forall x, (x < n)%nat -> R' x d0 = trunc (wdn F d0) (- R x a)).
    { intros x Hx. rewrite (Hval x Hx). unfold result0. cbn [result opv]. f_equal. ring. }
    destruct Hin as [<-|Hin].
    + destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs]. split; [lia|]. intros x Hx.
      rewrite partner_zero, (Hv x Hx), (Hother x a Hx Hs). reflexivity.
    + apply in_flat_map in Hin. destruct Hin as [[[m s0] g0] [Hr Hin]].
      destruct (sfun_eqb g0 GId) eqn:Hg; [|contradiction]. apply sfun_eqb_eq in Hg. subst g0.
      destruct Hin as [<-|[]]. destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs].
      destruct (rels_of_ok F n st R a m s0 GId Hok Hr) as [Hm Ha]. split; [exact Hm|]. intros x Hx.
      rewrite (Hv x Hx), (Ha x Hx). cbn [gval].
      rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs). reflexivity.
  - (* ICvt *)
    destruct signed; [contradiction|].
    assert (Hv : forall x, (x < n)%nat -> R' x d0 = trunc (wdn F d0) (trunc from (R x s)))
      by (intros x Hx; rewrite (Hval x Hx); reflexivity).
    destruct (Nat.eqb from 64 && Nat.eqb (wdn F d0) 32 && Nat.eqb (wdn F s) 64) eqn:H64.
    + rewrite !andb_true_iff in H64. destruct H64 as [[Hf Hwd] Hws]. apply Nat.eqb_eq in Hf, Hwd, Hws.
      subst from.
      assert (Hv' : forall x, (x < n)%nat -> R' x d0 = trunc 32 (R x s)).
      { intros x Hx. rewrite (Hv x Hx), Hwd. rewrite (trunc_small 64 (R x s)); [reflexivity|].
        rewrite <- Hws. apply range_of. exact Hx. }
      destruct Hin as [<-|Hin].
      * destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs]. split; [lia|]. intros x Hx.
        rewrite partner_zero, (Hv' x Hx), (Hother x s Hx Hs). reflexivity.
      * apply in_flat_map in Hin. destruct Hin as [[[m s0] g0] [Hr Hin]].
        destruct (rels_of_ok F n st R s m s0 g0 Hok Hr) as [Hm Ha].
        destruct (sfun_eqb g0 GId) eqn:Hg; [|destruct (sfun_eqb g0 GHi) eqn:Hg'; [|contradiction]].
        -- apply sfun_eqb_eq in Hg. subst g0. destruct Hin as [<-|[]].
           destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs]. split; [exact Hm|]. intros x Hx.
           rewrite (Hv' x Hx), (Ha x Hx). cbn [gval].
           rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs). reflexivity.
        -- apply sfun_eqb_eq in Hg'. subst g0. destruct Hin as [<-|[]].
           destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs]. split; [exact Hm|]. intros x Hx.
           rewrite (Hv' x Hx), (Ha x Hx). cbn [gval]. unfold trunc. rewrite Z.mod_mod by (apply Z.pow_nonzero; lia).
           rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs). reflexivity.
    + destruct (Nat.eqb from 32 && Nat.leb 32 (wdn F d0) && Nat.eqb (wdn F s) 32) eqn:H32; [|contradiction].
      rewrite !andb_true_iff in H32. destruct H32 as [[Hf Hwd] Hws]. apply Nat.eqb_eq in Hf, Hws.
      apply Nat.leb_le in Hwd. subst from.
      assert (Hv' : forall x, (x < n)%nat -> R' x d0 = R x s).
      { intros x Hx. assert (Hr := range_of x s Hx). rewrite Hws in Hr.
        rewrite (Hv x Hx), (trunc_small 32 (R x s)) by exact Hr. apply trunc_small.
        split; [lia|]. apply (Z.lt_le_trans _ _ _ (proj2 Hr)). apply Z.pow_le_mono_r; lia. }
      apply in_flat_map in Hin. destruct Hin as [[[m s0] g0] [Hr Hin]].
      destruct (sfun_eqb g0 GLo || sfun_eqb g0 GHi) eqn:Hg; [|contradiction]. destruct Hin as [<-|[]].
      destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs].
      destruct (rels_of_ok F n st R s m s0 g0 Hok Hr) as [Hm Ha]. split; [exact Hm|]. intros x Hx.
      rewrite (Hv' x Hx), (Ha x Hx). rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs).
      apply orb_true_iff in Hg. destruct Hg as [Hg|Hg]; apply sfun_eqb_eq in Hg; subst g0; reflexivity.
  - (* IBfe *)
    destruct (Nat.eqb w 64 && Nat.eqb p 32 && Nat.eqb ln 32 && Nat.leb 32 (wdn F d0) && Nat.eqb (wdn F a) 64)
      eqn:Hc; [|contradiction].
    rewrite !andb_true_iff in Hc. destruct Hc as [[[[Hw Hp] Hl] Hwd] Hwa].
    apply Nat.eqb_eq in Hw, Hp, Hl, Hwa. apply Nat.leb_le in Hwd. subst w p ln.
    assert (Hv' : forall x, (x < n)%nat -> R' x d0 = trunc 32 (R x a / 2 ^ 32)).
    { intros x Hx. rewrite (Hval x Hx). unfold result0. cbn [result].
      rewrite (trunc_small 64 (R x a)) by (rewrite <- Hwa; apply range_of; exact Hx).
      change (2 ^ Z.of_nat 32) with (2 ^ 32). apply trunc_small.
      pose proof (trunc_range 32 (R x a / 2 ^ 32)) as Hr. split; [lia|].
      apply (Z.lt_le_trans _ _ _ (proj2 Hr)). apply Z.pow_le_mono_r; lia. }
    destruct Hin as [<-|Hin].
    + destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs]. split; [lia|]. intros x Hx.
      rewrite partner_zero, (Hv' x Hx), (Hother x a Hx Hs). reflexivity.
    + apply in_flat_map in Hin. destruct Hin as [[[m s0] g0] [Hr Hin]].
      destruct (sfun_eqb g0 GId) eqn:Hg; [|contradiction]. apply sfun_eqb_eq in Hg. subst g0.
      destruct Hin as [<-|[]]. destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hs].
      destruct (rels_of_ok F n st R a m s0 GId Hok Hr) as [Hm Ha]. split; [exact Hm|]. intros x Hx.
      rewrite (Hv' x Hx), (Ha x Hx). cbn [gval].
      rewrite (Hother (partner x m) s0 (partner_lt n x m Hx Hn Hm) Hs). reflexivity.
Qed.

Definition masks_valid (f : nat -> Z) (M : Z) : Prop :=
  forall D, (forall x y, (x < n)%nat -> (y < n)%nat -> agree D x y -> f x = f y) ->
  forall x y, (x < n)%nat -> (y < n)%nat -> agree (Z.land D (Z.lnot M)) x y -> f x = f y.

Lemma masks_valid_zero : forall f, masks_valid f 0.
Proof. intros f D H x y Hx Hy Ha. apply H; try assumption. rewrite Z.lnot_0, Z.land_m1_r in Ha. exact Ha. Qed.

Lemma masks_valid_add : forall f M m, masks_valid f M -> single_lane m = true ->
  (forall x, (x < n)%nat -> f (partner x m) = f x) -> masks_valid f (Z.lor M m).
Proof.
  intros f M m HM Hm Hinv D H x y Hx Hy Ha.
  rewrite Z.lnot_lor, Z.land_assoc in Ha.
  exact (reduce_single n f (Z.land D (Z.lnot M)) m Hm Hn (HM D H) Hinv x y Hx Hy Ha).
Qed.

Lemma fold_masks_valid : forall f (A : Type) (l : list A) (step : Z -> A -> Z) acc,
  masks_valid f acc ->
  (forall acc e, In e l -> step acc e = acc \/
     exists m, step acc e = Z.lor acc m /\ single_lane m = true /\ forall x, (x < n)%nat -> f (partner x m) = f x) ->
  masks_valid f (fold_left step l acc).
Proof.
  intros f A l step. induction l as [|e l IH]; intros acc Hacc H; cbn [fold_left]; [exact Hacc|].
  apply IH; [|intros acc' e' He'; apply H; right; exact He'].
  destruct (H acc e (or_introl eq_refl)) as [->|[m [-> [Hm Hinv]]]]; [exact Hacc|].
  apply masks_valid_add; assumption.
Qed.

Lemma swap_masks_valid : forall st a b (op : Z -> Z -> Z), sstate_ok F n st R ->
  (forall u v, op u v = op v u) -> masks_valid (fun x => op (R x a) (R x b)) (swap_masks st a b).
Proof.
  intros st a b op Hok Hc. unfold swap_masks.
  apply fold_masks_valid; [apply fold_masks_valid; [apply masks_valid_zero|]|].
  - intros acc [[m s] g] Hin. destruct (sfun_eqb g GId && single_lane m && Pos.eqb s b) eqn:Hsel; [|left; reflexivity].
    right. exists m. split; [reflexivity|]. rewrite !andb_true_iff in Hsel. destruct Hsel as [[Hg Hm] Hs].
    apply sfun_eqb_eq in Hg. apply Pos.eqb_eq in Hs. subst g s. split; [exact Hm|].
    destruct (rels_of_ok F n st R a m b GId Hok Hin) as [Hm' Ha]. intros x Hx. cbn [gval] in Ha.
    assert (Hp : (partner x m < n)%nat) by (apply partner_lt; assumption).
    rewrite (Ha x Hx), (Ha (partner x m) Hp), partner_self by lia. apply Hc.
  - intros acc [[m s] g] Hin. destruct (sfun_eqb g GId && single_lane m && Pos.eqb s a) eqn:Hsel; [|left; reflexivity].
    right. exists m. split; [reflexivity|]. rewrite !andb_true_iff in Hsel. destruct Hsel as [[Hg Hm] Hs].
    apply sfun_eqb_eq in Hg. apply Pos.eqb_eq in Hs. subst g s. split; [exact Hm|].
    destruct (rels_of_ok F n st R b m a GId Hok Hin) as [Hm' Hb]. intros x Hx. cbn [gval] in Hb.
    assert (Hp : (partner x m < n)%nat) by (apply partner_lt; assumption).
    rewrite (Hb x Hx), (Hb (partner x m) Hp), partner_self by lia. apply Hc.
Qed.

Lemma eq_cmp : forall c w sg u v, 0 <= u < 2 ^ Z.of_nat w -> 0 <= v < 2 ^ Z.of_nat w ->
  (c = CEq \/ c = CNe) ->
  cmp_eval c (extend w sg u) (extend w sg v) = if (match c with CEq => true | _ => false end)
                                               then Z.eqb u v else negb (Z.eqb u v).
Proof.
  intros c w sg u v Hu Hv Hc.
  assert (E : Z.eqb (extend w sg u) (extend w sg v) = Z.eqb u v).
  { destruct (Z.eqb_spec u v) as [->|Hne]; [apply Z.eqb_refl|].
    apply Z.eqb_neq. intros He. apply Hne. exact (extend_inj w sg u v Hu Hv He). }
  destruct Hc as [->| ->]; cbn [cmp_eval]; rewrite E; reflexivity.
Qed.

Lemma neg_masks_valid : forall st a b c w (out : bool -> Z), sstate_ok F n st R ->
  wdn F a = w -> wdn F b = w -> (c = CEq \/ c = CNe) ->
  forall sg, masks_valid (fun x => out (cmp_eval c (extend w sg (R x a)) (extend w sg (R x b)))) (neg_masks st a b).
Proof.
  intros st a b c w out Hok Ha Hb Hc sg.
  assert (Hra : forall x, (x < n)%nat -> 0 <= R x a < 2 ^ Z.of_nat w) by (intros x Hx; rewrite <- Ha; apply range_of; exact Hx).
  assert (Hrb : forall x, (x < n)%nat -> 0 <= R x b < 2 ^ Z.of_nat w) by (intros x Hx; rewrite <- Hb; apply range_of; exact Hx).
  assert (Hval : forall x, (x < n)%nat -> cmp_eval c (extend w sg (R x a)) (extend w sg (R x b)) =
    if (match c with CEq => true | _ => false end) then Z.eqb (R x a) (R x b) else negb (Z.eqb (R x a) (R x b)))
    by (intros x Hx; apply eq_cmp; auto).
  unfold neg_masks.
  apply fold_masks_valid; [apply fold_masks_valid; [apply masks_valid_zero|]|].
  - intros acc [[m s] g] Hin. destruct (sfun_eqb g GNeg && single_lane m && Pos.eqb s b) eqn:Hsel; [|left; reflexivity].
    right. exists m. split; [reflexivity|]. rewrite !andb_true_iff in Hsel. destruct Hsel as [[Hg Hm] Hs].
    apply sfun_eqb_eq in Hg. apply Pos.eqb_eq in Hs. subst g s. split; [exact Hm|].
    destruct (rels_of_ok F n st R a m b GNeg Hok Hin) as [Hm' Hav]. intros x Hx. cbn [gval] in Hav.
    assert (Hp : (partner x m < n)%nat) by (apply partner_lt; assumption).
    rewrite (Hval _ Hp), (Hval x Hx). rewrite (Hav x Hx), (Hav (partner x m) Hp), partner_self, Ha by lia.
    assert (E : Z.eqb (trunc w (- R x b)) (R (partner x m) b) = Z.eqb (trunc w (- R (partner x m) b)) (R x b)).
    { rewrite (Z.eqb_sym (trunc w (- R (partner x m) b))).
      destruct (Z.eqb_spec (trunc w (- R x b)) (R (partner x m) b)) as [He|He];
      destruct (Z.eqb_spec (R x b) (trunc w (- R (partner x m) b))) as [He'|He']; try reflexivity.
      - exfalso. apply He'. apply (neg_sym w (R x b) (R (partner x m) b) (Hrb x Hx) (Hrb _ Hp)). symmetry. exact He.
      - exfalso. apply He. symmetry. apply (neg_sym w (R x b) (R (partner x m) b) (Hrb x Hx) (Hrb _ Hp)). exact He'. }
    rewrite E. reflexivity.
  - intros acc [[m s] g] Hin. destruct (sfun_eqb g GNeg && single_lane m && Pos.eqb s a) eqn:Hsel; [|left; reflexivity].
    right. exists m. split; [reflexivity|]. rewrite !andb_true_iff in Hsel. destruct Hsel as [[Hg Hm] Hs].
    apply sfun_eqb_eq in Hg. apply Pos.eqb_eq in Hs. subst g s. split; [exact Hm|].
    destruct (rels_of_ok F n st R b m a GNeg Hok Hin) as [Hm' Hbv]. intros x Hx. cbn [gval] in Hbv.
    assert (Hp : (partner x m < n)%nat) by (apply partner_lt; assumption).
    rewrite (Hval _ Hp), (Hval x Hx). rewrite (Hbv x Hx), (Hbv (partner x m) Hp), partner_self, Hb by lia.
    assert (E : Z.eqb (R (partner x m) a) (trunc w (- R x a)) = Z.eqb (R x a) (trunc w (- R (partner x m) a))).
    { destruct (Z.eqb_spec (R (partner x m) a) (trunc w (- R x a))) as [He|He];
      destruct (Z.eqb_spec (R x a) (trunc w (- R (partner x m) a))) as [He'|He']; try reflexivity.
      - exfalso. apply He'. apply (neg_sym w (R x a) (R (partner x m) a) (Hra x Hx) (Hra _ Hp)). exact He.
      - exfalso. apply He. apply (neg_sym w (R x a) (R (partner x m) a) (Hra x Hx) (Hra _ Hp)). exact He'. }
    rewrite E. reflexivity.
Qed.


Lemma masks_valid_ext : forall f g M, (forall x, (x < n)%nat -> f x = g x) -> masks_valid f M -> masks_valid g M.
Proof.
  intros f g M He H D HD x y Hx Hy Ha. rewrite <- (He x Hx), <- (He y Hy).
  apply (H D); try assumption. intros u v Hu Hv Hag. rewrite (He u Hu), (He v Hv). exact (HD u v Hu Hv Hag).
Qed.

Lemma field_step_guard : forall st ins f, field_step F tidr st ins = Some f -> guard_of ins = None.
Proof. intros st ins f H. destruct ins; try discriminate H; destruct g; try discriminate H; reflexivity. Qed.

Lemma pure_dep_ok : forall st ins d, sstate_ok F n st R -> pure_exact ins = true -> dest ins = Some d ->
  (forall x, (x < n)%nat -> R' x d = if holds (guard_of ins) (R x) then trunc (wdn F d) (result0 (R x) ins) else R x d) ->
  dep_spec n (pure_dep F tidr st ins d) R' d.
Proof.
  intros st ins d Hok Hp Hd Hval.
  set (base0 := dep_regs st (instruction_inputs ins)).
  set (base := match guard_of ins with Some _ => dunion base0 (dep_of st d) | None => base0 end).
  assert (Hbase : dep_spec n base R' d).
  { intros z Hz x y Hx Hy Ha. rewrite (Hval x Hx), (Hval y Hy). unfold base in Hz.
    destruct (guard_of ins) as [gg|] eqn:Hg.
    - unfold dunion in Hz. destruct base0 as [z0|] eqn:Hb0; [|discriminate].
      destruct (dep_of st d) as [zd|] eqn:Hzd; [|discriminate]. injection Hz as <-.
      destruct (agree_lor _ _ _ _ Ha) as [Ha0 Had].
      assert (Hin : forall r, In r (instruction_inputs ins) -> R x r = R y r)
        by exact (dep_regs_ok F n st R _ z0 Hok Hb0 x y Hx Hy Ha0).
      rewrite <- Hg, (holds_inputs ins (R x) (R y) Hin), Hg.
      destruct (holds (Some gg) (R y)).
      + rewrite (result_inputs ins (R x) (R y) Hp Hin). reflexivity.
      + exact (proj1 Hok d zd x y Hzd Hx Hy Had).
    - assert (Hin : forall r, In r (instruction_inputs ins) -> R x r = R y r)
        by exact (dep_regs_ok F n st R _ z Hok Hz x y Hx Hy Ha).
      cbn [holds]. rewrite (result_inputs ins (R x) (R y) Hp Hin). reflexivity. }
  assert (Hunguarded : guard_of ins = None ->
    forall x, (x < n)%nat -> R' x d = trunc (wdn F d) (result0 (R x) ins)).
  { intros Hg x Hx. rewrite (Hval x Hx), Hg. reflexivity. }
  unfold pure_dep. fold base0. fold base.
  destruct (field_step F tidr st ins) as [f|] eqn:Hf.
  - intros z Hz. injection Hz as <-. intros x y Hx Hy Ha.
    pose proof (field_step_guard st ins f Hf) as Hg.
    rewrite (field_step_ok st ins d f Hok Hd Hf (Hunguarded Hg) x Hx).
    rewrite (field_step_ok st ins d f Hok Hd Hf (Hunguarded Hg) y Hy).
    apply field_agree. exact Ha.
  - assert (Hred : forall M, guard_of ins = None -> masks_valid (fun x => R' x d) M ->
              dep_spec n (reduce base M) R' d).
    { intros M Hg HM z Hz x y Hx Hy Ha. unfold reduce in Hz. destruct base as [zb|] eqn:Hb; [|discriminate].
      injection Hz as <-. apply (HM zb); try assumption. intros u v Hu Hv Hag. exact (Hbase zb eq_refl u v Hu Hv Hag). }
    destruct ins; try exact Hbase; destruct g; try exact Hbase; cbn [dest] in Hd; injection Hd as <-.
    + destruct b as [b|]; [|exact Hbase]. destruct (commutative o) eqn:Hco; [|exact Hbase].
      apply Hred; [reflexivity|].
      apply (masks_valid_ext (fun x => trunc (wdn F d0) (binop_eval o w (R x a) (R x b))));
        [intros x Hx; rewrite (Hunguarded eq_refl x Hx); reflexivity|].
      apply (swap_masks_valid st a b (fun u v => trunc (wdn F d0) (binop_eval o w u v)) Hok).
      intros u v. rewrite binop_comm by exact Hco. reflexivity.
    + destruct b as [b|]; [|exact Hbase].
      destruct c; try exact Hbase;
        (destruct (Nat.eqb (wdn F a) w && Nat.eqb (wdn F b) w) eqn:Hw; [|exact Hbase]);
        apply Hred; [reflexivity| |reflexivity|];
        apply andb_true_iff in Hw; destruct Hw as [Hwa Hwb]; apply Nat.eqb_eq in Hwa, Hwb.
      * apply (masks_valid_ext (fun x => trunc (wdn F d0) (bool_z (cmp_eval CEq (extend w signed (R x a))
                                                                  (extend w signed (R x b))))));
          [intros x Hx; rewrite (Hunguarded eq_refl x Hx); reflexivity|].
        exact (neg_masks_valid st a b CEq w (fun v => trunc (wdn F d0) (bool_z v)) Hok Hwa Hwb (or_introl eq_refl) signed).
      * apply (masks_valid_ext (fun x => trunc (wdn F d0) (bool_z (cmp_eval CNe (extend w signed (R x a))
                                                                  (extend w signed (R x b))))));
          [intros x Hx; rewrite (Hunguarded eq_refl x Hx); reflexivity|].
        exact (neg_masks_valid st a b CNe w (fun v => trunc (wdn F d0) (bool_z v)) Hok Hwa Hwb (or_intror eq_refl) signed).
    + destruct b as [b|]; [|exact Hbase]. apply Hred; [reflexivity|].
      apply (masks_valid_ext (fun x => trunc (wdn F d0)
        (if is_min then Z.min (extend w signed (R x a)) (extend w signed (R x b))
         else Z.max (extend w signed (R x a)) (extend w signed (R x b)))));
        [intros x Hx; rewrite (Hunguarded eq_refl x Hx); reflexivity|].
      apply (swap_masks_valid st a b (fun u v => trunc (wdn F d0)
        (if is_min then Z.min (extend w signed u) (extend w signed v)
         else Z.max (extend w signed u) (extend w signed v))) Hok).
      intros u v. destruct is_min; [rewrite Z.min_comm|rewrite Z.max_comm]; reflexivity.
Qed.


Lemma pure_dest : forall ins, pure_exact ins = true -> exists d, dest ins = Some d.
Proof. intros ins H. destruct ins; try discriminate H; eexists; reflexivity. Qed.

Lemma cell_regs_is_cell : forall r, existsb (Pos.eqb r) (cell_regs F) = false -> is_cell F r = false.
Proof.
  intros r H. unfold is_cell. unfold cell_regs in H. induction (lf_cells F) as [|[[p u] c] cs IH]; [reflexivity|].
  cbn [existsb map] in *. apply orb_false_iff in H. destruct H as [H1 H2].
  rewrite Pos.eqb_sym in H1. rewrite H1. apply IH. exact H2.
Qed.

Lemma mask_ok_full : forall st mask, sstate_ok F n st R -> mask_ok st mask = true ->
  match mask with Some r => forall x, (x < n)%nat -> R x r = full_mask | None => True end.
Proof.
  intros st [r|] Hok H; [|exact I]. unfold mask_ok in H. destruct (const_of st r) as [z|] eqn:Hc; [|discriminate].
  apply Z.eqb_eq in H. subst z. exact (const_of_ok F n st R r full_mask Hok Hc).
Qed.

Lemma transfer_ok : forall dirty p s st ins (A V : nat -> Z),
  sstate_ok F n st R ->
  (pure_exact ins = true -> forall d, dest ins = Some d -> forall x, (x < n)%nat ->
     R' x = if holds (guard_of ins) (R x) then upd (wdn F) (R x) d (result0 (R x) ins) else R x) ->
  (forall d, ins = IDef None [d] -> forall x, (x < n)%nat -> R' x = upd (wdn F) (R x) d (A x)) ->
  (forall bits d a off, ins = ILdS None bits d a off -> forall x, (x < n)%nat ->
     R' x = upd (wdn F) (R x) d (trunc bits (V x))) ->
  (forall x r, (x < n)%nat -> existsb (Pos.eqb r) (loop_writes ins) = false ->
     (writes_cells ins = false \/ is_cell F r = false) -> R' x r = R x r) ->
  (forall d src m mask, ins = IDef None [d] -> collective_at cols p = Some (CBfly d src m mask) ->
     mask_ok st mask = true -> 0 <= m < 32 ->
     forall x, (x < n)%nat -> trunc 32 (A x) = trunc 32 (R (partner x m) src)) ->
  (forall d src mask, ins = IDef None [d] -> collective_at cols p = Some (CRedux d src mask) ->
     mask_ok st mask = true -> forall x y, (x < n)%nat -> (y < n)%nat -> (x / 32 = y / 32)%nat ->
     trunc 32 (A x) = trunc 32 (A y)) ->
  (forall bits d a off, ins = ILdS None bits d a off -> Z.land s dirty = 0 ->
     forall x y, (x < n)%nat -> (y < n)%nat -> R x a = R y a -> trunc bits (V x) = trunc bits (V y)) ->
  sstate_ok F n (ss_transfer F tidr cols dirty p s st ins) R'.
Proof.
  intros dirty p s st ins A V Hok Hpure Hdef Hlds Hkeep Hbfly Hredux Hload.
  unfold ss_transfer. destruct (pure_exact ins) eqn:Hp.
  - destruct (pure_dest ins Hp) as [d Hd]. rewrite Hd.
    assert (HR := Hpure eq_refl d Hd).
    assert (Hother : forall x r, (x < n)%nat -> r <> d -> R' x r = R x r).
    { intros x r Hx Hr. rewrite (HR x Hx). destruct (holds (guard_of ins) (R x)); [|reflexivity].
      unfold upd. apply Pos.eqb_neq in Hr. rewrite Hr. reflexivity. }
    assert (Hval : forall x, (x < n)%nat -> R' x d =
      if holds (guard_of ins) (R x) then trunc (wdn F d) (result0 (R x) ins) else R x d).
    { intros x Hx. rewrite (HR x Hx). destruct (holds (guard_of ins) (R x)); [|reflexivity].
      unfold upd. rewrite Pos.eqb_refl. reflexivity. }
    apply (ss_write_ok F n st d _ _ R R' Hn Hok Hother (pure_dep_ok st ins d Hok Hp Hd Hval)).
    intros f Hin Hfresh. destruct (guard_of ins) as [gg|] eqn:Hg; [contradiction|].
    assert (Hval' : forall x, (x < n)%nat -> R' x d = trunc (wdn F d) (result0 (R x) ins)).
    { intros x Hx. rewrite (Hval x Hx). reflexivity. }
    destruct (field_step F tidr st ins) as [ff|] eqn:Hf.
    + destruct Hin as [<-|Hin].
      * intros x Hx. exact (field_step_ok st ins d ff Hok Hd Hf Hval' x Hx).
      * exact (rel_step_ok st ins d Hok Hg Hd Hval' Hother f Hin Hfresh).
    + exact (rel_step_ok st ins d Hok Hg Hd Hval' Hother f Hin Hfresh).
  - assert (Htop : forall rs, (forall x r, (x < n)%nat -> existsb (Pos.eqb r) rs = false -> R' x r = R x r) ->
      sstate_ok F n (ss_top st rs) R') by (intros rs H; exact (ss_top_ok F n st rs R R' Hn Hok H)).
    assert (Hgen : sstate_ok F n (ss_top st (loop_writes ins ++ (if writes_cells ins then cell_regs F else []))) R').
    { apply Htop. intros x r Hx Hr. rewrite existsb_app in Hr. apply orb_false_iff in Hr. destruct Hr as [Hr1 Hr2].
      apply Hkeep; [exact Hx|exact Hr1|]. destruct (writes_cells ins); [right|left; reflexivity].
      apply cell_regs_is_cell. exact Hr2. }
    assert (Hwrite1 : forall d dep fs, (forall x, (x < n)%nat -> R' x = upd (wdn F) (R x) d (A x)) ->
      dep_spec n dep R' d -> (forall f, In f fs -> fresh_fact d f = true -> fact_holds F n R' f) ->
      sstate_ok F n (ss_write st d dep fs) R').
    { intros d dep fs HA Hdep Hfs. apply (ss_write_ok F n st d dep fs R R' Hn Hok); [|exact Hdep|exact Hfs].
      intros x r Hx Hr. rewrite (HA x Hx). unfold upd. apply Pos.eqb_neq in Hr. rewrite Hr. reflexivity. }
    assert (Hnone : forall d, (forall x, (x < n)%nat -> R' x = upd (wdn F) (R x) d (A x)) ->
      sstate_ok F n (ss_write st d None []) R').
    { intros d HA. apply (Hwrite1 d None [] HA); [intros z Hz; discriminate|intros f []]. }
    destruct ins; try discriminate Hp; try exact Hgen.
    + (* IDef *)
      destruct g; [exact Hgen|]. destruct ds as [|d [|d' ds]]; try exact Hgen.
      specialize (Hdef d eq_refl). destruct (collective_at cols p) as [[d1 src mask|d1 src m mask]|] eqn:Hc.
      * destruct (Pos.eqb d1 d && Nat.eqb (wdn F d) 32 && mask_ok st mask) eqn:Hcond; [|exact (Hnone d Hdef)].
        rewrite !andb_true_iff in Hcond. destruct Hcond as [[Hd1 Hw] Hm]. apply Pos.eqb_eq in Hd1. subst d1.
        apply Nat.eqb_eq in Hw. apply (Hwrite1 d _ _ Hdef); [|intros f []].
        intros z Hz. injection Hz as <-. intros x y Hx Hy Ha.
        rewrite (Hdef x Hx), (Hdef y Hy). unfold upd. rewrite Pos.eqb_refl, Hw.
        exact (Hredux d src mask eq_refl eq_refl Hm x y Hx Hy (agree_warp x y Ha)).
      * destruct (Pos.eqb d1 d && negb (Pos.eqb src d) && (0 <=? m) && (m <? 32) &&
                  Nat.eqb (wdn F d) 32 && Nat.eqb (wdn F src) 32 && mask_ok st mask) eqn:Hcond;
          [|exact (Hnone d Hdef)].
        rewrite !andb_true_iff in Hcond. destruct Hcond as [[[[[[Hd1 Hsd] Hm0] Hm32] Hw] Hws] Hm].
        apply Pos.eqb_eq in Hd1. subst d1. apply negb_true_iff, Pos.eqb_neq in Hsd.
        apply Z.leb_le in Hm0. apply Z.ltb_lt in Hm32. apply Nat.eqb_eq in Hw, Hws.
        assert (HB := Hbfly d src m mask eq_refl eq_refl Hm (conj Hm0 Hm32)).
        assert (Hsrc : forall x, (x < n)%nat -> R' x d = R (partner x m) src).
        { intros x Hx. rewrite (Hdef x Hx). unfold upd. rewrite Pos.eqb_refl, Hw.
          rewrite (HB x Hx). apply trunc_small. rewrite <- Hws. apply range_of. apply partner_lt; auto. }
        assert (Hsame : forall x r, (x < n)%nat -> r <> d -> R' x r = R x r).
        { intros x r Hx Hr. rewrite (Hdef x Hx). unfold upd. apply Pos.eqb_neq in Hr. rewrite Hr. reflexivity. }
        apply (Hwrite1 d _ _ Hdef).
        -- intros z Hz x y Hx Hy Ha. rewrite (Hsrc x Hx), (Hsrc y Hy).
           apply (proj1 Hok src z); try exact Hz; try (apply partner_lt; auto).
           apply agree_partner; [lia|exact Ha].
        -- intros f Hin Hfresh. destruct Hin as [<-|Hin].
           ++ split; [lia|]. intros x Hx. rewrite (Hsrc x Hx). cbn [gval].
              symmetry. apply Hsame; [apply partner_lt; auto|exact Hsd].
           ++ apply in_app_or in Hin. destruct Hin as [Hin|Hin].
              ** apply in_map_iff in Hin. destruct Hin as [[[m0 q] g] [<- Hin]].
                 destruct (fresh_rel _ _ _ _ _ Hfresh) as [_ Hq].
                 destruct (rels_of_ok F n st R src m0 q g Hok Hin) as [Hm0' Hr].
                 split; [apply lxor_lt32; lia|]. intros x Hx.
                 assert (Hpx : (partner x m < n)%nat) by (apply partner_lt; auto; lia).
                 rewrite (Hsrc x Hx), (Hr _ Hpx), Hw, Hws, partner_partner by lia.
                 rewrite (Hsame (partner x (Z.lxor m m0)) q); [reflexivity| |exact Hq].
                 apply partner_lt; auto. apply lxor_lt32; lia.
              ** destruct (const_of st src) as [z|] eqn:Hz; [|contradiction]. destruct Hin as [<-|[]].
                 intros x Hx. rewrite (Hsrc x Hx).
                 rewrite (const_of_ok F n st R src z Hok Hz _ (partner_lt n x m Hx Hn (conj Hm0 Hm32))).
                 symmetry. apply trunc_small. rewrite <- (const_of_ok F n st R src z Hok Hz x Hx), <- Hws.
                 apply range_of. exact Hx.
      * exact (Hnone d Hdef).
    + (* ILdS *)
      destruct g; [exact Hgen|].
      specialize (Hlds bits d a off eq_refl).
      assert (Hsame : forall x r, (x < n)%nat -> r <> d -> R' x r = R x r).
      { intros x r Hx Hr. rewrite (Hlds x Hx). unfold upd. apply Pos.eqb_neq in Hr. rewrite Hr. reflexivity. }
      destruct (Z.eqb (Z.land s dirty) 0) eqn:Hclean.
      * apply Z.eqb_eq in Hclean.
        apply (ss_write_ok F n st d _ _ R R' Hn Hok Hsame); [|intros f []].
        intros z Hz x y Hx Hy Ha. rewrite (Hlds x Hx), (Hlds y Hy). unfold upd. rewrite Pos.eqb_refl.
        rewrite (Hload bits d a off eq_refl Hclean x y Hx Hy (proj1 Hok a z x y Hz Hx Hy Ha)). reflexivity.
      * apply (ss_write_ok F n st d None [] R R' Hn Hok Hsame); [intros z Hz; discriminate|intros f []].
Qed.

End Transfer.

(* ------------------------------------------------------------------ *)
(* The walks, position by position.                                   *)

Lemma zsub_bits : forall a b, zsub a b = true <-> forall i, 0 <= i -> Z.testbit a i = true -> Z.testbit b i = true.
Proof.
  intros a b. unfold zsub. rewrite Z.eqb_eq. split.
  - intros H i Hi Ha. assert (B := f_equal (fun v => Z.testbit v i) H). cbn beta in B.
    rewrite Z.land_spec, Z.lnot_spec, Ha, Z.testbit_0_l in B by exact Hi.
    destruct (Z.testbit b i); [reflexivity|discriminate].
  - intros H. apply Z.bits_inj'. intros i Hi. rewrite Z.land_spec, Z.lnot_spec, Z.testbit_0_l by exact Hi.
    destruct (Z.testbit a i) eqn:Ha; [rewrite (H i Hi Ha)|]; reflexivity.
Qed.

Lemma zsub_refl : forall z, zsub z z = true.
Proof. intros z. apply zsub_bits. auto. Qed.

Lemma zsub_trans : forall a b c, zsub a b = true -> zsub b c = true -> zsub a c = true.
Proof. intros a b c H1 H2. rewrite zsub_bits in *. auto. Qed.

Lemma zsub_lor : forall a b, zsub a (Z.lor b a) = true /\ zsub b (Z.lor b a) = true.
Proof.
  intros a b. split; apply zsub_bits; intros i Hi H; rewrite Z.lor_spec, H; [apply orb_true_r|reflexivity].
Qed.

Section Walk.

Variable F : lfacts.
Variable tidr : reg.
Variable cols : list (pc * collective).
Variable cta : list pc.
Variable dirty : Z.

Fixpoint phase_iter (b j : nat) (s : Z) (k : nat) : Z :=
  match k with O => s | S k' => phase_iter b (S j) (phase_next cta (b, j) s) k' end.

Fixpoint ss_iter (b j : nat) (s : Z) (st : sstate) (body : list instr) (k : nat) {struct k} : sstate :=
  match k, body with
  | S k', ins :: rest => ss_iter b (S j) (phase_next cta (b, j) s) (ss_transfer F tidr cols dirty (b, j) s st ins) rest k'
  | _, _ => st
  end.

Lemma phase_iter_succ : forall k b j s, phase_iter b j s (S k) = phase_next cta (b, (j + k)%nat) (phase_iter b j s k).
Proof.
  induction k as [|k IH]; intros b j s; [cbn; rewrite Nat.add_0_r; reflexivity|].
  change (phase_iter b j s (S (S k))) with (phase_iter b (S j) (phase_next cta (b, j) s) (S k)).
  rewrite IH. replace (S j + k)%nat with (j + S k)%nat by lia. reflexivity.
Qed.

Lemma ss_iter_succ : forall body k b j s st ins, nth_error body k = Some ins ->
  ss_iter b j s st body (S k) =
  ss_transfer F tidr cols dirty (b, (j + k)%nat) (phase_iter b j s k) (ss_iter b j s st body k) ins.
Proof.
  induction body as [|i0 body IH]; intros k b j s st ins H; [destruct k; discriminate|].
  destruct k as [|k].
  - cbn in H. injection H as <-. cbn. rewrite Nat.add_0_r. reflexivity.
  - cbn in H. change (ss_iter b j s st (i0 :: body) (S (S k))) with
      (ss_iter b (S j) (phase_next cta (b, j) s) (ss_transfer F tidr cols dirty (b, j) s st i0) body (S k)).
    rewrite (IH k b (S j) _ _ ins H). replace (S j + k)%nat with (j + S k)%nat by lia. reflexivity.
Qed.

Lemma ss_walk_spec : forall body b j s st edges st' edges',
  ss_walk F tidr cols cta dirty b j s st body edges = Some (st', edges') ->
  st' = ss_iter b j s st body (length body) /\ (forall e, In e edges -> In e edges') /\
  forall k ins, nth_error body k = Some ins ->
    branch_ok (ss_iter b j s st body k) ins = true /\
    (forall tg, jump_target ins = Some tg -> In (tg, ss_iter b j s st body (S k)) edges').
Proof.
  induction body as [|ins body IH]; intros b j s st edges st' edges' H; cbn [ss_walk] in H.
  - injection H as <- <-. split; [reflexivity|]. split; [auto|]. intros k ins Hk. destruct k; discriminate.
  - destruct (branch_ok st ins) eqn:Hb; [|discriminate]. cbn [negb] in H.
    destruct (IH _ _ _ _ _ _ _ H) as [Hst [Hinc Hk]]. split; [exact Hst|]. split.
    + intros e He. apply Hinc. destruct (jump_target ins); [right|]; exact He.
    + intros k i Hki. destruct k as [|k].
      * cbn in Hki. injection Hki as <-. split; [exact Hb|]. intros tg Htg. apply Hinc. rewrite Htg. left. reflexivity.
      * cbn in Hki. exact (Hk k i Hki).
Qed.

Lemma phase_walk_spec : forall body b j s edges d0 s' edges' d',
  phase_walk cta b j s body edges d0 = (s', edges', d') ->
  s' = phase_iter b j s (length body) /\ (forall e, In e edges -> In e edges') /\ zsub d0 d' = true /\
  forall k ins, nth_error body k = Some ins ->
    (forall tg, jump_target ins = Some tg -> In (tg, phase_iter b j s (S k)) edges') /\
    (shared_write ins = true -> zsub (phase_iter b j s k) d' = true).
Proof.
  pose proof zsub_refl as Hsub_refl. pose proof zsub_trans as Hsub_trans. pose proof zsub_lor as Hsub_lor.
  induction body as [|ins body IH]; intros b j s edges d0 s' edges' d' H; cbn [phase_walk] in H.
  - injection H as <- <- <-. split; [reflexivity|]. split; [auto|]. split; [apply Hsub_refl|].
    intros k ins Hk. destruct k; discriminate.
  - destruct (IH _ _ _ _ _ _ _ _ H) as [Hs [Hinc [Hd Hk]]]. split; [exact Hs|]. split.
    + intros e He. apply Hinc. destruct (jump_target ins); [right|]; exact He.
    + split.
      * eapply Hsub_trans; [|exact Hd]. destruct (shared_write ins); [apply Hsub_lor|apply Hsub_refl].
      * intros k i Hki. destruct k as [|k].
        -- cbn in Hki. injection Hki as <-. split.
           ++ intros tg Htg. apply Hinc. rewrite Htg. left. reflexivity.
           ++ intros Hw. rewrite Hw in Hd. cbn [phase_iter]. eapply Hsub_trans; [|exact Hd]. apply Hsub_lor.
        -- cbn in Hki. exact (Hk k i Hki).
Qed.

End Walk.

(* ------------------------------------------------------------------ *)
(* Barrier phases of one thread.                                       *)

Lemma last_instr_nth : forall (body : list instr), last_instr body = nth_error body (pred (length body)).
Proof.
  intros body. unfold last_instr. induction body as [|a l IH]; [reflexivity|].
  destruct l as [|b l]; [reflexivity|]. cbn [rev]. cbn [rev] in IH.
  destruct (rev l ++ [b]) as [|c r] eqn:Hr.
  - apply app_eq_nil in Hr. destruct Hr as [_ Hr]. discriminate.
  - cbn. rewrite IH. reflexivity.
Qed.

Lemma fetch_in : forall prog b i ins, fetch prog b i = Some ins -> In ins (concat prog).
Proof.
  intros prog b i ins H. unfold fetch, block_body in H. destruct (nth_error prog b) as [body|] eqn:Hb;
    [|destruct i; discriminate]. apply nth_error_In in H. apply nth_error_In in Hb.
  apply in_concat. exists body. split; assumption.
Qed.

Lemma fetch_block : forall prog b i ins, fetch prog b i = Some ins -> (b < length prog)%nat.
Proof.
  intros prog b i ins H. unfold fetch, block_body in H. destruct (nth_error prog b) eqn:Hb;
    [apply nth_error_Some; rewrite Hb; discriminate|destruct i; discriminate].
Qed.

Section Thread.

Variable F : lfacts.
Variable prog : program.
Variable cta : list pc.

Definition tpos (E : env) (t : nat) : option pc := option_map pc_of (run' F prog E t).

Definition at_cta (o : option pc) : bool := match o with Some q => is_cta cta q | None => false end.

Fixpoint barriers_before (E : env) (t : nat) : nat :=
  match t with O => O | S u => (barriers_before E u + if at_cta (tpos E u) then 1 else 0)%nat end.

Fixpoint last_start (E : env) (t : nat) : option pc :=
  match t with O => None | S u => if at_cta (tpos E u) then tpos E u else last_start E u end.

Definition start_bit (o : option pc) : Z := match o with None => 1 | Some q => phase_bit cta q end.

Definition phase_at (P : list Z) (q : pc) : Z := phase_iter cta (fst q) 0 (nth (fst q) P 0) (snd q).

Lemma run_succ : forall E t, run' F prog E (S t) =
  match run' F prog E t with Some s => phase_step F prog E t s | None => None end.
Proof. reflexivity. Qed.

Lemma barriers_mono : forall E t k, (barriers_before E t <= barriers_before E (t + k))%nat.
Proof.
  intros E t k. induction k as [|k IH]; [rewrite Nat.add_0_r; lia|].
  rewrite Nat.add_succ_r. cbn [barriers_before]. lia.
Qed.

Lemma last_start_stable : forall E t k, barriers_before E (t + k) = barriers_before E t ->
  last_start E (t + k) = last_start E t.
Proof.
  intros E t k. induction k as [|k IH]; intros H; [rewrite Nat.add_0_r; reflexivity|].
  rewrite Nat.add_succ_r in *. cbn [barriers_before last_start] in *.
  pose proof (barriers_mono E t k) as Hm.
  destruct (at_cta (tpos E (t + k))); [lia|]. apply IH. lia.
Qed.

(* A run that passes a position of cta at step u has, at any later step,
   a last start that it passed at or after u. *)
Lemma last_start_after : forall E u t q, tpos E u = Some q -> is_cta cta q = true -> (u < t)%nat ->
  exists v p, (u <= v < t)%nat /\ last_start E t = Some p /\ tpos E v = Some p /\ is_cta cta p = true.
Proof.
  intros E u t q Hu Hq. induction t as [|t IH]; intros Ht; [lia|].
  cbn [last_start]. destruct (at_cta (tpos E t)) eqn:Ha.
  - unfold at_cta in Ha. destruct (tpos E t) as [p|] eqn:Hp; [|discriminate Ha].
    exists t, p. split; [lia|]. split; [reflexivity|]. split; [exact Hp|exact Ha].
  - destruct (Nat.eq_dec u t) as [->|Hne].
    + unfold at_cta in Ha. rewrite Hu in Ha. congruence.
    + destruct (IH ltac:(lia)) as [v [p [Hv [Hl [Hp Hc]]]]]. exists v, p. split; [lia|]. auto.
Qed.

Lemma start_bit_pos : forall o, Z.testbit (start_bit o) (match o with None => 0 | Some q => Z.of_nat (S (cta_index cta q)) end) = true.
Proof. intros [q|]; [unfold start_bit, phase_bit; apply Z.pow2_bits_true; lia|reflexivity]. Qed.

Lemma phase_block_spec : forall P b,
  let '(s', edges, d) := phase_block cta prog P b in
  s' = phase_iter cta b 0 (nth b P 0) (length (block_body prog b)) /\
  (falls_through (block_body prog b) = true -> In (S b, s') edges) /\
  forall k ins, nth_error (block_body prog b) k = Some ins ->
    (forall tg, jump_target ins = Some tg -> In (tg, phase_iter cta b 0 (nth b P 0) (S k)) edges) /\
    (shared_write ins = true -> zsub (phase_iter cta b 0 (nth b P 0) k) d = true).
Proof.
  intros P b. unfold phase_block.
  destruct (phase_walk cta b 0 (nth b P 0) (block_body prog b) [] 0) as [[s' edges] d] eqn:Hw.
  destruct (phase_walk_spec cta (block_body prog b) b 0 (nth b P 0) [] 0 s' edges d Hw) as [Hs [_ [_ Hk]]].
  split; [exact Hs|]. split.
  - intros Hf. rewrite Hf. left. reflexivity.
  - intros k ins Hki. destruct (Hk k ins Hki) as [Hj Hd]. split; [|exact Hd].
    intros tg Htg. specialize (Hj tg Htg). destruct (falls_through (block_body prog b)); [right|]; exact Hj.
Qed.

Lemma phase_verify_edge : forall P b tg z, phase_verify cta prog P = true -> (b < length prog)%nat ->
  (tg < length prog)%nat ->
  In (tg, z) (let '(_, edges, _) := phase_block cta prog P b in edges) -> zsub z (nth tg P 0) = true.
Proof.
  intros P b tg z H Hb Htg Hin. unfold phase_verify in H. apply andb_true_iff in H. destruct H as [_ H].
  rewrite forallb_forall in H. specialize (H b (proj2 (in_seq _ _ _) (conj (Nat.le_0_l b) Hb))).
  destruct (phase_block cta prog P b) as [[s' edges] d]. unfold phase_edges_ok in H.
  rewrite forallb_forall in H. apply (H (tg, z)). apply filter_In. split; [exact Hin|]. apply Nat.ltb_lt. exact Htg.
Qed.

Lemma phase_reach : forall P, phase_verify cta prog P = true ->
  forall E u s ins, run' F prog E u = Some s -> fetch prog (blk s) (idx s) = Some ins ->
  zsub (start_bit (last_start E u)) (phase_at P (pc_of s)) = true.
Proof.
  intros P HP E u. induction u as [|u IH]; intros s ins Hrun Hf.
  - injection Hrun as <-. cbn. unfold phase_verify in HP. apply andb_true_iff in HP. destruct HP as [H0 _].
    apply zsub_bits. intros i Hi H1. destruct (Z.eq_dec i 0) as [->|Hne]; [exact H0|].
    rewrite Z.bits_above_log2 in H1 by (cbn; lia). discriminate.
  - rewrite run_succ in Hrun. destruct (run' F prog E u) as [s0|] eqn:Hr0; [|discriminate].
    assert (Hf0 : exists i0, fetch prog (blk s0) (idx s0) = Some i0).
    { unfold phase_step, step in Hrun. destruct (fetch prog (blk s0) (idx s0)) as [i0|];
        [exists i0; reflexivity|discriminate]. }
    destruct Hf0 as [i0 Hf0].
    pose proof (IH s0 i0 eq_refl Hf0) as Hih.
    set (b := blk s0) in *. set (i := idx s0) in *.
    assert (Hb : (b < length prog)%nat) by exact (fetch_block prog b i i0 Hf0).
    assert (Hafter : zsub (start_bit (last_start E (S u))) (phase_at P (b, S i)) = true).
    { unfold phase_at. cbn [fst snd]. rewrite (phase_iter_succ cta i b 0). cbn [Nat.add].
      cbn [last_start]. unfold tpos. rewrite Hr0. cbn [option_map at_cta]. unfold pc_of. fold b i.
      unfold phase_next. destruct (is_cta cta (b, i)).
      - apply zsub_refl.
      - exact Hih. }
    pose proof (step_pos F prog E u s0 s i0 Hf0) as Hpos.
    destruct (Hpos Hrun) as [[Hn [Hbra Hexit]]|[[g [tg [Hi [Hg Hp]]]]|[g [Hi [Hg Hp]]]]].
    + unfold next_pos in Hn. fold b i in Hn.
      destruct (Nat.ltb (S i) (block_len prog b)) eqn:Hlt.
      * unfold pc_of. rewrite Hn. exact Hafter.
      * assert (Hlast : S i = length (block_body prog b)).
        { apply Nat.ltb_ge in Hlt. unfold block_len in Hlt. assert (i < length (block_body prog b))%nat
            by (apply nth_error_Some; unfold fetch in Hf0; fold b i in Hf0; rewrite Hf0; discriminate). lia. }
        unfold pc_of. rewrite Hn. cbn [fst snd]. unfold phase_at. cbn [fst snd phase_iter].
        assert (Hft : falls_through (block_body prog b) = true).
        { unfold falls_through. rewrite last_instr_nth, <- Hlast. cbn [pred].
          unfold fetch in Hf0. fold b i in Hf0. rewrite Hf0.
          destruct i0 as [| | | | | | | | | | | | | | |gg tg0|gg| | | | | | | |]; try reflexivity;
            destruct gg; try reflexivity.
          - specialize (Hbra None tg0 eq_refl). discriminate.
          - specialize (Hexit None eq_refl). discriminate. }
        pose proof (phase_block_spec P b) as Hspec.
        destruct (phase_block cta prog P b) as [[s' edges] d] eqn:Hpb. destruct Hspec as [Hs' [Hfall _]].
        assert (Hsb : (S b < length prog)%nat).
        { assert (Hn' := Hn). injection Hn' as Hb1 Hi1. rewrite Hb1, Hi1 in Hf.
          exact (fetch_block prog (S b) 0 ins Hf). }
        apply (zsub_trans _ (phase_at P (b, S i))); [exact Hafter|].
        unfold phase_at. cbn [fst snd]. rewrite Hlast, <- Hs'.
        apply (phase_verify_edge P b (S b) s' HP Hb Hsb). rewrite Hpb. apply Hfall. exact Hft.
    + subst i0. unfold pc_of. rewrite Hp. cbn [fst snd].
      assert (Htg : (tg < length prog)%nat).
      { assert (Hp' := Hp). injection Hp' as Hb1 Hi1. rewrite Hb1, Hi1 in Hf.
        exact (fetch_block prog tg 0 ins Hf). }
      pose proof (phase_block_spec P b) as Hspec.
      destruct (phase_block cta prog P b) as [[s' edges] d] eqn:Hpb. destruct Hspec as [_ [_ Hk]].
      unfold fetch in Hf0. fold b i in Hf0. destruct (Hk i _ Hf0) as [Hj _].
      apply (zsub_trans _ (phase_at P (b, S i))); [exact Hafter|].
      unfold phase_at. cbn [fst snd phase_iter].
      apply (phase_verify_edge P b tg _ HP Hb Htg). rewrite Hpb. apply Hj. reflexivity.
    + exfalso. assert (Hp' := Hp). injection Hp' as Hb1 Hi1. rewrite Hb1, Hi1 in Hf.
      pose proof (fetch_block prog (length prog) 0 ins Hf). lia.
Qed.

Lemma phase_dirty_spec : forall P b i ins, (b < length prog)%nat ->
  fetch prog b i = Some ins -> shared_write ins = true ->
  zsub (phase_at P (b, i)) (phase_dirty cta prog P) = true.
Proof.
  intros P b i ins Hb Hf Hw. unfold phase_dirty.
  pose proof (phase_block_spec P b) as Hspec.
  destruct (phase_block cta prog P b) as [[s' edges] d] eqn:Hpb. destruct Hspec as [_ [_ Hk]].
  destruct (Hk i ins Hf) as [_ Hd]. specialize (Hd Hw).
  assert (Hgen : forall l acc, (In b l -> True) -> zsub acc (fold_left (fun acc b0 =>
    let '(_, _, d0) := phase_block cta prog P b0 in Z.lor acc d0) l acc) = true /\
    (In b l -> zsub d (fold_left (fun acc b0 => let '(_, _, d0) := phase_block cta prog P b0 in Z.lor acc d0) l acc) = true)).
  { induction l as [|b0 l IH]; intros acc _; cbn [fold_left].
    - split; [apply zsub_refl|intros []].
    - destruct (IH (let '(_, _, d0) := phase_block cta prog P b0 in Z.lor acc d0) (fun _ => I)) as [H1 H2].
      split.
      + eapply zsub_trans; [|exact H1]. destruct (phase_block cta prog P b0) as [[? ?] d0]. apply zsub_lor.
      + intros [<-|Hin]; [|exact (H2 Hin)]. eapply zsub_trans; [|exact H1]. rewrite Hpb. apply zsub_lor. }
  destruct (Hgen (seq 0 (length prog)) 0 (fun _ => I)) as [_ H2].
  apply (zsub_trans _ d); [exact Hd|]. apply H2. apply in_seq. lia.
Qed.

End Thread.

(* ------------------------------------------------------------------ *)
(* The dependences at the starts of blocks.                            *)

Lemma pm_fold_and : forall (m : PM.t Z) (f : reg -> Z -> bool),
  PM.fold (fun r z acc => acc && f r z) m true = true -> forall r z, PM.find r m = Some z -> f r z = true.
Proof.
  intros m f H r z Hf. rewrite PM.fold_1 in H. apply PM.elements_correct in Hf.
  assert (Hgen : forall l acc, fold_left (fun a p => a && f (fst p) (snd p)) l acc = true ->
    acc = true /\ forall p, In p l -> f (fst p) (snd p) = true).
  { induction l as [|p l IH]; intros acc Hl; cbn [fold_left] in Hl; [split; [exact Hl|intros _ []]|].
    destruct (IH _ Hl) as [Ha Hp]. apply andb_true_iff in Ha. destruct Ha as [Ha Hfp].
    split; [exact Ha|]. intros p' [<-|Hin]; [exact Hfp|exact (Hp p' Hin)]. }
  destruct (Hgen _ _ H) as [_ Hall]. exact (Hall (r, z) Hf).
Qed.

Lemma dep_le_ok : forall n m1 m2 R, dep_le m1 m2 = true -> deps_ok n m1 R -> deps_ok n m2 R.
Proof.
  intros n m1 m2 R Hle Hok r z x y Hr Hx Hy Ha. unfold dep_le in Hle.
  pose proof (pm_fold_and m2 (fun r z => match PM.find r m1 with Some z1 => zsub z1 z | None => false end) Hle r z Hr)
    as H. cbn beta in H. destruct (PM.find r m1) as [z1|] eqn:Hz1; [|discriminate].
  exact (Hok r z1 x y Hz1 Hx Hy (agree_sub z1 z x y H Ha)).
Qed.

Lemma dep_initial_spec : forall F uniform r z, PM.find r (dep_initial F uniform) = Some z -> z = 0 /\ uniform r = true.
Proof.
  intros F uniform r z. unfold dep_initial.
  assert (Hgen : forall (l : list (String.string * reg)) m, (forall r z, PM.find r m = Some z -> z = 0 /\ uniform r = true) ->
    forall r z, PM.find r (fold_left (fun m x => if uniform (snd x) then PM.add (snd x) 0 m else m) l m) = Some z ->
    z = 0 /\ uniform r = true).
  { induction l as [|[nm q] l IH]; intros m Hm r0 z0 H; cbn [fold_left] in H; [exact (Hm r0 z0 H)|].
    refine (IH _ _ r0 z0 H). intros r1 z1 H1. cbn [snd] in H1. destruct (uniform q) eqn:Hq; [|exact (Hm r1 z1 H1)].
    destruct (Pos.eq_dec r1 q) as [->|Hne].
    - rewrite PM.gss in H1. injection H1 as <-. split; [reflexivity|exact Hq].
    - rewrite PM.gso in H1 by exact Hne. exact (Hm r1 z1 H1). }
  apply Hgen. intros r0 z0 H. rewrite PM.gempty in H. discriminate.
Qed.

Lemma dep_verify_entry : forall F tidr cols G uniform cta events prog P dirty D,
  dep_verify F tidr cols G uniform cta events prog P dirty D = true ->
  exists e0, nth 0 D None = Some e0 /\ dep_le (dep_initial F uniform) e0 = true.
Proof.
  intros F tidr cols G uniform cta events prog P dirty D H. unfold dep_verify in H. apply andb_true_iff in H.
  destruct H as [H _]. destruct (nth 0 D None) as [e0|]; [|discriminate]. exists e0. split; [reflexivity|exact H].
Qed.

Lemma sfact_eqb_eq : forall f g, sfact_eqb f g = true -> f = g.
Proof.
  intros [d m s h|d sh mk|d z|d u sh mk] [d' m' s' h'|d' sh' mk'|d' z'|d' u' sh' mk'] H; try discriminate H;
    cbn [sfact_eqb] in H; rewrite ?andb_true_iff, ?Pos.eqb_eq, ?Z.eqb_eq in H.
  - destruct H as [[[-> ->] ->] Hh]. destruct h, h'; try discriminate Hh; reflexivity.
  - destruct H as [[-> ->] ->]. reflexivity.
  - destruct H as [-> ->]. reflexivity.
  - destruct H as [[[-> ->] ->] ->]. reflexivity.
Qed.

Lemma facts_incl_in : forall G fs, facts_incl G fs = true -> forall g, In g G -> In g fs.
Proof.
  intros G fs H g Hg. unfold facts_incl in H. rewrite forallb_forall in H. specialize (H g Hg).
  apply existsb_exists in H. destruct H as [f [Hf Heq]]. apply sfact_eqb_eq in Heq. subst f. exact Hf.
Qed.

(* An edge with the state st to block tg: its dependences are at least as
   precise as tg's, and st has the facts G if tg is not the first block. *)
Definition edge_ok (G : list sfact) (D : list (option (PM.t Z))) (tg : nat) (st : sstate) : Prop :=
  exists m, nth tg D None = Some m /\ dep_le (ss_dep st) m = true /\
    (tg = 0%nat \/ facts_incl G (ss_facts st) = true).

Definition edges_ok (G : list sfact) (prog : program) (D : list (option (PM.t Z))) (edges : list (nat * sstate))
    : Prop :=
  forall tg st, In (tg, st) edges -> (tg < length prog)%nat -> edge_ok G D tg st.

Lemma edge_entry_ok : forall F n G D tg st R, edge_ok G D tg st -> sstate_ok F n st R ->
  exists m, nth tg D None = Some m /\ sstate_ok F n (entry_state G tg m) R.
Proof.
  intros F n G D tg st R [m [Hm [Hle Hf]]] [Hdeps Hfacts]. exists m. split; [exact Hm|].
  split; [exact (dep_le_ok n _ m R Hle Hdeps)|]. unfold entry_state. cbn [ss_facts].
  destruct (Nat.eqb tg 0) eqn:H0; [intros f []|]. destruct Hf as [Hf|Hf]; [apply Nat.eqb_neq in H0; lia|].
  intros f Hin. exact (Hfacts f (facts_incl_in G _ Hf f Hin)).
Qed.

Lemma dep_verify_block : forall F tidr cols G uniform cta events prog P dirty D b e,
  dep_verify F tidr cols G uniform cta events prog P dirty D = true -> (b < length prog)%nat -> nth b D None = Some e ->
  (exists st edges,
    ss_walk F tidr cols cta dirty b 0 (nth b P 0) (entry_state G b e) (block_body prog b) [] = Some (st, edges) /\
    edges_ok G prog D (if falls_through (block_body prog b) then (S b, st) :: edges else edges)) \/
  (exists g tg st edges,
    let J := region_join prog b (length prog) tg in
    let R := region_blocks prog b J in
    ss_walk F tidr cols cta dirty b 0 (nth b P 0) (entry_state G b e) (block_body prog b) [] = None /\
    last_instr (block_body prog b) = Some (IBra (Some g) tg) /\
    region_ok cta events prog b J R = true /\
    is_cta cta (b, pred (length (block_body prog b))) = false /\
    is_cta events (b, pred (length (block_body prog b))) = false /\
    ss_walk F tidr cols cta dirty b 0 (nth b P 0) (entry_state G b e)
      (removelast (block_body prog b)) [] = Some (st, edges) /\
    edges_ok G prog D ((J, ss_top st (region_writes F prog R)) :: edges)).
Proof.
  intros F tidr cols G uniform cta events prog P dirty D b e H Hb He. unfold dep_verify in H. apply andb_true_iff in H.
  destruct H as [_ H]. rewrite forallb_forall in H. specialize (H b (proj2 (in_seq _ _ _) (conj (Nat.le_0_l b) Hb))).
  assert (Hedges : forall edges, forallb (fun e => match nth (fst e) D None with
                    | Some m => dep_le (ss_dep (snd e)) m && (Nat.eqb (fst e) 0 || facts_incl G (ss_facts (snd e)))
                    | None => false end)
                    (filter (fun e => Nat.ltb (fst e) (length prog)) edges) = true -> edges_ok G prog D edges).
  { intros edges Hall tg st Hin Htg. rewrite forallb_forall in Hall.
    specialize (Hall (tg, st) (proj2 (filter_In _ _ _) (conj Hin (proj2 (Nat.ltb_lt _ _) Htg)))). cbn [fst snd] in Hall.
    unfold edge_ok. destruct (nth tg D None) as [m|]; [|discriminate]. apply andb_true_iff in Hall.
    destruct Hall as [Hle Hf]. exists m. split; [reflexivity|]. split; [exact Hle|]. apply orb_true_iff in Hf.
    destruct Hf as [Hf|Hf]; [left; apply Nat.eqb_eq; exact Hf|right; exact Hf]. }
  unfold dep_block in H. rewrite He in H.
  destruct (ss_walk F tidr cols cta dirty b 0 (nth b P 0) (entry_state G b e) (block_body prog b) [])
    as [[st edges]|] eqn:Hw.
  - left. exists st, edges. split; [reflexivity|]. apply Hedges. exact H.
  - right. unfold diverge_block in H.
    destruct (last_instr (block_body prog b)) as [ins|] eqn:Hl; [|discriminate H].
    destruct ins; try discriminate H. rename g into gg, target into tg. destruct gg as [g|]; [|discriminate H].
    set (J := region_join prog b (length prog) tg) in *.
    set (R := region_blocks prog b J) in *.
    destruct (region_ok cta events prog b J R &&
              negb (is_cta cta (b, pred (length (block_body prog b)))) &&
              negb (is_cta events (b, pred (length (block_body prog b))))) eqn:Hc; [|discriminate H].
    rewrite !andb_true_iff in Hc. destruct Hc as [[HR Hc1] Hc2].
    destruct (ss_walk F tidr cols cta dirty b 0 (nth b P 0) (entry_state G b e)
                (removelast (block_body prog b)) []) as [[st edges]|] eqn:Hw'; [|discriminate H].
    exists g, tg, st, edges. cbv zeta. fold J. fold R.
    split; [reflexivity|]. split; [reflexivity|]. split; [exact HR|]. split; [apply negb_true_iff; exact Hc1|].
    split; [apply negb_true_iff; exact Hc2|]. split; [reflexivity|]. apply Hedges. exact H.
Qed.

Lemma ss_iter_app : forall F tidr cols cta dirty l1 l2 b j s st k, (k <= length l1)%nat ->
  ss_iter F tidr cols cta dirty b j s st (l1 ++ l2) k = ss_iter F tidr cols cta dirty b j s st l1 k.
Proof.
  intros F tidr cols cta dirty l1. induction l1 as [|i l1 IH]; intros l2 b j s st k Hk.
  - cbn in Hk. assert (k = 0%nat) by lia. subst k. reflexivity.
  - destruct k as [|k]; [reflexivity|]. cbn [app ss_iter]. apply IH. cbn in Hk. lia.
Qed.

(* ------------------------------------------------------------------ *)
(* A block of threads, and the facts of the PTX semantics it assumes.  *)

Definition next_pc (prog : program) (q : pc) (ins : instr) (h : bool) : pc :=
  match ins with
  | IBra _ tg => if h then (tg, 0%nat) else next_pos prog (fst q) (snd q)
  | IExit _ => if h then (length prog, 0%nat) else next_pos prog (fst q) (snd q)
  | _ => next_pos prog (fst q) (snd q)
  end.

Lemma step_pc : forall F prog E t s s' ins, fetch prog (blk s) (idx s) = Some ins ->
  phase_step F prog E t s = Some s' -> pc_of s' = next_pc prog (pc_of s) ins (holds (guard_of ins) (regs s)).
Proof.
  intros F prog E t s s' ins Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs. injection Hs as <-.
  destruct (holds (guard_of ins) (regs s)) eqn:Hg; cbn [negb].
  - destruct ins; cbn [guard_of next_pc dest pc_of goto blk idx fst snd] in *; try rewrite Hg;
      destruct (next_pos prog (blk s) (idx s)); reflexivity.
  - destruct ins; cbn [guard_of next_pc dest pc_of goto blk idx fst snd] in *; try rewrite Hg;
      destruct (next_pos prog (blk s) (idx s)); reflexivity.
Qed.

Lemma step_some : forall F prog E t s ins, fetch prog (blk s) (idx s) = Some ins ->
  exists s', phase_step F prog E t s = Some s'.
Proof. intros F prog E t s ins Hf. unfold phase_step, step. rewrite Hf. eexists. reflexivity. Qed.

Lemma step_none : forall F prog E t s, fetch prog (blk s) (idx s) = None -> phase_step F prog E t s = None.
Proof. intros F prog E t s Hf. unfold phase_step, step. rewrite Hf. reflexivity. Qed.

Section Cta.

Variable F : lfacts.
Variable prog : program.
Variable cta : list pc.
Variable events : list pc.
Variable cols : list (pc * collective).
Variable n : nat.
Variable E : nat -> env.

Definition run_at (x t : nat) : option state := run' F prog (E x) t.

Definition pos_at (x t : nat) : option pc := tpos F prog (E x) t.

(* The waiting sites thread x executed before step t, the latest first. *)
Fixpoint evhist (x t : nat) : list pc :=
  match t with
  | O => []
  | S u => match pos_at x u with
           | Some q => if is_cta events q then q :: evhist x u else evhist x u
           | None => evhist x u
           end
  end.

Definition phase_of (x t : nat) : nat := barriers_before F prog cta (E x) t.

(* Thread z at step u writes shared memory that may overlap the w bytes at
   shared address a: a shared store or atomic whose bytes overlap them, or
   a generic store or atomic. *)
Definition writes_shared (z u : nat) (a w : Z) : Prop :=
  exists s i, run_at z u = Some s /\ fetch prog (blk s) (idx s) = Some i /\ holds (guard_of i) (regs s) = true /\
  match i with
  | IShA _ k r off width _ =>
      k <> KLoad /\ trunc 32 (regs s r + off) < a + w /\ a < trunc 32 (regs s r + off) + width
  | IMem _ k true _ _ _ _ _ _ => k <> KLoad
  | _ => False
  end.

(* Thread z at step u makes an atomic read-modify-write of w bytes of shared
   memory. *)
Definition atomic_at (z u : nat) (w : Z) : Prop :=
  exists s g r off o, run_at z u = Some s /\ fetch prog (blk s) (idx s) = Some (IShA g KRmw r off w o) /\
    holds g (regs s) = true.

(* Thread z at step u stores v, the low w bytes of a register or number, to
   shared memory. *)
Definition stores_at (z u : nat) (w v : Z) : Prop :=
  exists s g r off o, run_at z u = Some s /\ fetch prog (blk s) (idx s) = Some (IShA g KStore r off w (Some o)) /\
    holds g (regs s) = true /\ v = trunc (Z.to_nat (8 * w)) (opv (regs s) o).

(* Two writes that PTX's memory model orders as one: atomic
   read-modify-writes of one width, which coherence orders, or stores of
   one value of one width. Two writes of one width that overlap lie at one
   address, since every shared access of a run is aligned (the gate proves
   it, and a misaligned access faults). *)
Definition one_write (z1 u1 z2 u2 : nat) : Prop :=
  (exists w, atomic_at z1 u1 w /\ atomic_at z2 u2 w) \/
  (exists w v, stores_at z1 u1 w v /\ stores_at z2 u2 w v).

Definition mask_full (s : state) (mask : option reg) : Prop :=
  match mask with None => True | Some r => regs s r = full_mask end.

(* Every lane of x's warp reaches position p after the waiting sites that x
   executed before step tx, with a full member mask. *)
Definition warp_arrives (x tx : nat) (p : pc) (mask : option reg) : Prop :=
  forall z, (z < n)%nat -> (z / 32 = x / 32)%nat ->
  exists tz sz, run_at z tz = Some sz /\ pc_of sz = p /\ evhist z tz = evhist x tx /\ mask_full sz mask.

(* (a) Two threads of the block that load the same shared address in the
   same barrier phase read the same value, when no thread of the block
   writes the loaded bytes in that phase, and any two threads that write
   one of the loaded bytes in one earlier phase both make atomic
   read-modify-writes of one width or both store one value of one width.
   The barriers order the writes of earlier phases before the loads, and
   program order orders the writes of one thread, so each load reads, in
   each byte, a last write of the latest phase that writes the byte; the
   writes of different threads in that phase are atomics, which coherence
   orders, or stores of one value. *)
Definition shared_load_agrees : Prop :=
  forall x y tx ty sx sy g bits d a off,
  (x < n)%nat -> (y < n)%nat -> run_at x tx = Some sx -> run_at y ty = Some sy ->
  fetch prog (blk sx) (idx sx) = Some (ILdS g bits d a off) -> pc_of sy = pc_of sx ->
  evhist x tx = evhist y ty -> phase_of x tx = phase_of y ty ->
  holds g (regs sx) = true -> holds g (regs sy) = true ->
  trunc 32 (regs sx a + off) = trunc 32 (regs sy a + off) ->
  (forall z u, (z < n)%nat -> phase_of z u = phase_of x tx ->
     ~ writes_shared z u (trunc 32 (regs sx a + off)) (Z.of_nat bits / 8)) ->
  (forall z1 u1 z2 u2 c, (z1 < n)%nat -> (z2 < n)%nat -> z1 <> z2 ->
     phase_of z1 u1 = phase_of z2 u2 -> (phase_of z1 u1 < phase_of x tx)%nat ->
     trunc 32 (regs sx a + off) <= c < trunc 32 (regs sx a + off) + Z.of_nat bits / 8 ->
     writes_shared z1 u1 c 1 -> writes_shared z2 u2 c 1 -> one_write z1 u1 z2 u2) ->
  trunc bits (e_shm (E x) tx (trunc 32 (regs sx a + off))) =
  trunc bits (e_shm (E y) ty (trunc 32 (regs sy a + off))).

(* (b) Every lane of a warp gets the same result from a full-mask redux. *)
Definition redux_agrees : Prop :=
  forall x y tx ty sx sy d src mask,
  (x < n)%nat -> (y < n)%nat -> (x / 32 = y / 32)%nat ->
  run_at x tx = Some sx -> run_at y ty = Some sy -> pc_of sy = pc_of sx ->
  collective_at cols (pc_of sx) = Some (CRedux d src mask) ->
  evhist x tx = evhist y ty -> warp_arrives x tx (pc_of sx) mask ->
  trunc 32 (e_arb (E x) tx d) = trunc 32 (e_arb (E y) ty d).

(* (c) A full-mask butterfly shuffle with lane mask m returns to lane l the
   source operand of lane l xor m. *)
Definition bfly_reads_partner : Prop :=
  forall x tx ty sx sy d src m mask,
  (x < n)%nat -> (partner x m < n)%nat -> 0 <= m < 32 ->
  run_at x tx = Some sx -> run_at (partner x m) ty = Some sy -> pc_of sy = pc_of sx ->
  collective_at cols (pc_of sx) = Some (CBfly d src m mask) ->
  evhist x tx = evhist (partner x m) ty -> warp_arrives x tx (pc_of sx) mask ->
  trunc 32 (e_arb (E x) tx d) = trunc 32 (regs sy src).

(* The threads 0 to n - 1 of one block: each starts from the launch facts
   with its index in tidr, and the registers that uniform names hold the
   same launch value in every thread. *)
Record cta_family (uniform : reg -> bool) (tidr : reg) : Prop := {
  cf_launch : forall x, (x < n)%nat -> launch_ok F (E x);
  cf_tid : forall x, (x < n)%nat -> e_init (E x) tidr = Z.of_nat x;
  cf_uniform : forall x y r, (x < n)%nat -> (y < n)%nat -> uniform r = true -> e_init (E x) r = e_init (E y) r;
  cf_load : shared_load_agrees;
  cf_redux : redux_agrees;
  cf_bfly : bfly_reads_partner
}.

(* Every thread of the block executes the same sequence of waiting sites:
   what one thread executed before any step, another executed before some
   step. *)
Definition same_waits : Prop :=
  forall x y t, (x < n)%nat -> (y < n)%nat -> exists t', evhist x t = evhist y t'.

(* Steps t to t + d of thread x pass no position of the list L. *)
Definition passes_none (L : list pc) (x t d : nat) : Prop :=
  forall u, (t <= u < t + d)%nat -> at_cta L (pos_at x u) = false.

Lemma evhist_stable : forall x t d, passes_none events x t d -> evhist x (t + d) = evhist x t.
Proof.
  intros x t d. induction d as [|d IH]; intros H; [rewrite Nat.add_0_r; reflexivity|].
  rewrite Nat.add_succ_r. cbn [evhist]. specialize (H (t + d)%nat ltac:(lia)) as Hd.
  rewrite IH by (intros u Hu; apply H; lia). destruct (pos_at x (t + d)) as [q|]; [|reflexivity].
  cbn [at_cta] in Hd. rewrite Hd. reflexivity.
Qed.

Lemma phase_stable : forall x t d, passes_none cta x t d ->
  phase_of x (t + d) = phase_of x t /\ last_start F prog cta (E x) (t + d) = last_start F prog cta (E x) t.
Proof.
  intros x t d. induction d as [|d IH]; intros H; [rewrite Nat.add_0_r; split; reflexivity|].
  rewrite Nat.add_succ_r. unfold phase_of in *. cbn [barriers_before last_start].
  specialize (H (t + d)%nat ltac:(lia)) as Hd. unfold pos_at in Hd. rewrite Hd.
  rewrite Nat.add_0_r. apply IH. intros u Hu. apply H. lia.
Qed.

End Cta.

(* ------------------------------------------------------------------ *)
(* Soundness: every thread of the block executes the same positions.  *)

Lemma guard_uniform : forall F n st R ins x y, sstate_ok F n st R -> branch_ok st ins = true ->
  (x < n)%nat -> (y < n)%nat -> holds (guard_of ins) (R x) = holds (guard_of ins) (R y) \/ is_jump ins = false.
Proof.
  intros F n st R ins x y Hok Hb Hx Hy. destruct ins; try (right; reflexivity); left; cbn [branch_ok guard_of] in *;
    (destruct g as [[neg p]|]; [|reflexivity]); cbn [uniform_guard] in Hb;
    (destruct (dep_of st p) as [z|] eqn:Hz; [|discriminate]); apply Z.eqb_eq in Hb; subst z;
    unfold holds; rewrite (proj1 Hok p 0 x y Hz Hx Hy (agree_zero x y)); reflexivity.
Qed.

Lemma next_pc_same : forall prog q ins h1 h2, h1 = h2 \/ is_jump ins = false ->
  next_pc prog q ins h1 = next_pc prog q ins h2.
Proof. intros prog q ins h1 h2 [->|H]; [reflexivity|]. destruct ins; try discriminate H; reflexivity. Qed.

Lemma pos_run : forall F prog E x t q, pos_at F prog E x t = Some q ->
  exists s, run_at F prog E x t = Some s /\ pc_of s = q.
Proof.
  intros F prog E x t q H. unfold pos_at, tpos in H. unfold run_at.
  destruct (run' F prog (E x) t) as [s|]; [|discriminate]. injection H as <-. exists s. split; reflexivity.
Qed.

Lemma next_pos_cases : forall prog b i ins, fetch prog b i = Some ins ->
  (match ins with IBra None _ | IExit None => False | _ => True end) ->
  ((S i < block_len prog b)%nat /\ next_pos prog b i = (b, S i)) \/
  (S i = block_len prog b /\ falls_through (block_body prog b) = true /\ next_pos prog b i = (S b, 0%nat)).
Proof.
  intros prog b i ins Hf Hins.
  assert (Hi : (i < block_len prog b)%nat) by (apply nth_error_Some; unfold fetch in Hf; rewrite Hf; discriminate).
  unfold next_pos. destruct (Nat.ltb (S i) (block_len prog b)) eqn:Hlt.
  - left. split; [apply Nat.ltb_lt; exact Hlt|reflexivity].
  - right. apply Nat.ltb_ge in Hlt. assert (Hl : S i = block_len prog b) by lia. split; [exact Hl|].
    split; [|reflexivity]. unfold falls_through. rewrite last_instr_nth. unfold block_len in Hl.
    rewrite <- Hl. cbn [pred]. unfold fetch in Hf. rewrite Hf.
    destruct ins; try reflexivity; destruct g; try reflexivity; contradiction.
Qed.

Lemma next_pc_cases : forall prog b i ins rs, fetch prog b i = Some ins ->
  (exists tg, jump_target ins = Some tg /\ next_pc prog (b, i) ins (holds (guard_of ins) rs) = (tg, 0%nat)) \/
  next_pc prog (b, i) ins (holds (guard_of ins) rs) = (length prog, 0%nat) \/
  ((S i < block_len prog b)%nat /\ next_pc prog (b, i) ins (holds (guard_of ins) rs) = (b, S i)) \/
  (S i = block_len prog b /\ falls_through (block_body prog b) = true /\
   next_pc prog (b, i) ins (holds (guard_of ins) rs) = (S b, 0%nat)).
Proof.
  intros prog b i ins rs Hf.
  destruct ins; cbn [next_pc jump_target guard_of fst snd]; try (right; right; apply (next_pos_cases prog b i _ Hf I)).
  - destruct (holds g rs) eqn:Hh; [left; exists target; split; reflexivity|].
    right; right. apply (next_pos_cases prog b i _ Hf). destruct g; [exact I|discriminate Hh].
  - destruct (holds g rs) eqn:Hh; [right; left; reflexivity|].
    right; right. apply (next_pos_cases prog b i _ Hf). destruct g; [exact I|discriminate Hh].
Qed.

(* ------------------------------------------------------------------ *)
(* The blocks that the threads of a divergent branch pass.            *)

Lemma cell_regs_in : forall F r, is_cell F r = true -> In r (cell_regs F).
Proof.
  intros F r H. unfold is_cell in H. unfold cell_regs. apply existsb_exists in H.
  destruct H as [[[p u] c] [Hin Hp]]. apply Pos.eqb_eq in Hp. subst p. apply in_map_iff.
  exists (r, u, c). split; [reflexivity|exact Hin].
Qed.

(* After a run reaches a position without an instruction, it has no state. *)
Lemma run_after_end : forall F prog E T se k, run' F prog E T = Some se -> fetch prog (blk se) (idx se) = None ->
  run' F prog E (T + S k) = None.
Proof.
  intros F prog E T se k Hr Hf. induction k as [|k IH].
  - rewrite Nat.add_1_r, run_succ, Hr. apply step_none. exact Hf.
  - rewrite Nat.add_succ_r, run_succ, IH. reflexivity.
Qed.

(* A state of a run at an instruction comes before the run's end. *)
Lemma run_before_end : forall F prog E T se t s, run' F prog E T = Some se -> fetch prog (blk se) (idx se) = None ->
  run' F prog E t = Some s -> fetch prog (blk s) (idx s) <> None -> (t < T)%nat.
Proof.
  intros F prog E T se t s HT Hf Ht Hs. destruct (Nat.lt_ge_cases t T) as [Hlt|Hge]; [exact Hlt|exfalso].
  destruct (Nat.eq_dec t T) as [->|Hne].
  - rewrite HT in Ht. injection Ht as <-. exact (Hs Hf).
  - replace t with (T + S (t - S T))%nat in Ht by lia. rewrite (run_after_end F prog E T se _ HT Hf) in Ht. discriminate.
Qed.

(* Finitely many threads choose a step each. *)
Lemma finite_choice : forall (Q : nat -> nat -> Prop) n, (forall x, (x < n)%nat -> exists d, Q x d) ->
  exists f : nat -> nat, forall x, (x < n)%nat -> Q x (f x).
Proof.
  intros Q n. induction n as [|n IH]; intros H; [exists (fun _ => 0%nat); intros x Hx; lia|].
  destruct (IH (fun x Hx => H x ltac:(lia))) as [f Hf]. destruct (H n ltac:(lia)) as [d Hd].
  exists (fun x => if Nat.eqb x n then d else f x). intros x Hx.
  destruct (Nat.eqb_spec x n) as [->|Hne]; [exact Hd|apply Hf; lia].
Qed.

Lemma mem_nat_in : forall d R, mem_nat d R = true <-> In d R.
Proof.
  intros d R. unfold mem_nat. rewrite existsb_exists. split.
  - intros [x [Hx Heq]]. apply Nat.eqb_eq in Heq. subst x. exact Hx.
  - intros H. exists d. split; [exact H|apply Nat.eqb_refl].
Qed.

Lemma block_succs_jump : forall prog c i tg, In i (block_body prog c) -> jump_target i = Some tg ->
  In tg (block_succs prog c).
Proof.
  intros prog c i tg Hi Hj. unfold block_succs. apply in_app_iff. right. apply in_flat_map. exists i.
  split; [exact Hi|]. rewrite Hj. left. reflexivity.
Qed.

Lemma block_succs_next : forall prog c, falls_through (block_body prog c) = true -> In (S c) (block_succs prog c).
Proof. intros prog c H. unfold block_succs. rewrite H. left. reflexivity. Qed.

(* What region_ok states about the blocks R. *)
Lemma region_ok_parts : forall cta events prog b J R, region_ok cta events prog b J R = true ->
  (forall d, In d (block_succs prog b) -> d = J \/ In d R) /\
  forall c, In c R -> (0 < block_len prog c)%nat /\
    (forall k i, fetch prog c k = Some i -> region_instr_ok cta events c k i = true) /\
    (forall d, In d (block_succs prog c) -> d = J \/ In d R).
Proof.
  intros cta events prog b J R H. unfold region_ok in H. rewrite !andb_true_iff in H.
  destruct H as [Hstart Hall].
  assert (Hsucc : forall l, forallb (fun d => Nat.eqb d J || mem_nat d R) l = true ->
            forall d, In d l -> d = J \/ In d R).
  { intros l Hl d Hd. rewrite forallb_forall in Hl. specialize (Hl d Hd). apply orb_true_iff in Hl.
    destruct Hl as [Hl|Hl]; [left; apply Nat.eqb_eq; exact Hl|right; apply mem_nat_in; exact Hl]. }
  split; [exact (Hsucc _ Hstart)|].
  intros c Hc. rewrite forallb_forall in Hall. specialize (Hall c Hc). rewrite !andb_true_iff in Hall.
  destruct Hall as [[Hlen Hins] Hnext]. split; [apply negb_true_iff, Nat.eqb_neq in Hlen; lia|].
  split; [|exact (Hsucc _ Hnext)].
  intros k i Hf. rewrite forallb_forall in Hins.
  assert (Hk : (k < block_len prog c)%nat) by (apply nth_error_Some; unfold fetch in Hf; rewrite Hf; discriminate).
  specialize (Hins k (proj2 (in_seq _ _ _) (conj (Nat.le_0_l k) Hk))). rewrite Hf in Hins. exact Hins.
Qed.

Section Region.

Variable F : lfacts.
Variable prog : program.
Variable cta : list pc.
Variable events : list pc.
Variables b J : nat.
Variable R : list nat.
Hypothesis HR : region_ok cta events prog b J R = true.

Definition inreg (q : pc) : Prop := In (fst q) R /\ (snd q < block_len prog (fst q))%nat.

Definition quiet (W : list reg) (i : instr) : Prop :=
  (forall r, In r (loop_writes i) -> In r W) /\ (writes_cells i = true -> forall r, In r (cell_regs F) -> In r W).

Lemma region_facts : forall c k, inreg (c, k) ->
  exists i, fetch prog c k = Some i /\ region_instr_ok cta events c k i = true /\ In i (block_body prog c).
Proof.
  intros c k [Hc Hk]. cbn [fst snd] in *.
  destruct (proj2 (region_ok_parts cta events prog b J R HR) c Hc) as [_ [Hins _]].
  destruct (fetch prog c k) as [i|] eqn:Hf; [|apply nth_error_None in Hf; unfold block_len in Hk; lia].
  exists i. split; [reflexivity|]. split; [exact (Hins k i Hf)|]. unfold fetch in Hf. exact (nth_error_In _ _ Hf).
Qed.

(* The block that control enters from a block of R or from b is J or a
   block of R with an instruction. *)
Lemma region_enter : forall c d, (c = b \/ In c R) -> In d (block_succs prog c) ->
  d = J \/ inreg (d, 0%nat).
Proof.
  intros c d Hc Hd. destruct (region_ok_parts cta events prog b J R HR) as [Hstart Hall].
  assert (Hd' : d = J \/ In d R) by (destruct Hc as [->|Hc]; [exact (Hstart d Hd)|exact (proj2 (proj2 (Hall c Hc)) d Hd)]).
  destruct Hd' as [->|Hin]; [left; reflexivity|right]. split; [exact Hin|exact (proj1 (Hall d Hin))].
Qed.

Lemma region_quiet : forall c k i, inreg (c, k) -> In i (block_body prog c) -> quiet (region_writes F prog R) i.
Proof.
  intros c k i [Hc _] Hi. cbn [fst] in *. unfold region_writes. split.
  - intros r Hr. apply in_flat_map. exists c. split; [exact Hc|].
    apply in_flat_map. exists i. split; [exact Hi|]. apply in_app_iff. left. exact Hr.
  - intros Hw r Hr. apply in_flat_map. exists c. split; [exact Hc|].
    apply in_flat_map. exists i. split; [exact Hi|]. apply in_app_iff. right. rewrite Hw. exact Hr.
Qed.

Lemma keep_quiet : forall E t s s' i W r, fetch prog (blk s) (idx s) = Some i ->
  phase_step F prog E t s = Some s' -> quiet W i -> ~ In r W -> regs s' r = regs s r.
Proof.
  intros E t s s' i W r Hf Hs [Hw Hc] Hr. apply (step_keep F prog E t s s' i r Hf Hs).
  - destruct (existsb (Pos.eqb r) (loop_writes i)) eqn:He; [|reflexivity]. exfalso. apply Hr. apply Hw.
    apply existsb_exists in He. destruct He as [r' [Hin Heq]]. apply Pos.eqb_eq in Heq. subst r'. exact Hin.
  - destruct (writes_cells i) eqn:Hwc; [right|left; reflexivity].
    destruct (is_cell F r) eqn:Hcell; [|reflexivity]. exfalso. apply Hr. apply (Hc eq_refl). apply cell_regs_in. exact Hcell.
Qed.

(* The step after an instruction of block c that is not an exit enters a
   successor of c or stays in c. *)
Lemma step_succ : forall E t s s' c k i, pc_of s = (c, k) -> fetch prog c k = Some i ->
  (match i with IExit _ => False | _ => True end) -> phase_step F prog E t s = Some s' ->
  ((S k < block_len prog c)%nat /\ pc_of s' = (c, S k)) \/
  exists d, In d (block_succs prog c) /\ pc_of s' = (d, 0%nat).
Proof.
  intros E t s s' c k i Hpc Hf Hi Hs.
  assert (Hfs : fetch prog (blk s) (idx s) = Some i) by (unfold pc_of in Hpc; injection Hpc as -> ->; exact Hf).
  rewrite (step_pc F prog E t s s' i Hfs Hs), Hpc.
  assert (Hin : In i (block_body prog c)) by (unfold fetch in Hf; exact (nth_error_In _ _ Hf)).
  assert (Hpos : (match i with IBra None _ | IExit None => False | _ => True end) ->
            ((S k < block_len prog c)%nat /\ next_pos prog c k = (c, S k)) \/
            exists d, In d (block_succs prog c) /\ next_pos prog c k = (d, 0%nat)).
  { intros Hn. destruct (next_pos_cases prog c k i Hf Hn) as [H|[_ [Hft Hnext]]]; [left; exact H|].
    right. exists (S c). split; [apply block_succs_next; exact Hft|exact Hnext]. }
  destruct i; cbn [next_pc guard_of fst snd]; try (apply Hpos; exact I); [|contradiction].
  destruct (holds g (regs s)) eqn:Hh.
  - right. exists target. split; [apply (block_succs_jump prog c _ target Hin); reflexivity|reflexivity].
  - apply Hpos. destruct g; [exact I|discriminate Hh].
Qed.

Lemma region_step : forall E t s, run' F prog E t = Some s -> inreg (pc_of s) ->
  exists s', run' F prog E (S t) = Some s' /\ (pc_of s' = (J, 0%nat) \/ inreg (pc_of s')) /\
    forall r, ~ In r (region_writes F prog R) -> regs s' r = regs s r.
Proof.
  intros E t s Hr Hin. destruct (pc_of s) as [c k] eqn:Hpc.
  destruct (region_facts c k Hin) as [i [Hf [Hok Hbody]]].
  assert (Hfs : fetch prog (blk s) (idx s) = Some i) by (unfold pc_of in Hpc; injection Hpc as -> ->; exact Hf).
  destruct (step_some F prog E t s i Hfs) as [s' Hs]. exists s'.
  split; [rewrite run_succ, Hr; exact Hs|].
  split; [|intros r Hnot; exact (keep_quiet E t s s' i _ r Hfs Hs (region_quiet c k i Hin Hbody) Hnot)].
  unfold region_instr_ok in Hok. rewrite !andb_true_iff in Hok. destruct Hok as [_ Hok].
  assert (Hi : match i with IExit _ => False | _ => True end) by (destruct i; try exact I; discriminate Hok).
  destruct (step_succ E t s s' c k i Hpc Hf Hi Hs) as [[Hk Hnext]|[d [Hd Hnext]]]; rewrite Hnext.
  - right. split; [exact (proj1 Hin)|exact Hk].
  - destruct (region_enter c d (or_intror (proj1 Hin)) Hd) as [->|Hd']; [left; reflexivity|right; exact Hd'].
Qed.

(* A thread in R whose run ends reaches the start of J, passing only R and
   changing no register that R does not write. *)
Lemma region_run : forall E T se, run' F prog E T = Some se -> fetch prog (blk se) (idx se) = None ->
  forall m t s, (T - t < m)%nat -> run' F prog E t = Some s -> inreg (pc_of s) ->
  exists d s', (0 < d)%nat /\ run' F prog E (t + d) = Some s' /\ pc_of s' = (J, 0%nat) /\
    (forall r, ~ In r (region_writes F prog R) -> regs s' r = regs s r) /\
    (forall u, (t <= u < t + d)%nat -> exists q, tpos F prog E u = Some q /\ inreg q).
Proof.
  intros E T se HT Hend m. induction m as [|m IH]; intros t s Hm Hr Hin; [lia|].
  destruct (region_step E t s Hr Hin) as [s' [Hr' [[Hj|Hin'] Hregs]]].
  - exists 1%nat, s'. rewrite Nat.add_1_r. split; [lia|]. split; [exact Hr'|]. split; [exact Hj|].
    split; [exact Hregs|]. intros u Hu. assert (u = t) by lia. subst u. exists (pc_of s).
    unfold tpos. rewrite Hr. split; [reflexivity|exact Hin].
  - assert (Hlt : (S t < T)%nat).
    { destruct (pc_of s') as [c' k'] eqn:Hpc'. destruct (region_facts c' k' Hin') as [i [Hf _]].
      apply (run_before_end F prog E T se (S t) s' HT Hend Hr'). unfold pc_of in Hpc'.
      injection Hpc' as -> ->. rewrite Hf. discriminate. }
    destruct (IH (S t) s' ltac:(lia) Hr' Hin') as [d [s'' [Hd [Hr'' [Hj [Hregs' Hpos]]]]]].
    exists (S d), s''. split; [lia|]. rewrite Nat.add_succ_r, <- Nat.add_succ_l. split; [exact Hr''|].
    split; [exact Hj|]. split; [intros r Hn; rewrite (Hregs' r Hn); exact (Hregs r Hn)|].
    intros u Hu. destruct (Nat.eq_dec u t) as [->|Hne].
    + exists (pc_of s). unfold tpos. rewrite Hr. split; [reflexivity|exact Hin].
    + apply Hpos. lia.
Qed.

(* A thread at a branch at the end of block b whose run ends reaches the
   start of J, passing only the branch and R, and changes no register that
   R does not write. *)
Lemma branch_run : forall E T se t s g tg, run' F prog E T = Some se -> fetch prog (blk se) (idx se) = None ->
  run' F prog E t = Some s ->
  pc_of s = (b, pred (length (block_body prog b))) -> last_instr (block_body prog b) = Some (IBra (Some g) tg) ->
  exists d s', (0 < d)%nat /\ run' F prog E (t + d) = Some s' /\ pc_of s' = (J, 0%nat) /\
    (forall r, ~ In r (region_writes F prog R) -> regs s' r = regs s r) /\
    (forall u, (t <= u < t + d)%nat -> exists q, tpos F prog E u = Some q /\
       (q = (b, pred (length (block_body prog b))) \/ inreg q)).
Proof.
  intros E T se t s g tg HT Hend Hr Hpc Hl.
  assert (Hne : block_body prog b <> []) by (intros He; rewrite He in Hl; discriminate).
  assert (Hf : fetch prog b (pred (length (block_body prog b))) = Some (IBra (Some g) tg)).
  { unfold fetch. rewrite <- last_instr_nth. exact Hl. }
  assert (Hfs : fetch prog (blk s) (idx s) = Some (IBra (Some g) tg)) by
    (unfold pc_of in Hpc; injection Hpc as -> ->; exact Hf).
  destruct (step_some F prog E t s _ Hfs) as [s1 Hs1].
  assert (Hr1 : run' F prog E (S t) = Some s1) by (rewrite run_succ, Hr; exact Hs1).
  assert (Hregs1 : regs s1 = regs s) by exact (jump_regs F prog E t s s1 _ Hfs eq_refl Hs1).
  assert (Hpos1 : pc_of s1 = (J, 0%nat) \/ inreg (pc_of s1)).
  { destruct (step_succ E t s s1 b _ _ Hpc Hf I Hs1) as [[Hk _]|[d [Hd Hnext]]].
    - exfalso. unfold block_len in Hk. lia.
    - rewrite Hnext. destruct (region_enter b d (or_introl eq_refl) Hd) as [->|Hd']; [left; reflexivity|right; exact Hd']. }
  assert (Hfirst : forall u, (t <= u < S t)%nat -> exists q, tpos F prog E u = Some q /\
             (q = (b, pred (length (block_body prog b))) \/ inreg q)).
  { intros u Hu. assert (u = t) by lia. subst u. exists (pc_of s). unfold tpos. rewrite Hr.
    split; [reflexivity|left; exact Hpc]. }
  destruct Hpos1 as [Hj|Hin].
  - exists 1%nat, s1. rewrite Nat.add_1_r. split; [lia|]. split; [exact Hr1|]. split; [exact Hj|].
    split; [intros r _; rewrite Hregs1; reflexivity|exact Hfirst].
  - destruct (region_run E T se HT Hend (S (T - S t)) (S t) s1 ltac:(lia) Hr1 Hin)
      as [d [s' [Hd [Hr' [Hj [Hregs Hpos]]]]]].
    exists (S d), s'. split; [lia|]. rewrite Nat.add_succ_r, <- Nat.add_succ_l. split; [exact Hr'|].
    split; [exact Hj|]. split; [intros r Hn; rewrite (Hregs r Hn), Hregs1; reflexivity|].
    intros u Hu. destruct (Nat.eq_dec u t) as [->|Hne']; [apply Hfirst; lia|].
    destruct (Hpos u ltac:(lia)) as [q [Hq Hin']]. exists q. split; [exact Hq|right; exact Hin'].
Qed.

Lemma region_not_waiting : forall q, inreg q -> is_cta cta q = false /\ is_cta events q = false.
Proof.
  intros [c k] Hin. destruct (region_facts c k Hin) as [i [_ [Hok _]]].
  unfold region_instr_ok in Hok. rewrite !andb_true_iff in Hok. destruct Hok as [[H1 H2] _].
  apply negb_true_iff in H1, H2. split; assumption.
Qed.

End Region.

Lemma removelast_length : forall {A} (l : list A), length (removelast l) = pred (length l).
Proof.
  intros A l. induction l as [|a l IH]; [reflexivity|]. destruct l as [|b l]; [reflexivity|].
  change (removelast (a :: b :: l)) with (a :: removelast (b :: l)). cbn [length] in *. rewrite IH. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* The shared writes that the check orders.                            *)

Lemma write_class_none : forall F n st i, write_class F n st i = WNone <-> shared_write i = false.
Proof.
  intros F n st i. unfold write_class, shared_write.
  destruct i; try (split; intros; reflexivity).
  - destruct k; [split; intros; reflexivity| |]; destruct gen; split; intros H; try reflexivity; discriminate H.
  - destruct k; [split; intros; reflexivity| |].
    + destruct v as [o|]; [|split; intros H; discriminate H].
      destruct (zero_store st width o); [split; intros H; discriminate H|].
      destruct st as [st|]; [destruct (store_orderly F n st a width o)|]; split; intros H; discriminate H.
    + split; intros H; discriminate H.
Qed.

Lemma phase_iter_at_same : forall cta b j s k,
  phase_iter_at (pc_table (cta_bits cta 0)) b j s k = phase_iter cta b j s k.
Proof.
  intros cta b j s k. revert j s. induction k as [|k IH]; intros j s; [reflexivity|].
  cbn [phase_iter_at phase_iter]. rewrite phase_next_at_same. apply IH.
Qed.

(* The writes that the class walk of a block lists: one for each position
   of a write, with its phase starts and its class at the state there. *)
Lemma class_walk_in : forall F tidr n cols cta dirty body b j s st e,
  In e (class_walk F tidr n (pc_table (cta_bits cta 0)) (pc_table cols) dirty b j s st body) <->
  exists k ins, nth_error body k = Some ins /\ shared_write ins = true /\
    e = ((b, (j + k)%nat), phase_iter cta b j s k,
         write_class F n (option_map (fun st0 => ss_iter F tidr cols cta dirty b j s st0 body k) st) ins).
Proof.
  intros F tidr n cols cta dirty body. induction body as [|ins rest IH]; intros b j s st e.
  - cbn [class_walk]. split; [intros []|]. intros [k [ins [Hk _]]]. destruct k; discriminate Hk.
  - cbn [class_walk]. rewrite phase_next_at_same.
    set (st' := option_map (fun st0 => ss_transfer_at F tidr (pc_table cols) dirty (b, j) s st0 ins) st).
    assert (Hst' : st' = option_map (fun st0 => ss_transfer F tidr cols dirty (b, j) s st0 ins) st)
      by (unfold st'; destruct st; cbn [option_map]; [rewrite ss_transfer_at_same|]; reflexivity).
    assert (Hlater : In e (class_walk F tidr n (pc_table (cta_bits cta 0)) (pc_table cols) dirty b (S j)
                             (phase_next cta (b, j) s) st' rest) <->
              exists k i, nth_error rest k = Some i /\ shared_write i = true /\
                e = ((b, (j + S k)%nat), phase_iter cta b j s (S k),
                     write_class F n (option_map (fun st0 => ss_iter F tidr cols cta dirty b j s st0 (ins :: rest) (S k)) st) i)).
    { rewrite IH. rewrite Hst'. split; intros [k [i [Hk [Hw He]]]]; exists k, i; (split; [exact Hk|]); (split; [exact Hw|]);
        rewrite He; replace (S j + k)%nat with (j + S k)%nat by lia; destruct st; reflexivity. }
    assert (Hcases : forall Q : nat -> instr -> Prop,
              (exists k i, nth_error (ins :: rest) k = Some i /\ shared_write i = true /\ Q k i) <->
              ((shared_write ins = true /\ Q O ins) \/ exists k i, nth_error rest k = Some i /\ shared_write i = true /\ Q (S k) i))
      by (intros Q; split; [intros [[|k] [i [Hk Hw]]]; [cbn in Hk; injection Hk as <-; left; exact Hw|right; exists k, i; exact (conj Hk Hw)]
                           |intros [[Hw HP]|[k [i [Hk [Hw HP]]]]]; [exists O, ins; split; [reflexivity|exact (conj Hw HP)]|exists (S k), i; exact (conj Hk (conj Hw HP))]]).
    rewrite (Hcases (fun k i => e = ((b, (j + k)%nat), phase_iter cta b j s k,
       write_class F n (option_map (fun st0 => ss_iter F tidr cols cta dirty b j s st0 (ins :: rest) k) st) i))).
    rewrite <- Hlater. rewrite Nat.add_0_r. cbn [phase_iter].
    assert (Hst0 : option_map (fun st0 => ss_iter F tidr cols cta dirty b j s st0 (ins :: rest) 0) st = st)
      by (destruct st; reflexivity).
    rewrite Hst0.
    assert (Hwrite : forall c, write_class F n st ins = c -> c <> WNone -> shared_write ins = true).
    { intros c Hc Hne. destruct (shared_write ins) eqn:Hs; [reflexivity|].
      apply (write_class_none F n st ins) in Hs. congruence. }
    assert (Hsome : forall c, write_class F n st ins = c -> c <> WNone ->
              In e (((b, j), s, c) :: class_walk F tidr n (pc_table (cta_bits cta 0)) (pc_table cols) dirty b (S j)
                       (phase_next cta (b, j) s) st' rest) <->
              (shared_write ins = true /\ e = ((b, j), s, c)) \/
              In e (class_walk F tidr n (pc_table (cta_bits cta 0)) (pc_table cols) dirty b (S j)
                      (phase_next cta (b, j) s) st' rest)).
    { intros c Hc Hne. rewrite (Hwrite c Hc Hne). cbn [In].
      split; [intros [H|H]; [left; split; [reflexivity|symmetry; exact H]|right; exact H]
             |intros [[_ H]|H]; [left; symmetry; exact H|right; exact H]]. }
    destruct (write_class F n st ins) eqn:Hc.
    + assert (Hn : shared_write ins = false) by (apply (write_class_none F n st ins); exact Hc).
      rewrite Hn. split; [intros H; right; exact H|intros [[H _]|H]; [discriminate H|exact H]].
    + apply (Hsome _ eq_refl). discriminate.
    + apply (Hsome _ eq_refl). discriminate.
    + apply (Hsome _ eq_refl). discriminate.
    + apply (Hsome _ eq_refl). discriminate.
Qed.

(* The field of the thread index that two threads agree on determines the
   bits that they agree on. *)
Lemma field_agree_rev : forall f x y, field_value x f = field_value y f -> agree (field_dep f) x y.
Proof.
  intros [sh mk] x y H. apply agree_bits. intros i Hi Hb. unfold field_dep in Hb. cbn [fst snd] in Hb.
  rewrite Z.shiftr_spec in Hb by exact Hi.
  assert (Hish : 0 <= i + sh).
  { destruct (Z_lt_le_dec (i + sh) 0) as [Hl|Hl]; [rewrite Z.testbit_neg_r in Hb by exact Hl; discriminate|exact Hl]. }
  assert (B := f_equal (fun v => Z.testbit v (i + sh)) H). cbv beta in B. unfold field_value in B. cbn [fst snd] in B.
  rewrite !Z.land_spec, Hb, !andb_true_r, !Z.shiftl_spec in B by exact Hish.
  replace (i + sh - sh) with i in B by lia. exact B.
Qed.

Lemma field_value_range : forall x sh mk, 0 <= mk < 2 ^ 32 -> 0 <= field_value x (sh, mk) < 2 ^ 32.
Proof.
  intros x sh mk Hmk. unfold field_value. cbn [fst snd].
  assert (Hm : mk = Z.land mk (Z.ones 32)) by (rewrite Z.land_ones by lia; symmetry; apply Z.mod_small; exact Hmk).
  rewrite Hm, Z.land_assoc, Z.land_ones by lia. apply Z.mod_pos_bound. lia.
Qed.

Lemma field_value_multiple : forall x sh mk w, (w = 1 \/ w = 2 \/ w = 4 \/ w = 8) -> Z.land mk (w - 1) = 0 ->
  (w | field_value x (sh, mk)).
Proof.
  intros x sh mk w Hw H. unfold field_value. cbn [fst snd].
  assert (Hk : exists k, 0 <= k /\ w = 2 ^ k /\ w - 1 = Z.ones k).
  { destruct Hw as [-> | [-> | [-> | ->]]]; [exists 0|exists 1|exists 2|exists 3]; split; (lia || (split; reflexivity)). }
  destruct Hk as [k [Hk0 [-> Hones]]].
  apply Z.mod_divide; [apply Z.pow_nonzero; lia|].
  rewrite <- Z.land_ones by exact Hk0. rewrite <- Hones, <- Z.land_assoc, H. apply Z.land_0_r.
Qed.

(* Two stores of w bytes at a base plus two values that are multiples of w
   below 2^32, whose bytes meet, store at one address, and the values are
   equal. *)
Lemma overlap_same : forall K f1 f2 w c, 0 < w -> (w | 2 ^ 32) -> (w | f1) -> (w | f2) ->
  0 <= f1 < 2 ^ 32 -> 0 <= f2 < 2 ^ 32 ->
  trunc 32 (K + f1) <= c < trunc 32 (K + f1) + w -> trunc 32 (K + f2) <= c < trunc 32 (K + f2) + w ->
  trunc 32 (K + f1) = trunc 32 (K + f2) /\ f1 = f2.
Proof.
  intros K f1 f2 w c Hw Hw32 H1 H2 Hr1 Hr2 Hc1 Hc2. unfold trunc in *. change (2 ^ Z.of_nat 32) with (2 ^ 32) in *.
  set (M := 2 ^ 32) in *. assert (HM : 0 < M) by (unfold M; lia).
  pose proof (Z.div_mod (K + f1) M ltac:(lia)) as E1. pose proof (Z.div_mod (K + f2) M ltac:(lia)) as E2.
  set (A1 := (K + f1) mod M) in *. set (A2 := (K + f2) mod M) in *.
  set (q1 := (K + f1) / M) in *. set (q2 := (K + f2) / M) in *.
  assert (Hdiff : (w | A1 - A2)).
  { replace (A1 - A2) with ((f1 - f2) - M * (q1 - q2)) by lia.
    apply Z.divide_sub_r; [apply Z.divide_sub_r; assumption|]. apply Z.divide_mul_l. exact Hw32. }
  assert (HA : A1 = A2).
  { destruct Hdiff as [k Hk]. assert (k = 0) by nia. subst k. lia. }
  split; [exact HA|]. assert (Hq : f1 - f2 = M * (q1 - q2)) by lia.
  assert (q1 - q2 = 0) by nia. lia.
Qed.

Lemma pow32_multiple : forall w, (w = 1 \/ w = 2 \/ w = 4 \/ w = 8) -> (w | 2 ^ 32).
Proof.
  intros w [-> | [-> | [-> | ->]]]; [exists (2 ^ 32)|exists (2 ^ 31)|exists (2 ^ 30)|exists (2 ^ 29)]; reflexivity.
Qed.

Section Ordered.

Variable F : lfacts.
Variable tidr : reg.
Variable cols : list (pc * collective).
Variable G : list sfact.
Variable n : nat.
Variable cta : list pc.
Variable prog : program.
Variable P : list Z.
Variable dirty : Z.
Variable D : list (option (PM.t Z)).

Hypothesis Hord : writes_ordered_at F tidr G n cta (pc_table (cta_bits cta 0)) (pc_table cols) prog P dirty D = true.

(* The blocks that a divergent branch's threads may pass. *)
Definition ord_regions : list nat :=
  flat_map (diverged_blocks F tidr G (pc_table (cta_bits cta 0)) (pc_table cols) prog P dirty D) (seq 0 (length prog)).

(* The state at position q that the class of a write there reads: the
   analysis's, unless a divergent branch's threads may pass q's block. *)
Definition ord_state (q : pc) : option sstate :=
  if mem_nat (fst q) ord_regions then None
  else option_map (fun e => ss_iter F tidr cols cta dirty (fst q) 0 (nth (fst q) P 0)
                              (entry_state G (fst q) e) (block_body prog (fst q)) (snd q))
                  (nth (fst q) D None).

Definition class_at (q : pc) (i : instr) : wclass := write_class F n (ord_state q) i.

Definition ord_writes : list (pc * Z * wclass) :=
  program_writes F tidr G n (pc_table (cta_bits cta 0)) (pc_table cols) prog P dirty D.

Lemma ord_writes_in : forall q s c, In (q, s, c) ord_writes <->
  (fst q < length prog)%nat /\ exists i, fetch prog (fst q) (snd q) = Some i /\ shared_write i = true /\
    s = phase_iter cta (fst q) 0 (nth (fst q) P 0) (snd q) /\ c = class_at q i.
Proof.
  intros [b k] s c. unfold ord_writes, program_writes. fold ord_regions. rewrite in_flat_map. cbn [fst snd].
  split.
  - intros [b' [Hb' Hin]]. apply in_seq in Hb'. apply class_walk_in in Hin.
    destruct Hin as [k' [i [Hk [Hw He]]]]. injection He as Hb Hk' Hs Hc. subst b' k'. cbn [Nat.add] in *.
    split; [lia|]. exists i. split; [exact Hk|]. split; [exact Hw|]. split; [exact Hs|].
    rewrite Hc. unfold class_at, ord_state. cbn [fst snd].
    destruct (mem_nat b ord_regions); [reflexivity|]. destruct (nth b D None); reflexivity.
  - intros [Hb [i [Hk [Hw [Hs Hc]]]]]. exists b. split; [apply in_seq; lia|]. apply class_walk_in.
    exists k, i. split; [exact Hk|]. split; [exact Hw|]. cbn [Nat.add]. rewrite Hs, Hc.
    unfold class_at, ord_state. cbn [fst snd]. f_equal. f_equal.
    destruct (mem_nat b ord_regions); [reflexivity|]. destruct (nth b D None); reflexivity.
Qed.

Lemma ord_parts :
  (forall x, In x ord_writes -> snd x <> WOther) /\
  (forall x y, In x ord_writes -> In y ord_writes -> fst (fst x) <> fst (fst y) ->
     Z.land (snd (fst x)) (snd (fst y)) <> 0 -> compatible (snd x) (snd y) = true) /\
  let stores := flat_map (fun x => match snd x with WStore _ => [fst (fst x)] | _ => [] end) ord_writes in
  let cta' := cta ++ stores in
  let P' := phase_propose_at (pc_table (cta_bits cta' 0)) prog in
  phase_verify cta' prog P' = true /\
  forall q, In q stores -> Z.land (phase_iter cta' (fst q) 0 (nth (fst q) P' 0) (snd q)) (phase_bit cta' q) = 0.
Proof.
  pose proof Hord as H. unfold writes_ordered_at in H. fold ord_writes in H. cbv zeta in H.
  rewrite !andb_true_iff in H. destruct H as [[H1 H2] [H3 H4]]. split; [|split].
  - intros x Hx. rewrite forallb_forall in H1. specialize (H1 x Hx). destruct (snd x); congruence.
  - intros x y Hx Hy Hne Hand. rewrite forallb_forall in H2. specialize (H2 x Hx). rewrite forallb_forall in H2.
    specialize (H2 y Hy). rewrite !orb_true_iff in H2. destruct H2 as [[H2|H2]|H2]; [| |exact H2].
    + exfalso. apply Hne. destruct (fst (fst x)) as [a b], (fst (fst y)) as [c d]. unfold pc_eqb in H2.
      cbn [fst snd] in H2. rewrite andb_true_iff, !Nat.eqb_eq in H2. destruct H2 as [-> ->]. reflexivity.
    + exfalso. apply Hand. apply Z.eqb_eq. exact H2.
  - cbv zeta. rewrite phase_verify_at_same in H3. split; [exact H3|].
    intros q Hq. rewrite forallb_forall in H4. specialize (H4 q Hq). apply Z.eqb_eq in H4.
    rewrite phase_iter_at_same in H4. exact H4.
Qed.

End Ordered.

Lemma class_atomic : forall F n st i w, write_class F n st i = WAtomic w ->
  exists g r off o, i = IShA g KRmw r off w o.
Proof.
  intros F n st i w H. unfold write_class in H.
  destruct i; try discriminate H; [destruct k, gen; discriminate H|].
  destruct k; try discriminate H; [|injection H as <-; exists g, a, off, v; reflexivity].
  destruct v as [o|]; [|discriminate H]. destruct (zero_store st width o); [discriminate H|].
  destruct st as [st|]; [destruct (store_orderly F n st a width o)|]; discriminate H.
Qed.

Lemma class_zero : forall F n st i w, write_class F n st i = WZero w ->
  exists g r off o, i = IShA g KStore r off w (Some o) /\ zero_store st w o = true.
Proof.
  intros F n st i w H. unfold write_class in H.
  destruct i; try discriminate H; [destruct k, gen; discriminate H|].
  destruct k; try discriminate H.
  destruct v as [o|]; [|discriminate H]. destruct (zero_store st width o) eqn:Hz.
  - injection H as <-. exists g, a, off, o. split; [reflexivity|exact Hz].
  - destruct st as [st|]; [destruct (store_orderly F n st a width o)|]; discriminate H.
Qed.

Lemma class_store : forall F n st i w, write_class F n st i = WStore w ->
  exists g a off o st0, i = IShA g KStore a off w (Some o) /\ st = Some st0 /\ store_orderly F n st0 a w o = true.
Proof.
  intros F n st i w H. unfold write_class in H.
  destruct i; try discriminate H; [destruct k, gen; discriminate H|].
  destruct k; try discriminate H.
  destruct v as [o|]; [|discriminate H]. destruct (zero_store st width o); [discriminate H|].
  destruct st as [st|]; [|discriminate H]. destruct (store_orderly F n st a width o) eqn:Ho; [|discriminate H].
  injection H as <-. exists g, a, off, o, st. split; [reflexivity|]. split; [reflexivity|exact Ho].
Qed.

Lemma trunc32_wide : forall w X off, (32 <= w)%nat -> trunc 32 (trunc w X + off) = trunc 32 (X + off).
Proof.
  intros w X off Hw. unfold trunc.
  assert (Hp : 2 ^ Z.of_nat w = 2 ^ Z.of_nat 32 * 2 ^ (Z.of_nat w - 32)).
  { rewrite <- Z.pow_add_r by lia. f_equal. lia. }
  assert (Hq : 2 ^ Z.of_nat 32 <> 0) by (apply Z.pow_nonzero; lia).
  rewrite (Z.mod_eq X (2 ^ Z.of_nat w)) by (apply Z.pow_nonzero; lia).
  rewrite Hp.
  replace (X - 2 ^ Z.of_nat 32 * 2 ^ (Z.of_nat w - 32) * (X / (2 ^ Z.of_nat 32 * 2 ^ (Z.of_nat w - 32))) + off)
    with (X + off + - (2 ^ (Z.of_nat w - 32) * (X / (2 ^ Z.of_nat 32 * 2 ^ (Z.of_nat w - 32)))) * 2 ^ Z.of_nat 32)
    by ring.
  apply Z.mod_add. exact Hq.
Qed.

(* Threads 0 to n - 1 agree on every bit from log2 n up. *)
Lemma agree_below : forall n dv x y, (x < n)%nat -> (y < n)%nat ->
  agree (Z.land dv (index_bits n)) x y -> agree dv x y.
Proof.
  intros n dv x y Hx Hy H.
  destruct (Nat.lt_ge_cases n 2) as [Hn|Hn].
  { assert (x = 0%nat) by lia. assert (y = 0%nat) by lia. subst. unfold agree.
    rewrite Z.lxor_nilpotent, Z.land_0_r. reflexivity. }
  unfold index_bits in H. set (k := Z.log2_up (Z.of_nat n)) in *.
  assert (Hk : 0 <= k) by apply Z.log2_up_nonneg.
  assert (Hle : Z.of_nat n <= 2 ^ k) by (apply Z.log2_up_spec; lia).
  rewrite agree_bits in H |- *. intros i Hi Hb.
  destruct (Z_lt_le_dec i k) as [Hik|Hik].
  - apply H; [exact Hi|]. rewrite Z.land_spec, Z.ones_spec_low by lia. rewrite Hb. reflexivity.
  - assert (Hp : 2 ^ k <= 2 ^ i) by (apply Z.pow_le_mono_r; lia).
    rewrite <- (Z.mod_small (Z.of_nat x) (2 ^ i)) by lia. rewrite <- (Z.mod_small (Z.of_nat y) (2 ^ i)) by lia.
    rewrite !Z.mod_pow2_bits_high by lia. reflexivity.
Qed.

(* Two threads that store at overlapping addresses, at a store that
   store_orderly accepts in a state of their registers, store one value. *)
Lemma orderly_same : forall F n st (R : nat -> reg -> Z) a off w o x y c,
  sstate_ok F n st R -> store_orderly F n st a w o = true -> (x < n)%nat -> (y < n)%nat ->
  trunc 32 (R x a + off) <= c < trunc 32 (R x a + off) + w ->
  trunc 32 (R y a + off) <= c < trunc 32 (R y a + off) + w ->
  opv (R x) o = opv (R y) o.
Proof.
  intros F n st R a off w o x y c Hok Hord Hx Hy Hcx Hcy. unfold store_orderly in Hord.
  destruct (match o with OImm _ => Some 0 | OReg v => dep_of st v end) as [dv|] eqn:Hdvo; [|discriminate Hord].
  assert (Hval : agree dv x y -> opv (R x) o = opv (R y) o).
  { intros Ha. destruct o as [v|z]; cbn [opv]; [|reflexivity]. exact (proj1 Hok v dv x y Hdvo Hx Hy Ha). }
  apply orb_true_iff in Hord. destruct Hord as [Hu|Hb].
  - destruct (dep_of st a) as [da|] eqn:Hda; [|discriminate Hu]. apply andb_true_iff in Hu.
    destruct Hu as [Hda0 Hdv0]. apply Z.eqb_eq in Hda0, Hdv0. subst da. apply Hval.
    apply (agree_below n dv x y Hx Hy). rewrite Hdv0. apply agree_zero.
  - destruct (base_of st a) as [[[u sh] mk]|] eqn:Hba; [|discriminate Hb].
    destruct (dep_of st u) as [du|] eqn:Hdu; [|discriminate Hb].
    rewrite !andb_true_iff in Hb. destruct Hb as [[[[[[Hdu0 Hwa] Hmk0] Hmk32] Hw] Hal] Hsub].
    apply Z.eqb_eq in Hdu0. subst du. apply Nat.leb_le in Hwa. apply Z.leb_le in Hmk0. apply Z.ltb_lt in Hmk32.
    apply Z.eqb_eq in Hal.
    assert (Hw' : w = 1 \/ w = 2 \/ w = 4 \/ w = 8).
    { cbn [existsb] in Hw. rewrite !orb_true_iff, !Z.eqb_eq in Hw. destruct Hw as [H|[H|[H|[H|H]]]];
        [lia|lia|lia|lia|discriminate H]. }
    pose proof (base_of_ok F n st R a u sh mk Hok Hba) as Hbase. cbn [fact_holds] in Hbase.
    assert (Hux : R x u = R y u) by exact (proj1 Hok u 0 x y Hdu Hx Hy (agree_zero x y)).
    rewrite (Hbase x Hx), (trunc32_wide _ _ _ Hwa) in Hcx. rewrite (Hbase y Hy), (trunc32_wide _ _ _ Hwa) in Hcy.
    replace (R x u + field_value x (sh, mk) + off) with (R x u + off + field_value x (sh, mk)) in Hcx by ring.
    replace (R y u + field_value y (sh, mk) + off) with (R x u + off + field_value y (sh, mk)) in Hcy
      by (rewrite Hux; ring).
    assert (Hw0 : 0 < w) by (destruct Hw' as [-> | [-> | [-> | ->]]]; lia).
    destruct (overlap_same (R x u + off) _ _ w c Hw0 (pow32_multiple w Hw')
               (field_value_multiple x sh mk w Hw' Hal) (field_value_multiple y sh mk w Hw' Hal)
               (field_value_range x sh mk (conj Hmk0 Hmk32)) (field_value_range y sh mk (conj Hmk0 Hmk32)) Hcx Hcy)
      as [_ Hf].
    apply Hval. apply (agree_below n dv x y Hx Hy). apply (agree_sub _ (field_dep (sh, mk)) x y Hsub).
    apply field_agree_rev. exact Hf.
Qed.

Section Main.

Variable F : lfacts.
Variable prog : program.
Variable cta : list pc.
Variable events : list pc.
Variable cols : list (pc * collective).
Variable n : nat.
Variable E : nat -> env.
Variable uniform : reg -> bool.
Variable tidr : reg.
(* The phase starts in which the analysis takes a shared load's value as
   one that may differ in every thread, and the dependences at the starts
   of blocks. *)
Variable dirty : Z.
Variable G : list sfact.
Variable D : list (option (PM.t Z)).

Definition main_P := phase_propose_at (pc_table (cta_bits cta 0)) prog.

(* The shared writes are ordered. *)
Definition ordered : bool :=
  writes_ordered_at F tidr G n cta (pc_table (cta_bits cta 0)) (pc_table cols) prog main_P dirty D.

Hypothesis Hparts : (0 < n)%nat /\ (n mod 32 = 0)%nat /\ is_cell F tidr = false /\
  (forall ins, In ins (concat prog) -> loop_preserves tidr ins = true) /\
  phase_verify cta prog main_P = true /\
  dep_verify F tidr cols G uniform cta events prog main_P dirty D = true /\
  zsub (phase_dirty cta prog main_P) dirty = true /\ (dirty = all_dirty \/ ordered = true).
Hypothesis Hfam : cta_family F prog cta events cols n E uniform tidr.
Hypothesis Hends : kernel_ends F prog.

Lemma check_parts : (0 < n)%nat /\ (n mod 32 = 0)%nat /\ is_cell F tidr = false /\
  (forall ins, In ins (concat prog) -> loop_preserves tidr ins = true) /\
  phase_verify cta prog main_P = true /\
  dep_verify F tidr cols G uniform cta events prog main_P dirty D = true.
Proof.
  destruct Hparts as [H0 [H1 [H2 [H3 [H4 [H5 _]]]]]]. repeat split; assumption.
Qed.

Lemma thread_inv : forall x t s, (x < n)%nat -> run_at F prog E x t = Some s ->
  in_range F (regs s) /\ regs s tidr = Z.of_nat x.
Proof.
  destruct check_parts as [_ [_ [Hcell [Hpres _]]]].
  intros x t. induction t as [|t IH]; intros s Hx Hrun.
  - injection Hrun as <-. cbn [regs]. split; [|exact (cf_tid _ _ _ _ _ _ _ _ _ Hfam x Hx)].
    intros r. exact (lk_init F (E x) (cf_launch _ _ _ _ _ _ _ _ _ Hfam x Hx) r).
  - unfold run_at in Hrun. rewrite run_succ in Hrun. fold (run_at F prog E x t) in Hrun.
    destruct (run_at F prog E x t) as [s0|] eqn:Hr0; [|discriminate].
    destruct (IH s0 Hx eq_refl) as [Hrange Htid].
    destruct (fetch prog (blk s0) (idx s0)) as [i0|] eqn:Hf0; [|rewrite step_none in Hrun by exact Hf0; discriminate].
    split; [exact (step_range F prog (E x) t s0 s Hrange Hrun)|].
    rewrite (step_keep F prog (E x) t s0 s i0 tidr Hf0 Hrun); [exact Htid| |right; exact Hcell].
    pose proof (Hpres i0 (fetch_in prog _ _ i0 Hf0)) as Hp. unfold loop_preserves in Hp.
    apply negb_true_iff in Hp. exact Hp.
Qed.

Definition Rt (t x : nat) (r : reg) : Z := match run_at F prog E x t with Some s => regs s r | None => 0 end.

Definition ss_at (e : PM.t Z) (q : pc) : sstate :=
  ss_iter F tidr cols cta dirty (fst q) 0 (nth (fst q) main_P 0) (entry_state G (fst q) e)
    (block_body prog (fst q)) (snd q).

Lemma Rt_run : forall t x s, run_at F prog E x t = Some s -> Rt t x = regs s.
Proof. intros t x s H. unfold Rt. rewrite H. reflexivity. Qed.

(* The steps tau x at which the threads are realigned: the same waiting
   sites and barrier phase before them, and the same position. *)
Definition sync_same (tau : nat -> nat) : Prop :=
  forall x y, (x < n)%nat -> (y < n)%nat ->
  evhist F prog events E x (tau x) = evhist F prog events E y (tau y) /\
  phase_of F prog cta E x (tau x) = phase_of F prog cta E y (tau y) /\
  last_start F prog cta (E x) (tau x) = last_start F prog cta (E y) (tau y).

Definition sync_covered (tau : nat -> nat) : Prop :=
  forall x y t, (x < n)%nat -> (y < n)%nat -> (t <= tau x)%nat ->
  exists t', evhist F prog events E x t = evhist F prog events E y t'.

Definition sync_at (tau : nat -> nat) : Prop :=
  (forall x, (x < n)%nat -> run_at F prog E x (tau x) = None) \/
  exists q, (forall x, (x < n)%nat -> pos_at F prog E x (tau x) = Some q) /\
    (forall ins, fetch prog (fst q) (snd q) = Some ins ->
       exists e, nth (fst q) D None = Some e /\ sstate_ok F n (ss_at e q) (fun x => Rt (tau x) x)).

(* A write of thread z at step u: an instruction that may write shared
   memory, whose guard holds. *)
Definition write_at (z u : nat) (s : state) (i : instr) : Prop :=
  run_at F prog E z u = Some s /\ fetch prog (blk s) (idx s) = Some i /\
  holds (guard_of i) (regs s) = true /\ shared_write i = true.

Definition the_class (q : pc) (i : instr) : wclass := class_at F tidr cols G n cta prog main_P dirty D q i.

(* A write of class WZero stores the number 0. *)
Definition event_ok (z u : nat) : Prop :=
  forall s i, write_at z u s i ->
  match the_class (pc_of s) i with WZero w => stores_at F prog E z u w 0 | _ => True end.

(* The writes of the threads before the steps tau: each write of class
   WZero stores 0, and any two threads that write one byte in one phase
   make one write. *)
Definition WriteInv (tau : nat -> nat) : Prop :=
  (forall z u, (z < n)%nat -> (u < tau z)%nat -> event_ok z u) /\
  (forall z1 u1 z2 u2 c, (z1 < n)%nat -> (z2 < n)%nat -> z1 <> z2 -> (u1 < tau z1)%nat -> (u2 < tau z2)%nat ->
     phase_of F prog cta E z1 u1 = phase_of F prog cta E z2 u2 ->
     writes_shared F prog E z1 u1 c 1 -> writes_shared F prog E z2 u2 c 1 -> one_write F prog E z1 u1 z2 u2).

Definition SyncInv (k : nat) (tau : nat -> nat) : Prop :=
  (forall x, (x < n)%nat -> (k <= tau x)%nat) /\ sync_same tau /\ sync_covered tau /\ sync_at tau /\
  (ordered = true -> WriteInv tau).

Lemma inv_zero : SyncInv 0 (fun _ => 0%nat).
Proof.
  destruct check_parts as [Hn0 [Hn32 [_ [_ [_ HD]]]]]. split; [intros; lia|]. split; [|split; [|split]].
  - intros x y Hx Hy. split; [|split]; reflexivity.
  - intros x y t Hx Hy Ht. assert (t = 0%nat) by lia. subst t. exists 0%nat. reflexivity.
  - right. exists (0%nat, 0%nat). split; [intros x Hx; reflexivity|].
    intros ins Hf.
    destruct (dep_verify_entry F tidr cols G uniform cta events prog main_P dirty D HD) as [e0 [He0 Hle]].
    exists e0. split; [exact He0|]. unfold ss_at. cbn [fst snd ss_iter]. split; [|intros f []].
    apply (dep_le_ok n (dep_initial F uniform) e0 _ Hle).
    intros r z x y Hr Hx Hy _. destruct (dep_initial_spec F uniform r z Hr) as [_ Hu].
    unfold Rt, run_at. cbn [run' run regs]. exact (cf_uniform _ _ _ _ _ _ _ _ _ Hfam x y r Hx Hy Hu).
  - intros _. split; intros; lia.
Qed.

(* ------------------------------------------------------------------ *)
(* The writes of the threads.                                          *)

Lemma writes_shared_write : forall z u c w, writes_shared F prog E z u c w ->
  exists s i, write_at z u s i /\
    match i with
    | IShA _ _ r off width _ => trunc 32 (regs s r + off) < c + w /\ c < trunc 32 (regs s r + off) + width
    | _ => True
    end.
Proof.
  intros z u c w [s [i [Hr [Hf [Hg Hi]]]]]. exists s, i. split.
  - split; [exact Hr|]. split; [exact Hf|]. split; [exact Hg|].
    destruct i; try contradiction.
    + destruct gen; [|contradiction]. destruct k; [exfalso; apply Hi; reflexivity|reflexivity|reflexivity].
    + destruct Hi as [Hk _]. destruct k; [exfalso; apply Hk; reflexivity|reflexivity|reflexivity].
  - destruct i; try exact I. exact (proj2 Hi).
Qed.

Lemma write_entry : forall z u s i, ordered = true -> write_at z u s i ->
  In (pc_of s, phase_at cta main_P (pc_of s), the_class (pc_of s) i) (ord_writes F tidr cols G n cta prog main_P dirty D) /\
  the_class (pc_of s) i <> WOther /\ the_class (pc_of s) i <> WNone.
Proof.
  intros z u s i Hord [Hr [Hf [Hg Hw]]].
  assert (Hin : In (pc_of s, phase_at cta main_P (pc_of s), the_class (pc_of s) i)
                   (ord_writes F tidr cols G n cta prog main_P dirty D)).
  { apply (ord_writes_in F tidr cols G n cta prog main_P dirty D). unfold pc_of. cbn [fst snd].
    split; [exact (fetch_block prog _ _ _ Hf)|]. exists i. split; [exact Hf|]. split; [exact Hw|].
    split; reflexivity. }
  split; [exact Hin|]. split.
  - exact (proj1 (ord_parts F tidr cols G n cta prog main_P dirty D Hord) _ Hin).
  - unfold the_class, class_at. intros Hn. apply (write_class_none F n _ i) in Hn. congruence.
Qed.

Lemma atomic_event : forall z u s i w, write_at z u s i -> the_class (pc_of s) i = WAtomic w ->
  atomic_at F prog E z u w.
Proof.
  intros z u s i w [Hr [Hf [Hg _]]] Hc. unfold the_class, class_at in Hc.
  destruct (class_atomic _ _ _ _ _ Hc) as [g [r [off [o ->]]]].
  exists s, g, r, off, o. split; [exact Hr|]. split; [exact Hf|]. exact Hg.
Qed.

Lemma zero_event : forall z u s i w, write_at z u s i -> event_ok z u -> the_class (pc_of s) i = WZero w ->
  stores_at F prog E z u w 0.
Proof. intros z u s i w Hw He Hc. specialize (He s i Hw). rewrite Hc in He. exact He. Qed.

(* Two writes whose positions' phase starts meet make one write, unless
   they are at one position of class WStore. *)
Lemma pair_static : forall z1 u1 s1 i1 z2 u2 s2 i2, ordered = true ->
  write_at z1 u1 s1 i1 -> write_at z2 u2 s2 i2 -> event_ok z1 u1 -> event_ok z2 u2 ->
  Z.land (phase_at cta main_P (pc_of s1)) (phase_at cta main_P (pc_of s2)) <> 0 ->
  (pc_of s1 <> pc_of s2 \/ forall w, the_class (pc_of s1) i1 <> WStore w) ->
  one_write F prog E z1 u1 z2 u2.
Proof.
  intros z1 u1 s1 i1 z2 u2 s2 i2 Hord H1 H2 He1 He2 Hand Hcase.
  destruct (write_entry z1 u1 s1 i1 Hord H1) as [Hin1 [Ho1 Hn1]].
  destruct (write_entry z2 u2 s2 i2 Hord H2) as [Hin2 [Ho2 Hn2]].
  pose proof (proj1 (proj2 (ord_parts F tidr cols G n cta prog main_P dirty D Hord))) as Hcomp.
  assert (Hcl : (the_class (pc_of s1) i1 = the_class (pc_of s2) i2 /\ forall w, the_class (pc_of s1) i1 <> WStore w) \/
                compatible (the_class (pc_of s1) i1) (the_class (pc_of s2) i2) = true).
  { destruct (pc_of s1) as [b1 k1] eqn:Hp1, (pc_of s2) as [b2 k2] eqn:Hp2.
    destruct (Nat.eq_dec b1 b2) as [<-|Hb]; [destruct (Nat.eq_dec k1 k2) as [<-|Hk]|].
    - left. destruct Hcase as [Hne|Hns]; [exfalso; apply Hne; reflexivity|]. split; [|exact Hns].
      destruct H1 as [_ [Hf1 _]]. destruct H2 as [_ [Hf2 _]]. unfold pc_of in Hp1, Hp2.
      injection Hp1 as Hb1 Hk1. injection Hp2 as Hb2 Hk2. rewrite Hb1, Hk1 in Hf1. rewrite Hb2, Hk2 in Hf2.
      rewrite Hf1 in Hf2. injection Hf2 as <-. reflexivity.
    - right. apply (Hcomp _ _ Hin1 Hin2); [cbn [fst]; congruence|exact Hand].
    - right. apply (Hcomp _ _ Hin1 Hin2); [cbn [fst]; congruence|exact Hand]. }
  destruct Hcl as [[Heq Hns]|Hc].
  - destruct (the_class (pc_of s1) i1) as [|w|w|w|] eqn:Hc1; [congruence| | |exfalso; exact (Hns w eq_refl)|congruence].
    + left. exists w. split; [exact (atomic_event _ _ _ _ _ H1 Hc1)|exact (atomic_event _ _ _ _ _ H2 (eq_sym Heq))].
    + right. exists w, 0. split; [exact (zero_event _ _ _ _ _ H1 He1 Hc1)|exact (zero_event _ _ _ _ _ H2 He2 (eq_sym Heq))].
  - destruct (the_class (pc_of s1) i1) as [|w1|w1|w1|] eqn:Hc1, (the_class (pc_of s2) i2) as [|w2|w2|w2|] eqn:Hc2;
      cbn [compatible] in Hc; try discriminate Hc; apply Z.eqb_eq in Hc; subst w2.
    + left. exists w1. split; [exact (atomic_event _ _ _ _ _ H1 Hc1)|exact (atomic_event _ _ _ _ _ H2 Hc2)].
    + right. exists w1, 0. split; [exact (zero_event _ _ _ _ _ H1 He1 Hc1)|exact (zero_event _ _ _ _ _ H2 He2 Hc2)].
Qed.

(* Two steps whose last phase starts are one have phase starts that meet. *)
Lemma bits_meet : forall z1 u1 s1 i1 z2 u2 s2 i2,
  run_at F prog E z1 u1 = Some s1 -> fetch prog (blk s1) (idx s1) = Some i1 ->
  run_at F prog E z2 u2 = Some s2 -> fetch prog (blk s2) (idx s2) = Some i2 ->
  last_start F prog cta (E z1) u1 = last_start F prog cta (E z2) u2 ->
  Z.land (phase_at cta main_P (pc_of s1)) (phase_at cta main_P (pc_of s2)) <> 0.
Proof.
  destruct check_parts as [_ [_ [_ [_ [HP _]]]]].
  intros z1 u1 s1 i1 z2 u2 s2 i2 Hr1 Hf1 Hr2 Hf2 Hls.
  pose proof (phase_reach F prog cta main_P HP (E z1) u1 s1 i1 Hr1 Hf1) as H1.
  pose proof (phase_reach F prog cta main_P HP (E z2) u2 s2 i2 Hr2 Hf2) as H2.
  rewrite Hls in H1.
  set (o := last_start F prog cta (E z2) u2) in *.
  set (k := match o with None => 0 | Some q => Z.of_nat (S (cta_index cta q)) end).
  pose proof (start_bit_pos cta o) as Hb. fold k in Hb.
  assert (Hk : 0 <= k) by (unfold k; destruct o; lia).
  rewrite zsub_bits in H1, H2. intros Hand.
  assert (Hbit := f_equal (fun v => Z.testbit v k) Hand). cbv beta in Hbit.
  rewrite Z.land_spec, Z.testbit_0_l, (H1 k Hk Hb), (H2 k Hk Hb) in Hbit. discriminate.
Qed.

Lemma last_start_same : forall z u u', phase_of F prog cta E z u = phase_of F prog cta E z u' ->
  last_start F prog cta (E z) u = last_start F prog cta (E z) u'.
Proof.
  intros z u u' H. unfold phase_of in H. destruct (Nat.le_gt_cases u u') as [Hle|Hgt].
  - replace u' with (u + (u' - u))%nat by lia. symmetry. apply last_start_stable.
    replace (u + (u' - u))%nat with u' by lia. symmetry. exact H.
  - replace u with (u' + (u - u'))%nat by lia. apply last_start_stable.
    replace (u' + (u - u'))%nat with u by lia. exact H.
Qed.

Lemma phase_lt_step : forall z u u', (phase_of F prog cta E z u < phase_of F prog cta E z u')%nat -> (u < u')%nat.
Proof.
  intros z u u' H. destruct (Nat.lt_ge_cases u u') as [Hlt|Hge]; [exact Hlt|].
  unfold phase_of in H. pose proof (barriers_mono F prog cta (E z) u' (u - u')) as Hm.
  replace (u' + (u - u'))%nat with u in Hm by lia. lia.
Qed.

Lemma pc_eqb_true : forall p q, pc_eqb p q = true -> p = q.
Proof.
  intros [a b] [c d] H. unfold pc_eqb in H. cbn [fst snd] in H. apply andb_true_iff in H.
  destruct H as [H1 H2]. apply Nat.eqb_eq in H1, H2. subst. reflexivity.
Qed.

Lemma pc_eqb_refl : forall p, pc_eqb p p = true.
Proof. intros [a b]. unfold pc_eqb. cbn [fst snd]. rewrite !Nat.eqb_refl. reflexivity. Qed.

(* A thread does not reach a position of class WStore twice in one phase. *)
Lemma no_revisit : forall z u u' s s' i w, ordered = true -> (u < u')%nat ->
  run_at F prog E z u = Some s -> run_at F prog E z u' = Some s' -> pc_of s' = pc_of s ->
  fetch prog (blk s) (idx s) = Some i -> the_class (pc_of s) i = WStore w ->
  phase_of F prog cta E z u = phase_of F prog cta E z u' -> False.
Proof.
  intros z u u' s s' i w Hord Hlt Hr Hr' Hpc Hf Hc Hph.
  destruct check_parts as [_ [_ [_ [_ [HP _]]]]].
  destruct (ord_parts F tidr cols G n cta prog main_P dirty D Hord) as [_ [Hcomp HC4]]. cbv zeta in HC4.
  set (W := ord_writes F tidr cols G n cta prog main_P dirty D) in *.
  set (stores := flat_map (fun x => match snd x with WStore _ => [fst (fst x)] | _ => [] end) W) in *.
  set (cta' := cta ++ stores) in *.
  set (P' := phase_propose_at (pc_table (cta_bits cta' 0)) prog) in *.
  destruct HC4 as [HP' Hbits].
  assert (Hw : shared_write i = true).
  { destruct (shared_write i) eqn:Hs; [reflexivity|]. apply (write_class_none F n (ord_state F tidr cols G cta prog main_P dirty D (pc_of s)) i) in Hs.
    unfold the_class, class_at in Hc. congruence. }
  assert (Hentry : forall q iq, fetch prog (fst q) (snd q) = Some iq -> shared_write iq = true ->
            In (q, phase_at cta main_P q, the_class q iq) W).
  { intros q iq Hfq Hwq. apply (ord_writes_in F tidr cols G n cta prog main_P dirty D).
    split; [exact (fetch_block prog _ _ _ Hfq)|]. exists iq. split; [exact Hfq|]. split; [exact Hwq|].
    split; reflexivity. }
  assert (Hq : In (pc_of s) stores).
  { unfold stores. apply in_flat_map. exists (pc_of s, phase_at cta main_P (pc_of s), WStore w).
    split; [rewrite <- Hc; apply Hentry; [exact Hf|exact Hw]|cbn; left; reflexivity]. }
  assert (Htp : tpos F prog (E z) u = Some (pc_of s)) by (unfold tpos; unfold run_at in Hr; rewrite Hr; reflexivity).
  assert (Hcta' : is_cta cta' (pc_of s) = true).
  { unfold is_cta. apply existsb_exists. exists (pc_of s). split; [apply in_or_app; right; exact Hq|apply pc_eqb_refl]. }
  destruct (last_start_after F prog cta' (E z) u u' (pc_of s) Htp Hcta' Hlt) as [v [p [Hv [Hls [Hvp Hpc']]]]].
  assert (Hf' : fetch prog (blk s') (idx s') = Some i) by (unfold pc_of in Hpc; injection Hpc as -> ->; exact Hf).
  pose proof (phase_reach F prog cta' P' HP' (E z) u' s' i Hr' Hf') as Hreach. rewrite Hls, Hpc in Hreach.
  (* the phase of the thread is the same from u to u' *)
  assert (Hphv : phase_of F prog cta E z u = phase_of F prog cta E z v /\
                 phase_of F prog cta E z (S v) = phase_of F prog cta E z u').
  { unfold phase_of in *. pose proof (barriers_mono F prog cta (E z) u (v - u)) as M1.
    replace (u + (v - u))%nat with v in M1 by lia.
    pose proof (barriers_mono F prog cta (E z) v 1) as M2. rewrite Nat.add_1_r in M2.
    pose proof (barriers_mono F prog cta (E z) (S v) (u' - S v)) as M3.
    replace (S v + (u' - S v))%nat with u' in M3 by lia. lia. }
  unfold is_cta in Hpc'. unfold cta' in Hpc'. rewrite existsb_app in Hpc'.
  destruct (existsb (pc_eqb p) cta) eqn:Hp.
  - (* a CTA barrier between u and u' would change the phase *)
    destruct Hphv as [H1 H2]. unfold phase_of in H1, H2, Hph. cbn [barriers_before] in H2. unfold at_cta in H2.
    rewrite Hvp in H2. unfold is_cta in H2. rewrite Hp in H2. lia.
  - cbn [orb] in Hpc'. destruct (pc_eqb p (pc_of s)) eqn:Hps.
    + (* the start bit of the position itself is a phase start there *)
      apply pc_eqb_true in Hps. subst p. specialize (Hbits (pc_of s) Hq).
      unfold phase_at in Hreach. cbn [start_bit] in Hreach. unfold zsub in Hreach. apply Z.eqb_eq in Hreach.
      unfold phase_bit in *. set (bit := 2 ^ Z.of_nat (S (cta_index cta' (pc_of s)))) in *.
      assert (Hbit0 : bit <> 0) by (unfold bit; apply Z.pow_nonzero; lia).
      apply Hbit0. rewrite <- Hbits. apply Z.bits_inj'. intros j Hj.
      assert (B := f_equal (fun x => Z.testbit x j) Hreach). cbv beta in B.
      rewrite Z.land_spec, Z.lnot_spec, Z.testbit_0_l in B by exact Hj.
      rewrite Z.land_spec. destruct (Z.testbit bit j); destruct (Z.testbit _ j); cbn in B |- *; congruence.
    + (* another WStore position in the same phase: the check refuses it *)
      apply existsb_exists in Hpc'. destruct Hpc' as [p' [Hp' Heq]]. apply pc_eqb_true in Heq. subst p'.
      unfold stores in Hp'. apply in_flat_map in Hp'. destruct Hp' as [[[p0 sp] cp] [Hin Hpin]].
      destruct cp as [| |w'| |]; cbn [snd fst] in Hpin; try contradiction. destruct Hpin as [<-|[]].
      apply (ord_writes_in F tidr cols G n cta prog main_P dirty D) in Hin as Hin'.
      destruct Hin' as [_ [ip [Hfp [Hwp [Hsp Hcp]]]]].
      destruct (pos_run F prog E z v p0 Hvp) as [sv [Hrv Hpcv]].
      assert (Hfv : fetch prog (blk sv) (idx sv) = Some ip)
        by (unfold pc_of in Hpcv; rewrite <- Hpcv in Hfp; exact Hfp).
      assert (Hmeet := bits_meet z u s i z v sv ip Hr Hf Hrv Hfv (last_start_same z u v (proj1 Hphv))).
      rewrite Hpcv in Hmeet. assert (Hsp2 : sp = phase_at cta main_P p0) by exact Hsp. rewrite <- Hsp2 in Hmeet.
      assert (Hcomp' := Hcomp _ _ (Hentry (pc_of s) i Hf Hw) Hin).
      cbn [fst snd] in Hcomp'. rewrite Hc in Hcomp'.
      assert (Hne : pc_of s <> p0) by (intros Heq; rewrite <- Heq, pc_eqb_refl in Hps; discriminate Hps).
      specialize (Hcomp' Hne). cbn [compatible] in Hcomp'.
      discriminate (Hcomp' Hmeet).
Qed.

Lemma one_write_sym : forall z1 u1 z2 u2, one_write F prog E z1 u1 z2 u2 -> one_write F prog E z2 u2 z1 u1.
Proof.
  intros z1 u1 z2 u2 [[w [H1 H2]]|[w [v [H1 H2]]]]; [left; exists w|right; exists w, v]; split; assumption.
Qed.

(* A shared load in a phase that no write reaches, which the analysis
   takes as a value that depends only on its address, lies in a program
   whose writes are ordered. *)
Lemma clean_ordered : forall x t s ins, run_at F prog E x t = Some s -> fetch prog (blk s) (idx s) = Some ins ->
  Z.land (phase_at cta main_P (pc_of s)) dirty = 0 -> ordered = true.
Proof.
  destruct check_parts as [_ [_ [_ [_ [HP _]]]]].
  intros x t s ins Hr Hf Hclean.
  destruct Hparts as [_ [_ [_ [_ [_ [_ [_ [Hd|Ho]]]]]]]]; [|exact Ho]. exfalso.
  rewrite Hd in Hclean. unfold all_dirty in Hclean. rewrite Z.land_m1_r in Hclean.
  pose proof (phase_reach F prog cta main_P HP (E x) t s ins Hr Hf) as H. rewrite Hclean in H.
  unfold zsub in H. rewrite Z.lnot_0, Z.land_m1_r in H. apply Z.eqb_eq in H.
  destruct (last_start F prog cta (E x) t) as [q|]; cbn [start_bit] in H; [|discriminate H].
  unfold phase_bit in H. revert H. apply Z.pow_nonzero; lia.
Qed.

(* A step at which no thread writes keeps the writes ordered. *)
Lemma idle_writes : forall tau, (forall x s i, (x < n)%nat -> ~ write_at x (tau x) s i) ->
  WriteInv tau -> WriteInv (fun x => S (tau x)).
Proof.
  intros tau Hno [Hev Hpair]. split.
  - intros z u Hz Hu. destruct (Nat.eq_dec u (tau z)) as [->|Hne]; [|apply Hev; [exact Hz|lia]].
    intros s i Hw. exfalso. exact (Hno z s i Hz Hw).
  - intros z1 u1 z2 u2 c Hz1 Hz2 Hne Hu1 Hu2 Hph Hw1 Hw2.
    destruct (Nat.eq_dec u1 (tau z1)) as [->|H1].
    { exfalso. destruct (writes_shared_write _ _ _ _ Hw1) as [s [i [Hw _]]]. exact (Hno z1 s i Hz1 Hw). }
    destruct (Nat.eq_dec u2 (tau z2)) as [->|H2].
    { exfalso. destruct (writes_shared_write _ _ _ _ Hw2) as [s [i [Hw _]]]. exact (Hno z2 s i Hz2 Hw). }
    apply (Hpair z1 u1 z2 u2 c); try assumption; lia.
Qed.

(* A step at which every thread is at one position keeps the writes
   ordered: the new writes of class WZero store 0 by the analysis's
   constants, two new writes of class WStore store one value, and a new
   write and an earlier one of one phase lie at different positions, whose
   classes the check orders, or at one position of class WAtomic or
   WZero; a thread does not reach a position of class WStore twice in one
   phase. *)
Lemma sync_writes : forall k tau b i ins e, SyncInv k tau ->
  (forall x, (x < n)%nat -> pos_at F prog E x (tau x) = Some (b, i)) -> fetch prog b i = Some ins ->
  nth b D None = Some e -> sstate_ok F n (ss_at e (b, i)) (fun x => Rt (tau x) x) ->
  ordered = true -> WriteInv (fun x => S (tau x)).
Proof.
  intros k tau b i ins e [_ [Hsame [_ [_ Hwold]]]] Hq Hf He Hok Hord.
  destruct (Hwold Hord) as [Hev Hpair].
  assert (Hthr : forall x, (x < n)%nat -> exists s, run_at F prog E x (tau x) = Some s /\ pc_of s = (b, i) /\
            fetch prog (blk s) (idx s) = Some ins).
  { intros x Hx. destruct (pos_run F prog E x (tau x) _ (Hq x Hx)) as [s [Hr Hpc]]. exists s.
    split; [exact Hr|]. split; [exact Hpc|]. unfold pc_of in Hpc. injection Hpc as -> ->. exact Hf. }
  assert (Hstate : forall st, ord_state F tidr cols G cta prog main_P dirty D (b, i) = Some st -> st = ss_at e (b, i)).
  { intros st H. unfold ord_state in H. cbn [fst snd] in H.
    destruct (mem_nat b (ord_regions F tidr cols G cta prog main_P dirty D)); [discriminate H|].
    rewrite He in H. cbn [option_map] in H. injection H as <-. reflexivity. }
  assert (Hat : forall x s0 i0, (x < n)%nat -> write_at x (tau x) s0 i0 -> pc_of s0 = (b, i) /\ i0 = ins).
  { intros x s0 i0 Hx [Hr0 [Hf0 _]]. destruct (Hthr x Hx) as [s [Hr [Hpc Hfs]]]. rewrite Hr in Hr0.
    injection Hr0 as <-. rewrite Hfs in Hf0. injection Hf0 as <-. split; [exact Hpc|reflexivity]. }
  assert (Hnew : forall x, (x < n)%nat -> event_ok x (tau x)).
  { intros x Hx s0 i0 Hw. destruct (Hat x s0 i0 Hx Hw) as [Hpc ->]. rewrite Hpc.
    destruct (the_class (b, i) ins) as [|w|w|w|] eqn:Hc; try exact I.
    unfold the_class, class_at in Hc. destruct (class_zero _ _ _ _ _ Hc) as [g [r [off [o [Hi Hz]]]]].
    destruct Hw as [Hr0 [Hf0 [Hg0 _]]]. exists s0, g, r, off, o. split; [exact Hr0|].
    split; [rewrite Hf0, Hi; reflexivity|]. split; [rewrite Hi in Hg0; exact Hg0|].
    unfold zero_store in Hz. destruct o as [v|z].
    - destruct (ord_state F tidr cols G cta prog main_P dirty D (b, i)) as [st|] eqn:Hst; [|discriminate Hz].
      rewrite (Hstate st eq_refl) in Hz. destruct (const_of (ss_at e (b, i)) v) as [z|] eqn:Hcv; [|discriminate Hz].
      apply Z.eqb_eq in Hz. cbn [opv]. rewrite <- (Rt_run (tau x) x s0 Hr0).
      pose proof (const_of_ok F n (ss_at e (b, i)) (fun x => Rt (tau x) x) v z Hok Hcv x Hx) as Hz'.
      cbv beta in Hz'. rewrite Hz'. symmetry. exact Hz.
    - apply Z.eqb_eq in Hz. cbn [opv]. symmetry. exact Hz. }
  assert (Hls : forall x y, (x < n)%nat -> (y < n)%nat ->
            last_start F prog cta (E x) (tau x) = last_start F prog cta (E y) (tau y))
    by (intros x y Hx Hy; exact (proj2 (proj2 (Hsame x y Hx Hy)))).
  (* a new write and an earlier one *)
  assert (Hmixed : forall z1 z2 u2 c, (z1 < n)%nat -> (z2 < n)%nat -> (u2 < tau z2)%nat ->
            phase_of F prog cta E z1 (tau z1) = phase_of F prog cta E z2 u2 ->
            writes_shared F prog E z1 (tau z1) c 1 -> writes_shared F prog E z2 u2 c 1 ->
            one_write F prog E z1 (tau z1) z2 u2).
  { intros z1 z2 u2 c Hz1 Hz2 Hu2 Hph Hw1 Hw2.
    destruct (writes_shared_write z1 (tau z1) c 1 Hw1) as [s1 [i1 [Hwa1 _]]].
    destruct (writes_shared_write z2 u2 c 1 Hw2) as [s2 [i2 [Hwa2 _]]].
    destruct (Hat z1 s1 i1 Hz1 Hwa1) as [Hpc1 ->].
    assert (Hph2 : phase_of F prog cta E z2 u2 = phase_of F prog cta E z2 (tau z2))
      by (rewrite <- Hph; exact (proj1 (proj2 (Hsame z1 z2 Hz1 Hz2)))).
    assert (Hl : last_start F prog cta (E z1) (tau z1) = last_start F prog cta (E z2) u2)
      by (rewrite (Hls z1 z2 Hz1 Hz2); symmetry; exact (last_start_same z2 u2 (tau z2) Hph2)).
    pose proof Hwa1 as [Hr1 [Hf1 _]]. pose proof Hwa2 as [Hr2 [Hf2 _]].
    assert (Hmeet := bits_meet z1 (tau z1) s1 ins z2 u2 s2 i2 Hr1 Hf1 Hr2 Hf2 Hl).
    destruct (the_class (pc_of s1) ins) as [|w|w|w|] eqn:Hc;
      try (apply (pair_static z1 (tau z1) s1 ins z2 u2 s2 i2 Hord Hwa1 Hwa2 (Hnew z1 Hz1) (Hev z2 u2 Hz2 Hu2) Hmeet);
           right; intros w' Hw'; rewrite Hc in Hw'; discriminate Hw').
    destruct (pc_eqb (pc_of s1) (pc_of s2)) eqn:Hpq.
    - apply pc_eqb_true in Hpq. exfalso. destruct (Hthr z2 Hz2) as [s [Hr [Hpc Hfs]]].
      assert (Hi2 : i2 = ins).
      { assert (Hp2 : pc_of s2 = (b, i)) by (rewrite <- Hpq; exact Hpc1).
        unfold pc_of in Hp2. injection Hp2 as Hb2 Hk2. rewrite Hb2, Hk2, Hf in Hf2. injection Hf2 as <-.
        reflexivity. }
      subst i2. apply (no_revisit z2 u2 (tau z2) s2 s ins w Hord Hu2 Hr2 Hr).
      + rewrite Hpc, <- Hpc1, Hpq. reflexivity.
      + exact Hf2.
      + rewrite <- Hpq. exact Hc.
      + exact Hph2.
    - apply (pair_static z1 (tau z1) s1 ins z2 u2 s2 i2 Hord Hwa1 Hwa2 (Hnew z1 Hz1) (Hev z2 u2 Hz2 Hu2) Hmeet).
      left. intros Heq. rewrite Heq, pc_eqb_refl in Hpq. discriminate Hpq. }
  split.
  - intros z u Hz Hu. destruct (Nat.eq_dec u (tau z)) as [->|Hne]; [exact (Hnew z Hz)|].
    apply Hev; [exact Hz|lia].
  - intros z1 u1 z2 u2 c Hz1 Hz2 Hne Hu1 Hu2 Hph Hw1 Hw2.
    destruct (Nat.eq_dec u1 (tau z1)) as [E1|E1]; destruct (Nat.eq_dec u2 (tau z2)) as [E2|E2]; subst.
    + (* two new writes at one position *)
      destruct (writes_shared_write z1 (tau z1) c 1 Hw1) as [s1 [i1 [Hwa1 Hc1]]].
      destruct (writes_shared_write z2 (tau z2) c 1 Hw2) as [s2 [i2 [Hwa2 Hc2]]].
      destruct (Hat z1 s1 i1 Hz1 Hwa1) as [Hpc1 ->]. destruct (Hat z2 s2 i2 Hz2 Hwa2) as [Hpc2 ->].
      pose proof Hwa1 as [Hr1 [Hf1 [Hg1 _]]]. pose proof Hwa2 as [Hr2 [Hf2 [Hg2 _]]].
      assert (Hmeet := bits_meet z1 (tau z1) s1 ins z2 (tau z2) s2 ins Hr1 Hf1 Hr2 Hf2 (Hls z1 z2 Hz1 Hz2)).
      destruct (the_class (pc_of s1) ins) as [|w|w|w|] eqn:Hc;
        try (apply (pair_static z1 (tau z1) s1 ins z2 (tau z2) s2 ins Hord Hwa1 Hwa2 (Hnew z1 Hz1) (Hnew z2 Hz2) Hmeet);
             right; intros w' Hw'; rewrite Hc in Hw'; discriminate Hw').
      rewrite Hpc1 in Hc. unfold the_class, class_at in Hc.
      destruct (class_store _ _ _ _ _ Hc) as [g [a [off [o [st0 [Hi [Hst Hordly]]]]]]].
      rewrite (Hstate st0 Hst) in Hordly. subst ins. cbv beta iota in Hc1, Hc2.
      rewrite <- (Rt_run (tau z1) z1 s1 Hr1) in Hc1. rewrite <- (Rt_run (tau z2) z2 s2 Hr2) in Hc2.
      assert (Hv := orderly_same F n (ss_at e (b, i)) (fun x => Rt (tau x) x) a off w o z1 z2 c Hok Hordly Hz1 Hz2
                      ltac:(cbv beta; lia) ltac:(cbv beta; lia)).
      cbv beta in Hv. rewrite (Rt_run (tau z1) z1 s1 Hr1), (Rt_run (tau z2) z2 s2 Hr2) in Hv.
      right. exists w, (trunc (Z.to_nat (8 * w)) (opv (regs s1) o)). split.
      * exists s1, g, a, off, o. split; [exact Hr1|]. split; [exact Hf1|]. split; [exact Hg1|reflexivity].
      * exists s2, g, a, off, o. split; [exact Hr2|]. split; [exact Hf2|]. split; [exact Hg2|]. rewrite Hv. reflexivity.
    + exact (Hmixed z1 z2 u2 c Hz1 Hz2 ltac:(lia) Hph Hw1 Hw2).
    + apply one_write_sym. exact (Hmixed z2 z1 u1 c Hz2 Hz1 ltac:(lia) (eq_sym Hph) Hw2 Hw1).
    + apply (Hpair z1 u1 z2 u2 c); try assumption; lia.
Qed.

(* The steps of the threads that pass a divergent branch's blocks keep the
   writes ordered: a write there has the class of its instruction alone,
   WAtomic or WZero of a number, and its phase is the branch's. *)
Lemma diverge_writes : forall k tau tau' b L (R : list nat), SyncInv k tau ->
  (forall c, In c R -> In c (ord_regions F tidr cols G cta prog main_P dirty D)) ->
  (forall i, fetch prog b L = Some i -> shared_write i = false) ->
  (forall x, (x < n)%nat -> (tau x <= tau' x)%nat /\ passes_none F prog E cta x (tau x) (tau' x - tau x) /\
     forall u, (tau x <= u < tau' x)%nat -> exists q, pos_at F prog E x u = Some q /\ (q = (b, L) \/ inreg prog R q)) ->
  ordered = true -> WriteInv tau'.
Proof.
  intros k tau tau' b L R [_ [Hsame [_ [_ Hwold]]]] Hsub Hbr Hspan Hord.
  destruct (Hwold Hord) as [Hev Hpair].
  (* a write in a thread's span lies in a region block, whose class has no state *)
  assert (Hreg : forall x u s i, (x < n)%nat -> (tau x <= u < tau' x)%nat -> write_at x u s i ->
            ord_state F tidr cols G cta prog main_P dirty D (pc_of s) = None).
  { intros x u s i Hx Hu [Hr [Hf [_ Hw]]]. destruct (Hspan x Hx) as [_ [_ Hpos]].
    destruct (Hpos u Hu) as [q [Hq [Hbl|[Hin _]]]].
    - exfalso. unfold pos_at, tpos in Hq. unfold run_at in Hr. rewrite Hr in Hq. injection Hq as Hq. subst q.
      unfold pc_of in Hbl. injection Hbl as Hb Hl. rewrite Hb, Hl in Hf. rewrite (Hbr i Hf) in Hw. discriminate Hw.
    - unfold pos_at, tpos in Hq. unfold run_at in Hr. rewrite Hr in Hq. injection Hq as Hq. subst q.
      unfold ord_state. replace (mem_nat (fst (pc_of s)) (ord_regions F tidr cols G cta prog main_P dirty D)) with true
        by (symmetry; apply mem_nat_in; apply Hsub; exact Hin).
      reflexivity. }
  assert (Hnew : forall x u, (x < n)%nat -> (tau x <= u < tau' x)%nat -> event_ok x u).
  { intros x u Hx Hu s i Hw. pose proof (Hreg x u s i Hx Hu Hw) as Hst.
    destruct (the_class (pc_of s) i) as [|w|w|w|] eqn:Hc; try exact I.
    unfold the_class, class_at in Hc. rewrite Hst in Hc. destruct (class_zero _ _ _ _ _ Hc) as [g [r [off [o [Hi Hz]]]]].
    destruct Hw as [Hr [Hf [Hg _]]]. exists s, g, r, off, o. split; [exact Hr|].
    split; [rewrite Hf, Hi; reflexivity|]. split; [rewrite Hi in Hg; exact Hg|].
    destruct o as [v|z]; [discriminate Hz|]. cbn [zero_store] in Hz. apply Z.eqb_eq in Hz. cbn [opv]. symmetry. exact Hz. }
  assert (Hnostore : forall x u s i w, (x < n)%nat -> (tau x <= u < tau' x)%nat -> write_at x u s i ->
            the_class (pc_of s) i <> WStore w).
  { intros x u s i w Hx Hu Hw Hc. unfold the_class, class_at in Hc. rewrite (Hreg x u s i Hx Hu Hw) in Hc.
    destruct (class_store _ _ _ _ _ Hc) as [g [a [off [o [st0 [Hi [Hn _]]]]]]]. discriminate Hn. }
  (* the last start of a step in a span is that of the thread's step tau x *)
  assert (Hspan_ls : forall x u, (x < n)%nat -> (tau x <= u < tau' x)%nat ->
            phase_of F prog cta E x u = phase_of F prog cta E x (tau x) /\
            last_start F prog cta (E x) u = last_start F prog cta (E x) (tau x)).
  { intros x u Hx Hu. destruct (Hspan x Hx) as [_ [Hpn _]].
    replace u with (tau x + (u - tau x))%nat by lia.
    apply (phase_stable F prog cta E x (tau x) (u - tau x)). intros v Hv. apply Hpn. lia. }
  (* any two writes of one phase, one of them in a span *)
  assert (Hone : forall z1 u1 z2 u2 c, (z1 < n)%nat -> (z2 < n)%nat -> (u1 < tau' z1)%nat -> (u2 < tau' z2)%nat ->
            (tau z1 <= u1)%nat -> phase_of F prog cta E z1 u1 = phase_of F prog cta E z2 u2 ->
            writes_shared F prog E z1 u1 c 1 -> writes_shared F prog E z2 u2 c 1 ->
            one_write F prog E z1 u1 z2 u2).
  { intros z1 u1 z2 u2 c Hz1 Hz2 Hu1 Hu2 Ht1 Hph Hw1 Hw2.
    destruct (writes_shared_write z1 u1 c 1 Hw1) as [s1 [i1 [Hwa1 _]]].
    destruct (writes_shared_write z2 u2 c 1 Hw2) as [s2 [i2 [Hwa2 _]]].
    pose proof Hwa1 as [Hr1 [Hf1 _]]. pose proof Hwa2 as [Hr2 [Hf2 _]].
    destruct (Hspan_ls z1 u1 Hz1 ltac:(lia)) as [Hp1 Hl1].
    assert (Hph2 : phase_of F prog cta E z2 u2 = phase_of F prog cta E z2 (tau z2))
      by (rewrite <- Hph, Hp1; exact (proj1 (proj2 (Hsame z1 z2 Hz1 Hz2)))).
    assert (Hl : last_start F prog cta (E z1) u1 = last_start F prog cta (E z2) u2)
      by (rewrite Hl1, (proj2 (proj2 (Hsame z1 z2 Hz1 Hz2))); symmetry; exact (last_start_same z2 u2 (tau z2) Hph2)).
    assert (Hmeet := bits_meet z1 u1 s1 i1 z2 u2 s2 i2 Hr1 Hf1 Hr2 Hf2 Hl).
    assert (He2 : event_ok z2 u2).
    { destruct (Nat.lt_ge_cases u2 (tau z2)) as [Hlt|Hge]; [exact (Hev z2 u2 Hz2 Hlt)|exact (Hnew z2 u2 Hz2 (conj Hge Hu2))]. }
    apply (pair_static z1 u1 s1 i1 z2 u2 s2 i2 Hord Hwa1 Hwa2 (Hnew z1 u1 Hz1 (conj Ht1 Hu1)) He2 Hmeet).
    right. intros w. exact (Hnostore z1 u1 s1 i1 w Hz1 (conj Ht1 Hu1) Hwa1). }
  split.
  - intros z u Hz Hu. destruct (Nat.lt_ge_cases u (tau z)) as [Hlt|Hge]; [exact (Hev z u Hz Hlt)|].
    exact (Hnew z u Hz (conj Hge Hu)).
  - intros z1 u1 z2 u2 c Hz1 Hz2 Hne Hu1 Hu2 Hph Hw1 Hw2.
    destruct (Nat.lt_ge_cases u1 (tau z1)) as [H1|H1]; destruct (Nat.lt_ge_cases u2 (tau z2)) as [H2|H2].
    + exact (Hpair z1 u1 z2 u2 c Hz1 Hz2 Hne H1 H2 Hph Hw1 Hw2).
    + apply one_write_sym. exact (Hone z2 u2 z1 u1 c Hz2 Hz1 Hu2 Hu1 H2 (eq_sym Hph) Hw2 Hw1).
    + exact (Hone z1 u1 z2 u2 c Hz1 Hz2 Hu1 Hu2 H1 Hph Hw1 Hw2).
    + exact (Hone z1 u1 z2 u2 c Hz1 Hz2 Hu1 Hu2 H1 Hph Hw1 Hw2).
Qed.

Lemma one_step : forall k tau, SyncInv k tau ->
  (forall x y, (x < n)%nat -> (y < n)%nat -> pos_at F prog E x (tau x) = pos_at F prog E y (tau y)) ->
  (forall x, (x < n)%nat -> (S k <= S (tau x))%nat) /\ sync_same (fun x => S (tau x)) /\
  sync_covered (fun x => S (tau x)).
Proof.
  intros k tau [Hk [Hsame [Hcov _]]] Hpos.
  assert (Hsame' : sync_same (fun x => S (tau x))).
  { intros x y Hx Hy. destruct (Hsame x y Hx Hy) as [H1 [H2 H3]]. pose proof (Hpos x y Hx Hy) as Hp.
    unfold phase_of in *. cbn [evhist barriers_before last_start]. unfold pos_at in *. rewrite Hp, H1, H2, H3.
    split; [|split]; reflexivity. }
  split; [intros x Hx; specialize (Hk x Hx); lia|]. split; [exact Hsame'|].
  intros x y t Hx Hy Ht. destruct (Nat.eq_dec t (S (tau x))) as [->|Hne].
  - exists (S (tau y)). exact (proj1 (Hsame' x y Hx Hy)).
  - apply Hcov; [exact Hx|exact Hy|lia].
Qed.

Lemma no_shared_write : forall tau x sx ix, sync_same tau -> (x < n)%nat ->
  run_at F prog E x (tau x) = Some sx -> fetch prog (blk sx) (idx sx) = Some ix ->
  Z.land (phase_at cta main_P (pc_of sx)) dirty = 0 ->
  forall z u sz iz, (z < n)%nat -> phase_of F prog cta E z u = phase_of F prog cta E x (tau x) ->
  run_at F prog E z u = Some sz -> fetch prog (blk sz) (idx sz) = Some iz -> shared_write iz = false.
Proof.
  destruct check_parts as [_ [_ [_ [_ [HP _]]]]].
  intros tau x sx ix Hsame Hx Hrx Hfx Hclean z u sz iz Hz Hph Hrz Hfz.
  destruct (shared_write iz) eqn:Hw; [exfalso|reflexivity].
  assert (Hlast : last_start F prog cta (E z) u = last_start F prog cta (E x) (tau x)).
  { destruct (Hsame z x Hz Hx) as [_ [Hp Hls]]. rewrite <- Hls. unfold phase_of in *. rewrite <- Hp in Hph.
    destruct (Nat.le_gt_cases u (tau z)) as [Hut|Hut].
    - replace (tau z) with (u + (tau z - u))%nat by lia. symmetry. apply last_start_stable.
      replace (u + (tau z - u))%nat with (tau z) by lia. symmetry. exact Hph.
    - replace u with (tau z + (u - tau z))%nat by lia. apply last_start_stable.
      replace (tau z + (u - tau z))%nat with u by lia. exact Hph. }
  pose proof (phase_reach F prog cta main_P HP (E z) u sz iz Hrz Hfz) as Hz1.
  pose proof (phase_dirty_spec prog cta main_P (blk sz) (idx sz) iz (fetch_block prog _ _ _ Hfz) Hfz Hw) as Hz2.
  pose proof (phase_reach F prog cta main_P HP (E x) (tau x) sx ix Hrx Hfx) as Hx1.
  rewrite Hlast in Hz1. assert (Hsub : zsub (phase_dirty cta prog main_P) dirty = true)
    by (destruct Hparts as [_ [_ [_ [_ [_ [_ [Hs _]]]]]]]; exact Hs).
  pose proof (zsub_trans _ _ _ Hz2 Hsub) as Hz3. clear Hz2. rename Hz3 into Hz2.
  set (o := last_start F prog cta (E x) (tau x)) in *.
  set (k := match o with None => 0 | Some q => Z.of_nat (S (cta_index cta q)) end).
  pose proof (start_bit_pos cta o) as Hb. fold k in Hb.
  assert (Hk : 0 <= k) by (unfold k; destruct o; lia).
  rewrite zsub_bits in Hz1, Hz2, Hx1.
  assert (Hbit := f_equal (fun v => Z.testbit v k) Hclean). cbn beta in Hbit.
  rewrite Z.land_spec, Z.testbit_0_l in Hbit.
  rewrite (Hx1 k Hk Hb), (Hz2 k Hk (Hz1 k Hk Hb)) in Hbit. discriminate.
Qed.

Lemma single_step : forall k tau b i ins e, SyncInv k tau ->
  (forall x, (x < n)%nat -> pos_at F prog E x (tau x) = Some (b, i)) -> fetch prog b i = Some ins ->
  nth b D None = Some e -> sstate_ok F n (ss_at e (b, i)) (fun x => Rt (tau x) x) ->
  branch_ok (ss_at e (b, i)) ins = true ->
  (forall tg, jump_target ins = Some tg -> (tg < length prog)%nat -> edge_ok G D tg (ss_at e (b, S i))) ->
  (S i = block_len prog b -> falls_through (block_body prog b) = true -> (S b < length prog)%nat ->
     edge_ok G D (S b) (ss_at e (b, S i))) ->
  SyncInv (S k) (fun x => S (tau x)).
Proof.
  destruct check_parts as [Hn0 [Hn32 [Hcell [Hpres [HP HD]]]]].
  intros k tau b i ins e Hinv Hq Hf He Hok Hbranch Hjump Hfall.
  pose proof Hinv as [_ [Hsame [_ [_ Hwold]]]].
  destruct (one_step k tau Hinv) as [Hk' [Hsame' Hcov']];
    [intros x y Hx Hy; rewrite (Hq x Hx), (Hq y Hy); reflexivity|].
  split; [exact Hk'|]. split; [exact Hsame'|]. split; [exact Hcov'|].
  split; [|intros Hord; exact (sync_writes k tau b i ins e Hinv Hq Hf He Hok Hord)].
  assert (Hthr : forall x, (x < n)%nat -> exists s s', run_at F prog E x (tau x) = Some s /\ pc_of s = (b, i) /\
            fetch prog (blk s) (idx s) = Some ins /\ phase_step F prog (E x) (tau x) s = Some s' /\
            run_at F prog E x (S (tau x)) = Some s').
  { intros x Hx. destruct (pos_run F prog E x (tau x) _ (Hq x Hx)) as [s [Hr Hpc]].
    assert (Hfs : fetch prog (blk s) (idx s) = Some ins) by (unfold pc_of in Hpc; injection Hpc as -> ->; exact Hf).
    destruct (step_some F prog (E x) (tau x) s ins Hfs) as [s' Hs'].
    exists s, s'. split; [exact Hr|]. split; [exact Hpc|]. split; [exact Hfs|]. split; [exact Hs'|].
    unfold run_at. rewrite run_succ. fold (run_at F prog E x (tau x)). rewrite Hr. exact Hs'. }
  set (R := fun x => Rt (tau x) x). set (R' := fun x => Rt (S (tau x)) x).
  fold R in Hok.
  set (q' := next_pc prog (b, i) ins (holds (guard_of ins) (R 0%nat))).
  right. exists q'. split.
  { intros x Hx. destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]].
    unfold pos_at, tpos. fold (run_at F prog E x (S (tau x))). rewrite Hr'. cbn [option_map]. f_equal.
    rewrite (step_pc F prog (E x) (tau x) s s' ins Hfs Hs'), Hpc. unfold q'. apply next_pc_same.
    assert (HRx : R x = regs s) by exact (Rt_run (tau x) x s Hr). rewrite <- HRx.
    exact (guard_uniform F n _ R ins x 0 Hok Hbranch Hx Hn0). }
  set (A := fun x => match ins with IDef _ (d :: _) => e_arb (E x) (tau x) d | _ => 0 end).
  set (V := fun x => match ins with ILdS _ _ _ a off => e_shm (E x) (tau x) (trunc 32 (R x a + off)) | _ => 0 end).
  assert (Htr : sstate_ok F n (ss_transfer F tidr cols dirty (b, i) (phase_at cta main_P (b, i)) (ss_at e (b, i)) ins) R').
  { apply (transfer_ok F tidr cols n Hn32 R R'
             (fun x Hx => match Hthr x Hx with ex_intro _ s (ex_intro _ _ (conj Hr _)) =>
                            eq_ind_r (fun f => in_range F f) (proj1 (thread_inv x (tau x) s Hx Hr)) (Rt_run (tau x) x s Hr) end)
             (fun x Hx => match Hthr x Hx with ex_intro _ s (ex_intro _ _ (conj Hr _)) =>
                            eq_ind_r (fun f => f tidr = Z.of_nat x) (proj2 (thread_inv x (tau x) s Hx Hr)) (Rt_run (tau x) x s Hr) end)
             dirty (b, i) (phase_at cta main_P (b, i)) (ss_at e (b, i)) ins A V Hok).
    - intros Hp d Hd x Hx. destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]]. unfold R, R'.
      rewrite (Rt_run (S (tau x)) x s' Hr'), (Rt_run (tau x) x s Hr).
      exact (step_pure_regs F prog (E x) (tau x) s s' ins d Hfs Hp Hd Hs').
    - intros d Hi x Hx. destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]]. subst ins. unfold R, R'.
      rewrite (Rt_run (S (tau x)) x s' Hr'), (Rt_run (tau x) x s Hr).
      exact (step_def_regs F prog (E x) (tau x) s s' d Hfs Hs').
    - intros bits d a off Hi x Hx. destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]]. subst ins.
      unfold R'. rewrite (Rt_run (S (tau x)) x s' Hr'). unfold V, R. rewrite (Rt_run (tau x) x s Hr).
      exact (step_lds_regs F prog (E x) (tau x) s s' bits d a off Hfs Hs').
    - intros x r Hx Hw Hc. destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]]. unfold R, R'.
      rewrite (Rt_run (S (tau x)) x s' Hr'), (Rt_run (tau x) x s Hr).
      exact (step_keep F prog (E x) (tau x) s s' ins r Hfs Hs' Hw Hc).
    - intros d src m mask Hi Hc Hm Hm32 x Hx. subst ins.
      destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]].
      assert (Hpx : (partner x m < n)%nat) by (apply partner_lt; assumption).
      destruct (Hthr (partner x m) Hpx) as [sp [sp' [Hrp [Hpcp _]]]].
      unfold A, R. rewrite (Rt_run (tau (partner x m)) (partner x m) sp Hrp).
      apply (cf_bfly _ _ _ _ _ _ _ _ _ Hfam x (tau x) (tau (partner x m)) s sp d src m mask Hx Hpx Hm32 Hr Hrp).
      + rewrite Hpcp, Hpc. reflexivity.
      + rewrite Hpc. exact Hc.
      + exact (proj1 (Hsame x (partner x m) Hx Hpx)).
      + intros z Hz Hwz. destruct (Hthr z Hz) as [sz [_ [Hrz [Hpcz _]]]].
        exists (tau z), sz. split; [exact Hrz|]. split; [rewrite Hpcz, Hpc; reflexivity|].
        split; [exact (proj1 (Hsame z x Hz Hx))|].
        pose proof (mask_ok_full F n R _ mask Hok Hm) as Hfull.
        destruct mask as [r|]; [|exact I]. cbn [mask_full]. rewrite <- (Rt_run (tau z) z sz Hrz). exact (Hfull z Hz).
    - intros d src mask Hi Hc Hm x y Hx Hy Hxy. subst ins.
      destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]].
      destruct (Hthr y Hy) as [sy [sy' [Hry [Hpcy _]]]].
      unfold A. apply (cf_redux _ _ _ _ _ _ _ _ _ Hfam x y (tau x) (tau y) s sy d src mask Hx Hy Hxy Hr Hry).
      + rewrite Hpcy, Hpc. reflexivity.
      + rewrite Hpc. exact Hc.
      + exact (proj1 (Hsame x y Hx Hy)).
      + intros z Hz Hwz. destruct (Hthr z Hz) as [sz [_ [Hrz [Hpcz _]]]].
        exists (tau z), sz. split; [exact Hrz|]. split; [rewrite Hpcz, Hpc; reflexivity|].
        split; [exact (proj1 (Hsame z x Hz Hx))|].
        pose proof (mask_ok_full F n R _ mask Hok Hm) as Hfull.
        destruct mask as [r|]; [|exact I]. cbn [mask_full]. rewrite <- (Rt_run (tau z) z sz Hrz). exact (Hfull z Hz).
    - intros bits d a off Hi Hclean x y Hx Hy Ha. subst ins.
      destruct (Hthr x Hx) as [s [s' [Hr [Hpc [Hfs [Hs' Hr']]]]]].
      destruct (Hthr y Hy) as [sy [sy' [Hry [Hpcy [Hfsy _]]]]].
      unfold V, R in *. rewrite (Rt_run (tau x) x s Hr), (Rt_run (tau y) y sy Hry) in *.
      destruct (Hsame x y Hx Hy) as [Hev [Hph _]].
      apply (cf_load _ _ _ _ _ _ _ _ _ Hfam x y (tau x) (tau y) s sy None bits d a off Hx Hy Hr Hry Hfs);
        try reflexivity; try assumption.
      + rewrite Hpcy, Hpc. reflexivity.
      + rewrite Ha. reflexivity.
      + intros z u Hz Hzu Hws. destruct (writes_shared_write z u _ _ Hws) as [sz [iz [[Hrz [Hfz [_ Hsw]]] _]]].
        rewrite (no_shared_write tau x s _ Hsame Hx Hr Hfs) with (z := z) (u := u) (sz := sz) (iz := iz) in Hsw;
          try assumption; [discriminate|].
        rewrite Hpc. exact Hclean.
      + (* the writes of earlier phases are ordered *)
        intros z1 u1 z2 u2 c Hz1 Hz2 Hne Hph12 Hlt _ Hw1 Hw2.
        assert (Hord : ordered = true) by (apply (clean_ordered x (tau x) s _ Hr Hfs); rewrite Hpc; exact Hclean).
        destruct (Hwold Hord) as [_ Hpair].
        apply (Hpair z1 u1 z2 u2 c Hz1 Hz2 Hne); [| |exact Hph12|exact Hw1|exact Hw2].
        * apply (phase_lt_step z1). rewrite (proj1 (proj2 (Hsame z1 x Hz1 Hx))). exact Hlt.
        * apply (phase_lt_step z2). rewrite (proj1 (proj2 (Hsame z2 x Hz2 Hx))). rewrite <- Hph12. exact Hlt. }
  assert (Hsucc : ss_at e (b, S i) =
    ss_transfer F tidr cols dirty (b, i) (phase_at cta main_P (b, i)) (ss_at e (b, i)) ins)
    by exact (ss_iter_succ F tidr cols cta dirty (block_body prog b) i b 0 (nth b main_P 0)
                (entry_state G b e) ins Hf).
  rewrite <- Hsucc in Htr.
  intros ins' Hf'.
  assert (Hentry : forall tg, (tg < length prog)%nat -> edge_ok G D tg (ss_at e (b, S i)) ->
            exists e', nth tg D None = Some e' /\ sstate_ok F n (ss_at e' (tg, 0%nat)) R').
  { intros tg Htg Hedge. exact (edge_entry_ok F n G D tg _ R' Hedge Htr). }
  unfold q' in *.
  destruct (next_pc_cases prog b i ins (R 0%nat) Hf) as [[tg [Htg Heq]]|[Heq|[[Hlt Heq]|[Hlen [Hft Heq]]]]];
    rewrite Heq in Hf' |- *; cbn [fst snd] in *.
  - apply Hentry; [exact (fetch_block prog _ _ _ Hf')|]. exact (Hjump tg Htg (fetch_block prog _ _ _ Hf')).
  - exfalso. pose proof (fetch_block prog _ _ _ Hf'). lia.
  - exists e. split; [exact He|exact Htr].
  - apply Hentry; [exact (fetch_block prog _ _ _ Hf')|]. exact (Hfall Hlen Hft (fetch_block prog _ _ _ Hf')).
Qed.

Lemma diverge_step : forall k tau b e g tg st edges, SyncInv k tau ->
  let L := pred (length (block_body prog b)) in
  let J := region_join prog b (length prog) tg in
  let R := region_blocks prog b J in
  (forall x, (x < n)%nat -> pos_at F prog E x (tau x) = Some (b, L)) ->
  nth b D None = Some e -> sstate_ok F n (ss_at e (b, L)) (fun x => Rt (tau x) x) ->
  last_instr (block_body prog b) = Some (IBra (Some g) tg) ->
  region_ok cta events prog b J R = true -> is_cta cta (b, L) = false -> is_cta events (b, L) = false ->
  ss_walk F tidr cols cta dirty b 0 (nth b main_P 0) (entry_state G b e) (block_body prog b) [] = None ->
  ss_walk F tidr cols cta dirty b 0 (nth b main_P 0) (entry_state G b e)
    (removelast (block_body prog b)) [] = Some (st, edges) ->
  edges_ok G prog D ((J, ss_top st (region_writes F prog R)) :: edges) ->
  exists tau', SyncInv (S k) tau'.
Proof.
  destruct check_parts as [Hn0 [Hn32 [Hcell [Hpres [HP HD]]]]].
  intros k tau b e g tg st edges Hinv L J R Hq He Hok Hl HR Hc1 Hc2 Hnone Hw Hedges.
  pose proof Hinv as [Hk [Hsame [Hcov [Hat Hwr]]]].
  set (W := region_writes F prog R).
  assert (Hrun : forall x, (x < n)%nat -> exists d s s', (0 < d)%nat /\
            run_at F prog E x (tau x) = Some s /\ run_at F prog E x (tau x + d) = Some s' /\
            pc_of s' = (J, 0%nat) /\ (forall r, ~ In r W -> regs s' r = regs s r) /\
            (forall u, (tau x <= u < tau x + d)%nat -> exists q, pos_at F prog E x u = Some q /\
               (q = (b, L) \/ inreg prog R q))).
  { intros x Hx. destruct (pos_run F prog E x (tau x) _ (Hq x Hx)) as [s [Hr Hpc]].
    destruct (Hends (E x) (cf_launch _ _ _ _ _ _ _ _ _ Hfam x Hx)) as [T [se [HT Hend]]].
    destruct (branch_run F prog cta events b J R HR (E x) T se (tau x) s g tg HT Hend Hr Hpc Hl)
      as [d [s' [Hd [Hr' [Hj [Hregs Hpos]]]]]].
    exists d, s, s'. split; [exact Hd|]. split; [exact Hr|]. split; [exact Hr'|]. split; [exact Hj|].
    split; [exact Hregs|exact Hpos]. }
  destruct (finite_choice _ n Hrun) as [choose Hchoose].
  set (tau' := fun x => (tau x + choose x)%nat).
  assert (Htau : forall x, (x < n)%nat -> exists d s s', (0 < d)%nat /\ tau' x = (tau x + d)%nat /\
            run_at F prog E x (tau x) = Some s /\ run_at F prog E x (tau' x) = Some s' /\
            pc_of s' = (J, 0%nat) /\ (forall r, ~ In r W -> regs s' r = regs s r) /\
            passes_none F prog E cta x (tau x) d /\ passes_none F prog E events x (tau x) d).
  { intros x Hx. destruct (Hchoose x Hx) as [s [s' [Hd [Hr [Hr' [Hj [Hregs Hpos]]]]]]].
    assert (Hquiet : forall Lst, (forall q, q = (b, L) \/ inreg prog R q -> is_cta Lst q = false) ->
              passes_none F prog E Lst x (tau x) (choose x)).
    { intros Lst HL u Hu. destruct (Hpos u Hu) as [q [Hp Hq']]. rewrite Hp. cbn [at_cta].
      exact (HL q Hq'). }
    exists (choose x), s, s'. split; [exact Hd|]. split; [reflexivity|]. split; [exact Hr|]. split; [exact Hr'|].
    split; [exact Hj|]. split; [exact Hregs|]. split; apply Hquiet; intros q [->|Hin];
      try assumption; exact (proj1 (region_not_waiting prog cta events b J R HR q Hin))
      || exact (proj2 (region_not_waiting prog cta events b J R HR q Hin)). }
  assert (HfL : fetch prog b L = Some (IBra (Some g) tg)) by (unfold fetch, L; rewrite <- last_instr_nth; exact Hl).
  exists tau'. split; [|split; [|split; [|split]]].
  5: { (* the writes stay ordered *)
       intros Hord. apply (diverge_writes k tau tau' b L R Hinv); [| | |exact Hord].
       - intros c Hc. unfold ord_regions. apply in_flat_map. exists b. split.
         + apply in_seq. split; [lia|]. exact (fetch_block prog b L _ HfL).
         + unfold diverged_blocks. rewrite He, ss_walk_at_same, Hnone, Hl. exact Hc.
       - intros i Hfi. rewrite HfL in Hfi. injection Hfi as <-. reflexivity.
       - intros x Hx. destruct (Hchoose x Hx) as [s [s' [Hd [Hr [Hr' [Hj [Hregs Hpos]]]]]]].
         destruct (Htau x Hx) as [d [_ [_ [_ [Hx' [_ [_ [_ [_ [Hcx _]]]]]]]]]].
         assert (Hd' : tau' x = (tau x + choose x)%nat) by reflexivity.
         split; [lia|]. split.
         + replace (tau' x - tau x)%nat with d by lia. exact Hcx.
         + intros u Hu. apply Hpos. lia. }
  - intros x Hx. destruct (Htau x Hx) as [d [_ [_ [Hd [-> _]]]]]. specialize (Hk x Hx). lia.
  - intros x y Hx Hy. destruct (Htau x Hx) as [dx [_ [_ [_ [Hx' [_ [_ [_ [_ [Hcx Hex]]]]]]]]]].
    destruct (Htau y Hy) as [dy [_ [_ [_ [Hy' [_ [_ [_ [_ [Hcy Hey]]]]]]]]]].
    rewrite Hx', Hy'. rewrite (evhist_stable F prog events E x (tau x) dx Hex), (evhist_stable F prog events E y (tau y) dy Hey).
    destruct (phase_stable F prog cta E x (tau x) dx Hcx) as [Hpx Hlx].
    destruct (phase_stable F prog cta E y (tau y) dy Hcy) as [Hpy Hly].
    rewrite Hpx, Hpy, Hlx, Hly. exact (Hsame x y Hx Hy).
  - intros x y t Hx Hy Ht. destruct (Htau x Hx) as [d [_ [_ [_ [Hx' [_ [_ [_ [_ [_ Hex]]]]]]]]]].
    rewrite Hx' in Ht. destruct (Nat.le_gt_cases t (tau x)) as [Hle|Hgt]; [exact (Hcov x y t Hx Hy Hle)|].
    replace t with (tau x + (t - tau x))%nat by lia.
    rewrite (evhist_stable F prog events E x (tau x) (t - tau x)) by (intros u Hu; apply Hex; lia).
    exact (Hcov x y (tau x) Hx Hy (le_n _)).
  - right. exists (J, 0%nat). split.
    + intros x Hx. destruct (Htau x Hx) as [d [s [s' [_ [_ [_ [Hr' [Hj _]]]]]]]].
      unfold pos_at, tpos. fold (run_at F prog E x (tau' x)). rewrite Hr'. cbn [option_map]. rewrite Hj. reflexivity.
    + intros ins' Hf'. cbn [fst snd] in *. assert (HJl : (J < length prog)%nat) by exact (fetch_block prog _ _ _ Hf').
      assert (Hst : st = ss_at e (b, L)).
      { destruct (ss_walk_spec F tidr cols cta dirty _ b 0 (nth b main_P 0) _ [] st edges Hw) as [Hst _].
        rewrite Hst. unfold ss_at. cbn [fst snd]. rewrite removelast_length. fold L.
        assert (Hne : block_body prog b <> []) by (intros Hb; rewrite Hb in Hl; discriminate).
        rewrite (app_removelast_last INop Hne) at 2. rewrite ss_iter_app; [reflexivity|].
        rewrite removelast_length. lia. }
      assert (Htop : sstate_ok F n (ss_top st W) (fun x => Rt (tau' x) x)).
      { rewrite Hst. apply (ss_top_ok F n _ W (fun x => Rt (tau x) x)); [exact Hn32|exact Hok|].
        intros x r Hx Hr. destruct (Htau x Hx) as [d [s [s' [_ [_ [Hrx [Hr' [_ [Hregs _]]]]]]]]].
        unfold Rt. rewrite Hr', Hrx. apply Hregs. intros Hin.
        assert (existsb (Pos.eqb r) W = true) by (apply existsb_exists; exists r; split; [exact Hin|apply Pos.eqb_refl]).
        congruence. }
      exact (edge_entry_ok F n G D J _ _ (Hedges J _ (or_introl eq_refl) HJl) Htop).
Qed.

Lemma inv_step : forall k tau, SyncInv k tau -> exists tau', SyncInv (S k) tau'.
Proof.
  destruct check_parts as [Hn0 [Hn32 [Hcell [Hpres [HP HD]]]]].
  intros k tau Hinv. pose proof Hinv as [Hk [Hsame [Hcov [Hat Hwr]]]].
  destruct Hat as [Hall|[q [Hq Hrel]]].
  - exists (fun x => S (tau x)). destruct (one_step k tau Hinv) as [H1 [H2 H3]].
    { intros x y Hx Hy. unfold pos_at, tpos. fold (run_at F prog E x (tau x)) (run_at F prog E y (tau y)).
      rewrite (Hall x Hx), (Hall y Hy). reflexivity. }
    split; [exact H1|]. split; [exact H2|]. split; [exact H3|].
    split; [|intros Hord; apply idle_writes; [intros x s i Hx [Hr _]; rewrite (Hall x Hx) in Hr; discriminate Hr|exact (Hwr Hord)]].
    left. intros x Hx.
    unfold run_at. rewrite run_succ. fold (run_at F prog E x (tau x)). rewrite (Hall x Hx). reflexivity.
  - destruct (fetch prog (fst q) (snd q)) as [ins|] eqn:Hf.
    2: { exists (fun x => S (tau x)). destruct (one_step k tau Hinv) as [H1 [H2 H3]].
         { intros x y Hx Hy. rewrite (Hq x Hx), (Hq y Hy). reflexivity. }
         split; [exact H1|]. split; [exact H2|]. split; [exact H3|].
         split; [|intros Hord; apply idle_writes; [|exact (Hwr Hord)]].
         2: { intros x s0 i0 Hx [Hr0 [Hf0 _]]. destruct (pos_run F prog E x (tau x) q (Hq x Hx)) as [s [Hr Hpc]].
              rewrite Hr in Hr0. injection Hr0 as <-.
              assert (Hfs : fetch prog (blk s) (idx s) = None) by (rewrite <- Hf, <- Hpc; reflexivity).
              rewrite Hfs in Hf0. discriminate Hf0. }
         left. intros x Hx.
         destruct (pos_run F prog E x (tau x) q (Hq x Hx)) as [s [Hr Hpc]].
         unfold run_at. rewrite run_succ. fold (run_at F prog E x (tau x)). rewrite Hr. apply step_none.
         rewrite <- Hf, <- Hpc. reflexivity. }
    destruct (Hrel ins eq_refl) as [e [He Hok]].
    destruct q as [b i]. cbn [fst snd] in *.
    assert (Hb : (b < length prog)%nat) by exact (fetch_block prog b i ins Hf).
    destruct (dep_verify_block F tidr cols G uniform cta events prog main_P dirty D b e HD Hb He)
      as [[stw [edges [Hwalk Hedges]]]|[g [tg [st [edges [Hnone [Hl [HR [Hc1 [Hc2 [Hw Hedges]]]]]]]]]]].
    + destruct (ss_walk_spec F tidr cols cta dirty (block_body prog b) b 0 (nth b main_P 0)
                  (entry_state G b e) [] stw edges Hwalk) as [Hstw [_ Hk']].
      destruct (Hk' i ins Hf) as [Hbranch Hjump].
      exists (fun x => S (tau x)). apply (single_step k tau b i ins e Hinv Hq Hf He Hok Hbranch).
      * intros tg Htg Htl. apply Hedges; [|exact Htl].
        destruct (falls_through (block_body prog b)); [right|]; exact (Hjump tg Htg).
      * intros Hlen Hft Hsb. apply Hedges; [|exact Hsb]. rewrite Hft. left. f_equal. rewrite Hstw.
        unfold ss_at. cbn [fst snd]. unfold block_len in Hlen. rewrite <- Hlen. reflexivity.
    + set (L := pred (length (block_body prog b))).
      assert (Hne : block_body prog b <> []) by (intros Hb0; rewrite Hb0 in Hl; discriminate).
      assert (Hi : (i < length (block_body prog b))%nat) by
        (apply nth_error_Some; unfold fetch in Hf; rewrite Hf; discriminate).
      destruct (Nat.eq_dec i L) as [HiL|HiL].
      * subst i. exact (diverge_step k tau b e g tg st edges Hinv Hq He Hok Hl HR Hc1 Hc2 Hnone Hw Hedges).
      * assert (HiL' : (i < L)%nat) by (unfold L in *; lia).
        destruct (ss_walk_spec F tidr cols cta dirty (removelast (block_body prog b)) b 0 (nth b main_P 0)
                    (entry_state G b e) [] st edges Hw) as [_ [_ Hk']].
        assert (Hsplit := app_removelast_last INop Hne).
        assert (Hnth : nth_error (removelast (block_body prog b)) i = Some ins).
        { unfold fetch in Hf. rewrite Hsplit in Hf. rewrite nth_error_app1 in Hf; [exact Hf|].
          rewrite removelast_length. exact HiL'. }
        assert (Hpre : forall j, (j <= L)%nat -> ss_iter F tidr cols cta dirty b 0 (nth b main_P 0)
                   (entry_state G b e) (removelast (block_body prog b)) j = ss_at e (b, j)).
        { intros j Hj. unfold ss_at. cbn [fst snd]. rewrite Hsplit at 2. symmetry. apply ss_iter_app.
          rewrite removelast_length. exact Hj. }
        destruct (Hk' i ins Hnth) as [Hbranch Hjump].
        rewrite (Hpre i ltac:(lia)) in Hbranch.
        exists (fun x => S (tau x)). apply (single_step k tau b i ins e Hinv Hq Hf He Hok Hbranch).
        -- intros tg' Htg Htl. apply Hedges; [|exact Htl]. right. rewrite <- (Hpre (S i) ltac:(lia)).
           exact (Hjump tg' Htg).
        -- intros Hlen. exfalso. unfold block_len in Hlen. unfold L in HiL'. lia.
Qed.

Theorem main_same_waits : same_waits F prog events n E.
Proof.
  intros x y t Hx Hy.
  assert (H : exists tau, SyncInv t tau).
  { induction t as [|t IH]; [exists (fun _ => 0%nat); exact inv_zero|].
    destruct IH as [tau Htau]. exact (inv_step t tau Htau). }
  destruct H as [tau [Hk [_ [Hcov _]]]]. exact (Hcov x y t Hx Hy (Hk x Hx)).
Qed.

End Main.

(* Every thread of a block whose oracles satisfy the three facts executes
   the same sequence of waiting sites, when every thread's run ends. *)
Definition shared_schedule_safe (F : lfacts) (uniform : reg -> bool) (tidr : reg) (n : nat)
    (prog : program) (cta events : list pc) (cols : list (pc * collective)) : Prop :=
  kernel_ends F prog ->
  forall E, cta_family F prog cta events cols n E uniform tidr -> same_waits F prog events n E.

Theorem shared_check_sound : forall F uniform tidr n prog cta events cols,
  shared_check F uniform tidr n prog cta events cols = true ->
  shared_schedule_safe F uniform tidr n prog cta events cols.
Proof.
  intros F uniform tidr n prog cta events cols H Hends E Hfam.
  unfold shared_check in H. cbv zeta in H. rewrite !andb_true_iff in H.
  destruct H as [[[[[H0 H1] H2] H3] H4] [H5 H6]].
  rewrite phase_verify_at_same in H5. rewrite phase_dirty_at_same in H6.
  apply Nat.ltb_lt in H0. apply Nat.eqb_eq in H2. apply negb_true_iff in H3. rewrite forallb_forall in H4.
  match type of H6 with context [if ?c then _ else _] => destruct c eqn:Hc end.
  - apply andb_true_iff in Hc. destruct Hc as [Hd Ho]. rewrite dep_verify_at_same in Hd.
    eapply (main_same_waits F prog cta events cols n E uniform tidr _ _); [|exact Hfam|exact Hends].
    split; [exact H0|]. split; [exact H2|]. split; [exact H3|]. split; [exact H4|]. split; [exact H5|].
    split; [exact Hd|]. split; [apply zsub_refl|]. right. exact Ho.
  - rewrite dep_verify_at_same in H6.
    eapply (main_same_waits F prog cta events cols n E uniform tidr all_dirty _); [|exact Hfam|exact Hends].
    split; [exact H0|]. split; [exact H2|]. split; [exact H3|]. split; [exact H4|]. split; [exact H5|].
    split; [exact H6|]. split; [|left; reflexivity].
    unfold zsub. change (Z.lnot all_dirty) with 0. rewrite Z.land_0_r. reflexivity.
Qed.
