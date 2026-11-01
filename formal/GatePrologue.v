(* GatePrologue.v: the lowered zeroing prologue of a kernel with shared
   memory, and what a thread does before it reaches the prologue's barrier.

   The gate requires every kernel to begin with the zeroing prologue that
   `prologue` in GateFront.v states on the PTX text (design decision 46):
   zero and the thread's linear index in a block, a loop for each dynamic
   shared variable (GateZero.v), the stores of each other shared variable in
   rounds, and a `bar.sync 0`. `prologue_of` recognizes the lowered form of
   this prologue in the program that the translation gives, from the launch
   facts and the binding alone, and names the barrier's position. It
   decides; nothing here trusts how it was selected.

   Proved, in the one-thread semantics of GateIsa.v, for a program that
   `prologue_of` recognizes and a thread whose launch values put its index
   in the block, each size in 64 bits, and each shared variable in the
   window:
   - `prologue_run`: the thread's run reaches the barrier at some step M,
     every earlier step is at a position before the barrier, and before M
     the thread stores zero at every byte of its share: in each shared
     variable, the elements of the store's width whose index has the
     thread's linear index as its remainder modulo the block's threads;
   - `region_quiet`: at every position before the barrier the instruction
     reads neither memory nor the results of unmodeled instructions, and
     writes nothing but registers and, by a store of the zero register,
     shared memory;
   - `after_bar_closed`: once past the barrier a run never returns to it,
     so a thread executes the barrier at most once.
   Every element of every shared variable is in the share of the thread of
   the block whose linear index is the element's remainder modulo the
   block's threads (`share_covers`), so the threads of a block store zero
   at every byte of its shared variables before any of them passes the
   barrier (StoreLaunch.v). *)

From Coq Require Import ZArith.ZArith.
From Coq Require Import PArith.PArith.
From Coq Require Import Arith.PeanoNat.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
From Coq Require Import Strings.String.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.
Require Import OpGate.GateDomain.
Require Import OpGate.GateMeaning.
Require Import OpGate.GateThread.
Require Import OpGate.GateFront.
Require Import OpGate.GateZero.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* The shape of the lowered prologue.                                  *)

(* A shared variable that the prologue clears in rounds: thread t of the N
   threads stores the elements t, t + N, ..., t + (full - 1) N, and the
   element t + full N when t < left. *)
Record static_var := { sv_base : reg; sv_full : nat; sv_left : Z }.

Record prologue := {
  pr_zero : reg; pr_index : reg; pr_address : reg; pr_pred : reg;
  pr_tx : reg; pr_ty : reg; pr_tz : reg;
  pr_bx : Z; pr_by : Z; pr_bz : Z;
  pr_width : Z;
  pr_loops : option zero_plan;
  pr_static : list static_var
}.

Definition pr_threads (ps : prologue) : Z := pr_bx ps * pr_by ps * pr_bz ps.

Definition pr_blocks (ps : prologue) : nat :=
  match pr_loops ps with None => O | Some zp => (3 * List.length (zp_bases zp))%nat end.

(* The bytes of a static variable. *)
Definition sv_bytes (ps : prologue) (v : static_var) : Z :=
  pr_width ps * (pr_threads ps * Z.of_nat (sv_full v) + sv_left v).

(* The plan of the index computation alone. *)
Definition index_plan (ps : prologue) : zero_plan :=
  {| zp_regs := {| zr_zero := pr_zero ps; zr_index := pr_index ps; zr_address := pr_address ps;
                   zr_count := pr_pred ps; zr_element := pr_pred ps; zr_pred := pr_pred ps |};
     zp_tx := pr_tx ps; zp_ty := pr_ty ps; zp_tz := pr_tz ps;
     zp_bx := pr_bx ps; zp_by := pr_by ps; zp_bz := pr_bz ps;
     zp_width := pr_width ps; zp_const := 0; zp_terms := []; zp_bases := [] |}.

Definition index_ir (ps : prologue) : list instr := zero_index_ir (index_plan ps).

(* The registers that the prologue writes before its barrier: its own, and
   those of the loops with their 64-bit scratch registers. *)
Definition pr_written (prog : program) (ps : prologue) : list reg :=
  [pr_zero ps; pr_index ps; pr_address ps; pr_pred ps] ++
  match pr_loops ps with
  | Some zp => zero_regs_list (zp_regs zp) ++
               List.concat (map (fun j => map fst (zero_temps_of prog zp j)) (seq 0 (List.length (zp_bases zp))))
  | None => []
  end.

(* The stores of one static variable. *)
Definition static_ir (ps : prologue) (v : static_var) : list instr :=
  let z := pr_zero ps in let t := pr_index ps in let a := pr_address ps in let p := pr_pred ps in
  let w := pr_width ps in let N := pr_threads ps in
  IMov None a (sv_base v) :: IMad None 32 a t (OImm w) (OReg a) ::
  map (fun k => IShA None KStore a (w * N * Z.of_nat k) w (Some (OReg z))) (seq 0 (sv_full v)) ++
  (if Z.eqb (sv_left v) 0 then [] else
   [ISetp None CLt 32 false p t (OImm (sv_left v));
    IShA (Some (false, p)) KStore a (w * N * Z.of_nat (sv_full v)) w (Some (OReg z))]).

Definition statics_ir (ps : prologue) : list instr := flat_map (static_ir ps) (pr_static ps).

(* Where the static stores start: after the index in block 0, or at the
   start of the block after the loops. *)
Definition static_start (ps : prologue) : nat * nat :=
  match pr_loops ps with None => (O, 6%nat) | Some _ => (pr_blocks ps, O) end.

(* The barrier's position. *)
Definition bar_of (ps : prologue) : nat * nat :=
  (fst (static_start ps), (snd (static_start ps) + List.length (statics_ir ps))%nat).

(* ------------------------------------------------------------------ *)
(* Recognition.                                                        *)

Definition pro_instr_eqb (a b : instr) : bool :=
  match a, b with
  | INop, INop => true
  | ISetp g c w s d x y, ISetp h c' w' s' d' x' y' =>
      guard_eqb g h && Nat.eqb w w' && Bool.eqb s s' && Pos.eqb d d' && Pos.eqb x x' && opnd_eqb y y' &&
      match c, c' with
      | CLt, CLt | CLe, CLe | CGt, CGt | CGe, CGe | CEq, CEq | CNe, CNe => true
      | _, _ => false
      end
  | _, _ => zero_instr_eqb a b
  end.

Fixpoint prefix_eqb (expected actual : list instr) : bool :=
  match expected, actual with
  | [], _ => true
  | e :: es, x :: xs => pro_instr_eqb e x && prefix_eqb es xs
  | _ :: _, [] => false
  end.

Definition index_regs_of (prog : program) : option (reg * reg * reg) :=
  match block_body prog 0 with
  | IImm None z _ :: IMov None t _ :: IMov None a _ :: _ => Some (z, t, a)
  | _ => None
  end.

(* The width of the stores, from the first address computation after the
   index: the loop body's for a prologue with loops, else the first
   static variable's. *)
Definition width_of (prog : program) (m : nat) : option Z :=
  match m with
  | O => match skipn 7 (block_body prog 0) with
         | IMad None 32 _ _ (OImm w) _ :: _ => Some w
         | _ => None
         end
  | _ => match block_body prog 2 with
         | _ :: IMad None 32 _ _ (OImm w) _ :: _ => Some w
         | _ => None
         end
  end.

(* The predicate register, from the head of the first loop or from the
   first comparison of the static stores. *)
(* The prologue's predicate register: the one its first comparison writes,
   or, when every round is full and no loop runs, a register apart from the
   registers the prologue names, which the prologue then never touches: a
   declared 1-bit register, or a register above every declared one. *)
Definition spare_pred (F : lfacts) (taken : list reg) : reg :=
  match find (fun e => Nat.eqb (snd e) 1 && negb (existsb (Pos.eqb (fst e)) taken)) (PM.elements (lf_widths F)) with
  | Some (r, _) => r
  | None => Pos.succ (fold_left Pos.max (map fst (PM.elements (lf_widths F)) ++ taken) 1%positive)
  end.

Definition pred_of (F : lfacts) (prog : program) (m : nat) (taken : list reg) : reg :=
  match m with
  | O => match find (fun i => match i with ISetp None CLt 32 false _ _ _ => true | _ => false end)
                    (skipn 6 (block_body prog 0)) with
         | Some (ISetp _ _ _ _ p _ _) => p
         | _ => spare_pred F taken
         end
  | _ => match block_body prog 1 with
         | ISetp None _ _ _ p _ _ :: _ => p
         | _ => spare_pred F taken
         end
  end.

Definition loop_bases (prog : program) (m : nat) : option (list reg) :=
  option_all (map (fun j => match block_body prog (3 * j + 2) with
                            | IMov None _ b :: _ => Some b
                            | _ => None
                            end) (seq 0 m)).

(* The shared variables with sizes among their bytes, and those of fixed
   positive bytes. *)
Definition dynamic_vars (F : lfacts) : list (reg * form) :=
  filter (fun v => negb (null (ft (snd v)))) (lf_shared F).
Definition static_vars (F : lfacts) : list (reg * form) :=
  filter (fun v => null (ft (snd v)) && Z.ltb 0 (flo (snd v))) (lf_shared F).

Definition static_of (w N : Z) (v : reg * form) : option static_var :=
  let bytes := flo (snd v) in
  if Z.eqb (fhi (snd v)) bytes && Z.eqb (bytes mod w) 0
  then Some {| sv_base := fst v; sv_full := Z.to_nat (bytes / w / N); sv_left := (bytes / w) mod N |}
  else None.

(* The form of a dynamic variable states the count of a loop times the
   width: the constant and each coefficient, divided by the width. *)
Definition dynamic_form_ok (zp : zero_plan) (f : form) : bool :=
  Z.eqb (flo f) (zp_const zp * zp_width zp) && Z.eqb (fhi f) (flo f) &&
  (fix same (ts : terms) (ps : list (reg * Z)) : bool :=
     match ts, ps with
     | [], [] => true
     | ((s, EX), c) :: ts', (s', c') :: ps' => Pos.eqb s s' && Z.eqb c (c' * zp_width zp) && same ts' ps'
     | _, _ => false
     end) (ft f) (zp_terms zp).

(* A dynamic variable's extent, at launch values init, is the loop's count
   of elements times the width. *)
Lemma dynamic_form_value : forall zp f wd init, dynamic_form_ok zp f = true ->
    tsum wd init (ft f) + flo f = zero_elements zp init * zp_width zp.
Proof.
  intros zp f wd init H. unfold dynamic_form_ok in H. rewrite !andb_true_iff in H. destruct H as [[Hlo _] Hsame].
  apply Z.eqb_eq in Hlo.
  assert (Hgen : tsum wd init (ft f) = zero_sum init (zp_terms zp) * zp_width zp).
  { revert Hsame. generalize (zp_terms zp) as ps. generalize (ft f) as ts.
    induction ts as [| [[s e] c] ts IH]; intros ps Hsame; destruct ps as [| [s' c'] ps]; cbn in Hsame;
      try discriminate Hsame; try (destruct e; discriminate Hsame).
    - cbn [tsum zero_sum]. ring.
    - destruct e; try discriminate Hsame. rewrite !andb_true_iff in Hsame. destruct Hsame as [[Hs Hc] Hrest].
      apply Pos.eqb_eq in Hs. apply Z.eqb_eq in Hc. subst s' c. cbn [tsum zero_sum atom_val].
      rewrite (IH ps Hrest). ring. }
  unfold zero_elements. rewrite Hgen, Hlo. ring.
Qed.

Definition reg_in (r : reg) (l : list reg) : bool := existsb (Pos.eqb r) l.

(* A fixed register whose declared bounds keep its value, plus bytes,
   inside the window. *)
Definition bounded_in (F : lfacts) (r : reg) (bytes : Z) : bool :=
  is_fixed F r &&
  match lookup (lf_bounds F) r with
  | Some (lo, hi) => Z.leb 0 lo && Z.leb (hi + bytes) window_size
  | None => false
  end.

(* A fixed register whose declared bounds keep its value in [0, n). *)
Definition bounded_below (F : lfacts) (r : reg) (n : Z) : bool :=
  is_fixed F r &&
  match lookup (lf_bounds F) r with
  | Some (lo, hi) => Z.leb 0 lo && Z.ltb hi n
  | None => false
  end.

Definition branches_after (prog : program) (b : nat) : bool :=
  forallb (fun body => forallb (fun i => match i with IBra _ tg => Nat.ltb b tg | _ => true end) body)
          (skipn b prog).

(* The static shared variables in the order of the prologue's rounds: the
   order in which the prologue moves their addresses into its address
   register, then any the prologue does not name. *)
Definition ordered_vars (vars : list (reg * form)) (order : list reg) : list (reg * form) :=
  flat_map (fun u => filter (fun v => Pos.eqb (fst v) u) vars) order ++
  filter (fun v => negb (reg_in (fst v) order)) vars.

Definition body_order (prog : program) (a : reg) (start : nat * nat) : list reg :=
  flat_map (fun i => match i with IMov None d s => if Pos.eqb d a then [s] else [] | _ => [] end)
           (skipn (snd start) (block_body prog (fst start))).

Lemma ordered_vars_in : forall vars order x, In x (ordered_vars vars order) <-> In x vars.
Proof.
  intros vars order x. unfold ordered_vars. rewrite in_app_iff, in_flat_map. split.
  - intros [(u & _ & Hx) | Hx]; apply filter_In in Hx; exact (proj1 Hx).
  - intros Hx. destruct (reg_in (fst x) order) eqn:Hr.
    + unfold reg_in in Hr. apply existsb_exists in Hr. destruct Hr as [u [Hu Heq]]. apply Pos.eqb_eq in Heq. subst u.
      left. exists (fst x). split; [exact Hu|]. apply filter_In. split; [exact Hx|apply Pos.eqb_refl].
    + right. apply filter_In. split; [exact Hx|rewrite Hr; reflexivity].
Qed.

(* The category of a register, as the binding names it: a thread index in
   dimension d, or a platform value (a shared variable's address or a
   window's base). *)
Definition is_tid_cat (F : lfacts) (Mf : binding) (r : reg) (d : nat) : bool :=
  match category_reg F Mf r with Some (CTid d') => Nat.eqb d' d | _ => false end.

Definition is_platform_cat (F : lfacts) (Mf : binding) (r : reg) : bool :=
  match category_reg F Mf r with Some CPlatform => true | _ => false end.

Lemma is_tid_cat_spec : forall F Mf r d, is_tid_cat F Mf r d = true -> category_reg F Mf r = Some (CTid d).
Proof.
  intros F Mf r d H. unfold is_tid_cat in H.
  destruct (category_reg F Mf r) as [[k | i h | d' | d' | d' | d' | |] |]; try discriminate H.
  apply Nat.eqb_eq in H. subst d'. reflexivity.
Qed.

Lemma is_platform_cat_spec : forall F Mf r, is_platform_cat F Mf r = true -> category_reg F Mf r = Some CPlatform.
Proof.
  intros F Mf r H. unfold is_platform_cat in H.
  destruct (category_reg F Mf r) as [[k | i h | d' | d' | d' | d' | |] |]; try discriminate H. reflexivity.
Qed.

Definition prologue_of (F : lfacts) (Mf : binding) (prog : program) : option prologue :=
  let '(bx, by_, bz) := b_block Mf in
  let m := List.length (dynamic_vars F) in
  match reg_named F "%tid.x", reg_named F "%tid.y", reg_named F "%tid.z", index_regs_of prog,
        width_of prog m with
  | Some tx, Some ty, Some tz, Some (z, t, a), Some w =>
      let N := bx * by_ * bz in
      let p := pred_of F prog m (z :: t :: a :: tx :: ty :: tz :: map fst (lf_shared F)) in
      let loops :=
        match m with
        | O => Some None
        | _ => match zero_regs_of prog, zero_terms_of Mf F w, loop_bases prog m with
               | Some zr, Some terms, Some bases =>
                   Some (Some {| zp_regs := zr; zp_tx := tx; zp_ty := ty; zp_tz := tz;
                                 zp_bx := bx; zp_by := by_; zp_bz := bz; zp_width := w;
                                 zp_const := af_const (b_shared Mf) / w; zp_terms := terms; zp_bases := bases |})
               | _, _, _ => None
               end
        end in
      let order := match loops with
                   | Some None => body_order prog a (O, 6%nat)
                   | Some (Some zp) => body_order prog a (3 * List.length (zp_bases zp), O)%nat
                   | None => []
                   end in
      match loops, option_all (map (static_of w N) (ordered_vars (static_vars F) order)) with
      | Some loops, Some statics =>
          let ps := {| pr_zero := z; pr_index := t; pr_address := a; pr_pred := p;
                       pr_tx := tx; pr_ty := ty; pr_tz := tz; pr_bx := bx; pr_by := by_; pr_bz := bz;
                       pr_width := w; pr_loops := loops; pr_static := statics |} in
          let own := [z; t; a; p] in
          let read := tx :: ty :: tz :: map fst (lf_shared F) in
          let loops_ok :=
            match loops with
            | None => zero_instrs_eqb (firstn 6 (block_body prog 0)) (index_ir ps) &&
                      Pos.eqb (zr_zero (zp_regs (index_plan ps))) z
            | Some zp => zero_program_ok (wdn F) zp prog &&
                         Pos.eqb (zr_zero (zp_regs zp)) z && Pos.eqb (zr_index (zp_regs zp)) t &&
                         Pos.eqb (zr_address (zp_regs zp)) a && Pos.eqb (zr_pred (zp_regs zp)) p &&
                         Z.eqb (af_const (b_shared Mf) mod w) 0 &&
                         forallb (fun x => Z.eqb (snd x mod w) 0) (af_terms (b_shared Mf)) &&
                         forallb (fun v => reg_in (fst v) (zp_bases zp) && dynamic_form_ok zp (snd v) &&
                                           bounded_in F (fst v) shared_max)
                                 (dynamic_vars F) &&
                         forallb (fun b => existsb (fun v => Pos.eqb (fst v) b) (dynamic_vars F)) (zp_bases zp) &&
                         defined (size_terms F Mf (af_terms (b_shared Mf)))
            end in
          let cats_ok :=
            is_tid_cat F Mf tx O && is_tid_cat F Mf ty 1%nat && is_tid_cat F Mf tz 2%nat &&
            forallb (fun v => is_platform_cat F Mf (fst v)) (lf_shared F) in
          if loops_ok && cats_ok &&
             prefix_eqb (statics_ir ps ++ [INop])
                        (skipn (snd (static_start ps)) (block_body prog (fst (static_start ps)))) &&
             Nat.ltb (pr_blocks ps) (List.length prog) &&
             forallb (fun v => bounded_in F (sv_base v) (sv_bytes ps v)) statics &&
             bounded_below F tx bx && bounded_below F ty by_ && bounded_below F tz bz &&
             Z.ltb 0 bx && Z.ltb 0 by_ && Z.ltb 0 bz && Z.ltb N (2 ^ 31) && Z.ltb 0 w && Z.ltb w (2 ^ 31) &&
             Nat.eqb (wdn F z) 32 && Nat.eqb (wdn F t) 32 && Nat.eqb (wdn F a) 32 && Nat.leb 1 (wdn F p) &&
             pos_nodup own && pos_apart own read &&
             forallb (fun v => negb (reg_in (fst v) (pr_written prog ps))) (lf_shared F) &&
             branches_after prog (pr_blocks ps)
          then Some ps else None
      | _, _ => None
      end
  | _, _, _, _, _ => None
  end.

(* A kernel with shared variables has a recognized prologue; one without
   reads no shared memory (`private_in` in GateMeaning.v). *)
Definition prologue_ok (F : lfacts) (Mf : binding) (prog : program) : bool :=
  match lf_shared F with
  | [] => true
  | _ => match prologue_of F Mf prog with Some _ => true | None => false end
  end.

(* ------------------------------------------------------------------ *)
(* Positions.                                                          *)

Definition pc := (nat * nat)%type.

(* A position before the barrier: in a block of the loops, or in the
   barrier's block before it. *)
Definition before_bar (ps : prologue) (q : pc) : Prop :=
  (fst q < fst (bar_of ps))%nat \/ (fst q = fst (bar_of ps) /\ snd q < snd (bar_of ps))%nat.

Definition after_bar (ps : prologue) (q : pc) : Prop :=
  (fst (bar_of ps) < fst q)%nat \/ (fst q = fst (bar_of ps) /\ snd (bar_of ps) < snd q)%nat.

(* An instruction of the prologue before its barrier: one whose effect is
   a function of the registers, and a store of the zero register to shared
   memory. *)
Definition quiet (z : reg) (i : instr) : bool :=
  match i with
  | IImm None _ _ | IMov None _ _ | IMad None _ _ _ _ _ | ICvt None _ _ _ _ _
  | IBin None _ _ _ _ _ | ISetp None _ _ _ _ _ _ | IBra _ _ => true
  | IShA _ KStore _ _ _ (Some (OReg r)) => Pos.eqb r z
  | _ => false
  end.

(* The thread's linear index in its block, and its share of each shared
   variable: the elements whose index has that remainder modulo the
   block's threads. *)
Definition thread_of (ps : prologue) (init : reg -> Z) : Z :=
  (init (pr_tz ps) * pr_by ps + init (pr_ty ps)) * pr_bx ps + init (pr_tx ps).

Definition dyn_count (ps : prologue) (init : reg -> Z) : Z :=
  match pr_loops ps with Some zp => zero_elements zp init | None => 0 end.

Definition share_at (ps : prologue) (init : reg -> Z) (T x : Z) : Prop :=
  (exists v, In v (pr_static ps) /\ init (sv_base v) <= x < init (sv_base v) + sv_bytes ps v /\
             ((x - init (sv_base v)) / pr_width ps) mod pr_threads ps = T) \/
  (exists zp B, pr_loops ps = Some zp /\ In B (zp_bases zp) /\
                init B <= x < init B + dyn_count ps init * pr_width ps /\
                ((x - init B) / pr_width ps) mod pr_threads ps = T).

Definition share (ps : prologue) (init : reg -> Z) (x : Z) : Prop := share_at ps init (thread_of ps init) x.

(* The bytes of the shared variables, at the launch values. *)
Definition fval (F : lfacts) (init : reg -> Z) (f : form) : Z := tsum (wdn F) init (ft f) + flo f.

Definition declared (F : lfacts) (init : reg -> Z) (x : Z) : Prop :=
  exists u f, In (u, f) (lf_shared F) /\ init u <= x < init u + fval F init f.

(* ------------------------------------------------------------------ *)
(* What recognition states.                                            *)

Lemma pro_instr_eqb_eq : forall a b, pro_instr_eqb a b = true -> a = b.
Proof.
  intros a b H. destruct a, b; cbn in H; try discriminate; try (apply zero_instr_eqb_eq; exact H);
    try reflexivity.
  rewrite !andb_true_iff in H. destruct H as [[[[[[Hg Hw] Hs] Hd] Hx] Hy] Hc].
  apply guard_eqb_eq in Hg. apply Nat.eqb_eq in Hw. apply Bool.eqb_prop in Hs. apply Pos.eqb_eq in Hd.
  apply Pos.eqb_eq in Hx. apply opnd_eqb_eq in Hy. subst.
  destruct c, c0; try discriminate; reflexivity.
Qed.

Lemma prefix_eqb_app : forall e l, prefix_eqb e l = true -> exists rest, l = e ++ rest.
Proof.
  induction e as [|x e IH]; intros l H; [exists l; reflexivity|].
  destruct l as [|y l]; [discriminate|]. cbn [prefix_eqb] in H. apply andb_true_iff in H.
  destruct H as [Hx Hl]. apply pro_instr_eqb_eq in Hx. subst y.
  destruct (IH l Hl) as [rest ->]. exists rest. reflexivity.
Qed.

Lemma option_all_in : forall {A B} (f : A -> option B) l xs,
  option_all (map f l) = Some xs ->
  (forall x, In x xs -> exists y, In y l /\ f y = Some x) /\
  (forall y, In y l -> exists x, In x xs /\ f y = Some x).
Proof.
  intros A B f l. induction l as [|y l IH]; intros xs H.
  - cbn in H. injection H as <-. split; intros x Hx; destruct Hx.
  - cbn [map option_all] in H. destruct (f y) as [x|] eqn:Hy; [|discriminate].
    destruct (option_all (map f l)) as [xs'|] eqn:Hl; [|discriminate]. injection H as <-.
    destruct (IH xs' eq_refl) as [H1 H2]. split.
    + intros x' [<-|Hx']; [exists y; split; [left; reflexivity|exact Hy]|].
      destruct (H1 x' Hx') as [y' [Hin Hf]]. exists y'. split; [right; exact Hin|exact Hf].
    + intros y' [<-|Hy']; [exists x; split; [left; reflexivity|exact Hy]|].
      destruct (H2 y' Hy') as [x' [Hin Hf]]. exists x'. split; [right; exact Hin|exact Hf].
Qed.

Lemma option_all_length : forall {A B} (f : A -> option B) l xs,
  option_all (map f l) = Some xs -> List.length xs = List.length l.
Proof.
  intros A B f l. induction l as [|y l IH]; intros xs H; cbn [map option_all] in H.
  - injection H as <-. reflexivity.
  - destruct (f y); [|discriminate]. destruct (option_all (map f l)) as [xs'|] eqn:Hl; [|discriminate].
    injection H as <-. cbn [List.length]. rewrite (IH xs' eq_refl). reflexivity.
Qed.

Lemma reg_in_spec : forall r l, reg_in r l = true -> In r l.
Proof. intros r l H. unfold reg_in in H. apply existsb_exists in H. destruct H as [x [Hx E]]. apply Pos.eqb_eq in E. subst. exact Hx. Qed.

Lemma static_of_some : forall w N v sv, 0 < N -> static_of w N v = Some sv ->
  sv_base sv = fst v /\ ft (snd v) = ft (snd v) /\ fhi (snd v) = flo (snd v) /\ flo (snd v) mod w = 0 /\
  sv_full sv = Z.to_nat (flo (snd v) / w / N) /\ sv_left sv = (flo (snd v) / w) mod N /\
  0 <= sv_left sv < N.
Proof.
  intros w N v sv HN H. unfold static_of in H.
  destruct (Z.eqb (fhi (snd v)) (flo (snd v)) && Z.eqb (flo (snd v) mod w) 0) eqn:E; [|discriminate].
  rewrite andb_true_iff, !Z.eqb_eq in E. injection H as <-. cbn [sv_base sv_full sv_left].
  repeat split; try tauto; try reflexivity; apply Z.mod_pos_bound; exact HN.
Qed.

(* The facts that the recognized shape gives. *)
Record shape (F : lfacts) (Mf : binding) (prog : program) (ps : prologue) : Prop := {
  sh_tx : reg_named F "%tid.x" = Some (pr_tx ps);
  sh_ty : reg_named F "%tid.y" = Some (pr_ty ps);
  sh_tz : reg_named F "%tid.z" = Some (pr_tz ps);
  sh_block : b_block Mf = (pr_bx ps, pr_by ps, pr_bz ps);
  sh_index : firstn 6 (block_body prog 0) = index_ir ps;
  sh_loops : forall zp, pr_loops ps = Some zp ->
    zero_form (wdn F) zp prog /\ zp_bases zp <> [] /\
    zr_zero (zp_regs zp) = pr_zero ps /\ zr_index (zp_regs zp) = pr_index ps /\
    zr_address (zp_regs zp) = pr_address ps /\ zr_pred (zp_regs zp) = pr_pred ps /\
    zp_tx zp = pr_tx ps /\ zp_ty zp = pr_ty ps /\ zp_tz zp = pr_tz ps /\
    zp_bx zp = pr_bx ps /\ zp_by zp = pr_by ps /\ zp_bz zp = pr_bz ps /\ zp_width zp = pr_width ps /\
    (forall B, In B (zp_bases zp) ->
       exists f, In (B, f) (lf_shared F) /\ dynamic_form_ok zp f = true /\ bounded_in F B shared_max = true);
  sh_no_loops : pr_loops ps = None -> dynamic_vars F = [];
  sh_static : exists rest,
    skipn (snd (static_start ps)) (block_body prog (fst (static_start ps))) = statics_ir ps ++ INop :: rest;
  sh_blocks : (pr_blocks ps < List.length prog)%nat;
  sh_static_vars : forall v, In v (pr_static ps) ->
    exists f, In (sv_base v, f) (lf_shared F) /\ ft f = [] /\ flo f = sv_bytes ps v /\ fhi f = flo f /\
              bounded_in F (sv_base v) (sv_bytes ps v) = true;
  sh_left : forall v, In v (pr_static ps) -> 0 <= sv_left v < pr_threads ps;
  sh_covered : forall u f, In (u, f) (lf_shared F) ->
    (ft f = [] /\ flo f <= 0) \/
    (exists v, In v (pr_static ps) /\ sv_base v = u /\ ft f = [] /\ flo f = sv_bytes ps v) \/
    (exists zp, pr_loops ps = Some zp /\ In u (zp_bases zp) /\ dynamic_form_ok zp f = true);
  sh_tids : bounded_below F (pr_tx ps) (pr_bx ps) = true /\ bounded_below F (pr_ty ps) (pr_by ps) = true /\
            bounded_below F (pr_tz ps) (pr_bz ps) = true;
  sh_dims : 0 < pr_bx ps /\ 0 < pr_by ps /\ 0 < pr_bz ps /\ pr_threads ps < 2 ^ 31 /\
            0 < pr_width ps < 2 ^ 31;
  sh_widths : wdn F (pr_zero ps) = 32%nat /\ wdn F (pr_index ps) = 32%nat /\
              wdn F (pr_address ps) = 32%nat /\ (1 <= wdn F (pr_pred ps))%nat;
  sh_distinct : NoDup [pr_zero ps; pr_index ps; pr_address ps; pr_pred ps];
  sh_apart : forall r, In r (pr_tx ps :: pr_ty ps :: pr_tz ps :: map fst (lf_shared F)) ->
             ~ In r [pr_zero ps; pr_index ps; pr_address ps; pr_pred ps];
  sh_written_apart : forall r, In r (map fst (lf_shared F)) -> ~ In r (pr_written prog ps);
  sh_branches : branches_after prog (pr_blocks ps) = true;
  sh_tid_cat : category_reg F Mf (pr_tx ps) = Some (CTid O) /\ category_reg F Mf (pr_ty ps) = Some (CTid 1%nat) /\
               category_reg F Mf (pr_tz ps) = Some (CTid 2%nat);
  sh_shared_cat : forall u f, In (u, f) (lf_shared F) -> category_reg F Mf u = Some CPlatform;
  sh_sizes : forall zp, pr_loops ps = Some zp ->
    zero_terms_of Mf F (pr_width ps) = Some (zp_terms zp) /\ zp_const zp = af_const (b_shared Mf) / pr_width ps /\
    af_const (b_shared Mf) mod pr_width ps = 0 /\
    (forall k c, In (k, c) (af_terms (b_shared Mf)) -> c mod pr_width ps = 0) /\
    (exists ts, size_terms F Mf (af_terms (b_shared Mf)) = Some ts)
}.

Lemma dynamic_vars_in : forall F u f, In (u, f) (dynamic_vars F) <-> In (u, f) (lf_shared F) /\ ft f <> [].
Proof.
  intros F u f. unfold dynamic_vars. rewrite filter_In. cbn [snd]. split.
  - intros [Hin Hn]. split; [exact Hin|]. apply negb_true_iff in Hn. destruct (ft f); [discriminate|].
    discriminate.
  - intros [Hin Hn]. split; [exact Hin|]. apply negb_true_iff. destruct (ft f); [contradiction|reflexivity].
Qed.

Lemma static_vars_in : forall F u f, In (u, f) (static_vars F) <-> In (u, f) (lf_shared F) /\ ft f = [] /\ 0 < flo f.
Proof.
  intros F u f. unfold static_vars. rewrite filter_In. cbn [snd]. rewrite andb_true_iff, Z.ltb_lt. split.
  - intros [Hin [Hn Hpos]]. split; [exact Hin|]. split; [|exact Hpos]. destruct (ft f); [reflexivity|discriminate].
  - intros [Hin [Hn Hpos]]. split; [exact Hin|]. split; [|exact Hpos]. rewrite Hn. reflexivity.
Qed.

Lemma prologue_shape : forall F Mf prog ps, prologue_of F Mf prog = Some ps -> shape F Mf prog ps.
Proof.
  intros F Mf prog ps H. unfold prologue_of in H.
  destruct (b_block Mf) as [[bx by_] bz] eqn:Hblock.
  destruct (reg_named F "%tid.x") as [tx|] eqn:Htx; [|discriminate].
  destruct (reg_named F "%tid.y") as [ty|] eqn:Hty; [|discriminate].
  destruct (reg_named F "%tid.z") as [tz|] eqn:Htz; [|discriminate].
  destruct (index_regs_of prog) as [[[z t] a]|] eqn:Hidx; [|discriminate].
  destruct (width_of prog (List.length (dynamic_vars F))) as [w|] eqn:Hw; [|discriminate].
  set (N := bx * by_ * bz) in *.
  set (p := pred_of F prog (List.length (dynamic_vars F)) (z :: t :: a :: tx :: ty :: tz :: map fst (lf_shared F))) in *.
  match type of H with context [match ?L with Some _ => _ | None => None end] =>
    destruct L as [loops|] eqn:Hloops; [|discriminate] end.
  match type of H with context [option_all ?S] => destruct (option_all S) as [statics|] eqn:Hstatics; [|discriminate] end.
  match type of H with (if ?C then _ else _) = _ => destruct C eqn:Hc; [|discriminate] end.
  injection H as <-.
  rewrite !andb_true_iff in Hc.
  destruct Hc as [[[[[[[[[[[[[[[[[[[[[Hlo Hcat] Hpre] Hblk] Hsb] Htxb] Htyb] Htzb] Hbx] Hby] Hbz] HN] Hw0] Hw1]
    Hwz] Hwt] Hwa] Hwp] Hnodup] Hapart] Hwr] Hbr].
  apply Nat.ltb_lt in Hblk. apply Z.ltb_lt in Hbx, Hby, Hbz, HN, Hw0, Hw1.
  apply Nat.eqb_eq in Hwz, Hwt, Hwa. apply Nat.leb_le in Hwp.
  rewrite forallb_forall in Hsb.
  assert (HN0 : 0 < N) by (apply Z.mul_pos_pos; [apply Z.mul_pos_pos|]; lia).
  destruct (option_all_in _ _ _ Hstatics) as [Hst1 Hst2].
  (* the loops *)
  assert (Hloop_facts :
    (forall zp, loops = Some zp ->
       zero_form (wdn F) zp prog /\ zp_bases zp <> [] /\
       zr_zero (zp_regs zp) = z /\ zr_index (zp_regs zp) = t /\ zr_address (zp_regs zp) = a /\
       zr_pred (zp_regs zp) = p /\ zp_tx zp = tx /\ zp_ty zp = ty /\ zp_tz zp = tz /\
       zp_bx zp = bx /\ zp_by zp = by_ /\ zp_bz zp = bz /\ zp_width zp = w /\
       (forall B, In B (zp_bases zp) ->
          exists f, In (B, f) (lf_shared F) /\ dynamic_form_ok zp f = true /\ bounded_in F B shared_max = true)) /\
    (loops = None -> dynamic_vars F = []) /\
    (loops = None -> firstn 6 (block_body prog 0) = zero_index_ir (index_plan
       {| pr_zero := z; pr_index := t; pr_address := a; pr_pred := p; pr_tx := tx; pr_ty := ty; pr_tz := tz;
          pr_bx := bx; pr_by := by_; pr_bz := bz; pr_width := w; pr_loops := loops; pr_static := statics |})) /\
    (forall zp, loops = Some zp ->
       zero_terms_of Mf F w = Some (zp_terms zp) /\ zp_const zp = af_const (b_shared Mf) / w /\
       af_const (b_shared Mf) mod w = 0 /\ (forall k c, In (k, c) (af_terms (b_shared Mf)) -> c mod w = 0) /\
       (exists ts, size_terms F Mf (af_terms (b_shared Mf)) = Some ts))).
  { destruct (List.length (dynamic_vars F)) as [|m'] eqn:Hm.
    - injection Hloops as <-. rewrite andb_true_iff in Hlo. destruct Hlo as [Hlo _].
      split; [intros zp E; discriminate E|]. split; [| split; [| intros zp E; discriminate E]].
      + intros _. destruct (dynamic_vars F); [reflexivity|discriminate].
      + intros _. apply zero_instrs_eqb_eq. exact Hlo.
    - destruct (zero_regs_of prog) as [zr|]; [|discriminate].
      destruct (zero_terms_of Mf F w) as [terms|] eqn:Hterms; [|discriminate].
      destruct (loop_bases prog (S m')) as [bases|] eqn:Hbases; [|discriminate].
      injection Hloops as <-. split; [| split; [intros E; discriminate E | split; [intros E; discriminate E |]]].
      2: { intros zp E. injection E as <-. cbn [zp_terms zp_const].
           rewrite !andb_true_iff in Hlo. destruct Hlo as [[[[[[[[[_ _] _] _] _] Hc0] Hcs] _] _] Hst].
           split; [first [exact Hterms | reflexivity]|]. split; [reflexivity|]. split; [apply Z.eqb_eq; exact Hc0|]. split.
           - intros k c Hin. rewrite forallb_forall in Hcs. specialize (Hcs (k, c) Hin). cbn [snd] in Hcs.
             apply Z.eqb_eq in Hcs. exact Hcs.
           - unfold defined in Hst. destruct (size_terms F Mf (af_terms (b_shared Mf))) as [ts|];
               [exists ts; reflexivity | discriminate Hst]. }
      intros zp E. injection E as <-. cbn [zp_regs zp_tx zp_ty zp_tz zp_bx zp_by zp_bz zp_width zp_bases].
      rewrite !andb_true_iff in Hlo.
      destruct Hlo as [[[[[[[[[Hok Hz] Ht] Ha] Hp] Hc0] Hcs] Hdyn] Hall] _].
      apply Pos.eqb_eq in Hz, Ht, Ha, Hp.
      split; [exact (zero_program_ok_form _ _ _ Hok)|]. split.
      { intros Hnil. unfold loop_bases in Hbases.
        pose proof (option_all_length _ _ _ Hbases) as Hlen. rewrite Hnil, seq_length in Hlen.
        discriminate. }
      repeat (split; [assumption || reflexivity|]).
      intros B HB. rewrite forallb_forall in Hall. specialize (Hall B HB).
      apply existsb_exists in Hall. destruct Hall as [[u f] [Hin E]]. cbn [fst] in E.
      apply Pos.eqb_eq in E. subst u. rewrite forallb_forall in Hdyn. specialize (Hdyn (B, f) Hin).
      cbn [fst snd] in Hdyn. rewrite !andb_true_iff in Hdyn. destruct Hdyn as [[_ Hform] Hbd].
      exists f. split; [exact (proj1 (proj1 (dynamic_vars_in F B f) Hin))|]. split; assumption. }
  destruct Hloop_facts as [Hlf [Hnone [Hindex0 Hsz]]].
  constructor; cbn [pr_zero pr_index pr_address pr_pred pr_tx pr_ty pr_tz pr_bx pr_by pr_bz pr_width pr_loops
                    pr_static].
  - exact Htx.
  - exact Hty.
  - exact Htz.
  - exact Hblock.
  - destruct loops as [zp|].
    + destruct (Hlf zp eq_refl) as (Hform & _ & Hz & Ht & Ha & Hp & Htx' & Hty' & Htz' & Hbx' & Hby' & Hbz' & Hw' & _).
      rewrite (zf_index _ _ _ Hform). unfold index_ir, index_plan, zero_index_ir. cbn.
      rewrite Hz, Ht, Ha, Htx', Hty', Htz', Hbx', Hby'. reflexivity.
    + exact (Hindex0 eq_refl).
  - intros zp E. exact (Hlf zp E).
  - exact Hnone.
  - unfold statics_ir. cbn [pr_static]. apply prefix_eqb_app in Hpre.
    destruct Hpre as [rest Hrest]. exists rest. rewrite <- app_assoc in Hrest. exact Hrest.
  - exact Hblk.
  - intros v Hv. destruct (Hst1 v Hv) as [[u f] [Hin Hsv]].
    destruct (static_of_some w N (u, f) v HN0 Hsv) as (Hb & _ & Hhi & Hmod & Hfull & Hleft & Hl). cbn [fst snd] in *.
    apply ordered_vars_in in Hin. apply static_vars_in in Hin. destruct Hin as [Hin [Hft Hpos]].
    exists f. split; [rewrite Hb; exact Hin|]. split; [exact Hft|]. split.
    + unfold sv_bytes, pr_threads. cbn [pr_width pr_bx pr_by pr_bz]. rewrite Hfull, Hleft. fold N.
      rewrite Z2Nat.id by (apply Z.div_pos; [apply Z.div_pos|]; lia).
      rewrite <- (Z.div_mod (flo f / w) N) by lia.
      apply (proj2 (Z.div_exact (flo f) w ltac:(lia))). exact Hmod.
    + split; [exact Hhi|]. exact (Hsb v Hv).
  - intros v Hv. destruct (Hst1 v Hv) as [[u f] [Hin Hsv]].
    destruct (static_of_some w N (u, f) v HN0 Hsv) as (_ & _ & _ & _ & _ & _ & Hl). exact Hl.
  - intros u f Hin. destruct (ft f) as [|x ts] eqn:Hft.
    + destruct (Z.le_gt_cases (flo f) 0) as [Hle|Hgt]; [left; split; [reflexivity|exact Hle]|].
      right. left. assert (Hs : In (u, f) (static_vars F)) by (apply static_vars_in; tauto).
      destruct (Hst2 (u, f) (proj2 (ordered_vars_in _ _ _) Hs)) as [v [Hv Hsv]].
      destruct (static_of_some w N (u, f) v HN0 Hsv) as (Hb & _ & Hhi & Hmod & Hfull & Hleft & Hl). cbn [fst snd] in *.
      exists v. split; [exact Hv|]. split; [exact Hb|]. split; [reflexivity|].
      unfold sv_bytes, pr_threads. cbn [pr_width pr_bx pr_by pr_bz]. rewrite Hfull, Hleft. fold N.
      rewrite Z2Nat.id by (apply Z.div_pos; [apply Z.div_pos|]; lia).
      rewrite <- (Z.div_mod (flo f / w) N) by lia.
      apply (proj2 (Z.div_exact (flo f) w ltac:(lia))). exact Hmod.
    + right. right. assert (Hd : In (u, f) (dynamic_vars F)) by (apply dynamic_vars_in; split; [exact Hin|rewrite Hft; discriminate]).
      destruct loops as [zp|].
      * exists zp. split; [reflexivity|].
        destruct (List.length (dynamic_vars F)) as [|m'] eqn:Hm.
        { destruct (dynamic_vars F); [destruct Hd|discriminate]. }
        destruct (zero_regs_of prog) as [zr|]; [|discriminate].
        destruct (zero_terms_of Mf F w) as [terms|]; [|discriminate].
        destruct (loop_bases prog (S m')) as [bases|] eqn:Hbases; [|discriminate].
        injection Hloops as <-. cbn [zp_bases].
        rewrite !andb_true_iff in Hlo. destruct Hlo as [[[_ Hdyn] _] _].
        rewrite forallb_forall in Hdyn. specialize (Hdyn (u, f) Hd). cbn [fst snd] in Hdyn.
        rewrite !andb_true_iff in Hdyn. destruct Hdyn as [[Hb Hfo] _]. split; [exact (reg_in_spec _ _ Hb)|exact Hfo].
      * exfalso. rewrite (Hnone eq_refl) in Hd. destruct Hd.
  - split; [exact Htxb|]. split; [exact Htyb|exact Htzb].
  - unfold pr_threads. cbn. repeat split; assumption.
  - repeat split; assumption.
  - exact (pos_nodup_spec _ Hnodup).
  - intros r Hr Hown. exact (pos_apart_spec _ _ r Hapart Hown Hr).
  - intros r Hr Hin. rewrite forallb_forall in Hwr. apply in_map_iff in Hr. destruct Hr as [v [<- Hv]].
    specialize (Hwr v Hv). apply negb_true_iff in Hwr. unfold reg_in in Hwr.
    rewrite <- not_true_iff_false, existsb_exists in Hwr. apply Hwr. exists (fst v). split; [exact Hin|].
    apply Pos.eqb_refl.
  - exact Hbr.
  - destruct Hcat as [[[Hcx Hcy] Hcz] _].
    split; [exact (is_tid_cat_spec _ _ _ _ Hcx)|].
    split; [exact (is_tid_cat_spec _ _ _ _ Hcy)|exact (is_tid_cat_spec _ _ _ _ Hcz)].
  - destruct Hcat as [_ Hsh]. rewrite forallb_forall in Hsh.
    intros u f Hin. exact (is_platform_cat_spec _ _ _ (Hsh (u, f) Hin)).
  - intros zp E. exact (Hsz zp E).
Qed.

(* ------------------------------------------------------------------ *)
(* Positions before and after the barrier.                            *)

Lemma next_pos_spec : forall prog b i,
  next_pos prog b i = (b, S i) \/ (next_pos prog b i = (S b, O) /\ (block_len prog b <= S i)%nat).
Proof.
  intros prog b i. unfold next_pos. destruct (Nat.ltb_spec (S i) (block_len prog b)); [left; reflexivity|].
  right. split; [reflexivity|lia].
Qed.

Lemma next_pos_inside : forall prog b i, (S i < block_len prog b)%nat -> next_pos prog b i = (b, S i).
Proof. intros prog b i H. unfold next_pos. destruct (Nat.ltb_spec (S i) (block_len prog b)); [reflexivity|lia]. Qed.

Lemma fetch_some_len : forall prog b i ins, fetch prog b i = Some ins -> (i < block_len prog b)%nat.
Proof. intros prog b i ins H. unfold fetch in H. apply nth_error_Some. rewrite H. discriminate. Qed.

Lemma in_skipn_nth : forall {A} (l : list A) k b x, (k <= b)%nat -> nth_error l b = Some x -> In x (skipn k l).
Proof.
  intros A l. induction l as [|y l IH]; intros k b x Hk H; [destruct b; discriminate|].
  destruct k as [|k]; [apply nth_error_In in H; exact H|].
  destruct b as [|b]; [lia|]. cbn [skipn]. exact (IH k b x ltac:(lia) H).
Qed.

(* Where a step goes: the next position, a branch's target, or the end. *)
Lemma step_pos : forall prog wd mem arb shm init cells locals lwin t s s' ins,
  fetch prog (blk s) (idx s) = Some ins ->
  step prog wd mem arb shm init cells locals lwin t s = Some s' ->
  (blk s', idx s') = next_pos prog (blk s) (idx s) \/
  (exists g tg, ins = IBra g tg /\ (blk s', idx s') = (tg, O)) \/
  (exists g, ins = IExit g /\ (blk s', idx s') = (List.length prog, O)).
Proof.
  intros prog wd mem arb shm init cells locals lwin t s s' ins Hf Hs. unfold step in Hs. rewrite Hf in Hs.
  destruct (next_pos prog (blk s) (idx s)) as [nb ni] eqn:Hn.
  destruct (negb (holds (guard_of ins) (regs s))); [injection Hs as <-; left; reflexivity|].
  destruct ins; try (injection Hs as <-; left; reflexivity).
  - right. left. injection Hs as <-. exists g, target. split; reflexivity.
  - right. right. injection Hs as <-. exists g. split; reflexivity.
Qed.

(* A quiet instruction's step is a function of the registers. *)
Lemma quiet_step : forall prog z wd mem1 mem2 arb1 arb2 shm1 shm2 init cells locals lwin t s ins,
  fetch prog (blk s) (idx s) = Some ins -> quiet z ins = true ->
  step prog wd mem1 arb1 shm1 init cells locals lwin t s = step prog wd mem2 arb2 shm2 init cells locals lwin t s.
Proof.
  intros prog z wd mem1 mem2 arb1 arb2 shm1 shm2 init cells locals lwin t s ins Hf Hq.
  unfold step. rewrite Hf. destruct ins; cbn [quiet] in Hq; try discriminate;
    try (destruct g; [discriminate|]); try reflexivity.
Qed.

(* A quiet instruction writes no memory but shared memory, by a store of
   the zero register. *)
Lemma quiet_access : forall z ins win rs, quiet z ins = true -> access_at win rs ins = None.
Proof.
  intros z ins win rs Hq. unfold access_at. destruct (negb (holds (guard_of ins) rs)); [reflexivity|].
  destruct ins; cbn [quiet] in Hq; try discriminate; reflexivity.
Qed.

Ltac in_list :=
  cbn; first [left; reflexivity | right; left; reflexivity | right; right; left; reflexivity
             | right; right; right; left; reflexivity].

Lemma nodup4 : forall a b c d : reg, NoDup [a; b; c; d] ->
  a <> b /\ a <> c /\ a <> d /\ b <> c /\ b <> d /\ c <> d.
Proof.
  intros a b c d H. rewrite !NoDup_cons_iff in H. cbn [In] in H.
  repeat split; intros Heq; subst; tauto.
Qed.

Section Positions.

Variable F : lfacts.
Variable Mf : binding.
Variable prog : program.
Variable ps : prologue.
Hypothesis Hsh : shape F Mf prog ps.

Local Notation z := (pr_zero ps).
Local Notation t := (pr_index ps).
Local Notation a := (pr_address ps).
Local Notation p := (pr_pred ps).
Local Notation m := (pr_blocks ps).
Local Notation bar := (bar_of ps).

Lemma static_start_fst : fst (static_start ps) = m.
Proof. unfold static_start, pr_blocks. destruct (pr_loops ps); reflexivity. Qed.

Lemma bar_fst : fst bar = m.
Proof. unfold bar_of. cbn [fst]. apply static_start_fst. Qed.

Lemma bar_snd : snd bar = (snd (static_start ps) + List.length (statics_ir ps))%nat.
Proof. reflexivity. Qed.

Lemma own_distinct : z <> t /\ z <> a /\ z <> p /\ t <> a /\ t <> p /\ a <> p.
Proof. exact (nodup4 _ _ _ _ (sh_distinct _ _ _ _ Hsh)). Qed.

Lemma index_nth : forall i ins, nth_error (index_ir ps) i = Some ins ->
  quiet z ins = true /\ (i <> O -> dest ins <> Some z) /\ (forall d, dest ins = Some d -> In d [z; t; a]) /\
  (forall g tg, ins <> IBra g tg).
Proof.
  destruct own_distinct as (Hzt & Hza & _ & _ & _ & _).
  intros i ins H. unfold index_ir, zero_index_ir, index_plan in H. cbn in H.
  destruct i as [|[|[|[|[|[|i]]]]]]; cbn in H; try (destruct i; discriminate); injection H as <-; cbn [quiet dest].
  all: split; [reflexivity|]; split; [intros Hne E; first [exfalso; exact (Hne eq_refl) | injection E as E; congruence]|];
    split; [intros d E; injection E as <-; cbn;
            first [left; reflexivity | right; left; reflexivity | right; right; left; reflexivity]|];
    intros g tg E; discriminate E.
Qed.

Lemma statics_in : forall ins, In ins (statics_ir ps) ->
  quiet z ins = true /\ dest ins <> Some z /\ (forall d, dest ins = Some d -> In d [a; p]) /\
  (forall g tg, ins <> IBra g tg).
Proof.
  destruct own_distinct as (_ & Hza & Hzp & _ & _ & _).
  intros ins H. unfold statics_ir in H. apply in_flat_map in H. destruct H as [v [_ H]].
  unfold static_ir in H. cbn [In] in H.
  assert (Hcase : forall ins', (ins' = IMov None a (sv_base v) \/ ins' = IMad None 32 a t (OImm (pr_width ps)) (OReg a) \/
      (exists off, ins' = IShA None KStore a off (pr_width ps) (Some (OReg z))) \/
      ins' = ISetp None CLt 32 false p t (OImm (sv_left v)) \/
      (exists off, ins' = IShA (Some (false, p)) KStore a off (pr_width ps) (Some (OReg z)))) ->
      quiet z ins' = true /\ dest ins' <> Some z /\ (forall d, dest ins' = Some d -> In d [a; p]) /\
      (forall g tg, ins' <> IBra g tg)).
  { intros ins' [->|[->|[[off ->]|[->|[off ->]]]]]; cbn [quiet dest]; try rewrite Pos.eqb_refl.
    all: split; [reflexivity|]; split; [intros E; first [discriminate E | injection E as E; congruence]|];
      split; [intros d E; first [discriminate E | injection E as <-; cbn; first [left; reflexivity | right; left; reflexivity]]|];
      intros g tg E; discriminate E. }
  apply Hcase. destruct H as [<-|[<-|H]]; [left; reflexivity | right; left; reflexivity |].
  apply in_app_or in H. destruct H as [H|H].
  - apply in_map_iff in H. destruct H as [k [<- _]]. right. right. left. eexists. reflexivity.
  - destruct (Z.eqb (sv_left v) 0); [destruct H|]. cbn [In] in H. destruct H as [<-|[<-|[]]].
    + right. right. right. left. reflexivity.
    + right. right. right. right. eexists. reflexivity.
Qed.

(* The barrier's block: the index when there are no loops, the static
   stores, the barrier, and the rest. *)
Lemma block_bar : exists rest,
  block_body prog m = (match pr_loops ps with None => index_ir ps | Some _ => [] end) ++
                      statics_ir ps ++ INop :: rest.
Proof.
  destruct (sh_static _ _ _ _ Hsh) as [rest Hrest]. exists rest.
  rewrite static_start_fst in Hrest. unfold static_start in Hrest. unfold pr_blocks in *.
  destruct (pr_loops ps) as [zp|]; cbn [fst snd] in Hrest.
  - cbn [skipn] in Hrest. exact Hrest.
  - rewrite <- (firstn_skipn 6 (block_body prog 0)), Hrest, (sh_index _ _ _ _ Hsh). reflexivity.
Qed.

Lemma pre_length : List.length (match pr_loops ps with None => index_ir ps | Some _ => [] end) = snd (static_start ps).
Proof. unfold static_start. destruct (pr_loops ps); reflexivity. Qed.

Lemma fetch_bar : fetch prog (fst bar) (snd bar) = Some INop.
Proof.
  destruct block_bar as [rest Hb]. rewrite bar_fst, bar_snd. unfold fetch. rewrite Hb.
  rewrite nth_error_app2 by (rewrite pre_length; lia). rewrite pre_length.
  rewrite nth_error_app2 by lia. replace (snd (static_start ps) + List.length (statics_ir ps) - snd (static_start ps) - List.length (statics_ir ps))%nat with O by lia.
  reflexivity.
Qed.

Lemma bar_block_len : (snd bar < block_len prog m)%nat.
Proof.
  pose proof fetch_bar as H. rewrite bar_fst in H. exact (fetch_some_len _ _ _ _ H).
Qed.

(* A block index as three times a loop's index plus a remainder. *)
Lemma div3 : forall b, exists j r, b = (3 * j + r)%nat /\ (r < 3)%nat.
Proof.
  intros b. exists (b / 3)%nat, (b mod 3)%nat. split; [apply Nat.div_mod; lia | apply Nat.mod_upper_bound; lia].
Qed.

(* What the loops' blocks hold. *)
Lemma loops_blocks : forall zp, pr_loops ps = Some zp ->
  forall j, (j < List.length (zp_bases zp))%nat ->
  exists base, nth_error (zp_bases zp) j = Some base /\
  (j = O -> block_body prog 0 = zero_index_ir zp ++ zero_setup_ir zp (zero_temps_of prog zp 0)) /\
  (j <> O -> block_body prog (3 * j) = zero_setup_ir zp (zero_temps_of prog zp j)) /\
  block_body prog (3 * j + 1) = zero_head_ir (zp_regs zp) (3 * j + 3) /\
  block_body prog (3 * j + 2) = zero_body_ir zp base (3 * j + 1).
Proof.
  intros zp Hl j Hj. destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & _).
  destruct (nth_error (zp_bases zp) j) as [base|] eqn:Hbase; [|apply nth_error_None in Hbase; lia].
  exists base. split; [reflexivity|].
  destruct (zf_loops _ _ _ Hform j base Hbase) as (Hsetup & Hhead & Hbody & _).
  split; [|split; [|split; [exact Hhead|exact Hbody]]].
  - intros ->. unfold zero_setup_block in Hsetup.
    rewrite <- (firstn_skipn 6 (block_body prog 0)), (zf_index _ _ _ Hform), Hsetup. reflexivity.
  - intros Hj0. destruct j as [|j']; [contradiction|]. unfold zero_setup_block in Hsetup. exact Hsetup.
Qed.

Lemma loops_count : forall zp, pr_loops ps = Some zp -> m = (3 * List.length (zp_bases zp))%nat.
Proof. intros zp Hl. unfold pr_blocks. rewrite Hl. reflexivity. Qed.

Lemma setup_in : forall zp temps ins, In ins (zero_setup_ir zp temps) ->
  quiet (zr_zero (zp_regs zp)) ins = true /\
  (forall d, dest ins = Some d -> d = zr_count (zp_regs zp) \/ d = zr_element (zp_regs zp) \/ In d (map fst temps)) /\
  (forall g tg, ins <> IBra g tg).
Proof.
  intros zp temps ins H. unfold zero_setup_ir in H. cbn [In] in H. destruct H as [<-|H].
  { cbn. repeat split; try discriminate. intros d E. injection E as <-. left. reflexivity. }
  apply in_app_or in H. destruct H as [H|[<-|[]]].
  - apply in_flat_map in H. destruct H as [[[s c] [v from]] [Hin H]]. unfold zero_term_ir in H.
    cbn [In] in H.
    assert (Hv : In v (map fst temps))
      by (apply in_map_iff; exists (v, from); split; [reflexivity|exact (in_combine_r _ _ _ _ Hin)]).
    destruct H as [<-|[<-|[<-|[<-|[]]]]]; cbn; (repeat split; try discriminate); intros d E; injection E as <-; tauto.
  - cbn. repeat split; try discriminate. intros d E. injection E as <-. right. left. reflexivity.
Qed.

Lemma pr_written_loop : forall zp, pr_loops ps = Some zp ->
  forall r, In r (zero_regs_list (zp_regs zp)) -> In r (pr_written prog ps).
Proof.
  intros zp Hl r Hr. unfold pr_written. rewrite Hl. apply in_or_app. right. apply in_or_app. left. exact Hr.
Qed.

Lemma pr_written_temp : forall zp, pr_loops ps = Some zp ->
  forall j r, (j < List.length (zp_bases zp))%nat -> In r (map fst (zero_temps_of prog zp j)) ->
  In r (pr_written prog ps).
Proof.
  intros zp Hl j r Hj Hr. unfold pr_written. rewrite Hl. apply in_or_app. right. apply in_or_app. right.
  apply in_concat. exists (map fst (zero_temps_of prog zp j)). split; [|exact Hr].
  apply in_map_iff. exists j. split; [reflexivity|]. apply in_seq. lia.
Qed.

Lemma pr_written_own : forall r, In r [z; t; a; p] -> In r (pr_written prog ps).
Proof. intros r Hr. unfold pr_written. apply in_or_app. left. exact Hr. Qed.

Lemma index_ir_of_plan : forall zp, pr_loops ps = Some zp -> zero_index_ir zp = index_ir ps.
Proof.
  intros zp Hl. destruct (sh_loops _ _ _ _ Hsh zp Hl) as (_ & _ & Hz & Ht & Ha & _ & Hx & Hy & Hz' & Hbx & Hby & _).
  unfold index_ir, index_plan, zero_index_ir. cbn. rewrite Hz, Ht, Ha, Hx, Hy, Hz', Hbx, Hby. reflexivity.
Qed.

(* The instructions of a block of the loops. *)
Lemma setup_case : forall zp, pr_loops ps = Some zp -> forall j i ins, (j < List.length (zp_bases zp))%nat ->
  fetch prog (3 * j) i = Some ins ->
  In ins (zero_setup_ir zp (zero_temps_of prog zp j)) \/
  (exists k, nth_error (index_ir ps) k = Some ins /\ (k = O <-> (j, i) = (O, O))).
Proof.
  intros zp Hl j i ins Hj Hf.
  destruct (loops_blocks zp Hl j Hj) as (base & Hbase & H0 & H0' & _ & _).
  unfold fetch in Hf.
  destruct (Nat.eq_dec j 0) as [Hj0|Hj0].
  - subst j. replace (3 * 0)%nat with O in Hf by lia. rewrite (H0 eq_refl) in Hf.
    destruct (Nat.lt_ge_cases i 6) as [Hi6|Hi6].
    + rewrite nth_error_app1 in Hf by (unfold zero_index_ir; cbn; lia). rewrite (index_ir_of_plan zp Hl) in Hf.
      right. exists i. split; [exact Hf|]. split; intros E; [subst i; reflexivity|injection E; intros; lia].
    + rewrite nth_error_app2 in Hf by (unfold zero_index_ir; cbn; lia). left. exact (nth_error_In _ _ Hf).
  - rewrite (H0' Hj0) in Hf. left. exact (nth_error_In _ _ Hf).
Qed.

Lemma setup_fetch : forall zp, pr_loops ps = Some zp -> forall j i ins, (j < List.length (zp_bases zp))%nat ->
  fetch prog (3 * j) i = Some ins ->
  quiet z ins = true /\ ((j, i) <> (O, O) -> dest ins <> Some z) /\
  (forall d, dest ins = Some d -> In d (pr_written prog ps)) /\ (forall g tg, ins <> IBra g tg).
Proof.
  intros zp Hl j i ins Hj Hf.
  destruct (loops_blocks zp Hl j Hj) as (base & Hbase & _).
  destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & _ & Hz & _).
  pose proof (nodup6 _ _ _ _ _ _ (zf_distinct _ _ _ Hform)) as Hd6.
  pose proof (proj2 (proj2 (proj2 (zf_loops _ _ _ Hform _ _ Hbase)))) as Htemps.
  destruct (setup_case zp Hl j i ins Hj Hf) as [Hin|[k [Hk Hk0]]].
  - destruct (setup_in _ _ _ Hin) as (Hq & Hdest & Hnb). rewrite Hz in Hq.
    split; [exact Hq|]. split; [|split; [|exact Hnb]].
    + intros _ E. destruct (Hdest z E) as [E'|[E'|E']]; rewrite <- Hz in E'.
      * destruct Hd6 as (_ & _ & E'' & _). exact (E'' E').
      * destruct Hd6 as (_ & _ & _ & E'' & _). exact (E'' E').
      * apply in_map_iff in E'. destruct E' as [[v from] [Hv Hin']]. cbn [fst] in Hv. subst v.
        destruct (Htemps _ _ Hin') as (_ & _ & Hnw & _). apply Hnw. unfold zero_written, zero_regs_list. cbn. tauto.
    + intros d Hd. destruct (Hdest d Hd) as [->|[->|E]].
      * apply (pr_written_loop zp Hl). unfold zero_regs_list. cbn. tauto.
      * apply (pr_written_loop zp Hl). unfold zero_regs_list. cbn. tauto.
      * exact (pr_written_temp zp Hl j d Hj E).
  - destruct (index_nth k ins Hk) as (Hq & Hdz & Hdest & Hnb).
    split; [exact Hq|]. split; [|split; [|exact Hnb]].
    + intros Hne. apply Hdz. intros ->. apply Hne. apply Hk0. reflexivity.
    + intros d Hd. apply pr_written_own. destruct (Hdest d Hd) as [E|[E|[E|[]]]]; rewrite <- E; in_list.
Qed.

Lemma head_fetch : forall zp, pr_loops ps = Some zp -> forall j i ins, (j < List.length (zp_bases zp))%nat ->
  fetch prog (3 * j + 1) i = Some ins ->
  quiet z ins = true /\ dest ins <> Some z /\
  (forall d, dest ins = Some d -> In d (pr_written prog ps)) /\
  (forall g tg, ins = IBra g tg -> tg = (3 * j + 3)%nat).
Proof.
  intros zp Hl j i ins Hj Hf.
  destruct (loops_blocks zp Hl j Hj) as (base & Hbase & _ & _ & H1 & _).
  destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & _ & Hz & _).
  pose proof (nodup6 _ _ _ _ _ _ (zf_distinct _ _ _ Hform)) as Hd6.
  unfold fetch in Hf. rewrite H1 in Hf. unfold zero_head_ir in Hf.
  remember (3 * j + 3)%nat as ex eqn:Hex in Hf.
  destruct i as [|[|i]]; cbn in Hf; try (destruct i; discriminate); injection Hf as <-; cbn [quiet dest].
  - split; [reflexivity|]. split; [|split].
    + intros E. injection E as E. rewrite <- Hz in E. destruct Hd6 as (_ & _ & _ & _ & E' & _). exact (E' (eq_sym E)).
    + intros d E. injection E as <-. apply (pr_written_loop zp Hl). unfold zero_regs_list. cbn. tauto.
    + intros g tg E. discriminate E.
  - split; [reflexivity|]. split; [|split].
    + intros E. discriminate E.
    + intros d E. discriminate E.
    + intros g tg E. injection E as _ <-. exact Hex.
Qed.

Lemma body_fetch : forall zp, pr_loops ps = Some zp -> forall j i ins, (j < List.length (zp_bases zp))%nat ->
  fetch prog (3 * j + 2) i = Some ins ->
  quiet z ins = true /\ dest ins <> Some z /\
  (forall d, dest ins = Some d -> In d (pr_written prog ps)) /\
  (forall g tg, ins = IBra g tg -> tg = (3 * j + 1)%nat).
Proof.
  intros zp Hl j i ins Hj Hf.
  destruct (loops_blocks zp Hl j Hj) as (base & Hbase & _ & _ & _ & H2).
  destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & _ & Hz & _).
  pose proof (nodup6 _ _ _ _ _ _ (zf_distinct _ _ _ Hform)) as Hd6.
  destruct Hd6 as (_ & Hza & _ & Hze & _).
  assert (Hw : forall r, In r (zero_regs_list (zp_regs zp)) -> In r (pr_written prog ps)) by exact (pr_written_loop zp Hl).
  unfold fetch in Hf. rewrite H2 in Hf. unfold zero_body_ir in Hf.
  remember (3 * j + 1)%nat as hd eqn:Hhd in Hf.
  destruct i as [|[|[|[|[|i]]]]]; cbn in Hf; try (destruct i; discriminate); injection Hf as <-; cbn [quiet dest].
  - split; [reflexivity|]. split; [intros E; injection E as E; rewrite <- Hz in E; exact (Hza (eq_sym E))|].
    split; [intros d E; injection E as <-; apply Hw; unfold zero_regs_list; cbn; tauto|]. intros g tg E. discriminate E.
  - split; [reflexivity|]. split; [intros E; injection E as E; rewrite <- Hz in E; exact (Hza (eq_sym E))|].
    split; [intros d E; injection E as <-; apply Hw; unfold zero_regs_list; cbn; tauto|]. intros g tg E. discriminate E.
  - rewrite Hz, Pos.eqb_refl. split; [reflexivity|]. split; [intros E; discriminate E|].
    split; [intros d E; discriminate E|]. intros g tg E. discriminate E.
  - split; [reflexivity|]. split; [intros E; injection E as E; rewrite <- Hz in E; exact (Hze (eq_sym E))|].
    split; [intros d E; injection E as <-; apply Hw; unfold zero_regs_list; cbn; tauto|]. intros g tg E. discriminate E.
  - split; [reflexivity|]. split; [intros E; discriminate E|]. split; [intros d E; discriminate E|].
    intros g tg E. injection E as _ <-. exact Hhd.
Qed.

(* The instruction at a position in a block of the loops. *)
Lemma loops_fetch : forall zp, pr_loops ps = Some zp -> forall b i ins, (b < m)%nat ->
  fetch prog b i = Some ins ->
  quiet z ins = true /\ ((b, i) <> (O, O) -> dest ins <> Some z) /\
  (forall d, dest ins = Some d -> In d (pr_written prog ps)) /\
  (forall g tg, ins = IBra g tg -> (1 <= tg <= m)%nat).
Proof.
  intros zp Hl b i ins Hb Hf. rewrite (loops_count zp Hl) in Hb |- *.
  destruct (div3 b) as (j & r & -> & Hr).
  assert (Hj : (j < List.length (zp_bases zp))%nat) by lia.
  destruct r as [|[|[|r]]]; [| | |lia].
  - rewrite Nat.add_0_r in Hf |- *.
    destruct (setup_fetch zp Hl j i ins Hj Hf) as (Hq & Hdz & Hdest & Hnb).
    split; [exact Hq|]. split; [|split; [exact Hdest|]].
    + intros Hne. apply Hdz. intros E. apply Hne. injection E as -> ->. reflexivity.
    + intros g tg E. exfalso. exact (Hnb g tg E).
  - destruct (head_fetch zp Hl j i ins Hj Hf) as (Hq & Hdz & Hdest & Hbr).
    split; [exact Hq|]. split; [intros _; exact Hdz|]. split; [exact Hdest|].
    intros g tg E. rewrite (Hbr g tg E). lia.
  - destruct (body_fetch zp Hl j i ins Hj Hf) as (Hq & Hdz & Hdest & Hbr).
    split; [exact Hq|]. split; [intros _; exact Hdz|]. split; [exact Hdest|].
    intros g tg E. rewrite (Hbr g tg E). lia.
Qed.

(* The instruction at a position of the barrier's block before the barrier. *)
Lemma pre_fetch : forall i ins, (i < snd bar)%nat -> fetch prog m i = Some ins ->
  quiet z ins = true /\ ((m, i) <> (O, O) -> dest ins <> Some z) /\
  (forall d, dest ins = Some d -> In d (pr_written prog ps)) /\ (forall g tg, ins <> IBra g tg).
Proof.
  intros i ins Hi Hf. destruct block_bar as [rest Hbb]. unfold fetch in Hf. rewrite Hbb in Hf. rewrite bar_snd in Hi.
  destruct (Nat.lt_ge_cases i (snd (static_start ps))) as [Hpre|Hpre].
  - rewrite nth_error_app1 in Hf by (rewrite pre_length; exact Hpre).
    assert (Hl : pr_loops ps = None).
    { unfold static_start in Hpre. destruct (pr_loops ps); [cbn in Hpre; lia|reflexivity]. }
    rewrite Hl in Hf.
    destruct (index_nth i ins Hf) as (Hq & Hdz & Hdest & Hnb). split; [exact Hq|]. split; [|split].
    + intros Hne. apply Hdz. intros ->. apply Hne. unfold pr_blocks. rewrite Hl. reflexivity.
    + intros d Hd. apply pr_written_own. destruct (Hdest d Hd) as [E|[E|[E|[]]]]; rewrite <- E; in_list.
    + exact Hnb.
  - rewrite nth_error_app2 in Hf by (rewrite pre_length; exact Hpre). rewrite pre_length in Hf.
    rewrite nth_error_app1 in Hf by lia. apply nth_error_In in Hf.
    destruct (statics_in ins Hf) as (Hq & Hdz & Hdest & Hnb). split; [exact Hq|]. split; [|split].
    + intros _. exact Hdz.
    + intros d Hd. apply pr_written_own. destruct (Hdest d Hd) as [E|[E|[]]]; rewrite <- E; in_list.
    + exact Hnb.
Qed.

(* The instruction at a position before the barrier: quiet, writing no
   register outside those of the prologue, writing the zero register only
   at the start, branching only before or to the barrier, and followed by
   a position before or at the barrier. *)
Lemma region_fetch : forall b i ins, before_bar ps (b, i) -> fetch prog b i = Some ins ->
  quiet z ins = true /\
  ((b, i) <> (O, O) -> dest ins <> Some z) /\
  (forall d, dest ins = Some d -> In d (pr_written prog ps)) /\
  (forall g tg, ins = IBra g tg -> (1 <= tg)%nat /\ (before_bar ps (tg, O) \/ (tg, O) = bar)) /\
  (before_bar ps (next_pos prog b i) \/ next_pos prog b i = bar).
Proof.
  intros b i ins Hpos Hf. unfold before_bar in Hpos. rewrite bar_fst in Hpos. cbn [fst snd] in Hpos.
  (* a block start at or before the barrier's block *)
  assert (Hblock : forall tg, (tg <= m)%nat -> before_bar ps (tg, O) \/ (tg, O) = bar).
  { intros tg Htg. unfold before_bar. rewrite bar_fst. cbn [fst snd].
    destruct (Nat.lt_ge_cases tg m) as [Hlt|Hge]; [left; left; exact Hlt|].
    assert (E : tg = m) by lia. subst tg.
    destruct (Nat.eq_dec (snd bar) O) as [Hz|Hnz].
    - right. rewrite (surjective_pairing bar), bar_fst, Hz. reflexivity.
    - left. right. split; [reflexivity|lia]. }
  (* where the next position lies *)
  assert (Hnext : before_bar ps (next_pos prog b i) \/ next_pos prog b i = bar).
  { destruct Hpos as [H|[-> H]].
    - destruct (next_pos_spec prog b i) as [E|[E _]]; rewrite E.
      + left. unfold before_bar. rewrite bar_fst. cbn [fst snd]. left. exact H.
      + apply Hblock. lia.
    - rewrite next_pos_inside by (pose proof bar_block_len; lia).
      destruct (Nat.lt_ge_cases (S i) (snd bar)) as [Hlt|Hge].
      + left. unfold before_bar. rewrite bar_fst. cbn [fst snd]. right. split; [reflexivity|exact Hlt].
      + right. rewrite (surjective_pairing bar), bar_fst. f_equal. lia. }
  destruct Hpos as [Hb|[-> Hi]].
  - destruct (pr_loops ps) as [zp|] eqn:Hl; [|unfold pr_blocks in Hb; rewrite Hl in Hb; lia].
    destruct (loops_fetch zp Hl b i ins Hb Hf) as (Hq & Hdz & Hdest & Hbr).
    split; [exact Hq|]. split; [exact Hdz|]. split; [exact Hdest|]. split; [|exact Hnext].
    intros g tg E. destruct (Hbr g tg E) as [H1 H2]. split; [exact H1|]. apply Hblock. exact H2.
  - destruct (pre_fetch i ins Hi Hf) as (Hq & Hdz & Hdest & Hnb).
    split; [exact Hq|]. split; [exact Hdz|]. split; [exact Hdest|]. split; [|exact Hnext].
    intros g tg E. exfalso. exact (Hnb g tg E).
Qed.

Lemma region_quiet : forall b i ins, before_bar ps (b, i) -> fetch prog b i = Some ins -> quiet z ins = true.
Proof. intros b i ins Hp Hf. exact (proj1 (region_fetch b i ins Hp Hf)). Qed.

(* Positions after the barrier are closed under every step. *)
Lemma after_next : forall b i, after_bar ps (b, i) -> after_bar ps (next_pos prog b i).
Proof.
  intros b i H. unfold after_bar in *. cbn [fst snd] in *.
  destruct (next_pos_spec prog b i) as [E|[E _]]; rewrite E; cbn [fst snd]; lia.
Qed.

Lemma after_branch : forall b i g tg, after_bar ps (b, i) -> fetch prog b i = Some (IBra g tg) -> after_bar ps (tg, O).
Proof.
  intros b i g tg H Hf. pose proof (sh_branches _ _ _ _ Hsh) as Hbr. unfold branches_after in Hbr.
  rewrite forallb_forall in Hbr. unfold fetch, block_body in Hf.
  destruct (nth_error prog b) as [body|] eqn:Hb; [|destruct i; discriminate].
  assert (Hin : In body (skipn m prog)).
  { apply (in_skipn_nth prog m b body); [|exact Hb]. unfold after_bar in H. rewrite bar_fst in H. cbn [fst snd] in H. lia. }
  specialize (Hbr body Hin). rewrite forallb_forall in Hbr. apply nth_error_In in Hf. specialize (Hbr _ Hf).
  apply Nat.ltb_lt in Hbr. unfold after_bar. rewrite bar_fst. cbn [fst snd]. left. exact Hbr.
Qed.

Lemma after_exit : after_bar ps (List.length prog, O).
Proof. unfold after_bar. rewrite bar_fst. cbn [fst snd]. left. exact (sh_blocks _ _ _ _ Hsh). Qed.

Lemma bar_next : after_bar ps (next_pos prog (fst bar) (snd bar)).
Proof.
  destruct (next_pos_spec prog (fst bar) (snd bar)) as [E|[E _]]; rewrite E; unfold after_bar; cbn [fst snd]; lia.
Qed.

Lemma not_after_bar : ~ after_bar ps bar.
Proof. unfold after_bar. lia. Qed.

Lemma not_before_bar : ~ before_bar ps bar.
Proof. unfold before_bar. lia. Qed.

Lemma start_region : before_bar ps (O, O) \/ (O, O) = bar.
Proof.
  unfold before_bar. rewrite bar_fst. cbn [fst snd]. unfold pr_blocks.
  destruct (pr_loops ps) as [zp|] eqn:Hl.
  - destruct (sh_loops _ _ _ _ Hsh zp Hl) as (_ & Hne & _). left. left.
    destruct (zp_bases zp); [contradiction|]. cbn. lia.
  - left. right. split; [reflexivity|]. rewrite bar_snd. unfold static_start. rewrite Hl. cbn. lia.
Qed.

End Positions.

(* ------------------------------------------------------------------ *)
(* A thread's run to the barrier.                                      *)

Lemma nth_error_skipn' : forall {A} (l : list A) k j, nth_error (skipn k l) j = nth_error l (k + j).
Proof.
  intros A l. induction l as [|x l IH]; intros k j.
  - rewrite skipn_nil. destruct j as [|j]; [destruct (k + 0)%nat; reflexivity | destruct (k + S j)%nat; reflexivity].
  - destruct k as [|k]; [reflexivity|]. cbn [skipn nth_error plus]. exact (IH k j).
Qed.

Lemma nth_error_seq0 : forall n k, (k < n)%nat -> nth_error (seq 0 n) k = Some k.
Proof.
  intros n k Hk. rewrite (nth_error_nth' _ 0%nat) by (rewrite seq_length; exact Hk).
  rewrite seq_nth by exact Hk. reflexivity.
Qed.

Lemma step_fetch : forall prog wd mem arb shm init cells locals lwin t s s',
  step prog wd mem arb shm init cells locals lwin t s = Some s' -> exists ins, fetch prog (blk s) (idx s) = Some ins.
Proof.
  intros prog wd mem arb shm init cells locals lwin t s s' H. unfold step in H.
  destruct (fetch prog (blk s) (idx s)) as [ins|]; [exists ins; reflexivity|discriminate].
Qed.

Lemma trunc32_small : forall v, 0 <= v < 2 ^ 32 -> trunc 32 v = v.
Proof. intros v H. unfold trunc. apply Z.mod_small. exact H. Qed.

Lemma extend32_small : forall v, 0 <= v < 2 ^ 32 -> extend 32 false v = v.
Proof. intros v H. unfold extend. apply trunc32_small. exact H. Qed.

Section Runs.

Variable F : lfacts.
Variable Mf : binding.
Variable prog : program.
Variable ps : prologue.
Hypothesis Hsh : shape F Mf prog ps.
Variable Ev : env.

Local Notation z := (pr_zero ps).
Local Notation t := (pr_index ps).
Local Notation a := (pr_address ps).
Local Notation p := (pr_pred ps).
Local Notation m := (pr_blocks ps).
Local Notation bar := (bar_of ps).
Local Notation N := (pr_threads ps).
Local Notation w := (pr_width ps).
Local Notation init := (e_init Ev).
Local Notation run_ := (run prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                            (lf_local_window F)).
Local Notation step_ := (step prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                             (lf_local_window F)).
Local Notation exec := (exec1 (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev)).
Local Notation at_ b i rs := ({| blk := b; idx := i; regs := rs |}).
Local Notation T := (thread_of ps init).

Hypothesis Hinit : forall r, 0 <= init r < 2 ^ wdz F r.
Hypothesis Hbounds : forall r lo hi, is_fixed F r = true -> lookup (lf_bounds F) r = Some (lo, hi) ->
                     lo <= init r <= hi.
Hypothesis Hsizes : forall zp s c, pr_loops ps = Some zp -> In (s, c) (zp_terms zp) -> 0 <= init s < 2 ^ 64.
Hypothesis Hshared : forall zp, pr_loops ps = Some zp -> 0 <= zero_elements zp init * zp_width zp <= shared_max.

Lemma bounded_below_init : forall r n, bounded_below F r n = true -> 0 <= init r < n.
Proof.
  intros r n H. unfold bounded_below in H. apply andb_true_iff in H. destruct H as [Hf Hb].
  destruct (lookup (lf_bounds F) r) as [[lo hi]|] eqn:Hl; [|discriminate].
  apply andb_true_iff in Hb. destruct Hb as [H0 Hn]. apply Z.leb_le in H0. apply Z.ltb_lt in Hn.
  pose proof (Hbounds r lo hi Hf Hl). lia.
Qed.

Lemma bounded_in_init : forall r bytes, bounded_in F r bytes = true -> 0 <= init r /\ init r + bytes <= window_size.
Proof.
  intros r bytes H. unfold bounded_in in H. apply andb_true_iff in H. destruct H as [Hf Hb].
  destruct (lookup (lf_bounds F) r) as [[lo hi]|] eqn:Hl; [|discriminate].
  apply andb_true_iff in Hb. destruct Hb as [H0 Hn]. apply Z.leb_le in H0, Hn.
  pose proof (Hbounds r lo hi Hf Hl). lia.
Qed.

Lemma tid_bounds : 0 <= init (pr_tx ps) < pr_bx ps /\ 0 <= init (pr_ty ps) < pr_by ps /\ 0 <= init (pr_tz ps) < pr_bz ps.
Proof.
  destruct (sh_tids _ _ _ _ Hsh) as (Hx & Hy & Hz).
  split; [|split]; apply bounded_below_init; assumption.
Qed.

Lemma dims : 0 < pr_bx ps /\ 0 < pr_by ps /\ 0 < pr_bz ps /\ N < 2 ^ 31 /\ 0 < w < 2 ^ 31.
Proof. exact (sh_dims _ _ _ _ Hsh). Qed.

Lemma threads_pos' : 0 < N.
Proof. destruct dims as (Hx & Hy & Hz & _). unfold pr_threads. apply Z.mul_pos_pos; [apply Z.mul_pos_pos|]; lia. Qed.

Lemma thread_bounds' : 0 <= T < N.
Proof.
  unfold thread_of, pr_threads. destruct tid_bounds as (Hx & Hy & Hz). destruct dims as (Hbx & Hby & Hbz & _).
  destruct (index_arith _ _ _ _ _ _ Hbx Hby Hbz Hx Hy Hz) as (_ & _ & H). exact H.
Qed.

Lemma static_bounds : forall v, In v (pr_static ps) -> 0 <= init (sv_base v) /\ init (sv_base v) + sv_bytes ps v <= window_size.
Proof.
  intros v Hv. destruct (sh_static_vars _ _ _ _ Hsh v Hv) as (f & _ & _ & _ & _ & Hb). exact (bounded_in_init _ _ Hb).
Qed.

Lemma base_apart : forall v, In v (pr_static ps) -> ~ In (sv_base v) (pr_written prog ps).
Proof.
  intros v Hv. destruct (sh_static_vars _ _ _ _ Hsh v Hv) as (f & Hin & _). apply (sh_written_apart _ _ _ _ Hsh).
  apply in_map_iff. exists (sv_base v, f). split; [reflexivity|exact Hin].
Qed.

Lemma tid_apart : ~ In (pr_tx ps) (pr_written prog ps) /\ ~ In (pr_ty ps) (pr_written prog ps) /\
                  ~ In (pr_tz ps) (pr_written prog ps).
Proof.
  pose proof (sh_apart _ _ _ _ Hsh) as Hap.
  (* only the own registers may equal a thread register; the loops' registers are apart by the plan *)
  assert (Hown : forall r, In r [pr_tx ps; pr_ty ps; pr_tz ps] -> ~ In r [z; t; a; p]).
  { intros r Hr. apply Hap. cbn [In] in Hr |- *. tauto. }
  assert (Hloops : forall r, In r [pr_tx ps; pr_ty ps; pr_tz ps] ->
            ~ In r (match pr_loops ps with
                    | Some zp => zero_regs_list (zp_regs zp) ++
                                 List.concat (map (fun j => map fst (zero_temps_of prog zp j))
                                                  (seq 0 (List.length (zp_bases zp))))
                    | None => [] end)).
  { intros r Hr. destruct (pr_loops ps) as [zp|] eqn:Hl; [|intros []].
    destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & _ & _ & _ & _ & _ & Hx & Hy & Hz & _).
    intros Hin. apply in_app_or in Hin. destruct Hin as [Hin|Hin].
    - apply (zf_apart _ _ _ Hform r Hin). unfold zero_read. apply in_or_app. left. rewrite Hx, Hy, Hz. exact Hr.
    - apply in_concat in Hin. destruct Hin as [l [Hl' Hin]]. apply in_map_iff in Hl'. destruct Hl' as [j [<- Hj]].
      apply in_map_iff in Hin. destruct Hin as [[v from] [Hv Hin]]. cbn [fst] in Hv. subst v.
      apply in_seq in Hj.
      destruct (nth_error (zp_bases zp) j) as [base|] eqn:Hbase; [|apply nth_error_None in Hbase; lia].
      destruct (zf_loops _ _ _ Hform j base Hbase) as (_ & _ & _ & Htemps).
      destruct (Htemps _ _ Hin) as (_ & _ & _ & Hnr). apply Hnr. unfold zero_read. apply in_or_app. left.
      rewrite Hx, Hy, Hz. exact Hr. }
  unfold pr_written.
  repeat split; intros Hin; apply in_app_or in Hin; destruct Hin as [Hin|Hin];
    first [ eapply Hown; [|exact Hin]; in_list | eapply Hloops; [|exact Hin]; in_list ].
Qed.

(* The six instructions of the index. *)
Lemma index_rest : skipn 6 (block_body prog 0) <> [].
Proof.
  destruct (pr_loops ps) as [zp|] eqn:Hl.
  - destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & Hne & _).
    destruct (zp_bases zp) as [|b0 bs] eqn:Hb; [contradiction|].
    destruct (zf_loops _ _ _ Hform 0 b0 ltac:(rewrite Hb; reflexivity)) as (Hs & _).
    unfold zero_setup_block in Hs. rewrite Hs. unfold zero_setup_ir. discriminate.
  - destruct (sh_static _ _ _ _ Hsh) as [rest Hrest]. unfold static_start in Hrest. rewrite Hl in Hrest.
    cbn [fst snd] in Hrest. rewrite Hrest. intros E. apply app_eq_nil in E. destruct E as [_ E]. discriminate E.
Qed.

Lemma index_run : exists rs, run_ 6 = Some (at_ O 6 rs) /\ rs z = 0 /\ rs t = T /\
  forall r, r <> z -> r <> t -> r <> a -> rs r = init r.
Proof.
  pose proof (sh_index _ _ _ _ Hsh) as Hidx.
  assert (Hskip : skipn 0 (block_body prog 0) = index_ir ps ++ skipn 6 (block_body prog 0)).
  { cbn [skipn]. rewrite <- (firstn_skipn 6 (block_body prog 0)) at 1. rewrite Hidx. reflexivity. }
  assert (Hplain : forallb plain (index_ir ps) = true) by reflexivity.
  pose proof (run_plain_inside prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                (lf_local_window F) (index_ir ps) 0 0 0 (e_init Ev) _ eq_refl Hskip index_rest Hplain) as Hrun.
  change (List.length (index_ir ps)) with 6%nat in Hrun. cbn [plus] in Hrun.
  exists (fold_left exec (index_ir ps) (e_init Ev)). split; [exact Hrun|].
  destruct (own_distinct _ _ _ _ Hsh) as (Hzt & Hza & _ & Hta & _ & _).
  destruct (sh_widths _ _ _ _ Hsh) as (Wz & Wt & Wa & _).
  pose proof (sh_apart _ _ _ _ Hsh) as Hap.
  assert (Hx : z <> pr_tx ps /\ t <> pr_tx ps /\ a <> pr_tx ps)
    by (pose proof (Hap (pr_tx ps) ltac:(left; reflexivity)); cbn in *; tauto).
  assert (Hy : z <> pr_ty ps /\ t <> pr_ty ps /\ a <> pr_ty ps)
    by (pose proof (Hap (pr_ty ps) ltac:(right; left; reflexivity)); cbn in *; tauto).
  assert (Hz : z <> pr_tz ps /\ t <> pr_tz ps /\ a <> pr_tz ps)
    by (pose proof (Hap (pr_tz ps) ltac:(right; right; left; reflexivity)); cbn in *; tauto).
  destruct Hx as (Hxz & Hxt & Hxa). destruct Hy as (Hyz & Hyt & Hya). destruct Hz as (Hzz & Hzt' & Hza').
  destruct tid_bounds as (Htx & Hty & Htz). destruct dims as (Hbx & Hby & Hbz & HN & _).
  destruct (index_arith _ _ _ _ _ _ Hbx Hby Hbz Htx Hty Htz) as ((Ix & Iy & Iz) & Ia & Ib).
  unfold pr_threads in HN.
  assert (P : 2 ^ Z.of_nat 32 = 2 ^ 32) by reflexivity.
  unfold index_ir, index_plan, zero_index_ir. cbn [fold_left exec1 dest result opv zp_regs zr_zero zr_index zr_address
    zp_tx zp_ty zp_tz zp_bx zp_by].
  split; [|split].
  - upd_simpl. rewrite Wz. reflexivity.
  - upd_simpl. rewrite Wt, Wa. unfold thread_of.
    rewrite (trunc_small 32 (init (pr_tz ps))) by (rewrite P; lia).
    rewrite (trunc_small 32 (init (pr_ty ps))) by (rewrite P; lia).
    rewrite (trunc_small 32 (init (pr_tx ps))) by (rewrite P; lia).
    rewrite (trunc_small 32 (init (pr_tz ps) * pr_by ps + init (pr_ty ps))) by (rewrite P; lia).
    rewrite trunc_small by (rewrite P; lia). reflexivity.
  - intros r Hrz Hrt Hra. upd_simpl. reflexivity.
Qed.

(* The loops, from GateZero.v. *)
Lemma loops_done : forall zp, pr_loops ps = Some zp ->
  exists M0 rs, run_ M0 = Some (at_ m O rs) /\ rs z = 0 /\ rs t = T /\
    forall j B x, nth_error (zp_bases zp) j = Some B -> 0 <= x < zero_elements zp init -> x mod N = T ->
      exists M' s, (M' < M0)%nat /\ run_ M' = Some s /\
        fetch prog (blk s) (idx s) = Some (IShA None KStore a 0 w (Some (OReg z))) /\
        regs s a = init B + x * w /\ regs s z = 0.
Proof.
  intros zp Hl.
  destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & Hne & Hz & Ht & Ha & Hp & Hx & Hy & Hz' & Hbx & Hby & Hbz & Hw & Hbases).
  destruct tid_bounds as (Htx & Hty & Htz).
  assert (Htx' : 0 <= init (zp_tx zp) < zp_bx zp) by (rewrite Hx, Hbx; exact Htx).
  assert (Hty' : 0 <= init (zp_ty zp) < zp_by zp) by (rewrite Hy, Hby; exact Hty).
  assert (Htz' : 0 <= init (zp_tz zp) < zp_bz zp) by (rewrite Hz', Hbz; exact Htz).
  assert (Hsz : forall s c, In (s, c) (zp_terms zp) -> 0 <= init s < 2 ^ 64) by (intros s c Hin; exact (Hsizes zp s c Hl Hin)).
  pose proof (Hshared zp Hl) as HD.
  pose proof (zf_width _ _ _ Hform) as Hw01. pose proof (zf_threads _ _ _ Hform) as HN.
  pose proof (zf_bx _ _ _ Hform). pose proof (zf_by _ _ _ Hform). pose proof (zf_bz _ _ _ Hform).
  assert (HN0 : 0 < zp_threads zp) by (unfold zp_threads; apply Z.mul_pos_pos; [apply Z.mul_pos_pos|]; lia).
  set (C := zero_elements zp init) in *.
  assert (HC : 0 <= C <= shared_max).
  { split.
    - destruct (Z.lt_ge_cases C 0) as [Hneg|]; [|assumption].
      assert (C * zp_width zp < 0) by (apply Z.mul_neg_pos; lia). lia.
    - assert (C * 1 <= C * zp_width zp) by (apply Z.mul_le_mono_nonneg_l; lia). lia. }
  assert (Hcount : 0 <= C /\ C + zp_threads zp < 2 ^ 32) by (unfold shared_max in HC; lia).
  assert (Hb : forall B, In B (zp_bases zp) -> 0 <= init B /\ init B + C * zp_width zp < 2 ^ 32).
  { intros B HB. destruct (Hbases B HB) as (f & _ & _ & Hbd). destruct (bounded_in_init B shared_max Hbd) as [Hb0 Hb1].
    unfold window_size in Hb1. lia. }
  destruct (loops_run prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
              (lf_local_window F) zp Hform Htx' Hty' Htz' Hsz Hcount Hb Hne (List.length (zp_bases zp)) (le_n _))
    as (M0 & rs & Hrun & Hz0 & Ht0 & _ & Hstores).
  exists M0, rs.
  assert (Hzs : zero_start (List.length (zp_bases zp)) = (pr_blocks ps, O)).
  { destruct (zp_bases zp) as [|b0 bs] eqn:Hbl; [contradiction|]. cbn [zero_start List.length].
    unfold pr_blocks. rewrite Hl, Hbl. cbn [List.length]. f_equal. lia. }
  rewrite Hzs in Hrun. cbn [fst snd] in Hrun.
  split; [exact Hrun|]. rewrite Hz in Hz0. rewrite Ht in Ht0. split; [exact Hz0|]. split.
  { rewrite Ht0. unfold zero_thread, thread_of. rewrite Hx, Hy, Hz', Hbx, Hby. reflexivity. }
  intros j B x Hj Hx' Hmod.
  assert (Hjl : (j < List.length (zp_bases zp))%nat) by (apply nth_error_Some; rewrite Hj; discriminate).
  assert (Hmod' : x mod zp_threads zp = zero_thread zp init).
  { unfold zero_thread, zp_threads. rewrite Hx, Hy, Hz', Hbx, Hby, Hbz. exact Hmod. }
  destruct (Hstores j B x Hjl Hj Hx' Hmod') as (M' & s & HM' & Hrun' & Hf & _ & Hreg & Hz'').
  exists M', s. rewrite Ha, Hw, Hz in Hf. rewrite Ha, Hw in Hreg. rewrite Hz in Hz''.
  split; [exact HM'|]. split; [exact Hrun'|]. split; [exact Hf|]. split; [exact Hreg|exact Hz''].
Qed.

(* The static stores. *)
Definition group_stores (v : static_var) : list instr :=
  map (fun k => IShA None KStore a (w * N * Z.of_nat k) w (Some (OReg z))) (seq 0 (sv_full v)).

Definition group_tail (v : static_var) : list instr :=
  if Z.eqb (sv_left v) 0 then [] else
  [ISetp None CLt 32 false p t (OImm (sv_left v));
   IShA (Some (false, p)) KStore a (w * N * Z.of_nat (sv_full v)) w (Some (OReg z))].

Lemma static_ir_eq : forall v,
  static_ir ps v = IMov None a (sv_base v) :: IMad None 32 a t (OImm w) (OReg a) :: group_stores v ++ group_tail v.
Proof. intros v. reflexivity. Qed.

Lemma stores_regs : forall l rs,
  fold_left exec (map (fun k => IShA None KStore a (w * N * Z.of_nat k) w (Some (OReg z))) l) rs = rs.
Proof. induction l as [|k l IH]; intros rs; [reflexivity|]. cbn [map fold_left]. apply IH. Qed.

Lemma stores_plain : forall l,
  forallb plain (map (fun k => IShA None KStore a (w * N * Z.of_nat k) w (Some (OReg z))) l) = true.
Proof. intros l. apply forallb_forall. intros i Hi. apply in_map_iff in Hi. destruct Hi as [k [<- _]]. reflexivity. Qed.

Lemma step_sha : forall n b k rs g x off wd' v,
  fetch prog b k = Some (IShA g KStore x off wd' v) ->
  step_ n (at_ b k rs) = Some (goto (next_pos prog b k) rs).
Proof.
  intros n b k rs g x off wd' v Hf. unfold step. cbn [blk idx regs]. rewrite Hf. cbn [guard_of].
  destruct (holds g rs); reflexivity.
Qed.

(* An element of the thread's share of a static variable: its index in
   elements, and the round and the store that write it. *)
Lemma share_element : forall v x, In v (pr_static ps) ->
  init (sv_base v) <= x < init (sv_base v) + sv_bytes ps v ->
  ((x - init (sv_base v)) / w) mod N = T ->
  exists e k, e = T + N * Z.of_nat k /\ 0 <= e /\ (e + 1) * w <= sv_bytes ps v /\
    init (sv_base v) + e * w <= x < init (sv_base v) + e * w + w /\
    ((k < sv_full v)%nat \/ (k = sv_full v /\ T < sv_left v)).
Proof.
  intros v x Hv Hx Hmod. pose proof threads_pos' as HN. destruct dims as (_ & _ & _ & _ & Hw0 & _).
  pose proof (sh_left _ _ _ _ Hsh v Hv) as Hleft. pose proof thread_bounds' as HT.
  set (e := (x - init (sv_base v)) / w).
  assert (He : 0 <= e) by (apply Z.div_pos; lia).
  assert (Hex : init (sv_base v) + e * w <= x < init (sv_base v) + e * w + w).
  { pose proof (Z.mul_div_le (x - init (sv_base v)) w ltac:(lia)).
    pose proof (Z.mod_pos_bound (x - init (sv_base v)) w ltac:(lia)).
    pose proof (Z.div_mod (x - init (sv_base v)) w ltac:(lia)). unfold e. lia. }
  assert (Hlt : (e + 1) * w <= sv_bytes ps v).
  { unfold sv_bytes in *. pose proof (Z.mod_pos_bound (x - init (sv_base v)) w ltac:(lia)).
    pose proof (Z.div_mod (x - init (sv_base v)) w ltac:(lia)).
    assert (Hdiv : (x - init (sv_base v)) / w < N * Z.of_nat (sv_full v) + sv_left v).
    { apply Z.div_lt_upper_bound; [lia|]. rewrite Z.mul_comm. lia. }
    fold e in Hdiv. nia. }
  set (k := Z.to_nat (e / N)).
  assert (Hk : e = T + N * Z.of_nat k).
  { unfold k. rewrite Z2Nat.id by (apply Z.div_pos; lia). rewrite <- Hmod. fold e.
    rewrite (Z.mul_comm N). pose proof (Z.div_mod e N ltac:(lia)). lia. }
  exists e, k. split; [exact Hk|]. split; [exact He|]. split; [exact Hlt|]. split; [exact Hex|].
  unfold sv_bytes in Hlt.
  assert (Hew : e < N * Z.of_nat (sv_full v) + sv_left v) by nia.
  destruct (Nat.lt_ge_cases k (sv_full v)) as [Hlt'|Hge]; [left; exact Hlt'|]. right.
  assert (Hkf : k = sv_full v) by (apply Nat2Z.inj; nia). split; [exact Hkf|]. subst k. lia.
Qed.

Lemma group_run : forall v b k n rs rest, In v (pr_static ps) ->
  run_ n = Some (at_ b k rs) -> skipn k (block_body prog b) = static_ir ps v ++ rest -> rest <> [] ->
  rs z = 0 -> rs t = T -> rs (sv_base v) = init (sv_base v) ->
  exists rs', run_ (n + List.length (static_ir ps v)) = Some (at_ b (k + List.length (static_ir ps v)) rs') /\
    rs' z = 0 /\ rs' t = T /\ (forall r, r <> a -> r <> p -> rs' r = rs r) /\
    forall x, init (sv_base v) <= x < init (sv_base v) + sv_bytes ps v ->
      ((x - init (sv_base v)) / w) mod N = T ->
      exists n' s, (n <= n' < n + List.length (static_ir ps v))%nat /\ run_ n' = Some s /\
        exists g off, fetch prog (blk s) (idx s) = Some (IShA g KStore a off w (Some (OReg z))) /\
          holds g (regs s) = true /\ regs s z = 0 /\
          trunc 32 (regs s a + off) <= x < trunc 32 (regs s a + off) + w.
Proof.
  intros v b k n rs rest Hv Hrun Hskip Hrest Hz0 Ht0 Hbase.
  rewrite static_ir_eq in Hskip |- *.
  destruct (own_distinct _ _ _ _ Hsh) as (Hzt & Hza & Hzp & Hta & Htp & Hap).
  destruct (sh_widths _ _ _ _ Hsh) as (Wz & Wt & Wa & Wp).
  destruct (static_bounds v Hv) as [Hb0 Hb1]. pose proof (sh_left _ _ _ _ Hsh v Hv) as Hleft.
  destruct dims as (Hbx & Hby & Hbz & HN & Hw0 & Hw1). pose proof thread_bounds' as HT. pose proof threads_pos' as HN0.
  assert (P : 2 ^ Z.of_nat 32 = 2 ^ 32) by reflexivity.
  unfold window_size in Hb1.
  assert (Hbytes : 0 <= sv_bytes ps v) by (unfold sv_bytes; nia).
  set (full := sv_full v) in *. set (left := sv_left v) in *. set (base := sv_base v) in *.
  (* the registers after the address computation *)
  set (rs1 := upd (wdn F) (upd (wdn F) rs a (init base)) a (T * w + init base)).
  assert (Hrs1 : forall r, r <> a -> rs1 r = rs r).
  { intros r Hr. unfold rs1. upd_simpl. reflexivity. }
  assert (Hrs1a : rs1 a = trunc 32 (T * w + init base)).
  { unfold rs1. upd_simpl. rewrite Wa. reflexivity. }
  assert (Hfold : fold_left exec (IMov None a base :: IMad None 32 a t (OImm w) (OReg a) :: group_stores v) rs = rs1).
  { cbn [fold_left exec1 dest result opv]. unfold group_stores. rewrite stores_regs. unfold rs1.
    rewrite Hbase. f_equal. upd_simpl. rewrite Wa. rewrite Ht0. rewrite (trunc_small 32 (init base)) by (rewrite P; lia).
    reflexivity. }
  (* the run through the stores *)
  assert (Hplain1 : forallb plain (IMov None a base :: IMad None 32 a t (OImm w) (OReg a) :: group_stores v) = true).
  { cbn [forallb plain]. unfold group_stores. rewrite stores_plain. reflexivity. }
  assert (Hskip1 : skipn k (block_body prog b) =
                   (IMov None a base :: IMad None 32 a t (OImm w) (OReg a) :: group_stores v) ++ (group_tail v ++ rest)).
  { rewrite Hskip. cbn [app]. rewrite <- app_assoc. reflexivity. }
  assert (Hrest1 : group_tail v ++ rest <> []) by (intros E; apply app_eq_nil in E; destruct E as [_ E]; exact (Hrest E)).
  pose proof (run_plain_inside prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                (lf_local_window F) _ b k n rs _ Hrun Hskip1 Hrest1 Hplain1) as H1.
  rewrite Hfold in H1. cbn [List.length] in H1. unfold group_stores in H1. rewrite map_length, seq_length in H1.
  fold full in H1.
  replace (n + S (S full))%nat with (n + 2 + full)%nat in H1 by lia.
  replace (k + S (S full))%nat with (k + 2 + full)%nat in H1 by lia.
  (* a store of round k' *)
  assert (Hstore : forall k', (k' < full)%nat ->
            run_ (n + 2 + k') = Some (at_ b (k + 2 + k') rs1) /\
            fetch prog b (k + 2 + k') = Some (IShA None KStore a (w * N * Z.of_nat k') w (Some (OReg z)))).
  { intros k' Hk'.
    assert (Hpre : skipn k (block_body prog b) =
                   (IMov None a base :: IMad None 32 a t (OImm w) (OReg a) :: firstn k' (group_stores v)) ++
                   (skipn k' (group_stores v) ++ group_tail v ++ rest)).
    { pose proof Hskip1 as Hs. rewrite <- (firstn_skipn k' (group_stores v)) in Hs. rewrite Hs. cbn [app].
      rewrite <- app_assoc. reflexivity. }
    assert (Hlenk : List.length (firstn k' (group_stores v)) = k').
    { rewrite firstn_length. unfold group_stores. rewrite map_length, seq_length. lia. }
    assert (Hrestk : skipn k' (group_stores v) ++ group_tail v ++ rest <> []).
    { intros E. apply app_eq_nil in E. destruct E as [E _]. unfold group_stores in E.
      pose proof (f_equal (@List.length instr) E) as El. rewrite skipn_length, map_length, seq_length in El. cbn in El. lia. }
    assert (Hplk : forallb plain (IMov None a base :: IMad None 32 a t (OImm w) (OReg a) :: firstn k' (group_stores v)) = true).
    { cbn [forallb plain]. apply forallb_forall. intros i Hi. apply (proj1 (forallb_forall _ _) (stores_plain (seq 0 full))).
      unfold group_stores in Hi. rewrite <- (firstn_skipn k' (map (fun k0 => IShA None KStore a (w * N * Z.of_nat k0) w (Some (OReg z))) (seq 0 full))).
      apply in_or_app. left. exact Hi. }
    pose proof (run_plain_inside prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                  (lf_local_window F) _ b k n rs _ Hrun Hpre Hrestk Hplk) as Hk.
    cbn [List.length] in Hk. rewrite Hlenk in Hk.
    assert (Hfoldk : fold_left exec (IMov None a base :: IMad None 32 a t (OImm w) (OReg a) :: firstn k' (group_stores v)) rs = rs1).
    { cbn [fold_left]. unfold group_stores. rewrite firstn_map, stores_regs. cbn [exec1 dest result opv].
      unfold rs1. rewrite Hbase. f_equal. upd_simpl. rewrite Wa, Ht0, (trunc_small 32 (init base)) by (rewrite P; lia).
      reflexivity. }
    rewrite Hfoldk in Hk. replace (n + S (S k'))%nat with (n + 2 + k')%nat in Hk by lia.
    replace (k + S (S k'))%nat with (k + 2 + k')%nat in Hk by lia. split; [exact Hk|].
    unfold fetch. replace (k + 2 + k')%nat with (k + (2 + k'))%nat by lia.
    rewrite <- nth_error_skipn', Hskip. replace (2 + k')%nat with (S (S k')) by lia. cbn [nth_error app].
    rewrite <- app_assoc, nth_error_app1.
    - unfold group_stores. rewrite nth_error_map, nth_error_seq0 by exact Hk'. reflexivity.
    - unfold group_stores. rewrite map_length, seq_length. exact Hk'. }
  (* the address of a store that runs *)
  assert (Haddr : forall k', (T + N * Z.of_nat k' + 1) * w <= sv_bytes ps v ->
            trunc 32 (rs1 a + w * N * Z.of_nat k') = init base + (T + N * Z.of_nat k') * w).
  { intros k' Hk'. rewrite Hrs1a.
    assert (HTw : 0 <= T * w) by nia.
    assert (Hkw : 0 <= N * Z.of_nat k' * w) by nia.
    assert (HTw' : T * w <= (T + N * Z.of_nat k') * w) by nia.
    assert (Hsum : (T + N * Z.of_nat k') * w + w <= sv_bytes ps v) by nia.
    rewrite (trunc32_small (T * w + init base)) by lia.
    replace (T * w + init base + w * N * Z.of_nat k') with (init base + (T + N * Z.of_nat k') * w) by ring.
    rewrite trunc32_small; [reflexivity|lia]. }
  destruct (Z.eqb_spec left 0) as [Hl0|Hl0].
  - (* no last round *)
    assert (Htail : group_tail v = []) by (unfold group_tail; fold left; rewrite Hl0; reflexivity).
    rewrite Htail, app_nil_r. cbn [List.length] in *.
    unfold group_stores. rewrite map_length, seq_length.
    exists rs1. fold full. replace (n + S (S full))%nat with (n + 2 + full)%nat by lia.
    replace (k + S (S full))%nat with (k + 2 + full)%nat by lia.
    split; [exact H1|]. split; [rewrite Hrs1 by exact Hza; exact Hz0|].
    split; [rewrite Hrs1 by exact Hta; exact Ht0|]. split; [intros r Hr _; exact (Hrs1 r Hr)|].
    intros x Hx Hmod. destruct (share_element v x Hv Hx Hmod) as (e & k' & Hek & He & Hew & Hex & Hcase).
    destruct Hcase as [Hk'|[_ Hc]]; [|lia].
    destruct (Hstore k' Hk') as [Hrunk Hfk]. exists (n + 2 + k')%nat, (at_ b (k + 2 + k') rs1).
    split; [lia|]. split; [exact Hrunk|]. exists None, (w * N * Z.of_nat k'). cbn [blk idx regs].
    split; [exact Hfk|]. split; [reflexivity|]. split; [rewrite Hrs1 by exact Hza; exact Hz0|].
    rewrite (Haddr k' ltac:(rewrite <- Hek; exact Hew)). rewrite <- Hek. unfold base in *. lia.
  - (* the last round *)
    assert (Htail : group_tail v = [ISetp None CLt 32 false p t (OImm left);
                                    IShA (Some (false, p)) KStore a (w * N * Z.of_nat full) w (Some (OReg z))])
      by (unfold group_tail; fold left; destruct (Z.eqb_spec left 0) as [E|_]; [exfalso; exact (Hl0 E)|reflexivity]).
    rewrite Htail.
    set (rs2 := upd (wdn F) rs1 p (bool_z (T <? left))).
    (* the comparison *)
    assert (Hskip2 : skipn (k + 2 + full) (block_body prog b) =
                     [ISetp None CLt 32 false p t (OImm left)] ++
                     (IShA (Some (false, p)) KStore a (w * N * Z.of_nat full) w (Some (OReg z)) :: rest)).
    { replace (k + 2 + full)%nat with (k + List.length (IMov None a base :: IMad None 32 a t (OImm w) (OReg a) :: group_stores v))%nat
        by (cbn [List.length]; unfold group_stores; rewrite map_length, seq_length; lia).
      rewrite (skipn_app_rest _ _ _ _ Hskip1). rewrite Htail. reflexivity. }
    pose proof (run_plain_inside prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                  (lf_local_window F) [ISetp None CLt 32 false p t (OImm left)] b (k + 2 + full) (n + 2 + full) rs1 _
                  H1 Hskip2 ltac:(discriminate) eq_refl) as H2.
    cbn [List.length fold_left exec1 dest result opv] in H2.
    assert (Hrs2eq : upd (wdn F) rs1 p (bool_z (cmp_eval CLt (extend 32 false (rs1 t)) (extend 32 false left))) = rs2).
    { unfold rs2. rewrite Hrs1 by exact Hta. rewrite Ht0.
      rewrite (extend32_small T) by lia. rewrite (extend32_small left) by lia. reflexivity. }
    rewrite Hrs2eq in H2.
    (* the guarded store *)
    assert (Hf3 : fetch prog b (k + 2 + full + 1) = Some (IShA (Some (false, p)) KStore a (w * N * Z.of_nat full) w (Some (OReg z)))).
    { unfold fetch. rewrite <- nth_error_skipn', Hskip2. reflexivity. }
    assert (H3 : run_ (n + 2 + full + 2) = Some (at_ b (k + 2 + full + 2) rs2)).
    { replace (n + 2 + full + 2)%nat with (S (n + 2 + full + 1))%nat by lia.
      rewrite (run_succ _ _ _ _ _ _ _ _ _ _ _ H2), (step_sha _ _ _ _ _ _ _ _ _ Hf3).
      rewrite next_pos_inside.
      - replace (S (k + 2 + full + 1))%nat with (k + 2 + full + 2)%nat by lia. reflexivity.
      - pose proof (skipn_cons_length _ _ _ _ (skipn_app_rest _ _ _ _ Hskip2)) as Hl. cbn [List.length] in Hl.
        destruct rest as [|r0 rest]; [destruct (Hrest eq_refl)|]. cbn [List.length] in Hl. unfold block_len. lia. }
    assert (Hrs2 : forall r, r <> p -> rs2 r = rs1 r) by (intros r Hr; unfold rs2; upd_simpl; reflexivity).
    exists rs2. cbn [List.length]. unfold group_stores. rewrite app_length, map_length, seq_length. cbn [List.length].
    fold full. replace (n + S (S (full + 2)))%nat with (n + 2 + full + 2)%nat by lia.
    replace (k + S (S (full + 2)))%nat with (k + 2 + full + 2)%nat by lia.
    split; [exact H3|].
    split; [rewrite Hrs2 by exact Hzp; rewrite Hrs1 by exact Hza; exact Hz0|].
    split; [rewrite Hrs2 by exact Htp; rewrite Hrs1 by exact Hta; exact Ht0|].
    split; [intros r Hra Hrp; rewrite Hrs2 by exact Hrp; exact (Hrs1 r Hra)|].
    intros x Hx Hmod. destruct (share_element v x Hv Hx Hmod) as (e & k' & Hek & He & Hew & Hex & Hcase).
    destruct Hcase as [Hk'|[Hkf HTl]].
    + destruct (Hstore k' Hk') as [Hrunk Hfk]. exists (n + 2 + k')%nat, (at_ b (k + 2 + k') rs1).
      split; [lia|]. split; [exact Hrunk|]. exists None, (w * N * Z.of_nat k'). cbn [blk idx regs].
      split; [exact Hfk|]. split; [reflexivity|]. split; [rewrite Hrs1 by exact Hza; exact Hz0|].
      rewrite (Haddr k' ltac:(rewrite <- Hek; exact Hew)). rewrite <- Hek. unfold base in *. lia.
    + exists (n + 2 + full + 1)%nat, (at_ b (k + 2 + full + 1) rs2).
      split; [lia|]. split; [exact H2|]. exists (Some (false, p)), (w * N * Z.of_nat full). cbn [blk idx regs].
      split; [exact Hf3|]. split.
      { unfold holds. unfold rs2. upd_simpl. destruct (Z.ltb_spec T left); [|lia]. cbn [bool_z].
        rewrite (trunc_small _ 1); [reflexivity|].
        split; [lia|]. apply Z.lt_le_trans with (2 ^ 1); [lia|]. apply Z.pow_le_mono_r; lia. }
      split; [rewrite Hrs2 by exact Hzp; rewrite Hrs1 by exact Hza; exact Hz0|].
      rewrite Hrs2 by exact Hap. unfold full in *. rewrite <- Hkf.
      rewrite (Haddr k' ltac:(rewrite <- Hek; exact Hew)). rewrite <- Hek. unfold base in *. lia.
Qed.

Lemma statics_run : forall vs b k n rs rest, (forall v, In v vs -> In v (pr_static ps)) ->
  run_ n = Some (at_ b k rs) -> skipn k (block_body prog b) = flat_map (static_ir ps) vs ++ rest -> rest <> [] ->
  rs z = 0 -> rs t = T -> (forall v, In v vs -> rs (sv_base v) = init (sv_base v)) ->
  exists rs', run_ (n + List.length (flat_map (static_ir ps) vs)) =
                Some (at_ b (k + List.length (flat_map (static_ir ps) vs)) rs') /\
    rs' z = 0 /\ rs' t = T /\ (forall r, r <> a -> r <> p -> rs' r = rs r) /\
    forall v x, In v vs -> init (sv_base v) <= x < init (sv_base v) + sv_bytes ps v ->
      ((x - init (sv_base v)) / w) mod N = T ->
      exists n' s, (n <= n' < n + List.length (flat_map (static_ir ps) vs))%nat /\ run_ n' = Some s /\
        exists g off, fetch prog (blk s) (idx s) = Some (IShA g KStore a off w (Some (OReg z))) /\
          holds g (regs s) = true /\ regs s z = 0 /\
          trunc 32 (regs s a + off) <= x < trunc 32 (regs s a + off) + w.
Proof.
  induction vs as [|v vs IH]; intros b k n rs rest Hvs Hrun Hskip Hrest Hz0 Ht0 Hbases.
  - exists rs. cbn [flat_map List.length]. rewrite !Nat.add_0_r. split; [exact Hrun|].
    split; [exact Hz0|]. split; [exact Ht0|]. split; [intros; reflexivity|]. intros v x [] _ _.
  - cbn [flat_map] in Hskip |- *. rewrite <- app_assoc in Hskip.
    assert (Hrest1 : flat_map (static_ir ps) vs ++ rest <> [])
      by (intros E; apply app_eq_nil in E; destruct E as [_ E]; exact (Hrest E)).
    destruct (group_run v b k n rs _ (Hvs v (or_introl eq_refl)) Hrun Hskip Hrest1 Hz0 Ht0
                (Hbases v (or_introl eq_refl))) as (rs1 & H1 & Hz1 & Ht1 & Hkeep1 & Hwit1).
    pose proof (skipn_app_rest _ _ _ _ Hskip) as Hskip2.
    destruct (own_distinct _ _ _ _ Hsh) as (_ & _ & _ & _ & _ & _).
    assert (Hbases2 : forall v', In v' vs -> rs1 (sv_base v') = init (sv_base v')).
    { intros v' Hv'. rewrite Hkeep1; [exact (Hbases v' (or_intror Hv'))| |].
      - intros E. apply (base_apart v' (Hvs v' (or_intror Hv'))). rewrite E. apply pr_written_own. cbn. tauto.
      - intros E. apply (base_apart v' (Hvs v' (or_intror Hv'))). rewrite E. apply pr_written_own. cbn. tauto. }
    destruct (IH b (k + List.length (static_ir ps v))%nat (n + List.length (static_ir ps v))%nat rs1 rest
                (fun v' Hv' => Hvs v' (or_intror Hv')) H1 Hskip2 Hrest Hz1 Ht1 Hbases2)
      as (rs2 & H2 & Hz2 & Ht2 & Hkeep2 & Hwit2).
    exists rs2. rewrite app_length.
    replace (n + (List.length (static_ir ps v) + List.length (flat_map (static_ir ps) vs)))%nat
      with (n + List.length (static_ir ps v) + List.length (flat_map (static_ir ps) vs))%nat by lia.
    replace (k + (List.length (static_ir ps v) + List.length (flat_map (static_ir ps) vs)))%nat
      with (k + List.length (static_ir ps v) + List.length (flat_map (static_ir ps) vs))%nat by lia.
    split; [exact H2|]. split; [exact Hz2|]. split; [exact Ht2|].
    split; [intros r Hra Hrp; rewrite (Hkeep2 r Hra Hrp); exact (Hkeep1 r Hra Hrp)|].
    intros v' x [<-|Hv'] Hx Hmod.
    + destruct (Hwit1 x Hx Hmod) as (n' & s & Hn' & Hrun' & Hw). exists n', s. split; [lia|]. split; [exact Hrun'|exact Hw].
    + destruct (Hwit2 v' x Hv' Hx Hmod) as (n' & s & Hn' & Hrun' & Hw). exists n', s. split; [lia|]. split; [exact Hrun'|exact Hw].
Qed.

(* Registers outside the prologue's own keep their launch values while the
   run is before the barrier. *)
Lemma quiet_step_regs : forall n s s' ins, fetch prog (blk s) (idx s) = Some ins -> quiet z ins = true ->
  step_ n s = Some s' -> forall r, (forall d, dest ins = Some d -> d <> r) -> regs s' r = regs s r.
Proof.
  intros n s s' ins Hf Hq Hs r Hd. unfold step in Hs. rewrite Hf in Hs.
  destruct (negb (holds (guard_of ins) (regs s))); [injection Hs as <-; reflexivity|].
  destruct ins; cbn [quiet] in Hq; try discriminate; try (injection Hs as <-; reflexivity);
    try (injection Hs as <-; cbn [goto regs]; unfold upd; cbn [dest] in Hd;
         destruct (Pos.eqb_spec r d) as [->|]; [exfalso; exact (Hd d eq_refl eq_refl)|reflexivity]).
Qed.

Lemma run_region : forall n s, run_ n = Some s ->
  (forall n' s', (n' < n)%nat -> run_ n' = Some s' -> (blk s', idx s') <> bar) ->
  before_bar ps (blk s, idx s) \/ (blk s, idx s) = bar.
Proof.
  induction n as [|n IH]; intros s Hrun Hnb.
  - cbn in Hrun. injection Hrun as <-. cbn [blk idx]. exact (start_region _ _ _ _ Hsh).
  - destruct (run_ n) as [s0|] eqn:Hr0; [|cbn [run] in Hrun; rewrite Hr0 in Hrun; discriminate].
    rewrite (run_succ _ _ _ _ _ _ _ _ _ n s0 Hr0) in Hrun.
    destruct (IH s0 eq_refl (fun n' s' Hn' => Hnb n' s' ltac:(lia))) as [Hpos|Hpos];
      [|exfalso; exact (Hnb n s0 ltac:(lia) Hr0 Hpos)].
    destruct (step_fetch _ _ _ _ _ _ _ _ _ _ _ _ Hrun) as [ins Hf].
    destruct (region_fetch _ _ _ _ Hsh (blk s0) (idx s0) ins Hpos Hf) as (Hq & _ & _ & Hbr & Hnext).
    destruct (step_pos prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                (lf_local_window F) n s0 s ins Hf Hrun) as [Hn|[[g [tg [-> Hn]]]|[g [-> _]]]].
    + rewrite Hn. exact Hnext.
    + rewrite Hn. exact (proj2 (Hbr g tg eq_refl)).
    + cbn [quiet] in Hq. discriminate.
Qed.

Lemma run_after : forall n s s', run_ n = Some s -> after_bar ps (blk s, idx s) -> run_ (S n) = Some s' ->
  after_bar ps (blk s', idx s').
Proof.
  intros n s s' Hrun Hafter Hrun'. rewrite (run_succ _ _ _ _ _ _ _ _ _ n s Hrun) in Hrun'.
  destruct (step_fetch _ _ _ _ _ _ _ _ _ _ _ _ Hrun') as [ins Hf].
  destruct (step_pos prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
              (lf_local_window F) n s s' ins Hf Hrun') as [Hn|[[g [tg [-> Hn]]]|[g [-> Hn]]]].
  - rewrite Hn. exact (after_next prog ps (blk s) (idx s) Hafter).
  - rewrite Hn. exact (after_branch _ _ _ _ Hsh (blk s) (idx s) g tg Hafter Hf).
  - rewrite Hn. exact (after_exit _ _ _ _ Hsh).
Qed.

Lemma run_after_many : forall n s, run_ n = Some s -> after_bar ps (blk s, idx s) ->
  forall k s', run_ (n + k) = Some s' -> after_bar ps (blk s', idx s').
Proof.
  intros n s Hrun Hafter k. induction k as [|k IH]; intros s' Hrun'.
  - rewrite Nat.add_0_r in Hrun'. rewrite Hrun in Hrun'. injection Hrun' as <-. exact Hafter.
  - rewrite Nat.add_succ_r in Hrun'.
    destruct (run_ (n + k)) as [s0|] eqn:Hr0; [|cbn [run] in Hrun'; rewrite Hr0 in Hrun'; discriminate].
    exact (run_after (n + k) s0 s' Hr0 (IH s0 eq_refl) Hrun').
Qed.

(* The step at the barrier leaves it behind. *)
Lemma bar_step : forall n s s', run_ n = Some s -> (blk s, idx s) = bar -> run_ (S n) = Some s' ->
  after_bar ps (blk s', idx s').
Proof.
  intros n s s' Hrun Hbar Hrun'. rewrite (run_succ _ _ _ _ _ _ _ _ _ n s Hrun) in Hrun'.
  pose proof (fetch_bar _ _ _ _ Hsh) as Hf. rewrite <- Hbar in Hf. cbn [fst snd] in Hf.
  destruct (step_pos prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
              (lf_local_window F) n s s' INop Hf Hrun') as [Hn|[[g [tg [E _]]]|[g [E _]]]]; try discriminate.
  rewrite Hn.
  assert (E1 : blk s = fst bar) by (rewrite <- Hbar; reflexivity).
  assert (E2 : idx s = snd bar) by (rewrite <- Hbar; reflexivity).
  rewrite E1, E2. exact (bar_next prog ps).
Qed.

Lemma run_some_before : forall M s, run_ M = Some s -> forall n, (n <= M)%nat -> exists s', run_ n = Some s'.
Proof.
  induction M as [|M IH]; intros s HM n Hn.
  - assert (n = O) by lia. subst n. exists s. exact HM.
  - cbn [run] in HM. destruct (run_ M) as [s0|] eqn:H0; [|discriminate].
    destruct (Nat.eq_dec n (S M)) as [->|Hne]; [exists s; cbn [run]; rewrite H0; exact HM|].
    exact (IH s0 eq_refl n ltac:(lia)).
Qed.

(* Every step before the first arrival at the barrier is before it. *)
Lemma steps_before : forall M s, run_ M = Some s -> (blk s, idx s) = bar ->
  forall n s', (n < M)%nat -> run_ n = Some s' -> before_bar ps (blk s', idx s').
Proof.
  intros M s HM Hbar.
  assert (Hall : forall n, (n <= M)%nat -> forall n' s', (n' < n)%nat -> run_ n' = Some s' -> before_bar ps (blk s', idx s')).
  { induction n as [|n IH]; intros Hn n' s' Hn' Hrun'; [lia|].
    destruct (Nat.lt_ge_cases n' n) as [Hlt|Hge]; [exact (IH ltac:(lia) n' s' Hlt Hrun')|].
    assert (n' = n) by lia. subst n'.
    assert (Hnb : forall n'' s'', (n'' < n)%nat -> run_ n'' = Some s'' -> (blk s'', idx s'') <> bar).
    { intros n'' s'' Hn'' Hr'' E. apply (not_before_bar ps). rewrite <- E. exact (IH ltac:(lia) n'' s'' Hn'' Hr''). }
    destruct (run_region n s' Hrun' Hnb) as [H|H]; [exact H|].
    exfalso.
    destruct (run_some_before M s HM (S n) ltac:(lia)) as [s1 Hs1].
    pose proof (bar_step n s' s1 Hrun' H Hs1) as Hafter.
    pose proof (run_after_many (S n) s1 Hs1 Hafter (M - S n) s ltac:(replace (S n + (M - S n))%nat with M by lia; exact HM)) as Hm.
    rewrite Hbar in Hm. exact (not_after_bar ps Hm). }
  intros n s' Hn Hrun'. exact (Hall M (le_n M) n s' Hn Hrun').
Qed.

(* ------------------------------------------------------------------ *)
(* The run to the barrier.                                             *)

Lemma pre_region_zero : forall n s, run_ n = Some s ->
  (forall n' s', (n' < n)%nat -> run_ n' = Some s' -> before_bar ps (blk s', idx s')) ->
  (1 <= n)%nat -> (blk s, idx s) <> (O, O) /\ regs s z = 0.
Proof.
  induction n as [|n IH]; intros s Hrun Hbefore Hn; [lia|].
  destruct (run_ n) as [s0|] eqn:H0; [|cbn [run] in Hrun; rewrite H0 in Hrun; discriminate].
  rewrite (run_succ _ _ _ _ _ _ _ _ _ n s0 H0) in Hrun.
  pose proof (Hbefore n s0 ltac:(lia) H0) as Hpos.
  destruct (step_fetch _ _ _ _ _ _ _ _ _ _ _ _ Hrun) as [ins Hf].
  destruct (region_fetch _ _ _ _ Hsh (blk s0) (idx s0) ins Hpos Hf) as (Hq & Hdz & _ & Hbr & _).
  (* the position after the step is not the start *)
  assert (Hnot : (blk s, idx s) <> (O, O)).
  { destruct (step_pos prog (wdn F) (e_mem Ev) (e_arb Ev) (e_shm Ev) (e_init Ev) (lf_cells F) (lf_locals F)
                (lf_local_window F) n s0 s ins Hf Hrun) as [Hn'|[[g [tg [-> Hn']]]|[g [-> _]]]].
    - rewrite Hn'. destruct (next_pos_spec prog (blk s0) (idx s0)) as [E|[E _]]; rewrite E; intros X; injection X; lia.
    - rewrite Hn'. destruct (Hbr g tg eq_refl) as [H1 _]. intros X. injection X. lia.
    - cbn [quiet] in Hq. discriminate. }
  split; [exact Hnot|].
  destruct n as [|n].
  - (* the first instruction writes zero *)
    cbn in H0. injection H0 as <-. cbn [blk idx] in Hf.
    pose proof (sh_index _ _ _ _ Hsh) as Hidx.
    assert (Hf0 : fetch prog 0 0 = Some (IImm None z 0)).
    { unfold fetch. rewrite <- (firstn_skipn 6 (block_body prog 0)), Hidx. reflexivity. }
    rewrite Hf0 in Hf. injection Hf as <-.
    unfold step in Hrun. cbn [blk idx regs] in Hrun. rewrite Hf0 in Hrun. injection Hrun as <-.
    cbn [goto regs]. unfold upd. rewrite Pos.eqb_refl. destruct (sh_widths _ _ _ _ Hsh) as (Wz & _). rewrite Wz. reflexivity.
  - destruct (IH s0 eq_refl (fun n' s' Hn' => Hbefore n' s' ltac:(lia)) ltac:(lia)) as [Hnot0 Hz0].
    rewrite <- Hz0. apply (quiet_step_regs (S n) s0 s ins Hf Hq Hrun).
    intros d Hd ->. exact (Hdz Hnot0 Hd).
Qed.

Lemma pre_region_kept : forall n s, run_ n = Some s ->
  (forall n' s', (n' < n)%nat -> run_ n' = Some s' -> before_bar ps (blk s', idx s')) ->
  forall r, ~ In r (pr_written prog ps) -> regs s r = init r.
Proof.
  induction n as [|n IH]; intros s Hrun Hbefore r Hr.
  - cbn in Hrun. injection Hrun as <-. reflexivity.
  - destruct (run_ n) as [s0|] eqn:H0; [|cbn [run] in Hrun; rewrite H0 in Hrun; discriminate].
    rewrite (run_succ _ _ _ _ _ _ _ _ _ n s0 H0) in Hrun.
    pose proof (Hbefore n s0 ltac:(lia) H0) as Hpos.
    destruct (step_fetch _ _ _ _ _ _ _ _ _ _ _ _ Hrun) as [ins Hf].
    destruct (region_fetch _ _ _ _ Hsh (blk s0) (idx s0) ins Hpos Hf) as (Hq & _ & Hdest & _ & _).
    rewrite <- (IH s0 eq_refl (fun n' s' Hn' => Hbefore n' s' ltac:(lia)) r Hr).
    apply (quiet_step_regs n s0 s ins Hf Hq Hrun).
    intros d Hd ->. exact (Hr (Hdest r Hd)).
Qed.

Theorem prologue_run : exists M rs,
  run_ M = Some (at_ (fst bar) (snd bar) rs) /\ rs z = 0 /\
  (forall n s, (n < M)%nat -> run_ n = Some s -> before_bar ps (blk s, idx s)) /\
  (forall x, share ps init x -> exists n s, (n < M)%nat /\ run_ n = Some s /\
     exists g off, fetch prog (blk s) (idx s) = Some (IShA g KStore a off w (Some (OReg z))) /\
       holds g (regs s) = true /\ regs s z = 0 /\
       trunc 32 (regs s a + off) <= x < trunc 32 (regs s a + off) + w).
Proof.
  destruct (sh_static _ _ _ _ Hsh) as [rest Hrest].
  (* the state at the start of the static stores *)
  assert (Hstart : exists M0 rs0, run_ M0 = Some (at_ (fst (static_start ps)) (snd (static_start ps)) rs0) /\
            rs0 z = 0 /\ rs0 t = T /\ (forall r, ~ In r (pr_written prog ps) -> rs0 r = init r) /\
            (forall x, (exists zp B, pr_loops ps = Some zp /\ In B (zp_bases zp) /\
                          init B <= x < init B + dyn_count ps init * w /\ ((x - init B) / w) mod N = T) ->
               exists n s, (n < M0)%nat /\ run_ n = Some s /\
                 exists g off, fetch prog (blk s) (idx s) = Some (IShA g KStore a off w (Some (OReg z))) /\
                   holds g (regs s) = true /\ regs s z = 0 /\
                   trunc 32 (regs s a + off) <= x < trunc 32 (regs s a + off) + w)).
  { assert (Hcases : (exists zp, pr_loops ps = Some zp) \/ pr_loops ps = None)
      by (destruct (pr_loops ps) as [zp|]; [left; exists zp; reflexivity|right; reflexivity]).
    destruct Hcases as [[zp Hl]|Hl].
    - destruct (loops_done zp Hl) as (M0 & rs0 & Hrun & Hz0 & Ht0 & Hstores).
      exists M0, rs0. unfold static_start. try rewrite Hl. cbn [fst snd]. split; [exact Hrun|].
      split; [exact Hz0|]. split; [exact Ht0|]. split.
      { intros r Hr. pose proof (steps_before M0 _ Hrun) as Hsb. cbn [blk idx] in Hsb.
        (* the start of the static stores is before or at the barrier: no earlier step reaches it *)
        assert (Hbefore : forall n s', (n < M0)%nat -> run_ n = Some s' -> before_bar ps (blk s', idx s')).
        { intros n s' Hn Hr'.
          destruct (Nat.eq_dec (snd bar) O) as [Hb0|Hb0].
          - apply (Hsb ltac:(rewrite (surjective_pairing bar), bar_fst, Hb0; unfold pr_blocks; rewrite Hl; reflexivity) n s' Hn Hr').
          - (* (m, 0) is before the barrier, and reaching the barrier earlier would leave it *)
            assert (Hnb : forall n'' s'', (n'' < M0)%nat -> run_ n'' = Some s'' -> (blk s'', idx s'') <> bar).
            { intros n'' s'' Hn'' Hr'' E. destruct (run_some_before M0 _ Hrun (S n'') ltac:(lia)) as [s1 Hs1].
              pose proof (bar_step n'' s'' s1 Hr'' E Hs1) as Hafter.
              pose proof (run_after_many (S n'') s1 Hs1 Hafter (M0 - S n'') _
                            ltac:(replace (S n'' + (M0 - S n''))%nat with M0 by lia; exact Hrun)) as Hm.
              cbn [blk idx] in Hm. unfold after_bar in Hm. rewrite bar_fst in Hm. unfold pr_blocks in Hm. rewrite Hl in Hm.
              cbn [fst snd] in Hm. lia. }
            assert (Hreach : forall n'' s'', (n'' <= M0)%nat -> run_ n'' = Some s'' ->
                      before_bar ps (blk s'', idx s'') \/ (blk s'', idx s'') = bar).
            { intros n'' s'' Hn'' Hr''. apply (run_region n'' s'' Hr''). intros n3 s3 H3 Hr3. exact (Hnb n3 s3 ltac:(lia) Hr3). }
            destruct (Hreach n s' ltac:(lia) Hr') as [H|H]; [exact H|exfalso; exact (Hnb n s' Hn Hr' H)]. }
        exact (pre_region_kept M0 _ Hrun Hbefore r Hr). }
      intros x (zp' & B & Hl' & HB & Hx & Hmod). try rewrite Hl in Hl'. injection Hl' as <-.
      unfold dyn_count in Hx. try rewrite Hl in Hx.
      destruct (sh_loops _ _ _ _ Hsh zp Hl) as (Hform & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hw & Hbases).
      destruct dims as (_ & _ & _ & _ & Hw0 & _).
      set (e := (x - init B) / w).
      assert (He : 0 <= e < zero_elements zp init).
      { split; [apply Z.div_pos; lia|]. apply Z.div_lt_upper_bound; [lia|]. rewrite Z.mul_comm. lia. }
      apply In_nth_error in HB. destruct HB as [j Hj].
      destruct (Hstores j B e Hj He Hmod) as (M' & s & HM' & Hrun' & Hf & Hreg & Hz').
      exists M', s. split; [exact HM'|]. split; [exact Hrun'|]. exists None, 0.
      split; [exact Hf|]. split; [reflexivity|]. split; [exact Hz'|].
      rewrite Hreg, Z.add_0_r.
      destruct (Hbases B (nth_error_In _ _ Hj)) as (f & _ & _ & Hbd). destruct (bounded_in_init B shared_max Hbd) as [H0 H1].
      pose proof (Hshared zp Hl) as HD. unfold window_size, shared_max in *.
      assert (Hew : e * w + w <= zero_elements zp init * zp_width zp) by (rewrite <- Hw; nia).
      rewrite trunc32_small by lia.
      pose proof (Z.mul_div_le (x - init B) w ltac:(lia)). pose proof (Z.mod_pos_bound (x - init B) w ltac:(lia)).
      pose proof (Z.div_mod (x - init B) w ltac:(lia)). unfold e. lia.
    - destruct index_run as (rs0 & Hrun & Hz0 & Ht0 & Hkeep).
      exists 6%nat, rs0. unfold static_start. try rewrite Hl. cbn [fst snd]. split; [exact Hrun|].
      split; [exact Hz0|]. split; [exact Ht0|]. split.
      { intros r Hr. apply Hkeep; intros ->; apply Hr; apply pr_written_own; cbn; tauto. }
      intros x (zp & _ & E & _). try rewrite Hl in E. discriminate E. }
  destruct Hstart as (M0 & rs0 & Hrun0 & Hz0 & Ht0 & Hkeep0 & Hdyn).
  assert (Hvs : forall v, In v (pr_static ps) -> In v (pr_static ps)) by (intros v Hv; exact Hv).
  assert (Hbases0 : forall v, In v (pr_static ps) -> rs0 (sv_base v) = init (sv_base v))
    by (intros v Hv; exact (Hkeep0 _ (base_apart v Hv))).
  unfold statics_ir in Hrest.
  destruct (statics_run (pr_static ps) _ _ M0 rs0 (INop :: rest) Hvs Hrun0 Hrest ltac:(discriminate) Hz0 Ht0 Hbases0)
    as (rs1 & Hrun1 & Hz1 & _ & _ & Hwit).
  set (M := (M0 + List.length (flat_map (static_ir ps) (pr_static ps)))%nat) in *.
  assert (Hbar : (fst (static_start ps), (snd (static_start ps) + List.length (flat_map (static_ir ps) (pr_static ps)))%nat) = bar)
    by reflexivity.
  exists M, rs1. split; [rewrite <- Hbar; exact Hrun1|]. split; [exact Hz1|].
  assert (Hbefore : forall n s, (n < M)%nat -> run_ n = Some s -> before_bar ps (blk s, idx s)).
  { apply (steps_before M _ Hrun1). cbn [blk idx]. exact Hbar. }
  split; [exact Hbefore|].
  intros x [Hs|Hd].
  - destruct Hs as (v & Hv & Hx & Hmod). destruct (Hwit v x Hv Hx Hmod) as (n & s & Hn & Hrun & Hw').
    exists n, s. split; [unfold M; lia|]. split; [exact Hrun|exact Hw'].
  - destruct (Hdyn x Hd) as (n & s & Hn & Hrun & Hw'). exists n, s. split; [unfold M; lia|]. split; [exact Hrun|exact Hw'].
Qed.

(* ------------------------------------------------------------------ *)
(* Every byte of a shared variable is in a thread's share.             *)

Lemma dynamic_form_val : forall zp f, dynamic_form_ok zp f = true ->
  fval F init f = zero_elements zp init * zp_width zp.
Proof.
  intros zp f H. unfold dynamic_form_ok in H. rewrite !andb_true_iff, !Z.eqb_eq in H.
  destruct H as [[Hlo Hhi] Hsame]. unfold fval, zero_elements.
  assert (Hterms : tsum (wdn F) init (ft f) = zero_sum init (zp_terms zp) * zp_width zp).
  { revert Hsame. generalize (zp_terms zp) as ts. generalize (ft f) as fs.
    induction fs as [|[[s e] c] fs IH]; intros ts Hsame; destruct ts as [|[s' c'] ts]; try discriminate;
      try (destruct e; discriminate).
    - reflexivity.
    - destruct e; try discriminate. rewrite !andb_true_iff, Pos.eqb_eq, Z.eqb_eq in Hsame.
      destruct Hsame as [[<- ->] Hrest]. cbn [tsum zero_sum atom_val]. rewrite (IH ts Hrest). ring. }
  rewrite Hterms, Hlo. ring.
Qed.

Lemma share_covers : forall x, declared F init x -> exists T', 0 <= T' < N /\ share_at ps init T' x.
Proof.
  intros x (u & f & Hin & Hx). pose proof threads_pos' as HN. destruct dims as (_ & _ & _ & _ & Hw0 & _).
  destruct (sh_covered _ _ _ _ Hsh u f Hin) as [[Hft Hle]|[(v & Hv & Hb & Hft & Hlo)|(zp & Hl & HB & Hfo)]].
  - exfalso. unfold fval in Hx. rewrite Hft in Hx. cbn [tsum] in Hx. lia.
  - exists (((x - init u) / w) mod N). split; [apply Z.mod_pos_bound; exact HN|].
    left. exists v. split; [exact Hv|]. rewrite Hb. unfold fval in Hx. rewrite Hft in Hx. cbn [tsum] in Hx.
    rewrite <- Hlo. split; [lia|reflexivity].
  - exists (((x - init u) / w) mod N). split; [apply Z.mod_pos_bound; exact HN|].
    right. exists zp, u. split; [exact Hl|]. split; [exact HB|].
    destruct (sh_loops _ _ _ _ Hsh zp Hl) as (_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hw & _).
    rewrite (dynamic_form_val zp f Hfo), Hw in Hx. unfold dyn_count. rewrite Hl. split; [exact Hx|reflexivity].
Qed.

End Runs.
