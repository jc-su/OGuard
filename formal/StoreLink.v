(* StoreLink.v: the launch facts hold of what a launch puts in the registers.

   A launch that the catalog accepts runs a verified operator. Each PTX
   parameter holds the argument that `check_launch` gives: a buffer's
   address, the pool's base plus the buffer's offset, or a value. The
   register that the facts name size:k holds size symbol k; the thread
   registers hold the thread's indices, the block size, and the grid size;
   the platform puts the windows' bases, the variables' addresses, and the
   cluster's index in their registers, and each other register holds 0.
   Each register holds its value modulo 2 to its width.

   Proved: `link_sound`, if the gate accepts the operator with its binding
   (`link_ok`, part of `verify_ptx`), then the environment of every thread
   of the launch has the launch facts that `kernel_safe` assumes, provided
   that every buffer the launch takes with an index range holds elements in
   the range (`indices_in`); `region_buffer`, every access that the facts
   allow lies in a buffer argument that the tenant may read for a load and
   may write for a store or an atomic, and a store to an index output writes
   one whole element in the range; so each write that the facts allow keeps
   `indices_in` (`indices_keep`).

   Trusted, the platform's facts (`platform_ok`): the windows' bases and the
   variables' addresses satisfy the bounds and alignments the translation
   states of them, distinct local variables of one frame, or of a frame and
   a frame it calls, do not overlap, and the pool lies apart from both
   windows, below 2^64, at a multiple of 256 bytes. Trusted too, that a run
   does not read the value 0 of a register that a launch value does not
   fill: the gate refuses a kernel that may read a declared register before
   writing it (`defined_check` in GateFront.v), the translation writes each
   register that it adds before reading it, and the cells of local
   variables hold local memory, which the zeroing prologue clears
   (StoreLaunch.v). So the value that the hardware leaves in such a
   register, which may be another tenant's, does not enter a run. *)

From Coq Require Import ZArith.ZArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Lia.
From Coq Require Strings.String.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.
Require Import OpGate.GateDomain.
Require Import OpGate.GateMeaning.
Require Import OpGate.GateSoundAccess.
Require Import OpGate.GateSoundLocal.
Require Import OpGate.GateThread.
Require Import OpGate.GateFront.
Require Import OpGate.GateBarrier.
Require Import OpGate.GateLoop.
Require Import OpGate.GateContext.
Require Import OpGate.GateTerm.
Require Import OpGate.GateEnds.
Require Import OpGate.GateShared.
Require Import OpGate.GateZero.
Require Import OpGate.CatalogStep.
Require Import OpGate.StoreLaunch.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Lists, bytes, and forms.                                            *)

Lemma in_combine_seq : forall (A : Type) (l : list A) s i x,
    In (i, x) (combine (seq s (List.length l)) l) -> (s <= i)%nat /\ nth_error l (i - s) = Some x.
Proof.
  intros A l. induction l as [| y l IH]; intros s i x H; [destruct H |].
  cbn [List.length seq combine] in H. destruct H as [H | H].
  - injection H as <- <-. split; [lia |]. rewrite Nat.sub_diag. reflexivity.
  - destruct (IH (S s) i x H) as [Hs Hn]. split; [lia |].
    replace (i - s)%nat with (S (i - S s)) by lia. exact Hn.
Qed.

Lemma Forall2_nth : forall (A B : Type) (R : A -> B -> Prop) l1 l2 i x,
    Forall2 R l1 l2 -> nth_error l1 i = Some x -> exists y, nth_error l2 i = Some y /\ R x y.
Proof.
  intros A B R l1 l2 i x H. revert i. induction H as [| a b l1 l2 Hab _ IH]; intros i Hi;
    [destruct i; discriminate |].
  destruct i as [| i]; cbn [nth_error] in Hi |- *; [| exact (IH i Hi)].
  injection Hi as <-. exists b. split; [reflexivity | exact Hab].
Qed.

(* A value that occurs once in l has one partner in combine l l'. *)
Lemma combine_once : forall (A : Type) (l : list Z) (l' : list A) x a b,
    In (x, a) (combine l l') -> In (x, b) (combine l l') -> count_occ Z.eq_dec l x = 1%nat -> a = b.
Proof.
  intros A l. induction l as [| y l IH]; intros l' x a b Ha Hb Hc; [destruct Ha |].
  destruct l' as [| y' l']; [destruct Ha |]. cbn [combine] in Ha, Hb.
  revert Hc. cbn [count_occ]. destruct (Z.eq_dec y x) as [-> | Hne]; intros Hc.
  - assert (Hn : ~ In x l) by (apply (count_occ_not_In Z.eq_dec); lia).
    destruct Ha as [Ha | Ha]; destruct Hb as [Hb | Hb].
    + injection Ha as <-. injection Hb as <-. reflexivity.
    + exfalso. apply Hn. exact (in_combine_l _ _ _ _ Hb).
    + exfalso. apply Hn. exact (in_combine_l _ _ _ _ Ha).
    + exfalso. apply Hn. exact (in_combine_l _ _ _ _ Ha).
  - destruct Ha as [Ha | Ha]; [injection Ha as E _; contradiction |].
    destruct Hb as [Hb | Hb]; [injection Hb as E _; contradiction |].
    exact (IH l' x a b Ha Hb Hc).
Qed.

(* A buffer that occurs once among the views of l has one view in l and one
   partner in combine l l'. *)
Lemma combine_once_view : forall (A : Type) (l : list (bid * Z)) (l' : list A) x o1 o2 a b,
    In ((x, o1), a) (combine l l') -> In ((x, o2), b) (combine l l') ->
    count_occ bid_eq_dec (map fst l) x = 1%nat -> o1 = o2 /\ a = b.
Proof.
  intros A l. induction l as [| [y oy] l IH]; intros l' x o1 o2 a b Ha Hb Hc; [destruct Ha |].
  destruct l' as [| y' l']; [destruct Ha |]. cbn [combine] in Ha, Hb.
  revert Hc. cbn [map count_occ fst]. destruct (bid_eq_dec y x) as [-> | Hne]; intros Hc.
  - assert (Hn : ~ In x (map fst l)) by (apply (count_occ_not_In bid_eq_dec); lia).
    destruct Ha as [Ha | Ha]; destruct Hb as [Hb | Hb].
    + inversion Ha. inversion Hb. subst. split; reflexivity.
    + exfalso. apply Hn. exact (in_map fst l (x, o2) (in_combine_l _ _ _ _ Hb)).
    + exfalso. apply Hn. exact (in_map fst l (x, o1) (in_combine_l _ _ _ _ Ha)).
    + exfalso. apply Hn. exact (in_map fst l (x, o1) (in_combine_l _ _ _ _ Ha)).
  - destruct Ha as [Ha | Ha]; [inversion Ha; contradiction |].
    destruct Hb as [Hb | Hb]; [inversion Hb; contradiction |].
    exact (IH l' x o1 o2 a b Ha Hb Hc).
Qed.

Lemma trunc_small : forall n v, 0 <= v < 2 ^ Z.of_nat n -> trunc n v = v.
Proof. intros n v H. unfold trunc. apply Z.mod_small. exact H. Qed.

Lemma fits_trunc : forall l h n v, fits l h n = true -> l <= v <= h -> trunc n v = v.
Proof.
  intros l h n v Hf Hv. unfold fits in Hf. rewrite !andb_true_iff, Z.leb_le, Z.leb_le, Z.ltb_lt in Hf.
  apply trunc_small. lia.
Qed.

Lemma extend_mod : forall n s z, extend n s (z mod 2 ^ Z.of_nat n) = extend n s z.
Proof.
  intros n s z. assert (Hp : 2 ^ Z.of_nat n <> 0) by (apply Z.pow_nonzero; lia).
  destruct s; unfold extend, sext, trunc; rewrite Z.mod_mod by exact Hp; reflexivity.
Qed.

(* The bytes a write puts at its address are the value's, least significant
   first. *)
Lemma word_write : forall n m a z, word_of (write_bytes m a (Z.of_nat n) z) a n = z mod 2 ^ (8 * Z.of_nat n).
Proof.
  induction n as [| n IH]; intros m a z.
  - cbn. rewrite Z.mod_1_r. reflexivity.
  - cbn [word_of]. unfold write_bytes at 1.
    replace ((a <=? a) && (a <? a + Z.of_nat (S n))) with true
      by (symmetry; apply andb_true_iff; split; [apply Z.leb_refl | apply Z.ltb_lt; lia]).
    rewrite Z.sub_diag, Z.mul_0_r, Z.pow_0_r, Z.div_1_r, Z.mod_mod by lia.
    rewrite (word_agree _ (write_bytes m (a + 1) (Z.of_nat n) (z / 256)) (a + 1) n).
    + rewrite IH, pow_bytes. rewrite (Z.rem_mul_r z 256) by (try apply Z.pow_pos_nonneg; lia). reflexivity.
    + intros y Hy. unfold write_bytes.
      replace ((a <=? y) && (y <? a + Z.of_nat (S n))) with true
        by (symmetry; apply andb_true_iff; split; [apply Z.leb_le | apply Z.ltb_lt]; lia).
      replace ((a + 1 <=? y) && (y <? a + 1 + Z.of_nat n)) with true
        by (symmetry; apply andb_true_iff; split; [apply Z.leb_le | apply Z.ltb_lt]; lia).
      rewrite Z.div_div by (try apply Z.pow_nonzero; try apply Z.pow_pos_nonneg; lia).
      replace (8 * (y - a)) with (8 + 8 * (y - (a + 1))) by lia.
      rewrite Z.pow_add_r by lia. reflexivity.
Qed.

Lemma write_outside : forall m x w v y, ~ (x <= y < x + w) -> write_bytes m x w v y = m y.
Proof.
  intros m x w v y H. unfold write_bytes.
  destruct ((x <=? y) && (y <? x + w)) eqn:E; [| reflexivity].
  apply andb_true_iff in E. rewrite Z.leb_le, Z.ltb_lt in E. contradiction.
Qed.

Lemma form_eqb_eq : forall f g, form_eqb f g = true -> f = g.
Proof.
  intros [m ts lo hi] [m' ts' lo' hi'] H. unfold form_eqb in H. cbn [fm ft flo fhi] in H.
  rewrite !andb_true_iff, !Z.eqb_eq, Nat.eqb_eq in H. destruct H as [[[[-> ->] ->] Hl] Hts].
  assert (E : ts = ts').
  { revert ts' Hl Hts. induction ts as [| [x c] ts IH]; intros [| [x' c'] ts'] Hl Hts;
      try discriminate Hl; [reflexivity |].
    cbn [combine forallb fst snd] in Hts. rewrite !andb_true_iff, atom_eqb_eq, Z.eqb_eq in Hts.
    destruct Hts as [[-> ->] Hts]. f_equal. apply IH; [cbn in Hl; lia | exact Hts]. }
  subst. reflexivity.
Qed.

Lemma rep_exact : forall x, exact64 x = true -> rep 64 x = x.
Proof.
  intros x H. unfold exact64 in H. rewrite andb_true_iff, Z.ltb_lt, Z.leb_le in H. unfold rep. rewrite !pow2_eq.
  change (2 ^ (64 - 1)) with (2 ^ 63). change (2 ^ 64) with (2 ^ 63 * 2) in *.
  destruct (Z.le_gt_cases 0 x) as [Hx | Hx].
  - rewrite Z.mod_small by lia. destruct (Z.ltb_spec (2 ^ 63) x); lia.
  - replace (x mod (2 ^ 63 * 2)) with (x + 2 ^ 63 * 2).
    + destruct (Z.ltb_spec (2 ^ 63) (x + 2 ^ 63 * 2)); lia.
    + apply (Z.mod_unique _ _ (-1)); lia.
Qed.

Lemma treduce_exact : forall w rs ts, forallb (fun t => exact64 (snd t)) ts = true ->
    tsum w rs (treduce 64 ts) = tsum w rs ts.
Proof.
  intros w rs ts. induction ts as [| [x c] ts IH]; intros H; [reflexivity |].
  cbn [forallb snd] in H. apply andb_true_iff in H. destruct H as [Hc H].
  cbn [treduce tsum]. rewrite pow2_eq, (rep_exact c Hc).
  destruct (Z.eqb (c mod 2 ^ 64) 0) eqn:E.
  - apply Z.eqb_eq in E. unfold exact64 in Hc. rewrite andb_true_iff, Z.ltb_lt, Z.leb_le in Hc.
    assert (c = 0).
    { destruct (Z.le_gt_cases 0 c).
      - rewrite Z.mod_small in E by lia. exact E.
      - assert (E' : c mod 2 ^ 64 = c + 2 ^ 64) by (symmetry; apply (Z.mod_unique _ _ (-1)); lia). lia. }
    subst c. rewrite IH by exact H. lia.
  - cbn [tsum]. rewrite IH by exact H. reflexivity.
Qed.

Lemma flo_mk : forall ts x, exact64 x = true -> flo (mk 64 ts x x) = x.
Proof.
  intros ts x H. unfold mk. destruct (canon 64 ts); [reflexivity |].
  rewrite Z.eqb_refl. cbn [flo]. exact (rep_exact x H).
Qed.

Lemma fold_sum : forall (A : Type) (g : A -> Z) l acc,
    fold_left (fun a y => a + g y) l acc = acc + fold_left (fun a y => a + g y) l 0.
Proof.
  intros A g l. induction l as [| y l IH]; intros acc; cbn [fold_left]; [lia |].
  rewrite (IH (acc + g y)), (IH (0 + g y)). lia.
Qed.

Lemma dir_eqb_eq : forall a b, dir_eqb a b = true -> a = b.
Proof. intros [] [] H; try discriminate H; reflexivity. Qed.

Lemma allows_writes : forall sp k, allows (bs_dir sp) k = true -> k <> KLoad -> writes sp = true.
Proof. intros sp k. unfold writes. destruct (bs_dir sp), k; cbn; congruence. Qed.

Lemma allows_reads : forall sp, allows (bs_dir sp) KLoad = true -> reads sp = true.
Proof. intros sp. unfold reads. destruct (bs_dir sp); cbn; congruence. Qed.

(* ------------------------------------------------------------------ *)
(* A launch the catalog accepts.                                       *)

Lemma physical_warp_partition : forall bx tx,
 0<bx -> bx mod 32=0 -> 0<=tx<bx ->
 let w:=Z.to_nat(tx/32) in
 (w<Z.to_nat(bx/32))%nat /\ 32*Z.of_nat w<=tx<32*(Z.of_nat w+1).
Proof.
 intros bx tx Hbx Hmultiple Htx. cbv zeta.
 assert(Hq:0<=tx/32)by(apply Z.div_pos;lia).
 assert(Hbq:0<=bx/32)by(apply Z.div_pos;lia).
 rewrite Z2Nat.id by exact Hq.
 pose proof(Z.div_mod tx 32 ltac:(lia))as Hdiv.
 pose proof(Z.mod_pos_bound tx 32 ltac:(lia))as Hmod.
 pose proof(Z.div_mod bx 32 ltac:(lia))as Hbdiv.
 rewrite Hmultiple in Hbdiv.
 split;[apply Z2Nat.inj_lt;lia|lia].
Qed.


Section Link.

Variable F : lfacts.
Variable Mf : binding.
Variable c : catalog.
Variable t : Z.
Variables op gx gy gz : Z.
Variable buffers : list (bid * Z).
Variables sizes scalars : list Z.
Variable guarded : bool.
(* The pool's device address. *)
Variable base : Z.

Hypothesis Hlf : lf_ok F = true.
Hypothesis Hlink : link_ok F Mf = true.
Hypothesis Hinv : inv c.
Hypothesis Hbind : find_binding c op = Some Mf.
Hypothesis Hacc : exists r, check_launch c t op gx gy gz buffers sizes scalars guarded = inl r.
Hypothesis Hbase : 0 <= base /\ base mod alignment = 0 /\ base + c_pool c <= 2 ^ 64.

(* What each PTX parameter holds, and each buffer argument with its span. *)
Definition args : list arg := args_of c sizes (b_params Mf) buffers scalars.
Definition pairs : list ((bid * Z) * bspan) := combine buffers (spans Mf).

Definition grid_at (d : nat) : Z := match d with 0%nat => gx | 1%nat => gy | _ => gz end.

Definition arg_value (a : option arg) : Z :=
  match a with Some (At o) => base + o | Some (Val v) => v | None => 0 end.

(* A thread: its block index and thread index in each dimension, its lane,
   and what the platform puts in its registers. *)
Record thread_regs := {
  tr_ctaid : nat -> Z;
  tr_tid : nat -> Z;
  tr_lane : Z;
  tr_platform : reg -> Z
}.

Definition raw_value (th : thread_regs) (r : reg) : Z :=
  match category_reg F Mf r with
  | Some (CSize k) => nth k sizes 0
  | Some (CParam i _) => arg_value (nth_error args i)
  | Some (CTid d) => tr_tid th d
  | Some (CNtid d) => block_dim Mf d
  | Some (CCtaid d) => tr_ctaid th d
  | Some (CNctaid d) => grid_at d
  | Some CLane => tr_lane th
  | Some CPlatform => tr_platform th r
  | None => 0
  end.

(* The value of register r at the thread's launch: its value modulo 2 to
   its width. *)
Definition launch_value (th : thread_regs) (r : reg) : Z := trunc (wdn F r) (raw_value th r).

Definition thread_in (th : thread_regs) : Prop :=
  (forall d, 0 <= tr_tid th d < block_dim Mf d) /\ (forall d, 0 <= tr_ctaid th d < grid_at d) /\
  0 <= tr_lane th <= 31.

Record platform_ok (init : reg -> Z) : Prop := {
  pf_bounds : forall r lo hi, category_reg F Mf r = Some CPlatform -> is_fixed F r = true ->
              lookup (lf_bounds F) r = Some (lo, hi) -> lo <= init r <= hi;
  pf_aligns : forall r k, category_reg F Mf r = Some CPlatform -> is_fixed F r = true ->
              lookup (lf_aligns F) r = Some k -> (2 ^ Z.max 0 k | init r);
  pf_disjoint : forall u1 s1 f1 u2 s2 f2, In (u1, (s1, f1)) (lf_locals F) -> In (u2, (s2, f2)) (lf_locals F) ->
                u1 <> u2 -> prefix f1 f2 = true -> init u1 + s1 <= init u2 \/ init u2 + s2 <= init u1;
  pf_apart : forall w, w = lf_shared_window F \/ w = lf_local_window F ->
             base + c_pool c <= init w \/ init w + window_size <= base;
  (* the windows of shared and local memory lie apart *)
  pf_windows : init (lf_shared_window F) + window_size <= init (lf_local_window F) \/
               init (lf_local_window F) + window_size <= init (lf_shared_window F)
}.

(* Every element of buffer b, of e bytes read signed or not, lies in
   [lo, hi] in memory m. *)
Definition elems_in (m : memory) (b : buffer) (e : Z) (s : bool) (lo hi : Z) : Prop :=
  forall k, 0 <= k -> k mod e = 0 -> k + e <= bf_bytes b ->
    lo <= extend (Z.to_nat (8 * e)) s (word_of m (base + bf_offset b + k) (Z.to_nat e)) <= hi.

(* Every buffer the launch takes with an index range holds elements in the
   range at the launch's sizes. *)
Definition indices_in (m : memory) : Prop :=
  forall id off sp ir b, In ((id, off), sp) pairs -> bs_index sp = Some ir -> find_buffer c id = Some b ->
    elems_in m b (ir_elem ir) (ir_signed ir) (eval (ir_lo ir) sizes) (eval (ir_hi ir) sizes).

(* ------------------------------------------------------------------ *)
(* What the check of the launch states.                                *)

Lemma accepted :
    (1 <= gx <= 2 ^ 31 - 1 /\ 1 <= gy <= 65535 /\ 1 <= gz <= 65535) /\
    (let '(mx, my, mz) := b_grid Mf in gx <= mx /\ gy <= my /\ gz <= mz) /\
    List.length sizes = List.length (b_sizes Mf) /\
    Forall (fun e => fst (fst e) <= snd e <= snd (fst e)) (combine (b_sizes Mf) sizes) /\
    List.length buffers = List.length (spans Mf) /\
    List.length scalars = List.length (bounds Mf) /\
    Forall (fun e => fst (fst e) <= snd e <= snd (fst e)) (combine (bounds Mf) scalars) /\
    Forall (arg_ok c t guarded buffers (spans Mf) sizes) pairs.
Proof.
  destruct Hacc as [r H]. destruct (check_launch_sound _ _ _ _ _ _ _ _ _ _ _ H) as [b [Hb Hrest]].
  rewrite Hbind in Hb. injection Hb as <-. exact Hrest.
Qed.

Lemma shared_bound : 0 <= eval (b_shared Mf) sizes <= shared_max.
Proof. destruct Hacc as [r H]. exact (check_launch_shared_sound _ _ _ _ _ _ _ _ _ _ _ _ Hbind H). Qed.

Lemma grid_le : forall d, 1 <= grid_at d <= grid_dim Mf d.
Proof.
  destruct accepted as [Hg [Hm _]]. unfold grid_dim. destruct (b_grid Mf) as [[mx my] mz].
  cbn beta iota zeta in Hm. intros [| [| d]]; cbn [grid_at]; lia.
Qed.

Lemma nth_size : forall k, fst (nth k (b_sizes Mf) (0, 0)) <= nth k sizes 0 <= snd (nth k (b_sizes Mf) (0, 0)).
Proof.
  destruct accepted as [_ [_ [Hl [Hs _]]]]. intros k.
  destruct (Nat.lt_ge_cases k (List.length (b_sizes Mf))) as [Hk | Hk].
  - rewrite Forall_forall in Hs.
    assert (Hin : In (nth k (b_sizes Mf) (0, 0), nth k sizes 0) (combine (b_sizes Mf) sizes)).
    { rewrite <- combine_nth by (symmetry; exact Hl). apply nth_In.
      rewrite combine_length. lia. }
    exact (Hs _ Hin).
  - rewrite !nth_overflow by lia. cbn. lia.
Qed.

Lemma pair_ok : forall id off sp, In ((id, off), sp) pairs -> arg_ok c t guarded buffers (spans Mf) sizes ((id, off), sp).
Proof.
  destruct accepted as [_ [_ [_ [_ [_ [_ [_ Hok]]]]]]]. intros id off sp H.
  rewrite Forall_forall in Hok. exact (Hok _ H).
Qed.

Lemma buffer_placed : forall id b, find_buffer c id = Some b ->
    0 <= bf_offset b /\ 1 <= bf_bytes b /\ bf_offset b + bf_bytes b <= c_pool c /\ bf_offset b mod alignment = 0.
Proof.
  intros id b H. destruct (find_buffer_in _ _ _ H) as [Hb _]. destruct Hinv as [Hp _]. exact (Hp b Hb).
Qed.

(* ------------------------------------------------------------------ *)
(* What each parameter holds.                                          *)

Definition span_of (p : String.string * holding) : list bspan :=
  match snd p with PBase sp => [sp] | _ => [] end.

Definition bound_of (p : String.string * holding) : list (Z * Z) :=
  match snd p with PBounded lo hi => [(lo, hi)] | _ => [] end.

Definition arg_fits (h : holding) (a : arg) : Prop :=
  match h with
  | PBase sp => exists id off b, In ((id, off), sp) pairs /\ find_buffer c id = Some b /\ a = At (bf_offset b + off)
  | PSize x => a = Val (eval x sizes)
  | PConst z => a = Val z
  | PBounded lo hi => exists v, a = Val v /\ lo <= v <= hi
  | PZero => a = Val 0
  end.

Lemma args_fit_from : forall ps bs ss,
    incl (combine bs (flat_map span_of ps)) pairs ->
    List.length bs = List.length (flat_map span_of ps) ->
    List.length ss = List.length (flat_map bound_of ps) ->
    Forall (arg_ok c t guarded buffers (spans Mf) sizes) (combine bs (flat_map span_of ps)) ->
    Forall (fun e => fst (fst e) <= snd e <= snd (fst e)) (combine (flat_map bound_of ps) ss) ->
    Forall2 (fun p a => arg_fits (snd p) a) ps (args_of c sizes ps bs ss).
Proof.
  induction ps as [| [name h] ps IH]; intros bs ss Hincl Hlb Hls Hok Hbd; [constructor |].
  destruct h as [sp | x | z | lo hi |]; cbn [flat_map span_of bound_of snd app] in *.
  - destruct bs as [| [id off] bs]; [discriminate Hlb |]. cbn [combine] in Hincl, Hok.
    inversion Hok as [| ? ? Hid Hrest]; subst. unfold arg_ok in Hid. destruct Hid as [b [Hb _]].
    cbn [args_of]. rewrite Hb. constructor.
    + exists id, off, b. split; [apply Hincl; left; reflexivity | split; [exact Hb | reflexivity]].
    + apply IH; [intros y Hy; apply Hincl; right; exact Hy | cbn in Hlb; lia | exact Hls | exact Hrest | exact Hbd].
  - cbn [args_of]. constructor; [reflexivity | apply IH; assumption].
  - cbn [args_of]. constructor; [reflexivity | apply IH; assumption].
  - destruct ss as [| v ss]; [discriminate Hls |]. cbn [combine] in Hbd.
    inversion Hbd as [| ? ? Hv Hrest]; subst. cbn [args_of]. constructor.
    + exists v. split; [reflexivity | exact Hv].
    + apply IH; [exact Hincl | exact Hlb | cbn in Hls; lia | exact Hok | exact Hrest].
  - cbn [args_of]. constructor; [reflexivity | apply IH; assumption].
Qed.

Lemma args_fit : Forall2 (fun p a => arg_fits (snd p) a) (b_params Mf) args.
Proof.
  destruct accepted as [_ [_ [_ [_ [Hlb [Hls [Hbd Hok]]]]]]].
  apply args_fit_from; [intros y Hy; exact Hy | exact Hlb | exact Hls | exact Hok | exact Hbd].
Qed.

Ltac cases := repeat match goal with |- context [match ?x with _ => _ end] => destruct x end.

Lemma category_param : forall r i h, category_reg F Mf r = Some (CParam i h) ->
    exists name, nth_error (b_params Mf) i = Some (name, h).
Proof.
  intros r i h. unfold category_reg. destruct (name_of F r) as [n |]; [| discriminate].
  unfold category_of. destruct (find _ (seq 0 _)); [discriminate |].
  destruct (find _ (combine _ _)) as [[i' [name h']] |] eqn:E.
  - intros H. injection H as <- <-. apply find_some in E. destruct E as [Hin _].
    apply in_combine_seq in Hin. exists name. rewrite Nat.sub_0_r in Hin. exact (proj2 Hin).
  - cases; discriminate.
Qed.

Lemma param_arg : forall r i h, category_reg F Mf r = Some (CParam i h) ->
    exists a, nth_error args i = Some a /\ arg_fits h a.
Proof.
  intros r i h H. destruct (category_param r i h H) as [name Hn].
  exact (Forall2_nth _ _ _ _ _ i _ args_fit Hn).
Qed.

(* The values the launch puts in a register lie in its category's range. *)
Lemma raw_range : forall th r cat l h, thread_in th -> category_reg F Mf r = Some cat ->
    value_range Mf cat = Some (l, h) -> l <= raw_value th r <= h.
Proof.
  intros th r cat l h [Htid [Hcta Hlane]] Hc Hv. unfold raw_value. rewrite Hc.
  destruct cat as [k | i hd | d | d | d | d | |]; cbn [value_range] in Hv;
    try (injection Hv as <- <-).
  - pose proof (nth_size k) as Hsz. injection Hv as Hpair. rewrite Hpair in Hsz. exact Hsz.
  - destruct (param_arg r i hd Hc) as [a [Ha Hfit]]. rewrite Ha.
    destruct hd as [sp | x | z | lo hi |]; cbn [value_range arg_fits] in Hv, Hfit; try discriminate Hv;
      injection Hv as <- <-.
    + subst a. cbn. lia.
    + destruct Hfit as [v [-> Hvr]]. cbn. lia.
    + subst a. cbn. lia.
  - specialize (Htid d). lia.
  - lia.
  - specialize (Hcta d). pose proof (grid_le d). lia.
  - pose proof (grid_le d). lia.
  - lia.
  - discriminate Hv.
Qed.

(* A thread index register holds the thread's index: its declared bounds
   fit its width. *)
Lemma tid_value : forall th r d lo hi, thread_in th -> category_reg F Mf r = Some (CTid d) -> is_fixed F r = true ->
    lookup (lf_bounds F) r = Some (lo, hi) -> launch_value th r = tr_tid th d.
Proof.
  intros th r d lo hi Hth Hc Hfix Hl. pose proof (lookup_in _ _ _ Hl) as Hin.
  assert (Hb : bound_ok F Mf (r, (lo, hi)) = true).
  { unfold link_ok in Hlink. rewrite !andb_true_iff in Hlink. destruct Hlink as [[[Hb _] _] _].
    rewrite forallb_forall in Hb. exact (Hb _ Hin). }
  unfold bound_ok in Hb. rewrite Hfix, Hc in Hb. cbn [negb orb value_range] in Hb.
  rewrite !andb_true_iff in Hb. destruct Hb as [[Hf _] _].
  unfold launch_value, raw_value. rewrite Hc. apply (fits_trunc _ _ _ _ Hf).
  destruct Hth as [Htid _]. pose proof (Htid d). lia.
Qed.

(* The register of a buffer parameter holds the buffer's address. *)
Lemma role_buffer : forall th P sp, role_of F P = Some sp ->
    exists bsp id off b, In ((id, off), bsp) pairs /\ find_buffer c id = Some b /\
      arg_ok c t guarded buffers (spans Mf) sizes ((id, off), bsp) /\
      launch_value th P = base + bf_offset b + off /\
      sp_dir sp = bs_dir bsp /\ form_is F Mf (sp_ext sp) (bs_ext bsp) = true /\ sp_flag sp = bs_flag bsp /\
      match sp_index sp, bs_index bsp with
      | None, None => True
      | Some ix, Some ir => ix_elem ix = ir_elem ir /\ ix_signed ix = ir_signed ir /\
                            form_is F Mf (ix_lo ix) (ir_lo ir) = true /\ form_is F Mf (ix_hi ix) (ir_hi ir) = true
      | _, _ => False
      end.
Proof.
  intros th P sp H. apply lookup_in in H. unfold link_ok in Hlink. rewrite !andb_true_iff in Hlink.
  destruct Hlink as [[_ Hro] _]. rewrite forallb_forall in Hro. specialize (Hro _ H). unfold role_ok in Hro.
  destruct (category_reg F Mf P) as [cat |] eqn:Ec; [| discriminate Hro].
  destruct cat as [| i [bsp | | | |] | | | | | |]; try discriminate Hro.
  rewrite !andb_true_iff in Hro. destruct Hro as [[[[Hw Hdir] Hext] Hflag] Hix].
  destruct (param_arg P i (PBase bsp) Ec) as [a [Ha [id [off [b [Hin [Hb ->]]]]]]].
  exists bsp, id, off, b. split; [exact Hin |]. split; [exact Hb |]. split; [exact (pair_ok id off bsp Hin) |].
  split.
  - unfold launch_value, raw_value. rewrite Ec, Ha. cbn [arg_value]. apply Nat.eqb_eq in Hw. rewrite Hw.
    destruct (buffer_placed id b Hb) as [Ho [Hn [Hop _]]].
    pose proof (pair_ok id off bsp Hin) as Hok. destruct Hok as [b' [Hb' [Hoff [_ [_ [Hbytes _]]]]]].
    rewrite Hb in Hb'. injection Hb' as <-. rewrite trunc_small; [lia | cbn; lia].
  - split; [exact (dir_eqb_eq _ _ Hdir) |]. split; [exact Hext |].
    split; [exact (Bool.eqb_prop _ _ Hflag) |].
    destruct (sp_index sp) as [ix |], (bs_index bsp) as [ir |]; try discriminate Hix; [| exact I].
    rewrite !andb_true_iff, Z.eqb_eq in Hix. destruct Hix as [[[He Hs] Hlo] Hhi].
    apply Bool.eqb_prop in Hs. exact (conj He (conj Hs (conj Hlo Hhi))).
Qed.

(* ------------------------------------------------------------------ *)
(* Forms over the size registers.                                      *)

Lemma size_terms_sum : forall th l ts acc, size_terms F Mf l = Some ts ->
    acc + tsum (wdn F) (launch_value th) ts =
    fold_left (fun a y => a + snd y * nth (Z.to_nat (fst y)) sizes 0) l acc.
Proof.
  intros th l. induction l as [| [k x] l IH]; intros ts acc H.
  - injection H as <-. cbn. lia.
  - cbn [size_terms] in H.
    destruct (reg_named F _) as [r |]; [| discriminate H].
    destruct (size_terms F Mf l) as [ts' |] eqn:El; [| discriminate H].
    destruct (category_reg F Mf r) as [[j | | | | | | |] |] eqn:Ec; try discriminate H.
    destruct (Z.leb 0 k && Nat.eqb j (Z.to_nat k) && fits (fst (nth j (b_sizes Mf) (0, 0))) (snd (nth j (b_sizes Mf) (0, 0))) (wdn F r)) eqn:Ecd;
      [| discriminate H].
    injection H as <-. rewrite !andb_true_iff, Nat.eqb_eq in Ecd. destruct Ecd as [[_ ->] Hf].
    cbn [tsum atom_val fold_left fst snd].
    assert (Hv : launch_value th r = nth (Z.to_nat k) sizes 0).
    { unfold launch_value, raw_value. rewrite Ec. apply (fits_trunc _ _ _ _ Hf). pose proof (nth_size (Z.to_nat k)). lia. }
    rewrite Hv, <- (IH ts' _ eq_refl). lia.
Qed.

(* The loop's size terms of the zeroing prologue, each coefficient divided
   by the width, sum to the affine's terms at the launch's sizes. *)
Lemma zero_terms_sum : forall th w l terms ts acc, 0 < w ->
    option_all (map (fun x => match reg_named F (size_name (fst x)) with
                              | Some s => Some (s, snd x / w) | None => None end) l) = Some terms ->
    size_terms F Mf l = Some ts -> (forall k c, In (k, c) l -> c mod w = 0) ->
    acc + zero_sum (launch_value th) terms * w = fold_left (fun a y => a + snd y * nth (Z.to_nat (fst y)) sizes 0) l acc.
Proof.
  intros th w l. induction l as [| [k x] l IH]; intros terms ts acc Hw Hz Hs Hdiv.
  - injection Hz as <-. cbn. lia.
  - cbn [map option_all size_terms fst snd] in Hz, Hs.
    destruct (reg_named F (size_name k)) as [r |] eqn:Er; [| discriminate Hz].
    unfold size_name in Er. rewrite Er in Hs.
    destruct (option_all (map _ l)) as [terms' |] eqn:El; [| discriminate Hz]. injection Hz as <-.
    destruct (size_terms F Mf l) as [ts' |] eqn:Es; [| discriminate Hs].
    destruct (category_reg F Mf r) as [[j | | | | | | |] |] eqn:Ec; try discriminate Hs.
    destruct (Z.leb 0 k && Nat.eqb j (Z.to_nat k) &&
              fits (fst (nth j (b_sizes Mf) (0, 0))) (snd (nth j (b_sizes Mf) (0, 0))) (wdn F r)) eqn:Ecd;
      [| discriminate Hs].
    injection Hs as <-. rewrite !andb_true_iff, Nat.eqb_eq in Ecd. destruct Ecd as [[_ ->] Hf].
    cbn [zero_sum fold_left fst snd].
    assert (Hv : launch_value th r = nth (Z.to_nat k) sizes 0).
    { unfold launch_value, raw_value. rewrite Ec. apply (fits_trunc _ _ _ _ Hf). pose proof (nth_size (Z.to_nat k)). lia. }
    rewrite Hv, <- (IH terms' ts' _ Hw ltac:(first [exact El | reflexivity]) ltac:(first [exact Es | reflexivity])
                      (fun k' c' H => Hdiv k' c' (or_intror H))).
    assert (Hx : x = w * (x / w)) by (apply Z.div_exact; [lia | exact (Hdiv k x (or_introl eq_refl))]).
    nia.
Qed.

(* A form that the check finds to be affine a has the affine's value at the
   launch's sizes. *)
Lemma form_is_eval : forall th f a, form_is F Mf f a = true ->
    tsum (wdn F) (launch_value th) (ft f) + flo f = eval a sizes.
Proof.
  intros th f a H. unfold form_is in H.
  destruct (size_terms F Mf (af_terms a)) as [ts |] eqn:Ets; [| discriminate H].
  rewrite !andb_true_iff in H. destruct H as [[Hex Hc] Heq]. apply form_eqb_eq in Heq. subst f.
  rewrite mk_ft. unfold canon. rewrite (treduce_exact _ _ _ Hex), tsum_tsort, (flo_mk _ _ Hc).
  unfold eval. rewrite <- (size_terms_sum th _ _ 0 Ets). lia.
Qed.

(* ------------------------------------------------------------------ *)
(* The launch facts.                                                   *)

Lemma role_span : forall P sp, role_of F P = Some sp -> span_ok F sp = true.
Proof.
  intros P sp H. apply lookup_in in H. unfold lf_ok in Hlf. rewrite !andb_true_iff in Hlf.
  destruct Hlf as [[[[[_ Hro] _] _] _] _]. rewrite forallb_forall in Hro. specialize (Hro _ H).
  apply andb_true_iff in Hro. exact (proj2 Hro).
Qed.

Lemma span_elem : forall sp ix, span_ok F sp = true -> sp_index sp = Some ix -> 1 <= ix_elem ix <= 8.
Proof.
  intros sp ix H Hix. unfold span_ok in H. rewrite Hix in H. rewrite !andb_true_iff in H.
  destruct H as [_ [[[H _] _] _]]. rewrite !orb_true_iff, !Z.eqb_eq in H. lia.
Qed.

Lemma pool_outside : forall init x w, platform_ok init -> base <= x < base + c_pool c ->
    w = lf_shared_window F \/ w = lf_local_window F -> in_space init w x = false.
Proof.
  intros init x w Hpf Hx Hw. unfold in_space. apply andb_false_iff.
  destruct (pf_apart _ Hpf w Hw); [left; apply Z.leb_gt | right; apply Z.ltb_ge]; lia.
Qed.

Lemma derived_affine_eval : forall d,
 eval (derived_affine d) sizes =
 eval (ds_numerator d) sizes - ds_divisor d * nth (Z.to_nat (ds_target d)) sizes 0.
Proof.
 intros d. unfold eval, derived_affine. cbn [af_const af_terms].
 rewrite fold_left_app. cbn [fold_left fst snd]. ring.
Qed.

Lemma derived_launch_sound : forall th k, derived_ok F Mf k = true ->
 cons_ok F (launch_value th) k.
Proof.
 intros th [[ts lo] hi] H. unfold derived_ok in H.
 apply existsb_exists in H. destruct H as [d [Hin H]].
 rewrite !andb_true_iff in H. destruct H as [[[Hd Hl] Hh] Hf].
 apply Z.eqb_eq in Hl, Hh. subst lo hi.
 pose proof (form_is_eval th _ _ Hf) as Hform. cbn [ft flo] in Hform.
 rewrite derived_affine_eval in Hform.
 assert (Hr : derived_holds sizes d).
 { destruct Hacc as [r Haccepted].
   pose proof (check_launch_derived_sound _ _ _ _ _ _ _ _ _ _ _ _ Hbind Haccepted) as Hall.
   rewrite Forall_forall in Hall. exact (Hall _ Hin). }
 destruct Hr as [Hpos Hquot].
 pose proof (Z.div_mod (eval (ds_numerator d) sizes) (ds_divisor d) ltac:(lia)) as Hdiv.
 pose proof (Z.mod_pos_bound (eval (ds_numerator d) sizes) (ds_divisor d) Hpos) as Hmod.
 unfold cons_ok. rewrite Hquot in Hform. lia.
Qed.

Theorem link_sound : forall th o hist sh lh, thread_in th -> platform_ok (launch_value th) ->
    (forall t', indices_in (hist t')) -> launch_ok F (env_of F (launch_value th) o hist sh lh).
Proof.
  intros th o hist sh lh Hth Hpf Hidx. constructor; cbn [env_of e_init].
  - (* every register holds a value of its width *)
    intros r. unfold wdz, wd, launch_value, trunc. apply Z.mod_pos_bound, Z.pow_pos_nonneg; lia.
  - (* bounds *)
    intros r lo hi Hfix Hl. pose proof (lookup_in _ _ _ Hl) as Hin.
    unfold link_ok in Hlink. rewrite !andb_true_iff in Hlink. destruct Hlink as [[[Hb _] _] _].
    rewrite forallb_forall in Hb. specialize (Hb _ Hin). unfold bound_ok in Hb. rewrite Hfix in Hb.
    cbn [negb orb] in Hb. destruct (category_reg F Mf r) as [cat |] eqn:Ec; [| discriminate Hb].
    assert (Hc : cat = CPlatform \/
                 match value_range Mf cat with
                 | Some (l, h) => fits l h (wdn F r) && Z.leb lo l && Z.leb h hi
                 | None => false
                 end = true) by (destruct cat; first [left; reflexivity | right; exact Hb]).
    destruct Hc as [-> | Hc]; [exact (pf_bounds _ Hpf r lo hi Ec Hfix Hl) |].
    destruct (value_range Mf cat) as [[l h] |] eqn:Ev; [| discriminate Hc].
    rewrite !andb_true_iff, !Z.leb_le in Hc. destruct Hc as [[Hf Hlo] Hhi].
    pose proof (raw_range th r cat l h Hth Ec Ev) as Hr.
    unfold launch_value. rewrite (fits_trunc _ _ _ _ Hf Hr). lia.
  - (* alignments *)
    intros r k Hfix Hl. pose proof (lookup_in _ _ _ Hl) as Hin.
    unfold link_ok in Hlink. rewrite !andb_true_iff in Hlink. destruct Hlink as [[[_ Ha] _] _].
    rewrite forallb_forall in Ha. specialize (Ha _ Hin). unfold align_ok in Ha. rewrite Hfix in Ha.
    cbn [negb orb] in Ha. destruct (Z.leb_spec k 0) as [Hk | Hk].
    { rewrite Z.max_l by exact Hk. apply Z.divide_1_l. }
    rewrite Z.max_r by lia. cbn [orb] in Ha.
    destruct (category_reg F Mf r) as [cat |] eqn:Ec; [| discriminate Ha].
    destruct cat as [| i [sp | x | z | lo hi |] | | | | | | ]; try discriminate Ha.
    + (* a buffer's base *)
      rewrite andb_true_iff, Z.leb_le, Nat.eqb_eq in Ha. destruct Ha as [H8 Hw].
      destruct (param_arg r i (PBase sp) Ec) as [a [Harg [id [off [b [Hpair [Hb ->]]]]]]].
      destruct (buffer_placed id b Hb) as [Ho [Hn [Hop Hoa]]]. destruct Hbase as [Hb0 [Hba Hbt]].
      pose proof (pair_ok id off sp Hpair) as Hok. destruct Hok as [b' [Hb' [Hoff [Hova [_ [Hbytes _]]]]]].
      rewrite Hb in Hb'. injection Hb' as <-.
      unfold launch_value, raw_value. rewrite Ec, Harg. cbn [arg_value]. rewrite Hw.
      rewrite trunc_small by (cbn; lia).
      assert (H256 : (2 ^ k | 256)).
      { exists (2 ^ (8 - k)). rewrite <- Z.pow_add_r by lia. replace (8 - k + k) with 8 by lia. reflexivity. }
      unfold alignment in Hba, Hoa, Hova. rewrite Z.add_assoc. apply Z.divide_add_r; [apply Z.divide_add_r |].
      * apply (Z.divide_trans _ 256); [exact H256 | apply Z.mod_divide; [lia | exact Hba]].
      * apply (Z.divide_trans _ 256); [exact H256 | apply Z.mod_divide; [lia | exact Hoa]].
      * apply (Z.divide_trans _ 256); [exact H256 | apply Z.mod_divide; [lia | exact Hova]].
    + (* a constant *)
      rewrite andb_true_iff, Z.eqb_eq in Ha. destruct Ha as [Hf Hm].
      destruct (param_arg r i (PConst z) Ec) as [a [Harg ->]].
      unfold launch_value, raw_value. rewrite Ec, Harg. cbn [arg_value].
      rewrite (fits_trunc z z _ z Hf) by lia. apply Z.mod_divide; [apply Z.pow_nonzero; lia | exact Hm].
    + (* zero *)
      destruct (param_arg r i PZero Ec) as [a [Harg ->]].
      unfold launch_value, raw_value. rewrite Ec, Harg. cbn [arg_value]. unfold trunc. rewrite Z.mod_0_l.
      * apply Z.divide_0_r.
      * apply Z.pow_nonzero; lia.
    + (* the platform's *)
      rewrite <- (Z.max_r 0 k) by lia. exact (pf_aligns _ Hpf r k Ec Hfix Hl).
  - (* index inputs *)
    intros P sp ix t' e Hrole Hix He0 Hmod Hle.
    destruct (role_buffer th P sp Hrole) as [bsp [id [off [b [Hin [Hb [Hok [Hval [_ [Hext [_ Hidx']]]]]]]]]]].
    rewrite Hix in Hidx'. destruct (bs_index bsp) as [ir |] eqn:Eir; [| contradiction].
    destruct Hidx' as [Hel [Hsg [Hlo Hhi]]].
    pose proof (span_elem sp ix (role_span P sp Hrole) Hix) as Helem.
    unfold bval in *. cbn [env_of e_init] in *. rewrite (form_is_eval th _ _ Hext) in Hle.
    rewrite (form_is_eval th _ _ Hlo), (form_is_eval th _ _ Hhi).
    unfold arg_ok in Hok. destruct Hok as [b' [Hb' [Hoff [_ [Hzero [Hbytes _]]]]]].
    rewrite Hb in Hb'. injection Hb' as <-.
    assert (Hoff0 : off = 0) by (apply Hzero; rewrite Eir; discriminate). subst off.
    rewrite Z.add_0_r in Hval.
    destruct (buffer_placed id b Hb) as [Ho [Hn [Hop _]]]. destruct Hbase as [Hb0 [_ Hbt]].
    unfold ix_read. cbn [env_of e_mem e_init]. rewrite Hval.
    rewrite (trunc_small 64) by (cbn; lia).
    rewrite (pool_outside _ (base + bf_offset b + e) _ Hpf ltac:(lia) (or_introl eq_refl)),
            (pool_outside _ (base + bf_offset b + e) _ Hpf ltac:(lia) (or_intror eq_refl)).
    assert (Hr : trunc (Z.to_nat (8 * ix_elem ix)) (read_word (hist t') (base + bf_offset b + e)) =
                 word_of (hist t') (base + bf_offset b + e) (Z.to_nat (ix_elem ix))).
    { rewrite <- read_word_low by lia. unfold trunc. f_equal. f_equal. lia. }
    rewrite Hr, Hel, Hsg. rewrite Hel in Hmod, Hle.
    apply (Hidx t' id 0 bsp ir b Hin Eir Hb e He0 Hmod). lia.
  - (* local variables *)
    exact (pf_disjoint _ Hpf).
  - (* no buffer lies in a window *)
    intros P sp w Hrole Hw.
    destruct (role_buffer th P sp Hrole) as [bsp [id [off [b [_ [Hb [Hok [Hval [_ [Hext _]]]]]]]]]].
    unfold bval. cbn [env_of e_init]. rewrite (form_is_eval th _ _ Hext), Hval.
    unfold arg_ok in Hok. destruct Hok as [b' [Hb' [Hoff [_ [_ [Hbytes _]]]]]].
    rewrite Hb in Hb'. injection Hb' as <-.
    destruct (buffer_placed id b Hb) as [Ho [Hn [Hop _]]]. destruct Hbase as [Hb0 [_ Hbt]].
    destruct (pf_apart _ Hpf w Hw); [left | right]; lia.
  - (* exact derived-size relations checked by the launch *)
    apply Forall_forall. intros k Hin.
    unfold link_ok in Hlink. apply andb_true_iff in Hlink. destruct Hlink as [_ Hd].
    rewrite forallb_forall in Hd. exact (derived_launch_sound th k (Hd k Hin)).
Qed.

(* ------------------------------------------------------------------ *)
(* The accesses of a launch.                                           *)

(* Each access the facts allow lies in a buffer argument, which the span
   allows the access's kind; a store to an index output writes one whole
   element in the range. *)
Lemma region_buffer : forall th o hist sh lh tt k x w v,
    platform_ok (launch_value th) ->
    GateMeaning.region F (env_of F (launch_value th) o hist sh lh) tt k x w v -> 1 <= w ->
    exists id off sp b e, In ((id, off), sp) pairs /\ find_buffer c id = Some b /\
      arg_ok c t guarded buffers (spans Mf) sizes ((id, off), sp) /\
      x = base + bf_offset b + e /\ 0 <= e /\ e + w <= bf_bytes b /\ allows (bs_dir sp) k = true /\
      (forall ir, bs_index sp = Some ir -> k <> KLoad ->
         k = KStore /\ w = ir_elem ir /\ e mod ir_elem ir = 0 /\
         exists z, v = Some z /\
           eval (ir_lo ir) sizes <= extend (Z.to_nat (8 * ir_elem ir)) (ir_signed ir) z <= eval (ir_hi ir) sizes) /\
      (bs_flag sp = true -> k = KStore /\ w = 4 /\ e mod 4 = 0 /\ v = Some 1).
Proof.
  intros th o hist sh lh tt k x w v Hpf H Hw.
  destruct H as [k P sp e w v Hrole Hallow He Hle _ Hidx Hfl]. cbn [env_of e_init].
  destruct (role_buffer th P sp Hrole) as [bsp [id [off [b [Hin [Hb [Hok [Hval [Hdir [Hext [Hflag Hix]]]]]]]]]]].
  pose proof Hok as Hok'. unfold arg_ok in Hok'. destruct Hok' as [b' [Hb' [Hoff [Hoffa [Hzero [Hbytes _]]]]]].
  rewrite Hb in Hb'. injection Hb' as <-.
  unfold bval in Hle. cbn [env_of e_init] in Hle. rewrite (form_is_eval th _ _ Hext) in Hle.
  destruct (buffer_placed id b Hb) as [Ho [Hn [Hop _]]]. destruct Hbase as [Hb0 [_ Hbt]].
  exists id, off, bsp, b, (off + e). split; [exact Hin |]. split; [exact Hb |]. split; [exact Hok |].
  split; [rewrite Hval; rewrite trunc_small; [lia | cbn; lia] |]. split; [lia |]. split; [lia |].
  split; [rewrite <- Hdir; exact Hallow |].
  split.
  2: { (* a write into a flag buffer stores 1 into one whole word *)
       intros Hf. rewrite <- Hflag in Hf. destruct (Hfl Hf) as [Hk [Hw4 [Hm4 Hv]]].
       split; [exact Hk |]. split; [exact Hw4 |]. split; [| exact Hv].
       unfold alignment in Hoffa. apply Z.mod_divide in Hoffa; [| lia]. apply Z.mod_divide in Hm4; [| lia].
       apply Z.mod_divide; [lia |]. destruct Hoffa as [q1 Hq1]. destruct Hm4 as [q2 Hq2].
       exists (64 * q1 + q2). lia. }
  intros ir Hir Hk. assert (Hoff0 : off = 0) by (apply Hzero; rewrite Hir; discriminate).
  rewrite Hoff0, Z.add_0_l.
  rewrite Hir in Hix. destruct (sp_index sp) as [ix |]; [| contradiction].
  destruct Hix as [Hel [Hsg [Hlo Hhi]]].
  destruct (Hidx ix eq_refl Hk) as [Hst [Hwe [Hm [z [Hz Hr]]]]].
  unfold bval in Hr. cbn [env_of e_init] in Hr. rewrite (form_is_eval th _ _ Hlo), (form_is_eval th _ _ Hhi) in Hr.
  rewrite <- Hel, <- Hsg. split; [exact Hst |]. split; [exact Hwe |]. split; [exact Hm |].
  exists z. split; [exact Hz | exact Hr].
Qed.

(* A write that the facts allow keeps every index buffer in its range. *)
Lemma indices_keep : forall th o hist sh lh tt k x w v wr m,
    platform_ok (launch_value th) ->
    GateMeaning.region F (env_of F (launch_value th) o hist sh lh) tt k x w v -> k <> KLoad -> indices_in m ->
    indices_in (write_bytes m x w (match v with Some z => z | None => wr end)).
Proof.
  intros th o hist sh lh tt k x w v wr m Hpf H Hk Hm id off sp ir b Hin Hix Hb kk Hk0 Hkm Hkle.
  specialize (Hm id off sp ir b Hin Hix Hb). set (val := match v with Some z => z | None => wr end).
  set (a := base + bf_offset b + kk).
  assert (Hsame : (forall y, a <= y < a + ir_elem ir -> ~ (x <= y < x + w)) ->
                  word_of (write_bytes m x w val) a (Z.to_nat (ir_elem ir)) = word_of m a (Z.to_nat (ir_elem ir))).
  { intros Hout. apply word_agree. intros y Hy. apply write_outside. apply Hout.
    rewrite Z2Nat.id in Hy by lia. exact Hy. }
  destruct (Z_lt_le_dec w 1) as [Hw0 | Hw].
  - rewrite Hsame by lia. exact (Hm kk Hk0 Hkm Hkle).
  - destruct (region_buffer th o hist sh lh tt k x w v Hpf H Hw)
      as [id' [off' [sp' [b' [e' [Hin' [Hb' [Hok' [-> [He' [Hle' [Hallow [Hidx' _]]]]]]]]]]]]].
    destruct (bid_eq_dec id id') as [<- | Hne].
    + (* the write is to this buffer: an index output, written one element at a time *)
      rewrite Hb in Hb'. injection Hb' as <-.
      assert (Hw1 : writes sp' = true) by exact (allows_writes sp' k Hallow Hk).
      unfold arg_ok in Hok'. destruct Hok' as [b' [_ [_ [_ [_ [_ [_ [Hwr _]]]]]]]].
      destruct (Hwr Hw1) as [_ [Hc | Hc]];
        [| exfalso; rewrite (proj1 (Hc off sp Hin)) in Hix; discriminate Hix].
      destruct (combine_once_view _ _ _ _ _ _ _ _ Hin Hin' Hc) as [<- <-].
      destruct (Hidx' ir Hix Hk) as [_ [-> [Hem [z [-> Hz]]]]]. cbn in val. subst val.
      destruct (Z.eq_dec kk e') as [-> | Hne'].
      * subst a.
        pose proof (word_write (Z.to_nat (ir_elem ir)) m (base + bf_offset b + e') z) as Hww.
        rewrite Z2Nat.id in Hww by lia. rewrite Hww.
        replace (2 ^ (8 * ir_elem ir)) with (2 ^ Z.of_nat (Z.to_nat (8 * ir_elem ir))) by (rewrite Z2Nat.id; lia).
        rewrite extend_mod. exact Hz.
      * rewrite Hsame; [exact (Hm kk Hk0 Hkm Hkle) |].
        intros y Hy Hy'. subst a.
        apply Z.mod_divide in Hkm; [| lia]. apply Z.mod_divide in Hem; [| lia].
        destruct Hkm as [q1 Hq1]. destruct Hem as [q2 Hq2]. subst kk e'.
        assert (q1 <> q2) by (intros ->; contradiction).
        nia.
    + (* the write is to another buffer, apart from this one *)
      rewrite Hsame; [exact (Hm kk Hk0 Hkm Hkle) |].
      destruct (find_buffer_in _ _ _ Hb) as [Hbin Hbid]. destruct (find_buffer_in _ _ _ Hb') as [Hbin' Hbid'].
      destruct Hinv as [_ [_ [Hap _]]].
      assert (Hapart : apart b b') by (apply Hap; [exact Hbin | exact Hbin' | congruence]).
      intros y Hy Hy'. subst a. unfold apart in Hapart. lia.
Qed.

(* A write that the facts allow touches no byte of a word of 4 bytes of a
   flag buffer, or sets the word to 1: a flag buffer that a launch writes
   is passed once, and a write through its flag span stores 1 into one
   whole word. *)
Lemma flags_set : forall th o hist sh lh tt k x w v wr m id off sp b kk,
    platform_ok (launch_value th) ->
    GateMeaning.region F (env_of F (launch_value th) o hist sh lh) tt k x w v -> k <> KLoad ->
    In ((id, off), sp) pairs -> bs_flag sp = true -> find_buffer c id = Some b ->
    0 <= kk -> kk mod 4 = 0 -> kk + 4 <= bf_bytes b ->
    (forall y, base + bf_offset b + kk <= y < base + bf_offset b + kk + 4 -> ~ (x <= y < x + w)) \/
    word_of (write_bytes m x w (match v with Some z => z | None => wr end)) (base + bf_offset b + kk) 4 = 1.
Proof.
  intros th o hist sh lh tt k x w v wr m id off sp b kk Hpf H Hk Hin Hflag Hb Hk0 Hkm Hkle.
  destruct (Z_lt_le_dec w 1) as [Hw0 | Hw]; [left; intros y _ Hy; lia |].
  destruct (region_buffer th o hist sh lh tt k x w v Hpf H Hw)
    as [id' [off' [sp' [b' [e' [Hin' [Hb' [Hok' [-> [He' [Hle' [Hallow [_ Hfl]]]]]]]]]]]]].
  destruct (bid_eq_dec id id') as [<- | Hne].
  - (* the write is to this buffer, through its one flag span *)
    rewrite Hb in Hb'. injection Hb' as <-.
    assert (Hw1 : writes sp' = true) by exact (allows_writes sp' k Hallow Hk).
    unfold arg_ok in Hok'. destruct Hok' as [b' [_ [_ [_ [_ [_ [_ [Hwr _]]]]]]]].
    destruct (Hwr Hw1) as [_ [Hc | Hc]];
      [| exfalso; rewrite (proj2 (Hc off sp Hin)) in Hflag; discriminate Hflag].
    destruct (combine_once_view _ _ _ _ _ _ _ _ Hin Hin' Hc) as [<- <-].
    destruct (Hfl Hflag) as [_ [-> [Hem ->]]].
    destruct (Z.eq_dec kk e') as [-> | Hne'].
    + right. exact (word_write 4 m (base + bf_offset b + e') 1).
    + left. intros y Hy Hy'.
      apply Z.mod_divide in Hkm; [| lia]. apply Z.mod_divide in Hem; [| lia].
      destruct Hkm as [q1 Hq1]. destruct Hem as [q2 Hq2]. subst kk e'.
      assert (q1 <> q2) by (intros ->; contradiction).
      nia.
  - (* the write is to another buffer, apart from this one *)
    left. destruct (find_buffer_in _ _ _ Hb) as [Hbin Hbid]. destruct (find_buffer_in _ _ _ Hb') as [Hbin' Hbid'].
    destruct Hinv as [_ [_ [Hap _]]].
    assert (Hapart : apart b b') by (apply Hap; [exact Hbin | exact Hbin' | congruence]).
    intros y Hy Hy'. unfold apart in Hapart. lia.
Qed.

(* So every write that the facts allow leaves each word of 4 bytes of a
   flag buffer at its value or sets it to 1. *)
Lemma flags_keep : forall th o hist sh lh tt k x w v wr m id off sp b kk,
    platform_ok (launch_value th) ->
    GateMeaning.region F (env_of F (launch_value th) o hist sh lh) tt k x w v -> k <> KLoad ->
    In ((id, off), sp) pairs -> bs_flag sp = true -> find_buffer c id = Some b ->
    0 <= kk -> kk mod 4 = 0 -> kk + 4 <= bf_bytes b ->
    word_of (write_bytes m x w (match v with Some z => z | None => wr end)) (base + bf_offset b + kk) 4 =
      word_of m (base + bf_offset b + kk) 4 \/
    word_of (write_bytes m x w (match v with Some z => z | None => wr end)) (base + bf_offset b + kk) 4 = 1.
Proof.
  intros th o hist sh lh tt k x w v wr m id off sp b kk Hpf H Hk Hin Hflag Hb Hk0 Hkm Hkle.
  destruct (flags_set th o hist sh lh tt k x w v wr m id off sp b kk Hpf H Hk Hin Hflag Hb Hk0 Hkm Hkle)
    as [Hout | Hone]; [left | right; exact Hone].
  apply word_agree. intros y Hy. apply write_outside. apply Hout. cbn in Hy. lia.
Qed.

(* A one-dimensional complete-warp shape partitions every valid concrete
   thread coordinate. This does not assert completeness of a separately
   supplied scheduler thread list or termination of hardware collectives. *)
Lemma launch_value_tid : forall th r,
 category_reg F Mf r=Some(CTid 0) ->
 fits 0 (block_dim Mf 0-1) (wdn F r)=true -> thread_in th ->
 launch_value th r=tr_tid th 0.
Proof.
 intros th r Hcat Hfit [Htid Hrest]. unfold launch_value,raw_value. rewrite Hcat.
 apply(fits_trunc _ _ _ _ Hfit). specialize(Htid 0%nat). lia.
Qed.

Definition linked_scoped_schedule (prog:program) (events cta:list pc) (count:nat) : Prop :=
 exists words common,
 map fst words=seq 0 count /\
 forall w word,In(w,word)words ->
 event_projection cta word=common /\
 forall th o hist sh lh,thread_in th -> w=Z.to_nat(tr_tid th 0/32) ->
 platform_ok (launch_value th) -> (forall t',indices_in(hist t')) ->
 occurrence_trace F prog (env_of F (launch_value th) o hist sh lh)
   events word 0 (phase_initial(env_of F (launch_value th) o hist sh lh)).

Theorem linked_scoped_schedule_sound : forall prog events cta r count,
 warp_launch_shape F Mf=Some(r,count) ->
 scoped_schedule_safe F prog events cta r count ->
 linked_scoped_schedule prog events cta count.
Proof.
 intros prog events cta r count Hshape [words [common [Hindices Hwords]]].
 destruct (warp_launch_shape_sound F Mf r count Hshape)
   as [Hpos [Hmax [Hy [Hz [Hmultiple [Hcount [Hcat Hfit]]]]]]].
 exists words,common. split;[exact Hindices|].
 intros w word Hin. destruct (Hwords w word Hin) as [Hproject Htrace].
 split;[exact Hproject|]. intros th o hist sh lh Hth Hw Hplatform Hindices_memory.
 apply Htrace.
 - apply link_sound;assumption.
 - cbn [env_of e_init]. rewrite (launch_value_tid th r Hcat Hfit Hth),Hw.
   apply (proj2 (physical_warp_partition (block_dim Mf 0) (tr_tid th 0)
     Hpos Hmultiple (proj1 Hth 0%nat))).
Qed.

(* The launch context is captured by this section. Only the CTA coordinates
   vary between these words; thread/lane identities and oracles remain free. *)
Lemma same_cta_launch_values : forall th1 th2,
 (forall d,tr_ctaid th1 d=tr_ctaid th2 d)->
 context_initial_agree (cta_uniform_key F Mf)
   {|e_mem:=fun _ _=>0;e_arb:=fun _ _=>0;e_shm:=fun _ _=>0;e_init:=launch_value th1|}
   {|e_mem:=fun _ _=>0;e_arb:=fun _ _=>0;e_shm:=fun _ _=>0;e_init:=launch_value th2|}.
Proof.
 intros th1 th2 Hcta r Hkey. cbn[e_init].
 unfold launch_value,raw_value,cta_uniform_key in *.
 destruct(category_reg F Mf r)as[k|];[|discriminate].
 destruct k;try discriminate;try reflexivity. now rewrite Hcta.
Qed.
Definition linked_context_schedule (prog:program) (events:list pc) : Prop :=
 forall coordinates:nat->Z, exists word,
 forall th o hist sh lh,
 thread_in th ->(forall d,tr_ctaid th d=coordinates d)->
 platform_ok(launch_value th)->(forall t',indices_in(hist t'))->
 occurrence_trace F prog(env_of F(launch_value th)o hist sh lh)
   events word 0(phase_initial(env_of F(launch_value th)o hist sh lh)).
Theorem linked_context_schedule_sound : forall prog events,
 context_schedule_safe F(cta_uniform_key F Mf)prog events ->
 linked_context_schedule prog events.
Proof.
 intros prog events [words[Htrace Hagree]] coordinates.
 (* This reference only indexes the logical word. It is never assumed to be
    a valid launch or used as an execution premise. *)
 set(reference:={|tr_ctaid:=coordinates;tr_tid:=fun _=>0;tr_lane:=0;tr_platform:=fun _=>0|}).
 set(Eref:={|e_mem:=fun _ _=>0;e_arb:=fun _ _=>0;e_shm:=fun _ _=>0;e_init:=launch_value reference|}).
 exists(words Eref). intros th o hist sh lh Hthread Hcoordinates Hplatform Hindices.
 assert(Hinitial:context_initial_agree(cta_uniform_key F Mf)Eref(env_of F(launch_value th)o hist sh lh)).
 {intros r Hkey. unfold Eref;cbn[e_init env_of].
  apply(same_cta_launch_values reference th);[|exact Hkey].
  intros d. unfold reference;cbn[tr_ctaid]. symmetry;apply Hcoordinates. }
 rewrite(Hagree Eref(env_of F(launch_value th)o hist sh lh)Hinitial).
 apply Htrace. apply link_sound;assumption.
Qed.

(* Every lane of warp w in the CTA at the given coordinates follows one word,
   and the words of all warps and CTAs have one projection onto the CTA
   barriers. *)
Definition linked_warp_exit_schedule (prog:program) (events cta:list pc) (count:nat) : Prop :=
 exists common,forall(coordinates:nat->Z)w,(w<count)%nat ->exists word,
 event_projection cta word=common /\
 forall th o hist sh lh,
 thread_in th ->(forall d,tr_ctaid th d=coordinates d) ->w=Z.to_nat(tr_tid th 0/32) ->
 platform_ok(launch_value th) ->(forall t',indices_in(hist t')) ->
 occurrence_trace F prog(env_of F(launch_value th)o hist sh lh)
   events word 0(phase_initial(env_of F(launch_value th)o hist sh lh)).
Theorem linked_warp_exit_schedule_sound : forall prog events cta r count,
 warp_launch_shape F Mf=Some(r,count) ->
 warp_exit_schedule_safe F(cta_uniform_key F Mf)prog events cta r count ->
 linked_warp_exit_schedule prog events cta count.
Proof.
 intros prog events cta r count Hshape [common Hwarps].
 destruct (warp_launch_shape_sound F Mf r count Hshape)
   as [Hpos [Hmax [Hy [Hz [Hmultiple [Hcount [Hcat Hfit]]]]]]].
 exists common. intros coordinates w Hw. destruct(Hwarps w Hw)as[choose[Hagree[Hproject Htrace]]].
 (* As in linked_context_schedule_sound, this reference only indexes the word. *)
 set(reference:={|tr_ctaid:=coordinates;tr_tid:=fun _=>0;tr_lane:=0;tr_platform:=fun _=>0|}).
 set(Eref:={|e_mem:=fun _ _=>0;e_arb:=fun _ _=>0;e_shm:=fun _ _=>0;e_init:=launch_value reference|}).
 exists(choose Eref). split;[exact(Hproject Eref)|].
 intros th o hist sh lh Hthread Hcoordinates Hwarp Hplatform Hindices.
 assert(Hinitial:context_initial_agree(cta_uniform_key F Mf)Eref(env_of F(launch_value th)o hist sh lh)).
 {intros q Hkey. unfold Eref;cbn[e_init env_of].
  apply(same_cta_launch_values reference th);[|exact Hkey].
  intros d. unfold reference;cbn[tr_ctaid]. symmetry;apply Hcoordinates. }
 rewrite(Hagree Eref(env_of F(launch_value th)o hist sh lh)Hinitial).
 apply Htrace.
 - apply link_sound;assumption.
 - cbn [env_of e_init]. rewrite (launch_value_tid th r Hcat Hfit Hthread),Hwarp.
   apply (proj2 (physical_warp_partition (block_dim Mf 0) (tr_tid th 0)
     Hpos Hmultiple (proj1 Hthread 0%nat))).
Qed.

(* Every thread of a block executes the same sequence of waiting sites, when
   every thread's run ends (which term_check proves), the oracles of the
   block's threads satisfy the three facts of the PTX semantics that
   GateShared.v states, and the platform gives every thread of the block
   the same address of each shared variable. *)
Definition linked_shared_schedule (prog:program) (cta events:list pc) (cols:list(pc*collective)) : Prop :=
 kernel_ends F prog ->
 forall (th:nat->thread_regs) (o:nat->oracle) (hist sh lh:nat->nat->memory),
 let n:=Z.to_nat(block_dim Mf 0) in
 let E:=fun x=>env_of F(launch_value(th x))(o x)(hist x)(sh x)(lh x) in
 (forall x,(x<n)%nat ->thread_in(th x) /\ tr_tid(th x)0=Z.of_nat x /\
   platform_ok(launch_value(th x)) /\ forall t',indices_in(hist x t')) ->
 (forall x y d,(x<n)%nat ->(y<n)%nat ->tr_ctaid(th x)d=tr_ctaid(th y)d) ->
 (forall x y r,(x<n)%nat ->(y<n)%nat ->shared_variable_key F r=true ->
   launch_value(th x)r=launch_value(th y)r) ->
 shared_load_agrees F prog cta events n E ->redux_agrees F prog events cols n E ->
 bfly_reads_partner F prog events cols n E ->
 same_waits F prog events n E.
Theorem linked_shared_schedule_sound : forall prog cta events cols r count,
 warp_launch_shape F Mf=Some(r,count) ->
 shared_schedule_safe F(shared_uniform F Mf)r(32*count)prog cta events cols ->
 linked_shared_schedule prog cta events cols.
Proof.
 intros prog cta events cols r count Hshape Hsafe Hends th o hist sh lh n E Hthreads Hcta Hplat Hload Hredux Hbfly.
 destruct (warp_launch_shape_sound F Mf r count Hshape)
   as [Hpos [Hmax [Hy1 [Hz1 [Hmultiple [Hcount [Hcat Hfit]]]]]]].
 assert(Hn:(32*count)%nat=n).
 { subst n. rewrite Hcount. apply Nat2Z.inj. rewrite Nat2Z.inj_mul, !Z2Nat.id by (try apply Z.div_pos; lia).
   pose proof (Z.div_mod (block_dim Mf 0) 32 ltac:(lia)). lia. }
 rewrite <-Hn in *. apply(Hsafe Hends E). constructor.
 - intros x Hx. destruct(Hthreads x Hx)as[Hin[_[Hpf Hidx]]]. apply link_sound;assumption.
 - intros x Hx. destruct(Hthreads x Hx)as[Hin[Htid _]]. cbn[E env_of e_init].
   rewrite(launch_value_tid(th x)r Hcat Hfit Hin). exact Htid.
 - intros x y q Hx Hy Hu. unfold shared_uniform in Hu. apply orb_true_iff in Hu. cbn[E env_of e_init].
   destruct Hu as[Hu|Hu].
   + exact(same_cta_launch_values(th x)(th y)(fun d=>Hcta x y d Hx Hy)q Hu).
   + exact(Hplat x y q Hx Hy Hu).
 - exact Hload.
 - exact Hredux.
 - exact Hbfly.
Qed.

(* The concrete guarantee is tied to the exact accepted source lowering,
   complete source-event alphabet and checked CTA-marker projection. The
   old common-word proposition remains an explicit alternative. *)
Theorem accepted_front_linked_events : forall K prog K' org,
 front K Mf=Ok(prog,F) -> inline K=Ok(K',org) ->
 (2<=List.length(source_events K' (expanded_places org)))%nat ->
 barrier_linear_case K'=false ->
 exists pp raw ts events,
 ((p <- translate_placed K' Mf org ;; F <- facts K' Mf ;; ret(p,F)) init)=Ok((pp,raw),ts) /\
 prog=map(map fst)pp /\ F=payload_facts raw prog /\
 placed_events K' (expanded_places org) (ts_ids ts) pp
   (source_events K' (expanded_places org))=Some events /\
 ((((fixed_progress_safe F prog events \/
   (exists cta r count,
    placed_events K' (expanded_places org) (ts_ids ts) pp
      (source_barriers K' (expanded_places org))=Some cta /\
    warp_launch_shape F Mf=Some(r,count) /\
    linked_scoped_schedule prog events cta count) \/
   exists cta r count,
    placed_events K' (expanded_places org) (ts_ids ts) pp
      (source_barriers K' (expanded_places org))=Some cta /\
    warp_launch_shape F Mf=Some(r,count) /\
    linked_warp_exit_schedule prog events cta count) \/
   (placed_events K' (expanded_places org) (ts_ids ts) pp
      (source_barriers K' (expanded_places org))=Some events /\
    linked_context_schedule prog events)) \/
   exists cta,placed_events K' (expanded_places org) (ts_ids ts) pp
      (source_barriers K' (expanded_places org))=Some cta /\
    cta<>[] /\event_word_eqb cta events=false /\
    linked_context_schedule prog events) \/
   exists cta r count,
    placed_events K' (expanded_places org) (ts_ids ts) pp
      (source_barriers K' (expanded_places org))=Some cta /\
    warp_launch_shape F Mf=Some(r,count) /\
    linked_shared_schedule prog cta events (source_collectives K' (expanded_places org) (ts_ids ts) pp)).
Proof.
 intros K prog K' org Hfront Hinline Hcount Hlinear.
 destruct (accepted_front_events K Mf prog F K' org Hfront Hinline Hcount Hlinear)
   as [pp [raw [ts [events [Hlower [Hprog [Hfacts [Hplaced Hsafe]]]]]]]].
 exists pp,raw,ts,events. split;[exact Hlower|]. split;[exact Hprog|].
 split;[exact Hfacts|]. split;[exact Hplaced|].
 destruct Hsafe as [Hsafe|[cta [r [count [Hcta [Hshape Hshared]]]]]].
 2:{ right. exists cta,r,count. split;[exact Hcta|]. split;[exact Hshape|].
     rewrite <-Hprog in Hshared. eapply linked_shared_schedule_sound;eassumption. }
 left.
 unfold front_events_safe in Hsafe.
 destruct Hsafe as[Hlegacy|[cta[Hcta[Hnonempty[Hdiff Hcontext]]]]].
 - left. unfold front_legacy_events_safe in Hlegacy.
   destruct Hlegacy as[Hold|[cta[Hcta[Hsame Hcontext]]]].
   + unfold front_fixed_events_safe in Hold. rewrite <-Hprog in Hold.
     left. destruct Hold as[Hold|[[cta[r[count[Hcta[Hshape Hscoped]]]]]|[cta[r[count[Hcta[Hshape Hexit]]]]]]].
     * left;exact Hold.
     * right. left. exists cta,r,count. split;[exact Hcta|]. split;[exact Hshape|].
       eapply linked_scoped_schedule_sound;eassumption.
     * right. right. exists cta,r,count. split;[exact Hcta|]. split;[exact Hshape|].
       eapply linked_warp_exit_schedule_sound;eassumption.
   + right. subst cta. split;[exact Hcta|]. rewrite <-Hprog in Hcontext.
     now apply linked_context_schedule_sound.
 - right. exists cta. split;[exact Hcta|]. split;[exact Hnonempty|]. split;[exact Hdiff|].
   rewrite <-Hprog in Hcontext. now apply linked_context_schedule_sound.
Qed.

End Link.
