(* StoreModel.v: the pool's bytes under the steps the catalog accepts.

   A world is the catalog and memory, a byte at each device address; the
   pool's byte at offset o lies at address base + o. An action that the
   catalog accepts runs its effects in order (`run_action`): a clear writes
   zeros, an upload the data's bytes, a memset the element's bytes at each
   element, and a kernel a launch of the verified operator (StoreLaunch.v)
   with the values that the launch puts in its registers (StoreLink.v), on
   the threads, oracles, and schedule that the hardware chooses (`choice`),
   from the shared memory that an earlier launch left in each block and
   the local memory that an earlier launch left in each thread, which are
   any contents for a kernel without local variables; a kernel with local
   variables starts with zeroed local memory (`local_start`, the fact of
   decision 46).
   A step runs its actions in order if the catalog accepts it, and leaves
   the world as it is otherwise (`run_step`).

   Proved: `run_steps_ok`, every run keeps the world's invariants
   (`world_ok`): the catalog's invariant (CatalogStep.v), every element of a
   checked buffer lies in its range (`status_ok`), every operator of the
   catalog is verified, and the pool lies below 2^64 at a multiple of 256
   bytes. A launch changes only bytes of buffers its tenant may write
   (`launch_writes`), and two runs of it with the same choice from memories
   that agree on every buffer its tenant may read agree on those bytes
   after it (`launch_reads`).

   Trusted, as the hardware's facts (`hardware_ok`): each thread's indices
   lie in the block and the grid, the platform's registers satisfy
   `platform_ok` (StoreLink.v) and hold the same values for the threads of
   a block, and every thread index of a block is a thread of the launch. *)

From Coq Require Import ZArith.ZArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Lia.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateDomain.
Require Import OpGate.GateMeaning.
Require Import OpGate.GateThread.
Require Import OpGate.GateFront.
Require Import OpGate.GateZero.
Require Import OpGate.GatePrologue.
Require Import OpGate.GateCertificate.
Require Import OpGate.CatalogStep.
Require Import OpGate.StoreLaunch.
Require Import OpGate.StoreLink.
Require OpGate.GateParse.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Host writes.                                                        *)

Definition upload_bytes (m : memory) (a : Z) (data : list Z) : memory :=
  fun y => if (a <=? y) && (y <? a + Z.of_nat (List.length data)) then nth (Z.to_nat (y - a)) data 0 else m y.

(* The n bytes at a, each element of e bytes the bytes of v. *)
Definition fill_bytes (m : memory) (a n e v : Z) : memory :=
  fun y => if (a <=? y) && (y <? a + n) then (v / 2 ^ (8 * ((y - a) mod e))) mod 256 else m y.

Lemma skipn_cons_nth : forall (A : Type) (l : list A) k d,
    (k < List.length l)%nat -> skipn k l = nth k l d :: skipn (S k) l.
Proof.
  intros A l. induction l as [| x l IH]; intros k d Hk; [cbn in Hk; lia |].
  destruct k as [| k]; [reflexivity |]. cbn [skipn nth]. apply IH. cbn in Hk. lia.
Qed.

Lemma upload_word : forall n k m a data, (k + n <= List.length data)%nat ->
    word_of (upload_bytes m a data) (a + Z.of_nat k) n = bytes_value (firstn n (skipn k data)).
Proof.
  induction n as [| n IH]; intros k m a data Hk; [reflexivity |].
  cbn [word_of]. rewrite (skipn_cons_nth _ data k 0) by lia. cbn [firstn bytes_value].
  unfold upload_bytes at 1.
  replace ((a <=? a + Z.of_nat k) && (a + Z.of_nat k <? a + Z.of_nat (List.length data))) with true
    by (symmetry; apply andb_true_iff; split; [apply Z.leb_le | apply Z.ltb_lt]; lia).
  replace (a + Z.of_nat k - a) with (Z.of_nat k) by lia. rewrite Nat2Z.id.
  replace (a + Z.of_nat k + 1) with (a + Z.of_nat (S k)) by lia. rewrite IH by lia. reflexivity.
Qed.

Lemma fill_word : forall m a n e v k, 1 <= e -> 0 <= k -> k mod e = 0 -> k + e <= n ->
    word_of (fill_bytes m a n e v) (a + k) (Z.to_nat e) = v mod 2 ^ (8 * e).
Proof.
  intros m a n e v k He Hk Hkm Hkn.
  rewrite (word_agree _ (write_bytes m (a + k) (Z.of_nat (Z.to_nat e)) v)).
  - rewrite word_write, Z2Nat.id by lia. reflexivity.
  - intros y Hy. rewrite Z2Nat.id in * by lia. unfold fill_bytes, write_bytes.
    replace ((a <=? y) && (y <? a + n)) with true
      by (symmetry; apply andb_true_iff; split; [apply Z.leb_le | apply Z.ltb_lt]; lia).
    replace ((a + k <=? y) && (y <? a + k + e)) with true
      by (symmetry; apply andb_true_iff; split; [apply Z.leb_le | apply Z.ltb_lt]; lia).
    apply Z.mod_divide in Hkm; [| lia]. destruct Hkm as [q ->].
    replace (y - a) with (y - (a + q * e) + q * e) by lia. rewrite Z.mod_add by lia.
    rewrite (Z.mod_small (y - (a + q * e)) e) by lia. reflexivity.
Qed.

(* An element's value read back from its bytes. *)
Lemma extend_value : forall e s v, 1 <= e ->
    fst (element_range e s) <= v <= snd (element_range e s) ->
    extend (Z.to_nat (8 * e)) s (v mod 2 ^ (8 * e)) = v.
Proof.
  intros e s v He Hv.
  assert (Hp : 2 ^ (8 * e) = 2 * 2 ^ (8 * e - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  assert (Hq : 0 < 2 ^ (8 * e - 1)) by (apply Z.pow_pos_nonneg; lia).
  destruct s; unfold element_range in Hv; cbn [fst snd] in Hv; unfold extend, sext, trunc;
    rewrite Z2Nat.id by lia; rewrite Z.mod_mod by lia.
  - destruct (Z.le_gt_cases 0 v) as [Hv0 | Hv0].
    + rewrite Z.mod_small by lia. destruct (Z.ltb_spec v (2 ^ (8 * e - 1))); lia.
    + replace (v mod 2 ^ (8 * e)) with (v + 2 ^ (8 * e)) by (apply (Z.mod_unique _ _ (-1)); lia).
      destruct (Z.ltb_spec (v + 2 ^ (8 * e)) (2 ^ (8 * e - 1))); lia.
  - apply Z.mod_small. lia.
Qed.

(* The element at byte offset k of data is one of its elements. *)
Lemma elements_at : forall e s data k, 1 <= e -> 0 <= k -> k mod e = 0 -> k + e <= Z.of_nat (List.length data) ->
    In (extend (Z.to_nat (8 * e)) s (bytes_value (firstn (Z.to_nat e) (skipn (Z.to_nat k) data))))
       (elements e s data).
Proof.
  intros e s data k He Hk Hkm Hkl. unfold elements. apply in_map_iff.
  apply Z.mod_divide in Hkm; [| lia]. destruct Hkm as [q ->].
  assert (Hq : 0 <= q) by nia.
  exists (Z.to_nat q). split.
  - rewrite <- Z2Nat.inj_mul by lia. reflexivity.
  - apply in_seq. split; [lia |]. cbn [plus]. apply Nat2Z.inj_lt.
    rewrite Nat2Z.inj_div, !Z2Nat.id by lia.
    apply (Z.lt_le_trans _ (q + 1)); [lia | apply Z.div_le_lower_bound; nia].
Qed.

Lemma fold_min_le : forall l x y, In y (x :: l) -> fold_left Z.min l x <= y.
Proof.
  induction l as [| z l IH]; intros x y H; cbn [fold_left].
  - destruct H as [<- | []]. lia.
  - destruct H as [-> | [-> | H]].
    + pose proof (IH (Z.min y z) (Z.min y z) (or_introl eq_refl)). lia.
    + pose proof (IH (Z.min x y) (Z.min x y) (or_introl eq_refl)). lia.
    + exact (IH (Z.min x z) y (or_intror H)).
Qed.

Lemma fold_max_ge : forall l x y, In y (x :: l) -> y <= fold_left Z.max l x.
Proof.
  induction l as [| z l IH]; intros x y H; cbn [fold_left].
  - destruct H as [<- | []]. lia.
  - destruct H as [-> | [-> | H]].
    + pose proof (IH (Z.max y z) (Z.max y z) (or_introl eq_refl)). lia.
    + pose proof (IH (Z.max x y) (Z.max x y) (or_introl eq_refl)). lia.
    + exact (IH (Z.max x z) y (or_intror H)).
Qed.

Lemma data_range_in : forall e s data y, In y (elements e s data) ->
    fst (data_range e s data) <= y <= snd (data_range e s data).
Proof.
  intros e s data y H. unfold data_range. destruct (elements e s data) as [| x rest]; [destruct H |].
  cbn [fst snd]. split; [apply fold_min_le | apply fold_max_ge]; exact H.
Qed.

Lemma width_pos : forall e, width_ok e = true -> 1 <= e <= 8.
Proof. intros e H. unfold width_ok in H. rewrite !orb_true_iff, !Z.eqb_eq in H. lia. Qed.

(* ------------------------------------------------------------------ *)
(* What each accepted action gives.                                    *)

Lemma allocate_next : forall c t id o n c' es, check_action c t (Allocate id o n) = inl (c', es) ->
    c' = {| c_pool := c_pool c;
            c_buffers := c_buffers c ++ [{| bf_id := id; bf_offset := o; bf_bytes := n; bf_owner := t;
                                            bf_status := SRaw |}];
            c_grants := c_grants c; c_freed := carve (c_freed c) o n; c_reserved := c_reserved c;
            c_operators := c_operators c |} /\
    ((es = [] /\ exists r, In r (c_freed c) /\ rg_owner r = t /\ rg_offset r <= o /\ o + n <= rg_offset r + rg_bytes r)
     \/ es = [Clear o n]) /\
    existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers c) = false /\
    find_buffer c id = None.
Proof.
  intros c t id o n c' es H. cbn [check_action] in H. destruct (negb (Z.eqb (fst id) t)); [discriminate H |].
  destruct (find_buffer c id) eqn:Fb; [discriminate H |].
  refusals H; injection H as <- <-; (split; [reflexivity | split; [| split; reflexivity]]).
  - left. split; [reflexivity |].
    match goal with Hk : existsb _ (c_freed c) = true |- _ =>
      apply existsb_exists in Hk; destruct Hk as [r [Hr Hrk]] end.
    rewrite !andb_true_iff, Z.eqb_eq, !Z.leb_le in Hrk. destruct Hrk as [[Ho Hlo] Hhi].
    exists r. repeat split; assumption.
  - right. reflexivity.
Qed.

(* An allocation takes an id of its tenant and a region of its tenant's
   reservation. *)
Lemma allocate_place : forall c t id o n c' es, check_action c t (Allocate id o n) = inl (c', es) ->
    fst id = t /\ in_reservation c t o n = true /\ 1 <= n.
Proof.
  intros c t id o n c' es H. cbn [check_action] in H.
  destruct (negb (Z.eqb (fst id) t)) eqn:Et; [discriminate H |]. apply negb_false_iff, Z.eqb_eq in Et.
  destruct (find_buffer c id); [discriminate H |].
  destruct (Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool c) (o + n)) eqn:E1; [discriminate H |].
  destruct (negb (Z.eqb (o mod alignment) 0)); [discriminate H |].
  destruct (negb (in_reservation c t o n)) eqn:Er; [discriminate H |].
  apply negb_false_iff in Er. rewrite !orb_false_iff, !Z.ltb_ge in E1. split; [exact Et | split; [exact Er | lia]].
Qed.

Lemma free_next : forall c t id c' es, check_action c t (Free id) = inl (c', es) ->
    exists b, find_buffer c id = Some b /\ bf_owner b = t /\ es = [] /\
      c' = {| c_pool := c_pool c; c_buffers := filter (fun x => negb (bid_eqb (bf_id x) id)) (c_buffers c);
              c_grants := filter (fun g => negb (bid_eqb (fst g) id)) (c_grants c);
              c_freed := release (c_freed c) (bf_offset b) (bf_bytes b) t; c_reserved := c_reserved c;
              c_operators := c_operators c |}.
Proof.
  intros c t id c' es H. cbn [check_action] in H. destruct (find_buffer c id) as [b |]; [| discriminate H].
  refusals H. injection H as <- <-. exists b. split; [reflexivity |].
  match goal with E : negb (bf_owner b =? t) = false |- _ => apply negb_false_iff, Z.eqb_eq in E; rename E into Ho end.
  split; [exact Ho |]. split; reflexivity.
Qed.

Lemma grant_next : forall c t id u c' es, check_action c t (Grant id u) = inl (c', es) ->
    exists b, find_buffer c id = Some b /\ bf_owner b = t /\ u <> t /\ es = [] /\
      c' = {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c ++ [(id, u)];
              c_freed := c_freed c; c_reserved := c_reserved c; c_operators := c_operators c |}.
Proof.
  intros c t id u c' es H. cbn [check_action] in H. destruct (find_buffer c id) as [b |]; [| discriminate H].
  refusals H. injection H as <- <-. exists b. split; [reflexivity |].
  match goal with E : (negb (bf_owner b =? t) || (u =? t)) = false |- _ =>
    apply orb_false_iff in E; destruct E as [Ho Hu] end.
  apply negb_false_iff, Z.eqb_eq in Ho. apply Z.eqb_neq in Hu.
  split; [exact Ho |]. split; [exact Hu |]. split; reflexivity.
Qed.

Lemma write_next : forall c t id data element c' es, check_action c t (Write id data element) = inl (c', es) ->
    exists b, find_buffer c id = Some b /\ es = [Upload (bf_offset b) data] /\
      Z.of_nat (List.length data) <= bf_bytes b /\
      match element with
      | None => c' = set_status c id SRaw
      | Some (e, s) => width_ok e = true /\ Z.of_nat (List.length data) = bf_bytes b /\ bf_bytes b mod e = 0 /\
                       c' = set_status c id (SChecked e s (fst (data_range e s data)) (snd (data_range e s data)))
      end.
Proof.
  intros c t id data element c' es H. cbn [check_action] in H.
  destruct (find_buffer c id) as [b |]; [| discriminate H]. exists b. split; [reflexivity |].
  destruct element as [[e s] |]; refusals H.
  - destruct (data_range e s data) as [lo hi] eqn:Er. injection H as <- <-.
    match goal with E : (width_ok e && _ && _) = true |- _ =>
      rewrite !andb_true_iff, !Z.eqb_eq in E; destruct E as [[Hw Hl] Hm] end.
    split; [reflexivity |]. split; [lia |]. cbn [fst snd].
    split; [exact Hw | split; [exact Hl | split; [exact Hm | reflexivity]]].
  - injection H as <- <-.
    match goal with E : Z.leb _ _ = true |- _ => apply Z.leb_le in E end.
    split; [reflexivity |]. split; [assumption | reflexivity].
Qed.

Lemma fill_next : forall c t id e s v c' es, check_action c t (Fill id e s v) = inl (c', es) ->
    exists b, find_buffer c id = Some b /\ es = [Memset (bf_offset b) (bf_bytes b) e s v] /\
      width_ok e = true /\ fst (element_range e s) <= v <= snd (element_range e s) /\
      bf_bytes b mod e = 0 /\ c' = set_status c id (SChecked e s v v).
Proof.
  intros c t id e s v c' es H. cbn [check_action] in H.
  destruct (find_buffer c id) as [b |]; [| discriminate H]. exists b.
  refusals H. destruct (element_range e s) as [lo hi] eqn:Er. refusals H. injection H as <- <-.
  match goal with E : (width_ok e && _ && _ && _) = true |- _ =>
    rewrite !andb_true_iff, !Z.leb_le, Z.eqb_eq in E; destruct E as [[[Hw Hlo] Hhi] Hm] end.
  cbn [fst snd]. repeat split; assumption.
Qed.

Lemma check_launch_next : forall c t op gx gy gz buffers sizes scalars guarded c' k Mf,
    find_binding c op = Some Mf -> check_launch c t op gx gy gz buffers sizes scalars guarded = inl (c', k) ->
    c' = fold_left (fun acc x => written acc sizes x) (combine buffers (spans Mf)) c /\
    k = Kernel op gx gy gz (eval (b_shared Mf) sizes) (args_of c sizes (b_params Mf) buffers scalars)
               (checks_of c guarded sizes (combine buffers (spans Mf))).
Proof.
  intros c t op gx gy gz buffers sizes scalars guarded c' k Mf Hb H. unfold check_launch, grid_limit in H.
  rewrite Hb in H. destruct (b_grid Mf) as [[mx my] mz]. cbn beta iota in H. refusals H.
  destruct (first_reason _ _); [discriminate H |]. injection H as <- <-. split; reflexivity.
Qed.

Lemma launch_next : forall c t op gx gy gz buffers sizes scalars guarded c' es,
    check_action c t (Launch op gx gy gz buffers sizes scalars guarded) = inl (c', es) ->
    exists Mf, find_binding c op = Some Mf /\
      (exists r, check_launch c t op gx gy gz buffers sizes scalars guarded = inl r) /\
      c' = fold_left (fun acc x => written acc sizes x) (combine buffers (spans Mf)) c /\
      es = [Kernel op gx gy gz (eval (b_shared Mf) sizes) (args_of c sizes (b_params Mf) buffers scalars)
                   (checks_of c guarded sizes (combine buffers (spans Mf)))].
Proof.
  intros c t op gx gy gz buffers sizes scalars guarded c' es H. cbn [check_action] in H.
  destruct (check_launch c t op gx gy gz buffers sizes scalars guarded) as [[c'' k] |] eqn:L; [| discriminate H].
  injection H as <- <-.
  destruct (find_binding c op) as [Mf |] eqn:Hb.
  - destruct (check_launch_next _ _ _ _ _ _ _ _ _ _ _ _ _ Hb L) as [-> ->].
    exists Mf. split; [reflexivity |]. split; [eexists; reflexivity |]. split; reflexivity.
  - unfold check_launch in L. rewrite Hb in L. discriminate L.
Qed.

Lemma reserve_next : forall c t o n c' es, check_action c t (Reserve o n) = inl (c', es) ->
    find_reservation c t = None /\ 1 <= n /\ 0 <= o /\ o + n <= c_pool c /\
    existsb (fun r => overlap (rs_offset r) (rs_bytes r) o n) (c_reserved c) = false /\ es = [] /\
    c' = {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c; c_freed := c_freed c;
            c_reserved := c_reserved c ++ [{| rs_tenant := t; rs_offset := o; rs_bytes := n |}];
            c_operators := c_operators c |}.
Proof.
  intros c t o n c' es H. cbn [check_action] in H. destruct (find_reservation c t); [discriminate H |].
  destruct (Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool c) (o + n)) eqn:E1; [discriminate H |].
  destruct (existsb (fun r => overlap (rs_offset r) (rs_bytes r) o n) (c_reserved c)) eqn:E2; [discriminate H |].
  injection H as <- <-. rewrite !orb_false_iff, !Z.ltb_ge in E1.
  split; [reflexivity |]. split; [lia |]. split; [lia |]. split; [lia |].
  split; [reflexivity |]. split; reflexivity.
Qed.

Lemma unreserve_next : forall c t c' es, check_action c t Unreserve = inl (c', es) ->
    find_reservation c t <> None /\ existsb (fun b => Z.eqb (bf_owner b) t) (c_buffers c) = false /\ es = [] /\
    c' = {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c;
            c_freed := filter (fun r => negb (Z.eqb (rg_owner r) t)) (c_freed c);
            c_reserved := filter (fun r => negb (Z.eqb (rs_tenant r) t)) (c_reserved c);
            c_operators := c_operators c |}.
Proof.
  intros c t c' es H. cbn [check_action] in H. destruct (find_reservation c t); [| discriminate H].
  destruct (existsb (fun b => Z.eqb (bf_owner b) t) (c_buffers c)) eqn:E; [discriminate H |].
  injection H as <- <-. split; [discriminate |]. split; [reflexivity |]. split; reflexivity.
Qed.

Lemma check_action_frame : forall c t a c' es, check_action c t a = inl (c', es) ->
    c_pool c' = c_pool c /\ c_operators c' = c_operators c.
Proof.
  intros c t a c' es H.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - destruct (allocate_next _ _ _ _ _ _ _ H) as [-> _]. split; reflexivity.
  - destruct (free_next _ _ _ _ _ H) as [b [_ [_ [_ ->]]]]. split; reflexivity.
  - destruct (grant_next _ _ _ _ _ _ H) as [b [_ [_ [_ [_ ->]]]]]. split; reflexivity.
  - destruct (write_next _ _ _ _ _ _ _ H) as [b [_ [_ [_ Hc]]]].
    destruct element as [[e s] |]; [destruct Hc as [_ [_ [_ ->]]] | rewrite Hc]; split; reflexivity.
  - destruct (fill_next _ _ _ _ _ _ _ _ H) as [b [_ [_ [_ [_ [_ ->]]]]]]. split; reflexivity.
  - destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [_ [_ [-> _]]]]. clear H.
    generalize c. induction (combine buffers (spans Mf)) as [| [[id off] sp] l IH]; intros c0; [split; reflexivity |].
    cbn [fold_left]. destruct (IH (written c0 sizes ((id, off), sp))) as [E1 E2]. rewrite E1, E2. unfold written.
    destruct (negb (writes sp)); [split; reflexivity |]. destruct (bs_index sp); split; reflexivity.
  - destruct (reserve_next _ _ _ _ _ _ H) as [_ [_ [_ [_ [_ [_ ->]]]]]]. split; reflexivity.
  - destruct (unreserve_next _ _ _ _ H) as [_ [_ [_ ->]]]. split; reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* The statuses a launch leaves.                                       *)

Definition with_status (b : buffer) (st : status) : buffer :=
  {| bf_id := bf_id b; bf_offset := bf_offset b; bf_bytes := bf_bytes b; bf_owner := bf_owner b; bf_status := st |}.

Lemma with_status_same : forall b, with_status b (bf_status b) = b.
Proof. intros []. reflexivity. Qed.

Definition span_status (sizes : list Z) (sp : bspan) : status :=
  match bs_index sp with
  | Some ir => SChecked (ir_elem ir) (ir_signed ir) (eval (ir_lo ir) sizes) (eval (ir_hi ir) sizes)
  | None => SRaw
  end.

Definition after_write (sizes : list Z) (id : bid) (st : status) (x : (bid * Z) * bspan) : status :=
  if bid_eqb id (fst (fst x)) && writes (snd x) then span_status sizes (snd x) else st.

Lemma written_one : forall sizes c0 x,
    c_buffers (written c0 sizes x) =
    map (fun b => with_status b (after_write sizes (bf_id b) (bf_status b) x)) (c_buffers c0).
Proof.
  intros sizes c0 [[id off] sp]. unfold written, after_write. cbn [fst snd].
  destruct (writes sp) eqn:W; cbn [negb].
  - destruct (bs_index sp) as [ir |] eqn:Ei; unfold set_status; cbn [c_buffers]; apply map_ext; intros b;
      rewrite andb_true_r; unfold span_status; rewrite Ei;
      (destruct (bid_eqb (bf_id b) id); [reflexivity | symmetry; apply with_status_same]).
  - rewrite <- (map_id (c_buffers c0)) at 1. apply map_ext. intros b. rewrite andb_false_r.
    symmetry. apply with_status_same.
Qed.

Lemma written_buffers : forall sizes l c0,
    c_buffers (fold_left (fun acc x => written acc sizes x) l c0) =
    map (fun b => with_status b (fold_left (after_write sizes (bf_id b)) l (bf_status b))) (c_buffers c0).
Proof.
  intros sizes l. induction l as [| x l IH]; intros c0.
  - cbn [fold_left]. rewrite <- (map_id (c_buffers c0)) at 1. apply map_ext. intros b.
    symmetry. apply with_status_same.
  - cbn [fold_left]. rewrite IH, written_one, map_map. apply map_ext. intros b. reflexivity.
Qed.

Lemma after_written : forall sizes id sp l st, writes sp = true ->
    (forall off' sp', In ((id, off'), sp') l -> sp' = sp) ->
    fold_left (after_write sizes id) l st =
    if existsb (fun x => bid_eqb id (fst (fst x))) l then span_status sizes sp else st.
Proof.
  intros sizes id sp l. induction l as [| [[i o0] sp0] l IH]; intros st Hw Hu; [reflexivity |].
  cbn [fold_left existsb fst]. unfold after_write at 2. cbn [fst snd].
  assert (Hu' : forall off' sp', In ((id, off'), sp') l -> sp' = sp)
    by (intros off' sp' H; apply (Hu off'); right; exact H).
  destruct (bid_eqb_spec id i) as [<- | Hne].
  - rewrite (Hu o0 sp0 (or_introl eq_refl)), Hw. cbn [andb orb]. rewrite (IH _ Hw Hu').
    destruct (existsb _ l); reflexivity.
  - cbn [andb orb]. exact (IH st Hw Hu').
Qed.

Lemma after_unwritten : forall sizes id l st, (forall off sp, In ((id, off), sp) l -> writes sp = false) ->
    fold_left (after_write sizes id) l st = st.
Proof.
  intros sizes id l. induction l as [| [[i o] sp] l IH]; intros st H; [reflexivity |].
  cbn [fold_left]. unfold after_write at 2. cbn [fst snd].
  assert (H' : forall off' sp', In ((id, off'), sp') l -> writes sp' = false)
    by (intros off' sp' Hin; apply (H off'); right; exact Hin).
  destruct (bid_eqb_spec id i) as [<- | Hne].
  - rewrite (H o sp (or_introl eq_refl)). cbn [andb]. exact (IH st H').
  - cbn [andb]. exact (IH st H').
Qed.

Lemma written_exists : forall id off sp (l : list ((bid * Z) * bspan)), In ((id, off), sp) l -> writes sp = true ->
    existsb (fun x => bid_eqb id (fst (fst x)) && writes (snd x)) l = true.
Proof.
  intros id off sp l Hin Hw. apply existsb_exists. exists ((id, off), sp).
  split; [exact Hin |]. cbn [fst snd]. rewrite bid_eqb_refl, Hw. reflexivity.
Qed.

(* A buffer that the launch writes, and passes with no index range, is raw after it. *)
Lemma after_unindexed : forall sizes id (l : list ((bid * Z) * bspan)) st,
    (forall off sp, In ((id, off), sp) l -> bs_index sp = None) ->
    existsb (fun x => bid_eqb id (fst (fst x)) && writes (snd x)) l = true ->
    fold_left (after_write sizes id) l st = SRaw.
Proof.
  intros sizes id l. induction l as [| [[i o] sp] l IH]; intros st Hn He; [discriminate He |].
  cbn [fold_left]. cbn [existsb fst snd] in He.
  assert (Hn' : forall off' sp', In ((id, off'), sp') l -> bs_index sp' = None)
    by (intros off' sp' Hin; apply (Hn off'); right; exact Hin).
  destruct (existsb (fun x => bid_eqb id (fst (fst x)) && writes (snd x)) l) eqn:Ht.
  - exact (IH _ Hn' eq_refl).
  - rewrite orb_false_r in He. apply andb_true_iff in He. destruct He as [Hid Hw]. apply bid_eqb_eq in Hid.
    subst i. rewrite after_unwritten.
    + unfold after_write. cbn [fst snd]. rewrite bid_eqb_refl, Hw. unfold span_status.
      rewrite (Hn o sp (or_introl eq_refl)). reflexivity.
    + intros off' sp' Hin. destruct (writes sp') eqn:Hw'; [| reflexivity].
      pose proof (existsb_false _ _ _ Ht ((id, off'), sp') Hin) as E. cbn [fst snd] in E.
      rewrite bid_eqb_refl, Hw' in E. discriminate E.
Qed.

Lemma NoDup_map_eq : forall (A B : Type) (f : A -> B) l x y,
    NoDup (map f l) -> In x l -> In y l -> f x = f y -> x = y.
Proof.
  intros A B f l. induction l as [| z l IH]; intros x y Hn Hx Hy E; [destruct Hx |].
  cbn [map] in Hn. inversion Hn as [| ? ? Hz Hl]; subst.
  destruct Hx as [<- | Hx], Hy as [<- | Hy]; [reflexivity | | | exact (IH x y Hl Hx Hy E)].
  - exfalso. apply Hz. rewrite E. apply in_map. exact Hy.
  - exfalso. apply Hz. rewrite <- E. apply in_map. exact Hx.
Qed.

Lemma find_buffer_of : forall c b, inv c -> In b (c_buffers c) -> find_buffer c (bf_id b) = Some b.
Proof.
  intros c b [_ [Hn _]] Hb. unfold find_buffer.
  destruct (find (fun x => bid_eqb (bf_id x) (bf_id b)) (c_buffers c)) as [b0 |] eqn:E.
  - apply find_some in E. destruct E as [Hb0 E]. apply bid_eqb_eq in E. f_equal.
    exact (NoDup_map_eq _ _ bf_id _ _ _ Hn Hb0 Hb E).
  - exfalso. apply (find_none _ _ E) in Hb. rewrite bid_eqb_refl in Hb. discriminate Hb.
Qed.

(* ------------------------------------------------------------------ *)
(* Worlds and runs.                                                    *)

Section Model.

(* The pool's device address. *)
Variable base : Z.
(* The program and launch facts of each verified operator: what `front`
   gives its kernel, which `verify_ptx_sound` states the check accepts. *)
Variable kernels : Z -> option (program * lfacts).

Record world := { w_cat : catalog; w_mem : memory }.

(* What the hardware chooses for a launch: the threads, each thread's
   indices and platform values, the oracles, and the schedule. The oracles
   are functions of the bytes of the launch's own shared variables and of
   its local memory (StoreLaunch.v). The shared memory that an earlier
   launch left in each block is a separate input of the run, which the
   choice leaves open. *)
Record choice := {
  ch_count : nat;
  ch_threads : nat -> thread_regs;
  ch_oracles : nat -> oracle;
  ch_sched : list nat
}.

Definition inits_of (F : lfacts) (Mf : binding) (c : catalog) (gx gy gz : Z) (buffers : list (bid * Z)) (sizes scalars : list Z)
    (ch : choice) (n : nat) : reg -> Z :=
  launch_value F Mf c gx gy gz buffers sizes scalars base (ch_threads ch n).

(* The block of a thread: its block indices, numbered across the grid. *)
Definition block_of (gx gy : Z) (th : thread_regs) : nat :=
  Z.to_nat (tr_ctaid th 0 + gx * (tr_ctaid th 1 + gy * tr_ctaid th 2)).

(* The hardware's facts of a launch: each thread's indices lie in the block
   and the grid, the platform's values satisfy `platform_ok` and are the
   same for the threads of a block, and every thread index of a block is a
   thread of the launch. *)
Record hardware_ok (F : lfacts) (Mf : binding) (c : catalog) (gx gy gz : Z) (buffers : list (bid * Z)) (sizes scalars : list Z)
    (ch : choice) : Prop := {
  hw_thread : forall n, thread_in Mf gx gy gz (ch_threads ch n);
  hw_platform : forall n, platform_ok F Mf c base (inits_of F Mf c gx gy gz buffers sizes scalars ch n);
  hw_uniform : forall n m, block_of gx gy (ch_threads ch n) = block_of gx gy (ch_threads ch m) ->
               forall r, tr_platform (ch_threads ch n) r = tr_platform (ch_threads ch m) r;
  hw_complete : forall n, (n < ch_count ch)%nat -> forall x y z,
                0 <= x < block_dim Mf 0 -> 0 <= y < block_dim Mf 1 -> 0 <= z < block_dim Mf 2 ->
                exists m, (m < ch_count ch)%nat /\
                          (forall d, tr_ctaid (ch_threads ch m) d = tr_ctaid (ch_threads ch n) d) /\
                          tr_tid (ch_threads ch m) 0 = x /\ tr_tid (ch_threads ch m) 1 = y /\ tr_tid (ch_threads ch m) 2 = z
}.

(* The barrier of the kernel's zeroing prologue, at which the threads of a
   block wait until every thread of the block has stored zero over its share
   of the shared variables. *)
Definition prologue_bar (F : lfacts) (Mf : binding) (prog : program) : option (nat * nat) :=
  match prologue_of F Mf prog with Some ps => Some (bar_of ps) | None => None end.

(* The launch, from the shared memory sh0 that each block starts with and
   the local memory lm0 that each thread starts with. *)
Definition kernel_run (F : lfacts) (prog : program) (Mf : binding) (c : catalog) (gx gy gz : Z)
    (buffers : list (bid * Z)) (sizes scalars : list Z) (ch : choice) (sh0 lm0 : nat -> memory) (m : memory) : memory :=
  g_mem (launch F prog (prologue_bar F Mf prog) (inits_of F Mf c gx gy gz buffers sizes scalars ch) (ch_oracles ch)
                (fun n => block_of gx gy (ch_threads ch n)) (ch_count ch) m sh0 lm0 (ch_sched ch)).

(* The local memory that the threads of a kernel start with: any contents
   when the kernel declares no local variables, since then the gate refuses
   every access to local memory; zero otherwise, the fact of decision 46
   that the prologue's stores of the local variables give. *)
Definition local_start (F : lfacts) (lm0 : nat -> memory) : Prop :=
  lf_locals F = [] \/ forall n, lm0 n = zero.

(* Every whole element of a check's bytes lies in its range in memory m. *)
Definition check_holds (m : memory) (k : check) : Prop :=
  forall j, 0 <= j -> j mod ck_elem k = 0 -> j + ck_elem k <= ck_bytes k ->
    ck_lo k <= extend (Z.to_nat (8 * ck_elem k)) (ck_signed k)
                      (word_of m (base + ck_offset k + j) (Z.to_nat (ck_elem k))) <= ck_hi k.

Definition checks_hold (m : memory) (checks : list check) : Prop := Forall (check_holds m) checks.

(* Memory after an effect of action a in catalog c. A kernel runs if its
   checks hold, from whatever shared memory the blocks start with and
   whatever local memory `local_start` allows the threads, and is skipped
   otherwise. *)
Inductive effect_run (ch : choice) (c : catalog) : action -> memory -> effect -> memory -> Prop :=
| RunClear : forall a m o n, effect_run ch c a m (Clear o n) (write_bytes m (base + o) n 0)
| RunUpload : forall a m o data, effect_run ch c a m (Upload o data) (upload_bytes m (base + o) data)
| RunMemset : forall a m o n e s v, effect_run ch c a m (Memset o n e s v) (fill_bytes m (base + o) n e v)
| RunKernel : forall m op gx gy gz buffers sizes scalars guarded args checks Mf prog F sh0 lm0,
    find_binding c op = Some Mf -> kernels op = Some (prog, F) ->
    hardware_ok F Mf c gx gy gz buffers sizes scalars ch -> checks_hold m checks -> local_start F lm0 ->
    effect_run ch c (Launch op gx gy gz buffers sizes scalars guarded) m (Kernel op gx gy gz (eval (b_shared Mf) sizes) args checks)
      (kernel_run F prog Mf c gx gy gz buffers sizes scalars ch sh0 lm0 m)
| RunSkipped : forall m op gx gy gz shared buffers sizes scalars guarded args checks,
    ~ checks_hold m checks ->
    effect_run ch c (Launch op gx gy gz buffers sizes scalars guarded) m (Kernel op gx gy gz shared args checks) m.

Inductive effects_run (ch : choice) (c : catalog) (a : action) : memory -> list effect -> memory -> Prop :=
| RunNone : forall m, effects_run ch c a m [] m
| RunMore : forall m e es m1 m2, effect_run ch c a m e m1 -> effects_run ch c a m1 es m2 ->
    effects_run ch c a m (e :: es) m2.

Inductive run_action (ch : choice) (t : Z) : world -> action -> world -> Prop :=
| RunAction : forall c m a c' es m', check_action c t a = inl (c', es) -> effects_run ch c a m es m' ->
    run_action ch t {| w_cat := c; w_mem := m |} a {| w_cat := c'; w_mem := m' |}.

Inductive run_actions (t : Z) : list choice -> world -> list action -> world -> Prop :=
| RunDone : forall w, run_actions t [] w [] w
| RunNext : forall ch chs w a w1 actions w2, run_action ch t w a w1 -> run_actions t chs w1 actions w2 ->
    run_actions t (ch :: chs) w (a :: actions) w2.

Inductive run_step (chs : list choice) : world -> step -> world -> Prop :=
| RunAccepted : forall w s c' es w', check_step (w_cat w) s = Accepted c' es ->
    run_actions (s_tenant s) chs w (s_actions s) w' -> run_step chs w s w'
| RunRefused : forall w s r, check_step (w_cat w) s = Refused r -> run_step chs w s w.

Inductive run_steps : list (list choice) -> world -> list step -> world -> Prop :=
| RunEnd : forall w, run_steps [] w [] w
| RunStep : forall chs chss w s w1 ss w2, run_step chs w s w1 -> run_steps chss w1 ss w2 ->
    run_steps (chs :: chss) w (s :: ss) w2.

Lemma effects_nil : forall ch c a m m', effects_run ch c a m [] m' -> m' = m.
Proof. intros ch c a m m' H. inversion H. reflexivity. Qed.

Lemma effects_one : forall ch c a m e m', effects_run ch c a m [e] m' -> effect_run ch c a m e m'.
Proof.
  intros ch c a m e m' H. inversion H as [| ? ? ? ? ? He Hn]; subst. inversion Hn; subst. exact He.
Qed.

Lemma kernel_inv : forall ch c op gx gy gz shared buffers sizes scalars guarded m args checks m',
    effect_run ch c (Launch op gx gy gz buffers sizes scalars guarded) m (Kernel op gx gy gz shared args checks) m' ->
    (exists Mf prog F sh0 lm0, find_binding c op = Some Mf /\ kernels op = Some (prog, F) /\
       hardware_ok F Mf c gx gy gz buffers sizes scalars ch /\ checks_hold m checks /\ local_start F lm0 /\
       m' = kernel_run F prog Mf c gx gy gz buffers sizes scalars ch sh0 lm0 m) \/
    (~ checks_hold m checks /\ m' = m).
Proof.
  intros ch c op gx gy gz shared buffers sizes scalars guarded m args checks m' H.
  inversion H as [| | | ? ? ? ? ? ? ? ? ? ? ? Mf prog F sh0 lm0 Hb Hk Hhw Hck Hl | ? ? ? ? ? ? ? ? ? ? ? ? Hck]; subst.
  - left. exists Mf, prog, F, sh0, lm0. split; [exact Hb |]. split; [exact Hk |]. split; [exact Hhw |].
    split; [exact Hck |]. split; [exact Hl | reflexivity].
  - right. split; [exact Hck | reflexivity].
Qed.

(* ------------------------------------------------------------------ *)
(* The world's invariants.                                             *)

Definition in_buffer (b : buffer) (y : Z) : Prop :=
  base + bf_offset b <= y < base + bf_offset b + bf_bytes b.

Definition status_holds (m : memory) (b : buffer) : Prop :=
  match bf_status b with
  | SChecked e s lo hi => elems_in base m b e s lo hi
  | SRaw => True
  end.

Definition status_ok (c : catalog) (m : memory) : Prop := forall b, In b (c_buffers c) -> status_holds m b.

Definition verified (c : catalog) : Prop :=
  forall op Mf, find_binding c op = Some Mf ->
    exists prog F cert, kernels op = Some (prog, F) /\ check_scalar_certificate F prog cert = true /\ link_ok F Mf = true /\
      prologue_ok F Mf prog = true.

(* A catalog is verified when the gate accepted, for each operator with a
   binding, a text with its certificate (`check_text_certificate_ok`, what
   the service runs at registration), the binding is the one checked for
   the text's entry kernel, and the kernel table holds the program and
   facts that `front` gives that kernel. *)
Theorem verified_of_accepted : forall c,
    (forall op Mf, find_binding c op = Some Mf ->
       exists a n mfs certs ks name K,
         GateParse.check_text_certificate_ok a n mfs certs = true /\
         GateParse.parse_text a n = Ok ks /\ In (name, K) (GateParse.ks_entries ks) /\
         GateParse.lookup mfs name = Some Mf /\
         kernels op = match front K Mf with Ok pf => Some pf | Err _ _ => None end) ->
    verified c.
Proof.
  intros c H op Mf Hb.
  destruct (H op Mf Hb) as (a & n & mfs & certs & ks & name & K & Hok & Hp & Hin & Hl & Hk).
  destruct (GateParse.verify_certificate_sound a n mfs certs Hok) as [ks' [Hp' Hall]].
  rewrite Hp in Hp'. injection Hp' as <-.
  destruct (Hall name K Hin) as [mf [cert [Hl' [prog [F [Hf [Hc [_ [_ [Hlink Hpro]]]]]]]]]].
  rewrite Hl in Hl'. injection Hl' as <-. rewrite Hf in Hk.
  exists prog, F, cert. split; [exact Hk |]. split; [exact Hc |]. split; [exact Hlink | exact Hpro].
Qed.

Definition base_ok (c : catalog) : Prop := 0 <= base /\ base mod alignment = 0 /\ base + c_pool c <= 2 ^ 64.

Definition world_ok (w : world) : Prop :=
  inv (w_cat w) /\ status_ok (w_cat w) (w_mem w) /\ verified (w_cat w) /\ base_ok (w_cat w).

Lemma elems_frame : forall m m' b e s lo hi, (forall y, in_buffer b y -> m' y = m y) ->
    elems_in base m b e s lo hi -> elems_in base m' b e s lo hi.
Proof.
  intros m m' b e s lo hi H He k Hk Hkm Hkl. rewrite (word_agree m' m); [exact (He k Hk Hkm Hkl) |].
  intros y Hy. apply H. unfold in_buffer. lia.
Qed.

Lemma status_frame : forall m m' b, (forall y, in_buffer b y -> m' y = m y) -> status_holds m b -> status_holds m' b.
Proof.
  intros m m' b H. unfold status_holds. destruct (bf_status b); [trivial | apply elems_frame; exact H].
Qed.

(* A catalog after set_status: the other buffers keep their bytes, and the
   buffer's own bytes hold its new status. *)
Lemma status_set : forall c m m' id st, status_ok c m ->
    (forall b, In b (c_buffers c) -> bf_id b <> id -> forall y, in_buffer b y -> m' y = m y) ->
    (forall b, In b (c_buffers c) -> bf_id b = id -> status_holds m' (with_status b st)) ->
    status_ok (set_status c id st) m'.
Proof.
  intros c m m' id st Hs Hother Hself b' Hb'. unfold set_status in Hb'. cbn [c_buffers] in Hb'.
  apply in_map_iff in Hb'. destruct Hb' as [b [<- Hb]].
  destruct (bid_eqb_spec (bf_id b) id) as [E | E].
  - exact (Hself b Hb E).
  - apply (status_frame m); [exact (Hother b Hb E) | exact (Hs b Hb)].
Qed.

(* ------------------------------------------------------------------ *)
(* A launch.                                                           *)

Section Kernel.

Variable ch : choice.
Variable c : catalog.
Variable t : Z.
Variables op gx gy gz : Z.
Variable buffers : list (bid * Z).
Variables sizes scalars : list Z.
Variable guarded : bool.
Variable Mf : binding.
Variable prog : program.
Variable F : lfacts.

Hypothesis Hinv : inv c.
Hypothesis Hbase : base_ok c.
Hypothesis Hacc : exists r, check_launch c t op gx gy gz buffers sizes scalars guarded = inl r.
Hypothesis Hbind : find_binding c op = Some Mf.
Hypothesis Hsafe : kernel_safe F prog.
Hypothesis Hlf : lf_ok F = true.
Hypothesis Hlink : link_ok F Mf = true.
Hypothesis Hpro : prologue_ok F Mf prog = true.
Hypothesis Hhw : hardware_ok F Mf c gx gy gz buffers sizes scalars ch.
(* The shared memory that each block starts with, and the local memory
   that each thread starts with. *)
Variable sh0 lm0 : nat -> memory.

Let inits := inits_of F Mf c gx gy gz buffers sizes scalars ch.
Let cta := fun n => block_of gx gy (ch_threads ch n).
Let P := indices_in Mf c buffers sizes base.
Let bar := prologue_bar F Mf prog.
Let run := kernel_run F prog Mf c gx gy gz buffers sizes scalars ch sh0 lm0.
Let Hth : forall n, thread_in Mf gx gy gz (ch_threads ch n) := hw_thread F Mf c gx gy gz buffers sizes scalars ch Hhw.
Let Hpf : forall n, platform_ok F Mf c base (inits n) := hw_platform F Mf c gx gy gz buffers sizes scalars ch Hhw.

(* The bytes of the buffers the launch writes. *)
Definition launch_writes_at (y : Z) : Prop :=
  exists id off sp b, In ((id, off), sp) (pairs Mf buffers) /\ writes sp = true /\ find_buffer c id = Some b /\
    in_buffer b y.

Lemma kernel_link : forall n hist sh lh, (forall t', P (hist t')) ->
    launch_ok F (env_of F (inits n) (ch_oracles ch n) hist sh lh).
Proof.
  intros n hist sh lh H.
  exact (link_sound F Mf c t op gx gy gz buffers sizes scalars guarded base Hlf Hlink Hinv Hbind Hacc Hbase
           (ch_threads ch n) (ch_oracles ch n) hist sh lh (Hth n) (Hpf n) H).
Qed.

Lemma kernel_keep : forall n hist sh lh t' k x width v wr m,
    (forall t'', P (hist t'')) -> GateMeaning.region F (env_of F (inits n) (ch_oracles ch n) hist sh lh) t' k x width v ->
    k <> KLoad -> P m -> P (write_bytes m x width (match v with Some z => z | None => wr end)).
Proof.
  intros n hist sh lh t' k x width v wr m _ H Hk Hm.
  exact (indices_keep F Mf c t op gx gy gz buffers sizes scalars guarded base Hlink Hinv Hbind Hacc Hbase
           (ch_threads ch n) (ch_oracles ch n) hist sh lh t' k x width v wr m (Hpf n) H Hk Hm).
Qed.

Lemma kernel_region : forall n hist sh lh t' k x width v,
    GateMeaning.region F (env_of F (inits n) (ch_oracles ch n) hist sh lh) t' k x width v ->
    forall y, x <= y < x + width ->
    exists id off sp b, In ((id, off), sp) (pairs Mf buffers) /\ find_buffer c id = Some b /\
      arg_ok c t guarded buffers (spans Mf) sizes ((id, off), sp) /\ allows (bs_dir sp) k = true /\ in_buffer b y.
Proof.
  intros n hist sh lh t' k x width v H y Hy.
  destruct (region_buffer F Mf c t op gx gy gz buffers sizes scalars guarded base Hlink Hinv Hbind Hacc Hbase
              (ch_threads ch n) (ch_oracles ch n) hist sh lh t' k x width v (Hpf n) H ltac:(lia))
    as [id [off [sp [b [e [Hin [Hb [Hok [-> [He [Hle [Hallow _]]]]]]]]]]]].
  exists id, off, sp, b. split; [exact Hin |]. split; [exact Hb |]. split; [exact Hok |].
  split; [exact Hallow |]. unfold in_buffer. lia.
Qed.

Lemma kernel_indices : forall m, P m -> P (run m).
Proof.
  intros m Hm.
  exact (proj1 (launch_inv F prog bar Hsafe inits (ch_oracles ch) cta P kernel_link kernel_keep (ch_count ch) m sh0
                           lm0 (ch_sched ch) Hm)).
Qed.

Lemma kernel_frame : forall m, P m -> forall y, ~ launch_writes_at y -> run m y = m y.
Proof.
  intros m Hm y Hy.
  assert (Hw : forall n hist sh lh t' k x w v, (forall t'', P (hist t'')) ->
                 GateMeaning.region F (env_of F (inits n) (ch_oracles ch n) hist sh lh) t' k x w v -> k <> KLoad ->
                 forall y', x <= y' < x + w -> launch_writes_at y').
  { intros n hist sh lh t' k x w v _ H Hk y' Hy'.
    destruct (kernel_region n hist sh lh t' k x w v H y' Hy') as [id [off [sp [b [Hin [Hb [_ [Hallow Hib]]]]]]]].
    exists id, off, sp, b. split; [exact Hin |]. split; [exact (allows_writes sp k Hallow Hk) |].
    split; [exact Hb | exact Hib]. }
  exact (launch_frame F prog bar Hsafe inits (ch_oracles ch) cta P kernel_link kernel_keep launch_writes_at Hw
           (ch_count ch) m sh0 lm0 (ch_sched ch) Hm y Hy).
Qed.

(* ------------------------------------------------------------------ *)
(* The launch values, as the zeroing prologue needs them.             *)

Lemma kernel_launch_ok : forall m, P m -> forall n,
    launch_ok F (env_of F (inits n) (ch_oracles ch n) (fun _ => m) (fun _ => zero) (fun _ => zero)).
Proof. intros m Hm n. apply kernel_link. intros _. exact Hm. Qed.

Lemma kernel_bounds : forall m, P m -> forall n r lo hi, is_fixed F r = true -> lookup (lf_bounds F) r = Some (lo, hi) ->
    lo <= inits n r <= hi.
Proof. intros m Hm n r lo hi Hfix Hl. exact (lk_bounds F _ (kernel_launch_ok m Hm n) r lo hi Hfix Hl). Qed.

Lemma kernel_values : forall m, P m -> forall n r, 0 <= inits n r < 2 ^ 64.
Proof.
  intros m Hm n r. pose proof (kernel_launch_ok m Hm n) as Hk. pose proof (env_of_launch F prog _ Hlf Hk) as Hok.
  pose proof (lk_init F _ Hk r) as Hi. cbn [env_of e_init] in Hi.
  assert (H64 : 2 ^ wdz F r <= 2 ^ 64).
  { apply Z.pow_le_mono_r; [lia |]. pose proof (ok_widths F _ Hok r). lia. }
  lia.
Qed.

Lemma kernel_windows : forall n, inits n (lf_shared_window F) + window_size <= inits n (lf_local_window F) \/
                                 inits n (lf_local_window F) + window_size <= inits n (lf_shared_window F).
Proof. intros n. exact (pf_windows _ _ _ _ _ (Hpf n)). Qed.

(* The loop's count of elements times the width is the launch's dynamic
   shared memory, which the catalog bounds. *)
Lemma kernel_elements : forall ps zp, prologue_of F Mf prog = Some ps -> pr_loops ps = Some zp ->
    forall n, zero_elements zp (inits n) * pr_width ps = eval (b_shared Mf) sizes.
Proof.
  intros ps zp Hps Hl n. pose proof (prologue_shape F Mf prog ps Hps) as Hsh.
  destruct (sh_sizes _ _ _ _ Hsh zp Hl) as (Hterms & Hconst & Hmod & Hcoef & [ts Hts]).
  destruct (sh_dims _ _ _ _ Hsh) as (_ & _ & _ & _ & Hw & _).
  unfold zero_elements, eval, inits, inits_of. rewrite Hconst.
  rewrite <- (zero_terms_sum F Mf c t op gx gy gz buffers sizes scalars guarded base Hbind Hacc (ch_threads ch n)
                (pr_width ps) (af_terms (b_shared Mf)) (zp_terms zp) ts 0 ltac:(lia) Hterms Hts Hcoef).
  assert (Hc : af_const (b_shared Mf) = pr_width ps * (af_const (b_shared Mf) / pr_width ps))
    by (apply Z.div_exact; [lia | exact Hmod]).
  nia.
Qed.

Lemma kernel_shared : forall ps zp, prologue_of F Mf prog = Some ps -> pr_loops ps = Some zp ->
    forall n, 0 <= zero_elements zp (inits n) * zp_width zp <= shared_max.
Proof.
  intros ps zp Hps Hl n. pose proof (prologue_shape F Mf prog ps Hps) as Hsh.
  destruct (sh_loops _ _ _ _ Hsh zp Hl) as (_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hw & _).
  rewrite Hw, (kernel_elements ps zp Hps Hl n).
  exact (shared_bound Mf c t op gx gy gz buffers sizes scalars guarded Hbind Hacc).
Qed.

(* The threads of a block share the addresses and extents of its shared
   variables: the addresses are platform values, and the extents are
   constants or the launch's dynamic shared memory. *)
Lemma kernel_uniform : forall ps, prologue_of F Mf prog = Some ps -> forall n m, cta n = cta m ->
    (forall u f, In (u, f) (lf_shared F) -> inits n u = inits m u /\ fval F (inits n) f = fval F (inits m) f) /\
    (forall zp, pr_loops ps = Some zp -> zero_elements zp (inits n) = zero_elements zp (inits m)).
Proof.
  intros ps Hps n m Hc. pose proof (prologue_shape F Mf prog ps Hps) as Hsh.
  destruct (sh_dims _ _ _ _ Hsh) as (_ & _ & _ & _ & Hw & _).
  assert (Hel : forall zp, pr_loops ps = Some zp -> zero_elements zp (inits n) = zero_elements zp (inits m)).
  { intros zp Hl. apply (proj1 (Z.mul_cancel_r _ _ (pr_width ps) ltac:(lia))).
    rewrite (kernel_elements ps zp Hps Hl n), (kernel_elements ps zp Hps Hl m). reflexivity. }
  split; [| exact Hel]. intros u f Hin. split.
  - pose proof (sh_shared_cat _ _ _ _ Hsh u f Hin) as Hcat.
    unfold inits, inits_of, launch_value, raw_value. rewrite Hcat.
    rewrite (hw_uniform _ _ _ _ _ _ _ _ _ _ Hhw n m Hc u). reflexivity.
  - destruct (sh_covered _ _ _ _ Hsh u f Hin) as [[Hft _] | [(v & _ & _ & Hft & _) | (zp & Hl & _ & Hform)]].
    + unfold fval. rewrite Hft. reflexivity.
    + unfold fval. rewrite Hft. reflexivity.
    + unfold fval. rewrite (dynamic_form_value zp f (wdn F) (inits n) Hform),
        (dynamic_form_value zp f (wdn F) (inits m) Hform), (Hel zp Hl). reflexivity.
Qed.

Lemma bounded_below_fixed : forall r k, bounded_below F r k = true ->
    is_fixed F r = true /\ exists lo hi, lookup (lf_bounds F) r = Some (lo, hi).
Proof.
  intros r k H. unfold bounded_below in H. rewrite andb_true_iff in H. destruct H as [Hfix H]. split; [exact Hfix |].
  destruct (lookup (lf_bounds F) r) as [[lo hi] |]; [exists lo, hi; reflexivity | discriminate H].
Qed.

(* Every thread index of a block is a thread of the launch. *)
Lemma kernel_complete : forall ps, prologue_of F Mf prog = Some ps -> forall n T, (n < ch_count ch)%nat ->
    0 <= T < pr_threads ps -> exists m, (m < ch_count ch)%nat /\ cta m = cta n /\ thread_of ps (inits m) = T.
Proof.
  intros ps Hps n T Hn HT. pose proof (prologue_shape F Mf prog ps Hps) as Hsh.
  destruct (sh_dims _ _ _ _ Hsh) as (Hbx & Hby & Hbz & _).
  destruct (sh_tid_cat _ _ _ _ Hsh) as (Hcx & Hcy & Hcz).
  destruct (sh_tids _ _ _ _ Hsh) as (Htx & Hty & Htz).
  pose proof (sh_block _ _ _ _ Hsh) as Hblock.
  assert (Hdim : block_dim Mf 0 = pr_bx ps /\ block_dim Mf 1 = pr_by ps /\ block_dim Mf 2 = pr_bz ps)
    by (unfold block_dim; rewrite Hblock; repeat split; reflexivity).
  destruct Hdim as (Hd0 & Hd1 & Hd2).
  unfold pr_threads in HT.
  set (x := T mod pr_bx ps). set (y := (T / pr_bx ps) mod pr_by ps). set (z := T / (pr_bx ps * pr_by ps)).
  assert (Hx : 0 <= x < pr_bx ps) by (apply Z.mod_pos_bound; lia).
  assert (Hy : 0 <= y < pr_by ps) by (apply Z.mod_pos_bound; lia).
  assert (Hz : 0 <= z < pr_bz ps).
  { split; [apply Z.div_pos; nia |]. apply Z.div_lt_upper_bound; nia. }
  assert (HTxyz : (z * pr_by ps + y) * pr_bx ps + x = T).
  { unfold x, y, z. rewrite <- Z.div_div by lia.
    pose proof (Z.div_mod T (pr_bx ps) ltac:(lia)). pose proof (Z.div_mod (T / pr_bx ps) (pr_by ps) ltac:(lia)). nia. }
  destruct (hw_complete _ _ _ _ _ _ _ _ _ _ Hhw n Hn x y z ltac:(lia) ltac:(lia) ltac:(lia))
    as (m & Hm & Hcta & Hmx & Hmy & Hmz).
  exists m. split; [exact Hm |]. split.
  { unfold cta, block_of. rewrite !Hcta. reflexivity. }
  assert (Hval : forall r d k, category_reg F Mf r = Some (CTid d) -> bounded_below F r k = true ->
                   inits m r = tr_tid (ch_threads ch m) d).
  { intros r d k Hcat Hb. destruct (bounded_below_fixed r k Hb) as [Hfix [lo [hi Hl]]].
    exact (tid_value F Mf c gx gy gz buffers sizes scalars base Hlink (ch_threads ch m) r d lo hi (Hth m) Hcat Hfix Hl). }
  unfold thread_of. rewrite (Hval _ _ _ Hcx Htx), (Hval _ _ _ Hcy Hty), (Hval _ _ _ Hcz Htz), Hmx, Hmy, Hmz.
  exact HTxyz.
Qed.

(* Two runs of the launch from memories that agree on every byte of R, which
   holds every buffer its tenant may read, from any two shared memories, and
   from any two local memories that `local_start` allows, agree on R after
   it. *)
Lemma kernel_ni : forall (R : Z -> Prop) m1 m2 sh1 sh2 lm1 lm2, local_start F lm1 -> local_start F lm2 ->
    (forall b y, In b (c_buffers c) -> readable c t (bf_id b) = true -> in_buffer b y -> R y) ->
    P m1 -> P m2 -> (forall y, R y -> m1 y = m2 y) ->
    forall y, R y -> kernel_run F prog Mf c gx gy gz buffers sizes scalars ch sh1 lm1 m1 y =
                     kernel_run F prog Mf c gx gy gz buffers sizes scalars ch sh2 lm2 m2 y.
Proof.
  intros R m1 m2 sh1 sh2 lm1 lm2 Hl1 Hl2 HR Hm1 Hm2 Ha.
  (* the two local memories agree on the bytes of the local variables: there are none, or both are zero *)
  assert (Hlm : forall n, lm_agree F (inits n) (lm1 n) (lm2 n)).
  { intros n x [u [f [Hin _]]].
    destruct Hl1 as [Hnone | Hz1]; [unfold local_vars in Hin; rewrite Hnone in Hin; cbn in Hin; destruct Hin |].
    destruct Hl2 as [Hnone | Hz2]; [unfold local_vars in Hin; rewrite Hnone in Hin; cbn in Hin; destruct Hin |].
    rewrite Hz1, Hz2. reflexivity. }
  assert (Hr : forall n hist sh lh t' x w v, (forall t'', P (hist t'')) ->
                 GateMeaning.region F (env_of F (inits n) (ch_oracles ch n) hist sh lh) t' KLoad x w v ->
                 forall y, x <= y < x + w -> R y).
  { intros n hist sh lh t' x w v _ H y Hy.
    destruct (kernel_region n hist sh lh t' KLoad x w v H y Hy) as [id [off [sp [b [_ [Hb [Hok [Hallow Hib]]]]]]]].
    unfold arg_ok in Hok. destruct Hok as [b' [Hb' [_ [_ [_ [_ [Hrd _]]]]]]]. rewrite Hb in Hb'. injection Hb' as <-.
    destruct (find_buffer_in _ _ _ Hb) as [Hbin Hid]. subst id.
    exact (HR b y Hbin (Hrd (allows_reads sp Hallow)) Hib). }
  destruct (lf_shared F) as [| v vs] eqn:Hshared.
  - (* no shared variables: the launch reads no shared memory *)
    exact (ni0_mem _ _ _ _ _ _ (launch_ni0 F prog Hsafe inits (ch_oracles ch) cta (ch_count ch) P kernel_link kernel_keep
                              R Hr Hlf kernel_windows (prologue_bar F Mf prog) Hshared m1 m2 sh1 sh2 lm1 lm2
                              (ch_sched ch) Hm1 Hm2 Ha Hlm)).
  - (* the zeroing prologue stores zero over the shared variables before a thread reads them *)
    unfold prologue_ok in Hpro. rewrite Hshared in Hpro.
    destruct (prologue_of F Mf prog) as [ps |] eqn:Hps; [| discriminate Hpro].
    unfold kernel_run, prologue_bar. rewrite Hps.
    exact (ni_mem _ _ _ _ _ _ _ _ _ _
             (launch_ni F Mf prog Hsafe inits (ch_oracles ch) cta (ch_count ch) P kernel_link kernel_keep R Hr Hlf
                kernel_windows (Some (bar_of ps)) ps Hps eq_refl (kernel_bounds m1 Hm1)
                (fun n zp s c0 _ _ => kernel_values m1 Hm1 n s) (fun n zp Hl => kernel_shared ps zp Hps Hl n)
                (kernel_uniform ps Hps) (kernel_complete ps Hps) m1 m2 sh1 sh2 lm1 lm2 (ch_sched ch) Hm1 Hm2 Ha Hlm)).
Qed.

(* ------------------------------------------------------------------ *)
(* The words of a flag buffer.                                         *)

(* The word of 4 bytes at a is a word of a flag buffer that the launch
   passes, as the status buffer of the guard pass's checks. *)
Definition flag_word (a : Z) : Prop :=
  exists id off sp b k, In ((id, off), sp) (pairs Mf buffers) /\ bs_flag sp = true /\ find_buffer c id = Some b /\
    0 <= k /\ k mod 4 = 0 /\ k + 4 <= bf_bytes b /\ a = base + bf_offset b + k.

Let gstep_of := gstep F prog bar inits (ch_oracles ch) cta.

(* A thread whose instruction accesses memory is not at the barrier, whose
   instruction is a nop. *)
Lemma access_not_bar : forall s i win rs k x w v, fetch prog (blk s) (idx s) = Some i ->
    access_at win rs i = Some (k, x, w, v) -> at_bar bar s = false.
Proof.
  intros s i win rs k x w v Hf Ha. unfold bar, prologue_bar.
  destruct (prologue_of F Mf prog) as [ps |] eqn:Hps; [| reflexivity].
  destruct (at_bar (Some (bar_of ps)) s) eqn:Hb; [| reflexivity]. exfalso.
  apply (at_bar_spec _ _ _ eq_refl) in Hb.
  pose proof (fetch_bar F Mf prog ps (prologue_shape F Mf prog ps Hps)) as Hnop.
  rewrite <- Hb in Hnop. cbn [fst snd] in Hnop. rewrite Hf in Hnop. injection Hnop as ->.
  discriminate Ha.
Qed.

(* A step of a thread touches no byte of a flag word, or sets the word to
   1. *)
Lemma flag_step : forall a G n, flag_word a -> grid_inv F prog inits (ch_oracles ch) P G ->
    (forall y, a <= y < a + 4 -> g_mem (gstep_of G n) y = g_mem G y) \/ word_of (g_mem (gstep_of G n)) a 4 = 1.
Proof.
  intros a G n [id [off [sp [b [k [Hin [Hf [Hb [Hk0 [Hkm [Hkb ->]]]]]]]]]]] [Hm Hthreads].
  unfold gstep_of, gstep. destruct (nth_error (g_threads G) n) as [th |] eqn:En; [| left; intros y _; reflexivity].
  destruct (at_bar bar (th_state th) && negb (block_arrived bar cta G n)); [left; intros y _; reflexivity |].
  destruct (tstep F prog bar (inits n) (ch_oracles ch n) (g_mem G) (g_shared G (cta n)) th)
    as [[[th' m'] sm'] |] eqn:Et; [| left; intros y _; reflexivity].
  destruct (Hthreads n th En) as [Hok Hh].
  destruct (tstep_safe F prog bar Hsafe inits (ch_oracles ch) P kernel_link kernel_keep n (g_mem G) _ th th' m' sm'
              Hok Hh Hm Et)
    as [Hh' [i [_ [_ [Hreg [_ [Hm' _]]]]]]].
  cbn [g_mem]. rewrite Hm'. unfold mem_write.
  destruct (access_at (win_of F (inits n)) (regs (th_state th)) i) as [[[[kk x] w] v] |] eqn:Ea;
    [| left; intros y _; reflexivity].
  destruct kk; cbv beta iota; [left; intros y _; reflexivity | |];
    (destruct (flags_set F Mf c t op gx gy gz buffers sizes scalars guarded base Hlink Hinv Hbind Hacc Hbase
                 (ch_threads ch n) (ch_oracles ch n) (th_hist th') (th_shist th') (th_lhist th') (th_count th) _
                 x w v (o_wr (ch_oracles ch n) (th_count th) (shared_bytes F (inits n) (g_shared G (cta n)))
                              (local_bytes F (inits n) (th_local th)))
                 (g_mem G) id off sp b k (Hpf n) (Hreg _ _ _ _ eq_refl) ltac:(discriminate) Hin Hf Hb Hk0 Hkm Hkb)
       as [Hout | Hone]; [left; intros y Hy; apply write_outside; exact (Hout y Hy) | right; exact Hone]).
Qed.

(* A flag word that is 0 or 1 before steps of the launch is 0 or 1 after
   them, and one that is 1 stays 1. *)
Lemma flag_steps : forall a l G, flag_word a -> grid_inv F prog inits (ch_oracles ch) P G ->
    (word_of (g_mem G) a 4 = 0 \/ word_of (g_mem G) a 4 = 1) ->
    (word_of (g_mem (fold_left gstep_of l G)) a 4 = 0 \/ word_of (g_mem (fold_left gstep_of l G)) a 4 = 1) /\
    (word_of (g_mem G) a 4 = 1 -> word_of (g_mem (fold_left gstep_of l G)) a 4 = 1).
Proof.
  intros a l. induction l as [| n l IH]; intros G Ha HG H01; cbn [fold_left]; [split; [exact H01 | auto] |].
  assert (HG' : grid_inv F prog inits (ch_oracles ch) P (gstep_of G n))
    by exact (gstep_inv F prog bar Hsafe inits (ch_oracles ch) cta P kernel_link kernel_keep G n HG).
  destruct (flag_step a G n Ha HG) as [Hsame | Hone].
  - assert (Hw : word_of (g_mem (gstep_of G n)) a 4 = word_of (g_mem G) a 4)
      by (apply word_agree; intros y Hy; apply Hsame; cbn in Hy; lia).
    rewrite <- Hw in H01 |- *. exact (IH _ Ha HG' H01).
  - destruct (IH _ Ha HG' (or_intror Hone)) as [H1 H2]. split; [exact H1 |]. intros _. exact (H2 Hone).
Qed.

Let prefix (m : memory) (j : nat) : grid :=
  fold_left gstep_of (firstn j (ch_sched ch)) (grid0 inits (ch_count ch) m sh0 lm0).

(* After the launch, a flag word that was 0 or 1 before it is 0 or 1, and
   it is 1 if it was 1 after any prefix of the launch's steps. *)
Theorem kernel_flag : forall m a, P m -> flag_word a ->
    (word_of m a 4 = 0 \/ word_of m a 4 = 1) ->
    (word_of (run m) a 4 = 0 \/ word_of (run m) a 4 = 1) /\
    (forall j, word_of (g_mem (prefix m j)) a 4 = 1 -> word_of (run m) a 4 = 1).
Proof.
  intros m a Hm Ha H01.
  assert (Hrun : forall j, run m = g_mem (fold_left gstep_of (skipn j (ch_sched ch)) (prefix m j))).
  { intros j. unfold run, kernel_run, launch, prefix, gstep_of, bar. rewrite <- fold_left_app, firstn_skipn. reflexivity. }
  assert (Hpre : forall j, grid_inv F prog inits (ch_oracles ch) P (prefix m j))
    by (intros j; exact (launch_inv F prog bar Hsafe inits (ch_oracles ch) cta P kernel_link kernel_keep (ch_count ch) m sh0
                           lm0 (firstn j (ch_sched ch)) Hm)).
  split.
  - rewrite (Hrun 0%nat). apply (flag_steps a _ _ Ha (Hpre 0%nat)). exact H01.
  - intros j Hj. rewrite (Hrun j).
    destruct (flag_steps a (firstn j (ch_sched ch)) (grid0 inits (ch_count ch) m sh0 lm0) Ha
                (grid0_inv F prog inits (ch_oracles ch) P (ch_count ch) m sh0 lm0 Hm) H01) as [Hj01 _].
    exact (proj2 (flag_steps a _ _ Ha (Hpre j) Hj01) Hj).
Qed.

(* A step of a thread whose write touches a byte of a flag word sets the
   word to 1: after the launch, the word is 1 if any such store ran. *)
Theorem kernel_flag_store : forall m a j G n th i k x w v,
    P m -> flag_word a ->
    prefix m j = G -> nth_error (g_threads G) n = Some th ->
    fetch prog (blk (th_state th)) (idx (th_state th)) = Some i ->
    access_at (win_of F (inits n)) (regs (th_state th)) i = Some (k, x, w, v) -> k <> KLoad ->
    (exists y, a <= y < a + 4 /\ x <= y < x + w) ->
    (exists s', tstep F prog bar (inits n) (ch_oracles ch n) (g_mem G) (g_shared G (cta n)) th = Some s') ->
    word_of (g_mem (gstep_of G n)) a 4 = 1.
Proof.
  intros m a j G n th i k x w v Hm Ha HG En Hf Ea Hk [y [Hya Hyx]] [[[th' m'] sm'] Et].
  assert (Hinv' : grid_inv F prog inits (ch_oracles ch) P G)
    by (rewrite <- HG; exact (launch_inv F prog bar Hsafe inits (ch_oracles ch) cta P kernel_link kernel_keep (ch_count ch) m
                               sh0 lm0 (firstn j (ch_sched ch)) Hm)).
  destruct Ha as [id [off [sp [b [kk [Hin [Hfl [Hb [Hk0 [Hkm [Hkb ->]]]]]]]]]]].
  destruct Hinv' as [HmG Hthreads]. destruct (Hthreads n th En) as [Hok Hh].
  destruct (tstep_safe F prog bar Hsafe inits (ch_oracles ch) P kernel_link kernel_keep n (g_mem G) _ th th' m' sm'
              Hok Hh HmG Et)
    as [Hh' [i' [Hf' [_ [Hreg [_ [Hm' _]]]]]]].
  rewrite Hf in Hf'. injection Hf' as <-.
  unfold gstep_of, gstep. rewrite En. cbv beta iota.
  rewrite (access_not_bar _ _ _ _ _ _ _ _ Hf Ea). cbn [andb]. cbv iota. rewrite Et. cbv beta iota. cbn [g_mem].
  rewrite Hm'. unfold mem_write. rewrite Ea.
  destruct k; [contradiction | |]; cbv beta iota;
    (destruct (flags_set F Mf c t op gx gy gz buffers sizes scalars guarded base Hlink Hinv Hbind Hacc Hbase
                 (ch_threads ch n) (ch_oracles ch n) (th_hist th') (th_shist th') (th_lhist th') (th_count th) _
                 x w v (o_wr (ch_oracles ch n) (th_count th) (shared_bytes F (inits n) (g_shared G (cta n)))
                              (local_bytes F (inits n) (th_local th)))
                 (g_mem G) id off sp b kk (Hpf n) (Hreg _ _ _ _ Ea) ltac:(discriminate) Hin Hfl Hb Hk0 Hkm Hkb)
       as [Hout | Hone]; [exfalso; exact (Hout y Hya Hyx) | exact Hone]).
Qed.

Let checks := checks_of c guarded sizes (combine buffers (spans Mf)).

(* A buffer that the catalog checked inside an index range holds elements
   in the range. *)
Lemma index_elems : forall m b ir, status_ok c m -> In b (c_buffers c) -> index_ok b ir sizes = true ->
    elems_in base m b (ir_elem ir) (ir_signed ir) (eval (ir_lo ir) sizes) (eval (ir_hi ir) sizes).
Proof.
  intros m b ir Hs Hb Hidx. unfold index_ok in Hidx. specialize (Hs b Hb). unfold status_holds in Hs.
  destruct (bf_status b) as [| e s lo hi]; [discriminate Hidx |].
  rewrite !andb_true_iff, Z.eqb_eq, !Z.leb_le in Hidx. destruct Hidx as [[[He Hsg] Hlo] Hhi].
  apply Bool.eqb_prop in Hsg. subst e s.
  intros k Hk Hkm Hkl. pose proof (Hs k Hk Hkm Hkl). lia.
Qed.

(* Before the launch runs, every buffer it takes with an index range holds
   elements in the range: the catalog checked the buffer inside it, or the
   launch checks it when it runs. *)
Lemma status_indices : forall m, status_ok c m -> checks_hold m checks -> P m.
Proof.
  intros m Hs Hck id off sp ir b Hin Hix Hb.
  pose proof (pair_ok Mf c t op gx gy gz buffers sizes scalars guarded Hbind Hacc id off sp Hin) as Hok.
  unfold arg_ok in Hok. destruct Hok as [b' [Hb' [_ [_ [_ [_ [_ [_ Hidx]]]]]]]]. rewrite Hb in Hb'. injection Hb' as <-.
  destruct (find_buffer_in _ _ _ Hb) as [Hbin _].
  destruct (Hidx ir Hix) as [Hok | Hrun]; [exact (index_elems m b ir Hs Hbin Hok) |].
  unfold checks_hold in Hck. rewrite Forall_forall in Hck.
  pose (k := {| ck_offset := bf_offset b; ck_bytes := bf_bytes b; ck_elem := ir_elem ir; ck_signed := ir_signed ir;
                ck_lo := eval (ir_lo ir) sizes; ck_hi := eval (ir_hi ir) sizes |}).
  assert (Hk : In k checks).
  { unfold checks, checks_of. apply in_flat_map. exists ((id, off), sp). split; [exact Hin |].
    rewrite Hb, Hix, Hrun. left. reflexivity. }
  exact (Hck k Hk).
Qed.

(* A launch whose checks fail changes no byte, and the statuses it leaves
   hold of the bytes before it: it writes an index output only if the
   catalog checked it inside its range. *)
Lemma skip_status : forall m, status_ok c m ->
    status_ok (fold_left (fun acc x => written acc sizes x) (combine buffers (spans Mf)) c) m.
Proof.
  intros m Hs b' Hb'. rewrite written_buffers in Hb'. apply in_map_iff in Hb'. destruct Hb' as [b [<- Hb]].
  destruct (existsb (fun x => bid_eqb (bf_id b) (fst (fst x)) && writes (snd x)) (combine buffers (spans Mf))) eqn:Ew.
  - apply existsb_exists in Ew. destruct Ew as [[[id off] sp] [Hin Ew]]. cbn [fst snd] in Ew.
    rewrite andb_true_iff, bid_eqb_eq in Ew. destruct Ew as [<- Hw].
    pose proof (pair_ok Mf c t op gx gy gz buffers sizes scalars guarded Hbind Hacc _ _ _ Hin) as Hok.
    unfold arg_ok in Hok. destruct Hok as [b1 [Hb1 [_ [_ [_ [_ [_ [Hwr Hidx]]]]]]]].
    destruct (Hwr Hw) as [_ [Hc | Hc]];
      [| rewrite (after_unindexed _ _ _ _ (fun o s H => proj1 (Hc o s H)) (written_exists _ _ _ _ Hin Hw)); exact I].
    rewrite (find_buffer_of c b Hinv Hb) in Hb1. injection Hb1 as <-.
    assert (Hu : forall off' sp', In ((bf_id b, off'), sp') (combine buffers (spans Mf)) -> sp' = sp)
      by (intros off' sp' H'; exact (proj2 (combine_once_view _ _ _ _ _ _ _ _ H' Hin Hc))).
    rewrite (after_written _ _ sp _ _ Hw Hu).
    replace (existsb (fun x => bid_eqb (bf_id b) (fst (fst x))) (combine buffers (spans Mf))) with true
      by (symmetry; apply existsb_exists; exists ((bf_id b, off), sp); split; [exact Hin | apply bid_eqb_refl]).
    unfold status_holds, span_status. cbn [bf_status with_status].
    destruct (bs_index sp) as [ir |] eqn:Ei; [| exact I].
    destruct (Hidx ir eq_refl) as [Hok | Hrun]; [exact (index_elems m b ir Hs Hb Hok) |].
    unfold checked_at_run in Hrun. rewrite Hw in Hrun. rewrite andb_false_r in Hrun. discriminate Hrun.
  - assert (Hno : forall off sp, In ((bf_id b, off), sp) (combine buffers (spans Mf)) -> writes sp = false).
    { intros off sp Hin. destruct (writes sp) eqn:Hw; [| reflexivity].
      pose proof (existsb_false _ _ _ Ew ((bf_id b, off), sp) Hin) as E. cbn [fst snd] in E.
      rewrite bid_eqb_refl, Hw in E. discriminate E. }
    rewrite (after_unwritten _ _ _ _ Hno), with_status_same. exact (Hs b Hb).
Qed.

(* The checks read only buffers the tenant may read. *)
Lemma checks_readable : forall k, In k checks ->
    exists b, In b (c_buffers c) /\ readable c t (bf_id b) = true /\
      ck_offset k = bf_offset b /\ ck_bytes k = bf_bytes b.
Proof.
  intros k Hk. unfold checks, checks_of in Hk. apply in_flat_map in Hk. destruct Hk as [[[id off] sp] [Hin Hk]].
  destruct (find_buffer c id) as [b |] eqn:Hb; [| destruct Hk].
  destruct (bs_index sp) as [ix |]; [| destruct Hk].
  destruct (checked_at_run guarded b sp ix sizes) eqn:Hrun; [| destruct Hk].
  destruct Hk as [<- | []]. cbn [ck_offset ck_bytes].
  pose proof (pair_ok Mf c t op gx gy gz buffers sizes scalars guarded Hbind Hacc id off sp Hin) as Hok.
  unfold arg_ok in Hok. destruct Hok as [b' [Hb' [_ [_ [_ [_ [Hrd _]]]]]]]. rewrite Hb in Hb'. injection Hb' as <-.
  destruct (find_buffer_in _ _ _ Hb) as [Hbin Hid]. subst id.
  exists b. split; [exact Hbin |]. split; [| split; reflexivity].
  apply Hrd. unfold checked_at_run in Hrun. rewrite !andb_true_iff, negb_true_iff in Hrun.
  destruct Hrun as [[_ Hw] _]. unfold reads. unfold writes in Hw. destruct (bs_dir sp); congruence.
Qed.

(* After the launch, every element of a checked buffer lies in its range. *)
Lemma kernel_status : forall m, status_ok c m -> checks_hold m checks ->
    status_ok (fold_left (fun acc x => written acc sizes x) (combine buffers (spans Mf)) c) (run m).
Proof.
  intros m Hs Hck. pose proof (status_indices m Hs Hck) as Hm.
  intros b' Hb'. rewrite written_buffers in Hb'. apply in_map_iff in Hb'. destruct Hb' as [b [<- Hb]].
  destruct (existsb (fun x => bid_eqb (bf_id b) (fst (fst x)) && writes (snd x)) (combine buffers (spans Mf))) eqn:Ew.
  - (* the launch writes b through its one span sp *)
    apply existsb_exists in Ew. destruct Ew as [[[id off] sp] [Hin Ew]]. cbn [fst snd] in Ew.
    rewrite andb_true_iff, bid_eqb_eq in Ew. destruct Ew as [<- Hw].
    pose proof (pair_ok Mf c t op gx gy gz buffers sizes scalars guarded Hbind Hacc _ _ _ Hin) as Hok.
    unfold arg_ok in Hok. destruct Hok as [b1 [_ [_ [_ [_ [_ [_ [Hwr _]]]]]]]].
    destruct (Hwr Hw) as [_ [Hc | Hc]];
      [| rewrite (after_unindexed _ _ _ _ (fun o s H => proj1 (Hc o s H)) (written_exists _ _ _ _ Hin Hw)); exact I].
    assert (Hu : forall off' sp', In ((bf_id b, off'), sp') (combine buffers (spans Mf)) -> sp' = sp)
      by (intros off' sp' H'; exact (proj2 (combine_once_view _ _ _ _ _ _ _ _ H' Hin Hc))).
    rewrite (after_written _ _ sp _ _ Hw Hu).
    replace (existsb (fun x => bid_eqb (bf_id b) (fst (fst x))) (combine buffers (spans Mf))) with true
      by (symmetry; apply existsb_exists; exists ((bf_id b, off), sp); split; [exact Hin | apply bid_eqb_refl]).
    unfold status_holds, span_status. cbn [bf_status with_status].
    destruct (bs_index sp) as [ir |] eqn:Ei; [| exact I].
    exact (kernel_indices m Hm (bf_id b) off sp ir b Hin Ei (find_buffer_of c b Hinv Hb)).
  - (* the launch does not write b, and changes none of its bytes *)
    assert (Hno : forall off sp, In ((bf_id b, off), sp) (combine buffers (spans Mf)) -> writes sp = false).
    { intros off sp Hin. destruct (writes sp) eqn:Hw; [| reflexivity].
      pose proof (existsb_false _ _ _ Ew ((bf_id b, off), sp) Hin) as E. cbn [fst snd] in E.
      rewrite bid_eqb_refl, Hw in E. discriminate E. }
    rewrite (after_unwritten _ _ _ _ Hno), with_status_same.
    apply (status_frame m); [| exact (Hs b Hb)].
    intros y Hy. apply kernel_frame; [exact Hm |]. intros [id [off [sp [b'' [Hin [Hw [Hb'' Hy'']]]]]]].
    destruct (bid_eq_dec (bf_id b) id) as [E | E].
    + rewrite <- E in Hin. rewrite (Hno off sp Hin) in Hw. discriminate Hw.
    + destruct (find_buffer_in _ _ _ Hb'') as [Hbin'' Hid''].
      destruct Hinv as [_ [_ [Hap _]]].
      assert (Hapart : apart b b'') by (apply Hap; [exact Hb | exact Hbin'' | congruence]).
      unfold in_buffer in Hy, Hy''. unfold apart in Hapart. lia.
Qed.

End Kernel.

Lemma check_holds_frame : forall m1 m2 k,
    (forall y, base + ck_offset k <= y < base + ck_offset k + ck_bytes k -> m1 y = m2 y) ->
    check_holds m1 k -> check_holds m2 k.
Proof.
  intros m1 m2 k H Hk j Hj Hjm Hjl. rewrite (word_agree m2 m1); [exact (Hk j Hj Hjm Hjl) |].
  intros y Hy. symmetry. apply H. lia.
Qed.

(* The facts of a launch that a world with its invariants runs: the kernel
   runs if its checks hold, and is skipped otherwise. *)
Lemma launch_facts : forall ch c t m op gx gy gz buffers sizes scalars guarded c' es m',
    inv c -> verified c ->
    check_action c t (Launch op gx gy gz buffers sizes scalars guarded) = inl (c', es) ->
    effects_run ch c (Launch op gx gy gz buffers sizes scalars guarded) m es m' ->
    exists Mf prog F, find_binding c op = Some Mf /\ kernels op = Some (prog, F) /\
      kernel_safe F prog /\ lf_ok F = true /\ link_ok F Mf = true /\ prologue_ok F Mf prog = true /\
      (exists r, check_launch c t op gx gy gz buffers sizes scalars guarded = inl r) /\
      c' = fold_left (fun acc x => written acc sizes x) (combine buffers (spans Mf)) c /\
      ((hardware_ok F Mf c gx gy gz buffers sizes scalars ch /\
        checks_hold m (checks_of c guarded sizes (combine buffers (spans Mf))) /\
        exists sh0 lm0, local_start F lm0 /\ m' = kernel_run F prog Mf c gx gy gz buffers sizes scalars ch sh0 lm0 m) \/
       (~ checks_hold m (checks_of c guarded sizes (combine buffers (spans Mf))) /\ m' = m)).
Proof.
  intros ch c t m op gx gy gz buffers sizes scalars guarded c' es m' Hinv Hver H Hr.
  destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [Hb [Hacc [Hc' ->]]]].
  apply effects_one in Hr.
  destruct (Hver op Mf Hb) as [prog [F [cert [Hk [Hcb [Hlink Hpro]]]]]].
  assert (Hlf : lf_ok F = true).
  { exact (check_scalar_certificate_lf_ok F prog cert Hcb). }
  exists Mf, prog, F. split; [exact Hb |]. split; [exact Hk |].
  split; [exact (check_scalar_certificate_sound F prog cert Hcb) |]. split; [exact Hlf |]. split; [exact Hlink |].
  split; [exact Hpro |]. split; [exact Hacc |]. split; [exact Hc' |].
  destruct (kernel_inv _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hr)
    as [[Mf' [prog' [F' [sh0 [lm0 [Hb' [Hk' [Hhw [Hck [Hl ->]]]]]]]]]] | [Hck ->]];
    [left | right; split; [exact Hck | reflexivity]].
  rewrite Hb in Hb'. injection Hb' as <-. rewrite Hk in Hk'. injection Hk' as <- <-.
  split; [exact Hhw |]. split; [exact Hck |]. exists sh0, lm0. split; [exact Hl | reflexivity].
Qed.

(* ------------------------------------------------------------------ *)
(* Every run keeps the invariants.                                     *)

Theorem run_action_ok : forall ch t w a w', world_ok w -> run_action ch t w a w' -> world_ok w'.
Proof.
  intros ch t w a w' Hw H. destruct H as [c m a c' es m' Hc He]. destruct Hw as [Hinv [Hs [Hver Hb]]].
  unfold world_ok. cbn [w_cat w_mem] in *.
  pose proof (check_action_inv _ _ _ _ _ Hinv Hc) as Hinv'.
  destruct (check_action_frame _ _ _ _ _ Hc) as [Hpool Hops].
  split; [exact Hinv' |]. split; [| split].
  2: { intros op Mf Hf. apply Hver. unfold find_binding in *. rewrite <- Hops. exact Hf. }
  2: { unfold base_ok. rewrite Hpool. exact Hb. }
  pose proof Hinv as [Hp [Hn [Ha Hg]]].
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - (* an allocation: the region lies apart from every live buffer, and the new buffer is raw *)
    destruct (allocate_next _ _ _ _ _ _ _ Hc) as [-> [Hes [Hov _]]].
    assert (Hkeep : forall b y, In b (c_buffers c) -> in_buffer b y -> m' y = m y).
    { destruct Hes as [[-> _] | ->].
      - rewrite (effects_nil _ _ _ _ _ He). reflexivity.
      - apply effects_one in He. inversion He; subst. intros b y Hb' Hy. apply write_outside. intros Hy'.
        pose proof (existsb_false _ _ _ Hov b Hb') as Ho.
        unfold overlap, in_buffer in *. rewrite andb_false_iff, !Z.ltb_ge in Ho. lia. }
    intros b Hb'. cbn [c_buffers] in Hb'. apply in_app_or in Hb'. destruct Hb' as [Hb' | [<- | []]]; [| exact I].
    apply (status_frame m); [intros y Hy; exact (Hkeep b y Hb' Hy) | exact (Hs b Hb')].
  - (* a free *)
    destruct (free_next _ _ _ _ _ Hc) as [fb [_ [_ [-> ->]]]]. rewrite (effects_nil _ _ _ _ _ He).
    intros b Hb'. cbn [c_buffers] in Hb'. apply filter_In in Hb'. exact (Hs b (proj1 Hb')).
  - (* a grant *)
    destruct (grant_next _ _ _ _ _ _ Hc) as [fb [_ [_ [_ [-> ->]]]]]. rewrite (effects_nil _ _ _ _ _ He).
    exact Hs.
  - (* a write: the data lies in the buffer, and checked data has its range *)
    destruct (write_next _ _ _ _ _ _ _ Hc) as [fb [Fb [-> [Hlen Hel]]]].
    apply effects_one in He. inversion He; subst.
    destruct (find_buffer_in _ _ _ Fb) as [Hfb Hid].
    assert (Hother : forall b, In b (c_buffers c) -> bf_id b <> id -> forall y, in_buffer b y ->
                       upload_bytes m (base + bf_offset fb) data y = m y).
    { intros b Hb' Hne y Hy. unfold upload_bytes.
      destruct ((base + bf_offset fb <=? y) && (y <? base + bf_offset fb + Z.of_nat (List.length data))) eqn:Ey;
        [| reflexivity].
      exfalso. apply andb_true_iff in Ey. rewrite Z.leb_le, Z.ltb_lt in Ey.
      assert (Hap : apart b fb) by (apply Ha; [exact Hb' | exact Hfb | congruence]).
      unfold apart, in_buffer in *. lia. }
    destruct element as [[e s] |].
    + destruct Hel as [Hwe [Hl [Hmod ->]]]. pose proof (width_pos e Hwe) as He1.
      apply (status_set _ m); [exact Hs | exact Hother |].
      intros b Hb' Hbid. assert (b = fb) by (apply (NoDup_map_eq _ _ bf_id _ _ _ Hn Hb' Hfb); congruence).
      subst b. unfold status_holds. cbn [bf_status with_status]. intros k Hk Hkm Hkl.
      cbn [bf_offset bf_bytes with_status] in *.
      replace (base + bf_offset fb + k) with (base + bf_offset fb + Z.of_nat (Z.to_nat k)) by lia.
      rewrite upload_word by lia.
      exact (data_range_in e s data _ (elements_at e s data k ltac:(lia) Hk Hkm ltac:(lia))).
    + rewrite Hel. apply (status_set _ m); [exact Hs | exact Hother |]. intros b _ _. exact I.
  - (* a fill: every element is the value *)
    destruct (fill_next _ _ _ _ _ _ _ _ Hc) as [fb [Fb [-> [Hwe [Hv [Hmod ->]]]]]].
    pose proof (width_pos e Hwe) as He1.
    apply effects_one in He. inversion He; subst.
    destruct (find_buffer_in _ _ _ Fb) as [Hfb Hid].
    apply (status_set _ m); [exact Hs | |].
    + intros b Hb' Hne y Hy. unfold fill_bytes.
      destruct ((base + bf_offset fb <=? y) && (y <? base + bf_offset fb + bf_bytes fb)) eqn:Ey; [| reflexivity].
      exfalso. apply andb_true_iff in Ey. rewrite Z.leb_le, Z.ltb_lt in Ey.
      assert (Hap : apart b fb) by (apply Ha; [exact Hb' | exact Hfb | congruence]).
      unfold apart, in_buffer in *. lia.
    + intros b Hb' Hbid. assert (b = fb) by (apply (NoDup_map_eq _ _ bf_id _ _ _ Hn Hb' Hfb); congruence).
      subst b. unfold status_holds. cbn [bf_status with_status]. intros k Hk Hkm Hkl.
      cbn [bf_offset bf_bytes with_status] in *.
      rewrite fill_word by lia. rewrite extend_value by (exact Hv || lia). lia.
  - (* a launch, which runs if its checks hold and changes no byte otherwise *)
    destruct (launch_facts ch c t m op gx gy gz buffers sizes scalars guarded c' es m' Hinv Hver Hc He)
      as [Mf [prog [F [Hbind [_ [Hsafe [Hlf [Hlink [_ [Hacc [-> [[Hhw [Hck [sh0 [lm0 [_ ->]]]]] | [_ ->]]]]]]]]]]]]].
    + exact (kernel_status ch c t op gx gy gz buffers sizes scalars guarded Mf prog F Hinv Hb Hacc Hbind Hsafe
               Hlf Hlink Hhw sh0 lm0 m Hs Hck).
    + exact (skip_status c t op gx gy gz buffers sizes scalars guarded Mf Hinv Hacc Hbind m Hs).
  - (* a reservation changes no buffer and no byte *)
    destruct (reserve_next _ _ _ _ _ _ Hc) as [_ [_ [_ [_ [_ [-> ->]]]]]].
    rewrite (effects_nil _ _ _ _ _ He). exact Hs.
  - (* nor does the end of one *)
    destruct (unreserve_next _ _ _ _ Hc) as [_ [_ [-> ->]]].
    rewrite (effects_nil _ _ _ _ _ He). exact Hs.
Qed.

Lemma run_actions_ok : forall t chs w actions w', world_ok w -> run_actions t chs w actions w' -> world_ok w'.
Proof.
  intros t chs w actions w' Hw H. induction H as [w | ch chs w a w1 actions w2 H1 _ IH]; [exact Hw |].
  apply IH. exact (run_action_ok _ _ _ _ _ Hw H1).
Qed.

Theorem run_steps_ok : forall chss w steps w', world_ok w -> run_steps chss w steps w' -> world_ok w'.
Proof.
  intros chss w steps w' Hw H. induction H as [w | chs chss w s w1 ss w2 H1 _ IH]; [exact Hw |].
  apply IH. destruct H1 as [w s c' es w' _ Hr | w s r _]; [exact (run_actions_ok _ _ _ _ _ Hw Hr) | exact Hw].
Qed.

(* ------------------------------------------------------------------ *)
(* What a launch writes and reads.                                     *)

(* A launch changes only bytes of buffers its tenant may write. *)
Theorem launch_writes : forall ch t c m op gx gy gz buffers sizes scalars guarded w',
    world_ok {| w_cat := c; w_mem := m |} ->
    run_action ch t {| w_cat := c; w_mem := m |} (Launch op gx gy gz buffers sizes scalars guarded) w' ->
    forall y, (forall b, In b (c_buffers c) -> writable c t (bf_id b) = true -> ~ in_buffer b y) ->
      w_mem w' y = m y.
Proof.
  intros ch t c m op gx gy gz buffers sizes scalars guarded w' [Hinv [Hs [Hver Hb]]] H y Hy.
  inversion H as [c0 m0 a c' es m' Hc He]; subst. cbn [w_mem].
  destruct (launch_facts ch c t m op gx gy gz buffers sizes scalars guarded c' es m' Hinv Hver Hc He)
    as [Mf [prog [F [Hbind [_ [Hsafe [Hlf [Hlink [_ [Hacc [_ [[Hhw [Hck [sh0 [lm0 [_ ->]]]]] | [_ ->]]]]]]]]]]]]];
    [| reflexivity].
  apply (kernel_frame ch c t op gx gy gz buffers sizes scalars guarded Mf prog F Hinv Hb Hacc Hbind Hsafe Hlf
           Hlink Hhw sh0 lm0);
    [exact (status_indices c t op gx gy gz buffers sizes scalars guarded Mf Hacc Hbind m Hs Hck) |].
  intros [id [off [sp [b [Hin [Hw [Hfb Hib]]]]]]].
  pose proof (pair_ok Mf c t op gx gy gz buffers sizes scalars guarded Hbind Hacc id off sp Hin) as Hok.
  unfold arg_ok in Hok. destruct Hok as [b' [Hb' [_ [_ [_ [_ [_ [Hwr _]]]]]]]]. rewrite Hfb in Hb'.
  injection Hb' as <-.
  destruct (find_buffer_in _ _ _ Hfb) as [Hbin Hid]. subst id.
  exact (Hy b Hbin (proj1 (Hwr Hw)) Hib).
Qed.

(* Two runs of a launch with the same choice, from memories that agree on
   every byte of R, which holds every buffer its tenant may read, and from
   any two shared memories and any two local memories that `local_start`
   allows, agree on R after it. *)
Theorem launch_reads : forall ch t c m1 m2 op gx gy gz buffers sizes scalars guarded w1' w2' (R : Z -> Prop),
    world_ok {| w_cat := c; w_mem := m1 |} -> world_ok {| w_cat := c; w_mem := m2 |} ->
    run_action ch t {| w_cat := c; w_mem := m1 |} (Launch op gx gy gz buffers sizes scalars guarded) w1' ->
    run_action ch t {| w_cat := c; w_mem := m2 |} (Launch op gx gy gz buffers sizes scalars guarded) w2' ->
    (forall b y, In b (c_buffers c) -> readable c t (bf_id b) = true -> in_buffer b y -> R y) ->
    (forall y, R y -> m1 y = m2 y) ->
    w_cat w1' = w_cat w2' /\ forall y, R y -> w_mem w1' y = w_mem w2' y.
Proof.
  intros ch t c m1 m2 op gx gy gz buffers sizes scalars guarded w1' w2' R [Hinv [Hs1 [Hver Hb]]] [_ [Hs2 _]]
    H1 H2 HR Ha.
  inversion H1 as [c0 m0 a c1 es1 m1' Hc1 He1]; subst.
  inversion H2 as [c0 m0 a c2 es2 m2' Hc2 He2]; subst. cbn [w_cat w_mem].
  rewrite Hc1 in Hc2. injection Hc2 as <- <-. split; [reflexivity |].
  destruct (launch_facts ch c t m1 op gx gy gz buffers sizes scalars guarded c1 es1 m1' Hinv Hver Hc1 He1)
    as [Mf [prog [F [Hbind [Hk [Hsafe [Hlf [Hlink [Hpro [Hacc [_ Hrun1]]]]]]]]]]].
  destruct (launch_facts ch c t m2 op gx gy gz buffers sizes scalars guarded c1 es1 m2' Hinv Hver Hc1 He2)
    as [Mf' [prog' [F' [Hbind' [Hk' [_ [_ [_ [_ [_ [_ Hrun2]]]]]]]]]]].
  rewrite Hbind in Hbind'. injection Hbind' as <-. rewrite Hk in Hk'. injection Hk' as <- <-.
  (* the checks read bytes of buffers the tenant may read, so both runs decide alike *)
  assert (Hsame : forall ma mb, (forall y, R y -> ma y = mb y) ->
                  checks_hold ma (checks_of c guarded sizes (combine buffers (spans Mf))) ->
                  checks_hold mb (checks_of c guarded sizes (combine buffers (spans Mf)))).
  { intros ma mb Hab Hck. unfold checks_hold in *. rewrite Forall_forall in *. intros k Hk0.
    destruct (checks_readable c t op gx gy gz buffers sizes scalars guarded Mf Hacc Hbind k Hk0)
      as [b [Hbin [Hrd [Ho Hby]]]].
    apply (check_holds_frame ma mb k); [| exact (Hck k Hk0)].
    intros y Hy. apply Hab. apply (HR b y Hbin Hrd). unfold in_buffer. lia. }
  destruct Hrun1 as [[Hhw [Hck1 [sh1 [lm1 [Hl1 ->]]]]] | [Hck1 ->]];
    destruct Hrun2 as [[_ [Hck2 [sh2 [lm2 [Hl2 ->]]]]] | [Hck2 ->]].
  - exact (kernel_ni ch c t op gx gy gz buffers sizes scalars guarded Mf prog F Hinv Hb Hacc Hbind Hsafe Hlf Hlink
             Hpro Hhw R m1 m2 sh1 sh2 lm1 lm2 Hl1 Hl2 HR
             (status_indices c t op gx gy gz buffers sizes scalars guarded Mf Hacc Hbind m1 Hs1 Hck1)
             (status_indices c t op gx gy gz buffers sizes scalars guarded Mf Hacc Hbind m2 Hs2 Hck2) Ha).
  - exfalso. exact (Hck2 (Hsame m1 m2 Ha Hck1)).
  - exfalso. apply Hck1. apply (Hsame m2 m1); [intros y Hy; symmetry; exact (Ha y Hy) | exact Hck2].
  - exact Ha.
Qed.

End Model.
