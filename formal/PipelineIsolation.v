(* PipelineIsolation.v: what a tenant observes of a pipeline depends only on
   what it observed before, whatever the other tenants do.

   Tenant A observes the bytes of each buffer it may read and of each freed
   region it last owned, which it gets back without a clear (`observable`).
   Proved: `tenant_isolation`, two runs whose steps of A are the same, each
   with the same choices of the hardware, from worlds whose own parts of A
   are equal (`own`: A's buffers, the grants on them, A's freed regions,
   and A's reservation) and whose memories agree on what A observes, end
   with equal own parts of A and with memories that agree on what A
   observes. The other tenants' steps, with the regions, sizes, written
   bytes, grids, and scalars of their actions and their reservations, the
   bytes of their buffers, and the hardware's choices for their launches
   may differ between the runs, and so may the order in which their steps
   interleave with A's. A holds its reservation through both runs, and no
   buffer of another tenant is granted to A, which is how a tenant shows A
   its data. `own_decision_same`: two catalogs with the same own part of A
   decide a step of A alike, the same refusal, or the same effects and own
   part after it. `tenant_isolation_same_steps`: the same for two runs of the
   same steps of every tenant from the same catalog, without the
   reservation's premises; the other tenants' data, and the hardware's
   choices for their launches, may differ. A launch's choices are its
   schedule, its threads' indices and the platform's addresses, and its
   oracles, which are functions of the step and of the bytes of the
   launch's shared and local variables (StoreLaunch.v). The shared memory
   that an earlier launch left in a block is any contents, and may differ
   between the runs: the gate's zeroing prologue stores zero over the
   launch's shared variables before a thread of the block reads them. The
   local memory that an earlier launch left in a thread is any contents
   for a kernel without local variables, since the gate refuses every
   access to local memory then, and zero for a kernel with them
   (`local_start` in StoreModel.v); a register that a launch value does not
   fill holds 0; the prologue's local stores and the gate's check of
   registers make these what the hardware runs.

   Each action of another tenant changes no byte that A observes after it,
   and A observes after it only bytes A observed before (`other_action`);
   each action of A, run with the same choice from memories that agree on
   what A observes, gives memories that agree on what A observes after it
   (`self_action`). Both rest on the invariants of StoreModel.v and on one
   more: every freed region lies apart from every live buffer
   (`freed_apart`). For runs whose other tenants differ, a further
   invariant (`res_inv`) keeps the reservations of two tenants apart and
   each buffer and freed region in its owner's reservation, with an id of
   its owner; so an action of A decides and runs on A's own part as on the
   whole catalog (`own_action`, `run_action_own`), and an action of another
   tenant leaves A's own part as it is (`own_other`). *)

From Coq Require Import ZArith.ZArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Lia.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateDomain.
Require Import OpGate.GateFront.
Require Import OpGate.CatalogStep.
Require Import OpGate.StoreLaunch.
Require Import OpGate.StoreLink.
Require Import OpGate.StoreModel.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Readers of a buffer after an action.                                *)

Lemma find_status_map : forall (g : buffer -> status) l x,
    find (fun b => bid_eqb (bf_id b) x) (map (fun b => with_status b (g b)) l) =
    option_map (fun b => with_status b (g b)) (find (fun b => bid_eqb (bf_id b) x) l).
Proof.
  intros g l x. induction l as [| b l IH]; [reflexivity |]. cbn [map find with_status bf_id].
  destruct (bid_eqb (bf_id b) x); [reflexivity | exact IH].
Qed.

Lemma set_status_map : forall c id st,
    c_buffers (set_status c id st) =
    map (fun b => with_status b (if bid_eqb (bf_id b) id then st else bf_status b)) (c_buffers c).
Proof.
  intros c id st. unfold set_status. cbn [c_buffers]. apply map_ext. intros b.
  destruct (bid_eqb (bf_id b) id); [reflexivity | symmetry; apply with_status_same].
Qed.

Lemma written_frame : forall sizes l c0,
    c_freed (fold_left (fun acc x => written acc sizes x) l c0) = c_freed c0 /\
    c_grants (fold_left (fun acc x => written acc sizes x) l c0) = c_grants c0.
Proof.
  intros sizes l. induction l as [| [[id off] sp] l IH]; intros c0; [split; reflexivity |]. cbn [fold_left].
  destruct (IH (written c0 sizes ((id, off), sp))) as [E1 E2]. rewrite E1, E2. unfold written.
  destruct (negb (writes sp)); [split; reflexivity |]. destruct (bs_index sp); split; reflexivity.
Qed.

(* A catalog whose buffers differ only in status has the same readers. *)
Lemma readable_restatus : forall c c' (g : buffer -> status),
    c_buffers c' = map (fun b => with_status b (g b)) (c_buffers c) -> c_grants c' = c_grants c ->
    forall t x, readable c' t x = readable c t x.
Proof.
  intros c c' g Hb Hg t x. unfold readable, find_buffer, has_grant. rewrite Hb, find_status_map, Hg.
  destruct (find _ (c_buffers c)); reflexivity.
Qed.

Lemma find_app : forall (A : Type) (f : A -> bool) l1 l2,
    find f (l1 ++ l2) = match find f l1 with Some x => Some x | None => find f l2 end.
Proof.
  intros A f l1 l2. induction l1 as [| x l1 IH]; [reflexivity |]. cbn [app find].
  destruct (f x); [reflexivity | exact IH].
Qed.

Lemma find_filter : forall (A : Type) (f p : A -> bool) l, (forall x, f x = true -> p x = true) ->
    find f (filter p l) = find f l.
Proof.
  intros A f p l H. induction l as [| x l IH]; [reflexivity |]. cbn [filter find].
  destruct (f x) eqn:Ef.
  - rewrite (H x Ef). cbn [find]. rewrite Ef. reflexivity.
  - destruct (p x); [cbn [find]; rewrite Ef |]; exact IH.
Qed.

Lemma existsb_filter : forall (A : Type) (f p : A -> bool) l, existsb f (filter p l) = true -> existsb f l = true.
Proof.
  intros A f p l H. apply existsb_exists in H. destruct H as [x [Hx E]]. apply filter_In in Hx.
  apply existsb_exists. exists x. split; [exact (proj1 Hx) | exact E].
Qed.

Lemma has_grant_live : forall c x t, inv c -> has_grant c x t = true -> find_buffer c x <> None.
Proof.
  intros c x t [_ [_ [_ Hg]]] H Hn. unfold has_grant in H. apply existsb_exists in H.
  destruct H as [g [Hin E]]. rewrite andb_true_iff, bid_eqb_eq, Z.eqb_eq in E. destruct E as [Ex _].
  destruct (Hg g Hin) as [b [Hb [Hid _]]]. unfold find_buffer in Hn.
  apply (find_none _ _ Hn) in Hb. rewrite Hid, Ex, bid_eqb_refl in Hb. discriminate Hb.
Qed.

Lemma carve_in : forall freed o n r', In r' (carve freed o n) ->
    exists r, In r freed /\ rg_owner r' = rg_owner r /\ rg_offset r <= rg_offset r' /\
      rg_offset r' + rg_bytes r' <= rg_offset r + rg_bytes r /\
      (rg_offset r' + rg_bytes r' <= o \/ o + n <= rg_offset r').
Proof.
  intros freed o n r' H. unfold carve in H. apply in_flat_map in H. destruct H as [r [Hr H]].
  exists r. split; [exact Hr |]. unfold overlap in H.
  destruct (Z.ltb (rg_offset r) (o + n) && Z.ltb o (rg_offset r + rg_bytes r)) eqn:E; cbn [negb] in H.
  - rewrite andb_true_iff, !Z.ltb_lt in E. apply in_app_or in H. destruct H as [H | H].
    + destruct (Z.ltb_spec (rg_offset r) o); [| destruct H].
      destruct H as [<- | []]. cbn [rg_owner rg_offset rg_bytes]. repeat split; lia.
    + destruct (Z.ltb_spec (o + n) (rg_offset r + rg_bytes r)); [| destruct H].
      destruct H as [<- | []]. cbn [rg_owner rg_offset rg_bytes]. repeat split; lia.
  - destruct H as [<- | []]. rewrite andb_false_iff, !Z.ltb_ge in E. repeat split; lia.
Qed.

Lemma fold_min_in : forall l x, fold_left Z.min l x = x \/ In (fold_left Z.min l x) l.
Proof.
  induction l as [| z l IH]; intros x; cbn [fold_left]; [left; reflexivity |].
  destruct (IH (Z.min x z)) as [E | E]; [rewrite E | right; right; exact E].
  destruct (Z.min_spec x z) as [[_ ->] | [_ ->]]; [left | right; left]; reflexivity.
Qed.

Lemma fold_max_in : forall l x, fold_left Z.max l x = x \/ In (fold_left Z.max l x) l.
Proof.
  induction l as [| z l IH]; intros x; cbn [fold_left]; [left; reflexivity |].
  destruct (IH (Z.max x z)) as [E | E]; [rewrite E | right; right; exact E].
  destruct (Z.max_spec x z) as [[_ ->] | [_ ->]]; [right; left | left]; reflexivity.
Qed.

(* A region after tenant t frees the bytes [o, o + n) is a region that does
   not touch them, or the merged region: it is t's, it holds the bytes, and
   each of its bytes is one of them or lies in a freed region of t. *)
Lemma release_in : forall freed o n t r', 0 <= n -> In r' (release freed o n t) ->
    (In r' freed /\ touches o n t r' = false) \/
    (rg_owner r' = t /\ rg_offset r' <= o /\ o + n <= rg_offset r' + rg_bytes r' /\
     forall y, rg_offset r' <= y < rg_offset r' + rg_bytes r' ->
       o <= y < o + n \/ exists r, In r freed /\ rg_owner r = t /\ rg_offset r <= y < rg_offset r + rg_bytes r).
Proof.
  intros freed o n t r' Hn H. unfold release in H. cbv zeta in H. apply in_app_or in H.
  destruct H as [H | [<- | []]].
  - left. apply filter_In in H. destruct H as [H E]. apply negb_true_iff in E. split; [exact H | exact E].
  - right. cbn [rg_owner rg_offset rg_bytes].
    set (near := filter (touches o n t) freed).
    set (lo := fold_left Z.min (map rg_offset near) o).
    set (hi := fold_left Z.max (map (fun r => rg_offset r + rg_bytes r) near) (o + n)).
    assert (Hlo : lo <= o) by (apply fold_min_le; left; reflexivity).
    assert (Hhi : o + n <= hi) by (apply fold_max_ge; left; reflexivity).
    split; [reflexivity |]. split; [exact Hlo |]. split; [lia |]. intros y Hy.
    destruct (Z_lt_le_dec y o) as [Hyo | Hyo].
    + (* below the bytes, lo is the offset of a region of t that ends at o *)
      destruct (fold_min_in (map rg_offset near) o) as [E | E];
        change (fold_left Z.min (map rg_offset near) o) with lo in E; [lia |].
      apply in_map_iff in E. destruct E as [r [Er Hr]]. apply filter_In in Hr. destruct Hr as [Hr Ht].
      unfold touches in Ht. rewrite andb_true_iff, orb_true_iff, !Z.eqb_eq in Ht.
      destruct Ht as [Ho [He | Hs]]; [| lia].
      right. exists r. split; [exact Hr |]. split; [exact Ho | lia].
    + destruct (Z_lt_le_dec y (o + n)) as [Hyn | Hyn]; [left; lia |].
      (* above the bytes, hi is the end of a region of t that starts at o + n *)
      destruct (fold_max_in (map (fun r => rg_offset r + rg_bytes r) near) (o + n)) as [E | E];
        change (fold_left Z.max (map (fun r => rg_offset r + rg_bytes r) near) (o + n)) with hi in E; [lia |].
      apply in_map_iff in E. destruct E as [r [Er Hr]]. apply filter_In in Hr. destruct Hr as [Hr Ht].
      unfold touches in Ht. rewrite andb_true_iff, orb_true_iff, !Z.eqb_eq in Ht.
      destruct Ht as [Ho [He | Hs]]; [lia |].
      right. exists r. split; [exact Hr |]. split; [exact Ho | lia].
Qed.

(* ------------------------------------------------------------------ *)
(* What a tenant observes.                                             *)

Section Isolation.

Variable base : Z.
Variable kernels : Z -> option (program * lfacts).

Definition in_region (r : region) (y : Z) : Prop := base + rg_offset r <= y < base + rg_offset r + rg_bytes r.

Definition observable (c : catalog) (A : Z) (y : Z) : Prop :=
  (exists b, In b (c_buffers c) /\ readable c A (bf_id b) = true /\ in_buffer base b y) \/
  (exists r, In r (c_freed c) /\ rg_owner r = A /\ in_region r y).

Definition freed_apart (c : catalog) : Prop :=
  forall r b, In r (c_freed c) -> In b (c_buffers c) ->
    rg_offset r + rg_bytes r <= bf_offset b \/ bf_offset b + bf_bytes b <= rg_offset r.

Definition sound (w : world) : Prop := world_ok base kernels w /\ freed_apart (w_cat w).

Definition agree_on (c : catalog) (A : Z) (m1 m2 : memory) : Prop := forall y, observable c A y -> m1 y = m2 y.

(* Bytes that A observes lie in no buffer that another tenant may write. *)
Lemma observable_not_writable : forall c A B y b', inv c -> freed_apart c -> A <> B ->
    observable c A y -> In b' (c_buffers c) -> writable c B (bf_id b') = true -> ~ in_buffer base b' y.
Proof.
  intros c A B y b' Hinv Hfa HAB Hobs Hb' Hw Hy.
  pose proof (find_buffer_of c b' Hinv Hb') as Hf'. unfold writable in Hw. rewrite Hf' in Hw.
  rewrite andb_true_iff, Z.eqb_eq, negb_true_iff in Hw. destruct Hw as [Hown Hng].
  destruct Hobs as [[b [Hb [Hr Hy']]] | [r [Hr [Hro Hy']]]].
  - destruct (bid_eq_dec (bf_id b) (bf_id b')) as [E | E].
    + rewrite E in Hr. unfold readable in Hr. rewrite Hf' in Hr. rewrite orb_true_iff, Z.eqb_eq in Hr.
      destruct Hr as [Ho | Hg]; [congruence |].
      unfold has_grant in Hg. apply existsb_exists in Hg. destruct Hg as [g [Hin Eg]].
      apply andb_true_iff in Eg. destruct Eg as [Eg _].
      unfold granted in Hng. rewrite (proj2 (existsb_exists _ _) (ex_intro _ g (conj Hin Eg))) in Hng.
      discriminate Hng.
    + destruct Hinv as [_ [_ [Hap _]]]. pose proof (Hap b b' Hb Hb' E) as Ha.
      unfold apart, in_buffer in *. lia.
  - pose proof (Hfa r b' Hr Hb') as Ha. unfold in_region, in_buffer in *. lia.
Qed.

(* After a write, a fill, or a launch, the buffers differ only in status. *)
Lemma observable_restatus : forall c c' (g : buffer -> status) A y,
    c_buffers c' = map (fun b => with_status b (g b)) (c_buffers c) -> c_grants c' = c_grants c ->
    c_freed c' = c_freed c -> observable c' A y -> observable c A y.
Proof.
  intros c c' g A y Hb Hg Hf [[b [Hb' [Hr Hy]]] | [r [Hr [Ho Hy]]]].
  - rewrite (readable_restatus c c' g Hb Hg) in Hr. rewrite Hb in Hb'. apply in_map_iff in Hb'.
    destruct Hb' as [b0 [<- Hb0]]. left. exists b0. split; [exact Hb0 |]. split; [exact Hr | exact Hy].
  - right. exists r. rewrite Hf in Hr. split; [exact Hr | split; [exact Ho | exact Hy]].
Qed.

(* After an allocation, A observes what it observed before, apart from the
   region, or the region itself if A allocated it. *)
Lemma observable_allocate : forall c t id o n A y, inv c -> find_buffer c id = None ->
    existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers c) = false ->
    observable {| c_pool := c_pool c;
                  c_buffers := c_buffers c ++ [{| bf_id := id; bf_offset := o; bf_bytes := n; bf_owner := t;
                                                  bf_status := SRaw |}];
                  c_grants := c_grants c; c_freed := carve (c_freed c) o n; c_reserved := c_reserved c;
                  c_operators := c_operators c |} A y ->
    (observable c A y /\ ~ (base + o <= y < base + o + n)) \/ (t = A /\ base + o <= y < base + o + n).
Proof.
  intros c t id o n A y Hinv Hnone Hov [[b [Hb [Hr Hy]]] | [r' [Hr' [Ho Hy]]]].
  - cbn [c_buffers] in Hb. unfold readable, find_buffer, has_grant in Hr. cbn [c_buffers c_grants] in Hr.
    rewrite find_app in Hr. apply in_app_or in Hb. destruct Hb as [Hb | [<- | []]].
    + left. pose proof (find_buffer_of c b Hinv Hb) as Hf. unfold find_buffer in Hf. rewrite Hf in Hr.
      split.
      * left. exists b. split; [exact Hb |]. split; [| exact Hy].
        unfold readable, find_buffer, has_grant. rewrite Hf. exact Hr.
      * pose proof (existsb_false _ _ _ Hov b Hb) as E. unfold overlap, in_buffer in *.
        rewrite andb_false_iff, !Z.ltb_ge in E. lia.
    + right. cbn [bf_id] in Hr. unfold find_buffer in Hnone. rewrite Hnone in Hr.
      cbn [find bf_id bf_owner] in Hr. rewrite bid_eqb_refl, orb_true_iff, Z.eqb_eq in Hr.
      unfold in_buffer in Hy. cbn [bf_offset bf_bytes] in Hy.
      destruct Hr as [Ht | Hg]; [split; [exact Ht | lia] |].
      exfalso. apply (has_grant_live c id A Hinv); [exact Hg | exact Hnone].
  - cbn [c_freed] in Hr'. destruct (carve_in _ _ _ _ Hr') as [r [Hr [Hro [Hlo [Hhi Hap]]]]].
    left. unfold in_region in Hy. split; [| lia].
    right. exists r. split; [exact Hr |]. split; [congruence |]. unfold in_region. lia.
Qed.

Lemma observable_free : forall c t id fb A y, inv c -> find_buffer c id = Some fb -> bf_owner fb = t ->
    observable {| c_pool := c_pool c; c_buffers := filter (fun x => negb (bid_eqb (bf_id x) id)) (c_buffers c);
                  c_grants := filter (fun g => negb (bid_eqb (fst g) id)) (c_grants c);
                  c_freed := release (c_freed c) (bf_offset fb) (bf_bytes fb) t; c_reserved := c_reserved c;
                  c_operators := c_operators c |} A y ->
    observable c A y.
Proof.
  intros c t id fb A y Hinv Hfb Ho [[b [Hb [Hr Hy]]] | [r [Hr [Hro Hy]]]].
  - cbn [c_buffers] in Hb. apply filter_In in Hb. destruct Hb as [Hb Hne].
    left. exists b. split; [exact Hb |]. split; [| exact Hy].
    unfold readable, find_buffer, has_grant in Hr |- *. cbn [c_buffers c_grants] in Hr.
    rewrite find_filter in Hr.
    + destruct (find _ (c_buffers c)) as [b0 |]; [| discriminate Hr].
      rewrite orb_true_iff in Hr |- *. destruct Hr as [Hr | Hr]; [left; exact Hr | right].
      exact (existsb_filter _ _ _ _ Hr).
    + intros x Ex. apply bid_eqb_eq in Ex. rewrite Ex. exact Hne.
  - destruct (find_buffer_in _ _ _ Hfb) as [Hin Hid].
    assert (Hn : 0 <= bf_bytes fb) by (destruct Hinv as [Hp _]; destruct (Hp fb Hin) as [_ [H1 _]]; lia).
    cbn [c_freed] in Hr. destruct (release_in _ _ _ _ _ Hn Hr) as [[Hr' _] | [Hrt [_ [_ Hcover]]]].
    + right. exists r. split; [exact Hr' | split; [exact Hro | exact Hy]].
    + (* the merged region is A's: each of its bytes lies in fb or in a region A freed before *)
      assert (Ht : t = A) by congruence. unfold in_region in Hy.
      destruct (Hcover (y - base) ltac:(lia)) as [Hyb | [r0 [Hr0 [Hro0 Hy0]]]].
      * left. exists fb. split; [exact Hin |]. split.
        -- unfold readable. rewrite Hid, Hfb, Ho, Ht, Z.eqb_refl. reflexivity.
        -- unfold in_buffer. lia.
      * right. exists r0. split; [exact Hr0 |]. split; [congruence |]. unfold in_region. lia.
Qed.

Lemma observable_grant : forall c id u A y, u <> A ->
    observable {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c ++ [(id, u)];
                  c_freed := c_freed c; c_reserved := c_reserved c; c_operators := c_operators c |} A y ->
    observable c A y.
Proof.
  intros c id u A y Hu [[b [Hb [Hr Hy]]] | Hf]; [left | right; exact Hf].
  exists b. split; [exact Hb |]. split; [| exact Hy].
  unfold readable, has_grant, find_buffer in Hr |- *. cbn [c_buffers c_grants] in Hr.
  rewrite existsb_app in Hr. cbn [existsb fst snd] in Hr. apply Z.eqb_neq in Hu. rewrite Hu in Hr.
  rewrite andb_false_r, !orb_false_r in Hr. exact Hr.
Qed.

(* A reservation changes nothing that a tenant observes. *)
Lemma observable_reserve : forall c t o n A y,
    observable {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c; c_freed := c_freed c;
                  c_reserved := c_reserved c ++ [{| rs_tenant := t; rs_offset := o; rs_bytes := n |}];
                  c_operators := c_operators c |} A y ->
    observable c A y.
Proof. intros c t o n A y [H | H]; [left | right]; exact H. Qed.

(* The end of a reservation drops freed regions. *)
Lemma observable_unreserve : forall c t A y,
    observable {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c;
                  c_freed := filter (fun r => negb (Z.eqb (rg_owner r) t)) (c_freed c);
                  c_reserved := filter (fun r => negb (Z.eqb (rs_tenant r) t)) (c_reserved c);
                  c_operators := c_operators c |} A y ->
    observable c A y.
Proof.
  intros c t A y [H | [r [Hr [Ho Hy]]]]; [left; exact H |].
  right. exists r. cbn [c_freed] in Hr. apply filter_In in Hr. split; [exact (proj1 Hr) | split; assumption].
Qed.

(* ------------------------------------------------------------------ *)
(* The invariants of isolation.                                        *)

Lemma freed_restatus : forall c c' (g : buffer -> status), freed_apart c ->
    c_buffers c' = map (fun b => with_status b (g b)) (c_buffers c) -> c_freed c' = c_freed c -> freed_apart c'.
Proof.
  intros c c' g Hfa Hb Hf r b Hr Hb'. rewrite Hf in Hr. rewrite Hb in Hb'.
  apply in_map_iff in Hb'. destruct Hb' as [b0 [<- Hb0]]. exact (Hfa r b0 Hr Hb0).
Qed.

Lemma sound_action : forall ch t w a w', sound w -> run_action base kernels ch t w a w' -> sound w'.
Proof.
  intros ch t w a w' [Hw Hfa] H. split; [exact (run_action_ok base kernels _ _ _ _ _ Hw H) |].
  destruct H as [c m a c' es m' Hc He]. cbn [w_cat] in *. destruct Hw as [Hinv _].
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - destruct (allocate_next _ _ _ _ _ _ _ Hc) as [-> [_ [Hov _]]].
    intros r' b Hr' Hb. cbn [c_freed c_buffers] in Hr', Hb.
    destruct (carve_in _ _ _ _ Hr') as [r [Hr [_ [Hlo [Hhi Hap]]]]].
    apply in_app_or in Hb. destruct Hb as [Hb | [<- | []]].
    + pose proof (Hfa r b Hr Hb). lia.
    + cbn [bf_offset bf_bytes]. lia.
  - destruct (free_next _ _ _ _ _ Hc) as [fb [Fb [_ [_ ->]]]].
    intros r b Hr Hb. cbn [c_freed c_buffers] in Hr, Hb. apply filter_In in Hb. destruct Hb as [Hb Hne].
    destruct (find_buffer_in _ _ _ Fb) as [Hfb Hid]. destruct Hinv as [Hp [_ [Hap _]]].
    destruct (Hp fb Hfb) as [_ [Hn _]]. destruct (Hp b Hb) as [_ [Hnb _]].
    destruct (release_in _ _ (bf_bytes fb) _ _ ltac:(lia) Hr) as [[Hr' _] | [_ [Hlo [Hhi Hcover]]]];
      [exact (Hfa r b Hr' Hb) |].
    assert (Ha : apart fb b)
      by (apply Hap; [exact Hfb | exact Hb | rewrite negb_true_iff, bid_eqb_neq in Hne; congruence]).
    unfold apart in Ha.
    (* a byte of b in the merged region would lie in fb or in a freed region *)
    destruct (Z_le_gt_dec (rg_offset r + rg_bytes r) (bf_offset b)) as [E | E]; [left; exact E |].
    destruct (Z_le_gt_dec (bf_offset b + bf_bytes b) (rg_offset r)) as [E' | E']; [right; exact E' |].
    exfalso. destruct (Hcover (Z.max (rg_offset r) (bf_offset b)) ltac:(lia)) as [Hy | [r0 [Hr0 [_ Hy]]]]; [lia |].
    pose proof (Hfa r0 b Hr0 Hb). lia.
  - destruct (grant_next _ _ _ _ _ _ Hc) as [fb [_ [_ [_ [_ ->]]]]]. exact Hfa.
  - destruct (write_next _ _ _ _ _ _ _ Hc) as [fb [_ [_ [_ Hel]]]].
    destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ ->]]] | rewrite Hel];
      exact (freed_restatus _ _ _ Hfa (set_status_map _ _ _) eq_refl).
  - destruct (fill_next _ _ _ _ _ _ _ _ Hc) as [fb [_ [_ [_ [_ [_ ->]]]]]].
    exact (freed_restatus _ _ _ Hfa (set_status_map _ _ _) eq_refl).
  - destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ Hc) as [Mf [_ [_ [-> _]]]].
    exact (freed_restatus _ _ _ Hfa (written_buffers _ _ _) (proj1 (written_frame _ _ _))).
  - destruct (reserve_next _ _ _ _ _ _ Hc) as [_ [_ [_ [_ [_ [_ ->]]]]]]. exact Hfa.
  - destruct (unreserve_next _ _ _ _ Hc) as [_ [_ [_ ->]]].
    intros r b Hr Hb. cbn [c_freed c_buffers] in Hr, Hb. apply filter_In in Hr. exact (Hfa r b (proj1 Hr) Hb).
Qed.

(* ------------------------------------------------------------------ *)
(* An action of another tenant.                                        *)

Lemma other_action : forall ch A B c m a c' m', A <> B -> sound {| w_cat := c; w_mem := m |} ->
    (forall id u, a = Grant id u -> u <> A) ->
    run_action base kernels ch B {| w_cat := c; w_mem := m |} a {| w_cat := c'; w_mem := m' |} ->
    forall y, observable c' A y -> observable c A y /\ m' y = m y.
Proof.
  intros ch A B c m a c' m' HAB Hsound Hng H y Hy. pose proof Hsound as [Hw Hfa].
  pose proof Hw as [Hinv [Hs [Hver Hb]]]. cbn [w_cat w_mem] in *.
  inversion H as [c0 m0 a0 c1 es m1 Hc He]; subst.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - (* B's allocation: the new buffer is B's, and a clear touches no byte A observes *)
    destruct (allocate_next _ _ _ _ _ _ _ Hc) as [-> [Hes [Hov Hnone]]].
    destruct (observable_allocate c B id o n A y Hinv Hnone Hov Hy) as [[Hobs Hout] | [E _]]; [| congruence].
    split; [exact Hobs |].
    destruct Hes as [[-> _] | ->]; [rewrite (effects_nil _ _ _ _ _ _ _ He); reflexivity |].
    apply effects_one in He. inversion He; subst. apply write_outside. exact Hout.
  - (* B's free: what A observes shrinks *)
    destruct (free_next _ _ _ _ _ Hc) as [fb [Fb [Ho [-> ->]]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He).
    split; [exact (observable_free c B id fb A y Hinv Fb Ho Hy) | reflexivity].
  - (* a grant to a tenant other than A *)
    destruct (grant_next _ _ _ _ _ _ Hc) as [fb [_ [_ [_ [-> ->]]]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He).
    split; [exact (observable_grant c id u A y (Hng id u eq_refl) Hy) | reflexivity].
  - (* B's write lies in a buffer B may write *)
    destruct (write_next _ _ _ _ _ _ _ Hc) as [fb [Fb [-> [Hlen Hel]]]].
    assert (Hobs : observable c A y).
    { destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ Ec]]] | rename Hel into Ec]; rewrite Ec in Hy;
        exact (observable_restatus c _ _ A y (set_status_map c id _) eq_refl eq_refl Hy). }
    split; [exact Hobs |]. apply effects_one in He. inversion He; subst.
    assert (Hwr : writable c B id = true).
    { cbn [check_action] in Hc. rewrite Fb in Hc. destruct (writable c B id); [reflexivity | discriminate Hc]. }
    destruct (find_buffer_in _ _ _ Fb) as [Hfb Hid]. rewrite <- Hid in Hwr.
    pose proof (observable_not_writable c A B y fb Hinv Hfa HAB Hobs Hfb Hwr) as Hn.
    unfold upload_bytes. destruct ((base + bf_offset fb <=? y) && _) eqn:Ey; [| reflexivity].
    exfalso. apply Hn. apply andb_true_iff in Ey. rewrite Z.leb_le, Z.ltb_lt in Ey. unfold in_buffer. lia.
  - (* B's fill lies in a buffer B may write *)
    destruct (fill_next _ _ _ _ _ _ _ _ Hc) as [fb [Fb [-> [_ [_ [_ Ec]]]]]]. rewrite Ec in Hy.
    assert (Hobs : observable c A y) by exact (observable_restatus c _ _ A y (set_status_map c id _) eq_refl eq_refl Hy).
    split; [exact Hobs |]. apply effects_one in He. inversion He; subst.
    assert (Hwr : writable c B id = true).
    { cbn [check_action] in Hc. rewrite Fb in Hc. destruct (writable c B id); [reflexivity | discriminate Hc]. }
    destruct (find_buffer_in _ _ _ Fb) as [Hfb Hid]. rewrite <- Hid in Hwr.
    pose proof (observable_not_writable c A B y fb Hinv Hfa HAB Hobs Hfb Hwr) as Hn.
    unfold fill_bytes. destruct ((base + bf_offset fb <=? y) && _) eqn:Ey; [| reflexivity].
    exfalso. apply Hn. apply andb_true_iff in Ey. rewrite Z.leb_le, Z.ltb_lt in Ey. unfold in_buffer. lia.
  - (* B's launch writes only buffers B may write *)
    destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ Hc) as [Mf [_ [_ [Ec _]]]].
    assert (Hobs : observable c A y).
    { rewrite Ec in Hy.
      exact (observable_restatus _ _ _ A y (written_buffers _ _ _) (proj2 (written_frame _ _ _))
               (proj1 (written_frame _ _ _)) Hy). }
    split; [exact Hobs |].
    exact (launch_writes base kernels ch B c m op gx gy gz buffers sizes scalars guarded _ Hw H y
             (fun b' Hb' Hwr => observable_not_writable c A B y b' Hinv Hfa HAB Hobs Hb' Hwr)).
  - (* B's reservation *)
    destruct (reserve_next _ _ _ _ _ _ Hc) as [_ [_ [_ [_ [_ [-> ->]]]]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He). split; [exact (observable_reserve c B o n A y Hy) | reflexivity].
  - (* the end of B's reservation *)
    destruct (unreserve_next _ _ _ _ Hc) as [_ [_ [-> ->]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He). split; [exact (observable_unreserve c B A y Hy) | reflexivity].
Qed.

(* ------------------------------------------------------------------ *)
(* An action of the tenant.                                            *)

Lemma self_action : forall ch A c m1 m2 a w1' w2',
    sound {| w_cat := c; w_mem := m1 |} -> sound {| w_cat := c; w_mem := m2 |} -> agree_on c A m1 m2 ->
    run_action base kernels ch A {| w_cat := c; w_mem := m1 |} a w1' ->
    run_action base kernels ch A {| w_cat := c; w_mem := m2 |} a w2' ->
    w_cat w1' = w_cat w2' /\ agree_on (w_cat w1') A (w_mem w1') (w_mem w2').
Proof.
  intros ch A c m1 m2 a w1' w2' Hs1 Hs2 Hag H1 H2. pose proof Hs1 as [Hw1 Hfa]. pose proof Hw1 as [Hinv _].
  inversion H1 as [c0 m0 a0 c1 es1 m1' Hc1 He1]; subst.
  inversion H2 as [c0 m0 a0 c2 es2 m2' Hc2 He2]; subst. cbn [w_cat w_mem].
  rewrite Hc1 in Hc2. injection Hc2 as <- <-. split; [reflexivity |].
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - (* A's allocation: a kept region was A's, a cleared one is zero in both *)
    destruct (allocate_next _ _ _ _ _ _ _ Hc1) as [-> [Hes [Hov Hnone]]].
    intros y Hy. destruct (observable_allocate c A id o n A y Hinv Hnone Hov Hy) as [[Hobs Hout] | [_ Hin]].
    + destruct Hes as [[-> _] | ->].
      * rewrite (effects_nil _ _ _ _ _ _ _ He1), (effects_nil _ _ _ _ _ _ _ He2). exact (Hag y Hobs).
      * apply effects_one in He1, He2. inversion He1; subst. inversion He2; subst.
        rewrite !write_outside by exact Hout. exact (Hag y Hobs).
    + destruct Hes as [[-> [r [Hr [Hro [Hlo Hhi]]]]] | ->].
      * rewrite (effects_nil _ _ _ _ _ _ _ He1), (effects_nil _ _ _ _ _ _ _ He2). apply Hag.
        right. exists r. split; [exact Hr |]. split; [exact Hro |]. unfold in_region. lia.
      * apply effects_one in He1, He2. inversion He1; subst. inversion He2; subst.
        unfold write_bytes. destruct (_ && _) eqn:E; [reflexivity |].
        exfalso. apply andb_false_iff in E. rewrite Z.leb_gt, Z.ltb_ge in E. lia.
  - (* A's free *)
    destruct (free_next _ _ _ _ _ Hc1) as [fb [Fb [Ho [-> ->]]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He1), (effects_nil _ _ _ _ _ _ _ He2).
    intros y Hy. exact (Hag y (observable_free c A id fb A y Hinv Fb Ho Hy)).
  - (* A's grant to another tenant *)
    destruct (grant_next _ _ _ _ _ _ Hc1) as [fb [_ [_ [Hu [-> ->]]]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He1), (effects_nil _ _ _ _ _ _ _ He2).
    intros y Hy. exact (Hag y (observable_grant c id u A y Hu Hy)).
  - (* A's write: the same data in both *)
    destruct (write_next _ _ _ _ _ _ _ Hc1) as [fb [Fb [-> [_ Hel]]]].
    apply effects_one in He1, He2. inversion He1; subst. inversion He2; subst.
    intros y Hy.
    assert (Hobs : observable c A y).
    { destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ Ec]]] | rename Hel into Ec]; rewrite Ec in Hy;
        exact (observable_restatus c _ _ A y (set_status_map c id _) eq_refl eq_refl Hy). }
    unfold upload_bytes. destruct (_ && _); [reflexivity | exact (Hag y Hobs)].
  - (* A's fill: the same value in both *)
    destruct (fill_next _ _ _ _ _ _ _ _ Hc1) as [fb [Fb [-> [_ [_ [_ ->]]]]]].
    apply effects_one in He1, He2. inversion He1; subst. inversion He2; subst.
    intros y Hy.
    assert (Hobs : observable c A y) by exact (observable_restatus c _ _ A y (set_status_map c id _) eq_refl eq_refl Hy).
    unfold fill_bytes. destruct (_ && _); [reflexivity | exact (Hag y Hobs)].
  - (* A's launch reads only what A observes *)
    destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ Hc1) as [Mf [_ [_ [Ec _]]]].
    destruct (launch_reads base kernels ch A c m1 m2 op gx gy gz buffers sizes scalars guarded _ _ (observable c A) Hw1
                (proj1 Hs2) H1 H2
                (fun b y Hb Hr Hy => or_introl (ex_intro _ b (conj Hb (conj Hr Hy)))) Hag) as [_ Ha].
    intros y Hy. apply Ha. rewrite Ec in Hy.
    exact (observable_restatus _ _ _ A y (written_buffers _ _ _) (proj2 (written_frame _ _ _))
             (proj1 (written_frame _ _ _)) Hy).
  - (* A's reservation *)
    destruct (reserve_next _ _ _ _ _ _ Hc1) as [_ [_ [_ [_ [_ [-> ->]]]]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He1), (effects_nil _ _ _ _ _ _ _ He2).
    intros y Hy. exact (Hag y (observable_reserve c A o n A y Hy)).
  - (* the end of A's reservation *)
    destruct (unreserve_next _ _ _ _ Hc1) as [_ [_ [-> ->]]].
    rewrite (effects_nil _ _ _ _ _ _ _ He1), (effects_nil _ _ _ _ _ _ _ He2).
    intros y Hy. exact (Hag y (observable_unreserve c A A y Hy)).
Qed.

(* ------------------------------------------------------------------ *)
(* Pipelines.                                                          *)

Lemma actions_isolation : forall A t chs1 chs2 w1 w2 actions w1' w2',
    sound w1 -> sound w2 -> w_cat w1 = w_cat w2 -> agree_on (w_cat w1) A (w_mem w1) (w_mem w2) ->
    (t = A -> chs1 = chs2) -> (t <> A -> forall id, ~ In (Grant id A) actions) ->
    run_actions base kernels t chs1 w1 actions w1' -> run_actions base kernels t chs2 w2 actions w2' ->
    sound w1' /\ sound w2' /\ w_cat w1' = w_cat w2' /\ agree_on (w_cat w1') A (w_mem w1') (w_mem w2').
Proof.
  intros A t chs1 chs2 w1 w2 actions w1' w2' Hs1 Hs2 Hc Hag Hch Hng H1. revert chs2 w2 w2' Hs2 Hc Hag Hch Hng.
  induction H1 as [w1 | ch1 chs1 w1 a w1m actions w1' Ha1 Hr1 IH];
    intros chs2 w2 w2' Hs2 Hc Hag Hch Hng H2; inversion H2 as [w2e | ch2 chs2' w2e a2 w2m actions2 w2e' Ha2 Hr2]; subst.
  - split; [exact Hs1 |]. split; [exact Hs2 |]. split; [exact Hc | exact Hag].
  - destruct w1 as [c m1]. destruct w2 as [c2 m2]. cbn [w_cat w_mem] in Hc, Hag. subst c2.
    assert (Hstep : w_cat w1m = w_cat w2m /\ agree_on (w_cat w1m) A (w_mem w1m) (w_mem w2m)).
    { destruct (Z.eq_dec t A) as [-> | Hne].
      - injection (Hch eq_refl) as <- _. exact (self_action ch1 A c m1 m2 a w1m w2m Hs1 Hs2 Hag Ha1 Ha2).
      - destruct w1m as [c1 m1']. destruct w2m as [c2 m2'].
        assert (Hc12 : c1 = c2).
        { inversion Ha1 as [? ? ? ? ? ? Hc1 _]; subst. inversion Ha2 as [? ? ? ? ? ? Hc2 _]; subst.
          rewrite Hc1 in Hc2. injection Hc2 as <- _. reflexivity. }
        subst c2. cbn [w_cat w_mem]. split; [reflexivity |].
        assert (Hga : forall id u, a = Grant id u -> u <> A)
          by (intros id u -> <-; exact (Hng Hne id (or_introl eq_refl))).
        intros y Hy.
        destruct (other_action ch1 A t c m1 a c1 m1' (not_eq_sym Hne) Hs1 Hga Ha1 y Hy) as [Hobs E1].
        destruct (other_action ch2 A t c m2 a c1 m2' (not_eq_sym Hne) Hs2 Hga Ha2 y Hy) as [_ E2].
        rewrite E1, E2. exact (Hag y Hobs). }
    destruct Hstep as [Hcm Hagm].
    apply (IH (sound_action _ _ _ _ _ Hs1 Ha1) chs2' w2m w2' (sound_action _ _ _ _ _ Hs2 Ha2) Hcm Hagm).
    + intros E. injection (Hch E) as _ E'. exact E'.
    + intros Hne id Hin. exact (Hng Hne id (or_intror Hin)).
    + exact Hr2.
Qed.

(* A's choices of the hardware for A's steps are the same in both runs:
   the schedule, the threads, and the oracles, which read the shared and
   local memory of the launch. *)
Fixpoint same_choices (A : Z) (steps : list step) (chss1 chss2 : list (list choice)) : Prop :=
  match steps, chss1, chss2 with
  | s :: ss, chs1 :: rest1, chs2 :: rest2 => (s_tenant s = A -> chs1 = chs2) /\ same_choices A ss rest1 rest2
  | _, _, _ => True
  end.

(* No other tenant grants A a buffer. *)
Definition no_grant_to (A : Z) (steps : list step) : Prop :=
  forall s id, In s steps -> s_tenant s <> A -> ~ In (Grant id A) (s_actions s).

Theorem tenant_isolation_same_steps : forall A steps chss1 chss2 w1 w2 w1' w2',
    sound w1 -> sound w2 -> w_cat w1 = w_cat w2 -> agree_on (w_cat w1) A (w_mem w1) (w_mem w2) ->
    no_grant_to A steps -> same_choices A steps chss1 chss2 ->
    run_steps base kernels chss1 w1 steps w1' -> run_steps base kernels chss2 w2 steps w2' ->
    w_cat w1' = w_cat w2' /\ agree_on (w_cat w1') A (w_mem w1') (w_mem w2').
Proof.
  intros A steps chss1 chss2 w1 w2 w1' w2' Hs1 Hs2 Hc Hag Hng Hsame H1. revert chss2 w2 w2' Hs2 Hc Hag Hng Hsame.
  induction H1 as [w1 | chs1 chss1 w1 s w1m ss w1' Hst1 Hr1 IH];
    intros chss2 w2 w2' Hs2 Hc Hag Hng Hsame H2;
    inversion H2 as [w2e | chs2 chss2' w2e s2 w2m ss2 w2e' Hst2 Hr2]; subst; [split; assumption |].
  cbn [same_choices] in Hsame. destruct Hsame as [Hch Hsame].
  assert (Hstep : sound w1m /\ sound w2m /\ w_cat w1m = w_cat w2m /\ agree_on (w_cat w1m) A (w_mem w1m) (w_mem w2m)).
  { destruct Hst1 as [w1 s c1' es1 w1m Hcs1 Ha1 | w1 s r1 Hcs1];
      destruct Hst2 as [w2 s c2' es2 w2m Hcs2 Ha2 | w2 s r2 Hcs2]; rewrite Hc in Hcs1; rewrite Hcs1 in Hcs2;
      try discriminate Hcs2.
    - apply (actions_isolation A (s_tenant s) chs1 chs2 w1 w2 (s_actions s) w1m w2m Hs1 Hs2 Hc Hag Hch); [| exact Ha1 | exact Ha2].
      intros Hne id Hin. exact (Hng s id (or_introl eq_refl) Hne Hin).
    - split; [exact Hs1 |]. split; [exact Hs2 |]. split; [exact Hc | exact Hag]. }
  destruct Hstep as [Hs1m [Hs2m [Hcm Hagm]]].
  apply (IH Hs1m chss2' w2m w2' Hs2m Hcm Hagm); [| exact Hsame | exact Hr2].
  intros s' id Hin. exact (Hng s' id (or_intror Hin)).
Qed.

(* ------------------------------------------------------------------ *)
(* The part of the catalog that a tenant owns.                        *)

(* A's own part of the catalog: its buffers, the grants on its buffers, its
   freed regions, and its reservation, with the pool and the operators. *)
Definition own (A : Z) (c : catalog) : catalog :=
  {| c_pool := c_pool c;
     c_buffers := filter (fun b => Z.eqb (bf_owner b) A) (c_buffers c);
     c_grants := filter (fun g => Z.eqb (fst (fst g)) A) (c_grants c);
     c_freed := filter (fun r => Z.eqb (rg_owner r) A) (c_freed c);
     c_reserved := filter (fun r => Z.eqb (rs_tenant r) A) (c_reserved c);
     c_operators := c_operators c |}.

(* No buffer is granted to A. *)
Definition ungranted (A : Z) (c : catalog) : Prop := forall g, In g (c_grants c) -> snd g <> A.

Definition inside_r (r : reservation) (o n : Z) : Prop := rs_offset r <= o /\ o + n <= rs_offset r + rs_bytes r.

(* The reservations of two tenants lie apart; every buffer has an id of its
   owner and lies in its owner's reservation, and every freed region in its
   last owner's. *)
Definition res_inv (c : catalog) : Prop :=
  (forall r1 r2, In r1 (c_reserved c) -> In r2 (c_reserved c) -> rs_tenant r1 <> rs_tenant r2 ->
     rs_offset r1 + rs_bytes r1 <= rs_offset r2 \/ rs_offset r2 + rs_bytes r2 <= rs_offset r1) /\
  (forall b, In b (c_buffers c) -> fst (bf_id b) = bf_owner b /\
     exists r, find_reservation c (bf_owner b) = Some r /\ inside_r r (bf_offset b) (bf_bytes b)) /\
  (forall g, In g (c_freed c) ->
     exists r, find_reservation c (rg_owner g) = Some r /\ inside_r r (rg_offset g) (rg_bytes g)).

Lemma filter_comm : forall (X : Type) (p q : X -> bool) l, filter p (filter q l) = filter q (filter p l).
Proof.
  intros X p q l. induction l as [| x l IH]; [reflexivity |]. cbn [filter].
  destruct (q x) eqn:Q, (p x) eqn:P; cbn [filter]; rewrite ?Q, ?P, IH; reflexivity.
Qed.

Lemma filter_implied : forall (X : Type) (f p : X -> bool) l, (forall x, In x l -> f x = true -> p x = true) ->
    filter f (filter p l) = filter f l.
Proof.
  intros X f p l. induction l as [| x l IH]; intros H; [reflexivity |]. cbn [filter].
  assert (H' : forall y, In y l -> f y = true -> p y = true) by (intros y Hy; apply H; right; exact Hy).
  destruct (p x) eqn:P; cbn [filter].
  - destruct (f x); rewrite (IH H'); reflexivity.
  - destruct (f x) eqn:F; [| exact (IH H')]. rewrite (H x (or_introl eq_refl) F) in P. discriminate P.
Qed.

Lemma filter_true : forall (X : Type) (p : X -> bool) l, (forall x, In x l -> p x = true) -> filter p l = l.
Proof.
  intros X p l. induction l as [| x l IH]; intros H; [reflexivity |]. cbn [filter].
  rewrite (H x (or_introl eq_refl)). f_equal. apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma filter_false : forall (X : Type) (p : X -> bool) l, (forall x, In x l -> p x = false) -> filter p l = [].
Proof.
  intros X p l. induction l as [| x l IH]; intros H; [reflexivity |]. cbn [filter].
  rewrite (H x (or_introl eq_refl)). apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma existsb_implied : forall (X : Type) (f p : X -> bool) l, (forall x, In x l -> f x = true -> p x = true) ->
    existsb f (filter p l) = existsb f l.
Proof.
  intros X f p l. induction l as [| x l IH]; intros H; [reflexivity |]. cbn [filter existsb].
  assert (H' : forall y, In y l -> f y = true -> p y = true) by (intros y Hy; apply H; right; exact Hy).
  destruct (p x) eqn:P; cbn [existsb].
  - rewrite (IH H'). reflexivity.
  - destruct (f x) eqn:F; [| exact (IH H')]. rewrite (H x (or_introl eq_refl) F) in P. discriminate P.
Qed.

Lemma find_none_all : forall (X : Type) (f : X -> bool) l, (forall x, In x l -> f x = false) -> find f l = None.
Proof.
  intros X f l. induction l as [| x l IH]; intros H; [reflexivity |]. cbn [find].
  rewrite (H x (or_introl eq_refl)). apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma find_buffer_filter : forall (p : buffer -> bool) l id, NoDup (map bf_id l) ->
    find (fun b => bid_eqb (bf_id b) id) (filter p l) =
    match find (fun b => bid_eqb (bf_id b) id) l with Some b => if p b then Some b else None | None => None end.
Proof.
  intros p l id. induction l as [| x l IH]; intros Hn; [reflexivity |].
  inversion Hn as [| ? ? Hx Hl]; subst. cbn [filter find].
  destruct (bid_eqb (bf_id x) id) eqn:E.
  - destruct (p x) eqn:P; cbn [find]; [rewrite E; reflexivity |].
    rewrite (IH Hl). apply bid_eqb_eq in E. subst id.
    rewrite find_none_all; [reflexivity |]. intros y Hy.
    destruct (bid_eqb (bf_id y) (bf_id x)) eqn:E'; [| reflexivity].
    exfalso. apply Hx. apply bid_eqb_eq in E'. rewrite <- E'. apply in_map. exact Hy.
  - destruct (p x); cbn [find]; rewrite ?E; exact (IH Hl).
Qed.

Lemma find_own : forall A c id, inv c ->
    find_buffer (own A c) id =
    match find_buffer c id with Some b => if Z.eqb (bf_owner b) A then Some b else None | None => None end.
Proof.
  intros A c id [_ [Hn _]]. unfold find_buffer, own. cbn [c_buffers]. exact (find_buffer_filter _ _ _ Hn).
Qed.

Lemma find_reservation_own : forall A c, find_reservation (own A c) A = find_reservation c A.
Proof.
  intros A c. unfold find_reservation, own. cbn [c_reserved]. apply find_filter. intros r H. exact H.
Qed.

Lemma has_grant_ungranted : forall A c id, ungranted A c -> has_grant c id A = false.
Proof.
  intros A c id H. unfold has_grant. destruct (existsb _ _) eqn:E; [| reflexivity].
  apply existsb_exists in E. destruct E as [g [Hg E]]. rewrite andb_true_iff, Z.eqb_eq in E.
  exfalso. exact (H g Hg (proj2 E)).
Qed.

Lemma owner_of_find : forall c id b, res_inv c -> find_buffer c id = Some b -> fst id = bf_owner b.
Proof. intros c id b [_ [Hb _]] H. destruct (find_buffer_in _ _ _ H) as [Hin <-]. exact (proj1 (Hb b Hin)). Qed.

Lemma readable_own : forall A c id, inv c -> ungranted A c -> readable (own A c) A id = readable c A id.
Proof.
  intros A c id Hinv Hu. unfold readable. rewrite (find_own A c id Hinv), (has_grant_ungranted A c id Hu).
  destruct (find_buffer c id) as [b |]; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E; cbn [orb]; [rewrite E; reflexivity | reflexivity].
Qed.

Lemma granted_own : forall A c id, fst id = A -> granted (own A c) id = granted c id.
Proof.
  intros A c id Hid. unfold granted, own. cbn [c_grants].
  apply existsb_implied. intros [gid u] _ E. cbn [fst snd] in *. apply bid_eqb_eq in E. subst gid. rewrite Hid. apply Z.eqb_refl.
Qed.

Lemma has_grant_own : forall A c id u, fst id = A -> has_grant (own A c) id u = has_grant c id u.
Proof.
  intros A c id u Hid. unfold has_grant, own. cbn [c_grants].
  apply existsb_implied. intros [gid v] _ E. cbn [fst snd] in *. apply andb_true_iff in E. destruct E as [E _].
  apply bid_eqb_eq in E. subst gid. rewrite Hid. apply Z.eqb_refl.
Qed.

Lemma writable_own : forall A c id, inv c -> res_inv c -> writable (own A c) A id = writable c A id.
Proof.
  intros A c id Hinv Hr. unfold writable. rewrite (find_own A c id Hinv).
  destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E; [| reflexivity]. rewrite E. cbn [andb].
  rewrite granted_own; [reflexivity |]. rewrite (owner_of_find c id b Hr Fb). apply Z.eqb_eq. exact E.
Qed.

Lemma inv_own : forall A c, inv c -> res_inv c -> inv (own A c).
Proof.
  intros A c [Hp [Hn [Ha Hg]]] [_ [Hrb _]]. unfold own.
  split; [| split; [| split]]; cbn [c_buffers c_pool c_grants].
  - intros b Hb. apply filter_In in Hb. exact (Hp b (proj1 Hb)).
  - apply NoDup_filter_map. exact Hn.
  - intros a b HA HB. apply filter_In in HA, HB. exact (Ha a b (proj1 HA) (proj1 HB)).
  - intros [[gt gn] u] Hin. apply filter_In in Hin. destruct Hin as [Hin E]. cbn [fst] in E.
    apply Z.eqb_eq in E. destruct (Hg _ Hin) as [b [Hb [Hid Ho]]]. exists b. split; [| split; assumption].
    apply filter_In. split; [exact Hb |]. destruct (Hrb b Hb) as [Hf _]. rewrite Hid in Hf. cbn [fst] in Hf.
    apply Z.eqb_eq. congruence.
Qed.

Lemma sound_own : forall A c m, sound {| w_cat := c; w_mem := m |} -> res_inv c ->
    sound {| w_cat := own A c; w_mem := m |}.
Proof.
  intros A c m [[Hinv [Hs [Hver Hb]]] Hfa] Hr. split; [split; [| split; [| split]] |]; cbn [w_cat w_mem] in *.
  - exact (inv_own A c Hinv Hr).
  - intros b Hin. unfold own in Hin. cbn [c_buffers] in Hin. apply filter_In in Hin. exact (Hs b (proj1 Hin)).
  - intros op Mf Hf. apply Hver. exact Hf.
  - exact Hb.
  - intros r b Hr' Hb'. unfold own in Hr', Hb'. cbn [c_freed c_buffers] in Hr', Hb'.
    apply filter_In in Hr', Hb'. exact (Hfa r b (proj1 Hr') (proj1 Hb')).
Qed.

Lemma observable_own : forall A c y, inv c -> ungranted A c -> (observable (own A c) A y <-> observable c A y).
Proof.
  intros A c y Hinv Hu. split.
  - intros [[b [Hb [Hrd Hy]]] | [r [Hr [Ho Hy]]]].
    + left. exists b. unfold own in Hb. cbn [c_buffers] in Hb. apply filter_In in Hb.
      split; [exact (proj1 Hb) |]. split; [| exact Hy]. rewrite <- (readable_own A c _ Hinv Hu). exact Hrd.
    + right. exists r. unfold own in Hr. cbn [c_freed] in Hr. apply filter_In in Hr.
      split; [exact (proj1 Hr) | split; assumption].
  - intros [[b [Hb [Hrd Hy]]] | [r [Hr [Ho Hy]]]].
    + left. exists b. split.
      * unfold own. cbn [c_buffers]. apply filter_In. split; [exact Hb |].
        unfold readable in Hrd. rewrite (find_buffer_of c b Hinv Hb), (has_grant_ungranted A c _ Hu) in Hrd.
        rewrite orb_false_r in Hrd. exact Hrd.
      * split; [| exact Hy]. rewrite (readable_own A c _ Hinv Hu). exact Hrd.
    + right. exists r. split; [| split; assumption]. unfold own. cbn [c_freed]. apply filter_In.
      split; [exact Hr | rewrite Ho; apply Z.eqb_refl].
Qed.

Lemma agree_own : forall A c m1 m2, inv c -> ungranted A c ->
    (agree_on (own A c) A m1 m2 <-> agree_on c A m1 m2).
Proof.
  intros A c m1 m2 Hinv Hu. split; intros H y Hy; apply H; apply (observable_own A c y Hinv Hu); exact Hy.
Qed.

(* ------------------------------------------------------------------ *)
(* Every accepted action keeps the reservations' invariant.           *)

Lemma find_reservation_in : forall c t r, find_reservation c t = Some r -> In r (c_reserved c) /\ rs_tenant r = t.
Proof.
  intros c t r H. unfold find_reservation in H. apply find_some in H. destruct H as [H E].
  apply Z.eqb_eq in E. split; assumption.
Qed.

Lemma reservations_apart : forall c A B rA rB, res_inv c -> A <> B ->
    find_reservation c A = Some rA -> find_reservation c B = Some rB ->
    rs_offset rA + rs_bytes rA <= rs_offset rB \/ rs_offset rB + rs_bytes rB <= rs_offset rA.
Proof.
  intros c A B rA rB [Hap _] HAB HA HB.
  destruct (find_reservation_in _ _ _ HA) as [InA TA]. destruct (find_reservation_in _ _ _ HB) as [InB TB].
  apply Hap; [exact InA | exact InB | congruence].
Qed.

Lemma in_reservation_some : forall c t o n, in_reservation c t o n = true ->
    exists r, find_reservation c t = Some r /\ inside_r r o n.
Proof.
  intros c t o n H. unfold in_reservation in H. destruct (find_reservation c t) as [r |]; [| discriminate H].
  exists r. split; [reflexivity |]. rewrite andb_true_iff, !Z.leb_le in H. exact H.
Qed.

Lemma written_reserved : forall sizes l c0,
    c_reserved (fold_left (fun acc x => written acc sizes x) l c0) = c_reserved c0.
Proof.
  intros sizes l. induction l as [| [[id off] sp] l IH]; intros c0; [reflexivity |]. cbn [fold_left].
  rewrite IH. unfold written. destruct (negb (writes sp)); [reflexivity |]. destruct (bs_index sp); reflexivity.
Qed.

(* A catalog whose buffers differ only in status, with the same freed regions
   and reservations, keeps the invariant. *)
Lemma res_restatus : forall c c' (g : buffer -> status), res_inv c ->
    c_buffers c' = map (fun b => with_status b (g b)) (c_buffers c) -> c_freed c' = c_freed c ->
    c_reserved c' = c_reserved c -> res_inv c'.
Proof.
  intros c c' g [Hap [Hb Hf]] Eb Ef Er. unfold res_inv, find_reservation in *. split; [| split].
  - rewrite Er. exact Hap.
  - intros b' Hb'. rewrite Eb in Hb'. apply in_map_iff in Hb'. destruct Hb' as [b [<- Hin]].
    cbn [with_status bf_id bf_owner bf_offset bf_bytes]. rewrite Er. exact (Hb b Hin).
  - intros r Hr. rewrite Ef in Hr. rewrite Er. exact (Hf r Hr).
Qed.

Lemma res_action : forall c t a c' es, inv c -> res_inv c -> check_action c t a = inl (c', es) -> res_inv c'.
Proof.
  intros c t a c' es Hinv Hres H. pose proof Hres as [Hap [Hb Hf]].
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - (* an allocation: its buffer lies in its tenant's reservation, and carve keeps every piece in its region *)
    destruct (allocate_place _ _ _ _ _ _ _ H) as [Hid [Hin Hn]].
    destruct (allocate_next _ _ _ _ _ _ _ H) as [-> _]. unfold find_reservation in *.
    split; [| split]; cbn [c_reserved c_buffers c_freed].
    + exact Hap.
    + intros b' Hb'. apply in_app_or in Hb'. destruct Hb' as [Hb' | [<- | []]]; [exact (Hb b' Hb') |].
      cbn [bf_id bf_owner bf_offset bf_bytes]. split; [exact Hid |]. exact (in_reservation_some _ _ _ _ Hin).
    + intros r' Hr'. destruct (carve_in _ _ _ _ Hr') as [r [Hr [Ho [Hlo [Hhi _]]]]].
      destruct (Hf r Hr) as [rs [Hrs [Hl Hh]]]. exists rs. rewrite Ho. split; [exact Hrs |]. split; lia.
  - (* a free: the merged region lies in its tenant's reservation *)
    destruct (free_next _ _ _ _ _ H) as [fb [Fb [Ho [_ ->]]]]. unfold find_reservation in *.
    destruct (find_buffer_in _ _ _ Fb) as [Hfb _].
    assert (Hn : 1 <= bf_bytes fb) by (destruct Hinv as [Hp _]; exact (proj1 (proj2 (Hp fb Hfb)))).
    split; [| split]; cbn [c_reserved c_buffers c_freed].
    + exact Hap.
    + intros b' Hb'. apply filter_In in Hb'. exact (Hb b' (proj1 Hb')).
    + intros r' Hr'. destruct (release_in _ _ (bf_bytes fb) _ _ ltac:(lia) Hr') as [[Hr _] | [Hro [Hlo [Hhi Hcover]]]];
        [exact (Hf r' Hr) |].
      destruct (Hb fb Hfb) as [_ [rs [Hrs [Hl Hh]]]]. rewrite Ho in Hrs. rewrite Hro. exists rs.
      split; [exact Hrs |].
      (* the first and the last byte of the merged region lie in the buffer or in a freed region of t *)
      assert (Hin : forall y, rg_offset r' <= y < rg_offset r' + rg_bytes r' -> rs_offset rs <= y < rs_offset rs + rs_bytes rs).
      { intros y Hy. destruct (Hcover y Hy) as [Hy' | [r0 [Hr0 [Ho0 Hy0]]]]; [lia |].
        destruct (Hf r0 Hr0) as [rs0 [Hrs0 [Hl0 Hh0]]]. rewrite Ho0 in Hrs0. rewrite Hrs in Hrs0.
        injection Hrs0 as <-. lia. }
      pose proof (Hin (rg_offset r') ltac:(lia)). pose proof (Hin (rg_offset r' + rg_bytes r' - 1) ltac:(lia)).
      unfold inside_r. lia.
  - (* a grant *)
    destruct (grant_next _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ ->]]]]]. exact Hres.
  - (* a write *)
    destruct (write_next _ _ _ _ _ _ _ H) as [fb [_ [_ [_ Hel]]]].
    destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ ->]]] | rewrite Hel];
      exact (res_restatus _ _ _ Hres (set_status_map _ _ _) eq_refl eq_refl).
  - (* a fill *)
    destruct (fill_next _ _ _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ [_ ->]]]]]].
    exact (res_restatus _ _ _ Hres (set_status_map _ _ _) eq_refl eq_refl).
  - (* a launch *)
    destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [_ [_ [-> _]]]].
    exact (res_restatus _ _ _ Hres (written_buffers _ _ _) (proj1 (written_frame _ _ _)) (written_reserved _ _ _)).
  - (* a reservation: the tenant held none, so no buffer or freed region names it *)
    destruct (reserve_next _ _ _ _ _ _ H) as [Hnone [Hn [Ho [Hp [Hov [_ ->]]]]]].
    assert (Hkeep : forall u r, find_reservation c u = Some r ->
                      find_reservation {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c;
                                          c_freed := c_freed c;
                                          c_reserved := c_reserved c ++ [{| rs_tenant := t; rs_offset := o; rs_bytes := n |}];
                                          c_operators := c_operators c |} u = Some r).
    { intros u r Hu. unfold find_reservation in *. cbn [c_reserved]. rewrite find_app, Hu. reflexivity. }
    split; [| split]; cbn [c_reserved c_buffers c_freed].
    + intros r1 r2 H1 H2 Ht. apply in_app_or in H1, H2.
      destruct H1 as [H1 | [<- | []]], H2 as [H2 | [<- | []]].
      * exact (Hap r1 r2 H1 H2 Ht).
      * pose proof (existsb_false _ _ _ Hov r1 H1) as E. cbn beta in E. unfold overlap in E.
        cbn [rs_offset rs_bytes]. rewrite andb_false_iff, !Z.ltb_ge in E. lia.
      * pose proof (existsb_false _ _ _ Hov r2 H2) as E. cbn beta in E. unfold overlap in E.
        cbn [rs_offset rs_bytes]. rewrite andb_false_iff, !Z.ltb_ge in E. lia.
      * exfalso. apply Ht. reflexivity.
    + intros b Hb'. destruct (Hb b Hb') as [Hid [r [Hr Hi]]]. split; [exact Hid |].
      exists r. split; [exact (Hkeep _ _ Hr) | exact Hi].
    + intros g Hg. destruct (Hf g Hg) as [r [Hr Hi]]. exists r. split; [exact (Hkeep _ _ Hr) | exact Hi].
  - (* the end of a reservation: its tenant holds no buffer, and its freed regions go *)
    destruct (unreserve_next _ _ _ _ H) as [_ [Hno [_ ->]]].
    assert (Hkeep : forall u, u <> t ->
                      find_reservation {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c;
                                          c_freed := filter (fun r => negb (Z.eqb (rg_owner r) t)) (c_freed c);
                                          c_reserved := filter (fun r => negb (Z.eqb (rs_tenant r) t)) (c_reserved c);
                                          c_operators := c_operators c |} u = find_reservation c u).
    { intros u Hu. unfold find_reservation. cbn [c_reserved]. apply find_filter. intros r E.
      apply Z.eqb_eq in E. apply negb_true_iff, Z.eqb_neq. congruence. }
    split; [| split]; cbn [c_reserved c_buffers c_freed].
    + intros r1 r2 H1 H2. apply filter_In in H1, H2. exact (Hap r1 r2 (proj1 H1) (proj1 H2)).
    + intros b Hb'. pose proof (existsb_false _ _ _ Hno b Hb') as E. cbn beta in E. apply Z.eqb_neq in E.
      rewrite (Hkeep _ E). exact (Hb b Hb').
    + intros g Hg. apply filter_In in Hg. destruct Hg as [Hg E]. apply negb_true_iff, Z.eqb_neq in E.
      rewrite (Hkeep _ E). exact (Hf g Hg).
Qed.

(* No action grants A a buffer that A did not own, and no action but the end
   of A's reservation ends it. *)
Lemma ungranted_action : forall A c t a c' es, ungranted A c ->
    (forall id u, a = Grant id u -> u <> A \/ t = A) ->
    check_action c t a = inl (c', es) -> ungranted A c'.
Proof.
  intros A c t a c' es Hu Hg H.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - destruct (allocate_next _ _ _ _ _ _ _ H) as [-> _]. exact Hu.
  - destruct (free_next _ _ _ _ _ H) as [fb [_ [_ [_ ->]]]].
    intros g Hin. cbn [c_grants] in Hin. apply filter_In in Hin. exact (Hu g (proj1 Hin)).
  - destruct (grant_next _ _ _ _ _ _ H) as [fb [_ [_ [Hut [_ ->]]]]].
    intros g Hin. cbn [c_grants] in Hin. apply in_app_or in Hin. destruct Hin as [Hin | [<- | []]]; [exact (Hu g Hin) |].
    cbn [snd]. destruct (Hg id u eq_refl) as [Hne | ->]; [exact Hne | exact Hut].
  - destruct (write_next _ _ _ _ _ _ _ H) as [fb [_ [_ [_ Hel]]]].
    destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ ->]]] | rewrite Hel]; exact Hu.
  - destruct (fill_next _ _ _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ [_ ->]]]]]]. exact Hu.
  - destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [_ [_ [-> _]]]].
    intros g Hin. rewrite (proj2 (written_frame _ _ _)) in Hin. exact (Hu g Hin).
  - destruct (reserve_next _ _ _ _ _ _ H) as [_ [_ [_ [_ [_ [_ ->]]]]]]. exact Hu.
  - destruct (unreserve_next _ _ _ _ H) as [_ [_ [_ ->]]]. exact Hu.
Qed.

Lemma reservation_action : forall A c t a c' es r, find_reservation c A = Some r ->
    (a = Unreserve -> t <> A) -> check_action c t a = inl (c', es) -> find_reservation c' A = Some r.
Proof.
  intros A c t a c' es r Hr Hun H.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - destruct (allocate_next _ _ _ _ _ _ _ H) as [-> _]. exact Hr.
  - destruct (free_next _ _ _ _ _ H) as [fb [_ [_ [_ ->]]]]. exact Hr.
  - destruct (grant_next _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ ->]]]]]. exact Hr.
  - destruct (write_next _ _ _ _ _ _ _ H) as [fb [_ [_ [_ Hel]]]].
    destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ ->]]] | rewrite Hel]; exact Hr.
  - destruct (fill_next _ _ _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ [_ ->]]]]]]. exact Hr.
  - destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [_ [_ [-> _]]]].
    unfold find_reservation in *. rewrite written_reserved. exact Hr.
  - destruct (reserve_next _ _ _ _ _ _ H) as [_ [_ [_ [_ [_ [_ ->]]]]]].
    unfold find_reservation in *. cbn [c_reserved]. rewrite find_app, Hr. reflexivity.
  - destruct (unreserve_next _ _ _ _ H) as [_ [_ [_ ->]]]. pose proof (Hun eq_refl) as Ht.
    unfold find_reservation in *. cbn [c_reserved]. rewrite find_filter; [exact Hr |].
    intros r' E. apply Z.eqb_eq in E. apply negb_true_iff, Z.eqb_neq. congruence.
Qed.

(* ------------------------------------------------------------------ *)
(* How the own part changes.                                           *)

Lemma filter_map_comm : forall (X : Type) (p : X -> bool) (f : X -> X) l, (forall x, p (f x) = p x) ->
    filter p (map f l) = map f (filter p l).
Proof.
  intros X p f l H. induction l as [| x l IH]; [reflexivity |]. cbn [map filter]. rewrite H.
  destruct (p x); cbn [map]; rewrite IH; reflexivity.
Qed.

Lemma own_set_status : forall A c id st, own A (set_status c id st) = set_status (own A c) id st.
Proof.
  intros A c id st. unfold own, set_status. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators].
  f_equal. apply filter_map_comm. intros b. destruct (bid_eqb (bf_id b) id); reflexivity.
Qed.

Lemma own_written : forall A c sizes x, own A (written c sizes x) = written (own A c) sizes x.
Proof.
  intros A c sizes [[id off] sp]. unfold written. destruct (negb (writes sp)); [reflexivity |].
  destruct (bs_index sp); apply own_set_status.
Qed.

Lemma own_fold_written : forall A sizes l c,
    own A (fold_left (fun acc x => written acc sizes x) l c) = fold_left (fun acc x => written acc sizes x) l (own A c).
Proof.
  intros A sizes l. induction l as [| x l IH]; intros c; [reflexivity |]. cbn [fold_left].
  rewrite IH, own_written. reflexivity.
Qed.

(* The pieces of a region that carve leaves have its owner. *)
Lemma carve_one_owner : forall r o n,
    carve [r] o n = filter (fun r' => Z.eqb (rg_owner r') (rg_owner r)) (carve [r] o n).
Proof.
  intros r o n. symmetry. apply filter_true. intros r' Hr'. destruct (carve_in _ _ _ _ Hr') as [r0 [[<- | []] [Ho _]]].
  rewrite Ho. apply Z.eqb_refl.
Qed.

Lemma carve_cons : forall r l o n, carve (r :: l) o n = carve [r] o n ++ carve l o n.
Proof. intros r l o n. unfold carve. cbn [flat_map]. rewrite app_nil_r. reflexivity. Qed.

Lemma carve_filter : forall (p : Z -> bool) freed o n,
    filter (fun r => p (rg_owner r)) (carve freed o n) = carve (filter (fun r => p (rg_owner r)) freed) o n.
Proof.
  intros p freed o n. induction freed as [| r l IH]; [reflexivity |].
  rewrite carve_cons, filter_app, IH. cbn [filter]. destruct (p (rg_owner r)) eqn:P.
  - rewrite (carve_cons r (filter (fun r => p (rg_owner r)) l) o n). f_equal. apply filter_true. intros r' Hr'.
    destruct (carve_in _ _ _ _ Hr') as [r0 [[<- | []] [Ho _]]]. rewrite Ho. exact P.
  - rewrite filter_false; [reflexivity |]. intros r' Hr'.
    destruct (carve_in _ _ _ _ Hr') as [r0 [[<- | []] [Ho _]]]. rewrite Ho. exact P.
Qed.

(* carve leaves a region that the bytes do not touch as it is. *)
Lemma carve_apart : forall freed o n,
    (forall r, In r freed -> rg_offset r + rg_bytes r <= o \/ o + n <= rg_offset r) -> carve freed o n = freed.
Proof.
  intros freed o n. induction freed as [| r l IH]; intros H; [reflexivity |].
  rewrite carve_cons, IH by (intros r' Hr'; apply H; right; exact Hr').
  unfold carve. cbn [flat_map]. unfold overlap.
  destruct (H r (or_introl eq_refl)) as [E | E].
  - replace (Z.ltb o (rg_offset r + rg_bytes r)) with false by (symmetry; apply Z.ltb_ge; lia).
    rewrite andb_false_r. reflexivity.
  - replace (Z.ltb (rg_offset r) (o + n)) with false by (symmetry; apply Z.ltb_ge; lia). reflexivity.
Qed.

Lemma release_own : forall A freed o n,
    filter (fun r => Z.eqb (rg_owner r) A) (release freed o n A) = release (filter (fun r => Z.eqb (rg_owner r) A) freed) o n A.
Proof.
  intros A freed o n. unfold release.
  assert (Hnear : filter (touches o n A) (filter (fun r => Z.eqb (rg_owner r) A) freed) = filter (touches o n A) freed).
  { apply filter_implied. intros r _ E. unfold touches in E. apply andb_true_iff in E. exact (proj1 E). }
  rewrite Hnear, filter_app, filter_comm. cbn [filter rg_owner]. rewrite Z.eqb_refl. reflexivity.
Qed.

Lemma release_other : forall A B freed o n, A <> B ->
    filter (fun r => Z.eqb (rg_owner r) A) (release freed o n B) = filter (fun r => Z.eqb (rg_owner r) A) freed.
Proof.
  intros A B freed o n HAB. unfold release. rewrite filter_app. cbn [filter rg_owner].
  replace (Z.eqb B A) with false by (symmetry; apply Z.eqb_neq; congruence). rewrite app_nil_r, filter_comm.
  apply filter_true. intros r Hr. apply filter_In in Hr. destruct Hr as [_ E]. apply Z.eqb_eq in E.
  unfold touches. rewrite E. replace (Z.eqb A B) with false by (symmetry; apply Z.eqb_neq; exact HAB). reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* An action of A decides on A's own part as on the whole catalog.     *)

Lemma find_own_tenant : forall A c id, inv c -> res_inv c -> fst id = A -> find_buffer (own A c) id = find_buffer c id.
Proof.
  intros A c id Hinv Hr Hid. rewrite (find_own A c id Hinv).
  destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  rewrite <- (owner_of_find c id b Hr Fb), Hid, Z.eqb_refl. reflexivity.
Qed.

Definition own_result (A : Z) (d : catalog * list effect + String.string) : catalog * list effect + String.string :=
  match d with inl (c', es) => inl (own A c', es) | inr r => inr r end.

Lemma own_allocate : forall A c id o n, inv c -> res_inv c ->
    check_action (own A c) A (Allocate id o n) = own_result A (check_action c A (Allocate id o n)).
Proof.
  intros A c id o n Hinv Hr. cbn [check_action].
  destruct (negb (Z.eqb (fst id) A)) eqn:E0; [reflexivity |]. apply negb_false_iff, Z.eqb_eq in E0.
  rewrite (find_own_tenant A c id Hinv Hr E0). destruct (find_buffer c id); [reflexivity |].
  change (c_pool (own A c)) with (c_pool c).
  destruct (Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool c) (o + n)); [reflexivity |].
  destruct (negb (Z.eqb (o mod alignment) 0)); [reflexivity |].
  assert (Hres : in_reservation (own A c) A o n = in_reservation c A o n)
    by (unfold in_reservation; rewrite find_reservation_own; reflexivity).
  rewrite Hres. destruct (in_reservation c A o n) eqn:Ein; cbn [negb]; [| reflexivity].
  (* the region lies in A's reservation, so only A's buffers can overlap it *)
  assert (Hov : existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers (own A c)) =
                existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers c)).
  { unfold own. cbn [c_buffers]. apply existsb_implied. intros b Hb E.
    apply Z.eqb_eq. destruct (Z.eq_dec (bf_owner b) A) as [Eq | Ne]; [exact Eq |]. exfalso.
    destruct (in_reservation_some _ _ _ _ Ein) as [rA [HA [HAl HAh]]].
    pose proof Hr as [_ [Hb' _]]. destruct (Hb' b Hb) as [_ [rB [HB [HBl HBh]]]].
    unfold overlap in E. rewrite andb_true_iff, !Z.ltb_lt in E.
    destruct (reservations_apart c A (bf_owner b) rA rB Hr (fun H => Ne (eq_sym H)) HA HB); lia. }
  rewrite Hov. destruct (existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers c)); [reflexivity |].
  assert (Hkept : existsb (fun r => Z.eqb (rg_owner r) A && Z.leb (rg_offset r) o &&
                                     Z.leb (o + n) (rg_offset r + rg_bytes r)) (c_freed (own A c)) =
                  existsb (fun r => Z.eqb (rg_owner r) A && Z.leb (rg_offset r) o &&
                                     Z.leb (o + n) (rg_offset r + rg_bytes r)) (c_freed c)).
  { unfold own. cbn [c_freed]. apply existsb_implied. intros r _ E. rewrite !andb_true_iff in E. exact (proj1 (proj1 E)). }
  cbv zeta. rewrite Hkept. cbn [own_result]. f_equal. f_equal.
  unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators bf_owner]. f_equal.
  - rewrite filter_app. cbn [filter bf_owner]. rewrite Z.eqb_refl. reflexivity.
  - symmetry. exact (carve_filter (fun x => Z.eqb x A) (c_freed c) o n).
Qed.

Lemma own_free : forall A c id, inv c -> res_inv c ->
    check_action (own A c) A (Free id) = own_result A (check_action c A (Free id)).
Proof.
  intros A c id Hinv Hr. cbn [check_action]. rewrite (find_own A c id Hinv).
  destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E; cbv beta iota; rewrite ?E; cbn [negb]; [| reflexivity].
  cbn [own_result]. f_equal. f_equal. unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators].
  f_equal.
  - apply filter_comm.
  - apply filter_comm.
  - symmetry. apply release_own.
Qed.

Lemma own_grant : forall A c id u, inv c -> res_inv c ->
    check_action (own A c) A (Grant id u) = own_result A (check_action c A (Grant id u)).
Proof.
  intros A c id u Hinv Hr. cbn [check_action]. rewrite (find_own A c id Hinv).
  destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E; cbv beta iota; rewrite ?E; cbn [negb orb]; [| reflexivity].
  assert (Hid : fst id = A) by (rewrite (owner_of_find c id b Hr Fb); apply Z.eqb_eq; exact E).
  destruct (Z.eqb u A); [reflexivity |]. rewrite (has_grant_own A c id u Hid).
  destruct (has_grant c id u); [reflexivity |].
  cbn [own_result]. f_equal. f_equal. unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators].
  f_equal. rewrite filter_app. cbn [filter fst]. rewrite Hid, Z.eqb_refl. reflexivity.
Qed.

Lemma own_write : forall A c id data element, inv c -> res_inv c ->
    check_action (own A c) A (Write id data element) = own_result A (check_action c A (Write id data element)).
Proof.
  intros A c id data element Hinv Hr. cbn [check_action]. rewrite (writable_own A c id Hinv Hr).
  rewrite (find_own A c id Hinv). destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E.
  - destruct (negb (writable c A id)); [reflexivity |].
    destruct element as [[e s] |].
    + destruct (width_ok e && _ && _); [| reflexivity]. destruct (data_range e s data) as [lo hi].
      cbn [own_result]. rewrite own_set_status. reflexivity.
    + destruct (Z.leb _ _); [| reflexivity]. cbn [own_result]. rewrite own_set_status. reflexivity.
  - (* another tenant's buffer: A may not write it *)
    replace (writable c A id) with false; [reflexivity |].
    unfold writable. rewrite Fb, E. reflexivity.
Qed.

Lemma own_fill : forall A c id e s v, inv c -> res_inv c ->
    check_action (own A c) A (Fill id e s v) = own_result A (check_action c A (Fill id e s v)).
Proof.
  intros A c id e s v Hinv Hr. cbn [check_action]. rewrite (writable_own A c id Hinv Hr).
  rewrite (find_own A c id Hinv). destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E.
  - destruct (negb (writable c A id)); [reflexivity |]. destruct (element_range e s) as [lo hi].
    destruct (_ && _ && _ && _); [| reflexivity]. cbn [own_result]. rewrite own_set_status. reflexivity.
  - replace (writable c A id) with false; [reflexivity |].
    unfold writable. rewrite Fb, E. reflexivity.
Qed.

Lemma own_reserve : forall A c o n, find_reservation c A <> None ->
    check_action (own A c) A (Reserve o n) = own_result A (check_action c A (Reserve o n)).
Proof.
  intros A c o n Hres. cbn [check_action]. rewrite find_reservation_own.
  destruct (find_reservation c A); [reflexivity | exfalso; exact (Hres eq_refl)].
Qed.

Lemma own_unreserve : forall A c,
    check_action (own A c) A Unreserve = own_result A (check_action c A Unreserve).
Proof.
  intros A c. cbn [check_action]. rewrite find_reservation_own.
  destruct (find_reservation c A); [| reflexivity].
  assert (E : existsb (fun b => Z.eqb (bf_owner b) A) (c_buffers (own A c)) =
              existsb (fun b => Z.eqb (bf_owner b) A) (c_buffers c))
    by (unfold own; cbn [c_buffers]; apply existsb_implied; intros b _ H; exact H).
  rewrite E. destruct (existsb _ (c_buffers c)); [reflexivity |].
  cbn [own_result]. f_equal. f_equal. unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators].
  f_equal; apply filter_comm.
Qed.

Lemma arg_reason_own : forall A c guarded buffers sps sizes x, inv c -> res_inv c -> ungranted A c ->
    arg_reason (own A c) A guarded buffers sps sizes x = arg_reason c A guarded buffers sps sizes x.
Proof.
  intros A c guarded buffers sps sizes [[id off] sp] Hinv Hr Hu. unfold arg_reason.
  rewrite (readable_own A c id Hinv Hu), (writable_own A c id Hinv Hr), (find_own A c id Hinv).
  destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E; [reflexivity |].
  (* another tenant's buffer, which A may not read *)
  replace (readable c A id) with false; [reflexivity |].
  unfold readable. rewrite Fb, E, (has_grant_ungranted A c id Hu). reflexivity.
Qed.

Lemma first_reason_ext : forall (X : Type) (f g : X -> option String.string) l, (forall x, f x = g x) ->
    first_reason f l = first_reason g l.
Proof.
  intros X f g l H. induction l as [| x l IH]; [reflexivity |]. cbn [first_reason]. rewrite H, IH. reflexivity.
Qed.

Lemma in_combine_left : forall (X Y : Type) (l : list X) (l' : list Y) x, List.length l = List.length l' -> In x l ->
    exists y, In (x, y) (combine l l').
Proof.
  intros X Y l. induction l as [| a l IH]; intros l' x Hl Hx; [destruct Hx |].
  destruct l' as [| b l']; [discriminate Hl |]. cbn [combine]. destruct Hx as [<- | Hx].
  - exists b. left. reflexivity.
  - cbn [List.length] in Hl. injection Hl as Hl. destruct (IH l' x Hl Hx) as [y Hy]. exists y. right. exact Hy.
Qed.

Lemma args_of_ext : forall c1 c2 sizes ps buffers scalars,
    (forall id off, In (id, off) buffers -> find_buffer c1 id = find_buffer c2 id) ->
    args_of c1 sizes ps buffers scalars = args_of c2 sizes ps buffers scalars.
Proof.
  intros c1 c2 sizes ps. induction ps as [| [name p] ps IH]; intros buffers scalars H; [reflexivity |].
  cbn [args_of]. destruct p as [sp | a | z | lo hi |].
  - destruct buffers as [| [id off] bs]; [reflexivity |]. rewrite (H id off (or_introl eq_refl)).
    rewrite (IH bs scalars); [reflexivity |]. intros id' off' Hin. apply (H id' off'). right. exact Hin.
  - rewrite (IH buffers scalars H). reflexivity.
  - rewrite (IH buffers scalars H). reflexivity.
  - destruct scalars as [| v vs]; [reflexivity |]. rewrite (IH buffers vs H). reflexivity.
  - rewrite (IH buffers scalars H). reflexivity.
Qed.

Lemma checks_of_ext : forall c1 c2 guarded sizes (pairs : list ((bid * Z) * bspan)),
    (forall id off sp, In ((id, off), sp) pairs -> find_buffer c1 id = find_buffer c2 id) ->
    checks_of c1 guarded sizes pairs = checks_of c2 guarded sizes pairs.
Proof.
  intros c1 c2 guarded sizes pairs. induction pairs as [| [[id off] sp] l IH]; intros H; [reflexivity |].
  unfold checks_of in *. cbn [flat_map]. rewrite (H id off sp (or_introl eq_refl)).
  rewrite IH; [reflexivity |]. intros id' off' sp' Hin. apply (H id' off' sp'). right. exact Hin.
Qed.

(* Every buffer argument of an accepted launch of A is A's. *)
Lemma launch_args_own : forall A c op gx gy gz buffers sizes scalars guarded r, inv c -> res_inv c -> ungranted A c ->
    check_launch c A op gx gy gz buffers sizes scalars guarded = inl r ->
    forall id off, In (id, off) buffers -> find_buffer (own A c) id = find_buffer c id.
Proof.
  intros A c op gx gy gz buffers sizes scalars guarded r Hinv Hr Hu H id off Hin.
  destruct (check_launch_sound _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [_ [_ [_ [_ [_ [Hlen [_ [_ Hok]]]]]]]]].
  destruct (in_combine_left _ _ _ _ _ Hlen Hin) as [sp Hsp].
  rewrite Forall_forall in Hok. pose proof (Hok _ Hsp) as Hx. unfold arg_ok in Hx.
  destruct Hx as [b [Fb [_ [_ [_ [_ [Hrd [Hwr _]]]]]]]].
  rewrite (find_own A c id Hinv), Fb.
  assert (Ho : bf_owner b = A).
  { destruct (reads sp) eqn:R.
    - pose proof (Hrd eq_refl) as E. unfold readable in E. rewrite Fb, (has_grant_ungranted A c id Hu) in E.
      rewrite orb_false_r in E. apply Z.eqb_eq. exact E.
    - assert (W : writes sp = true) by (unfold reads, writes in *; destruct (bs_dir sp); congruence).
      destruct (Hwr W) as [E _]. unfold writable in E. rewrite Fb in E. apply andb_true_iff in E.
      apply Z.eqb_eq. exact (proj1 E). }
  rewrite Ho, Z.eqb_refl. reflexivity.
Qed.

Lemma own_check_launch : forall A c op gx gy gz buffers sizes scalars guarded, inv c -> res_inv c -> ungranted A c ->
    check_launch (own A c) A op gx gy gz buffers sizes scalars guarded =
    match check_launch c A op gx gy gz buffers sizes scalars guarded with
    | inl (c', k) => inl (own A c', k)
    | inr r => inr r
    end.
Proof.
  intros A c op gx gy gz buffers sizes scalars guarded Hinv Hr Hu.
  destruct (check_launch c A op gx gy gz buffers sizes scalars guarded) as [[c' k] | r] eqn:L.
  - pose proof (launch_args_own A c op gx gy gz buffers sizes scalars guarded _ Hinv Hr Hu L) as Hargs.
    unfold check_launch in *.
    change (find_binding (own A c) op) with (find_binding c op).
    destruct (find_binding c op) as [Mf |]; [| discriminate L].
    destruct grid_limit as [[lx ly] lz]. destruct (b_grid Mf) as [[mx my] mz].
    rewrite (first_reason_ext _ _ _ _ (fun x => arg_reason_own A c guarded buffers (spans Mf) sizes x Hinv Hr Hu)).
    refusals L. destruct (first_reason _ _) eqn:F; [discriminate L |]. injection L as <- <-.
    rewrite own_fold_written, (args_of_ext (own A c) c sizes (b_params Mf) buffers scalars Hargs).
    rewrite (checks_of_ext (own A c) c guarded sizes (combine buffers (spans Mf))); [reflexivity |].
    intros id off sp Hin. apply (Hargs id off). exact (in_combine_l _ _ _ _ Hin).
  - unfold check_launch in *.
    change (find_binding (own A c) op) with (find_binding c op).
    destruct (find_binding c op) as [Mf |]; [| exact L].
    destruct grid_limit as [[lx ly] lz]. destruct (b_grid Mf) as [[mx my] mz].
    rewrite (first_reason_ext _ _ _ _ (fun x => arg_reason_own A c guarded buffers (spans Mf) sizes x Hinv Hr Hu)).
    refusals L; try exact L.
    destruct (first_reason _ _); [exact L | discriminate L].
Qed.

Lemma own_action : forall A c a, inv c -> res_inv c -> ungranted A c -> find_reservation c A <> None ->
    check_action (own A c) A a = own_result A (check_action c A a).
Proof.
  intros A c a Hinv Hr Hu Hres.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - exact (own_allocate A c id o n Hinv Hr).
  - exact (own_free A c id Hinv Hr).
  - exact (own_grant A c id u Hinv Hr).
  - exact (own_write A c id data element Hinv Hr).
  - exact (own_fill A c id e s v Hinv Hr).
  - cbn [check_action]. rewrite (own_check_launch A c op gx gy gz buffers sizes scalars guarded Hinv Hr Hu).
    destruct (check_launch c A op gx gy gz buffers sizes scalars guarded) as [[c' k] | r]; reflexivity.
  - exact (own_reserve A c o n Hres).
  - exact (own_unreserve A c).
Qed.

(* ------------------------------------------------------------------ *)
(* An action of another tenant leaves A's own part as it is.           *)

Lemma carve_filter_owner : forall A freed o n,
    filter (fun r => Z.eqb (rg_owner r) A) (carve freed o n) = carve (filter (fun r => Z.eqb (rg_owner r) A) freed) o n.
Proof. intros A freed o n. exact (carve_filter (fun x => Z.eqb x A) freed o n). Qed.

Lemma set_status_absent : forall c id st, (forall b, In b (c_buffers c) -> bf_id b <> id) -> set_status c id st = c.
Proof.
  intros c id st H. destruct c as [pool bufs grants freed reserved ops]. unfold set_status.
  cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators] in *. f_equal.
  rewrite <- (map_id bufs) at 2. apply map_ext_in. intros b Hb.
  destruct (bid_eqb (bf_id b) id) eqn:E; [| reflexivity]. exfalso. apply (H b Hb). apply bid_eqb_eq. exact E.
Qed.

Lemma written_absent : forall c sizes id off sp, (forall b, In b (c_buffers c) -> bf_id b <> id) ->
    written c sizes ((id, off), sp) = c.
Proof.
  intros c sizes id off sp H. unfold written. destruct (negb (writes sp)); [reflexivity |].
  destruct (bs_index sp); apply set_status_absent; exact H.
Qed.

Lemma fold_written_absent : forall c sizes (l : list ((bid * Z) * bspan)),
    (forall id off sp, In ((id, off), sp) l -> writes sp = true -> forall b, In b (c_buffers c) -> bf_id b <> id) ->
    fold_left (fun acc x => written acc sizes x) l c = c.
Proof.
  intros c sizes l. induction l as [| [[id off] sp] l IH]; intros H; [reflexivity |]. cbn [fold_left].
  assert (E : written c sizes ((id, off), sp) = c).
  { unfold written. destruct (writes sp) eqn:W; cbn [negb]; [| reflexivity].
    destruct (bs_index sp); apply set_status_absent; exact (H id off sp (or_introl eq_refl) W). }
  rewrite E. apply IH. intros id' off' sp' Hin. apply (H id' off' sp'). right. exact Hin.
Qed.

(* A's own buffers hold no id of a buffer of another tenant. *)
Lemma own_absent : forall A c id b, inv c -> find_buffer c id = Some b -> bf_owner b <> A ->
    forall b', In b' (c_buffers (own A c)) -> bf_id b' <> id.
Proof.
  intros A c id b Hinv Fb Ho b' Hb' E. unfold own in Hb'. cbn [c_buffers] in Hb'. apply filter_In in Hb'.
  destruct Hb' as [Hb' Eo]. apply Z.eqb_eq in Eo. rewrite <- E in Fb. rewrite (find_buffer_of c b' Hinv Hb') in Fb.
  injection Fb as ->. contradiction.
Qed.

Lemma own_other : forall A B c a c' es, A <> B -> inv c -> res_inv c ->
    check_action c B a = inl (c', es) -> own A c' = own A c.
Proof.
  intros A B c a c' es HAB Hinv Hr H.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - (* B's allocation lies in B's reservation, apart from every freed region of A *)
    destruct (allocate_place _ _ _ _ _ _ _ H) as [_ [Hin _]].
    destruct (allocate_next _ _ _ _ _ _ _ H) as [-> _].
    destruct (in_reservation_some _ _ _ _ Hin) as [rB [HB [HBl HBh]]].
    unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators]. f_equal.
    + rewrite filter_app. cbn [filter bf_owner]. replace (Z.eqb B A) with false by (symmetry; apply Z.eqb_neq; congruence).
      apply app_nil_r.
    + rewrite carve_filter_owner. apply carve_apart. intros r Hrin. apply filter_In in Hrin. destruct Hrin as [Hrin Eo].
      apply Z.eqb_eq in Eo. pose proof Hr as [_ [_ Hf]]. destruct (Hf r Hrin) as [rA [HA [HAl HAh]]]. rewrite Eo in HA.
      destruct (reservations_apart c A B rA rB Hr HAB HA HB); lia.
  - (* B's free: the buffer and its grants are B's, and the merged region is B's *)
    destruct (free_next _ _ _ _ _ H) as [fb [Fb [Ho [_ ->]]]].
    assert (Hid : fst id = B) by (rewrite (owner_of_find c id fb Hr Fb); exact Ho).
    unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators]. f_equal.
    + rewrite filter_comm. apply filter_true. intros b Hb. apply negb_true_iff, bid_eqb_neq.
      apply (own_absent A c id fb Hinv Fb ltac:(congruence)). unfold own. cbn [c_buffers]. exact Hb.
    + rewrite filter_comm. apply filter_true. intros [[gt gn] u] Hg. apply filter_In in Hg. destruct Hg as [_ E].
      cbn [fst] in E. apply Z.eqb_eq in E. apply negb_true_iff, bid_eqb_neq. cbn [fst]. intros Eq.
      rewrite <- Eq in Hid. cbn [fst] in Hid. congruence.
    + exact (release_other A B _ _ _ HAB).
  - (* B's grant is on B's buffer *)
    destruct (grant_next _ _ _ _ _ _ H) as [fb [Fb [Ho [_ [_ ->]]]]].
    assert (Hid : fst id = B) by (rewrite (owner_of_find c id fb Hr Fb); exact Ho).
    unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators]. f_equal.
    rewrite filter_app. cbn [filter fst]. rewrite Hid. replace (Z.eqb B A) with false by (symmetry; apply Z.eqb_neq; congruence).
    apply app_nil_r.
  - (* B's write is to a buffer of B *)
    destruct (write_next _ _ _ _ _ _ _ H) as [fb [Fb [_ [_ Hel]]]].
    assert (Hwr : writable c B id = true).
    { cbn [check_action] in H. rewrite Fb in H. destruct (writable c B id); [reflexivity | discriminate H]. }
    assert (Ho : bf_owner fb <> A).
    { unfold writable in Hwr. rewrite Fb in Hwr. apply andb_true_iff in Hwr. destruct Hwr as [E _].
      apply Z.eqb_eq in E. congruence. }
    destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ ->]]] | rewrite Hel];
      rewrite own_set_status; apply set_status_absent; exact (own_absent A c id fb Hinv Fb Ho).
  - (* B's fill is of a buffer of B *)
    destruct (fill_next _ _ _ _ _ _ _ _ H) as [fb [Fb [_ [_ [_ [_ ->]]]]]].
    assert (Hwr : writable c B id = true).
    { cbn [check_action] in H. rewrite Fb in H. destruct (writable c B id); [reflexivity | discriminate H]. }
    assert (Ho : bf_owner fb <> A).
    { unfold writable in Hwr. rewrite Fb in Hwr. apply andb_true_iff in Hwr. destruct Hwr as [E _].
      apply Z.eqb_eq in E. congruence. }
    rewrite own_set_status. apply set_status_absent. exact (own_absent A c id fb Hinv Fb Ho).
  - (* B's launch writes only buffers of B *)
    destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [Hbind [[r L] [-> _]]]].
    rewrite own_fold_written. apply fold_written_absent. intros id off sp Hin W.
    destruct (check_launch_sound _ _ _ _ _ _ _ _ _ _ _ L) as [Mf' [Hb' [_ [_ [_ [_ [_ [_ [_ Hok]]]]]]]]].
    rewrite Hbind in Hb'. injection Hb' as <-.
    rewrite Forall_forall in Hok. pose proof (Hok _ Hin) as Hx. unfold arg_ok in Hx.
    destruct Hx as [b [Fb [_ [_ [_ [_ [_ [Hw _]]]]]]]]. destruct (Hw W) as [E _].
    unfold writable in E. rewrite Fb in E. apply andb_true_iff in E. destruct E as [E _]. apply Z.eqb_eq in E.
    apply (own_absent A c id b Hinv Fb). congruence.
  - (* B's reservation *)
    destruct (reserve_next _ _ _ _ _ _ H) as [_ [_ [_ [_ [_ [_ ->]]]]]].
    unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators]. f_equal.
    rewrite filter_app. cbn [filter rs_tenant]. replace (Z.eqb B A) with false by (symmetry; apply Z.eqb_neq; congruence).
    apply app_nil_r.
  - (* the end of B's reservation drops B's entries only *)
    destruct (unreserve_next _ _ _ _ H) as [_ [_ [_ ->]]].
    unfold own. cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators]. f_equal.
    + rewrite filter_comm. apply filter_true. intros r Hrin. apply filter_In in Hrin. destruct Hrin as [_ E].
      apply Z.eqb_eq in E. rewrite E. apply negb_true_iff, Z.eqb_neq. exact HAB.
    + rewrite filter_comm. apply filter_true. intros r Hrin. apply filter_In in Hrin. destruct Hrin as [_ E].
      apply Z.eqb_eq in E. rewrite E. apply negb_true_iff, Z.eqb_neq. exact HAB.
Qed.

(* ------------------------------------------------------------------ *)
(* An action of A runs on A's own part as on the whole catalog.        *)

Lemma platform_pool : forall F Mf c1 c2 init, c_pool c1 = c_pool c2 ->
    platform_ok F Mf c1 base init -> platform_ok F Mf c2 base init.
Proof.
  intros F Mf c1 c2 init Hp [H1 H2 H3 H4 H5]. constructor; try assumption.
  intros w Hw. rewrite <- Hp. exact (H4 w Hw).
Qed.

Lemma inits_same : forall F Mf c1 c2 gx gy gz buffers sizes scalars ch,
    args_of c1 sizes (b_params Mf) buffers scalars = args_of c2 sizes (b_params Mf) buffers scalars ->
    inits_of base F Mf c1 gx gy gz buffers sizes scalars ch = inits_of base F Mf c2 gx gy gz buffers sizes scalars ch.
Proof.
  intros F Mf c1 c2 gx gy gz buffers sizes scalars ch H. unfold inits_of, launch_value, raw_value, args.
  rewrite H. reflexivity.
Qed.

Lemma kernel_run_same : forall F prog Mf c1 c2 gx gy gz buffers sizes scalars ch sh0 lm0 m,
    args_of c1 sizes (b_params Mf) buffers scalars = args_of c2 sizes (b_params Mf) buffers scalars ->
    kernel_run base F prog Mf c1 gx gy gz buffers sizes scalars ch sh0 lm0 m =
    kernel_run base F prog Mf c2 gx gy gz buffers sizes scalars ch sh0 lm0 m.
Proof.
  intros F prog Mf c1 c2 gx gy gz buffers sizes scalars ch sh0 lm0 m H. unfold kernel_run.
  rewrite (inits_same F Mf c1 c2 gx gy gz buffers sizes scalars ch H). reflexivity.
Qed.

Lemma hardware_same : forall F Mf c1 c2 gx gy gz buffers sizes scalars ch, c_pool c1 = c_pool c2 ->
    args_of c1 sizes (b_params Mf) buffers scalars = args_of c2 sizes (b_params Mf) buffers scalars ->
    hardware_ok base F Mf c1 gx gy gz buffers sizes scalars ch -> hardware_ok base F Mf c2 gx gy gz buffers sizes scalars ch.
Proof.
  intros F Mf c1 c2 gx gy gz buffers sizes scalars ch Hp Ha [H1 H2 H3 H4]. constructor; try assumption.
  intros n. rewrite <- (inits_same F Mf c1 c2 gx gy gz buffers sizes scalars ch Ha).
  exact (platform_pool F Mf c1 c2 _ Hp (H2 n)).
Qed.

Lemma effect_own : forall ch A c a m e m',
    (forall op gx gy gz buffers sizes scalars guarded, a = Launch op gx gy gz buffers sizes scalars guarded ->
       forall id off, In (id, off) buffers -> find_buffer (own A c) id = find_buffer c id) ->
    effect_run base kernels ch c a m e m' -> effect_run base kernels ch (own A c) a m e m'.
Proof.
  intros ch A c a m e m' Hargs H. destruct H as [a m o n | a m o data | a m o n e s v
    | m op gx gy gz buffers sizes scalars guarded args checks Mf prog F sh0 lm0 Hb Hk Hhw Hck Hl
    | m op gx gy gz shared buffers sizes scalars guarded args checks Hck].
  - apply RunClear.
  - apply RunUpload.
  - apply RunMemset.
  - assert (Ha : args_of (own A c) sizes (b_params Mf) buffers scalars = args_of c sizes (b_params Mf) buffers scalars)
      by (apply args_of_ext; exact (Hargs op gx gy gz buffers sizes scalars guarded eq_refl)).
    rewrite <- (kernel_run_same F prog Mf (own A c) c gx gy gz buffers sizes scalars ch sh0 lm0 m Ha).
    apply (RunKernel base kernels ch (own A c) m op gx gy gz buffers sizes scalars guarded args checks Mf prog F sh0 lm0);
      [exact Hb | exact Hk | | exact Hck | exact Hl].
    apply (hardware_same F Mf c (own A c) gx gy gz buffers sizes scalars ch eq_refl); [symmetry; exact Ha | exact Hhw].
  - apply RunSkipped. exact Hck.
Qed.

Lemma effects_own : forall ch A c a m es m',
    (forall op gx gy gz buffers sizes scalars guarded, a = Launch op gx gy gz buffers sizes scalars guarded ->
       forall id off, In (id, off) buffers -> find_buffer (own A c) id = find_buffer c id) ->
    effects_run base kernels ch c a m es m' -> effects_run base kernels ch (own A c) a m es m'.
Proof.
  intros ch A c a m es m' Hargs H. induction H as [m | m e es m1 m2 He Hr IH].
  - apply RunNone.
  - apply (RunMore base kernels ch (own A c) a m e es m1 m2); [exact (effect_own ch A c a m e m1 Hargs He) | exact IH].
Qed.

Lemma run_action_own : forall ch A c m a c' m', inv c -> res_inv c -> ungranted A c -> find_reservation c A <> None ->
    run_action base kernels ch A {| w_cat := c; w_mem := m |} a {| w_cat := c'; w_mem := m' |} ->
    run_action base kernels ch A {| w_cat := own A c; w_mem := m |} a {| w_cat := own A c'; w_mem := m' |}.
Proof.
  intros ch A c m a c' m' Hinv Hr Hu Hres H. inversion H as [c0 m0 a0 c1 es m1 Hc He]; subst.
  apply (RunAction base kernels ch A (own A c) m a (own A c') es m').
  - rewrite (own_action A c a Hinv Hr Hu Hres), Hc. reflexivity.
  - apply effects_own; [| exact He]. intros op gx gy gz buffers sizes scalars guarded -> id off Hin.
    destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ Hc) as [Mf [_ [[r L] _]]].
    exact (launch_args_own A c op gx gy gz buffers sizes scalars guarded r Hinv Hr Hu L id off Hin).
Qed.

(* ------------------------------------------------------------------ *)
(* Steps.                                                              *)

Definition own_decision (A : Z) (d : decision) : decision :=
  match d with Accepted c es => Accepted (own A c) es | Refused r => Refused r end.

Lemma own_check_actions : forall A acts c pos done, inv c -> res_inv c -> ungranted A c ->
    find_reservation c A <> None -> ~ In Unreserve acts ->
    check_actions (own A c) A pos acts done = own_decision A (check_actions c A pos acts done).
Proof.
  intros A acts. induction acts as [| a acts IH]; intros c pos done Hinv Hr Hu Hres Hun; [reflexivity |].
  cbn [check_actions]. rewrite (own_action A c a Hinv Hr Hu Hres).
  destruct (check_action c A a) as [[c1 es1] | r] eqn:E; cbn [own_result]; [| reflexivity].
  apply IH.
  - exact (check_action_inv _ _ _ _ _ Hinv E).
  - exact (res_action _ _ _ _ _ Hinv Hr E).
  - apply (ungranted_action A c A a c1 es1 Hu); [intros id u _; right; reflexivity | exact E].
  - destruct (find_reservation c A) as [r |] eqn:Er; [| exfalso; exact (Hres eq_refl)].
    rewrite (reservation_action A c A a c1 es1 r Er (fun H => False_ind _ (Hun (or_introl H))) E).
    discriminate.
  - intros H. exact (Hun (or_intror H)).
Qed.

(* A's step is decided on A's own part as on the whole catalog. *)
Lemma own_check_step : forall A c s, s_tenant s = A -> inv c -> res_inv c -> ungranted A c ->
    find_reservation c A <> None -> ~ In Unreserve (s_actions s) ->
    check_step (own A c) s = own_decision A (check_step c s).
Proof.
  intros A c s Ht Hinv Hr Hu Hres Hun. unfold check_step. rewrite Ht.
  exact (own_check_actions A _ c 0 [] Hinv Hr Hu Hres Hun).
Qed.

(* So two catalogs with the same own part of A decide A's step alike: the
   same refusal, or the same effects and own part after it. *)
Theorem own_decision_same : forall A c1 c2 s, s_tenant s = A -> inv c1 -> inv c2 -> res_inv c1 -> res_inv c2 ->
    ungranted A c1 -> ungranted A c2 -> find_reservation c1 A <> None -> ~ In Unreserve (s_actions s) ->
    own A c1 = own A c2 -> own_decision A (check_step c1 s) = own_decision A (check_step c2 s).
Proof.
  intros A c1 c2 s Ht Hi1 Hi2 Hr1 Hr2 Hu1 Hu2 Hres Hun Hown.
  assert (Hres2 : find_reservation c2 A <> None) by (rewrite <- find_reservation_own, <- Hown, find_reservation_own; exact Hres).
  rewrite <- (own_check_step A c1 s Ht Hi1 Hr1 Hu1 Hres Hun), <- (own_check_step A c2 s Ht Hi2 Hr2 Hu2 Hres2 Hun), Hown.
  reflexivity.
Qed.

(* The invariants that every run of actions keeps. *)
Lemma ok_actions : forall A t chs w acts w', sound w -> res_inv (w_cat w) -> ungranted A (w_cat w) ->
    (forall id u, In (Grant id u) acts -> u <> A \/ t = A) ->
    run_actions base kernels t chs w acts w' -> sound w' /\ res_inv (w_cat w') /\ ungranted A (w_cat w').
Proof.
  intros A t chs w acts w' Hs Hr Hu Hg H. induction H as [w | ch chs w a w1 acts w2 H1 _ IH].
  - split; [exact Hs | split; assumption].
  - pose proof (sound_action _ _ _ _ _ Hs H1) as Hs1.
    inversion H1 as [c m a0 c1 es m1 Hc _]; subst. cbn [w_cat] in *.
    destruct Hs as [[Hinv _] _].
    apply IH; [exact Hs1 | exact (res_action _ _ _ _ _ Hinv Hr Hc) | |].
    + apply (ungranted_action A c t a c1 es Hu); [| exact Hc]. intros id u ->. exact (Hg id u (or_introl eq_refl)).
    + intros id u Hin. exact (Hg id u (or_intror Hin)).
Qed.

Lemma reservation_actions : forall A t chs w acts w' r, find_reservation (w_cat w) A = Some r ->
    (In Unreserve acts -> t <> A) -> run_actions base kernels t chs w acts w' -> find_reservation (w_cat w') A = Some r.
Proof.
  intros A t chs w acts w' r Hr Hun H. induction H as [w | ch chs w a w1 acts w2 H1 _ IH]; [exact Hr |].
  inversion H1 as [c m a0 c1 es m1 Hc _]; subst. cbn [w_cat] in *. apply IH.
  - apply (reservation_action A c t a c1 es r Hr); [| exact Hc]. intros ->. exact (Hun (or_introl eq_refl)).
  - intros Hin. exact (Hun (or_intror Hin)).
Qed.

(* A run of actions of another tenant leaves A's own part and the bytes that A
   observes as they are. *)
Lemma other_actions : forall A B chs w acts w', A <> B -> sound w -> res_inv (w_cat w) -> ungranted A (w_cat w) ->
    (forall id, ~ In (Grant id A) acts) -> run_actions base kernels B chs w acts w' ->
    own A (w_cat w') = own A (w_cat w) /\ forall y, observable (w_cat w) A y -> w_mem w' y = w_mem w y.
Proof.
  intros A B chs w acts w' HAB Hs Hr Hu Hg H. induction H as [w | ch chs w a w1 acts w2 H1 _ IH].
  - split; [reflexivity | intros y _; reflexivity].
  - pose proof (sound_action _ _ _ _ _ Hs H1) as Hs1.
    destruct w as [c m]. destruct w1 as [c1 m1].
    inversion H1 as [c0 m0 a0 c1' es m1' Hc _]; subst. cbn [w_cat w_mem] in *.
    pose proof Hs as [[Hinv _] _]. pose proof Hs1 as [[Hinv1 _] _].
    assert (Hga : forall id u, a = Grant id u -> u <> A) by (intros id u -> <-; exact (Hg id (or_introl eq_refl))).
    pose proof (res_action _ _ _ _ _ Hinv Hr Hc) as Hr1.
    assert (Hu1 : ungranted A c1).
    { apply (ungranted_action A c B a c1 es Hu); [| exact Hc]. intros id u E. left. exact (Hga id u E). }
    pose proof (own_other A B c a c1 es HAB Hinv Hr Hc) as Hown.
    destruct (IH Hs1 Hr1 Hu1 (fun id Hin => Hg id (or_intror Hin))) as [Hown' Hmem'].
    split; [rewrite Hown', Hown; reflexivity |].
    intros y Hy.
    assert (Hy1 : observable c1 A y).
    { apply (observable_own A c1 y Hinv1 Hu1). rewrite Hown. apply (observable_own A c y Hinv Hu). exact Hy. }
    rewrite (Hmem' y Hy1). exact (proj2 (other_action ch A B c m a c1 m1 HAB Hs Hga H1 y Hy1)).
Qed.

(* A run of A's actions runs on A's own part as on the whole catalog. *)
Lemma run_actions_own : forall A chs w acts w', sound w -> res_inv (w_cat w) -> ungranted A (w_cat w) ->
    find_reservation (w_cat w) A <> None -> ~ In Unreserve acts ->
    run_actions base kernels A chs w acts w' ->
    run_actions base kernels A chs {| w_cat := own A (w_cat w); w_mem := w_mem w |} acts
                                   {| w_cat := own A (w_cat w'); w_mem := w_mem w' |}.
Proof.
  intros A chs w acts w' Hs Hr Hu Hres Hun H. induction H as [w | ch chs w a w1 acts w2 H1 _ IH].
  - apply RunDone.
  - pose proof (sound_action _ _ _ _ _ Hs H1) as Hs1.
    destruct w as [c m]. destruct w1 as [c1 m1].
    inversion H1 as [c0 m0 a0 c1' es m1' Hc _]; subst. cbn [w_cat w_mem] in *.
    pose proof Hs as [[Hinv _] _].
    apply (RunNext base kernels A ch chs _ a {| w_cat := own A c1; w_mem := m1 |}).
    + exact (run_action_own ch A c m a c1 m1 Hinv Hr Hu Hres H1).
    + apply IH; [exact Hs1 | exact (res_action _ _ _ _ _ Hinv Hr Hc) | | |].
      * apply (ungranted_action A c A a c1 es Hu); [intros id u _; right; reflexivity | exact Hc].
      * destruct (find_reservation c A) as [r |] eqn:Er; [| exfalso; exact (Hres eq_refl)].
        rewrite (reservation_action A c A a c1 es r Er (fun E => False_ind _ (Hun (or_introl E))) Hc).
        discriminate.
      * intros Hin. exact (Hun (or_intror Hin)).
Qed.

(* The two runs' worlds agree for A: their invariants hold, A holds its
   reservation, their own parts of A are equal, and their memories agree on
   what A observes. *)
Definition related (A : Z) (w1 w2 : world) : Prop :=
  sound w1 /\ sound w2 /\ res_inv (w_cat w1) /\ res_inv (w_cat w2) /\
  ungranted A (w_cat w1) /\ ungranted A (w_cat w2) /\ find_reservation (w_cat w1) A <> None /\
  own A (w_cat w1) = own A (w_cat w2) /\ agree_on (w_cat w1) A (w_mem w1) (w_mem w2).

Lemma observable_same : forall A c1 c2 y, inv c1 -> inv c2 -> ungranted A c1 -> ungranted A c2 ->
    own A c1 = own A c2 -> observable c1 A y -> observable c2 A y.
Proof.
  intros A c1 c2 y Hi1 Hi2 Hu1 Hu2 Hown Hy. apply (observable_own A c2 y Hi2 Hu2). rewrite <- Hown.
  apply (observable_own A c1 y Hi1 Hu1). exact Hy.
Qed.

(* A step of another tenant in either run keeps the runs related. *)
Lemma other_step : forall A chs w s w', s_tenant s <> A -> (forall id, ~ In (Grant id A) (s_actions s)) ->
    sound w -> res_inv (w_cat w) -> ungranted A (w_cat w) -> run_step base kernels chs w s w' ->
    sound w' /\ res_inv (w_cat w') /\ ungranted A (w_cat w') /\ own A (w_cat w') = own A (w_cat w) /\
    (forall r, find_reservation (w_cat w) A = Some r -> find_reservation (w_cat w') A = Some r) /\
    (forall y, observable (w_cat w) A y -> w_mem w' y = w_mem w y).
Proof.
  intros A chs w s w' Ht Hg Hs Hr Hu H.
  destruct H as [w s c' es w' Hc Hrun | w s r Hc].
  - assert (HAB : A <> s_tenant s) by (intros E; apply Ht; symmetry; exact E).
    destruct (ok_actions A (s_tenant s) chs w (s_actions s) w' Hs Hr Hu) as (Hs' & Hr' & Hu'); [| exact Hrun |].
    { intros id u Hin. left. intros ->. exact (Hg id Hin). }
    destruct (other_actions A (s_tenant s) chs w (s_actions s) w' HAB Hs Hr Hu Hg Hrun) as [Hown Hmem].
    split; [exact Hs' |]. split; [exact Hr' |]. split; [exact Hu' |]. split; [exact Hown |].
    split; [| exact Hmem].
    intros r Er. exact (reservation_actions A (s_tenant s) chs w (s_actions s) w' r Er (fun _ => Ht) Hrun).
  - split; [exact Hs |]. split; [exact Hr |]. split; [exact Hu |]. split; [reflexivity |].
    split; [intros r' Er; exact Er | intros y _; reflexivity].
Qed.

Lemma other_step1 : forall A chs w1 s w1' w2, s_tenant s <> A -> (forall id, ~ In (Grant id A) (s_actions s)) ->
    related A w1 w2 -> run_step base kernels chs w1 s w1' -> related A w1' w2.
Proof.
  intros A chs w1 s w1' w2 Ht Hg (Hs1 & Hs2 & Hr1 & Hr2 & Hu1 & Hu2 & Hres & Hown & Hag) H.
  destruct (other_step A chs w1 s w1' Ht Hg Hs1 Hr1 Hu1 H) as (Hs1' & Hr1' & Hu1' & Hown' & Hkeep & Hmem).
  split; [exact Hs1' |]. split; [exact Hs2 |]. split; [exact Hr1' |]. split; [exact Hr2 |].
  split; [exact Hu1' |]. split; [exact Hu2 |].
  split; [destruct (find_reservation (w_cat w1) A) as [r |] eqn:Er; [rewrite (Hkeep r eq_refl); discriminate | contradiction] |].
  split; [rewrite Hown'; exact Hown |].
  intros y Hy. pose proof Hs1 as [[Hi1 _] _]. pose proof Hs1' as [[Hi1' _] _].
  assert (Hy0 : observable (w_cat w1) A y) by (apply (observable_same A (w_cat w1') (w_cat w1) y Hi1' Hi1 Hu1' Hu1 Hown' Hy)).
  rewrite (Hmem y Hy0). exact (Hag y Hy0).
Qed.

Lemma other_step2 : forall A chs w1 w2 s w2', s_tenant s <> A -> (forall id, ~ In (Grant id A) (s_actions s)) ->
    related A w1 w2 -> run_step base kernels chs w2 s w2' -> related A w1 w2'.
Proof.
  intros A chs w1 w2 s w2' Ht Hg (Hs1 & Hs2 & Hr1 & Hr2 & Hu1 & Hu2 & Hres & Hown & Hag) H.
  destruct (other_step A chs w2 s w2' Ht Hg Hs2 Hr2 Hu2 H) as (Hs2' & Hr2' & Hu2' & Hown' & _ & Hmem).
  split; [exact Hs1 |]. split; [exact Hs2' |]. split; [exact Hr1 |]. split; [exact Hr2' |].
  split; [exact Hu1 |]. split; [exact Hu2' |]. split; [exact Hres |].
  split; [rewrite Hown'; exact Hown |].
  intros y Hy. pose proof Hs1 as [[Hi1 _] _]. pose proof Hs2 as [[Hi2 _] _].
  assert (Hy2 : observable (w_cat w2) A y) by (apply (observable_same A (w_cat w1) (w_cat w2) y Hi1 Hi2 Hu1 Hu2 Hown Hy)).
  rewrite (Hmem y Hy2). exact (Hag y Hy).
Qed.

(* A step of A, in both runs with the same choices of the hardware, keeps the
   runs related: it is decided alike, and runs alike on A's own part. *)
Lemma self_step : forall A chs w1 w2 s w1' w2', s_tenant s = A -> ~ In Unreserve (s_actions s) ->
    related A w1 w2 -> run_step base kernels chs w1 s w1' -> run_step base kernels chs w2 s w2' ->
    related A w1' w2'.
Proof.
  intros A chs w1 w2 s w1' w2' Ht Hun (Hs1 & Hs2 & Hr1 & Hr2 & Hu1 & Hu2 & Hres & Hown & Hag) H1 H2.
  pose proof Hs1 as [[Hi1 _] _]. pose proof Hs2 as [[Hi2 _] _].
  assert (Hres2 : find_reservation (w_cat w2) A <> None)
    by (rewrite <- find_reservation_own, <- Hown, find_reservation_own; exact Hres).
  pose proof (own_decision_same A (w_cat w1) (w_cat w2) s Ht Hi1 Hi2 Hr1 Hr2 Hu1 Hu2 Hres Hun Hown) as Hdec.
  destruct H1 as [w1 s c1' es1 w1' Hc1 Hrun1 | w1 s r1 Hc1];
    destruct H2 as [w2 s c2' es2 w2' Hc2 Hrun2 | w2 s r2 Hc2]; rewrite ?Hc1, ?Hc2 in Hdec; cbn [own_decision] in Hdec;
    try discriminate Hdec.
  - (* both accept: the runs on the own parts start from one catalog *)
    rewrite Ht in Hrun1, Hrun2.
    pose proof (run_actions_own A chs w1 (s_actions s) w1' Hs1 Hr1 Hu1 Hres Hun Hrun1) as O1.
    pose proof (run_actions_own A chs w2 (s_actions s) w2' Hs2 Hr2 Hu2 Hres2 Hun Hrun2) as O2.
    rewrite <- Hown in O2.
    assert (S1 : sound {| w_cat := own A (w_cat w1); w_mem := w_mem w1 |}).
    { apply sound_own; [destruct w1; exact Hs1 | exact Hr1]. }
    assert (S2 : sound {| w_cat := own A (w_cat w1); w_mem := w_mem w2 |}).
    { rewrite Hown. apply sound_own; [destruct w2; exact Hs2 | exact Hr2]. }
    assert (G : agree_on (own A (w_cat w1)) A (w_mem w1) (w_mem w2))
      by (apply (agree_own A (w_cat w1) _ _ Hi1 Hu1); exact Hag).
    destruct (actions_isolation A A chs chs _ _ (s_actions s) _ _ S1 S2 eq_refl G
                (fun _ => eq_refl) (fun H => False_ind _ (H eq_refl)) O1 O2)
      as (_ & _ & Hc' & Hag'). cbn [w_cat w_mem] in Hc', Hag'.
    assert (Hg : forall id u, In (Grant id u) (s_actions s) -> u <> A \/ A = A) by (intros; right; reflexivity).
    destruct (ok_actions A A chs w1 (s_actions s) w1' Hs1 Hr1 Hu1 Hg Hrun1) as (Hs1' & Hr1' & Hu1').
    destruct (ok_actions A A chs w2 (s_actions s) w2' Hs2 Hr2 Hu2 Hg Hrun2) as (Hs2' & Hr2' & Hu2').
    pose proof Hs1' as [[Hi1' _] _].
    split; [exact Hs1' |]. split; [exact Hs2' |]. split; [exact Hr1' |]. split; [exact Hr2' |].
    split; [exact Hu1' |]. split; [exact Hu2' |].
    split; [destruct (find_reservation (w_cat w1) A) as [r |] eqn:Er; [| contradiction];
            rewrite (reservation_actions A A chs w1 (s_actions s) w1' r Er (fun H => False_ind _ (Hun H)) Hrun1);
            discriminate |].
    split; [exact Hc' |]. apply (agree_own A (w_cat w1') _ _ Hi1' Hu1'). exact Hag'.
  - (* both refuse *)
    split; [exact Hs1 |]. split; [exact Hs2 |]. split; [exact Hr1 |]. split; [exact Hr2 |].
    split; [exact Hu1 |]. split; [exact Hu2 |]. split; [exact Hres |]. split; [exact Hown | exact Hag].
Qed.

(* ------------------------------------------------------------------ *)
(* Pipelines whose other tenants differ.                               *)

(* A's steps of a run, each with the choices of the hardware for it. *)
Definition own_steps (A : Z) (steps : list step) (chss : list (list choice)) : list (step * list choice) :=
  filter (fun p => Z.eqb (s_tenant (fst p)) A) (combine steps chss).

(* A's reservation lasts the whole run: no step of A ends it. *)
Definition keeps_reservation (A : Z) (steps : list step) : Prop :=
  forall s, In s steps -> s_tenant s = A -> ~ In Unreserve (s_actions s).

Lemma own_steps_cons : forall A s ss chs chss,
    own_steps A (s :: ss) (chs :: chss) =
    if Z.eqb (s_tenant s) A then (s, chs) :: own_steps A ss chss else own_steps A ss chss.
Proof. intros A s ss chs chss. reflexivity. Qed.

Lemma isolation_runs : forall n A steps1 steps2 chss1 chss2 w1 w2 w1' w2',
    (List.length steps1 + List.length steps2 <= n)%nat -> related A w1 w2 ->
    no_grant_to A steps1 -> no_grant_to A steps2 -> keeps_reservation A steps1 ->
    own_steps A steps1 chss1 = own_steps A steps2 chss2 ->
    run_steps base kernels chss1 w1 steps1 w1' -> run_steps base kernels chss2 w2 steps2 w2' -> related A w1' w2'.
Proof.
  induction n as [| n IH]; intros A steps1 steps2 chss1 chss2 w1 w2 w1' w2' Hlen Hrel Hng1 Hng2 Hkeep Hown H1 H2.
  - destruct steps1; [| cbn in Hlen; lia]. destruct steps2; [| cbn in Hlen; lia].
    inversion H1; subst. inversion H2; subst. exact Hrel.
  - destruct steps1 as [| s1 ss1].
    + destruct steps2 as [| s2 ss2]; [inversion H1; subst; inversion H2; subst; exact Hrel |].
      inversion H1; subst.
      inversion H2 as [| chs2 chss2' ? ? w2m ? ? Hs2 Hr2]; subst.
      rewrite own_steps_cons in Hown. destruct (Z.eqb_spec (s_tenant s2) A) as [E2 | N2]; [discriminate Hown |].
      apply (IH A [] ss2 [] chss2' w1' w2m w1' w2'); [cbn in Hlen |- *; lia | | exact Hng1 | | exact Hkeep | exact Hown |
                                                  apply RunEnd | exact Hr2].
      * apply (other_step2 A chs2 w1' w2 s2 w2m N2); [| exact Hrel | exact Hs2].
        intros id Hin. exact (Hng2 s2 id (or_introl eq_refl) N2 Hin).
      * intros s' id Hin. exact (Hng2 s' id (or_intror Hin)).
    + pose proof H1 as H1c. inversion H1 as [| chs1 chss1' ? ? w1m ? ? Hs1 Hr1]; subst.
      destruct (Z.eqb_spec (s_tenant s1) A) as [E1 | N1].
      * destruct steps2 as [| s2 ss2].
        { rewrite own_steps_cons, (proj2 (Z.eqb_eq _ _) E1) in Hown. cbn in Hown. discriminate Hown. }
        pose proof H2 as H2c. inversion H2 as [| chs2 chss2' ? ? w2m ? ? Hs2 Hr2]; subst chss2.
        destruct (Z.eqb_spec (s_tenant s2) A) as [E2 | N2].
        -- (* both runs take a step of A: the same step, with the same choices *)
           rewrite !own_steps_cons, (proj2 (Z.eqb_eq _ _) E1), (proj2 (Z.eqb_eq _ _) E2) in Hown.
           injection Hown as <- <- Hown.
           apply (IH A ss1 ss2 chss1' chss2' w1m w2m w1' w2'); [cbn in Hlen; lia | | | | | exact Hown | exact Hr1 | exact Hr2].
           ++ exact (self_step A chs1 w1 w2 s1 w1m w2m E1 (Hkeep s1 (or_introl eq_refl) E1) Hrel Hs1 Hs2).
           ++ intros s' id Hin. exact (Hng1 s' id (or_intror Hin)).
           ++ intros s' id Hin. exact (Hng2 s' id (or_intror Hin)).
           ++ intros s' Hin. exact (Hkeep s' (or_intror Hin)).
        -- (* the second run takes a step of another tenant first *)
           rewrite (own_steps_cons A s2 ss2), (proj2 (Z.eqb_neq _ _) N2) in Hown.
           apply (IH A (s1 :: ss1) ss2 (chs1 :: chss1') chss2' w1 w2m w1' w2'); [cbn in Hlen |- *; lia | | exact Hng1 | |
                                                                                  exact Hkeep | exact Hown | exact H1c | exact Hr2].
           ++ apply (other_step2 A chs2 w1 w2 s2 w2m N2); [| exact Hrel | exact Hs2].
              intros id Hin. exact (Hng2 s2 id (or_introl eq_refl) N2 Hin).
           ++ intros s' id Hin. exact (Hng2 s' id (or_intror Hin)).
      * (* the first run takes a step of another tenant *)
        rewrite (own_steps_cons A s1 ss1), (proj2 (Z.eqb_neq _ _) N1) in Hown.
        apply (IH A ss1 steps2 chss1' chss2 w1m w2 w1' w2'); [cbn in Hlen; lia | | | exact Hng2 | | exact Hown | exact Hr1 | exact H2].
        -- apply (other_step1 A chs1 w1 s1 w1m w2 N1); [| exact Hrel | exact Hs1].
           intros id Hin. exact (Hng1 s1 id (or_introl eq_refl) N1 Hin).
        -- intros s' id Hin. exact (Hng1 s' id (or_intror Hin)).
        -- intros s' Hin. exact (Hkeep s' (or_intror Hin)).
Qed.

(* Tenant isolation: two runs whose steps of A are the same, each with the
   same choices of the hardware, from worlds whose own parts of A are equal
   and whose memories agree on what A observes, end with equal own parts of
   A and memories that agree on what A observes. The other tenants' steps,
   with their regions, sizes, written bytes, grids, and scalars, their
   reservations, their memory, and the hardware's choices for their launches
   may differ between the runs. A holds its reservation through both runs,
   and no buffer of another tenant is granted to A. *)
Theorem tenant_isolation : forall A steps1 steps2 chss1 chss2 w1 w2 w1' w2',
    sound w1 -> sound w2 -> res_inv (w_cat w1) -> res_inv (w_cat w2) ->
    ungranted A (w_cat w1) -> ungranted A (w_cat w2) -> find_reservation (w_cat w1) A <> None ->
    own A (w_cat w1) = own A (w_cat w2) -> agree_on (w_cat w1) A (w_mem w1) (w_mem w2) ->
    no_grant_to A steps1 -> no_grant_to A steps2 -> keeps_reservation A steps1 ->
    own_steps A steps1 chss1 = own_steps A steps2 chss2 ->
    run_steps base kernels chss1 w1 steps1 w1' -> run_steps base kernels chss2 w2 steps2 w2' ->
    own A (w_cat w1') = own A (w_cat w2') /\ agree_on (w_cat w1') A (w_mem w1') (w_mem w2').
Proof.
  intros A steps1 steps2 chss1 chss2 w1 w2 w1' w2' Hs1 Hs2 Hr1 Hr2 Hu1 Hu2 Hres Hown Hag Hng1 Hng2 Hkeep Hsteps H1 H2.
  destruct (isolation_runs (List.length steps1 + List.length steps2) A steps1 steps2 chss1 chss2 w1 w2 w1' w2'
              (le_n _) (conj Hs1 (conj Hs2 (conj Hr1 (conj Hr2 (conj Hu1 (conj Hu2 (conj Hres (conj Hown Hag))))))))
              Hng1 Hng2 Hkeep Hsteps H1 H2)
    as (_ & _ & _ & _ & _ & _ & _ & Hown' & Hag').
  split; [exact Hown' | exact Hag'].
Qed.

End Isolation.
