(* PipelineGrants.v: what a tenant observes of a pipeline is a function of
   the tenant's steps and of what the other tenants grant it.

   `tenant_isolation` (PipelineIsolation.v) covers a tenant A without
   grants to it (`ungranted`, `no_grant_to`). A grant is how a tenant shows
   another tenant its data: after `Grant id A` by the owner of buffer id, A
   may read the buffer, and the buffer keeps its bytes (the step check
   refuses every write of a granted buffer: `writable`) until its owner
   frees it, which ends the grant. Here A may receive grants.

   What is released to A is the sequence of events of A (`event`): A's own
   steps, each with the choices of the hardware for it (`OStep`); each grant
   to A (`OGrant`), with the buffer's record (its id, offset in the pool,
   size, owner, and status) and its bytes when granted; and each end of a
   grant to A (`ORevoke`), the owner's free of a buffer granted to A, with
   the buffer's id. `observe_steps` is a run of steps (`run_steps`) together
   with its sequence of events of A; every run has one (`run_observe_steps`).
   The record holds the offset and the status because A's launches read
   them: a launch passes the address of a granted buffer to the kernel, and
   an index span of a granted buffer is checked by its status.

   Proved: `tenant_declassification`, two runs with the same sequence of
   events of A, from worlds whose own parts of A are equal, whose buffers
   granted to A are equal (`received`), and whose memories agree on what A
   observes, end with equal own parts of A, equal buffers granted to A, and
   memories that agree on what A observes. The other tenants' steps, their
   memory, and the hardware's choices for their launches may differ between
   the runs, and so may the order of their other actions relative to A's
   events.
   A holds its reservation through the first run (`keeps_reservation`), as
   in `tenant_isolation`, and the grants of each catalog are distinct
   (`NoDup (c_grants c)`), which every accepted action keeps
   (`grants_nodup_action`).

   The proof follows PipelineIsolation.v. An action of A is decided and run
   on `view A c`, A's own part with the buffers granted to A, as on the
   whole catalog (`view_action`, `run_action_view`); a step of another
   tenant is a sequence of actions, and each of them keeps every byte that
   A observes after it (`other_action`), or grants a buffer to A, or ends a
   grant to A (`silent_action`, `grant_sync`, `revoke_sync`). *)

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
Require Import OpGate.PipelineIsolation.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Lists and catalogs.                                                 *)

Lemma existsb_none : forall (X : Type) (f : X -> bool) l, (forall x, In x l -> f x = false) -> existsb f l = false.
Proof.
  intros X f l H. induction l as [| x l IH]; [reflexivity |]. cbn [existsb].
  rewrite (H x (or_introl eq_refl)). apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma NoDup_app_apart : forall (X : Type) (l1 l2 : list X), NoDup l1 -> NoDup l2 ->
    (forall x, In x l1 -> ~ In x l2) -> NoDup (l1 ++ l2).
Proof.
  intros X l1 l2 H1. induction H1 as [| x l1 Hx H1 IH]; intros H2 H; [exact H2 |].
  cbn [app]. constructor.
  - intros Hin. apply in_app_or in Hin. destruct Hin as [Hin | Hin]; [contradiction |].
    exact (H x (or_introl eq_refl) Hin).
  - apply IH; [exact H2 |]. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma map_eq_in : forall (X Y : Type) (f g : X -> Y) l x, map f l = map g l -> In x l -> f x = g x.
Proof.
  intros X Y f g l x. induction l as [| z l IH]; intros H Hin; [destruct Hin |].
  cbn [map] in H. injection H as Hz Hl. destruct Hin as [<- | Hin]; [exact Hz | exact (IH Hl Hin)].
Qed.

Lemma catalog_ext : forall c1 c2, c_pool c1 = c_pool c2 -> c_buffers c1 = c_buffers c2 -> c_grants c1 = c_grants c2 ->
    c_freed c1 = c_freed c2 -> c_reserved c1 = c_reserved c2 -> c_operators c1 = c_operators c2 -> c1 = c2.
Proof. intros [p1 b1 g1 f1 r1 o1] [p2 b2 g2 f2 r2 o2]; cbn; intros -> -> -> -> -> ->; reflexivity. Qed.

Lemma has_grant_granted : forall c id u, has_grant c id u = true -> granted c id = true.
Proof.
  intros c id u H. unfold has_grant in H. unfold granted. apply existsb_exists in H. destruct H as [g [Hg E]].
  apply andb_true_iff in E. apply existsb_exists. exists g. split; [exact Hg | exact (proj1 E)].
Qed.

(* A buffer that a tenant may write is ungranted. *)
Lemma writable_ungranted : forall c t id u, writable c t id = true -> has_grant c id u = false.
Proof.
  intros c t id u H. unfold writable in H. destruct (find_buffer c id); [| discriminate H].
  apply andb_true_iff in H. destruct H as [_ H]. apply negb_true_iff in H.
  destruct (has_grant c id u) eqn:E; [| reflexivity]. rewrite (has_grant_granted c id u E) in H. discriminate H.
Qed.

Lemma has_grant_exists : forall c id u, has_grant c id u = true <-> exists g, In g (c_grants c) /\ fst g = id /\ snd g = u.
Proof.
  intros c id u. unfold has_grant. rewrite existsb_exists. split.
  - intros [g [Hg E]]. rewrite andb_true_iff, bid_eqb_eq, Z.eqb_eq in E. exists g. split; [exact Hg | exact E].
  - intros [g [Hg [Ei Eu]]]. exists g. split; [exact Hg |]. rewrite Ei, Eu, bid_eqb_refl, Z.eqb_refl. reflexivity.
Qed.

(* The owner of a granted buffer differs from its grantee. *)
Lemma has_grant_owner : forall c id u b, inv c -> has_grant c id u = true -> find_buffer c id = Some b -> bf_owner b <> u.
Proof.
  intros c id u b Hinv Hg Fb. apply has_grant_exists in Hg. destruct Hg as [g [Hg [Ei Eu]]].
  pose proof Hinv as [_ [_ [_ Hgr]]]. destruct (Hgr g Hg) as [b' [Hb' [Hid Ho]]].
  rewrite <- Ei, <- Hid, (find_buffer_of c b' Hinv Hb') in Fb. injection Fb as <-. congruence.
Qed.

(* ------------------------------------------------------------------ *)
(* The buffers granted to a tenant.                                    *)

(* The grants to A, in the order of the catalog. *)
Definition grants_to (A : Z) (c : catalog) : list (bid * Z) := filter (fun g => Z.eqb (snd g) A) (c_grants c).

Definition granted_buffer (c : catalog) (g : bid * Z) : list buffer :=
  match find_buffer c (fst g) with Some b => [b] | None => [] end.

(* The buffers granted to A, in the order of their grants. *)
Definition received (A : Z) (c : catalog) : list buffer := flat_map (granted_buffer c) (grants_to A c).

(* A's own part o with the buffers sh granted to A. *)
Definition extend (A : Z) (sh : list buffer) (o : catalog) : catalog :=
  {| c_pool := c_pool o; c_buffers := sh ++ c_buffers o;
     c_grants := map (fun b => (bf_id b, A)) sh ++ c_grants o;
     c_freed := c_freed o; c_reserved := c_reserved o; c_operators := c_operators o |}.

(* What an action of A sees of the catalog: A's own part and the buffers
   granted to it. *)
Definition view (A : Z) (c : catalog) : catalog := extend A (received A c) (own A c).

Lemma view_same : forall A c1 c2, own A c1 = own A c2 -> received A c1 = received A c2 -> view A c1 = view A c2.
Proof. intros A c1 c2 Ho Hr. unfold view. rewrite Ho, Hr. reflexivity. Qed.

(* The own part is the view without the buffers granted to A. *)
Lemma view_own : forall A c1 c2, received A c1 = received A c2 -> view A c1 = view A c2 -> own A c1 = own A c2.
Proof.
  intros A c1 c2 Hr Hv. unfold view, extend in Hv. rewrite Hr in Hv. injection Hv as Hp Hb Hg Hf Hres Hop.
  apply app_inv_head in Hb. apply app_inv_head in Hg. apply catalog_ext; assumption.
Qed.

Lemma in_received : forall A c b, In b (received A c) -> In b (c_buffers c) /\ has_grant c (bf_id b) A = true.
Proof.
  intros A c b H. unfold received in H. apply in_flat_map in H. destruct H as [g [Hg Hb]].
  unfold grants_to in Hg. apply filter_In in Hg. destruct Hg as [Hg Eu]. apply Z.eqb_eq in Eu.
  unfold granted_buffer in Hb. destruct (find_buffer c (fst g)) as [b' |] eqn:F; [| destruct Hb].
  destruct Hb as [<- | []]. destruct (find_buffer_in _ _ _ F) as [Hin Hid]. split; [exact Hin |].
  apply has_grant_exists. exists g. split; [exact Hg |]. split; [exact (eq_sym Hid) | exact Eu].
Qed.

Lemma received_in : forall A c b, inv c -> In b (c_buffers c) -> has_grant c (bf_id b) A = true -> In b (received A c).
Proof.
  intros A c b Hinv Hb Hg. apply has_grant_exists in Hg. destruct Hg as [g [Hg [Ei Eu]]].
  unfold received. apply in_flat_map. exists g. split.
  - unfold grants_to. apply filter_In. split; [exact Hg | rewrite Eu; apply Z.eqb_refl].
  - unfold granted_buffer. rewrite Ei, (find_buffer_of c b Hinv Hb). left. reflexivity.
Qed.

Lemma received_owner : forall A c b, inv c -> In b (received A c) -> bf_owner b <> A.
Proof.
  intros A c b Hinv H. destruct (in_received A c b H) as [Hb Hg].
  exact (has_grant_owner c (bf_id b) A b Hinv Hg (find_buffer_of c b Hinv Hb)).
Qed.

Lemma received_tenant : forall A c b, inv c -> res_inv c -> In b (received A c) -> fst (bf_id b) <> A.
Proof.
  intros A c b Hinv [_ [Hb _]] H. pose proof (received_owner A c b Hinv H) as Ho.
  destruct (in_received A c b H) as [Hin _]. rewrite (proj1 (Hb b Hin)). exact Ho.
Qed.

(* The grants to A name distinct buffers, so the buffers granted to A have
   distinct ids. *)
Lemma nodup_fst : forall (A : Z) (l : list (bid * Z)), NoDup l -> (forall g, In g l -> snd g = A) -> NoDup (map fst l).
Proof.
  intros A l Hn. induction Hn as [| g l Hg Hn IH]; intros H; cbn [map]; [constructor |].
  constructor; [| apply IH; intros g' Hg'; apply H; right; exact Hg'].
  intros Hin. apply in_map_iff in Hin. destruct Hin as [g' [Ef Hg']]. apply Hg.
  assert (Es : snd g' = snd g) by (rewrite (H g' (or_intror Hg')), (H g (or_introl eq_refl)); reflexivity).
  destruct g as [x u], g' as [x' u']. cbn [fst snd] in Ef, Es. subst. exact Hg'.
Qed.

Lemma nodup_received : forall A c, NoDup (c_grants c) -> NoDup (map bf_id (received A c)).
Proof.
  intros A c Hn. assert (Hf : NoDup (map fst (grants_to A c))).
  { apply (nodup_fst A); [apply NoDup_filter; exact Hn |].
    intros g Hg. unfold grants_to in Hg. apply filter_In in Hg. apply Z.eqb_eq. exact (proj2 Hg). }
  unfold received. induction (grants_to A c) as [| g l IH]; [constructor |].
  cbn [flat_map map] in *. inversion Hf as [| ? ? Hg Hl]; subst.
  unfold granted_buffer at 1. destruct (find_buffer c (fst g)) as [b |] eqn:F; [| exact (IH Hl)].
  cbn [app map]. constructor; [| exact (IH Hl)].
  intros Hin. apply Hg. apply in_map_iff in Hin. destruct Hin as [b' [Eb Hb']].
  apply in_flat_map in Hb'. destruct Hb' as [g' [Hg' Hb']]. unfold granted_buffer in Hb'.
  destruct (find_buffer c (fst g')) as [b'' |] eqn:F'; [| destruct Hb'].
  destruct Hb' as [<- | []]. destruct (find_buffer_in _ _ _ F) as [_ Hid]. destruct (find_buffer_in _ _ _ F') as [_ Hid'].
  rewrite <- Hid. rewrite <- Eb, Hid'. apply in_map. exact Hg'.
Qed.

(* received is a function of the grants to A and of the buffers they name. *)
Lemma received_ext : forall A c c', grants_to A c' = grants_to A c ->
    (forall g, In g (grants_to A c) -> find_buffer c' (fst g) = find_buffer c (fst g)) -> received A c' = received A c.
Proof.
  intros A c c' Hg Hf. unfold received. rewrite Hg. clear Hg. revert Hf. generalize (grants_to A c) as l.
  induction l as [| g l IH]; intros Hf; [reflexivity |].
  cbn [flat_map]. unfold granted_buffer at 1 3. rewrite (Hf g (or_introl eq_refl)).
  rewrite IH; [reflexivity |]. intros g' Hg'. apply Hf. right. exact Hg'.
Qed.

Lemma received_set_status : forall A c id st,
    received A (set_status c id st) =
    map (fun b => with_status b (if bid_eqb (bf_id b) id then st else bf_status b)) (received A c).
Proof.
  intros A c id st. unfold received. change (grants_to A (set_status c id st)) with (grants_to A c).
  induction (grants_to A c) as [| g l IH]; [reflexivity |]. cbn [flat_map]. rewrite map_app, IH. f_equal.
  unfold granted_buffer, find_buffer. rewrite set_status_map, find_status_map.
  destruct (find _ (c_buffers c)); reflexivity.
Qed.

(* A status change of a buffer without a grant to A leaves the buffers
   granted to A as they are. *)
Lemma received_restatus : forall A c id st, has_grant c id A = false -> received A (set_status c id st) = received A c.
Proof.
  intros A c id st Hg. rewrite received_set_status. rewrite <- (map_id (received A c)) at 2.
  apply map_ext_in. intros b Hb. destruct (bid_eqb (bf_id b) id) eqn:E; [| apply with_status_same].
  exfalso. apply bid_eqb_eq in E. destruct (in_received A c b Hb) as [_ Hb']. rewrite E, Hg in Hb'. discriminate Hb'.
Qed.

Lemma received_written : forall A sizes l c,
    (forall id off sp, In ((id, off), sp) l -> writes sp = true -> has_grant c id A = false) ->
    received A (fold_left (fun acc x => written acc sizes x) l c) = received A c.
Proof.
  intros A sizes l. induction l as [| [[id off] sp] l IH]; intros c H; [reflexivity |]. cbn [fold_left].
  assert (E : received A (written c sizes ((id, off), sp)) = received A c).
  { unfold written. destruct (writes sp) eqn:W; cbn [negb]; [| reflexivity].
    destruct (bs_index sp); apply received_restatus; exact (H id off sp (or_introl eq_refl) W). }
  rewrite IH; [exact E |]. intros id' off' sp' Hin W.
  assert (Eg : has_grant (written c sizes ((id, off), sp)) id' A = has_grant c id' A).
  { unfold written. destruct (negb (writes sp)); [reflexivity |]. destruct (bs_index sp); reflexivity. }
  rewrite Eg. exact (H id' off' sp' (or_intror Hin) W).
Qed.

(* Every buffer argument that a launch writes is a buffer its tenant may write. *)
Lemma launch_written_writable : forall c t op gx gy gz buffers sizes scalars guarded r Mf,
    check_launch c t op gx gy gz buffers sizes scalars guarded = inl r -> find_binding c op = Some Mf ->
    forall id off sp, In ((id, off), sp) (combine buffers (spans Mf)) -> writes sp = true -> writable c t id = true.
Proof.
  intros c t op gx gy gz buffers sizes scalars guarded r Mf L Hb id off sp Hin W.
  destruct (check_launch_sound _ _ _ _ _ _ _ _ _ _ _ L) as [Mf' [Hb' [_ [_ [_ [_ [_ [_ [_ Hok]]]]]]]]].
  rewrite Hb in Hb'. injection Hb' as <-. rewrite Forall_forall in Hok. pose proof (Hok _ Hin) as Hx.
  unfold arg_ok in Hx. destruct Hx as [b [_ [_ [_ [_ [_ [_ [Hw _]]]]]]]]. exact (proj1 (Hw W)).
Qed.

(* An accepted action that neither grants a buffer to A nor ends a grant to
   A leaves the buffers granted to A as they are. *)
Lemma received_action : forall A c t a c' es, inv c -> check_action c t a = inl (c', es) ->
    (forall id, a <> Grant id A) -> (forall id, a = Free id -> has_grant c id A = false) ->
    received A c' = received A c.
Proof.
  intros A c t a c' es Hinv H Hng Hnf.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - destruct (allocate_next _ _ _ _ _ _ _ H) as [-> [_ [_ Hnone]]].
    apply received_ext; [reflexivity |]. intros g Hg. unfold find_buffer. cbn [c_buffers]. rewrite find_app.
    destruct (find _ (c_buffers c)) eqn:F; [reflexivity | exfalso].
    unfold grants_to in Hg. apply filter_In in Hg. destruct Hg as [Hg _].
    apply (has_grant_live c (fst g) (snd g) Hinv); [| exact F].
    apply has_grant_exists. exists g. split; [exact Hg | split; reflexivity].
  - destruct (free_next _ _ _ _ _ H) as [fb [Fb [_ [_ ->]]]]. pose proof (Hnf id eq_refl) as Hg0.
    assert (Hne : forall g, In g (grants_to A c) -> fst g <> id).
    { intros g Hg E. unfold grants_to in Hg. apply filter_In in Hg. destruct Hg as [Hg Eu]. apply Z.eqb_eq in Eu.
      assert (Ht : has_grant c id A = true) by (apply has_grant_exists; exists g; split; [exact Hg | split; assumption]).
      rewrite Ht in Hg0. discriminate Hg0. }
    apply received_ext.
    + unfold grants_to. cbn [c_grants]. rewrite filter_comm. apply filter_true. intros g Hg.
      apply negb_true_iff, bid_eqb_neq. exact (Hne g Hg).
    + intros g Hg. unfold find_buffer. cbn [c_buffers]. apply find_filter. intros b E.
      apply bid_eqb_eq in E. apply negb_true_iff, bid_eqb_neq. rewrite E. exact (Hne g Hg).
  - destruct (grant_next _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ ->]]]]].
    assert (Hu : u <> A) by (intros ->; exact (Hng id eq_refl)).
    apply received_ext; [| intros g _; reflexivity].
    unfold grants_to. cbn [c_grants]. rewrite filter_app. cbn [filter snd].
    replace (Z.eqb u A) with false by (symmetry; apply Z.eqb_neq; exact Hu). apply app_nil_r.
  - destruct (write_next _ _ _ _ _ _ _ H) as [fb [Fb [_ [_ Hel]]]].
    assert (Hwr : writable c t id = true).
    { cbn [check_action] in H. rewrite Fb in H. destruct (writable c t id); [reflexivity | discriminate H]. }
    destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ ->]]] | rewrite Hel];
      apply received_restatus; exact (writable_ungranted c t id A Hwr).
  - destruct (fill_next _ _ _ _ _ _ _ _ H) as [fb [Fb [_ [_ [_ [_ ->]]]]]].
    assert (Hwr : writable c t id = true).
    { cbn [check_action] in H. rewrite Fb in H. destruct (writable c t id); [reflexivity | discriminate H]. }
    apply received_restatus. exact (writable_ungranted c t id A Hwr).
  - destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [Hbind [[r L] [-> _]]]].
    apply received_written. intros id off sp Hin W.
    exact (writable_ungranted c t id A (launch_written_writable c t op gx gy gz buffers sizes scalars guarded r Mf L Hbind
                                          id off sp Hin W)).
  - destruct (reserve_next _ _ _ _ _ _ H) as [_ [_ [_ [_ [_ [_ ->]]]]]]. reflexivity.
  - destruct (unreserve_next _ _ _ _ H) as [_ [_ [_ ->]]]. reflexivity.
Qed.

(* A grant to A adds the buffer, as it is, after the buffers granted before. *)
Lemma received_grant : forall A c t id c' es, check_action c t (Grant id A) = inl (c', es) ->
    exists b, find_buffer c id = Some b /\ received A c' = received A c ++ [b].
Proof.
  intros A c t id c' es H. destruct (grant_next _ _ _ _ _ _ H) as [fb [Fb [_ [_ [_ ->]]]]].
  exists fb. split; [exact Fb |]. unfold received at 1, grants_to. cbn [c_grants]. rewrite filter_app.
  cbn [filter snd]. rewrite Z.eqb_refl, flat_map_app. cbn [flat_map]. unfold granted_buffer at 2.
  change (find_buffer {| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c ++ [(id, A)];
                         c_freed := c_freed c; c_reserved := c_reserved c; c_operators := c_operators c |} (fst (id, A)))
    with (find_buffer c id). rewrite Fb. reflexivity.
Qed.

(* A free removes the buffer from the buffers granted to A. *)
Lemma received_free : forall A c t id c' es, check_action c t (Free id) = inl (c', es) ->
    received A c' = filter (fun b => negb (bid_eqb (bf_id b) id)) (received A c).
Proof.
  intros A c t id c' es H. destruct (free_next _ _ _ _ _ H) as [fb [_ [_ [_ ->]]]].
  unfold received, grants_to. cbn [c_grants]. rewrite filter_comm.
  induction (filter (fun g => Z.eqb (snd g) A) (c_grants c)) as [| g l IH]; [reflexivity |].
  cbn [filter flat_map]. rewrite filter_app, <- IH.
  unfold granted_buffer at 2. destruct (find_buffer c (fst g)) as [b |] eqn:F.
  - destruct (find_buffer_in _ _ _ F) as [_ Hid]. cbn [filter]. rewrite Hid.
    destruct (negb (bid_eqb (fst g) id)) eqn:E; cbn [flat_map app].
    + unfold granted_buffer at 1. unfold find_buffer at 1. cbn [c_buffers]. rewrite find_filter.
      * change (find (fun b0 => bid_eqb (bf_id b0) (fst g)) (c_buffers c)) with (find_buffer c (fst g)). rewrite F. reflexivity.
      * intros x Ex. apply bid_eqb_eq in Ex. rewrite Ex. exact E.
    + reflexivity.
  - cbn [filter app]. destruct (negb (bid_eqb (fst g) id)) eqn:E; cbn [flat_map]; [| reflexivity].
    unfold granted_buffer at 1. unfold find_buffer at 1. cbn [c_buffers]. rewrite find_filter.
    + change (find (fun b0 => bid_eqb (bf_id b0) (fst g)) (c_buffers c)) with (find_buffer c (fst g)). rewrite F. reflexivity.
    + intros x Ex. apply bid_eqb_eq in Ex. rewrite Ex. exact E.
Qed.

(* Every accepted action keeps the grants distinct. *)
Lemma grants_nodup_action : forall c t a c' es, NoDup (c_grants c) -> check_action c t a = inl (c', es) ->
    NoDup (c_grants c').
Proof.
  intros c t a c' es Hn H.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |].
  - destruct (allocate_next _ _ _ _ _ _ _ H) as [-> _]. exact Hn.
  - destruct (free_next _ _ _ _ _ H) as [fb [_ [_ [_ ->]]]]. apply NoDup_filter. exact Hn.
  - pose proof H as H'. destruct (grant_next _ _ _ _ _ _ H') as [fb [Fb [_ [_ [_ ->]]]]].
    cbn [check_action] in H. rewrite Fb in H.
    destruct (negb (Z.eqb (bf_owner fb) t) || Z.eqb u t); [discriminate H |].
    destruct (has_grant c id u) eqn:Eg; [discriminate H |].
    cbn [c_grants]. apply NoDup_snoc; [exact Hn |]. intros Hin.
    assert (Ht : has_grant c id u = true) by (apply has_grant_exists; exists (id, u); split; [exact Hin | split; reflexivity]).
    congruence.
  - destruct (write_next _ _ _ _ _ _ _ H) as [fb [_ [_ [_ Hel]]]].
    destruct element as [[e s] |]; [destruct Hel as [_ [_ [_ ->]]] | rewrite Hel]; exact Hn.
  - destruct (fill_next _ _ _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ [_ ->]]]]]]. exact Hn.
  - destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [_ [_ [-> _]]]].
    rewrite (proj2 (written_frame _ _ _)). exact Hn.
  - destruct (reserve_next _ _ _ _ _ _ H) as [_ [_ [_ [_ [_ [_ ->]]]]]]. exact Hn.
  - destruct (unreserve_next _ _ _ _ H) as [_ [_ [_ ->]]]. exact Hn.
Qed.

Lemma grants_nodup_actions : forall c t pos acts done c' es, NoDup (c_grants c) ->
    check_actions c t pos acts done = Accepted c' es -> NoDup (c_grants c').
Proof.
  intros c t pos acts. revert c pos. induction acts as [| a acts IH]; intros c pos done c' es Hn H; cbn in H.
  - injection H as <- _. exact Hn.
  - destruct (check_action c t a) as [[c1 es1] |] eqn:E; [| discriminate H].
    exact (IH c1 _ _ _ _ (grants_nodup_action c t a c1 es1 Hn E) H).
Qed.

(* So every accepted step keeps the grants distinct, as it keeps `inv`. *)
Theorem check_step_grants_nodup : forall c s c' es, NoDup (c_grants c) -> check_step c s = Accepted c' es ->
    NoDup (c_grants c').
Proof. intros c s c' es Hn H. exact (grants_nodup_actions c (s_tenant s) 0 (s_actions s) [] c' es Hn H). Qed.

(* ------------------------------------------------------------------ *)
(* What an action of A sees.                                           *)

Lemma find_extend : forall A sh o id,
    find_buffer (extend A sh o) id =
    match find (fun b => bid_eqb (bf_id b) id) sh with Some b => Some b | None => find_buffer o id end.
Proof. intros A sh o id. unfold find_buffer at 1, extend. cbn [c_buffers]. apply find_app. Qed.

Lemma find_view : forall A c id, inv c ->
    find_buffer (view A c) id =
    match find_buffer c id with
    | Some b => if Z.eqb (bf_owner b) A || has_grant c id A then Some b else None
    | None => None
    end.
Proof.
  intros A c id Hinv. unfold view. rewrite find_extend.
  destruct (find (fun b => bid_eqb (bf_id b) id) (received A c)) as [x |] eqn:Fs.
  - apply find_some in Fs. destruct Fs as [Hx E]. apply bid_eqb_eq in E. subst id.
    destruct (in_received A c x Hx) as [Hin Hg]. rewrite (find_buffer_of c x Hinv Hin), Hg, orb_true_r. reflexivity.
  - rewrite (find_own A c id Hinv). destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
    destruct (has_grant c id A) eqn:Hg.
    + exfalso. destruct (find_buffer_in _ _ _ Fb) as [Hb Hid]. subst id.
      pose proof (find_none _ _ Fs b (received_in A c b Hinv Hb Hg)) as E. cbn beta in E.
      rewrite bid_eqb_refl in E. discriminate E.
    + rewrite orb_false_r. reflexivity.
Qed.

Lemma find_view_some : forall A c id b, inv c -> find_buffer c id = Some b ->
    bf_owner b = A \/ has_grant c id A = true -> find_buffer (view A c) id = Some b.
Proof.
  intros A c id b Hinv Fb H. rewrite (find_view A c id Hinv), Fb.
  destruct H as [H | H]; [rewrite H, Z.eqb_refl | rewrite H, orb_true_r]; reflexivity.
Qed.

Lemma has_grant_view : forall A c id, inv c -> has_grant (view A c) id A = has_grant c id A.
Proof.
  intros A c id Hinv. apply eq_true_iff_eq. rewrite !has_grant_exists. split.
  - intros [g [Hg [Ei Eu]]]. unfold view, extend in Hg. cbn [c_grants] in Hg. apply in_app_or in Hg.
    destruct Hg as [Hg | Hg].
    + apply in_map_iff in Hg. destruct Hg as [b [<- Hb]]. cbn [fst] in Ei. subst id.
      apply has_grant_exists. exact (proj2 (in_received A c b Hb)).
    + unfold own in Hg. cbn [c_grants] in Hg. apply filter_In in Hg. exists g. split; [exact (proj1 Hg) | split; assumption].
  - intros Hex. assert (Hg : has_grant c id A = true) by (apply has_grant_exists; exact Hex).
    destruct (find_buffer c id) as [b |] eqn:Fb; [| exfalso; exact (has_grant_live c id A Hinv Hg Fb)].
    destruct (find_buffer_in _ _ _ Fb) as [Hb Hid]. subst id.
    exists (bf_id b, A). split; [| split; reflexivity].
    unfold view, extend. cbn [c_grants]. apply in_or_app. left.
    apply (in_map (fun b => (bf_id b, A))). exact (received_in A c b Hinv Hb Hg).
Qed.

Lemma readable_view : forall A c id, inv c -> readable (view A c) A id = readable c A id.
Proof.
  intros A c id Hinv. unfold readable. rewrite (find_view A c id Hinv), (has_grant_view A c id Hinv).
  destruct (find_buffer c id) as [b |]; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E1, (has_grant c id A); cbn [orb]; rewrite ?E1; reflexivity.
Qed.

(* A grant in the view on a buffer of A is a grant on A's own part. *)
Lemma view_grants_tenant : forall A c (f : bid * Z -> bool) id, inv c -> res_inv c -> fst id = A ->
    (forall g, f g = true -> fst g = id) ->
    existsb f (c_grants (view A c)) = existsb f (c_grants (own A c)).
Proof.
  intros A c f id Hinv Hr Hid Hf. unfold view, extend. cbn [c_grants]. rewrite existsb_app.
  rewrite existsb_none; [reflexivity |]. intros x Hx. apply in_map_iff in Hx. destruct Hx as [b [<- Hb]].
  destruct (f (bf_id b, A)) eqn:E; [| reflexivity]. exfalso. apply (received_tenant A c b Hinv Hr Hb).
  rewrite <- Hid. exact (f_equal fst (Hf _ E)).
Qed.

Lemma granted_view : forall A c id, inv c -> res_inv c -> fst id = A -> granted (view A c) id = granted c id.
Proof.
  intros A c id Hinv Hr Hid. unfold granted. rewrite (view_grants_tenant A c _ id Hinv Hr Hid).
  - exact (granted_own A c id Hid).
  - intros g E. apply bid_eqb_eq. exact E.
Qed.

Lemma has_grant_view_tenant : forall A c id u, inv c -> res_inv c -> fst id = A ->
    has_grant (view A c) id u = has_grant c id u.
Proof.
  intros A c id u Hinv Hr Hid. unfold has_grant. rewrite (view_grants_tenant A c _ id Hinv Hr Hid).
  - exact (has_grant_own A c id u Hid).
  - intros g E. apply andb_true_iff in E. apply bid_eqb_eq. exact (proj1 E).
Qed.

Lemma writable_view : forall A c id, inv c -> res_inv c -> writable (view A c) A id = writable c A id.
Proof.
  intros A c id Hinv Hr. unfold writable. rewrite (find_view A c id Hinv).
  destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A) eqn:E; cbn [orb].
  - rewrite E. cbn [andb]. rewrite granted_view; [reflexivity | exact Hinv | exact Hr |].
    rewrite (owner_of_find c id b Hr Fb). apply Z.eqb_eq. exact E.
  - destruct (has_grant c id A); [rewrite E |]; reflexivity.
Qed.

Lemma find_view_tenant : forall A c id, inv c -> res_inv c -> fst id = A -> find_buffer (view A c) id = find_buffer c id.
Proof.
  intros A c id Hinv Hr Hid. rewrite (find_view A c id Hinv). destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  rewrite <- (owner_of_find c id b Hr Fb), Hid, Z.eqb_refl. reflexivity.
Qed.

Lemma extend_set_status : forall A sh o id st,
    set_status (extend A sh o) id st =
    extend A (map (fun b => with_status b (if bid_eqb (bf_id b) id then st else bf_status b)) sh) (set_status o id st).
Proof.
  intros A sh o id st. apply catalog_ext; [reflexivity | | | reflexivity | reflexivity | reflexivity].
  - rewrite set_status_map. unfold extend. cbn [c_buffers]. rewrite set_status_map, map_app. reflexivity.
  - unfold extend. cbn [c_grants set_status]. rewrite map_map. reflexivity.
Qed.

Lemma view_set_status : forall A c id st, view A (set_status c id st) = set_status (view A c) id st.
Proof.
  intros A c id st. unfold view. rewrite extend_set_status, received_set_status, own_set_status. reflexivity.
Qed.

Lemma view_written : forall A c sizes x, view A (written c sizes x) = written (view A c) sizes x.
Proof.
  intros A c sizes [[id off] sp]. unfold written. destruct (negb (writes sp)); [reflexivity |].
  destruct (bs_index sp); apply view_set_status.
Qed.

Lemma view_fold_written : forall A sizes l c,
    view A (fold_left (fun acc x => written acc sizes x) l c) = fold_left (fun acc x => written acc sizes x) l (view A c).
Proof.
  intros A sizes l. induction l as [| x l IH]; intros c; [reflexivity |]. cbn [fold_left].
  rewrite IH, view_written. reflexivity.
Qed.

Lemma view_buffers_in : forall A c b, In b (c_buffers (view A c)) -> In b (c_buffers c).
Proof.
  intros A c b Hb. unfold view, extend in Hb. cbn [c_buffers] in Hb. apply in_app_or in Hb.
  destruct Hb as [Hb | Hb]; [exact (proj1 (in_received A c b Hb)) |].
  unfold own in Hb. cbn [c_buffers] in Hb. apply filter_In in Hb. exact (proj1 Hb).
Qed.

Lemma inv_view : forall A c, inv c -> res_inv c -> NoDup (c_grants c) -> inv (view A c).
Proof.
  intros A c Hinv Hr Hgn. pose proof Hinv as [Hp [Hn [Ha Hg]]]. split; [| split; [| split]].
  - intros b Hb. exact (Hp b (view_buffers_in A c b Hb)).
  - unfold view, extend. cbn [c_buffers]. rewrite map_app. apply NoDup_app_apart.
    + exact (nodup_received A c Hgn).
    + unfold own. cbn [c_buffers]. apply NoDup_filter_map. exact Hn.
    + intros x Hx1 Hx2. apply in_map_iff in Hx1, Hx2. destruct Hx1 as [b1 [E1 H1]]. destruct Hx2 as [b2 [E2 H2]].
      unfold own in H2. cbn [c_buffers] in H2. apply filter_In in H2. destruct H2 as [H2 Eo]. apply Z.eqb_eq in Eo.
      pose proof (received_owner A c b1 Hinv H1) as Ho1. destruct (in_received A c b1 H1) as [H1' _].
      assert (b1 = b2) by (apply (NoDup_map_eq _ _ bf_id (c_buffers c)); [exact Hn | exact H1' | exact H2 | congruence]).
      congruence.
  - intros a b HA HB Hd. exact (Ha a b (view_buffers_in A c a HA) (view_buffers_in A c b HB) Hd).
  - intros g Hin. unfold view, extend in Hin. cbn [c_grants] in Hin. apply in_app_or in Hin. destruct Hin as [Hin | Hin].
    + apply in_map_iff in Hin. destruct Hin as [b [<- Hb]]. exists b.
      split; [unfold view, extend; cbn [c_buffers]; apply in_or_app; left; exact Hb |].
      cbn [fst snd]. split; [reflexivity | exact (received_owner A c b Hinv Hb)].
    + destruct (proj2 (proj2 (proj2 (inv_own A c Hinv Hr))) g Hin) as [b [Hb [Hid Ho]]]. exists b.
      split; [unfold view, extend; cbn [c_buffers]; apply in_or_app; right; exact Hb | split; assumption].
Qed.

Section Grants.

Variable base : Z.
Variable kernels : Z -> option (program * lfacts).

Lemma observable_view : forall A c y, inv c -> (observable base (view A c) A y <-> observable base c A y).
Proof.
  intros A c y Hinv. split.
  - intros [[b [Hb [Hr Hy]]] | [r [Hr [Ho Hy]]]].
    + left. exists b. split; [exact (view_buffers_in A c b Hb) |].
      split; [rewrite <- (readable_view A c (bf_id b) Hinv); exact Hr | exact Hy].
    + right. exists r. unfold view, extend, own in Hr. cbn [c_freed] in Hr. apply filter_In in Hr.
      split; [exact (proj1 Hr) | split; assumption].
  - intros [[b [Hb [Hr Hy]]] | [r [Hr [Ho Hy]]]].
    + left. exists b. pose proof Hr as Hr'. unfold readable in Hr'. rewrite (find_buffer_of c b Hinv Hb) in Hr'.
      split.
      * unfold view, extend. cbn [c_buffers]. apply in_or_app. apply orb_true_iff in Hr'. destruct Hr' as [Ho | Hg].
        -- right. unfold own. cbn [c_buffers]. apply filter_In. split; [exact Hb | exact Ho].
        -- left. exact (received_in A c b Hinv Hb Hg).
      * split; [rewrite (readable_view A c (bf_id b) Hinv); exact Hr | exact Hy].
    + right. exists r. split; [| split; assumption]. unfold view, extend, own. cbn [c_freed]. apply filter_In.
      split; [exact Hr | rewrite Ho; apply Z.eqb_refl].
Qed.

Lemma agree_view : forall A c m1 m2, inv c -> (agree_on base (view A c) A m1 m2 <-> agree_on base c A m1 m2).
Proof.
  intros A c m1 m2 Hinv. split; intros H y Hy; apply H; apply (observable_view A c y Hinv); exact Hy.
Qed.

(* Two catalogs with the same own part of A and the same buffers granted to
   A show A the same bytes. *)
Lemma observable_related : forall A c1 c2 y, inv c1 -> inv c2 -> own A c1 = own A c2 -> received A c1 = received A c2 ->
    observable base c1 A y -> observable base c2 A y.
Proof.
  intros A c1 c2 y Hi1 Hi2 Ho Hr Hy. apply (observable_view A c2 y Hi2). rewrite <- (view_same A c1 c2 Ho Hr).
  apply (observable_view A c1 y Hi1). exact Hy.
Qed.

Lemma sound_view : forall A c m, sound base kernels {| w_cat := c; w_mem := m |} -> res_inv c -> NoDup (c_grants c) ->
    sound base kernels {| w_cat := view A c; w_mem := m |}.
Proof.
  intros A c m [[Hinv [Hs [Hver Hb]]] Hfa] Hr Hgn. split; [split; [| split; [| split]] |]; cbn [w_cat w_mem] in *.
  - exact (inv_view A c Hinv Hr Hgn).
  - intros b Hin. exact (Hs b (view_buffers_in A c b Hin)).
  - intros op Mf Hf. apply Hver. exact Hf.
  - exact Hb.
  - intros r b Hr' Hb'. unfold view, extend, own in Hr'. cbn [c_freed] in Hr'. apply filter_In in Hr'.
    exact (Hfa r b (proj1 Hr') (view_buffers_in A c b Hb')).
Qed.

End Grants.

(* ------------------------------------------------------------------ *)
(* An action of A decides on its view as on the whole catalog.        *)

(* The buffers sh granted to A, next to A's own part p: they are buffers of
   other tenants, outside p and apart from A's reservation. *)
Definition extend_ok (A : Z) (sh : list buffer) (p : catalog) : Prop :=
  forall b, In b sh -> fst (bf_id b) <> A /\ bf_owner b <> A /\ find_buffer p (bf_id b) = None /\
    forall r, find_reservation p A = Some r ->
      bf_offset b + bf_bytes b <= rs_offset r \/ rs_offset r + rs_bytes r <= bf_offset b.

Lemma extend_ok_view : forall A c, inv c -> res_inv c -> extend_ok A (received A c) (own A c).
Proof.
  intros A c Hinv Hr b Hb. pose proof (received_owner A c b Hinv Hb) as Ho.
  destruct (in_received A c b Hb) as [Hin _].
  split; [exact (received_tenant A c b Hinv Hr Hb) |]. split; [exact Ho |].
  split; [rewrite (find_own A c (bf_id b) Hinv), (find_buffer_of c b Hinv Hin);
          replace (Z.eqb (bf_owner b) A) with false by (symmetry; apply Z.eqb_neq; exact Ho); reflexivity |].
  intros r Er. rewrite find_reservation_own in Er. pose proof Hr as [_ [Hrb _]].
  destruct (Hrb b Hin) as [_ [rB [HB [HBl HBh]]]].
  destruct (reservations_apart c A (bf_owner b) r rB Hr (fun E => Ho (eq_sym E)) Er HB); lia.
Qed.

Definition extend_result (A : Z) (sh : list buffer) (d : catalog * list effect + String.string)
    : catalog * list effect + String.string :=
  match d with inl (p', es) => inl (extend A sh p', es) | inr r => inr r end.

Lemma find_sh_none : forall (sh : list buffer) id, find (fun b => bid_eqb (bf_id b) id) sh = None ->
    forall b, In b sh -> bid_eqb (bf_id b) id = false.
Proof. intros sh id H b Hb. exact (find_none _ _ H b Hb). Qed.

Lemma extend_grants_apart : forall A sh p id (f : bid * Z -> bool), (forall b, In b sh -> bid_eqb (bf_id b) id = false) ->
    (forall g, f g = true -> fst g = id) -> existsb f (c_grants (extend A sh p)) = existsb f (c_grants p).
Proof.
  intros A sh p id f Hsh Hf. unfold extend. cbn [c_grants]. rewrite existsb_app, existsb_none; [reflexivity |].
  intros x Hx. apply in_map_iff in Hx. destruct Hx as [b [<- Hb]]. destruct (f (bf_id b, A)) eqn:E; [| reflexivity].
  pose proof (Hf _ E) as Ei. cbn [fst] in Ei. pose proof (Hsh b Hb) as Hn. rewrite Ei, bid_eqb_refl in Hn.
  discriminate Hn.
Qed.

Lemma writable_extend_apart : forall A sh p t id, find (fun b => bid_eqb (bf_id b) id) sh = None ->
    writable (extend A sh p) t id = writable p t id.
Proof.
  intros A sh p t id Fs. unfold writable. rewrite find_extend, Fs. destruct (find_buffer p id); [| reflexivity].
  unfold granted. rewrite (extend_grants_apart A sh p id _ (find_sh_none sh id Fs)); [reflexivity |].
  intros g E. apply bid_eqb_eq. exact E.
Qed.

Lemma has_grant_extend_apart : forall A sh p id u, find (fun b => bid_eqb (bf_id b) id) sh = None ->
    has_grant (extend A sh p) id u = has_grant p id u.
Proof.
  intros A sh p id u Fs. unfold has_grant. apply (extend_grants_apart A sh p id _ (find_sh_none sh id Fs)).
  intros g E. apply andb_true_iff in E. apply bid_eqb_eq. exact (proj1 E).
Qed.

Lemma extend_set_status_apart : forall A sh p id st, (forall b, In b sh -> bid_eqb (bf_id b) id = false) ->
    set_status (extend A sh p) id st = extend A sh (set_status p id st).
Proof.
  intros A sh p id st Hsh. rewrite extend_set_status. f_equal. rewrite <- (map_id sh) at 2. apply map_ext_in.
  intros b Hb. rewrite (Hsh b Hb). apply with_status_same.
Qed.

(* A buffer granted to A gives the same refusal as a missing one. *)
Lemma find_sh_some : forall A sh p id b, extend_ok A sh p -> find (fun b => bid_eqb (bf_id b) id) sh = Some b ->
    find_buffer (extend A sh p) id = Some b /\ find_buffer p id = None /\ bf_owner b <> A /\ fst id <> A.
Proof.
  intros A sh p id b Hok Fs. pose proof (find_some _ _ Fs) as [Hb E]. apply bid_eqb_eq in E. subst id.
  destruct (Hok b Hb) as [Hi [Ho [Hn _]]]. rewrite find_extend, Fs.
  split; [reflexivity | split; [exact Hn | split; assumption]].
Qed.

Lemma extend_action : forall A sh p a, extend_ok A sh p ->
    (forall op gx gy gz buffers sizes scalars guarded, a <> Launch op gx gy gz buffers sizes scalars guarded) ->
    check_action (extend A sh p) A a = extend_result A sh (check_action p A a).
Proof.
  intros A sh p a Hok Hnl.
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |]; cbn [check_action].
  - (* an allocation: its region lies in A's reservation, apart from every buffer granted to A *)
    destruct (negb (Z.eqb (fst id) A)) eqn:E0; [reflexivity |]. apply negb_false_iff, Z.eqb_eq in E0.
    destruct (find (fun b => bid_eqb (bf_id b) id) sh) as [b |] eqn:Fs.
    { exfalso. destruct (find_sh_some A sh p id b Hok Fs) as [_ [_ [_ Hi]]]. exact (Hi E0). }
    rewrite find_extend, Fs. destruct (find_buffer p id); [reflexivity |].
    change (c_pool (extend A sh p)) with (c_pool p).
    destruct (Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool p) (o + n)); [reflexivity |].
    destruct (negb (Z.eqb (o mod alignment) 0)); [reflexivity |].
    change (in_reservation (extend A sh p) A o n) with (in_reservation p A o n).
    destruct (in_reservation p A o n) eqn:Ein; cbn [negb]; [| reflexivity].
    destruct (in_reservation_some _ _ _ _ Ein) as [r [Hr [Hrl Hrh]]].
    assert (Hov : existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers (extend A sh p)) =
                  existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers p)).
    { unfold extend. cbn [c_buffers]. rewrite existsb_app, existsb_none; [reflexivity |].
      intros b Hb. destruct (Hok b Hb) as [_ [_ [_ Hap]]]. pose proof (Hap r Hr) as H. unfold overlap.
      apply andb_false_iff. destruct H; [right | left]; apply Z.ltb_ge; lia. }
    rewrite Hov. destruct (existsb _ (c_buffers p)); [reflexivity |].
    cbv zeta. change (c_freed (extend A sh p)) with (c_freed p). cbn [extend_result]. f_equal. f_equal.
    apply catalog_ext; unfold extend; cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators];
      [reflexivity | rewrite app_assoc; reflexivity | reflexivity | reflexivity | reflexivity | reflexivity].
  - (* a free: a buffer granted to A belongs to another tenant *)
    destruct (find (fun b => bid_eqb (bf_id b) id) sh) as [b |] eqn:Fs.
    + destruct (find_sh_some A sh p id b Hok Fs) as [Fe [Fp [Ho _]]]. rewrite Fe, Fp.
      replace (Z.eqb (bf_owner b) A) with false by (symmetry; apply Z.eqb_neq; exact Ho). reflexivity.
    + rewrite find_extend, Fs. destruct (find_buffer p id) as [fb |]; [| reflexivity].
      destruct (negb (Z.eqb (bf_owner fb) A)); [reflexivity |]. cbn [extend_result]. f_equal. f_equal.
      apply catalog_ext; unfold extend; cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators];
        [reflexivity | | | reflexivity | reflexivity | reflexivity]; rewrite filter_app; f_equal; apply filter_true.
      * intros x Hx. rewrite (find_sh_none sh id Fs x Hx). reflexivity.
      * intros x Hx. apply in_map_iff in Hx. destruct Hx as [b [<- Hb]]. cbn [fst].
        rewrite (find_sh_none sh id Fs b Hb). reflexivity.
  - (* a grant *)
    destruct (find (fun b => bid_eqb (bf_id b) id) sh) as [b |] eqn:Fs.
    + destruct (find_sh_some A sh p id b Hok Fs) as [Fe [Fp [Ho _]]]. rewrite Fe, Fp.
      replace (Z.eqb (bf_owner b) A) with false by (symmetry; apply Z.eqb_neq; exact Ho). reflexivity.
    + rewrite find_extend, Fs. destruct (find_buffer p id) as [fb |]; [| reflexivity].
      destruct (negb (Z.eqb (bf_owner fb) A) || Z.eqb u A); [reflexivity |].
      rewrite (has_grant_extend_apart A sh p id u Fs). destruct (has_grant p id u); [reflexivity |].
      cbn [extend_result]. f_equal. f_equal.
      apply catalog_ext; unfold extend; cbn [c_pool c_buffers c_grants c_freed c_reserved c_operators];
        [reflexivity | reflexivity | rewrite app_assoc; reflexivity | reflexivity | reflexivity | reflexivity].
  - (* a write: the check refuses A's write of a buffer granted to A *)
    destruct (find (fun b => bid_eqb (bf_id b) id) sh) as [b |] eqn:Fs.
    + destruct (find_sh_some A sh p id b Hok Fs) as [Fe [Fp [Ho _]]]. rewrite Fe, Fp.
      replace (writable (extend A sh p) A id) with false; [reflexivity |].
      unfold writable. rewrite Fe. replace (Z.eqb (bf_owner b) A) with false by (symmetry; apply Z.eqb_neq; exact Ho).
      reflexivity.
    + rewrite find_extend, Fs, (writable_extend_apart A sh p A id Fs). destruct (find_buffer p id) as [fb |]; [| reflexivity].
      destruct (negb (writable p A id)); [reflexivity |].
      destruct element as [[e s] |].
      * destruct (width_ok e && _ && _); [| reflexivity]. destruct (data_range e s data) as [lo hi].
        cbn [extend_result]. rewrite (extend_set_status_apart A sh p id _ (find_sh_none sh id Fs)). reflexivity.
      * destruct (Z.leb _ _); [| reflexivity]. cbn [extend_result].
        rewrite (extend_set_status_apart A sh p id _ (find_sh_none sh id Fs)). reflexivity.
  - (* a fill *)
    destruct (find (fun b => bid_eqb (bf_id b) id) sh) as [b |] eqn:Fs.
    + destruct (find_sh_some A sh p id b Hok Fs) as [Fe [Fp [Ho _]]]. rewrite Fe, Fp.
      replace (writable (extend A sh p) A id) with false; [reflexivity |].
      unfold writable. rewrite Fe. replace (Z.eqb (bf_owner b) A) with false by (symmetry; apply Z.eqb_neq; exact Ho).
      reflexivity.
    + rewrite find_extend, Fs, (writable_extend_apart A sh p A id Fs). destruct (find_buffer p id) as [fb |]; [| reflexivity].
      destruct (negb (writable p A id)); [reflexivity |]. destruct (element_range e s) as [lo hi].
      destruct (_ && _ && _ && _); [| reflexivity]. cbn [extend_result].
      rewrite (extend_set_status_apart A sh p id _ (find_sh_none sh id Fs)). reflexivity.
  - exfalso. exact (Hnl op gx gy gz buffers sizes scalars guarded eq_refl).
  - (* a reservation *)
    change (find_reservation (extend A sh p) A) with (find_reservation p A).
    destruct (find_reservation p A); [reflexivity |].
    change (c_pool (extend A sh p)) with (c_pool p). change (c_reserved (extend A sh p)) with (c_reserved p).
    destruct (Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool p) (o + n)); [reflexivity |].
    destruct (existsb _ (c_reserved p)); [reflexivity |]. reflexivity.
  - (* the end of A's reservation: every buffer granted to A belongs to another tenant *)
    change (find_reservation (extend A sh p) A) with (find_reservation p A).
    destruct (find_reservation p A); [| reflexivity].
    assert (E : existsb (fun b => Z.eqb (bf_owner b) A) (c_buffers (extend A sh p)) =
                existsb (fun b => Z.eqb (bf_owner b) A) (c_buffers p)).
    { unfold extend. cbn [c_buffers]. rewrite existsb_app, existsb_none; [reflexivity |].
      intros b Hb. apply Z.eqb_neq. exact (proj1 (proj2 (Hok b Hb))). }
    rewrite E. destruct (existsb _ (c_buffers p)); reflexivity.
Qed.

Lemma arg_reason_view : forall A c guarded buffers sps sizes x, inv c -> res_inv c ->
    arg_reason (view A c) A guarded buffers sps sizes x = arg_reason c A guarded buffers sps sizes x.
Proof.
  intros A c guarded buffers sps sizes [[id off] sp] Hinv Hr. unfold arg_reason.
  rewrite (readable_view A c id Hinv), (writable_view A c id Hinv Hr), (find_view A c id Hinv).
  destruct (find_buffer c id) as [b |] eqn:Fb; [| reflexivity].
  destruct (Z.eqb (bf_owner b) A || has_grant c id A) eqn:E; [reflexivity |].
  (* a buffer that A may read neither as owner nor as grantee *)
  replace (readable c A id) with false; [reflexivity |].
  unfold readable. rewrite Fb. symmetry. exact E.
Qed.

(* Every buffer argument of an accepted launch of A is in A's view. *)
Lemma launch_args_view : forall A c op gx gy gz buffers sizes scalars guarded r, inv c ->
    check_launch c A op gx gy gz buffers sizes scalars guarded = inl r ->
    forall id off, In (id, off) buffers -> find_buffer (view A c) id = find_buffer c id.
Proof.
  intros A c op gx gy gz buffers sizes scalars guarded r Hinv H id off Hin.
  destruct (check_launch_sound _ _ _ _ _ _ _ _ _ _ _ H) as [Mf [_ [_ [_ [_ [_ [Hlen [_ [_ Hok]]]]]]]]].
  destruct (in_combine_left _ _ _ _ _ Hlen Hin) as [sp Hsp].
  rewrite Forall_forall in Hok. pose proof (Hok _ Hsp) as Hx. unfold arg_ok in Hx.
  destruct Hx as [b [Fb [_ [_ [_ [_ [Hrd [Hwr _]]]]]]]].
  rewrite Fb. apply (find_view_some A c id b Hinv Fb).
  destruct (reads sp) eqn:R.
  - pose proof (Hrd eq_refl) as E. unfold readable in E. rewrite Fb in E. apply orb_true_iff in E.
    destruct E as [E | E]; [left; apply Z.eqb_eq; exact E | right; exact E].
  - assert (W : writes sp = true) by (unfold reads, writes in *; destruct (bs_dir sp); congruence).
    destruct (Hwr W) as [E _]. unfold writable in E. rewrite Fb in E. apply andb_true_iff in E.
    left. apply Z.eqb_eq. exact (proj1 E).
Qed.

Lemma view_check_launch : forall A c op gx gy gz buffers sizes scalars guarded, inv c -> res_inv c ->
    check_launch (view A c) A op gx gy gz buffers sizes scalars guarded =
    match check_launch c A op gx gy gz buffers sizes scalars guarded with
    | inl (c', k) => inl (view A c', k)
    | inr r => inr r
    end.
Proof.
  intros A c op gx gy gz buffers sizes scalars guarded Hinv Hr.
  destruct (check_launch c A op gx gy gz buffers sizes scalars guarded) as [[c' k] | r] eqn:L.
  - pose proof (launch_args_view A c op gx gy gz buffers sizes scalars guarded _ Hinv L) as Hargs.
    unfold check_launch in *.
    change (find_binding (view A c) op) with (find_binding c op).
    destruct (find_binding c op) as [Mf |]; [| discriminate L].
    destruct grid_limit as [[lx ly] lz]. destruct (b_grid Mf) as [[mx my] mz].
    rewrite (first_reason_ext _ _ _ _ (fun x => arg_reason_view A c guarded buffers (spans Mf) sizes x Hinv Hr)).
    refusals L. destruct (first_reason _ _) eqn:F; [discriminate L |]. injection L as <- <-.
    rewrite view_fold_written, (args_of_ext (view A c) c sizes (b_params Mf) buffers scalars Hargs).
    rewrite (checks_of_ext (view A c) c guarded sizes (combine buffers (spans Mf))); [reflexivity |].
    intros id off sp Hin. apply (Hargs id off). exact (in_combine_l _ _ _ _ Hin).
  - unfold check_launch in *.
    change (find_binding (view A c) op) with (find_binding c op).
    destruct (find_binding c op) as [Mf |]; [| exact L].
    destruct grid_limit as [[lx ly] lz]. destruct (b_grid Mf) as [[mx my] mz].
    rewrite (first_reason_ext _ _ _ _ (fun x => arg_reason_view A c guarded buffers (spans Mf) sizes x Hinv Hr)).
    refusals L; try exact L.
    destruct (first_reason _ _); [exact L | discriminate L].
Qed.

(* An accepted action of A leaves the buffers granted to A as they are. *)
Lemma received_self_action : forall A c a c' es, inv c -> check_action c A a = inl (c', es) ->
    received A c' = received A c.
Proof.
  intros A c a c' es Hinv H. apply (received_action A c A a c' es Hinv H).
  - intros id ->. destruct (grant_next _ _ _ _ _ _ H) as [fb [_ [_ [Hu _]]]]. exact (Hu eq_refl).
  - intros id ->. destruct (free_next _ _ _ _ _ H) as [fb [Fb [Ho _]]].
    destruct (has_grant c id A) eqn:Hg; [| reflexivity].
    exfalso. exact (has_grant_owner c id A fb Hinv Hg Fb Ho).
Qed.

Definition view_result (A : Z) (d : catalog * list effect + String.string) : catalog * list effect + String.string :=
  match d with inl (c', es) => inl (view A c', es) | inr r => inr r end.

Lemma view_action : forall A c a, inv c -> res_inv c -> find_reservation c A <> None ->
    check_action (view A c) A a = view_result A (check_action c A a).
Proof.
  intros A c a Hinv Hr Hres.
  assert (Hlift : forall a, (forall op gx gy gz buffers sizes scalars guarded, a <> Launch op gx gy gz buffers sizes scalars guarded) ->
            check_action (own A c) A a = own_result A (check_action c A a) ->
            check_action (view A c) A a = view_result A (check_action c A a)).
  { intros a0 Hnl Hown. unfold view at 1. rewrite (extend_action A (received A c) (own A c) a0 (extend_ok_view A c Hinv Hr) Hnl).
    rewrite Hown. destruct (check_action c A a0) as [[c' es] | r] eqn:E; [| reflexivity].
    cbn [own_result extend_result view_result]. unfold view. rewrite (received_self_action A c a0 c' es Hinv E). reflexivity. }
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |];
    try (apply Hlift; [intros; discriminate |]).
  - exact (own_allocate A c id o n Hinv Hr).
  - exact (own_free A c id Hinv Hr).
  - exact (own_grant A c id u Hinv Hr).
  - exact (own_write A c id data element Hinv Hr).
  - exact (own_fill A c id e s v Hinv Hr).
  - cbn [check_action]. rewrite (view_check_launch A c op gx gy gz buffers sizes scalars guarded Hinv Hr).
    destruct (check_launch c A op gx gy gz buffers sizes scalars guarded) as [[c' k] | r]; reflexivity.
  - exact (own_reserve A c o n Hres).
  - exact (own_unreserve A c).
Qed.

Definition view_decision (A : Z) (d : decision) : decision :=
  match d with Accepted c es => Accepted (view A c) es | Refused r => Refused r end.

Lemma view_check_actions : forall A acts c pos done, inv c -> res_inv c ->
    find_reservation c A <> None -> ~ In Unreserve acts ->
    check_actions (view A c) A pos acts done = view_decision A (check_actions c A pos acts done).
Proof.
  intros A acts. induction acts as [| a acts IH]; intros c pos done Hinv Hr Hres Hun; [reflexivity |].
  cbn [check_actions]. rewrite (view_action A c a Hinv Hr Hres).
  destruct (check_action c A a) as [[c1 es1] | r] eqn:E; cbn [view_result]; [| reflexivity].
  apply IH.
  - exact (check_action_inv _ _ _ _ _ Hinv E).
  - exact (res_action _ _ _ _ _ Hinv Hr E).
  - destruct (find_reservation c A) as [r |] eqn:Er; [| exfalso; exact (Hres eq_refl)].
    rewrite (reservation_action A c A a c1 es1 r Er (fun H => False_ind _ (Hun (or_introl H))) E).
    discriminate.
  - intros H. exact (Hun (or_intror H)).
Qed.

(* A's step is decided on A's view as on the whole catalog. *)
Lemma view_check_step : forall A c s, s_tenant s = A -> inv c -> res_inv c ->
    find_reservation c A <> None -> ~ In Unreserve (s_actions s) ->
    check_step (view A c) s = view_decision A (check_step c s).
Proof.
  intros A c s Ht Hinv Hr Hres Hun. unfold check_step. rewrite Ht.
  exact (view_check_actions A _ c 0 [] Hinv Hr Hres Hun).
Qed.

Section Run.

Variable base : Z.
Variable kernels : Z -> option (program * lfacts).

Lemma effect_ext : forall ch c1 c2 a m e m', c_pool c1 = c_pool c2 -> c_operators c1 = c_operators c2 ->
    (forall op gx gy gz buffers sizes scalars guarded, a = Launch op gx gy gz buffers sizes scalars guarded ->
       forall id off, In (id, off) buffers -> find_buffer c2 id = find_buffer c1 id) ->
    effect_run base kernels ch c1 a m e m' -> effect_run base kernels ch c2 a m e m'.
Proof.
  intros ch c1 c2 a m e m' Hp Hop Hargs H. destruct H as [a m o n | a m o data | a m o n e s v
    | m op gx gy gz buffers sizes scalars guarded args checks Mf prog F sh0 lm0 Hb Hk Hhw Hck Hl
    | m op gx gy gz shared buffers sizes scalars guarded args checks Hck].
  - apply RunClear.
  - apply RunUpload.
  - apply RunMemset.
  - assert (Ha : args_of c2 sizes (b_params Mf) buffers scalars = args_of c1 sizes (b_params Mf) buffers scalars)
      by (apply args_of_ext; exact (Hargs op gx gy gz buffers sizes scalars guarded eq_refl)).
    rewrite <- (kernel_run_same base F prog Mf c2 c1 gx gy gz buffers sizes scalars ch sh0 lm0 m Ha).
    apply (RunKernel base kernels ch c2 m op gx gy gz buffers sizes scalars guarded args checks Mf prog F sh0 lm0);
      [| exact Hk | | exact Hck | exact Hl].
    + unfold find_binding in *. rewrite <- Hop. exact Hb.
    + apply (hardware_same base F Mf c1 c2 gx gy gz buffers sizes scalars ch Hp); [symmetry; exact Ha | exact Hhw].
  - apply RunSkipped. exact Hck.
Qed.

Lemma effects_ext : forall ch c1 c2 a m es m', c_pool c1 = c_pool c2 -> c_operators c1 = c_operators c2 ->
    (forall op gx gy gz buffers sizes scalars guarded, a = Launch op gx gy gz buffers sizes scalars guarded ->
       forall id off, In (id, off) buffers -> find_buffer c2 id = find_buffer c1 id) ->
    effects_run base kernels ch c1 a m es m' -> effects_run base kernels ch c2 a m es m'.
Proof.
  intros ch c1 c2 a m es m' Hp Hop Hargs H. induction H as [m | m e es m1 m2 He Hr IH].
  - apply RunNone.
  - apply (RunMore base kernels ch c2 a m e es m1 m2); [exact (effect_ext ch c1 c2 a m e m1 Hp Hop Hargs He) | exact IH].
Qed.

Lemma run_action_view : forall ch A c m a c' m', inv c -> res_inv c -> find_reservation c A <> None ->
    run_action base kernels ch A {| w_cat := c; w_mem := m |} a {| w_cat := c'; w_mem := m' |} ->
    run_action base kernels ch A {| w_cat := view A c; w_mem := m |} a {| w_cat := view A c'; w_mem := m' |}.
Proof.
  intros ch A c m a c' m' Hinv Hr Hres H. inversion H as [c0 m0 a0 c1 es m1 Hc He]; subst.
  apply (RunAction base kernels ch A (view A c) m a (view A c') es m').
  - rewrite (view_action A c a Hinv Hr Hres), Hc. reflexivity.
  - apply (effects_ext ch c (view A c)); [reflexivity | reflexivity | | exact He].
    intros op gx gy gz buffers sizes scalars guarded -> id off Hin.
    destruct (launch_next _ _ _ _ _ _ _ _ _ _ _ _ Hc) as [Mf [_ [[r L] _]]].
    exact (launch_args_view A c op gx gy gz buffers sizes scalars guarded r Hinv L id off Hin).
Qed.

(* A run of A's actions runs on A's view as on the whole catalog, and leaves
   the buffers granted to A as they are. *)
Lemma run_actions_view : forall A chs w acts w', sound base kernels w -> res_inv (w_cat w) ->
    find_reservation (w_cat w) A <> None -> ~ In Unreserve acts ->
    run_actions base kernels A chs w acts w' ->
    run_actions base kernels A chs {| w_cat := view A (w_cat w); w_mem := w_mem w |} acts
                                   {| w_cat := view A (w_cat w'); w_mem := w_mem w' |} /\
    received A (w_cat w') = received A (w_cat w).
Proof.
  intros A chs w acts w' Hs Hr Hres Hun H. induction H as [w | ch chs w a w1 acts w2 H1 _ IH].
  - split; [apply RunDone | reflexivity].
  - pose proof (sound_action _ _ _ _ _ _ _ Hs H1) as Hs1.
    destruct w as [c m]. destruct w1 as [c1 m1].
    inversion H1 as [c0 m0 a0 c1' es m1' Hc _]; subst. cbn [w_cat w_mem] in *.
    pose proof Hs as [[Hinv _] _].
    destruct IH as [IHrun IHrec]; [exact Hs1 | exact (res_action _ _ _ _ _ Hinv Hr Hc) | | |].
    + destruct (find_reservation c A) as [r |] eqn:Er; [| exfalso; exact (Hres eq_refl)].
      rewrite (reservation_action A c A a c1 es r Er (fun E => False_ind _ (Hun (or_introl E))) Hc).
      discriminate.
    + intros Hin. exact (Hun (or_intror Hin)).
    + split.
      * apply (RunNext base kernels A ch chs _ a {| w_cat := view A c1; w_mem := m1 |}); [| exact IHrun].
        exact (run_action_view ch A c m a c1 m1 Hinv Hr Hres H1).
      * rewrite IHrec. exact (received_self_action A c a c1 es Hinv Hc).
Qed.

End Run.

(* ------------------------------------------------------------------ *)
(* The events of a tenant.                                             *)

Section Pipeline.

Variable base : Z.
Variable kernels : Z -> option (program * lfacts).

(* The bytes of buffer b in memory m, from its first byte. *)
Definition bytes_at (m : memory) (b : buffer) : list Z :=
  map (fun k => m (base + bf_offset b + Z.of_nat k)) (seq 0 (Z.to_nat (bf_bytes b))).

Lemma bytes_at_agree : forall m1 m2 b y, bytes_at m1 b = bytes_at m2 b -> in_buffer base b y -> m1 y = m2 y.
Proof.
  intros m1 m2 b y H Hy. unfold in_buffer in Hy.
  assert (Hk : In (Z.to_nat (y - base - bf_offset b)) (seq 0 (Z.to_nat (bf_bytes b)))) by (apply in_seq; lia).
  pose proof (map_eq_in _ _ _ _ _ _ H Hk) as E. cbn beta in E.
  rewrite Z2Nat.id in E by lia. replace (base + bf_offset b + (y - base - bf_offset b)) with y in E by lia. exact E.
Qed.

(* What is released to A: A's own steps, each with the choices of the
   hardware for it; each grant to A, with the buffer's record and its bytes
   when granted; and each end of a grant to A, the owner's free of the
   buffer. *)
Inductive event :=
| OStep (s : step) (chs : list choice)
| OGrant (b : buffer) (data : list Z)
| ORevoke (id : bid).

(* The event of A, if any, of an action of another tenant in world w. *)
Definition action_event (A : Z) (w : world) (a : action) : option event :=
  match a with
  | Grant id u =>
      if Z.eqb u A then
        match find_buffer (w_cat w) id with Some b => Some (OGrant b (bytes_at (w_mem w) b)) | None => None end
      else None
  | Free id => if has_grant (w_cat w) id A then Some (ORevoke id) else None
  | _ => None
  end.

Definition cons_event (e : option event) (os : list event) : list event :=
  match e with Some x => x :: os | None => os end.

(* A run of the actions of another tenant t, with its events of A. *)
Inductive observe_actions (A t : Z) : list choice -> world -> list action -> world -> list event -> Prop :=
| ObsDone : forall w, observe_actions A t [] w [] w []
| ObsNext : forall ch chs w a w1 acts w2 os, run_action base kernels ch t w a w1 ->
    observe_actions A t chs w1 acts w2 os ->
    observe_actions A t (ch :: chs) w (a :: acts) w2 (cons_event (action_event A w a) os).

(* A run of a step, with its events of A. *)
Inductive observe_step (A : Z) (chs : list choice) : world -> step -> world -> list event -> Prop :=
| ObsOwn : forall w s w', s_tenant s = A -> run_step base kernels chs w s w' ->
    observe_step A chs w s w' [OStep s chs]
| ObsOther : forall w s c' es w' os, s_tenant s <> A -> check_step (w_cat w) s = Accepted c' es ->
    observe_actions A (s_tenant s) chs w (s_actions s) w' os -> observe_step A chs w s w' os
| ObsRefused : forall w s r, s_tenant s <> A -> check_step (w_cat w) s = Refused r -> observe_step A chs w s w [].

(* A run of steps, with its sequence of events of A. *)
Inductive observe_steps (A : Z) : list (list choice) -> world -> list step -> world -> list event -> Prop :=
| ObsEnd : forall w, observe_steps A [] w [] w []
| ObsStep : forall chs chss w s w1 ss w2 os1 os2, observe_step A chs w s w1 os1 ->
    observe_steps A chss w1 ss w2 os2 -> observe_steps A (chs :: chss) w (s :: ss) w2 (os1 ++ os2).

Lemma observe_actions_run : forall A t chs w acts w' os, observe_actions A t chs w acts w' os ->
    run_actions base kernels t chs w acts w'.
Proof.
  intros A t chs w acts w' os H. induction H as [w | ch chs w a w1 acts w2 os Ha _ IH];
    [apply RunDone | exact (RunNext base kernels t ch chs w a w1 acts w2 Ha IH)].
Qed.

Lemma run_observe_actions : forall A t chs w acts w', run_actions base kernels t chs w acts w' ->
    exists os, observe_actions A t chs w acts w' os.
Proof.
  intros A t chs w acts w' H. induction H as [w | ch chs w a w1 acts w2 Ha _ [os IH]].
  - exists []. apply ObsDone.
  - exists (cons_event (action_event A w a) os). exact (ObsNext A t ch chs w a w1 acts w2 os Ha IH).
Qed.

(* The events of a run are a record of the run: a run of steps is the run of
   its observation, and every run of steps has an observation. *)
Theorem observe_steps_run : forall A chss w steps w' os, observe_steps A chss w steps w' os ->
    run_steps base kernels chss w steps w'.
Proof.
  intros A chss w steps w' os H. induction H as [w | chs chss w s w1 ss w2 os1 os2 Hs _ IH]; [apply RunEnd |].
  apply (RunStep base kernels chs chss w s w1 ss w2); [| exact IH].
  destruct Hs as [w s w' Ht Hr | w s c' es w' os Ht Hc Ha | w s r Ht Hc].
  - exact Hr.
  - exact (RunAccepted base kernels chs w s c' es w' Hc (observe_actions_run _ _ _ _ _ _ _ Ha)).
  - exact (RunRefused base kernels chs w s r Hc).
Qed.

Theorem run_observe_steps : forall A chss w steps w', run_steps base kernels chss w steps w' ->
    exists os, observe_steps A chss w steps w' os.
Proof.
  intros A chss w steps w' H. induction H as [w | chs chss w s w1 ss w2 Hs _ [os2 IH]].
  - exists []. apply ObsEnd.
  - assert (Hstep : exists os1, observe_step A chs w s w1 os1).
    { destruct (Z.eq_dec (s_tenant s) A) as [Ht | Ht]; [exists [OStep s chs]; exact (ObsOwn A chs w s w1 Ht Hs) |].
      destruct Hs as [w s c' es w1 Hc Ha | w s r Hc].
      - destruct (run_observe_actions A (s_tenant s) chs w (s_actions s) w1 Ha) as [os1 Ho].
        exists os1. exact (ObsOther A chs w s c' es w1 os1 Ht Hc Ho).
      - exists []. exact (ObsRefused A chs w s r Ht Hc). }
    destruct Hstep as [os1 Hs1]. exists (os1 ++ os2). exact (ObsStep A chs chss w s w1 ss w2 os1 os2 Hs1 IH).
Qed.

(* ------------------------------------------------------------------ *)
(* Moves: the steps of A and the actions of the other tenants.         *)

Inductive move (A : Z) : world -> option event -> world -> Prop :=
| MoveOwn : forall chs w s w', s_tenant s = A -> run_step base kernels chs w s w' -> move A w (Some (OStep s chs)) w'
| MoveOther : forall ch t w a w', t <> A -> run_action base kernels ch t w a w' -> move A w (action_event A w a) w'.

Inductive moves (A : Z) : world -> list (option event) -> world -> Prop :=
| MovesEnd : forall w, moves A w [] w
| MovesNext : forall w e w1 l w2, move A w e w1 -> moves A w1 l w2 -> moves A w (e :: l) w2.

Definition events (l : list (option event)) : list event := fold_right cons_event [] l.

Lemma events_app : forall l1 l2, events (l1 ++ l2) = events l1 ++ events l2.
Proof.
  intros l1 l2. induction l1 as [| e l1 IH]; [reflexivity |]. cbn [app events fold_right].
  change (fold_right cons_event [] (l1 ++ l2)) with (events (l1 ++ l2)). rewrite IH. destruct e; reflexivity.
Qed.

Lemma moves_app : forall A w l1 w1 l2 w2, moves A w l1 w1 -> moves A w1 l2 w2 -> moves A w (l1 ++ l2) w2.
Proof.
  intros A w l1 w1 l2 w2 H1 H2. induction H1 as [w | w e wm l w' M _ IH]; [exact H2 |].
  exact (MovesNext A w e wm (l ++ l2) w2 M (IH H2)).
Qed.

Lemma action_event_step : forall A w a s chs, action_event A w a <> Some (OStep s chs).
Proof.
  intros A w a s chs. destruct a; cbn [action_event]; try discriminate.
  - destruct (has_grant (w_cat w) id A); discriminate.
  - destruct (Z.eqb tenant A); [destruct (find_buffer (w_cat w) id) |]; discriminate.
Qed.

Lemma observe_actions_moves : forall A t chs w acts w' os, t <> A -> observe_actions A t chs w acts w' os ->
    exists l, moves A w l w' /\ events l = os /\ forall s c, ~ In (Some (OStep s c)) l.
Proof.
  intros A t chs w acts w' os Ht H. induction H as [w | ch chs w a w1 acts w2 os Ha _ [l [Hl [He Hn]]]].
  - exists []. split; [apply MovesEnd | split; [reflexivity | intros s c []]].
  - exists (action_event A w a :: l). split; [exact (MovesNext A w _ w1 l w2 (MoveOther A ch t w a w1 Ht Ha) Hl) |].
    split; [cbn [events fold_right]; change (fold_right cons_event [] l) with (events l); rewrite He; reflexivity |].
    intros s c [E | Hin]; [exact (action_event_step A w a s c E) | exact (Hn s c Hin)].
Qed.

Lemma observe_steps_moves : forall A chss w steps w' os, observe_steps A chss w steps w' os ->
    exists l, moves A w l w' /\ events l = os /\
      forall s c, In (Some (OStep s c)) l -> In s steps /\ s_tenant s = A.
Proof.
  intros A chss w steps w' os H. induction H as [w | chs chss w s w1 ss w2 os1 os2 Hs _ [l2 [Hl2 [He2 Hn2]]]].
  - exists []. split; [apply MovesEnd | split; [reflexivity | intros s c []]].
  - assert (Hstep : exists l1, moves A w l1 w1 /\ events l1 = os1 /\
                      forall s' c, In (Some (OStep s' c)) l1 -> s' = s /\ s_tenant s = A).
    { destruct Hs as [w s w' Ht Hr | w s c' es w' os Ht Hc Ha | w s r Ht Hc].
      - exists [Some (OStep s chs)].
        split; [exact (MovesNext A w _ w' [] w' (MoveOwn A chs w s w' Ht Hr) (MovesEnd A w')) |].
        split; [reflexivity |]. intros s' c [E | []]. injection E as -> _. split; [reflexivity | exact Ht].
      - destruct (observe_actions_moves A (s_tenant s) chs w (s_actions s) w' os Ht Ha) as [l1 [Hl1 [He1 Hn1]]].
        exists l1. split; [exact Hl1 | split; [exact He1 |]]. intros s' c Hin. exfalso. exact (Hn1 s' c Hin).
      - exists []. split; [apply MovesEnd | split; [reflexivity | intros s' c []]]. }
    destruct Hstep as [l1 [Hl1 [He1 Hn1]]]. exists (l1 ++ l2).
    split; [exact (moves_app A w l1 w1 l2 w2 Hl1 Hl2) |]. split; [rewrite events_app, He1, He2; reflexivity |].
    intros s' c Hin. apply in_app_or in Hin. destruct Hin as [Hin | Hin].
    + destruct (Hn1 s' c Hin) as [-> Ht]. split; [left; reflexivity | exact Ht].
    + destruct (Hn2 s' c Hin) as [Hs' Ht]. split; [right; exact Hs' | exact Ht].
Qed.

Lemma move_cases : forall A w o w', move A w o w' ->
    (exists chs s, o = Some (OStep s chs) /\ s_tenant s = A /\ run_step base kernels chs w s w') \/
    (exists ch t a, o = action_event A w a /\ t <> A /\ run_action base kernels ch t w a w').
Proof.
  intros A w o w' H. destruct H as [chs w s w' Ht Hr | ch t w a w' Ht Ha].
  - left. exists chs, s. split; [reflexivity | split; assumption].
  - right. exists ch, t, a. split; [reflexivity | split; assumption].
Qed.

(* ------------------------------------------------------------------ *)
(* Two runs related for A.                                             *)

(* Their invariants hold, A holds its reservation, their own parts of A and
   their buffers granted to A are equal, and their memories agree on what A
   observes. *)
Definition grant_related (A : Z) (w1 w2 : world) : Prop :=
  sound base kernels w1 /\ sound base kernels w2 /\ res_inv (w_cat w1) /\ res_inv (w_cat w2) /\
  NoDup (c_grants (w_cat w1)) /\ NoDup (c_grants (w_cat w2)) /\ find_reservation (w_cat w1) A <> None /\
  own A (w_cat w1) = own A (w_cat w2) /\ received A (w_cat w1) = received A (w_cat w2) /\
  agree_on base (w_cat w1) A (w_mem w1) (w_mem w2).

(* An action of another tenant keeps the invariants and A's own part. *)
Lemma other_facts : forall A ch t w a w', t <> A -> sound base kernels w -> res_inv (w_cat w) ->
    NoDup (c_grants (w_cat w)) -> run_action base kernels ch t w a w' ->
    sound base kernels w' /\ res_inv (w_cat w') /\ NoDup (c_grants (w_cat w')) /\ own A (w_cat w') = own A (w_cat w) /\
    (forall r, find_reservation (w_cat w) A = Some r -> find_reservation (w_cat w') A = Some r).
Proof.
  intros A ch t w a w' Ht Hs Hr Hn H. pose proof (sound_action _ _ _ _ _ _ _ Hs H) as Hs'.
  inversion H as [c m a0 c' es m' Hc _]; subst. cbn [w_cat] in *. pose proof Hs as [[Hinv _] _].
  split; [exact Hs' |]. split; [exact (res_action _ _ _ _ _ Hinv Hr Hc) |].
  split; [exact (grants_nodup_action _ _ _ _ _ Hn Hc) |].
  split; [exact (own_other A t c a c' es (fun E => Ht (eq_sym E)) Hinv Hr Hc) |].
  intros r Er. exact (reservation_action A c t a c' es r Er (fun _ => Ht) Hc).
Qed.

(* An action of another tenant without an event of A leaves the buffers
   granted to A as they are, and keeps every byte that A observes after it. *)
Lemma silent_action : forall A ch t w a w', t <> A -> sound base kernels w -> action_event A w a = None ->
    run_action base kernels ch t w a w' ->
    received A (w_cat w') = received A (w_cat w) /\
    forall y, observable base (w_cat w') A y -> observable base (w_cat w) A y /\ w_mem w' y = w_mem w y.
Proof.
  intros A ch t w a w' Ht Hs He H. destruct w as [c m]. destruct w' as [c' m'].
  pose proof Hs as [[Hinv _] _]. inversion H as [c0 m0 a0 c1 es m1 Hc _]; subst. cbn [w_cat w_mem] in *.
  split.
  - apply (received_action A c t a c' es Hinv Hc).
    + intros id ->. cbn [action_event w_cat] in He. rewrite Z.eqb_refl in He.
      destruct (grant_next _ _ _ _ _ _ Hc) as [fb [Fb _]]. rewrite Fb in He. discriminate He.
    + intros id ->. cbn [action_event w_cat] in He. destruct (has_grant c id A); [discriminate He | reflexivity].
  - apply (other_action base kernels ch A t c m a c' m' (fun E => Ht (eq_sym E)) Hs); [| exact H].
    intros id u -> Hu. subst u. cbn [action_event w_cat] in He. rewrite Z.eqb_refl in He.
    destruct (grant_next _ _ _ _ _ _ Hc) as [fb [Fb _]]. rewrite Fb in He. discriminate He.
Qed.

Lemma silent1 : forall A ch t w1 a w1' w2, t <> A -> grant_related A w1 w2 -> action_event A w1 a = None ->
    run_action base kernels ch t w1 a w1' -> grant_related A w1' w2.
Proof.
  intros A ch t w1 a w1' w2 Ht (Hs1 & Hs2 & Hr1 & Hr2 & Hn1 & Hn2 & Hres & Hown & Hrec & Hag) He H.
  destruct (other_facts A ch t w1 a w1' Ht Hs1 Hr1 Hn1 H) as (Hs1' & Hr1' & Hn1' & Hown' & Hkeep).
  destruct (silent_action A ch t w1 a w1' Ht Hs1 He H) as [Hrec' Hmem].
  split; [exact Hs1' |]. split; [exact Hs2 |]. split; [exact Hr1' |]. split; [exact Hr2 |].
  split; [exact Hn1' |]. split; [exact Hn2 |].
  split; [destruct (find_reservation (w_cat w1) A) as [r |] eqn:Er; [rewrite (Hkeep r eq_refl); discriminate | contradiction] |].
  split; [rewrite Hown'; exact Hown |]. split; [rewrite Hrec'; exact Hrec |].
  intros y Hy. destruct (Hmem y Hy) as [Hy0 E]. rewrite E. exact (Hag y Hy0).
Qed.

Lemma silent2 : forall A ch t w1 w2 a w2', t <> A -> grant_related A w1 w2 -> action_event A w2 a = None ->
    run_action base kernels ch t w2 a w2' -> grant_related A w1 w2'.
Proof.
  intros A ch t w1 w2 a w2' Ht (Hs1 & Hs2 & Hr1 & Hr2 & Hn1 & Hn2 & Hres & Hown & Hrec & Hag) He H.
  destruct (other_facts A ch t w2 a w2' Ht Hs2 Hr2 Hn2 H) as (Hs2' & Hr2' & Hn2' & Hown' & _).
  destruct (silent_action A ch t w2 a w2' Ht Hs2 He H) as [Hrec' Hmem].
  split; [exact Hs1 |]. split; [exact Hs2' |]. split; [exact Hr1 |]. split; [exact Hr2' |].
  split; [exact Hn1 |]. split; [exact Hn2' |]. split; [exact Hres |].
  split; [rewrite Hown'; exact Hown |]. split; [rewrite Hrec'; exact Hrec |].
  intros y Hy. pose proof Hs1 as [[Hi1 _] _]. pose proof Hs2 as [[Hi2 _] _]. pose proof Hs2' as [[Hi2' _] _].
  assert (Hy2 : observable base (w_cat w2') A y).
  { apply (observable_related base A (w_cat w1) (w_cat w2') y Hi1 Hi2'); [congruence | congruence | exact Hy]. }
  destruct (Hmem y Hy2) as [_ E]. rewrite E. exact (Hag y Hy).
Qed.

(* After a grant of buffer b to A, A observes what it observed before and b. *)
Lemma observable_grant_to : forall A c t id b c' es y, inv c -> find_buffer c id = Some b ->
    check_action c t (Grant id A) = inl (c', es) ->
    observable base c' A y -> observable base c A y \/ in_buffer base b y.
Proof.
  intros A c t id b c' es y Hinv Fb H Hy. destruct (grant_next _ _ _ _ _ _ H) as [fb [_ [_ [_ [_ ->]]]]].
  destruct Hy as [[b0 [Hb0 [Hr Hy]]] | Hf]; [| left; right; exact Hf].
  cbn [c_buffers] in Hb0.
  assert (Hr' : Z.eqb (bf_owner b0) A || has_grant c (bf_id b0) A = true \/ id = bf_id b0).
  { unfold readable in Hr. unfold find_buffer in Hr at 1. cbn [c_buffers] in Hr. fold (find_buffer c (bf_id b0)) in Hr.
    rewrite (find_buffer_of c b0 Hinv Hb0) in Hr. apply orb_true_iff in Hr. destruct Hr as [Ho | Hg].
    - left. rewrite Ho. reflexivity.
    - apply has_grant_exists in Hg. destruct Hg as [g [Hg [Ei Eu]]]. cbn [c_grants] in Hg. apply in_app_or in Hg.
      destruct Hg as [Hg | [<- | []]].
      + left. apply orb_true_iff. right. apply has_grant_exists. exists g. split; [exact Hg | split; assumption].
      + right. exact Ei. }
  destruct Hr' as [Hr' | Ei].
  - left. left. exists b0. split; [exact Hb0 |]. split; [| exact Hy].
    unfold readable. rewrite (find_buffer_of c b0 Hinv Hb0). exact Hr'.
  - right. rewrite Ei, (find_buffer_of c b0 Hinv Hb0) in Fb. injection Fb as <-. exact Hy.
Qed.

Lemma action_event_grant : forall A w a b data, action_event A w a = Some (OGrant b data) ->
    exists id, a = Grant id A /\ find_buffer (w_cat w) id = Some b /\ data = bytes_at (w_mem w) b.
Proof.
  intros A w a b data H. destruct a; cbn [action_event] in H; try discriminate H.
  - destruct (has_grant (w_cat w) id A); discriminate H.
  - destruct (Z.eqb_spec tenant A) as [-> | _]; [| discriminate H].
    destruct (find_buffer (w_cat w) id) as [b' |] eqn:F; [| discriminate H]. injection H as <- <-.
    exists id. split; [reflexivity | split; [exact F | reflexivity]].
Qed.

Lemma action_event_revoke : forall A w a id, action_event A w a = Some (ORevoke id) ->
    a = Free id /\ has_grant (w_cat w) id A = true.
Proof.
  intros A w a id H. destruct a; cbn [action_event] in H; try discriminate H.
  - destruct (has_grant (w_cat w) id0 A) eqn:E; [| discriminate H]. injection H as <-. split; [reflexivity | exact E].
  - destruct (Z.eqb tenant A); [destruct (find_buffer (w_cat w) id0) |]; discriminate H.
Qed.

(* The same grant to A in both runs, of the same buffer with the same bytes. *)
Lemma grant_sync : forall A ch1 ch2 t1 t2 w1 w2 a1 a2 w1' w2' b data, t1 <> A -> t2 <> A ->
    grant_related A w1 w2 -> action_event A w1 a1 = Some (OGrant b data) -> action_event A w2 a2 = Some (OGrant b data) ->
    run_action base kernels ch1 t1 w1 a1 w1' -> run_action base kernels ch2 t2 w2 a2 w2' -> grant_related A w1' w2'.
Proof.
  intros A ch1 ch2 t1 t2 w1 w2 a1 a2 w1' w2' b data Ht1 Ht2
    (Hs1 & Hs2 & Hr1 & Hr2 & Hn1 & Hn2 & Hres & Hown & Hrec & Hag) He1 He2 H1 H2.
  destruct (other_facts A ch1 t1 w1 a1 w1' Ht1 Hs1 Hr1 Hn1 H1) as (Hs1' & Hr1' & Hn1' & Hown1 & Hkeep).
  destruct (other_facts A ch2 t2 w2 a2 w2' Ht2 Hs2 Hr2 Hn2 H2) as (Hs2' & Hr2' & Hn2' & Hown2 & _).
  destruct (action_event_grant A w1 a1 b data He1) as [id1 [-> [F1 D1]]].
  destruct (action_event_grant A w2 a2 b data He2) as [id2 [-> [F2 D2]]].
  destruct w1 as [c1 m1]. destruct w2 as [c2 m2]. cbn [w_cat w_mem] in *.
  inversion H1 as [? ? ? c1' es1 m1' Hc1 He1']; subst.
  inversion H2 as [? ? ? c2' es2 m2' Hc2 He2']; subst. cbn [w_cat w_mem] in *.
  destruct (grant_next _ _ _ _ _ _ Hc1) as [_ [_ [_ [_ [-> _]]]]]. pose proof (effects_nil _ _ _ _ _ _ _ He1') as Em1. subst m1'.
  destruct (grant_next _ _ _ _ _ _ Hc2) as [_ [_ [_ [_ [-> _]]]]]. pose proof (effects_nil _ _ _ _ _ _ _ He2') as Em2. subst m2'.
  destruct (received_grant A c1 t1 id1 c1' _ Hc1) as [b1 [F1' R1]]. rewrite F1 in F1'. injection F1' as <-.
  destruct (received_grant A c2 t2 id2 c2' _ Hc2) as [b2 [F2' R2]]. rewrite F2 in F2'. injection F2' as <-.
  pose proof Hs1 as [[Hi1 _] _]. unfold grant_related. cbn [w_cat w_mem] in *.
  split; [exact Hs1' |]. split; [exact Hs2' |]. split; [exact Hr1' |]. split; [exact Hr2' |].
  split; [exact Hn1' |]. split; [exact Hn2' |].
  split; [destruct (find_reservation c1 A) as [r |] eqn:Er; [rewrite (Hkeep r eq_refl); discriminate | contradiction] |].
  split; [rewrite Hown1, Hown2; exact Hown |]. split; [rewrite R1, R2, Hrec; reflexivity |].
  intros y Hy. destruct (observable_grant_to A c1 t1 id1 b c1' _ y Hi1 F1 Hc1 Hy) as [Hy0 | Hb].
  - exact (Hag y Hy0).
  - apply (bytes_at_agree m1 m2 b y); [congruence | exact Hb].
Qed.

(* The same end of a grant to A in both runs. *)
Lemma revoke_sync : forall A ch1 ch2 t1 t2 w1 w2 a1 a2 w1' w2' id, t1 <> A -> t2 <> A ->
    grant_related A w1 w2 -> action_event A w1 a1 = Some (ORevoke id) -> action_event A w2 a2 = Some (ORevoke id) ->
    run_action base kernels ch1 t1 w1 a1 w1' -> run_action base kernels ch2 t2 w2 a2 w2' -> grant_related A w1' w2'.
Proof.
  intros A ch1 ch2 t1 t2 w1 w2 a1 a2 w1' w2' id Ht1 Ht2
    (Hs1 & Hs2 & Hr1 & Hr2 & Hn1 & Hn2 & Hres & Hown & Hrec & Hag) He1 He2 H1 H2.
  destruct (other_facts A ch1 t1 w1 a1 w1' Ht1 Hs1 Hr1 Hn1 H1) as (Hs1' & Hr1' & Hn1' & Hown1 & Hkeep).
  destruct (other_facts A ch2 t2 w2 a2 w2' Ht2 Hs2 Hr2 Hn2 H2) as (Hs2' & Hr2' & Hn2' & Hown2 & _).
  destruct (action_event_revoke A w1 a1 id He1) as [-> _]. destruct (action_event_revoke A w2 a2 id He2) as [-> _].
  assert (Hrec' : received A (w_cat w1') = received A (w_cat w2')).
  { destruct w1 as [c1 m1]. destruct w2 as [c2 m2].
    inversion H1 as [? ? ? c1' es1 m1' Hc1 _]; subst. inversion H2 as [? ? ? c2' es2 m2' Hc2 _]; subst.
    cbn [w_cat] in *. rewrite (received_free A c1 t1 id c1' es1 Hc1), (received_free A c2 t2 id c2' es2 Hc2), Hrec.
    reflexivity. }
  assert (Hng : forall i u, Free id = Grant i u -> u <> A) by (intros i u E; discriminate E).
  split; [exact Hs1' |]. split; [exact Hs2' |]. split; [exact Hr1' |]. split; [exact Hr2' |].
  split; [exact Hn1' |]. split; [exact Hn2' |].
  split; [destruct (find_reservation (w_cat w1) A) as [r |] eqn:Er; [rewrite (Hkeep r eq_refl); discriminate | contradiction] |].
  split; [rewrite Hown1, Hown2; exact Hown |]. split; [exact Hrec' |].
  intros y Hy. pose proof Hs1' as [[Hi1' _] _]. pose proof Hs2' as [[Hi2' _] _].
  assert (Hy2 : observable base (w_cat w2') A y)
    by (apply (observable_related base A (w_cat w1') (w_cat w2') y Hi1' Hi2'); [congruence | exact Hrec' | exact Hy]).
  destruct w1 as [c1 m1]. destruct w2 as [c2 m2]. destruct w1' as [c1' m1']. destruct w2' as [c2' m2'].
  cbn [w_cat w_mem] in *.
  destruct (other_action base kernels ch1 A t1 c1 m1 (Free id) c1' m1' (fun E => Ht1 (eq_sym E)) Hs1 Hng H1 y Hy) as [Hy1 E1].
  destruct (other_action base kernels ch2 A t2 c2 m2 (Free id) c2' m2' (fun E => Ht2 (eq_sym E)) Hs2 Hng H2 y Hy2) as [_ E2].
  rewrite E1, E2. exact (Hag y Hy1).
Qed.

(* The invariants that every run of actions keeps. *)
Lemma ok_actions : forall t chs w acts w', sound base kernels w -> res_inv (w_cat w) -> NoDup (c_grants (w_cat w)) ->
    run_actions base kernels t chs w acts w' ->
    sound base kernels w' /\ res_inv (w_cat w') /\ NoDup (c_grants (w_cat w')).
Proof.
  intros t chs w acts w' Hs Hr Hn H. induction H as [w | ch chs w a w1 acts w2 H1 _ IH]; [split; [exact Hs | split; assumption] |].
  pose proof (sound_action _ _ _ _ _ _ _ Hs H1) as Hs1.
  inversion H1 as [c m a0 c1 es m1 Hc _]; subst. cbn [w_cat] in *. pose proof Hs as [[Hinv _] _].
  exact (IH Hs1 (res_action _ _ _ _ _ Hinv Hr Hc) (grants_nodup_action _ _ _ _ _ Hn Hc)).
Qed.

(* The same step of A in both runs, with the same choices of the hardware. *)
Lemma self_sync : forall A chs w1 w2 s w1' w2', s_tenant s = A -> ~ In Unreserve (s_actions s) ->
    grant_related A w1 w2 -> run_step base kernels chs w1 s w1' -> run_step base kernels chs w2 s w2' ->
    grant_related A w1' w2'.
Proof.
  intros A chs w1 w2 s w1' w2' Ht Hun (Hs1 & Hs2 & Hr1 & Hr2 & Hn1 & Hn2 & Hres & Hown & Hrec & Hag) H1 H2.
  pose proof Hs1 as [[Hi1 _] _]. pose proof Hs2 as [[Hi2 _] _].
  assert (Hres2 : find_reservation (w_cat w2) A <> None)
    by (rewrite <- find_reservation_own, <- Hown, find_reservation_own; exact Hres).
  pose proof (view_same A (w_cat w1) (w_cat w2) Hown Hrec) as Hview.
  assert (Hdec : view_decision A (check_step (w_cat w1) s) = view_decision A (check_step (w_cat w2) s)).
  { rewrite <- (view_check_step A (w_cat w1) s Ht Hi1 Hr1 Hres Hun), <- (view_check_step A (w_cat w2) s Ht Hi2 Hr2 Hres2 Hun).
    rewrite Hview. reflexivity. }
  destruct H1 as [w1 s c1' es1 w1' Hc1 Hrun1 | w1 s r1 Hc1];
    destruct H2 as [w2 s c2' es2 w2' Hc2 Hrun2 | w2 s r2 Hc2]; rewrite ?Hc1, ?Hc2 in Hdec; cbn [view_decision] in Hdec;
    try discriminate Hdec.
  - (* both accept: the runs on the views start from one catalog *)
    rewrite Ht in Hrun1, Hrun2.
    destruct (run_actions_view base kernels A chs w1 (s_actions s) w1' Hs1 Hr1 Hres Hun Hrun1) as [O1 R1].
    destruct (run_actions_view base kernels A chs w2 (s_actions s) w2' Hs2 Hr2 Hres2 Hun Hrun2) as [O2 R2].
    rewrite <- Hview in O2.
    assert (S1 : sound base kernels {| w_cat := view A (w_cat w1); w_mem := w_mem w1 |}).
    { apply sound_view; [destruct w1; exact Hs1 | exact Hr1 | exact Hn1]. }
    assert (S2 : sound base kernels {| w_cat := view A (w_cat w1); w_mem := w_mem w2 |}).
    { rewrite Hview. apply sound_view; [destruct w2; exact Hs2 | exact Hr2 | exact Hn2]. }
    assert (G : agree_on base (view A (w_cat w1)) A (w_mem w1) (w_mem w2))
      by (apply (agree_view base A (w_cat w1) _ _ Hi1); exact Hag).
    destruct (actions_isolation base kernels A A chs chs _ _ (s_actions s) _ _ S1 S2 eq_refl G
                (fun _ => eq_refl) (fun H => False_ind _ (H eq_refl)) O1 O2)
      as (_ & _ & Hc' & Hag'). cbn [w_cat w_mem] in Hc', Hag'.
    destruct (ok_actions A chs w1 (s_actions s) w1' Hs1 Hr1 Hn1 Hrun1) as (Hs1' & Hr1' & Hn1').
    destruct (ok_actions A chs w2 (s_actions s) w2' Hs2 Hr2 Hn2 Hrun2) as (Hs2' & Hr2' & Hn2').
    pose proof Hs1' as [[Hi1' _] _].
    assert (Hrec' : received A (w_cat w1') = received A (w_cat w2')) by (rewrite R1, R2; exact Hrec).
    split; [exact Hs1' |]. split; [exact Hs2' |]. split; [exact Hr1' |]. split; [exact Hr2' |].
    split; [exact Hn1' |]. split; [exact Hn2' |].
    split; [destruct (find_reservation (w_cat w1) A) as [r |] eqn:Er; [| contradiction];
            rewrite (reservation_actions base kernels A A chs w1 (s_actions s) w1' r Er (fun H => False_ind _ (Hun H)) Hrun1);
            discriminate |].
    split; [exact (view_own A _ _ Hrec' Hc') |]. split; [exact Hrec' |].
    apply (agree_view base A (w_cat w1') _ _ Hi1'). exact Hag'.
  - (* both refuse *)
    split; [exact Hs1 |]. split; [exact Hs2 |]. split; [exact Hr1 |]. split; [exact Hr2 |].
    split; [exact Hn1 |]. split; [exact Hn2 |]. split; [exact Hres |].
    split; [exact Hown |]. split; [exact Hrec | exact Hag].
Qed.

(* The same event of A in both runs keeps them related. *)
Lemma event_sync : forall A w1 w2 e w1' w2', grant_related A w1 w2 ->
    (forall s chs, e = OStep s chs -> ~ In Unreserve (s_actions s)) ->
    move A w1 (Some e) w1' -> move A w2 (Some e) w2' -> grant_related A w1' w2'.
Proof.
  intros A w1 w2 e w1' w2' Hrel Hun M1 M2.
  destruct (move_cases A w1 _ w1' M1) as [[chs1 [s1 [E1 [Ht1 Hr1]]]] | [ch1 [t1 [a1 [E1 [Ht1 Ha1]]]]]];
    destruct (move_cases A w2 _ w2' M2) as [[chs2 [s2 [E2 [Ht2 Hr2]]]] | [ch2 [t2 [a2 [E2 [Ht2 Ha2]]]]]].
  - injection E1 as ->. injection E2 as -> ->.
    exact (self_sync A chs2 w1 w2 s2 w1' w2' Ht2 (Hun s2 chs2 eq_refl) Hrel Hr1 Hr2).
  - injection E1 as ->. exfalso. exact (action_event_step A w2 a2 s1 chs1 (eq_sym E2)).
  - injection E2 as ->. exfalso. exact (action_event_step A w1 a1 s2 chs2 (eq_sym E1)).
  - destruct e as [s chs | b data | id].
    + exfalso. exact (action_event_step A w1 a1 s chs (eq_sym E1)).
    + exact (grant_sync A ch1 ch2 t1 t2 w1 w2 a1 a2 w1' w2' b data Ht1 Ht2 Hrel (eq_sym E1) (eq_sym E2) Ha1 Ha2).
    + exact (revoke_sync A ch1 ch2 t1 t2 w1 w2 a1 a2 w1' w2' id Ht1 Ht2 Hrel (eq_sym E1) (eq_sym E2) Ha1 Ha2).
Qed.

Lemma moves_related : forall n A l1 l2 w1 w2 w1' w2', (List.length l1 + List.length l2 <= n)%nat ->
    grant_related A w1 w2 -> (forall s chs, In (Some (OStep s chs)) l1 -> ~ In Unreserve (s_actions s)) ->
    events l1 = events l2 -> moves A w1 l1 w1' -> moves A w2 l2 w2' -> grant_related A w1' w2'.
Proof.
  induction n as [| n IH]; intros A l1 l2 w1 w2 w1' w2' Hlen Hrel Hun Hev H1 H2.
  - destruct l1; [| cbn in Hlen; lia]. destruct l2; [| cbn in Hlen; lia].
    inversion H1; subst. inversion H2; subst. exact Hrel.
  - destruct H1 as [w1 | w1 o1 w1m r1 w1' M1 R1].
    + destruct H2 as [w2 | w2 o2 w2m r2 w2' M2 R2]; [exact Hrel |].
      destruct o2 as [e2 |]; [cbn in Hev; discriminate Hev |].
      destruct (move_cases A w2 None w2m M2) as [[chs [s [E _]]] | [ch [t [a [E [Ht Ha]]]]]]; [discriminate E |].
      apply (IH A [] r2 w1 w2m w1 w2'); [cbn in Hlen |- *; lia | | exact Hun | exact Hev | apply MovesEnd | exact R2].
      exact (silent2 A ch t w1 w2 a w2m Ht Hrel (eq_sym E) Ha).
    + destruct o1 as [e1 |].
      * destruct H2 as [w2 | w2 o2 w2m r2 w2' M2 R2]; [cbn in Hev; discriminate Hev |].
        destruct o2 as [e2 |].
        -- cbn in Hev. injection Hev as <- Hev.
           apply (IH A r1 r2 w1m w2m w1' w2'); [cbn in Hlen; lia | | | exact Hev | exact R1 | exact R2].
           ++ apply (event_sync A w1 w2 e1 w1m w2m Hrel); [| exact M1 | exact M2].
              intros s chs ->. apply (Hun s chs). left. reflexivity.
           ++ intros s chs Hin. apply (Hun s chs). right. exact Hin.
        -- destruct (move_cases A w2 None w2m M2) as [[chs [s [E _]]] | [ch [t [a [E [Ht Ha]]]]]]; [discriminate E |].
           apply (IH A (Some e1 :: r1) r2 w1 w2m w1' w2'); [cbn in Hlen |- *; lia | | exact Hun | exact Hev
                                                         | exact (MovesNext A w1 _ w1m r1 w1' M1 R1) | exact R2].
           exact (silent2 A ch t w1 w2 a w2m Ht Hrel (eq_sym E) Ha).
      * destruct (move_cases A w1 None w1m M1) as [[chs [s [E _]]] | [ch [t [a [E [Ht Ha]]]]]]; [discriminate E |].
        apply (IH A r1 l2 w1m w2 w1' w2'); [cbn in Hlen; lia | | | exact Hev | exact R1 | exact H2].
        -- exact (silent1 A ch t w1 a w1m w2 Ht Hrel (eq_sym E) Ha).
        -- intros s chs Hin. apply (Hun s chs). right. exact Hin.
Qed.

(* Tenant declassification: two runs with the same sequence of events of A,
   from worlds whose own parts of A are equal, whose buffers granted to A are
   equal, and whose memories agree on what A observes, end with equal own
   parts of A, equal buffers granted to A, and memories that agree on what A
   observes. The other tenants' steps, with their regions, sizes, written
   bytes, grids, and scalars, their reservations, their memory, and the
   hardware's choices for their launches may differ between the runs, and so
   may the order of their actions relative to A's events. A holds its
   reservation through the first run. *)
Theorem tenant_declassification : forall A steps1 steps2 chss1 chss2 w1 w2 w1' w2' os,
    sound base kernels w1 -> sound base kernels w2 -> res_inv (w_cat w1) -> res_inv (w_cat w2) ->
    NoDup (c_grants (w_cat w1)) -> NoDup (c_grants (w_cat w2)) -> find_reservation (w_cat w1) A <> None ->
    own A (w_cat w1) = own A (w_cat w2) -> received A (w_cat w1) = received A (w_cat w2) ->
    agree_on base (w_cat w1) A (w_mem w1) (w_mem w2) ->
    keeps_reservation A steps1 ->
    observe_steps A chss1 w1 steps1 w1' os -> observe_steps A chss2 w2 steps2 w2' os ->
    own A (w_cat w1') = own A (w_cat w2') /\ received A (w_cat w1') = received A (w_cat w2') /\
    agree_on base (w_cat w1') A (w_mem w1') (w_mem w2').
Proof.
  intros A steps1 steps2 chss1 chss2 w1 w2 w1' w2' os Hs1 Hs2 Hr1 Hr2 Hn1 Hn2 Hres Hown Hrec Hag Hkeep O1 O2.
  destruct (observe_steps_moves A chss1 w1 steps1 w1' os O1) as [l1 [M1 [E1 S1]]].
  destruct (observe_steps_moves A chss2 w2 steps2 w2' os O2) as [l2 [M2 [E2 _]]].
  destruct (moves_related (List.length l1 + List.length l2) A l1 l2 w1 w2 w1' w2' (le_n _)
              (conj Hs1 (conj Hs2 (conj Hr1 (conj Hr2 (conj Hn1 (conj Hn2 (conj Hres (conj Hown (conj Hrec Hag)))))))))
              (fun s chs Hin => let '(conj Hs Ht) := S1 s chs Hin in Hkeep s Hs Ht)
              (eq_trans E1 (eq_sym E2)) M1 M2)
    as (_ & _ & _ & _ & _ & _ & _ & Hown' & Hrec' & Hag').
  split; [exact Hown' |]. split; [exact Hrec' | exact Hag'].
Qed.

End Pipeline.
