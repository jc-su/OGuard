(* GateParticipation.v: the check that the gate's query proves also proves
   the participation of every thread in the waiting sites of each entry
   kernel.

   The gate's query (opgate/gate) proves
   `check_text_certificate_ok text size mfs certificate = true`. That check
   reads `text_front front`, and every program and launch facts that `front`
   returns passed its participation check `barrier_boundary`
   (GateFront.v). `verify_certificate_sound` (GateParse.v) states the
   safety, the termination, the link and the prologue of each entry kernel;
   `verify_certificate_participation` below states the same facts together
   with `participation`, the fact that `barrier_boundary` proves on the
   route that it takes for the kernel:

   - the inlined kernel has zero waiting sites (CTA barriers, shfl, vote,
     match, and redux);
   - the inlined kernel is a straight-line body whose waiting sites are
     aligned `bar.sync` barriers without a guard (`barrier_linear_case`),
     a structural fact of the parsed text;
   - one waiting site: `single_phase_safe`, the run of every launch that
     the launch facts allow reaches the site (`phase_reaches`,
     GateBarrier.v);
   - two or more waiting sites: `front_events_safe`, the ordered-event
     check (GateFront.v), or `front_shared_safe`, the shared value check,
     which gives `same_waits` for every block of whole warps whose oracles
     satisfy `cta_family`, when every thread ends (GateShared.v). *)

From Coq Require Import ZArith.ZArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Strings.String.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateDomain.
Require Import OpGate.GateThread.
Require Import OpGate.GateBarrier.
Require Import OpGate.GateFront.
Require Import OpGate.GateTerm.
Require Import OpGate.GatePrologue.
Require Import OpGate.GateParse.

(* What the participation check of `front` proves of the program and launch
   facts that it gives for kernel K with binding Mf: the lowering of the
   inlined kernel K' with its places org gives the program and facts, and
   the waiting sites of K' satisfy the fact of the route that the check
   takes. *)
Definition participation (K : pkernel) (Mf : binding) (prog : program) (F : lfacts) : Prop :=
  exists K' org pp raw ts,
    inline K = Ok (K', org) /\
    ((p <- translate_placed K' Mf org ;; F0 <- facts K' Mf ;; ret (p, F0)) init) = Ok ((pp, raw), ts) /\
    prog = map (map fst) pp /\ F = payload_facts raw prog /\
    (source_events K' (expanded_places org) = [] \/
     barrier_linear_case K' = true \/
     (barrier_linear_case K' = false /\
      exists origin event, source_events K' (expanded_places org) = [origin] /\
        source_placed_event K' (expanded_places org) (ts_ids ts) pp origin = Some event /\
        single_phase_safe F prog event) \/
     (barrier_linear_case K' = false /\ (2 <= List.length (source_events K' (expanded_places org)))%nat /\
      exists events,
        placed_events K' (expanded_places org) (ts_ids ts) pp (source_events K' (expanded_places org)) = Some events /\
        (front_events_safe Mf K' (expanded_places org) (ts_ids ts) pp F events \/
         front_shared_safe Mf K' (expanded_places org) (ts_ids ts) pp F events))).

(* GateFront.v makes front opaque to conversion after it is defined. *)
Local Transparent front.
Local Opaque barrier_boundary inline translate_placed facts payload_facts.

Theorem front_participation : forall K Mf prog F, front K Mf = Ok (prog, F) -> participation K Mf prog F.
Proof.
  intros K Mf prog F Hfront. unfold front, front_core, front_lower in Hfront.
  destruct (zeroing_check K Mf); [| discriminate Hfront].
  destruct (inline K) as [[K' org] | e i] eqn:Hinline; [| discriminate Hfront].
  destruct (defined_check K' org); [| discriminate Hfront].
  destruct ((p <- translate_placed K' Mf org ;; F0 <- facts K' Mf ;; ret (p, F0)) init)
    as [[[pp raw] ts] | e i] eqn:Hlower; [| discriminate Hfront].
  destruct (zero_program_check K Mf (payload_facts raw (map (map fst) pp)) (map (map fst) pp));
    [| discriminate Hfront].
  set (F' := payload_facts raw (map (map fst) pp)) in *.
  destruct (barrier_boundary Mf K' (expanded_places org) (ts_ids ts) pp F') eqn:Hboundary; [| discriminate Hfront].
  injection Hfront as <- <-.
  exists K', org, pp, raw, ts. split; [exact Hinline |]. split; [exact Hlower |].
  split; [reflexivity |]. split; [reflexivity |].
  destruct (barrier_linear_case K') eqn:Hlinear; [right; left; reflexivity |].
  destruct (source_events K' (expanded_places org)) as [| origin [| second rest]] eqn:Hsource.
  - left. reflexivity.
  - right. right. left. split; [reflexivity |].
    destruct (barrier_boundary_phase_sound Mf K' (expanded_places org) (ts_ids ts) pp F' origin Hsource Hlinear
                Hboundary) as [event [He Hsafe]].
    exists origin, event. split; [reflexivity |]. split; [exact He | exact Hsafe].
  - right. right. right. split; [reflexivity |]. split; [cbn [List.length]; apply le_n_S, le_n_S, Nat.le_0_l |].
    destruct (barrier_boundary_events_sound Mf K' (expanded_places org) (ts_ids ts) pp F'
                (origin :: second :: rest) Hsource ltac:(cbn [List.length]; apply le_n_S, le_n_S, Nat.le_0_l)
                Hlinear Hboundary) as [events [He Hsafe]].
    exists events. split; [exact He | exact Hsafe].
Qed.

Local Transparent barrier_boundary inline translate_placed facts payload_facts.

(* The check that the gate's query proves gives, for every entry kernel of
   the text, what `verify_certificate_sound` gives and the participation of
   its waiting sites. *)
Theorem verify_certificate_participation : forall a n mfs certs,
  check_text_certificate_ok a n mfs certs = true ->
  exists ks, parse_text a n = Ok ks /\
    forall name K, In (name, K) (ks_entries ks) ->
    exists mf cert, lookup mfs name = Some mf /\
      exists prog F, front K mf = Ok (prog, F) /\
        check_scalar_certificate F prog cert = true /\ kernel_safe F prog /\ kernel_ends F prog /\
        link_ok F mf = true /\ prologue_ok F mf prog = true /\ participation K mf prog F.
Proof.
  intros a n mfs certs H. destruct (verify_certificate_sound a n mfs certs H) as [ks [Hparse Hall]].
  exists ks. split; [exact Hparse |]. intros name K Hin.
  destruct (Hall name K Hin) as [mf [cert [Hmf [prog [F [Hfront [Hc [Hsafe [Hends [Hlink Hpro]]]]]]]]]].
  exists mf, cert. split; [exact Hmf |]. exists prog, F.
  split; [exact Hfront |]. split; [exact Hc |]. split; [exact Hsafe |]. split; [exact Hends |].
  split; [exact Hlink |]. split; [exact Hpro |]. exact (front_participation K mf prog F Hfront).
Qed.
