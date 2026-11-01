(* Scalar certificate forms share one soundness boundary. The proposer selects
   its analysis from source structure and available template inputs before checking; a failed proposal
   cannot switch to a different analysis. Certificates remain untrusted data. *)
From Coq Require Import List Bool.
From OpGate Require Import GateIsa GateDomain GateThread GateInterval.
Import ListNotations.

Inductive scalar_certificate : Type :=
| ScalarBlocks (cert : block_certificate)
| ScalarCached (cert : cached_certificate)
| ScalarMissing.

Definition check_scalar_certificate (F : lfacts) (prog : program)
    (cert : scalar_certificate) : bool :=
 match cert with
 | ScalarBlocks block => check_block_certificate F prog block
 | ScalarCached cached => check_cached_certificate F prog cached
 | ScalarMissing => false
 end.

Definition propose_scalar_certificate (F : lfacts) (prog : program)
    : scalar_certificate :=
 match interval_prepare F prog with
 | Some prepared => match propose_prepared_interval F prog prepared with
   | Some cert => ScalarCached cert
   | None => ScalarMissing
   end
 | None => ScalarBlocks (propose_block_certificate F prog)
 end.

Definition scalar_certificate_present (cert : scalar_certificate) : bool :=
 match cert with
 | ScalarBlocks (CertSingle (Some _)) => true
 | ScalarBlocks (CertSplit _ _ (Some _) (Some _)) => true
 | ScalarCached _ => true
 | _ => false
 end.

Lemma missing_scalar_certificate : forall F prog cert,
 scalar_certificate_present cert = false ->
 check_scalar_certificate F prog cert = false.
Proof.
 intros F prog [block|cached|] H; [|discriminate|reflexivity].
 destruct block as [cases|r cut low high].
 - destruct cases; [discriminate|].
   unfold check_scalar_certificate,check_block_certificate,check_case_certificate.
   apply andb_false_r.
 - destruct low,high; try discriminate;
   unfold check_scalar_certificate,check_block_certificate,check_case_certificate;
   rewrite ?andb_false_l,?andb_false_r; reflexivity.
Qed.

Theorem check_scalar_certificate_sound : forall F prog cert,
 check_scalar_certificate F prog cert = true -> kernel_safe F prog.
Proof.
 intros F prog [block|cached|] H; [| |discriminate].
 - exact (check_block_certificate_sound F prog block H).
 - exact (check_cached_certificate_safe F prog cached H).
Qed.

Lemma check_scalar_certificate_lf_ok : forall F prog cert,
 check_scalar_certificate F prog cert = true -> lf_ok F = true.
Proof.
 intros F prog [block|cached|] H; [| |discriminate].
 - exact (check_block_certificate_lf_ok F prog block H).
 - exact (check_cached_certificate_lf_ok F prog cached H).
Qed.

Definition check_scalar (F : lfacts) (prog : program) : bool :=
 check_scalar_certificate F prog (propose_scalar_certificate F prog).
Theorem check_scalar_sound : forall F prog,
 check_scalar F prog = true -> kernel_safe F prog.
Proof. intros F prog H; exact (check_scalar_certificate_sound _ _ _ H). Qed.
