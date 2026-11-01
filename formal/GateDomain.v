(*
  The address checker's abstract domain and transfer functions.

  For each register the checker keeps one fact:
    FForm f al vr     the value satisfies form f, is a multiple of 2^al, and
                      lies in vr when vr is given
    FAlign k          the value is a multiple of 2^k
    FPred cs exact    a predicate: if it holds, every comparison in cs holds;
                      if exact, also conversely
    FNeg cs exact     a predicate: if it fails, every comparison in cs holds;
                      if exact, also conversely, as for a disjunction
  and a list of linear constraints over atoms. A register with no fact holds
  anything, except that a register no instruction writes keeps its launch
  value, which the launch facts may bound. Size symbols are such registers:
  extents and index ranges are forms over them, and the constraints relate
  the registers of a kernel to them.

  This file defines the functions. GateMeaning.v gives the facts their meaning,
  the files GateSound*.v prove the functions sound, and GateThread.v
  proves the top theorem.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import PArith.PArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
From Coq Require Import FSets.FMapPositive.
From Coq Require Strings.String.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.

Local Open Scope Z_scope.

Module PM := PositiveMap.

Definition mark_regs (rs : list reg) (m : PM.t unit) : PM.t unit :=
  fold_left (fun m r => PM.add r tt m) rs m.

Fixpoint register_marks_eqb (x y : PM.t unit) : bool :=
 match x,y with
 | PM.Leaf _, PM.Leaf _ => true
 | PM.Node l o r, PM.Node l' o' r' =>
   register_marks_eqb l l' &&
   (match o,o' with None,None | Some tt,Some tt => true | _,_ => false end) &&
   register_marks_eqb r r'
 | _,_ => false
 end.
Lemma register_marks_eqb_eq : forall x y,
 register_marks_eqb x y = true -> x = y.
Proof.
 intros x. induction x as [|l IHl o r IHr]; intros y;
 destruct y as [|l' o' r']; cbn [register_marks_eqb]; try discriminate; intros H.
 - reflexivity.
 - rewrite !andb_true_iff in H. destruct H as [[Hl Ho] Hr].
   apply IHl in Hl. apply IHr in Hr. subst l' r'.
   destruct o as [[]|], o' as [[]|]; try discriminate; reflexivity.
Qed.

Fixpoint register_reach (fuel : nat) (graph : PM.t (list reg))
    (pending : list reg) (seen : PM.t unit) : PM.t unit :=
  match fuel, pending with
  | S fuel', r :: rest =>
      if PM.mem r seen then register_reach fuel' graph rest seen else
      let deps := match PM.find r graph with Some rs => rs | None => [] end in
      register_reach fuel' graph (List.app deps rest) (PM.add r tt seen)
  | _, _ => seen
  end.

Definition register_closure (graph : PM.t (list reg)) : list reg -> PM.t unit :=
  (* Each seed is popped once. Each graph node expands at most once, so
     the total number of further work items is bounded by all graph edges. *)
  let edges := PM.fold (fun _ rs n => (List.length rs + n)%nat) graph 0%nat in
  fun seeds => register_reach (List.length seeds + edges)%nat graph seeds (PM.empty unit).

(* Explicit branches short-circuit under Rocq's strict VM evaluator. Keep
   the predicate outside the recursive fixpoint, as in List's searches:
   these definitions are convertible to the standard boolean functions. *)
Section BooleanSearch.
Variable A : Type.
Variable predicate : A -> bool.

Fixpoint short_all (xs : list A) : bool :=
  match xs with [] => true | x :: rest => if predicate x then short_all rest else false end.

Fixpoint short_any (xs : list A) : bool :=
  match xs with [] => false | x :: rest => if predicate x then true else short_any rest end.
End BooleanSearch.

Arguments short_all {A} predicate xs.
Arguments short_any {A} predicate xs.

Lemma short_all_same : forall A (p : A -> bool) xs, short_all p xs = forallb p xs.
Proof. reflexivity. Qed.

Lemma short_any_same : forall A (p : A -> bool) xs, short_any p xs = existsb p xs.
Proof. reflexivity. Qed.

Definition range := (Z * Z)%type.

(* ------------------------------------------------------------------ *)
(* Facts and states.                                                   *)

(* a cmp b, both read at width w, as signed integers if sg. *)
Record cmpf := { cc : cmp; cw : nat; csg : bool; ca : reg; cb : opnd }.

Inductive fact :=
| FForm (f : form) (al : Z) (vr : option range)
| FAlign (k : Z)
| FPred (cs : list cmpf) (exact : bool)
| FNeg (cs : list cmpf) (exact : bool).

(* lo <= sum of the terms <= hi. *)
Definition constraint := (terms * Z * Z)%type.

Record astate := { amap : PM.t fact; acons : list constraint }.

Definition get (st : astate) (r : reg) : option fact := PM.find r (amap st).

(* Retain whole map entries and whole constraints. The selector is
   untrusted; no fact or predicate conjunction is rewritten. *)
Definition filter_fact_map (keep : PM.t unit) (m : PM.t fact) : PM.t fact :=
  PM.fold (fun r f acc => if PM.mem r keep then PM.add r f acc else acc)
    m (PM.empty fact).

Definition whole_state_filter (keep : PM.t unit)
    (keep_constraint : constraint -> bool) (st : astate) : astate :=
  {| amap := filter_fact_map keep (amap st);
     acons := filter keep_constraint (acons st) |}.

(* ------------------------------------------------------------------ *)
(* Launch facts.                                                       *)

(* The direction of a span: the kinds of access it allows. An atomic reads
   and writes, so only a read-write span allows it. *)
Inductive dir := DRead | DWrite | DReadWrite.

Definition allows (d : dir) (k : mkind) : bool :=
  match d, k with
  | DRead, KLoad | DWrite, KStore | DReadWrite, _ => true
  | _, _ => false
  end.

(* An index input or output: elements of elem bytes, read as signed or
   unsigned integers, each in [lo, hi]. The bounds are forms over size
   registers with one constant. *)
Record index := { ix_elem : Z; ix_signed : bool; ix_lo : form; ix_hi : form }.

(* The region a global memory access may reach, by the span of its base
   parameter: the ext bytes from the parameter's value, where ext is a
   form over size registers with one constant. A flag span holds words of
   4 bytes that a launch may only set to 1, as the status buffer of the
   guard pass's checks: the host zeroes it before the launch, so after it
   each word is 0 or 1. *)
Record span := { sp_dir : dir; sp_ext : form; sp_index : option index; sp_flag : bool }.

Record lfacts := {
  lf_widths : PM.t nat;                (* declared register widths *)
  lf_fixed : list reg;                 (* registers no instruction may write *)
  lf_bounds : list (reg * range);      (* ranges of fixed registers *)
  lf_aligns : list (reg * Z);          (* alignments of fixed registers *)
  lf_roles : list (reg * span);        (* each base parameter's span *)
  lf_shared_window : reg;              (* the fixed registers that hold the bases of the windows
                                          of shared and of local memory in the generic address
                                          space *)
  lf_local_window : reg;
  lf_shared : list (reg * form);          (* the register that holds each shared variable's shared
                                          address, and the variable's affine byte extent *)
  lf_locals : list (reg * (Z * frame));
                                       (* the register that holds each local variable's local
                                          address, the variable's bytes, and its frame *)
  lf_cells : list (reg * reg * Z);     (* the cells of local memory: each cell register, its
                                          variable's register, and its offset in the variable *)
  (* Optional precision hints; forgetting a pure result is sound regardless
     of how this map was selected. No access or instruction is omitted. *)
  lf_payloads : PM.t unit;
  lf_derived : list constraint;        (* checked launch relations between size symbols *)
  lf_names : list (String.string * reg)
                                       (* the name of each fixed register: "param:x",
                                          "size:k", a special register, a window, or a
                                          variable; what the launch puts in it *)
}.

Section Domain.

Variable F : lfacts.

Definition wdn (r : reg) : nat := match PM.find r (lf_widths F) with Some n => n | None => 64%nat end.
Definition wd (r : reg) : Z := Z.of_nat (wdn r).


Definition is_fixed (r : reg) : bool := existsb (Pos.eqb r) (lf_fixed F).

Definition opnd_width_ok (o : opnd) (w : Z) : bool :=
  match o with OReg r => Z.eqb (wd r) w | OImm _ => true end.

(* Both operands of comparison k have its width. *)
Definition cmp_wf (k : cmpf) : bool :=
  Z.eqb (wd (ca k)) (Z.of_nat (cw k)) && opnd_width_ok (cb k) (Z.of_nat (cw k)).

(* ------------------------------------------------------------------ *)
(* Ranges.                                                             *)

(* A form fact kept as its register's atom: a bare range, or a relation that
   also records the register's own value range. *)
Definition interval_only (x : option fact) : bool :=
  match x with
  | Some (FForm f _ vr) => (match ft f with [] => negb (Z.eqb (flo f) (fhi f)) | _ => false end)
                           || match vr with Some _ => true | None => false end
  | _ => false
  end.

(* The range of the sum of f's terms and its constant, given a range for
   each atom. *)
Fixpoint plain_sum (ar : atom -> option range) (ts : terms) (lo hi : Z) : option range :=
  match ts with
  | [] => Some (lo, hi)
  | (x, c) :: ts' =>
      match ar x with
      | None => None
      | Some (a, b) => plain_sum ar ts' (lo + Z.min (c * a) (c * b)) (hi + Z.max (c * a) (c * b))
      end
  end.

Definition inter (x y : option range) : option range :=
  match x, y with
  | Some (a, b), Some (c, d) => Some (Z.max a c, Z.min b d)
  | Some r, None | None, Some r => Some r
  | None, None => None
  end.

(* Find a whole-register value and related bit atoms. A right shift can
   use the quotient/remainder identity even without exact cancellation:
   x - (x >> 1) becomes (x >> 1) + (x mod 2). *)
Definition low_bits (ts : terms) : option (reg * ext * ext * Z * Z) :=
  let whole := fun r c0 => find (fun u => match u with
                                          | ((r', EX), c) | ((r', EZ), c) => Pos.eqb r r' && Z.eqb c c0
                                          | _ => false
                                          end) ts in
  fold_left (fun acc t =>
    match acc, t with
    | Some _, _ => acc
    | None, ((r, EBf a n), c') =>
        if Z.eqb a 0 then
          match whole r (- c') with Some ((_, e), c) => Some (r, e, EShr n, n, c) | _ => None end
        else if Z.eqb (Z.rem c' (2 ^ a)) 0 then
          match whole r (- (c' / 2 ^ a)) with Some ((_, e), c) => Some (r, e, EBf a n, a, c) | _ => None end
        else None
    | None, ((r, EShr n), _) =>
        match find (fun u => match u with ((r',EX),_) | ((r',EZ),_) => Pos.eqb r r' | _ => false end) ts with
        | Some ((_,e),c) => Some (r,e,EShr n,n,c) | _ => None end
    | None, _ => None
    end) ts None.

(* The range of f from its atoms' ranges; and when f holds a register x
   less its low n bits, as a warp's start tid - (tid & 31), or less its
   bits from n, as Numba's tid % 32, tid - 32 (tid >> 5), also with x read
   as 2^n (x >> n) + x mod 2^n: the three terms added sum to 0, and the
   sum's terms then give the first 2^n (x >> n), which is not negative, and
   the second x mod 2^n, below 2^n. *)
Definition plain_range (ar : atom -> option range) (f : form) : option range :=
  let p := plain_sum ar (ft f) (flo f) (fhi f) in
  match low_bits (ft f) with
  | Some (r, e, hi, n, c) =>
      let whole := match e with EX | EZ => true | _ => false end in
      (* the atom of the high bits is x >> n *)
      let high := match hi with
                  | EShr k => Z.eqb k n
                  | EBf a m => Z.eqb a n && (0 <=? m) && (wd r <=? n + m)
                  | _ => false
                  end in
      if whole && high && (0 <=? n) then
        let ts := filter (fun t => negb (Z.eqb (snd t) 0))
                    (tsort (ft f ++ [((r, hi), c * 2 ^ n); ((r, e), - c); ((r, EBf 0 n), c)])) in
        inter p (plain_sum ar ts (flo f) (fhi f))
      else p
  | None => p
  end.

Fixpoint tget (ts : terms) (x : atom) : Z :=
  match ts with
  | [] => 0
  | (y, c) :: ts' => if atom_eqb x y then c else tget ts' x
  end.

Definition tmem (ts : terms) (x : atom) : bool := existsb (fun '(y, _) => atom_eqb x y) ts.

Fixpoint terms_eqb (a b : terms) : bool :=
  match a, b with
  | [], [] => true
  | (x, c) :: a', (y, d) :: b' => atom_eqb x y && Z.eqb c d && terms_eqb a' b'
  | _, _ => false
  end.

(* A constraint lo <= sum(ts) <= hi whose terms that occur in f occur there
   with q times their coefficient bounds that part of f, q times sum(ts)
   less its other terms, which their atoms bound; the rest of f is bounded
   by the atoms' ranges. *)
Fixpoint scaled (q : Z) (pick ts : terms) : bool :=
  match pick, ts with
  | [], [] => true
  | (x, c') :: pick', (y, c) :: ts' => atom_eqb x y && Z.eqb c' (q * c) && scaled q pick' ts'
  | _, _ => false
  end.

(* The unmatched case needs only the picked terms. Construct residual terms
   only after the shared terms and coefficient ratio pass their checks. *)
Definition cons_range (ar : atom -> option range) (f : form) (k : constraint) : option range :=
  let '(ts, clo, chi) := k in
  let pick := filter (fun t => tmem ts (fst t)) (ft f) in
  match pick with
  | [] => None
  | (_, c1) :: _ =>
      let core := filter (fun t => tmem (ft f) (fst t)) ts in
      match core with
      | [] => None
      | (_, c0) :: _ =>
          if Z.eqb c0 0 || negb (Z.eqb (Z.rem c1 c0) 0) then None else
          let q := Z.quot c1 c0 in
          if negb (scaled q pick core) then None else
          let rest := filter (fun t => negb (tmem ts (fst t))) (ft f) in
          let extra := filter (fun t => negb (tmem (ft f) (fst t))) ts in
          match plain_sum ar rest (flo f) (fhi f), plain_sum ar extra 0 0 with
          | Some (a, b), Some (u, v) =>
              Some (a + Z.min (q * (clo - v)) (q * (chi - u)), b + Z.max (q * (clo - v)) (q * (chi - u)))
          | _, _ => None
          end
      end
  end.


(* Cancel a shared coefficient after scaling f by the positive magnitude
   of the constraint's coefficient. Integer division of the resulting bounds
   handles relations such as 2*x <= 2*n - 1 without losing x < n. *)
Definition projection_at (ar : atom -> option range) (f : form) (k : constraint) (c1 c0 : Z) : option range :=
 let '(ts,clo,chi) := k in
      if Z.eqb c0 0 then None else
      let n := Z.abs c0 in
      let q := c1 * Z.sgn c0 in
      let rest := filter (fun t => negb (Z.eqb (snd t) 0)) (tsort (tscale n (ft f) ++ tscale (-q) ts)) in
      match plain_range ar {|fm:=64;ft:=rest;flo:=n*flo f;fhi:=n*fhi f|} with
      | Some (a, b) => Some (- ((- (a + Z.min (clo * q) (chi * q))) / n), (b + Z.max (clo * q) (chi * q)) / n)
      | None => None
      end.

(* Try every distinct shared coefficient ratio. Choosing only the first
   shared atom makes precision depend on register order; another ratio may
   cancel more terms and leave a residual with a useful range. *)
Definition projected_cons_range (ar : atom -> option range) (f : form) (k : constraint) : option range :=
 let '(ts,_,_) := k in
 let pivots := fold_left (fun acc t => let '(x,c1) := t in let c0 := tget ts x in
   if Z.eqb c0 0 || existsb (fun p => Z.eqb (fst p * c0) (c1 * snd p)) acc then acc else (c1,c0)::acc) (ft f) [] in
 fold_left (fun acc p => inter acc (projection_at ar f k (fst p) (snd p))) pivots None.

(* The registers that a constraint's terms read, each once. *)
Definition cons_regs (k : constraint) : list reg :=
  let '(ts, _, _) := k in
  fold_left (fun rs t => let r := fst (fst t) in if existsb (Pos.eqb r) rs then rs else r :: rs) ts [].
Definition cons_bucket (ix : PM.t (list constraint)) (r : reg) : list constraint :=
  match PM.find r ix with Some ks => ks | None => [] end.
(* The constraints of cs on each register, in their order. A constraint
   without an atom of a form gives the form no range, so the constraints on
   the form's registers give it the range that all of cs gives. The index
   is built once for each range evaluator, so that each form reads only the
   constraints on its own registers. *)
Definition cons_index (cs : list constraint) : PM.t (list constraint) :=
  fold_right (fun k ix => fold_left (fun ix r => PM.add r (k :: cons_bucket ix r) ix) (cons_regs k) ix)
    (PM.empty (list constraint)) cs.
(* The constraints on the registers of ts; one on two of them twice, which
   an intersection of ranges reads once. *)
Definition cons_near (ix : PM.t (list constraint)) (ts : terms) : list constraint :=
  flat_map (fun t => cons_bucket ix (fst (fst t))) ts.

(* Memoize atom ranges within one form evaluation. A recursive atom range
   is computed once even when several constraints use that atom. *)
Definition cached_range (ar : atom -> option range) (xs : list atom) : atom -> option range :=
  let cached := map (fun x => (x, ar x)) xs in
  fun x => match find (fun p => atom_eqb x (fst p)) cached with
           | Some (_, v) => v | None => ar x end.

Definition dependencies_in (cs : list constraint) (ts : terms) : list atom :=
 let relevant := filter (fun k => let '(ks,_,_) := k in
    existsb (fun t => tmem ks (fst t)) ts) cs in
 let xs := map fst ts ++ flat_map (fun k => let '(ks,_,_) := k in map fst ks) relevant in
 fold_right (fun x acc => if existsb (atom_eqb x) acc then acc else x::acc) [] xs.
(* The range of f through the constraints cs. *)
Definition form_range_in (cs : list constraint) (ar : atom -> option range) (f : form) : option range :=
  let unique := dependencies_in cs (ft f) in
  let ar := cached_range ar unique in
  fold_left (fun acc k => inter acc (cons_range ar f k)) cs (plain_range ar f).
Definition form_range_with (st : astate) (ar : atom -> option range) (f : form) : option range :=
  form_range_in (acons st) ar f.
(* The same range through the constraints on f's registers alone. *)
Definition form_range_ix (ix : PM.t (list constraint)) (ar : atom -> option range) (f : form) : option range :=
  form_range_in (cons_near ix (ft f)) ar f.

Definition sext_ok (w : Z) (r : range) : bool :=
  (- 2 ^ (w - 1) <=? fst r) && (snd r <? 2 ^ (w - 1)).

(* The residues modulo 2^w of the values in r, when r lies within one
   multiple of 2^w: a form keeps its constant unreduced, so 2048 - 1 at
   64 bits is 2^64 + 2047. *)
Definition reduce (w : Z) (r : range) : option range :=
  let '(x, y) := r in
  if Z.eqb (x / 2 ^ w) (y / 2 ^ w) then Some (x mod 2 ^ w, y mod 2 ^ w) else None.

Record rfuns := { f_rr : reg -> option range; f_sr : reg -> option range }.

Definition shr_range (k : Z) (r : option range) : range :=
  match r with Some (a, b) => (a / 2 ^ k, b / 2 ^ k) | None => (0, (2 ^ 64 - 1) / 2 ^ k) end.

Definition atom_range_with (fs : rfuns) (x : atom) : option range :=
  let (r, e) := x in
  match e with
  | EBf a n =>
      if (a <? 0) || (n <? 0) then None else
      match f_rr fs r with
      | Some (lo, hi) => if (0 <=? lo) && (hi / 2 ^ a <? 2 ^ n) then Some (lo / 2 ^ a, hi / 2 ^ a)
                         (* the bits of a constant *)
                         else if Z.eqb lo hi then Some ((lo / 2 ^ a) mod 2 ^ n, (lo / 2 ^ a) mod 2 ^ n)
                         else Some (0, 2 ^ n - 1)
      | None => Some (0, 2 ^ n - 1)
      end
  | EShr k => if k <? 0 then None else Some (shr_range k (f_rr fs r))
  (* a sign extension lies in [-2^(w-1), 2^(w-1) - 1], as for a 32-bit sum
     whose own range is unknown *)
  | ES => Some (match f_sr fs r with
                | Some rg => rg
                | None => (- 2 ^ (wd r - 1), 2 ^ (wd r - 1) - 1)
                end)
  (* a narrow register's value lies in [0, 2^w - 1] *)
  | EZ => Some (match f_rr fs r with
                | Some (lo, hi) => (Z.max lo 0, Z.min hi (2 ^ wd r - 1))
                | None => (0, 2 ^ wd r - 1)
                end)
  | EX => f_rr fs r
  end.

(* The range a register's own fact gives, without a form's range. *)
Definition fact_range (r : reg) (x : option fact) : option range :=
  match x with
  | None => if is_fixed r then lookup (lf_bounds F) r else None
  | Some (FForm _ _ (Some vr)) => Some vr
  | Some (FForm f _ None) =>
      if Nat.eqb (List.length (ft f)) 0 && (wd r <=? fm f) && (0 <=? flo f) && (fhi f <? 2 ^ wd r)
      then Some (flo f, fhi f) else None
  | _ => None
  end.

(* A map of the pairs' registers to their values, read by a lookup rather
   than a scan of the list. *)
Definition pair_table {A : Type} (pairs : list (reg * A)) : PM.t A :=
 fold_left (fun m p => PM.add (fst p) (snd p) m) pairs (PM.empty A).

Lemma pair_table_find : forall {A : Type} (v : reg -> A) regs r x,
 PM.find r (pair_table (map (fun u => (u, v u)) regs)) = Some x -> x = v r.
Proof.
 intros A v regs r x. unfold pair_table.
 assert (Hinv : forall m, (forall u y, PM.find u m = Some y -> y = v u) ->
   forall u y, PM.find u (fold_left (fun m p => PM.add (fst p) (snd p) m)
                            (map (fun u => (u, v u)) regs) m) = Some y -> y = v u).
 { induction regs as [|r0 rest IH]; intros m Hm; cbn [map fold_left fst snd]; [exact Hm|].
   apply IH. intros u y H. destruct (Pos.eq_dec u r0) as [->|Hne].
   - rewrite PM.gss in H. injection H as <-. reflexivity.
   - rewrite PM.gso in H by exact Hne. exact (Hm u y H). }
 apply Hinv. intros u y H. rewrite PM.gempty in H. discriminate.
Qed.

Definition cache_rf (fs : rfuns) (regs : list reg) : rfuns :=
 let cached := pair_table (map (fun r => (r,(f_rr fs r,f_sr fs r))) regs) in
 {| f_rr := fun r => match PM.find r cached with Some (rr,_) => rr | None => f_rr fs r end;
    f_sr := fun r => match PM.find r cached with Some (_,sr) => sr | None => f_sr fs r end |}.

(* One successor layer defines both uncached and cached range semantics. *)
Definition range_base (st:astate) : rfuns :=
{| f_rr := fun r => fact_range r (get st r);
            f_sr := fun r => match fact_range r (get st r) with
                             | Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd r - 1)) then Some (lo, hi) else None
                             | None => None
                             end |}.
Definition range_layer (ix:PM.t (list constraint)) (st:astate) (prev:rfuns) : rfuns :=
let fr := form_range_ix ix (atom_range_with prev) in
      let rr := fun r =>
        match get st r with
        | Some (FForm f _ None) =>
            if fm f >=? wd r then
              match fr f with Some g => reduce (wd r) g | None => None end
            else None
        | x => fact_range r x
        end in
      let sr := fun r =>
        match get st r with
        | Some (FForm f _ _) as x =>
            if fm f >=? wd r then
              match fr f with
              | Some g => if sext_ok (wd r) g then Some g else
                          match rr r with Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd r - 1)) then Some (lo, hi) else None | None => None end
              | None => match rr r with Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd r - 1)) then Some (lo, hi) else None | None => None end
              end
            else match rr r with Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd r - 1)) then Some (lo, hi) else None | None => None end
        | _ => match rr r with Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd r - 1)) then Some (lo, hi) else None | None => None end
        end in
      {| f_rr := rr; f_sr := sr |}.
Definition range_cache_regs (st:astate) :=
 flat_map (fun p=>match snd p with
 | FForm f _ _=>if Nat.eqb(List.length(ft f))0 then [] else[fst p]
 | _=>[] end)(PM.elements(amap st)).

(* Memoized layers compute a form range once for both register readings. *)
Definition range_unsigned_signed (r:reg) (rr:option range) : option range :=
 match rr with Some(lo,hi)=>if (0<=?lo)&&(hi<?2^(wd r-1)) then Some(lo,hi) else None|None=>None end.
Definition range_pair (ix:PM.t (list constraint)) (st:astate) (prev:rfuns) (r:reg) : option range * option range :=
 match get st r with
 | Some (FForm f al vr) as x =>
   if fm f >=? wd r then
     let g := form_range_ix ix (atom_range_with prev) f in
     let rr := match vr with Some v=>Some v | None=>match g with Some v=>reduce (wd r) v|None=>None end end in
     let sr := match g with Some v=>if sext_ok (wd r) v then Some v else range_unsigned_signed r rr|None=>range_unsigned_signed r rr end in
     (rr,sr)
   else let rr:=match vr with Some v=>Some v|None=>None end in (rr,range_unsigned_signed r rr)
 | x=>let rr:=fact_range r x in (rr,range_unsigned_signed r rr)
 end.
(* The cached registers regs with their ranges in the layer above prev. *)
Definition layer_pairs (regs:list reg) (ix:PM.t (list constraint)) (st:astate) (prev:rfuns)
  : list (reg * (option range * option range)) :=
 map (fun r=>(r,range_pair ix st prev r)) regs.
(* The layer above prev, with the cached registers' ranges read from pairs. *)
Definition cache_range_layer (ix:PM.t (list constraint)) (st:astate) (prev:rfuns)
  (pairs:list (reg * (option range * option range))) : rfuns :=
 let cached:=pair_table pairs in
 {|f_rr:=fun r=>match PM.find r cached with Some(rr,_)=>rr|None=>f_rr(range_layer ix st prev) r end;
 f_sr:=fun r=>match PM.find r cached with Some(_,sr)=>sr|None=>f_sr(range_layer ix st prev) r end|}.

Definition range_option_eqb (x y : option range) : bool :=
 match x, y with
 | Some (a, b), Some (c, d) => Z.eqb a c && Z.eqb b d
 | None, None => true
 | _, _ => false
 end.
Fixpoint range_pairs_eqb (a b : list (reg * (option range * option range))) : bool :=
 match a, b with
 | [], [] => true
 | (r, (x, y)) :: a', (r', (x', y')) :: b' =>
   Pos.eqb r r' && range_option_eqb x x' && range_option_eqb y y' && range_pairs_eqb a' b'
 | _, _ => false
 end.

Fixpoint rf_plain (ix:PM.t (list constraint)) (n:nat) (st:astate) : rfuns :=
 match n with O=>range_base st|S n=>range_layer ix st(rf_plain ix n st) end.
(* The cached layers above prev, whose cached registers regs have the
   ranges of pairs. A layer whose cached ranges equal those of the layer
   below is the same function as that layer on the cached registers and on
   the registers whose ranges read no register of the layer below. When
   the cached registers' ranges read only such registers, every later
   layer is then the same on them as well, so the iteration stops there. *)
Fixpoint rf_stable_layers (regs:list reg) (ix:PM.t (list constraint)) (st:astate) (n:nat) (prev:rfuns)
  (pairs:list (reg * (option range * option range))) : rfuns :=
 match n with
 | O=>prev
 | S n=>let next:=layer_pairs regs ix st prev in
   if range_pairs_eqb next pairs then prev
   else rf_stable_layers regs ix st n (cache_range_layer ix st prev next) next
 end.
Definition rf_cached_layers (regs:list reg) (ix:PM.t (list constraint)) (n:nat) (st:astate) : rfuns :=
 let base:=cache_rf(range_base st) regs in
 match n with
 | O=>base
 | S n=>let first:=layer_pairs regs ix st base in
   rf_stable_layers regs ix st n (cache_range_layer ix st base first) first
 end.
(* The range functions of depth n with the layers cached on regs. *)
Definition rf_on (regs:list reg) (ix:PM.t (list constraint)) (n:nat) (st:astate) : rfuns :=
 match n with O=>range_base st|S n=>range_layer ix st(rf_cached_layers regs ix n st) end.
Definition rf_with_cache (ix:PM.t (list constraint)) (memo:bool) (n:nat) (st:astate) : rfuns :=
 if memo then rf_on (range_cache_regs st) ix n st else rf_plain ix n st.

(* The range functions of depth n, with the state's constraints indexed once. *)
Definition rf (n:nat) (st:astate) : rfuns := rf_with_cache (cons_index (acons st)) false n st.

(* The registers of the terms ts. *)
Definition term_regs (ts : terms) : list reg := map (fun t => fst (fst t)) ts.
(* The registers whose ranges form_range_ix reads for a form with terms
   ts: those of the terms and of every constraint on them (cons_near). *)
Definition range_reads (ix : PM.t (list constraint)) (ts : terms) : list reg :=
 term_regs ts ++ flat_map (fun k => let '(ks, _, _) := k in term_regs ks) (cons_near ix ts).
(* A register r of a round of reach_graph: when r is not in nodes and a
   layer evaluates its form, r joins nodes and the registers that the
   form's range reads join the next round. *)
Definition reach_step (ix : PM.t (list constraint)) (st : astate) (acc : PM.t unit * list reg) (r : reg)
    : PM.t unit * list reg :=
 let '(nodes, next) := acc in
 if PM.mem r nodes then acc else
 match get st r with
 | Some (FForm f _ _) =>
   if (wd r <=? fm f) && negb (Nat.eqb (length (ft f)) 0) then
     (PM.add r tt nodes, range_reads ix (ft f) ++ next)
   else acc
 | _ => acc end.
(* nodes with the registers whose form a layer evaluates among pending
   and, round by round, among the registers that the added registers'
   layers read. A round that adds no register ends the search, and every
   other round adds a register with a fact, so fuel above the number of
   registers with a fact ends it. *)
Fixpoint reach_graph (fuel : nat) (ix : PM.t (list constraint)) (st : astate) (pending : list reg)
    (nodes : PM.t unit) : PM.t unit :=
 match fuel with
 | O => nodes
 | S fuel =>
   let '(nodes, next) := fold_left (reach_step ix st) pending (nodes, []) in
   match next with [] => nodes | _ => reach_graph fuel ix st next nodes end
 end.
(* The registers of regs whose form a layer evaluates, and with each of
   them every such register that its layer reads. A layer at one of these
   registers reads only these registers and registers whose ranges read no
   layer below. *)
Definition reach_regs (ix : PM.t (list constraint)) (st : astate) (regs : list reg) : list reg :=
 map fst (PM.elements (reach_graph (S (PM.cardinal (amap st))) ix st regs (PM.empty unit))).

Definition DEPTH : nat := 8.

(* A register's range from the constraints on its own atoms, reading the
   other atoms of each constraint through their facts alone: EX and EZ read
   the register's value, and so does ES when the range it gives lies at or
   above 0. A loop index bounded by a relation, such as j < end, gets its
   range this way. *)
Definition cons_reg_range (ks : list constraint) (st : astate) (r : reg) : option range :=
  let ar := atom_range_with (range_base st) in
  let via := fun e => fold_left (fun acc k => inter acc (cons_range ar {| fm := 64; ft := [((r, e), 1)]; flo := 0; fhi := 0 |} k))
                                ks None in
  let es := match via ES with Some (lo, hi) => if 0 <=? lo then Some (lo, hi) else None | None => None end in
  inter (inter (via EX) (via EZ)) es.

(* The ranges fs gives, tightened by the constraints on each register. *)
Definition tighten (ix : PM.t (list constraint)) (st : astate) (fs : rfuns) : rfuns :=
  let rr := fun r => inter (f_rr fs r) (cons_reg_range (cons_bucket ix r) st r) in
  {| f_rr := rr;
     f_sr := fun r => inter (f_sr fs r)
                        (match rr r with
                         | Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd r - 1)) then Some (lo, hi) else None
                         | None => None
                         end) |}.

(* The ranges of depth DEPTH with the layers cached on reach_regs of
   regs. At regs they are the ranges of the uncached layers (rft_on_exact
   in GateSoundRange.v), so a query that reads the ranges of regs alone
   gets the uncached result. *)
Definition rft_on (ix : PM.t (list constraint)) (st : astate) (regs : list reg) : rfuns :=
  tighten ix st (rf_on (reach_regs ix st regs) ix DEPTH st).

Definition reg_range (st : astate) (r : reg) : option range := f_rr (rft_on (cons_index (acons st)) st [r]) r.
Definition sreg_range (st : astate) (r : reg) : option range := f_sr (rft_on (cons_index (acons st)) st [r]) r.
Definition atom_range (st : astate) (x : atom) : option range :=
  atom_range_with (rft_on (cons_index (acons st)) st [fst x]) x.
Definition form_range (st : astate) (f : form) : option range :=
  let ix := cons_index (acons st) in form_range_ix ix (atom_range_with (rft_on ix st (range_reads ix (ft f)))) f.

(* The range of f at depth 1: from each register's own fact or form, and
   the constraints. Enough for a constant and for a fixed register, and
   cheap enough for a join to ask about every constraint. *)
Definition form_range_near (st : astate) (f : form) : option range :=
  let ix := cons_index (acons st) in form_range_ix ix (atom_range_with (rf_with_cache ix false 1 st)) f.

(* The range every value of a register has: below 2^w, and its sign
   extension in [-2^(w-1), 2^(w-1)). *)
Definition natural_range (x : atom) : option range :=
  let (r, e) := x in
  match e with
  | EX | EZ => Some (0, 2 ^ wd r - 1)
  | ES => Some (- 2 ^ (wd r - 1), 2 ^ (wd r - 1) - 1)
  | _ => None
  end.

(* A form's range with each atom of unknown range read in its natural
   range. form_range_with reads the constraints that share an atom with f,
   which are among the constraints on f's registers. *)
Definition form_range_nat (st : astate) (f : form) : option range :=
 let ix := cons_index (acons st) in
 let ar := atom_range_with (rft_on ix st (range_reads ix (ft f))) in
 form_range_with st (fun x => match ar x with Some g => Some g | None => natural_range x end) f.

(* ------------------------------------------------------------------ *)
(* Alignment.                                                          *)

Definition const_align (f : form) : Z :=
  if Z.eqb (flo f) (fhi f) then (if Z.eqb (flo f) 0 then fm f else Z.min (fm f) (v2 (flo f))) else 0.

Definition form_align_with (ra : reg -> Z) (f : form) : Z :=
  fold_left (fun k '(x, c) =>
               let ka := match snd x with EX | EZ | ES => ra (fst x) | _ => 0 end in
               Z.min k (v2 c + ka))
            (ft f) (const_align f).

Fixpoint ra (n : nat) (st : astate) (r : reg) : Z :=
  match get st r with
  | None => if is_fixed r then match lookup (lf_aligns F) r with Some k => k | None => 0 end else 0
  | Some (FAlign k) => k
  | Some (FForm f al _) =>
      match n with
      | O => Z.max 0 al
      | S n' => if (fm f >=? wd r) || (match ft f with [] => true | _ => false end)
                then Z.max al (Z.min (form_align_with (ra n' st) f) (fm f))
                else Z.max 0 al
      end
  | _ => 0
  end.

Definition reg_align (st : astate) (r : reg) : Z := ra DEPTH st r.
Definition form_align (st : astate) (f : form) : Z := form_align_with (ra DEPTH st) f.

(* The alignment of a register's value, given its form and value facts. *)
Definition value_align (st : astate) (f : form) (al : Z) : Z := Z.max al (Z.min (form_align st f) (fm f)).

(* Round a range inward to multiples of 2^k. *)
Definition round (k : Z) (r : range) : range :=
  let s := 2 ^ Z.max 0 k in (- ((- fst r) / s) * s, (snd r / s) * s).

Definition bitlen (z : Z) : Z := if Z.leb z 0 then 0 else Z.log2 z + 1.

Definition is_some {A} (o : option A) : bool := match o with Some _ => true | None => false end.

Definition null {A} (l : list A) : bool := match l with [] => true | _ => false end.

(* ------------------------------------------------------------------ *)
(* A register's value as a form.                                       *)

Definition atom64 (r : reg) (e : ext) : form := fatom 64 (r, e) 1.

(* Modulo 2^32, the zero and sign extensions of a 32-bit register agree:
   read each through the one whose range is known. *)
Definition swap_ext (st : astate) (ts : terms) : terms :=
  map (fun t =>
         let '((r, e), c) := t in
         match e with
         | EZ | ES =>
             if Z.eqb (wd r) 32 && negb (is_some (atom_range st (r, e))) then
               let o := match e with EZ => ES | _ => EZ end in
               if is_some (atom_range st (r, o)) then ((r, o), c) else t
             else t
         | _ => t
         end) ts.

(* A form modulo 2^32 of a 32-bit value, as a form modulo 2^64 of the
   value's zero (EZ) or sign (ES) extension. *)
(* Query only the bounds the operation needs before expanding recursive
   range dependencies. The returned interval must prove those bounds. *)
Definition bounded_form_range (st : astate) (f : form) (lo hi : Z) : option range :=
 match form_range_near st f with
 | Some (p,q) as found => if (lo <=? p) && (q <=? hi) then found else form_range st f
 | None => form_range st f
 end.

Definition lift (st : astate) (f : form) (e : ext) : option form :=
  if Z.ltb (fm f) 32 then None else
  let f := if Z.eqb (fm f) 32 then mk 32 (swap_ext st (ft f)) (flo f) (fhi f) else f in
  match bounded_form_range st f (match e with ES => -2^31 | _ => 0 end) (match e with ES => 2^31-1 | _ => 2^32-1 end) with
  | None => None
  | Some (lo, hi) =>
      let ok := match e with
                | EZ => (0 <=? lo) && (hi <? 2 ^ 32)
                | ES => (- 2 ^ 31 <=? lo) && (hi <? 2 ^ 31)
                | _ => false
                end in
      if ok then Some {| fm := 64; ft := ft f; flo := flo f; fhi := fhi f |} else None
  end.

(* The terms with each 32-bit register read through extension e, which
   modulo 2^32 is the register's value either way. *)
Definition swap_to (e : ext) (ts : terms) : terms :=
  map (fun t => let '((r, e'), c) := t in
                match e' with EZ | ES => if Z.eqb (wd r) 32 then ((r, e), c) else t | _ => t end) ts.

(* Register r's extension e, where f is r's value modulo 2^32, when the
   form's range and r's own range, which a guard's constraint may give,
   leave only one value congruent to it modulo 2^32. *)
Definition lift_by_range (st : astate) (r : reg) (f : form) (e : ext) : option form :=
  match e with
  | EZ | ES =>
      if negb (Z.eqb (wd r) 32) || negb (Z.eqb (fm f) 32) then None else
      let f' := mk 32 (swap_to e (ft f)) (flo f) (fhi f) in
      let ix := cons_index (acons st) in
      let ar := atom_range_with (rft_on ix st (range_reads ix (ft f' ++ [((r,e),1)]))) in
      let n := match e with ES => (- 2 ^ 31, 2 ^ 31 - 1) | _ => (0, 2 ^ 32 - 1) end in
      let '(a, b) := match form_range_with st ar {| fm := 64; ft := [((r, e), 1)]; flo := 0; fhi := 0 |} with
                     | Some (a, b) => (Z.max a (fst n), Z.min b (snd n))
                     | None => n
                     end in
      let '(a,b) := round (Z.min 32 (reg_align st r)) (a,b) in
      match form_range_with st (fun x => match ar x with Some g => Some g | None => natural_range x end) f' with
      | Some (p, q) =>
          if (q - a <? 2 ^ 32) && (b - p <? 2 ^ 32)
          then Some {| fm := 64; ft := ft f'; flo := flo f'; fhi := fhi f' |} else None
      | None => None
      end
  | _ => None
  end.

Definition lift_reg (st : astate) (r : reg) (f : form) (e : ext) : option form :=
  match lift st f e with Some g => Some g | None => lift_by_range st r f e end.

Definition as_form (st : astate) (r : reg) (e : ext) : form :=
  let x := get st r in
  if interval_only x then atom64 r e else
  match x with
  | Some (FForm f _ _) =>
      match e with
      | EX => if Z.eqb (fm f) 64 && Z.eqb (wd r) 64 then f else atom64 r e
      | EZ | ES =>
          if Z.eqb (wd r) 32 then match lift_reg st r (fat f 32) e with Some g => g | None => atom64 r e end
          else atom64 r e
      | _ => atom64 r e
      end
  | _ => atom64 r e
  end.

(* r's value modulo 2^w. A relation known only modulo a smaller power of two
   is kept as the register's atom unless relational is set. *)
Definition narrow (st : astate) (r : reg) (w : Z) (relational : bool) : form :=
  let x := get st r in
  match x with
  | Some (FForm f _ vr) =>
      if relational && negb (null (ft f)) then fat f w
      else if negb (interval_only x) then fat f w
      else if null (ft f) && negb (is_some vr) && Z.ltb (fm f) w then f
      else fatom w (r, EZ) 1
  | _ => fatom w (r, EZ) 1
  end.

Definition opnd_form (st : astate) (o : opnd) (w : Z) : form :=
  match o with
  | OImm z => fconst w (z mod 2 ^ w)
  | OReg r => if Z.eqb w 64 then as_form st r EX else narrow st r w false
  end.

(* The range of an operand read at width w, as a signed integer if sg. *)
Definition opnd_range (st : astate) (o : opnd) (w : Z) (sg : bool) : option range :=
  match o with
  | OImm z => let v := extend (Z.to_nat w) sg z in Some (v, v)
  | OReg r => if sg then sreg_range st r else reg_range st r
  end.

Definition const_of (st : astate) (o : opnd) (w : Z) (sg : bool) : option Z :=
  match opnd_range st o w sg with Some (a, b) => if Z.eqb a b then Some a else None | None => None end.

(* ------------------------------------------------------------------ *)
(* Normalization.                                                      *)

(* The register r holds a copy of, given r's fact: the fact says r = r2
   modulo at least r's width, and r2 is no wider than r. *)
Definition copy_of (r : reg) (x : option fact) : option reg :=
  match x with
  | Some (FForm f _ _) =>
      if (wd r <=? fm f) && Z.eqb (flo f) 0 && Z.eqb (fhi f) 0 then
        match ft f with
        | [((r2, EX), 1)] | [((r2, EZ), 1)] => if wd r2 <=? wd r then Some r2 else None
        | _ => None
        end
      else None
  | _ => None
  end.

Definition atom_align (st : astate) (x : atom) : Z :=
  match snd x with EX | EZ | ES => reg_align st (fst x) | _ => 0 end.

(* A register whose value is a multiple of 2^al: its value shifted right by
   j < k <= al is its value shifted right by k, times 2^(k - j). *)
Definition shr_of (e : ext) : option Z := match e with EX => Some 0 | EShr k => Some k | _ => None end.

Definition merge_reg (st : astate) (ts : terms) (r : reg) : terms :=
  let al := reg_align st r in
  let ks := flat_map (fun t => let '((r', e), _) := t in
                        if Pos.eqb r r' then match e with EShr k => if k <=? al then [k] else [] | _ => [] end
                        else []) ts in
  match ks with
  | [] => ts
  | k0 :: _ =>
      let k := fold_left Z.max ks k0 in
      map (fun t => let '((r', e), c) := t in
             if Pos.eqb r r' then
               match shr_of e with
               | Some j => if (0 <=? j) && (j <? k) then ((r, EShr k), c * 2 ^ (k - j)) else t
               | None => t
               end
             else t) ts
  end.

Definition shr_regs (ts : terms) : list reg :=
  flat_map (fun t => let '((r, e), _) := t in match e with EShr _ => [r] | _ => [] end) ts.

Definition merge_shifts (st : astate) (f : form) : form :=
  mk (fm f) (fold_left (merge_reg st) (shr_regs (ft f)) (ft f)) (flo f) (fhi f).

(* Substitute an atom by the form of its register. With ex set, only a form
   with a single constant is substituted, so the result allows exactly the
   values the input allows. *)
(* The value of atom (r, e) when r is fixed with one value at launch. *)
Definition fixed_atom (r : reg) (e : ext) : option Z :=
  if is_fixed r then
    match lookup (lf_bounds F) r with
    | Some (lo, hi) =>
        if Z.eqb lo hi then
          Some (match e with
                | EX | EZ => lo
                | ES => sext (wdn r) lo
                | EShr k => lo / 2 ^ k
                | EBf a n => (lo / 2 ^ a) mod 2 ^ n
                end)
        else None
    | None => None
    end
  else None.

Definition norm_term (ex : bool) (st : astate) (m : Z) (x : atom) (c : Z) : form * bool :=
  let (r, e) := x in
  match fixed_atom r e with Some v => ({| fm := m; ft := []; flo := c * v; fhi := c * v |}, true) | None =>
  let g := get st r in
  let point := fun h : form => negb ex || Z.eqb (flo h) (fhi h) in
  if interval_only g then
    match e with
    | EX => if Z.eqb (wd r) 64 then
              if match fact_range r g with
                 | Some (lo, hi) => (0 <=? lo) && (hi <? 2 ^ 63)
                 | None => false
                 end then (fatom m x c, false) else
              match sreg_range st r with
              | Some (lo, _) => if lo <? 0 then (fatom m (r, ES) c, false) else (fatom m x c, false)
              | None => (fatom m x c, false)
              end
            else (fatom m x c, false)
    | _ => (fatom m x c, false)
    end else
  match g, e with
  | Some (FForm f _ _), EX =>
      if Z.eqb (fm f) 64 && point f then (fscale f c 64, true) else (fatom m x c, false)
  | Some (FForm f _ _), ES =>
      if Z.eqb (wd r) 64 then
        match form_range st f with
        | Some rg => if Z.eqb (fm f) 64 && sext_ok 64 rg && point f then (fscale f c 64, true) else (fatom m x c, false)
        | None => (fatom m x c, false)
        end
      else if negb (Z.eqb (wd r) 32) then (fatom m x c, false) else
      match lift_reg st r (fat f 32) e with
      | Some h => if point h then (fscale h c 64, true) else (fatom m x c, false)
      | None => (fatom m x c, false)
      end
  | Some (FForm f _ _), EZ =>
      if negb (Z.eqb (wd r) 32) then (fatom m x c, false) else
      match lift_reg st r (fat f 32) e with
      | Some h => if point h then (fscale h c 64, true) else (fatom m x c, false)
      | None => (fatom m x c, false)
      end
  | _, EShr _ | _, EBf _ _ =>
      match copy_of r g with Some r2 => (fatom m (r2, e) c, true) | None => (fatom m x c, false) end
  | _, _ => (fatom m x c, false)
  end
  end.

Definition norm_once (ex : bool) (st : astate) (f : form) : form * bool :=
  let '(out, ch) :=
    fold_left (fun acc t => let '(o, ch) := acc in let '(x, c) := t in
                 let '(g, b) := norm_term ex st (fm f) x c in (fadd o g, ch || b))
              (ft f) ({| fm := fm f; ft := []; flo := flo f; fhi := fhi f |}, false) in
  (merge_shifts st (fat out (fm f)), ch).

Fixpoint normalize_n (ex : bool) (n : nat) (st : astate) (f : form) : form :=
  match n with
  | O => f
  | S n' => let '(g, ch) := norm_once ex st f in if ch then normalize_n ex n' st g else g
  end.

Definition normalize (st : astate) (f : form) : form := normalize_n false 12 st f.

(* A normal form that allows exactly the values f allows. *)
Definition normalize_exact (st : astate) (f : form) : form := normalize_n true 12 st f.

(* ------------------------------------------------------------------ *)
(* Masks and shifts.                                                   *)

(* k keeps exactly the bits [s, j). *)
Definition contiguous (k : Z) : option (Z * Z) :=
  if Z.leb k 0 then None else
  let s := v2 k in
  let j := Z.log2 (k + 2 ^ s) in
  if Z.eqb (k + 2 ^ s) (2 ^ j) then Some (s, j) else None.

(* The terms of f whose values are multiples of 2^s, the rest, and the
   multiple of 2^s in f's constant. *)
Definition split_at (st : astate) (f : form) (s : Z) : terms * terms * Z :=
  let A := filter (fun t => let '(x, c) := t in s <=? v2 c + atom_align st x) (ft f) in
  let B := filter (fun t => let '(x, c) := t in v2 c + atom_align st x <? s) (ft f) in
  let base := if Z.eqb (flo f) (fhi f) then flo f - flo f mod 2 ^ s else 0 in
  (A, B, base).

(* The register is known only as itself, so a bit field of it is an atom. *)
Definition plain (st : astate) (r : reg) : bool :=
  match get st r with
  | None | Some (FAlign _) => true
  | Some (FForm f _ _) as x => interval_only x || null (ft f)
  | _ => false
  end.

Definition range_form (w lo hi : Z) : form := {| fm := w; ft := []; flo := lo; fhi := hi |}.

(* The register whose value fa is exactly, when fa is a plain copy of one. *)
Definition mask_reg (st : astate) (fa : form) (a : reg) (bw : Z) : reg :=
  if negb (plain st a) && (bw <=? fm fa) && Z.eqb (flo fa) 0 && Z.eqb (fhi fa) 0 then
    match ft fa with
    | [((r, EX), 1)] | [((r, EZ), 1)] => if plain st r && (wd r <=? bw) then r else a
    | _ => a
    end
  else a.

Definition mask_rg (st : astate) (fa : form) (bw : Z) : option range :=
  if bw <=? fm fa then form_range st fa else None.

(* x & (2^n - 1): the terms below 2^n, or x itself, or the mask's range. *)
Definition mask_low_linear (st : astate) (fa : form) (k bw n : Z) : fact :=
  let rg := mask_rg st fa bw in
  let whole := match rg with
               | Some (x, y) => if (0 <=? x) && (y <=? k) then FForm fa 0 None else FForm (range_form bw 0 k) 0 None
               | None => FForm (range_form bw 0 k) 0 None
               end in
  let '(A, B, base) := split_at st fa n in
  match (if n <=? fm fa then form_range st (mk 64 B (flo fa - base) (fhi fa - base)) else None) with
  | Some (x, y) => if (0 <=? x) && (y <=? k) then FForm (mk bw B (flo fa - base) (fhi fa - base)) 0 None else whole
  | None => whole
  end.

Definition mask_low_residue (st : astate) (fa : form) (k bw n : Z) : fact :=
  let rg := mask_rg st fa bw in
  let whole := match rg with
               | Some (x, y) => if (0 <=? x) && (y <=? k) then FForm fa 0 None else FForm (range_form bw 0 k) 0 None
               | None => FForm (range_form bw 0 k) 0 None
               end in
  let '(A, B, base) := split_at st fa n in
  let residue := mk n B (flo fa - base) (fhi fa - base) in
  match (if n <=? fm fa then form_range st residue else None) with
  | Some (x, y) => if (0 <=? x) && (y <=? k) then FForm (mk bw (ft residue) (flo residue) (fhi residue)) 0 None else whole
  | None => whole
  end.

(* Preserve an existing exact linear projection; use the modular
   representative only when that projection supplies the mask interval. *)
Definition mask_low (st : astate) (fa : form) (k bw n : Z) : fact :=
 let linear := mask_low_linear st fa k bw n in
 match linear with
 | FForm f al None =>
   if null (ft f) && Z.eqb (flo f) 0 && Z.eqb (fhi f) k && (al <=? 0)
   then mask_low_residue st fa k bw n else linear
 | _=>linear end.

(* c * (r >> k) & (2^j - 2^s) for c a power of two at most 2^s, when the
   value lies below 2^j: 2^s times r shifted right by k + s - log2 c. With
   k = 0 the atom is r itself; with k > 0 it is a shifted register, as a
   Bloom filter's (h >> 3) & 0x1ffffffc, which is 4 * (h >> 5) for a hash
   h below 2^32. *)
Definition mask_shr (st : astate) (fa : form) (bw s j : Z) : option fact :=
  match ft fa, mask_rg st fa bw with
  | [((r, e), c)], Some (x, y) =>
      match shr_of e with
      | Some k =>
          if Z.eqb bw 64 && Z.eqb (flo fa) 0 && Z.eqb (fhi fa) 0 && (0 <=? k) && (0 <? c) && Z.eqb c (2 ^ v2 c)
             && (v2 c <=? s) && (0 <=? x) && (y <? 2 ^ j)
          then Some (FForm (fatom 64 (r, EShr (k + s - v2 c)) (2 ^ s)) 0 None) else None
      | None => None
      end
  | _, _ => None
  end.

(* x & (2^j - 2^s) is x less its low s bits, modulo 2^j. Below the full
   width, where that relation is weak, the fact is its range [0, k]. *)
Definition mask_rel (st : astate) (fa : form) (k bw s j : Z) : fact :=
  let rg := mask_rg st fa bw in
  let top := match rg with Some (x, y) => if 0 <=? x then Z.min k (y - y mod 2 ^ s) else k | None => k end in
  let rel := match rg with
             | Some (x, y) => if (0 <=? x) && (y <? 2 ^ j) then fshift fa (- (2 ^ s - 1)) 0
                              else fshift (fat fa j) (- (2 ^ s - 1)) 0
             | None => fshift (fat fa j) (- (2 ^ s - 1)) 0
             end in
  let low := if j <? bw then Some (0, k) else None in
  (* at the full width the relation is exact, and a value range would make
     the fact a range alone, which normalization does not read through, as
     in Numba's x - (x & -32) for x mod 32 *)
  FForm rel s (match rg with
               | Some (x, y) => if (0 <=? x) && (y <? 2 ^ j) && Z.eqb j bw then None
                                else if 0 <=? x then Some (0, top) else low
               | None => low
               end).

(* x & (2^j - 2^s), s > 0. *)
Definition mask_mid (st : astate) (fa : form) (k bw s j : Z) : fact :=
  match mask_shr st fa bw s j with Some x => x | None => mask_rel st fa k bw s j end.

(* x & k for a k whose bits are not contiguous. *)
Definition mask_other (st : astate) (fa : form) (k bw : Z) : fact :=
  let top := match mask_rg st fa bw with Some (x, y) => if 0 <=? x then Z.min k (y - y mod 2 ^ v2 k) else k | None => k end in
  FForm (range_form bw 0 top) (v2 k) None.

(* The fact of fa & k at width bw, where fa is the form of register a. *)
(* a's range is not negative, and k has none of the bits a value in it may
   have, as Numba's negative-index term (i & 2^63) has for an index i
   below 2^63 *)
Definition above_range (st : astate) (a : reg) (k : Z) : bool :=
  match reg_range st a with
  | Some (lo, hi) => (0 <=? lo) && Z.eqb (Z.land (Z.ones (bitlen hi)) k) 0
  | None => false
  end.

Definition mask (st : astate) (fa : form) (a : reg) (k bw : Z) : option fact :=
  (* k is zero, each bit of k is below the alignment of a, or above the
     bits of a's range *)
  if Z.eqb k 0 || (k <? 2 ^ reg_align st a) || above_range st a k then Some (FForm (fconst bw 0) 0 None) else
  let a := mask_reg st fa a bw in
  match contiguous k with
  | Some (s, j) =>
      (* a has no bit from its width up, so a mask to 2^j with j above the
         width, as Numba's x & -32 of a 32-bit x, keeps bits [s, wd a) *)
      if plain st a && (s <? wd a) then Some (FForm (fatom bw (a, EBf s (Z.min j (wd a) - s)) (2 ^ s)) 0 None)
      else if Z.eqb s 0 then
        (* low bits of which nothing is known are a's own bit field *)
        let low := mask_low st fa k bw j in
        if (j <=? wd a) && match low with
                           | FForm f al None => null (ft f) && Z.eqb (flo f) 0 && Z.eqb (fhi f) (2 ^ j - 1) && (al <=? 0)
                           | _ => false
                           end
        then Some (FForm (fatom bw (a, EBf s (j - s)) (2 ^ s)) 0 None) else Some low
      else Some (mask_mid st fa k bw s j)
  | None => Some (mask_other st fa k bw)
  end.

(* The value alignment of a form's value. *)
Definition falign (st : astate) (f : form) : Z := Z.min (form_align st f) (fm f).

(* fa has none of k's bits set, judged modulo 2^t for the t covering k: the
   low t bits of fa lie in [x, y] and are a multiple of 2^a, so k may have
   bits below a and from y's bit length, as Triton's offset of a thread's
   later element, (4 * tid & 508) | 513. *)
Definition low_bits_clear (st : astate) (fa : form) (k : Z) : bool :=
  let t := bitlen k in
  if Z.leb k 0 || Z.ltb (fm fa) t || negb (Z.eqb (flo fa) (fhi fa)) then false else
  let rest := mk 64 (filter (fun u => let '(x, c) := u in v2 c + atom_align st x <? t) (ft fa)) (flo fa) (flo fa) in
  match form_range st rest with
  | Some (x, y) =>
      let a := Z.min (Z.max 0 (falign st rest)) (bitlen y) in
      (0 <=? x) && (y <? 2 ^ t) && Z.eqb (Z.land (2 ^ bitlen y - 2 ^ a) k) 0
  | None => false
  end.

(* fa shifted right by k, when fa's value is a nonnegative integer below
   the width's bound and every term is a multiple of 2^k. *)
Definition aligned_shift (st : astate) (fa : form) (k bw : Z) (signed : bool) : option form :=
  if Z.ltb (fm fa) bw || negb (Z.eqb (flo fa) (fhi fa)) || negb (Z.eqb (flo fa mod 2 ^ k) 0)
     || negb (forallb (fun t => Z.eqb (snd t mod 2 ^ k) 0) (ft fa)) then None else
  match form_range st fa with
  | Some (x, y) =>
      if (0 <=? x) && (y <? (if signed then 2 ^ (bw - 1) else 2 ^ bw))
      then Some (mk bw (map (fun t => (fst t, snd t / 2 ^ k)) (ft fa)) (flo fa / 2 ^ k) (flo fa / 2 ^ k))
      else None
  | None => None
  end.

(* Split a nonnegative form into a multiple of 2^k and a proved low
   remainder. This retains packed fields when discarded low terms are
   nonzero, instead of requiring every coefficient to be divisible. *)
Definition quotient_shift (st : astate) (fa : form) (k bw : Z) (signed : bool) : option form :=
 match ft fa with _ :: _ :: _ =>
 if Z.ltb (fm fa) bw || negb (Z.eqb (flo fa) (fhi fa)) then None else
 let multiple := fun t : atom * Z => Z.eqb (snd t mod 2 ^ k) 0 in
 let high := filter multiple (ft fa) in
 let low := filter (fun t => negb (multiple t)) (ft fa) in
 if null high || null low then None else
 let remainder := {|fm:=64;ft:=low;flo:=flo fa mod 2 ^ k;fhi:=flo fa mod 2 ^ k|} in
 match form_range st fa, form_range st remainder with
 | Some(lo,hi),Some(rlo,rhi) =>
   if (0 <=? lo) && (hi <? (if signed then 2 ^ (bw-1) else 2 ^ bw))
      && (0 <=? rlo) && (rhi <? 2 ^ k)
   then Some (mk bw (map (fun t => (fst t,snd t / 2 ^ k)) high)
                    (flo fa / 2 ^ k) (flo fa / 2 ^ k)) else None
 | _,_=>None end
 | _=>None end.

Definition exact_shift (st : astate) (fa : form) (k bw : Z) (signed : bool) : option form :=
 match aligned_shift st fa k bw signed with Some g=>Some g
 | None=>quotient_shift st fa k bw signed end.

Definition opnd_align (st : astate) (o : opnd) : Z :=
  match o with OImm z => v2 z | OReg r => reg_align st r end.

(* a + sg * b at width bw: the sum of the two forms, or of the operands'
   relations modulo 2^bw when those cancel terms. *)
Definition addsub_fact (st : astate) (sg bw : Z) (a : reg) (b : opnd) : fact :=
  let fa := normalize st (opnd_form st (OReg a) bw) in
  let fb := normalize st (opnd_form st b bw) in
  let pl := fadd fa (fscale fb sg bw) in
  let al := Z.min (Z.min (reg_align st a) (opnd_align st b)) bw in
  if Z.eqb bw 64 then FForm pl al None else
  let ra := normalize st (narrow st a bw true) in
  let rb := normalize st (match b with OImm _ => opnd_form st b bw | OReg r => narrow st r bw true end) in
  let rel := fadd ra (fscale rb sg bw) in
  if Nat.ltb (length (ft rel)) (length (ft pl)) then FForm rel al None else FForm pl al None.

Definition mul_fact (st : astate) (bw : Z) (a : reg) (b : opnd) : option fact :=
  let fa := normalize st (opnd_form st (OReg a) bw) in
  match const_of st b bw false with
  | Some k => Some (FForm (fscale fa (k mod 2 ^ bw) bw) 0 None)
  | None =>
      match const_of st (OReg a) bw false with
      | Some k => Some (FForm (fscale (normalize st (opnd_form st b bw)) (k mod 2 ^ bw) bw) 0 None)
      | None =>
          let al := Z.min bw (reg_align st a + match b with OReg r => reg_align st r | OImm _ => 0 end) in
          match reg_range st a, opnd_range st b bw false with
          | Some (a0, a1), Some (b0, b1) =>
              if (0 <=? a0) && (0 <=? b0) && (a1 * b1 <? 2 ^ bw)
              then Some (FForm (range_form bw (a0 * b0) (a1 * b1)) al None)
              else if 0 <? al then Some (FAlign al) else None
          | _, _ => if 0 <? al then Some (FAlign al) else None
          end
      end
  end.

(* Restore an OR's relation through the complemented register when the
   normalized complement names that register exactly modulo64. *)
Definition or_complement (st : astate) (fa : form) (k : Z) : option fact :=
 let nf := normalize st (fshift (fscale fa (-1) 64) (-1) (-1)) in
 if negb (Z.eqb (fm nf) 64 && Z.eqb (flo nf) 0 && Z.eqb (fhi nf) 0) then None else
 match ft nf with
 | [((r, e), c)] =>
   if Z.eqb c 1 && (match e with EX | EZ => true | _ => false end) then
     match contiguous (k mod 2 ^ wd r) with
     | Some (s,j) => Some (FForm (fadd fa (fatom 64 (r,EBf s (j-s)) (2^s))) 0 None)
     | None => None end
   else None
 | _ => None end.

(* fa | k for a constant k. *)
Definition or_const (st : astate) (fa : form) (k : Z) : fact :=
  if low_bits_clear st fa k then FForm (fshift fa k k) 0 None else
  (* Every bit of k is already set when the complement has none of them. *)
  if low_bits_clear st (normalize st (fshift (fscale fa (-1) 64) (-1) (-1))) k
  then FForm fa 0 None else
  match or_complement st fa k with
  | Some f => f | None => FForm (fshift fa 0 k) 0 None end.

Definition or_fact (st : astate) (bw : Z) (a : reg) (b : opnd) : option fact :=
  let fa := normalize st (opnd_form st (OReg a) bw) in
  match b with
  | OReg rb =>
      match const_of st b bw false with
      | Some k => Some (or_const st fa (k mod 2 ^ bw))
      | None =>
          (* one side a multiple of 2^t, the other below 2^t *)
          let fb := normalize st (opnd_form st b bw) in
          let ra := reg_range st a in
          let rb' := reg_range st rb in
          let c1 := match rb' with Some (b0, b1) => (0 <=? b0) && (bitlen b1 <=? falign st fa) | None => false end in
          let c2 := match ra with Some (a0, a1) => (0 <=? a0) && (bitlen a1 <=? falign st fb) | None => false end in
          if c1 || c2 then Some (FForm (fadd fa fb) 0 None) else
          match ra, rb' with
          | Some (a0, a1), Some (b0, b1) =>
              (* each power of two that divides both divides a | b *)
              if (0 <=? a0) && (0 <=? b0) && (a1 + b1 <? 2 ^ bw)
              then Some (FForm (range_form bw 0 (a1 + b1)) (Z.min (falign st fa) (falign st fb)) None)
              else None
          | _, _ => None
          end
      end
  | OImm z => Some (or_const st fa (z mod 2 ^ bw))
  end.

(* a xor b of two registers whose values lie below 2^n lies below 2^n, and
   each power of two that divides both divides it, as a swizzled shared
   address of Triton's tl.sort. *)
Definition xor_fact (st : astate) (bw : Z) (a : reg) (b : opnd) : option fact :=
  match b with
  | OReg rb =>
      let fa := normalize st (opnd_form st (OReg a) bw) in
      let fb := normalize st (opnd_form st b bw) in
      match reg_range st a, reg_range st rb with
      | Some (a0, a1), Some (b0, b1) =>
          if (0 <=? a0) && (0 <=? b0) && (2 ^ Z.max (bitlen a1) (bitlen b1) <=? 2 ^ bw)
          then Some (FForm (range_form bw 0 (2 ^ Z.max (bitlen a1) (bitlen b1) - 1))
                           (Z.min (falign st fa) (falign st fb)) None)
          else None
      | _, _ => None
      end
  | OImm z => match const_of st (OReg a) bw false with
    | Some c => Some (FForm (fconst bw (Z.lxor c (z mod 2 ^ bw))) 0 None)
    | None =>
      match reg_range st a with
      | Some (a0, a1) =>
          let k := z mod 2 ^ bw in
          let n := Z.max (bitlen a1) (bitlen k) in
          if (0 <=? a0) && (2 ^ n <=? 2 ^ bw)
          then Some (FForm (range_form bw 0 (2 ^ n - 1))
                           (Z.min (reg_align st a) (v2 k)) None)
          else None
      | None => None end
    end
  end.

Definition shift_fact (st : astate) (signed : bool) (bw : Z) (a : reg) (k : Z) : fact :=
  let fa := normalize st (opnd_form st (OReg a) bw) in
  match exact_shift st fa k bw signed with
  | Some g => FForm g 0 None
  | None =>
      if Z.eqb bw 64 && negb signed then FForm (atom64 a (EShr k)) 0 None else
      match (if signed then sreg_range st a else reg_range st a) with
      | Some (x, y) => FForm (range_form bw (x / 2 ^ k) (y / 2 ^ k)) 0 None
      | None =>
          if signed then
            if k <? bw then FForm (range_form bw (- 2 ^ (bw - 1 - k)) (2 ^ (bw - 1 - k) - 1)) 0 None
            else FForm (range_form bw (-1) 0) 0 None
          else FForm (range_form bw 0 ((2 ^ bw - 1) / 2 ^ k)) 0 None
      end
  end.

Definition binop_fact (st : astate) (o : binop) (bw : Z) (a : reg) (b : opnd) : option fact :=
  let fa := normalize st (opnd_form st (OReg a) bw) in
  match o with
  | BAdd => Some (addsub_fact st 1 bw a b)
  | BSub => Some (addsub_fact st (-1) bw a b)
  | BMul => mul_fact st bw a b
  | BOr => or_fact st bw a b
  | BXor => xor_fact st bw a b
  (* a shift amount is read as a 32-bit value *)
  | BShl =>
      match const_of st b 32 false with
      | Some k => Some (FForm (fscale fa (2 ^ Z.min (k mod 2 ^ 32) bw) bw) (Z.min (k mod 2 ^ 32) bw) None)
      | None => None
      end
  | BAnd =>
      match const_of st b bw false, b with
      | Some k, _ => mask st fa a (k mod 2 ^ bw) bw
      | None, OImm _ => None
      | None, OReg rb =>
          match const_of st (OReg a) bw false with
          (* the constant may be the first operand, as in Numba's
             negative-index term (i >> 63) & shape *)
          | Some k => mask st (normalize st (opnd_form st b bw)) rb (k mod 2 ^ bw) bw
          | None =>
              (* a & b is at most either operand's value *)
              let hi := match reg_range st a, reg_range st rb with
                        | Some (_, x), Some (_, y) => Some (Z.min x y)
                        | Some (_, x), None | None, Some (_, x) => Some x
                        | None, None => None
                        end in
              match hi with
              | Some m => if m <? 0 then None else Some (FForm (range_form bw 0 m) 0 None)
              | None => None
              end
          end
      end
  | BShr | BShrs =>
      match const_of st b 32 false with
      | Some k => Some (shift_fact st (match o with BShrs => true | _ => false end) bw a (Z.min (k mod 2 ^ 32) bw))
      | None => None
      end
  end.

(* ------------------------------------------------------------------ *)
(* Refinement by a comparison.                                         *)

Definition put (st : astate) (r : reg) (x : fact) : astate :=
  {| amap := PM.add r x (amap st); acons := acons st |}.

Definition without (ts : terms) (x : atom) : terms := filter (fun t => negb (atom_eqb (fst t) x)) ts.

(* The sum of the coefficients of atom x in ts. *)
Definition coef (ts : terms) (x : atom) : Z :=
  fold_right (fun t s => if atom_eqb (fst t) x then snd t + s else s) 0 ts.

(* Terms in atom order with zero coefficients dropped: the sum is exact,
   unlike canon, and the order is the one canon leaves, so a constraint of
   these terms matches a form. *)
Definition sorted_terms (ts : terms) : terms := filter (fun t => negb (Z.eqb (snd t) 0)) (tsort ts).

Definition constraint_eqb (a b : constraint) : bool :=
 let '(ts,lo,hi) := a in let '(us,l,h) := b in
 terms_eqb ts us && Z.eqb lo l && Z.eqb hi h.

(* A constraint that the state already has adds nothing. A selection copies
   each operand's constraints to its result, so a chain of selections, as an
   unrolled binary search's bounds, would otherwise double them each step. *)
Definition add_cons (st : astate) (k : constraint) : astate :=
  if existsb (constraint_eqb k) (acons st) then st
  else {| amap := amap st; acons := k :: acons st |}.

(* The bounds of a value read at width w. *)
Definition lo_ok (w : Z) (sg : bool) : Z := if sg then - 2 ^ (w - 1) else 0.
Definition hi_ok (w : Z) (sg : bool) : Z := if sg then 2 ^ (w - 1) - 1 else 2 ^ w - 1.

(* A register's form f, compared at width w: f, or, when f is one register
   x of width w, x read with the comparison's extension, which agrees with f
   modulo 2^w, as for a signed test of a copy of a callee's 32-bit result. *)
Definition reading (f : form) (w : Z) (sg : bool) : form :=
  match ft f with
  | [((x, e), c)] =>
      if Z.eqb c 1 && Z.eqb (flo f) 0 && Z.eqb (fhi f) 0 && Z.eqb (wd x) w &&
         match e with EZ | ES => true | _ => false end
      then atom64 x (if sg then ES else EZ) else f
  | _ => f
  end.

(* The operand as a form whose integer value is the compared value whenever
   the form's range lies within the comparison's bounds. *)
Definition cmp_form (st : astate) (o : opnd) (w : Z) (sg : bool) : form :=
  match o with
  | OImm z => fconst 64 (extend (Z.to_nat w) sg z)
  | OReg r =>
      if Z.eqb (wd r) 64 then
        if sg && (match get st r with None | Some (FAlign _) => true | x => interval_only x end) &&
           (match fact_range r (get st r) with
            | Some (lo, hi) => negb ((0 <=? lo) && (hi <? 2 ^ 63))
            | None => true
            end)
        then atom64 r ES else normalize st (as_form st r EX)
      else
      match get st r with
      | Some (FForm f _ _) as x =>
          if (wd r <=? fm f) && negb (interval_only x) then reading f w sg else atom64 r (if sg then ES else EZ)
      | _ => atom64 r (if sg then ES else EZ)
      end
  end.

Definition aligned_range (st : astate) (o : opnd) (f : form) (w : Z) : option range :=
  match form_range st f with
  | None => None
  | Some g => match o with OImm _ => Some g | OReg r => Some (round (Z.min (reg_align st r) (w - 1)) g) end
  end.

Definition cmp_range (st : astate) (o : opnd) (w : Z) (sg : bool) : option range :=
  match aligned_range st o (cmp_form st o w sg) w with
  | Some (x, y) => if (lo_ok w sg <=? x) && (y <=? hi_ok w sg) then Some (x, y) else None
  | None => None
  end.

(* r with its value a multiple of 2^n. *)
Definition add_align (st : astate) (r : reg) (n : Z) : astate :=
  match get st r with
  | Some (FForm f al vr) => put st r (FForm f (Z.max al n) vr)
  | Some (FAlign k) => put st r (FAlign (Z.max k n))
  | None => put st r (FAlign n)
  | _ => st
  end.

(* Narrow register r, seen through extension e, to [lo, hi]. Low bits that
   are all zero make r a multiple of their power of two. *)
Definition tighten_reg (st : astate) (r : reg) (e : ext) (lo hi : Z) : option astate :=
  match e with
  | EBf p n => if Z.eqb p 0 && Z.eqb lo 0 && Z.eqb hi 0 && (0 <=? n) then Some (add_align st r n) else Some st
  | EShr _ => Some st
  | _ =>
      match get st r with
      | x =>
          let keep := match x with
                      | None | Some (FAlign _) => true
                      | Some (FForm f _ _) => interval_only x || null (ft f)
                      | _ => false
                      end in
          if negb keep then Some st else
          let al := reg_align st r in
          let w := wd r in
          let cur := match (match e with ES => sreg_range st r | _ => reg_range st r end) with
                     | Some g => g
                     | None => match e with ES => (- 2 ^ (w - 1), 2 ^ (w - 1) - 1) | _ => (0, 2 ^ w - 1) end
                     end in
          let g := round al (Z.max (fst cur) lo, Z.min (snd cur) hi) in
          if snd g <? fst g then None else Some (put st r (FForm (range_form w (fst g) (snd g)) al None))
      end
  end.

Definition ceil_div (a c : Z) : Z := - ((- a) / c).

(* out where the integer value of form f, whose range in st is rg if known,
   lies in [lo, hi]: each atom of f is narrowed. *)
Definition constrain_form (out st : astate) (f : form) (lo hi : Z) (rg : option range) : option astate :=
  let '(lo, hi) := match rg with Some (x, y) => (Z.max lo x, Z.min hi y) | None => (lo, hi) end in
  if hi <? lo then None else
  let res :=
    fold_left (fun acc t =>
      match acc with
      | None => None
      | Some o =>
          let x := fst t in
          (* x's coefficient over all its terms, and the other terms with
             the constant, unreduced so that their sum is exact *)
          let c := fold_right (fun u s => if atom_eqb (fst u) x then snd u + s else s) 0 (ft f) in
          let others := {| fm := 64; ft := filter (fun u => negb (atom_eqb (fst u) x)) (ft f);
                           flo := flo f; fhi := fhi f |} in
          if Z.eqb c 0 then Some o else
          match form_range st others with
          | None => Some o
          | Some (p, q) =>
              let a := lo - q in let b := hi - p in
              let '(rlo, rhi) := if 0 <? c then (ceil_div a c, b / c) else (ceil_div b c, a / c) in
              tighten_reg o (fst x) (snd x) rlo rhi
          end
      end) (ft f) (Some out) in
  match res with
  | None => None
  | Some o =>
      match ft f with
      | _ :: _ :: _ => Some (add_cons o (ft f, lo - fhi f, hi - flo f))
      | [((r, e), _)] =>
          (* a bit field or a shift of r, which tighten_reg does not narrow,
             is bounded by a constraint of the atom, as bits [0, 5) of %tid.x
             under the guard (tid & 31) <= 7 *)
          if match e with EBf _ _ | EShr _ => true | _ => false end
          then Some (add_cons o (ft f, lo - fhi f, hi - flo f)) else
          (* tighten_reg keeps no range for a register with a relational
             fact. The bound is a constraint of the atom, and of the atom's
             normal form with that constraint when the two cannot differ by
             a multiple of 2^64. *)
          match get o r with
          | Some (FForm g _ _) as x =>
              if negb (interval_only x) && negb (null (ft g)) then
                let o1 := add_cons o (ft f, lo - fhi f, hi - flo f) in
                let N := normalize o1 f in
                match form_range_nat o1 N with
                | Some (p, q) =>
                    if Z.eqb (fm f) 64 && negb (terms_eqb (ft N) (ft f)) && (q - lo <? 2 ^ 64) && (hi - p <? 2 ^ 64)
                    then Some (add_cons o1 (ft N, lo - fhi N, hi - flo N)) else Some o1
                | None => Some o1
                end
              else Some o
          | _ => Some o
          end
      | [] => Some o
      end
  end.

(* st where operand o's value, read at width w (signed if sg), lies in
   [lo, hi] (a missing bound is o's own). *)
Definition constrain (st : astate) (o : opnd) (lo hi : option Z) (w : Z) (sg : bool) : option astate :=
  match o with
  | OImm _ => Some st
  | OReg r =>
      let f := cmp_form st o w sg in
      let rg := aligned_range st o f w in
      (* the form's range, or with each atom of unknown range in its natural range *)
      let nr := match rg with Some g => Some g | None => form_range_nat st f end in
      let exact := match nr with Some (x, y) => (lo_ok w sg <=? x) && (y <=? hi_ok w sg) | None => false end in
      let single := match ft f with
                    | [((r', e), c)] =>
                        Z.eqb c 1 && Z.eqb (flo f) 0 && Z.eqb (fhi f) 0 && Pos.eqb r' r && Z.eqb (wd r) w &&
                        match e with EX | EZ => negb sg | ES => sg | _ => false end
                    | _ => false
                    end in
      (* an unsigned upper bound below every wrapped value of the form: the
         form's value is then the compared value *)
      let wrap := match sg, lo, hi, nr with
                  | false, None, Some h, Some (x, y) => (- 2 ^ w <? x) && (y <? 2 ^ w) && (h <? x + 2 ^ w)
                  | _, _, _, _ => false
                  end in
      if negb exact && negb single && negb wrap then Some st else
      let lo := match lo with
                | Some v => v
                | None => match rg with Some (x, _) => Z.max x (lo_ok w sg) | None => lo_ok w sg end
                end in
      let hi := match hi with
                | Some v => v
                | None => match rg with Some (_, y) => Z.min y (hi_ok w sg) | None => hi_ok w sg end
                end in
      let '(lo, hi) := round (Z.min (reg_align st r) (w - 1)) (lo, hi) in
      let own := match get st r with
                 | None | Some (FAlign _) => true
                 | Some (FForm g _ _) => null (ft g)
                 | _ => false
                 end in
      let out := if own then tighten_reg st r (if sg then ES else if Z.eqb (wd r) 64 then EX else EZ) lo hi else Some st in
      match out with
      | None => None
      | Some out => constrain_form out st f lo hi rg
      end
  end.

Definition flip (c : cmp) : cmp :=
  match c with CLt => CGe | CGe => CLt | CLe => CGt | CGt => CLe | CEq => CNe | CNe => CEq end.

(* The integer sum of a constraint's terms, as a form. *)
Definition form_of_terms (ts : terms) : form := {| fm := 64; ft := ts; flo := 0; fhi := 0 |}.

(* The state s that a refinement of st gives, unless no values satisfy the
   constraint it added last: the range of its terms in st lies outside its
   bounds, as for i >= n where st has i - n <= -1. *)
Definition settle (st s : astate) : option astate :=
  match acons s with
  | (ts, lo, hi) :: _ =>
      match form_range_near st (form_of_terms ts) with
      | Some (x, y) => if (y <? lo) || (hi <? x) then None else Some s
      | None => Some s
      end
  | [] => Some s
  end.

(* The relation dlo <= a - b <= dhi between the compared values of a and b,
   whose comparison forms fa and fb are exact (their ranges lie in the
   comparison's bounds), as constraints: on the difference of the forms,
   and, when fb is one atom, through each constraint that mentions the atom
   with coefficient 1, so that j < end and end - S <= 0 give j - S <= -1. *)
Definition relate (st : astate) (fa fb : form) (dlo dhi : Z) : astate :=
  let direct := (sorted_terms (ft fa ++ tscale (-1) (ft fb)), dlo - fhi fa + flo fb, dhi - flo fa + fhi fb) in
  let through :=
    match ft fb with
    | [(xb, 1)] =>
        if Z.eqb (flo fb) 0 && Z.eqb (fhi fb) 0 then
          flat_map (fun k => let '(ts, lo, hi) := k in
                      if Z.eqb (coef ts xb) 1
                      then [(sorted_terms (ft fa ++ without ts xb), dlo - fhi fa + lo, dhi - flo fa + hi)]
                      else []) (acons st)
        else []
    | _ => []
    end in
  {| amap := amap st; acons := direct :: through ++ acons st |}.

(* The compared register as its own atom, when the atom's value is the
   compared value. *)
Definition cmp_atom (st : astate) (r : reg) (w : Z) (sg : bool) : option form :=
  let g := atom64 r (if Z.eqb (wd r) 64 then EX else if sg then ES else EZ) in
  match form_range st g with
  | Some (x, y) => if (lo_ok w sg <=? x) && (y <=? hi_ok w sg) then Some g else None
  | None => None
  end.

(* The relation of the compared forms, and of register a's own atom to the
   other form: a join forgets the form of a register that steps, such as a
   loop counter, but keeps a relation on its atom. *)
Definition relate_cmp (st : astate) (a : reg) (fa fb : form) (w : Z) (sg : bool) (dlo dhi : Z) : astate :=
  let st1 := relate st fa fb dlo dhi in
  match cmp_atom st a w sg with
  | Some ga => if terms_eqb (ft fa) (ft ga) then st1 else relate st1 ga fb dlo dhi
  | None => st1
  end.

(* Whether constraint k has an atom of register r. *)
Definition cons_mentions (r : reg) (k : constraint) : bool :=
  existsb (fun t => Pos.eqb (fst (fst t)) r) (fst (fst k)).

(* The relation of register a's own atom to a constant's form fb, when a has
   constraints on that atom: a loop's copy of its position, whose form a
   step rewrote in other registers, is refined through its atom, and its
   constraints can show the comparison impossible, as Numba's i < -1 for a
   row i the loop keeps at least 0. *)
Definition relate_atom (st : astate) (a : reg) (fb : form) (w : Z) (sg : bool) (dlo dhi : Z) : option astate :=
  match cmp_atom st a w sg with
  | Some ga => if existsb (cons_mentions a) (acons st) then settle st (relate st ga fb dlo dhi) else Some st
  | None => Some st
  end.

Definition refine_cmp_raw (st : astate) (k : cmpf) : option astate :=
  let w := Z.of_nat (cw k) in
  let sg := csg k in
  let a := ca k in
  let b := cb k in
  if negb (cmp_wf k) then Some st else
  (* both compared values lie in [lo_ok, hi_ok], so their difference lies
     within 2^w - 1 of zero *)
  let far := 2 ^ w - 1 in
  match cc k with
  | CGt | CGe =>
      let d := match cc k with CGt => 1 | _ => 0 end in
      match b with
      | OReg rb =>
          (* a > b is b < a *)
          let ra := cmp_range st (OReg a) w sg in
          let rbr := cmp_range st b w sg in
          let st1 := match ra with Some (_, y) => constrain st b None (Some (y - d)) w sg | None => Some st end in
          match st1 with
          | None => None
          | Some s1 =>
              let s2 := match rbr with Some (x, _) => constrain s1 (OReg a) (Some (x + d)) None w sg | None => Some s1 end in
              match s2, ra, rbr with
              | Some s2, Some _, Some _ => settle s2 (relate_cmp s2 a (cmp_form st (OReg a) w sg) (cmp_form st b w sg) w sg d far)
              | _, _, _ => s2
              end
          end
      | OImm z =>
          match constrain st (OReg a) (Some (extend (cw k) sg z + d)) None w sg with
          | Some s1 => relate_atom s1 a (cmp_form st b w sg) w sg d far
          | None => None
          end
      end
  | CLt | CLe =>
      let d := match cc k with CLt => 1 | _ => 0 end in
      (* Only a register comparison constrains the other operand from a's
         range. An immediate comparison never consumes this range. *)
      let ra := match b with
                | OReg _ => cmp_range st (OReg a) w sg
                | OImm _ => None
                end in
      let rbr := cmp_range st b w sg in
      let st1 := match rbr with Some (_, y) => constrain st (OReg a) None (Some (y - d)) w sg | None => Some st end in
      match st1 with
      | None => None
      | Some s1 =>
          let s2 := match ra, b with Some (x, _), OReg _ => constrain s1 b (Some (x + d)) None w sg | _, _ => Some s1 end in
          match s2, ra, rbr, b with
          | Some s2, Some _, Some _, OReg _ => settle s2 (relate_cmp s2 a (cmp_form st (OReg a) w sg) (cmp_form st b w sg) w sg (- far) (- d))
          | Some s2, _, _, OImm _ => relate_atom s2 a (cmp_form st b w sg) w sg (- far) (- d)
          | _, _, _, _ => s2
          end
      end
  | CEq | CNe =>
      (* a = b exactly when D = a - b is 0 modulo 2^w; when D's range lies
         inside (-2^w, 2^w), exactly when D = 0 *)
      let D := normalize st (fadd (cmp_form st (OReg a) w sg) (fscale (cmp_form st b w sg) (-1) 64)) in
      if fm D <? w then Some st else
      if null (ft D) && Z.eqb (flo D) (fhi D) then
        let zero := Z.eqb (flo D mod 2 ^ w) 0 in
        if Bool.eqb zero (match cc k with CEq => true | _ => false end) then Some st else None
      else
      match form_range st D with
      | None => Some st
      | Some (x, y) =>
          if (x <=? - 2 ^ w) || (2 ^ w <=? y) then Some st else
          let al := Z.min (Z.min (reg_align st a) (opnd_align st b)) (w - 1) in
          let g := round al (x, y) in
          match cc k with
          | CEq => constrain_form st st D 0 0 (Some g)
          | _ => if Z.eqb (fst g) 0 then constrain_form st st D (2 ^ al) (snd g) (Some g)
                 else if Z.eqb (snd g) 0 then constrain_form st st D (fst g) (- 2 ^ al) (Some g)
                 else Some st
          end
      end
  end.

Definition unsigned_equality (k : cmpf) : cmpf :=
 match cc k with CEq | CNe => {|cc:=cc k;cw:=cw k;csg:=false;ca:=ca k;cb:=cb k|} | _ => k end.
Definition refine_cmp (st : astate) (k : cmpf) : option astate := refine_cmp_raw st (unsigned_equality k).

Fixpoint refine_all (st : astate) (cs : list cmpf) : option astate :=
  match cs with
  | [] => Some st
  | k :: cs' => match refine_cmp st k with None => None | Some s => refine_all s cs' end
  end.

Definition negate (k : cmpf) : cmpf := {| cc := flip (cc k); cw := cw k; csg := csg k; ca := ca k; cb := cb k |}.

(* A false exact conjunction is the union of its failing comparison branches.
   Retain only constraints shared by every reachable alternative, with bounds
   widened to include each branch. *)
Definition common_constraints (a b : list constraint) : list constraint :=
 flat_map (fun k => let '(ts,lo,hi) := k in
 match find (fun k' => let '(ts',_,_) := k' in terms_eqb ts ts') b with
 | Some (_,lo',hi') => [(ts,Z.min lo lo',Z.max hi hi')] | None => [] end) a.
Definition join_alternatives (st : astate) (states : list astate) : option astate :=
 match states with [] => None | [s] => Some s | s::rest =>
 Some {|amap:=amap st;acons:=fold_left (fun cs s => common_constraints cs (acons s)) rest (acons s)|} end.

Definition refine_cmps (st : astate) (cs : list cmpf) (exact all : bool) : option astate :=
  if all then refine_all st cs
  else if negb exact then Some st
  else join_alternatives st (flat_map (fun k => match refine_cmp st (negate k) with Some s => [s] | None => [] end) cs).

(* st with the comparisons of predicate register p refined, where p is
   (holds) true. *)
Definition refine_pred (st : astate) (p : reg) (holds : bool) : option astate :=
  match get st p with
  | Some (FPred cs exact) => refine_cmps st cs exact holds
  | Some (FNeg cs exact) => refine_cmps st cs exact (negb holds)
  | Some (FForm f _ _) =>
      (* a predicate set to a constant *)
      if null (ft f) && Z.eqb (flo f) (fhi f) && (1 <=? fm f) && Z.eqb (wd p) 1 then
        if Bool.eqb (negb (Z.eqb (flo f mod 2) 0)) holds then Some st else None
      else Some st
  | _ => Some st
  end.

(* An atom of register p that reads p's value. *)
Definition value_atom (p : reg) (x : atom) : bool :=
  Pos.eqb (fst x) p && match snd x with EX | EZ => true | _ => false end.

(* The terms with p's value atoms replaced by v: the other terms, and the
   constant they add. *)
Definition subst_terms (p : reg) (v : Z) (ts : terms) : terms * Z :=
  (filter (fun t => negb (value_atom p (fst t))) ts,
   fold_left (fun acc (t : atom * Z) => if value_atom p (fst t) then acc + snd t * v else acc) ts 0).

Definition subst_form (p : reg) (v : Z) (f : form) : form :=
  let '(ts, k) := subst_terms p v (ft f) in {| fm := fm f; ft := ts; flo := flo f + k; fhi := fhi f + k |}.

(* st with register p known to hold v. *)
Definition subst_pred (st : astate) (p : reg) (v : Z) : astate :=
  {| amap := PM.map (fun x => match x with FForm f al vr => FForm (subst_form p v f) al vr | _ => x end) (amap st);
     acons := map (fun k => let '(ts, lo, hi) := k in let '(ts', c) := subst_terms p v ts in (ts', lo - c, hi - c))
                  (acons st) |}.

(* st on an edge where predicate register p is (holds) true: its
   comparisons refined, and a one-bit p's value, 1 or 0, in every form, as
   for i := i + (p ? 1 : 0) on the edge where p holds. *)
Definition refine (st : astate) (p : reg) (holds : bool) : option astate :=
  match refine_pred st p holds with
  | Some s => Some (if Z.eqb (wd p) 1 then subst_pred s p (if holds then 1 else 0) else s)
  | None => None
  end.

(* st where guard g holds. *)
Definition refine_guard (st : astate) (g : guard) : option astate :=
  match g with None => Some st | Some (neg, p) => refine st p (negb neg) end.

(* ------------------------------------------------------------------ *)
(* Roles and equal forms.                                              *)

Definition role_of (P : reg) : option span := lookup (lf_roles F) P.

Definition form_eqb (f g : form) : bool :=
  Z.eqb (fm f) (fm g) && Z.eqb (flo f) (flo g) && Z.eqb (fhi f) (fhi g) &&
  (Nat.eqb (length (ft f)) (length (ft g))) &&
  forallb (fun p => atom_eqb (fst (fst p)) (fst (snd p)) && Z.eqb (snd (fst p)) (snd (snd p)))
          (combine (ft f) (ft g)).

(* ------------------------------------------------------------------ *)
(* Values of instructions.                                             *)

Definition pred_union (x y : option fact) : option fact :=
  let cx := match x with Some (FPred cs _) => cs | _ => [] end in
  let cy := match y with Some (FPred cs _) => cs | _ => [] end in
  let exact := match x, y with Some (FPred _ e1), Some (FPred _ e2) => e1 && e2 | _, _ => false end in
  match cx ++ cy with [] => None | cs => Some (FPred cs exact) end.

(* The comparisons that hold when a predicate of fact x fails, and whether
   they are exact: an exact comparison's negation. *)
Definition fails_cmps (x : option fact) : list cmpf * bool :=
  match x with
  | Some (FNeg cs e) => (cs, e)
  | Some (FPred [k] true) => ([negate k], true)
  | _ => ([], false)
  end.

(* The disjunction fails when both predicates fail. *)
Definition pred_disj (x y : option fact) : option fact :=
  let '(cx, ex) := fails_cmps x in
  let '(cy, ey) := fails_cmps y in
  match cx ++ cy with [] => None | cs => Some (FNeg cs (ex && ey)) end.

Definition cmpf_eqb (k l : cmpf) : bool :=
  match cc k, cc l with
  | CLt, CLt | CLe, CLe | CGt, CGt | CGe, CGe | CEq, CEq | CNe, CNe => true
  | _, _ => false
  end && Nat.eqb (cw k) (cw l) && Bool.eqb (csg k) (csg l) && Pos.eqb (ca k) (ca l) &&
  match cb k, cb l with
  | OReg a, OReg b => Pos.eqb a b
  | OImm a, OImm b => Z.eqb a b
  | _, _ => false
  end.

(* The exclusive or holds when exactly one predicate holds, so the
   comparisons of both hold, as for Triton's loop mask
   pending := pending xor stored, where stored implies pending. *)
Definition pred_common (x y : option fact) : option fact :=
  match x, y with
  | Some (FPred cx _), Some (FPred cy _) =>
      match filter (fun k => existsb (cmpf_eqb k) cy) cx with [] => None | cs => Some (FPred cs false) end
  | _, _ => None
  end.

(* The quotient or remainder by a positive constant k, which PTX defines for
   every dividend, from the dividend's range; and rule R4's range of the
   remainder of a nonnegative dividend by a register of positive range:
   below the register, and at most the dividend. *)
Definition div_fact (st : astate) (rm sg : bool) (w : Z) (a : reg) (b : opnd) : option fact :=
  match const_of st b w sg, opnd_range st (OReg a) w sg with
  | Some k, Some (x, y) =>
      if 0 <? k then
        let '(lo, hi) := if rm then (if 0 <=? x then (0, Z.min (k - 1) y) else (1 - k, k - 1))
                         else (Z.quot x k, Z.quot y k) in
        Some (FForm (range_form w lo hi) 0 None)
      else None
  | None, Some (x, y) =>
      if rm && (0 <=? x) then
        match opnd_range st b w sg with
        | Some (k0, k1) => if 0 <? k0 then Some (FForm (range_form w 0 (Z.min (k1 - 1) y)) 0 None) else None
        | None => None
        end
      else None
  | _, _ => None
  end.

(* Boolean identities use a stored constant bit. Recursively deriving a new
   range at every Boolean operation duplicates the refinement work. *)
Definition known_bit (st : astate) (r : reg) : option bool :=
 match fact_range r (get st r) with Some (lo,hi) =>
 if Z.eqb lo 0 && Z.eqb hi 0 then Some false else
 if Z.eqb lo 1 && Z.eqb hi 1 then Some true else None | None => None end.
Definition bit_fact (v : bool) : option fact := Some (FForm (fconst 1 (if v then 1 else 0)) 0 None).

(* Negation exchanges the meanings of true and false without discarding
   the comparisons or their exactness. *)
Definition opposite_predicate (x : option fact) : option fact :=
 match x with Some (FPred cs exact) => Some (FNeg cs exact)
 | Some (FNeg cs exact) => Some (FPred cs exact) | _ => None end.

(* A recovered one-bit predicate may be held in a wider integer register.
   Testing its exact copy against zero preserves the original comparisons. *)
Definition copied_zero_predicate (st : astate) (c : cmp) (w : nat) (a : reg) (b : opnd) : option fact :=
 match c,b with
 | CEq,OImm 0 | CNe,OImm 0 =>
   if Nat.leb 1 w then
    match copy_of a (get st a) with
    | Some p => if Z.eqb (wd p) 1 then
       match c with CEq=>opposite_predicate (get st p) | _=>get st p end
       else None
    | None=>None end
   else None
 | _,_=>None end.

(* Lift an exact full-source-width residue when its form already lies in
   the converted reading's range. This preserves non32-bit packed forms. *)
Definition converted_form (st : astate) (s : reg) (n : nat) (signed : bool) : option form :=
 match get st s with
 | Some (FForm f _ _) =>
   if null (ft f) || negb (Nat.leb 1 n) || (fm f <? Z.of_nat n) then None else
   match bounded_form_range st f (lo_ok (Z.of_nat n) signed) (hi_ok (Z.of_nat n) signed) with
   | Some(lo,hi)=>
     if (lo_ok (Z.of_nat n) signed <=? lo) && (hi <=? hi_ok (Z.of_nat n) signed)
     then Some {|fm:=64;ft:=ft f;flo:=flo f;fhi:=fhi f|} else None
   |None=>None end
 | _=>None end.

Definition value (st : astate) (i : instr) : option fact :=
  match i with
  | IMov _ d s =>
      match get st s with
      | Some (FAlign _) | Some (FPred _ _) | Some (FNeg _ _) => get st s
      | _ => if Z.eqb (wd s) (wd d) then Some (FForm (opnd_form st (OReg s) (wd d)) 0 None) else None
      end
  | IImm _ d z => Some (FForm (fconst (wd d) (z mod 2 ^ wd d)) 0 None)
  | IBin _ o w _ a b => binop_fact st o (Z.of_nat w) a b
  | IMad _ w _ a b c =>
      let w := Z.of_nat w in
      match const_of st b w false with
      | Some k => Some (FForm (fadd (fscale (normalize st (opnd_form st (OReg a) w)) (k mod 2 ^ w) w)
                                    (normalize st (opnd_form st c w))) 0 None)
      | None =>
          match const_of st (OReg a) w false with
          | Some k => Some (FForm (fadd (fscale (normalize st (opnd_form st b w)) (k mod 2 ^ w) w)
                                        (normalize st (opnd_form st c w))) 0 None)
          | None => None
          end
      end
  | IWide _ signed _ a b c =>
      let e := if signed then ES else EZ in
      let prod := match const_of st b 32 signed with
                  | Some k => Some (fscale (normalize st (as_form st a e)) k 64)
                  | None =>
                      (* the constant may be the first operand, as when
                         Numba multiplies the block size by the block index *)
                      match b, const_of st (OReg a) 32 signed with
                      | OReg rb, Some k => Some (fscale (normalize st (as_form st rb e)) k 64)
                      | _, _ => None
                      end
                  end in
      match prod with
      | Some f => Some (FForm (match c with Some r => fadd f (normalize st (as_form st r EX)) | None => f end) 0 None)
      | None => None
      end
  | ICvt _ to from signed _ s =>
      if Nat.eqb to 64 && Nat.eqb from 32 then
        let e := if signed then ES else EZ in
        if Z.eqb (wd s) 32 then Some (FForm (normalize st (as_form st s e)) 0 None) else
        (* a wider source: the extension of its low 32 bits *)
        match get st s with
        | Some (FForm f _ _) as x =>
            if interval_only x then None else
            match lift st (fat f 32) e with
            | Some g => Some (FForm (normalize st g) 0 None)
            (* the residue: the value agrees with the source modulo 2^32,
               as Triton's sign extension of a 64-bit sum begin + j whose
               range the guard of the access bounds, not the conversion.
               A zero extension lies in [0, 2^32); the transfer states the
               bound of a sign extension's signed reading as a constraint *)
            | None => Some (FForm (fat f 32) 0 (if signed then None else Some (0, 2 ^ 32 - 1)))
            end
        | _ => None
        end
      else if Nat.eqb to 64 && Nat.eqb from 64 then Some (FForm (normalize st (as_form st s EX)) 0 None)
      (* the low 32 bits of a 64-bit source, also when the conversion reads
         only those bits, as the load of a 64-bit parameter's low half *)
      else if Nat.eqb to 32 && (Nat.eqb from 64 || Nat.eqb from 32 && Z.eqb (wd s) 64) then
        Some (FForm (fat (normalize st (as_form st s EX)) 32) 0 None)
      else if Nat.eqb to from && Z.eqb (wd s) (Z.of_nat from) then get st s
      (* A full-width source at any integer width: preserve its signed or
         unsigned value, then reduce it to the destination's width. *)
      else if Nat.eqb from (wdn s) then
        match converted_form st s from signed with
        | Some f=>Some (FForm (fat f (Z.of_nat to)) 0 None)
        | None=>Some (FForm (fat (normalize st (as_form st s (if signed then ES else EZ))) (Z.of_nat to)) 0 None)
        end
      else None
  | IBfe _ w _ a p ln =>
      let w := Z.of_nat w in let p := Z.of_nat p in let ln := Z.of_nat ln in
      if p + ln <=? wd a then Some (FForm (fatom w (a, EBf p ln) 1) 0 None) else
      let hi := match reg_range st a with Some (_, y) => Z.min (2 ^ ln - 1) (y / 2 ^ p) | None => 2 ^ ln - 1 end in
      Some (FForm (range_form w 0 hi) 0 None)
  | ISetp _ c w signed _ a b =>
      let original := Some (FPred [{|cc:=c;cw:=w;csg:=signed;ca:=a;cb:=b|}] true) in
      match copied_zero_predicate st c w a b with
      | Some recovered => Some recovered
      | None => if Nat.eqb w 1 && Z.eqb (wd a) 1 then
       match c,b with CEq,OImm 0 =>
         match opposite_predicate (get st a) with Some f => Some f | None => original end
       | _,_ => original end else original end
  | IAndp _ _ a b => match known_bit st a,known_bit st b with
    | Some false,_ | _,Some false => bit_fact false
    | Some true,_ => get st b | _,Some true => get st a
    | _,_ => pred_union (get st a) (get st b) end
  | IOrp _ _ a b => match known_bit st a,known_bit st b with
    | Some true,_ | _,Some true => bit_fact true
    | Some false,_ => get st b | _,Some false => get st a
    | _,_ => pred_disj (get st a) (get st b) end
  | IXorp _ _ a b => pred_common (get st a) (get st b)
  | IMinMax _ is_min signed w _ a b =>
      let w := Z.of_nat w in
      (* a minimum or maximum is one of its operands, so it has the lesser of
         their alignments *)
      let al := Z.min (Z.min (reg_align st a) (opnd_align st b)) w in
      match opnd_range st (OReg a) w signed, opnd_range st b w signed with
      | Some (a0, a1), Some (b0, b1) =>
          (* the ranges decide which operand the result is, as for Numba's
             max(k, 0) of a size k *)
          if (if is_min then a1 <=? b0 else b1 <=? a0) then Some (FForm (normalize st (opnd_form st (OReg a) w)) 0 None)
          else if (if is_min then b1 <=? a0 else a1 <=? b0) then Some (FForm (normalize st (opnd_form st b w)) 0 None)
          else if is_min then Some (FForm (range_form w (Z.min a0 b0) (Z.min a1 b1)) al None)
          else Some (FForm (range_form w (Z.max a0 b0) (Z.max a1 b1)) al None)
      | ra, rb =>
          (* a minimum is at most either operand, and at least the least
             value of its type *)
          if is_min then
            match (match ra with Some r => Some r | None => rb end) with
            | Some (_, y) => Some (FForm (range_form w (if signed then - 2 ^ (w - 1) else 0) y) al None)
            | None => None
            end
          else None
      end
  | IDiv _ rm sg w _ a b => div_fact st rm sg (Z.of_nat w) a b
  | _ => None
  end.

(* ------------------------------------------------------------------ *)
(* Assignment.                                                         *)

Definition mentions (f : form) (d : reg) : bool := existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f).

(* f with d's atoms replaced: d's new value is its old one plus the value
   of the step, a form over other registers, so the old value is the new
   one less the step; without a step, each atom by its range. *)
(* d := d + s moves d's value by s's value exactly: neither end wraps. *)
Definition step_exact (st : astate) (d : reg) (s : form) : bool :=
  match reg_range st d, form_range st s with
  | Some (x, y), Some (p, q) => (0 <=? x + p) && (y + q <? 2 ^ wd d)
  | _, _ => false
  end.

Definition forget_step (st : astate) (f : form) (d : reg) (step : option form) (acc : option form) (t : atom * Z)
  : option form :=
  match acc with
  | None => None
  | Some g =>
      let '((r, e), c) := t in
      if negb (Pos.eqb r d) then Some (fadd g (fatom (fm f) (r, e) c)) else
      let by_range := match atom_range st (r, e) with
                      | Some (x, y) => Some (fshift g (Z.min (x * c) (y * c)) (Z.max (x * c) (y * c)))
                      | None => None
                      end in
      let by_step := fun s => Some (fadd g (fadd (fatom (fm f) (r, e) c) (fscale s (- c) (fm f)))) in
      (* the step moves d's value by the step's value modulo 2^(wd d), which
         is enough for a form no wider than d, and exactly when neither end
         wraps *)
      match step, e with
      | Some s, EX | Some s, EZ => if (fm f <=? wd d) || step_exact st d s then by_step s else by_range
      (* a narrow register's sign extension is its value while the old and
         the new value lie below its sign bit, as for a 32-bit loop index *)
      | Some s, ES =>
          if fm f <=? wd d then by_step s else
          match reg_range st d, form_range st s with
          | Some (x, y), Some (p, q) =>
              if (wd d <? 64) && (0 <=? x) && (y <? 2 ^ (wd d - 1)) && (0 <=? x + p) && (y + q <? 2 ^ (wd d - 1))
              then by_step s else by_range
          | _, _ => by_range
          end
      | _, _ => by_range
      end
  end.

Definition forget_reg (st : astate) (f : form) (d : reg) (step : option form) : option form :=
  fold_left (forget_step st f d step) (ft f) (Some (range_form (fm f) (flo f) (fhi f))).

Definition cmp_mentions (k : cmpf) (d : reg) : bool :=
  Pos.eqb (ca k) d || match cb k with OReg r => Pos.eqb r d | OImm _ => false end.

(* The constant c of a step d := d + c, and whether d's reading sg moves
   by exactly c: the unsigned reading stays in [0, 2^w), and the signed
   reading stays in [-2^(w-1), 2^(w-1)), as a count that reaches 0 and
   wraps below it as an unsigned value does in its signed reading. *)
Definition const_step (step : option form) : option Z :=
  match step with
  (* a step is modulo 2^64, so a constant 2^64 - 1 is the step -1 *)
  | Some s => if null (ft s) && Z.eqb (flo s) (fhi s) then Some (rep 64 (flo s)) else None
  | None => None
  end.

Definition step_keeps (st : astate) (d : reg) (c : Z) (sg : bool) : bool :=
  let w := wd d in
  if sg then
    match sreg_range st d with
    | Some (x, y) => (- 2 ^ (w - 1) <=? x + c) && (y + c <? 2 ^ (w - 1))
    | None => false
    end
  else
    match reg_range st d with
    | Some (x, y) => (0 <=? x + c) && (y + c <? 2 ^ w)
    | None => false
    end.

(* A comparison of d with a constant, after d := d + c: the comparison of
   the new value with the constant plus c, which holds exactly when the old
   one did while the step keeps d's reading. *)
(* Integer comparisons survive a unit step by exchanging strict and
   non-strict bounds. Both directions are exact, so predicate negation also
   remains valid. This covers a compiler that tests a cursor before moving it. *)
Definition step_cmp (c : Z) (k : cmp) : option cmp :=
  match k with
  | CLt => if Z.eqb c 1 then Some CLe else None
  | CGe => if Z.eqb c 1 then Some CGt else None
  | CLe => if Z.eqb c (-1) then Some CLt else None
  | CGt => if Z.eqb c (-1) then Some CGe else None
  | _ => None
  end.

Definition cmp_step_direct (st : astate) (d : reg) (step : option form) (k : cmpf) : option cmpf :=
  if negb (cmp_mentions k d) then Some k else
  match const_step step, cb k with
  | Some c, OImm z =>
      let z' := extend (cw k) (csg k) z + c in
      if Pos.eqb (ca k) d && Z.eqb (Z.of_nat (cw k)) (wd d) && step_keeps st d c (csg k) &&
         (lo_ok (wd d) (csg k) <=? z') && (z' <=? hi_ok (wd d) (csg k))
      then Some {| cc := cc k; cw := cw k; csg := csg k; ca := d; cb := OImm z' |}
      else None
  | Some c, OReg r =>
      if Pos.eqb (ca k) d && negb (Pos.eqb r d) &&
         Z.eqb (Z.of_nat (cw k)) (wd d) && step_keeps st d c (csg k) then
        match step_cmp c (cc k) with
        | Some cc' => Some {| cc := cc'; cw := cw k; csg := csg k; ca := d; cb := OReg r |}
        | None => None
        end
      else None
  | _, _ => None
  end.

(* A saved comparison operand may still be an exact copy of the register
   being updated. Transport that equivalent comparison only when the same
   checked no-wrap and step conditions apply. Otherwise retain the saved
   operand: it remains useful even when the updated register can wrap. *)
Definition cmp_rebase (d : reg) (k : cmpf) : cmpf :=
  {| cc := cc k; cw := cw k; csg := csg k; ca := d; cb := cb k |}.

Definition same_exact_value (st : astate) (a b : reg) : bool :=
 match get st a,get st b with
 | Some (FForm f _ _),Some (FForm g _ _) =>
   Z.eqb (wd a) (wd b) && (wd a <=? fm f) && Z.eqb (fm f) (fm g) &&
   Z.eqb (flo f) (fhi f) && Z.eqb (flo g) (fhi g) &&
   Z.eqb (flo f) (flo g) && terms_eqb (ft f) (ft g)
 | _,_ => false end.
Definition comparison_alias (st : astate) (a d : reg) : bool :=
 match copy_of a (get st a) with
 | Some r => if Pos.eqb r d then true else same_exact_value st a d
 | None => same_exact_value st a d end.

Definition cmp_step (st : astate) (d : reg) (step : option form) (k : cmpf) : option cmpf :=
  match const_step step with
  | Some _ =>
      if comparison_alias st (ca k) d then
        match cmp_step_direct st d step (cmp_rebase d k) with
        | Some k' => Some k'
        | None => cmp_step_direct st d step k
        end
      else cmp_step_direct st d step k
  | None => cmp_step_direct st d step k
  end.

(* A term of an atom of register d. *)
Definition on_reg (d : reg) (t : atom * Z) : bool := Pos.eqb (fst (fst t)) d.

(* A constraint after d := d + c: its bounds shifted by c times the
   coefficients of d's atoms, while the step keeps each atom's reading. *)
Definition cons_step (st : astate) (d : reg) (step : option form) (k : constraint) : option constraint :=
  let '(ts, lo, hi) := k in
  let dts := filter (on_reg d) ts in
  match dts, const_step step with
  | [], _ => Some k
  | _, Some c =>
      if forallb (fun t : atom * Z => match snd (fst t) with
                                     | EX | EZ => step_keeps st d c false
                                     | ES => step_keeps st d c true
                                     | _ => false
                                     end) dts
      then let q := fold_left (fun acc (t : atom * Z) => acc + snd t) dts 0 in Some (ts, lo + q * c, hi + q * c)
      else None
  | _, None => None
  end.

Definition forget_fact (st : astate) (d : reg) (step : option form) (x : fact) : option fact :=
  match x with
  (* the value keeps its alignment and range; only its form changes *)
  | FForm f al vr => if mentions f d then
                       match forget_reg st f d step with
                       | Some g => Some (FForm g (value_align st f al) vr)
                       | None => None
                       end
                     else Some x
  | FPred cs exact =>
      let cs' := flat_map (fun k => match cmp_step st d step k with Some k' => [k'] | None => [] end) cs in
      match cs' with [] => None | _ => Some (FPred cs' (exact && Nat.eqb (length cs') (length cs))) end
  | FNeg cs exact =>
      let cs' := flat_map (fun k => match cmp_step st d step k with Some k' => [k'] | None => [] end) cs in
      match cs' with [] => None | _ => Some (FNeg cs' (exact && Nat.eqb (length cs') (length cs))) end
  | FAlign _ => Some x
  end.

(* Traverse each existing trie node once instead of rebuilding the map
   with one insertion per surviving fact. Empty branches are removed so
   repeated overwrites do not retain dead register paths. *)
Definition fact_node (l : PM.t fact) (value : option fact) (r : PM.t fact) : PM.t fact :=
  match l, value, r with
  | PM.Leaf _, None, PM.Leaf _ => PM.Leaf fact
  | _, _, _ => PM.Node l value r
  end.

Fixpoint filter_facts (f : fact -> option fact) (m : PM.t fact) : PM.t fact :=
  match m with
  | PM.Leaf _ => PM.Leaf fact
  | PM.Node l value r =>
      fact_node (filter_facts f l)
        (match value with Some x => f x | None => None end)
        (filter_facts f r)
  end.

Definition assigned_facts (st : astate) (d : reg) (step : option form) : PM.t fact :=
  filter_facts (forget_fact st d step) (PM.remove d (amap st)).

Definition assign (st : astate) (d : reg) (new : option fact) (step : option form) : astate :=
  let m := assigned_facts st d step in
  let m := match new with Some x => PM.add d x m | None => m end in
  {| amap := m;
     acons := flat_map (fun k => match cons_step st d step k with Some k' => [k'] | None => [] end) (acons st) |}.

(* ------------------------------------------------------------------ *)
(* Order and join of facts.                                            *)

Definition cmps_sub (a b : list cmpf) : bool := short_all (fun k => short_any (cmpf_eqb k) b) a.

Definition value_range (x : fact) : option range :=
  match x with
  | FForm _ _ (Some vr) => Some vr
  | FForm f _ None => if null (ft f) then Some (flo f, fhi f) else None
  | _ => None
  end.

(* Term t occurs in ts with the same coefficient. *)
Definition has_term (ts : terms) (t : atom * Z) : bool :=
  existsb (fun u => atom_eqb (fst u) (fst t) && Z.eqb (snd u) (snd t)) ts.

(* The terms of form a that form b also has, and the rest of a. *)
Definition common_terms (a b : form) : terms := filter (has_term (ft b)) (ft a).

Definition rest_form (a b : form) : form :=
  {| fm := fm a; ft := filter (fun u => negb (has_term (ft b) u)) (ft a); flo := flo a; fhi := fhi a |}.

(* A range of a form fact's value modulo its form's modulus: its value
   range, or, when it has none, the range of its form in st. *)
Definition fact_range_in (st : astate) (x : fact) : option range :=
  match value_range x with
  | Some r => Some r
  | None => match x with FForm f _ _ => form_range st f | _ => None end
  end.

(* Every value that x allows, g allows (st describes the state). *)
Definition fact_le (st : astate) (x g : fact) : bool :=
  match x, g with
  | FForm f al vr, FForm h al' vr' =>
      let sub := fun f h => terms_eqb (ft f) (ft h) && (fm h <=? fm f) && (flo h <=? flo f) && (fhi f <=? fhi h) in
      (* Eligibility is cheap; defer recursive ranges until it holds. *)
      let in_range := fun (_ : unit) =>
        if null (ft h) then
          if match vr with Some (a,b) => (flo h <=? a) && (b <=? fhi h) | None => false end
          then true else if fm h <=? fm f then
            match form_range st f with Some(a,b)=>(flo h <=? a) && (b <=? fhi h)|None=>false end
          else false
        else false in
      let shared := fun (_ : unit) =>
        if negb (null (ft h)) && terms_eqb (common_terms f h) (ft h) && (fm h <=? fm f)
        then match form_range st (rest_form f h) with
             | Some(a,b)=>(flo h <=? a) && (b <=? fhi h)|None=>false end
        else false in
      if (0 <=? fm h) && (fm h <=? 64) then
        if (if (if (if sub f h then true else sub (normalize st f) (normalize_exact st h))
                   then true else in_range tt) then true else shared tt) then
          if al' <=? value_align st f al then
            match vr' with None=>true|Some(a,b)=>match vr with Some(c,d)=>(a<=?c)&&(d<=?b)|None=>false end end
          else false
        else false
      else false
  | FAlign k, FAlign k' => k' <=? k
  | FForm f al _, FAlign k => if Z.eqb (fm f) 64 then k <=? value_align st (normalize st f) al else false
  | FPred cs e, FPred cs' e' | FNeg cs e, FNeg cs' e' =>
      if cmps_sub cs' cs then if negb e' then true else if e then cmps_sub cs cs' else false else false
  | _, _ => false
  end.

Lemma fact_le_same : forall st x g, fact_le st x g =
match x, g with
  | FForm f al vr, FForm h al' vr' =>
      let sub := fun f h => terms_eqb (ft f) (ft h) && (fm h <=? fm f) && (flo h <=? flo f) && (fhi f <=? fhi h) in
      (* a value range of x, or the range of its form, within g's constant
         range *)
      let in_range := fun (_ : unit) => null (ft h) &&
                      (match vr with Some (a, b) => (flo h <=? a) && (b <=? fhi h) | None => false end ||
                       (fm h <=? fm f) &&
                       match form_range st f with Some (a, b) => (flo h <=? a) && (b <=? fhi h) | None => false end) in
      (* g's terms are the terms of x that g shares, and the range of the
         rest of x lies in g's *)
      let shared := fun (_ : unit) => negb (null (ft h)) && terms_eqb (common_terms f h) (ft h) && (fm h <=? fm f) &&
                    match form_range st (rest_form f h) with
                    | Some (a, b) => (flo h <=? a) && (b <=? fhi h)
                    | None => false
                    end in
      (0 <=? fm h) && (fm h <=? 64) &&
      (if (if (if sub f h then true else sub (normalize st f) (normalize_exact st h))
                then true else in_range tt) then true else shared tt) &&
      (al' <=? value_align st f al) &&
      match vr' with None => true | Some (a, b) => match vr with Some (c, d) => (a <=? c) && (d <=? b) | None => false end end
  | FAlign k, FAlign k' => k' <=? k
  | FForm f al _, FAlign k => (Z.eqb (fm f) 64) && (k <=? value_align st (normalize st f) al)
  | FPred cs e, FPred cs' e' | FNeg cs e, FNeg cs' e' => cmps_sub cs' cs && (negb e' || (e && cmps_sub cs cs'))
  | _, _ => false
  end.
Proof.
 assert (Hand : forall a b : bool, (if a then b else false) = a && b).
 { intros [] b; reflexivity. }
 assert (Hor : forall a b : bool, (if a then true else b) = a || b).
 { intros [] b; reflexivity. }
 intros st x g. unfold fact_le. destruct x, g; cbv beta iota zeta;
 rewrite ?Hand, ?Hor; repeat rewrite Hand; repeat rewrite Hor;
 repeat rewrite andb_assoc; reflexivity.
Qed.

Definition fact_le_o (st : astate) (x g : option fact) : bool :=
  match g with None => true | Some g => match x with Some x => fact_le st x g | None => false end end.

(* The join of forms a (in sf) and b (in sg) that share terms, such as the
   base of a pointer that a loop advances: the shared terms, and the hull
   of the ranges of the rest. *)
Definition join_common (sf sg : astate) (a b : form) (al : Z) : option fact :=
  if negb (null (common_terms a b)) && terms_eqb (common_terms a b) (common_terms b a) then
    match form_range sf (rest_form a b), form_range sg (rest_form b a) with
    | Some (p, q), Some (r, s) =>
        Some (FForm {| fm := Z.max 0 (Z.min (fm a) (fm b)); ft := common_terms a b;
                       flo := Z.min p r; fhi := Z.max q s |} al None)
    | _, _ => None
    end
  else None.

(* An upper bound of two facts of one register, the first described by
   state sf and the second by sg. *)
Definition join_fact (sf sg : astate) (x y : option fact) : option fact :=
  match x, y with
  | None, _ | _, None => None
  | Some f, Some g =>
      if fact_le sf f g then Some g else if fact_le sg g f then Some f else
      match f, g with
      | FForm a al vr, FForm b bl vr' =>
          if terms_eqb (ft a) (ft b) then
            Some (FForm {| fm := Z.max 0 (Z.min (fm a) (fm b)); ft := ft a; flo := Z.min (flo a) (flo b); fhi := Z.max (fhi a) (fhi b) |}
                        (Z.min (value_align sf a al) (value_align sg b bl)) None)
          else
          (* the hull of the two ranges, as when a probe loop joins
             h & mask with (slot + 1) & mask; without them, the terms the
             forms share, as for a pointer the loop advances *)
          match fact_range_in sf f, fact_range_in sg g with
          | Some (p, q), Some (r, s) =>
              Some (FForm (range_form (Z.max 0 (Z.min (fm a) (fm b))) (Z.min p r) (Z.max q s))
                          (Z.min (value_align sf a al) (value_align sg b bl)) None)
          | _, _ => join_common sf sg a b (Z.min (value_align sf a al) (value_align sg b bl))
          end
      | FPred cs e, FPred cs' e' =>
          match filter (fun k => existsb (cmpf_eqb k) cs') cs with
          | [] => None
          | c => Some (FPred c (e && e' && cmps_sub cs cs' && cmps_sub cs' cs))
          end
      | FNeg cs e, FNeg cs' e' =>
          match filter (fun k => existsb (cmpf_eqb k) cs') cs with
          | [] => None
          | c => Some (FNeg c (e && e' && cmps_sub cs cs' && cmps_sub cs' cs))
          end
      | FAlign k, FAlign k' => let k := Z.min k k' in if 0 <? k then Some (FAlign k) else None
      | _, _ => None
      end
  end.

(* ------------------------------------------------------------------ *)
(* Transfer.                                                           *)

Definition narrow_step (st : astate) (d : reg) (f : form) (g : guard) : option form :=
  (* d := d + step, where the step is the rest of the form: modulo 2^64 for
     a 64-bit register, whose step may be a register as in a grid-stride
     loop; a constant at d's width, without wrapping, for a narrow one *)
  match g, filter (fun t => Pos.eqb (fst (fst t)) d) (ft f) with
  | None, [((_, e), 1)] =>
      let rest := filter (fun t => negb (Pos.eqb (fst (fst t)) d)) (ft f) in
      let w := wd d in
      match e with
      | EX | ES => if Z.eqb (fm f) 64 && Z.eqb w 64 then Some {| fm := 64; ft := rest; flo := flo f; fhi := fhi f |}
              else None
      | EZ => if null rest && Z.eqb (flo f) (fhi f) && (w <? 64) && (w <=? fm f) then
                let s := rep w (flo f) in
                match reg_range st d with
                | Some (x, y) => if (0 <=? x + s) && (y + s <? 2 ^ w) then Some (fconst 64 s) else None
                | None => None
                end
              else None
      | _ => None
      end
  | _, _ => None
  end.

(* A step keeps any alignment proved for its complete affine value, including
   symbolic grid strides. A non-singleton residual range has no such fact. *)
Definition step_align (st : astate) (s : form) : Z :=
  if Z.eqb (flo s) (fhi s) then form_align st s else 0.

Definition step_fact (st : astate) (d : reg) (s : form) : option fact :=
  match get st d with
  | Some (FAlign k) => let k := Z.min (Z.min k (step_align st s)) (wd d) in if 0 <? k then Some (FAlign k) else None
  | None => match reg_range st d, form_range st s with
            | Some (x, y), Some (p, q) => Some (FForm (range_form (wd d) (x + p) (y + q)) 0 None)
            | _, _ => None
            end
  | Some (FForm f al vr) as x =>
      if interval_only x || null (ft f) then
        match form_range st s with
        | Some (p, q) =>
            let '(lo, hi) := match vr with Some r => r | None => (flo f, fhi f) end in
            Some (FForm (range_form (match vr with Some _ => wd d | None => Z.min (fm f) (wd d) end) (lo + p) (hi + q))
                        (Z.min (Z.min (reg_align st d) (step_align st s)) (wd d)) None)
        | None => None
        end
      else None
  | _ => None
  end.

(* The fact of d after an instruction with guard g: when g may fail, the
   join of the new fact and d's old one, unless the new fact also holds of
   d's old value. *)
Definition guard_join (st : astate) (d : reg) (g : guard) (total : bool) (x : option fact) : option fact :=
  match g with Some _ => if total then x else join_fact st st (get st d) x | None => x end.

(* The fact placed on d, which must not read d's old value. *)
Definition place (st : astate) (d : reg) (x : option fact) : option fact :=
  match x with Some y => forget_fact st d None y | None => None end.

(* d := f where d's old fact is the exact form h and f is h plus a
   constant: a constant step of d modulo 2^(wd d), although f does not
   mention d, as when a counter's form is a size and the counter counts
   down. *)
Definition form_step (st : astate) (d : reg) (f : form) (g : guard) : option form :=
  match g, get st d with
  | None, Some (FForm h _ _) =>
      if Z.eqb (fm f) (fm h) && (wd d <=? fm f) && Z.eqb (flo f) (fhi f) && Z.eqb (flo h) (fhi h) &&
         terms_eqb (ft f) (ft h)
      then Some (fconst 64 (rep (wd d) (flo f - flo h)))
      else None
  | _, _ => None
  end.

(* The state after an instruction with guard g writes d, given the fact of
   the value it writes and whether that fact also holds of d's old value. *)
(* An exact modular constant has a unique centered representative. This
   keeps a descending machine-width counter near zero when its instruction
   encodes -1 as the all-ones unsigned immediate. *)
Definition canonical_fact (d : reg) (new : option fact) : option fact :=
  match new with
    | Some (FForm f al vr) =>
       if null (ft f) && Z.eqb (flo f) (fhi f) && (wd d <=? fm f) then
        Some (FForm (fconst (wd d) (rep (wd d) (flo f))) al vr)
       else new
    | _ => new end.

Definition transfer_dest_raw (st : astate) (d : reg) (g : guard) (new : option fact) (total : bool) : astate :=
  match new with
  | Some (FForm f _ _) as x =>
      if mentions f d then
        match narrow_step st d f g with
        | Some s => assign st d (step_fact st d s) (Some s)
        (* d's old value in the form is read as its range *)
        | None => assign st d (place st d (guard_join st d g total x)) None
        end
      else match form_step st d f g with
           | Some s => assign st d (place st d (guard_join st d g total x)) (Some s)
           | None => assign st d (place st d (guard_join st d g total x)) None
           end
  | x => assign st d (place st d (guard_join st d g total x)) None
  end.

Definition transfer_dest (st : astate) (d : reg) (g : guard) (new : option fact) (total : bool) : astate :=
  transfer_dest_raw st d g (canonical_fact d new) total.

(* ------------------------------------------------------------------ *)
(* Access checks.                                                      *)

Definition guard_cmps (st : astate) (g : guard) : list cmpf :=
  match g with
  | Some (false, p) => match get st p with Some (FPred cs _) => cs | _ => [] end
  | _ => []
  end.

(* A range for rest modulo 2^64 that one of the guard's 64-bit comparisons
   gives: an equality of two registers whose difference is rest plus an
   interval, or an unsigned upper bound of a register that is rest plus an
   interval. Every value of rest is a multiple of 2^al. *)
Definition guard_bound (st : astate) (g : guard) (rest : form) (al : Z) : option range :=
  let from (k : cmpf) : option range :=
    if negb (Nat.eqb (cw k) 64) then None else
    let fx := normalize st (as_form st (ca k) EX) in
    match cc k, cb k with
    | CEq, OReg y =>
        let D := fadd (fadd fx (fscale (normalize st (as_form st y EX)) (-1) 64)) (fscale rest (-1) 64) in
        if null (ft D) && Z.eqb (fm D) 64 then Some (round al (- fhi D, - flo D)) else None
    | CLt, OImm z | CLe, OImm z =>
        if csg k then None else
        let c := z - match cc k with CLt => 1 | _ => 0 end in
        let D := fadd fx (fscale rest (-1) 64) in
        if null (ft D) && Z.eqb (fm D) 64 && (0 <=? c) && (c <? 2 ^ 64) then Some (round al (- fhi D, c - flo D)) else None
    | _, _ => None
    end in
  fold_left (fun acc k => match acc with Some r => Some r | None => from k end) (guard_cmps st g) None.

(* The alignment of rest, the address a + off less base P: its form's, or
   the alignment the three have in common. *)
Definition rest_align (st : astate) (a : reg) (off : Z) (P : reg) (rest : form) : Z :=
  Z.max (form_align st rest) (Z.min (Z.min (reg_align st a) (v2 off)) (Z.min (reg_align st P) 64)).

Definition base_params (A : form) : list reg :=
  flat_map (fun t => let '((r, e), c) := t in
              match e with EX => if Z.eqb c 1 && is_some (role_of r) then [r] else [] | _ => [] end) (ft A).

(* Whether the access of kind k at address a + off and width bytes, in state
   st before an instruction with guard g, stays in its region. *)
(* The width bytes at rest lie inside a span of extent ext, given a range
   [x, y] of a representative of rest modulo 2^64: x is at least 0, and
   ext - rest, read through the constraints that relate the kernel's
   registers to the size registers, is at least width. The representatives
   of rest and of ext - rest sum to less than 2^64, so they sum to ext. *)
Definition inside (sg : astate) (r : option range) (rest ext : form) (width : Z) : bool :=
  match r with
  | None => false
  | Some (x, y) =>
      let direct := null (ft ext) && (y + width <=? flo ext) in
      let D := fadd ext (fscale rest (-1) 64) in
      let through := match form_range_near sg D, form_range_near sg ext with
                     | Some (xd, yd), Some (xe, ye) =>
                         (width <=? xd) && (y + yd <? 2 ^ 64) && (0 <=? xe) && (ye <? 2 ^ 64)
                     | _, _ => false
                     end in
      (0 <=? x) && (direct || through)
  end.

(* ------------------------------------------------------------------ *)
(* The constraints of an access check.                                 *)

(* One atom for each value of a register: EX and EZ read the same value,
   and so does ES when the register's range lies below its sign bit, as for
   a 32-bit offset that one instruction reads zero-extended and another
   sign-extended. *)
Definition canon_atom_with (ranges : rfuns) (x : atom) : atom :=
  let '(r, e) := x in
  match e with
  | EX | EZ => (r, EX)
  | ES => match f_rr ranges r with
          | Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd r - 1)) then (r, EX) else x
          | None => x
          end
  | _ => x
  end.

Definition canon_terms_with (ranges : rfuns) (ts : terms) : terms :=
  sorted_terms (map (fun t => (canon_atom_with ranges (fst t), snd t)) ts).

Definition canon_cons_with (ranges : rfuns) (k : constraint) : constraint :=
  let '(ts, lo, hi) := k in (canon_terms_with ranges ts, lo, hi).

Definition canon_atom (st : astate) (x : atom) : atom :=
  canon_atom_with (rft_on (cons_index (acons st)) st [fst x]) x.
Definition canon_terms (st : astate) (ts : terms) : terms :=
  canon_terms_with (rft_on (cons_index (acons st)) st (term_regs ts)) ts.
Definition canon_cons (st : astate) (k : constraint) : constraint :=
  let '(ts, _, _) := k in canon_cons_with (rft_on (cons_index (acons st)) st (term_regs ts)) k.

(* a * k1 + b * k2, for a >= 0. *)
Definition cons_lin (a b : Z) (k1 k2 : constraint) : constraint :=
  let '(t1, l1, h1) := k1 in
  let '(t2, l2, h2) := k2 in
  (sorted_terms (tscale a t1 ++ tscale b t2),
   a * l1 + (if 0 <=? b then b * l2 else b * h2),
   a * h1 + (if 0 <=? b then b * h2 else b * l2)).

(* k1 and k2 combined so that atom x cancels. *)
Definition eliminate (x : atom) (k1 k2 : constraint) : constraint :=
  let '(t1, _, _) := k1 in
  let '(t2, _, _) := k2 in
  cons_lin (Z.abs (tget t2 x)) (- (Z.sgn (tget t1 x) * Z.sgn (tget t2 x) * Z.abs (tget t1 x))) k1 k2.

(* At most this many combinations are kept at each elimination. *)
Definition FM_CAP : nat := 64.

(* Build only the requested prefix. The callback receives the remaining
   budget, so a nested enumeration can stop before constructing discarded
   pairs. vm_compute eagerly constructs a full flat_map before firstn. *)
Fixpoint flat_map_budget {A B : Type} (n : nat) (f : nat -> A -> list B) (xs : list A) : list B :=
  match n, xs with
  | O, _ | _, [] => []
  | S _, x :: xs' =>
      let batch := f n x in
      batch ++ flat_map_budget (n - length batch) f xs'
  end.

(* One step of Fourier-Motzkin elimination: the constraints without x, and
   the combinations of two constraints with x in which x cancels. When an
   equality has x, each other constraint with x is combined with it alone:
   the projection is the same, and the constraints are no more, so none is
   lost to the cap. A register's exact form is such an equality. *)
Definition fm_step (x : atom) (cs : list constraint) : list constraint :=
  let has := fun k : constraint => let '(ts, _, _) := k in negb (Z.eqb (tget ts x) 0) in
  let with_x := filter has cs in
  filter (fun k => negb (has k)) cs ++
  match find (fun k : constraint => let '(_, lo, hi) := k in Z.eqb lo hi) with_x with
  | Some ke =>
      let '(te, _, _) := ke in
      flat_map (fun k => let '(t, _, _) := k in if terms_eqb t te then [] else [eliminate x k ke]) with_x
  | None =>
      flat_map_budget FM_CAP (fun n k1 => flat_map_budget n (fun m k2 =>
                       firstn m (let '(t1, _, _) := k1 in let '(t2, _, _) := k2 in
                       if terms_eqb t1 t2 then [] else [eliminate x k1 k2])) with_x) with_x
  end.

Fixpoint atoms_nodup (xs acc : list atom) : list atom :=
  match xs with
  | [] => rev acc
  | x :: xs' => if existsb (atom_eqb x) acc then atoms_nodup xs' acc else atoms_nodup xs' (x :: acc)
  end.

(* The value of a 64-bit register of atom x as its exact form, an equality
   that lets a projection eliminate the atom: the two agree modulo 2^64.
   Intersect the inferred register range with its unsigned machine range;
   when the register's range and the form's lie within 2^64 of each other
   they are equal. A loop that copies its position and then advances it has
   the copy as the new position less the step, and the bound of the copy
   then bounds that. *)
Definition form_eq_with (ranges : rfuns) (st : astate) (x : atom) : list constraint :=
  let '(r, e) := x in
  match e, get st r with
  | EX, Some (FForm f _ _) | EZ, Some (FForm f _ _) =>
      if Z.eqb (fm f) 64 && Z.eqb (wd r) 64 && Z.eqb (flo f) (fhi f) && negb (null (ft f)) then
        match inter (f_rr ranges r) (natural_range (r,EX)), form_range_with st (atom_range_with ranges) f with
        | Some (lo, hi), Some (p, q) =>
            if (- 2 ^ 64 <? lo - q) && (hi - p <? 2 ^ 64)
            then [(sorted_terms ([(x, 1)] ++ tscale (-1) (ft f)), flo f, flo f)] else []
        | _, _ => []
        end
      else []
  | _, _ => []
  end.

(* The atom x >> k of a register and the atom x of the register itself:
   x is 2^k times x >> k plus its low k bits, so
   0 <= x - 2^k (x >> k) <= 2^k - 1. A bound on x, as a mask's
   p - 32 words <= -1, then reaches p >> 5 through the elimination of p,
   which bounds the word index of a Bloom filter's bit p. The link is
   stated only where both atoms occur, so that a shifted hash whose
   register nothing bounds adds no constraint and no atom to eliminate. *)
Definition shift_links (xs : list atom) : list constraint :=
  flat_map (fun x => let '(r, e) := x in
    match e with
    | EBf a n =>
      if Z.eqb a 0 && (0 <=? n) &&
         (existsb (atom_eqb (r,EX)) xs || existsb (atom_eqb (r,EZ)) xs) then
        let raw := if existsb (atom_eqb (r,EX)) xs then EX else EZ in
        [(sorted_terms [((r,raw),1);((r,EBf 0 n),-1)],0,2^64-1)] else []
    | EShr k => if (0 <=? k) && existsb (atom_eqb (r, EX)) xs
                then [(sorted_terms [((r, EX), 1); ((r, EShr k), - 2 ^ k)], 0, 2 ^ k - 1)] else []
    | _ => []
    end) xs.

(* The range of f through the constraints projected onto f's atoms: the
   constraints, the ranges of their atoms, and the exact forms of their
   registers, all over canonical atoms, with every atom that f does not have
   eliminated, as j < k and k <= end - start and end <= C bound start + j by
   C - 1. *)
Definition orient_cons (k : constraint) : constraint :=
 let '(ts,lo,hi) := k in
 match ts with (_,c)::_ => if c <? 0 then (tscale (-1) ts,-hi,-lo) else k | [] => k end.
(* A constraint whose coefficients share a factor g > 1, divided by g with
   its bounds rounded inward: lo <= g * v <= hi gives
   ceil(lo / g) <= v <= floor(hi / g). The elimination of x from a mask's
   bound x - 32 words <= -1 and the shift link x - 32 (x >> 5) >= 0 leaves
   32 (x >> 5) - 32 words <= -1, and the projection of an access of width 4
   needs it as (x >> 5) - words <= -1: scaled by 32, the difference
   4 words - 4 (x >> 5) would only be proved at least 1. *)
Definition reduce_cons (k : constraint) : constraint :=
  let '(ts, lo, hi) := k in
  let g := fold_left (fun a t => Z.gcd a (snd t)) ts 0 in
  if g <=? 1 then k else (map (fun t => (fst t, snd t / g)) ts, - ((- lo) / g), hi / g).

Definition compact_projection (cs : list constraint) : list constraint :=
 fold_right (fun k cs => let '(ts,lo,hi) := reduce_cons (orient_cons k) in
   let same := fun k' => let '(ts',_,_) := k' in terms_eqb ts ts' in
   let '(lo,hi) := fold_left (fun acc k' => let '(l,h) := acc in let '(_,l',h') := k' in
      (Z.max l l',Z.min h h')) (filter same cs) (lo,hi) in
   (ts,lo,hi) :: filter (fun k' => negb (same k')) cs) [] cs.

(* Fixed launch parameters are boundaries between independent dynamic
   components. A query keeps the component of its non-fixed atoms and all
   fixed-only constraints. This is a deliberate weakening: an omitted
   component may constrain a fixed parameter, so equivalence is not assumed.
   Soundness follows from selecting a subset of already justified facts. *)
Definition projection_moving (ts : terms) : list atom :=
  filter (fun x => negb (is_fixed (fst x))) (map fst ts).

Definition projection_touches (xs : list atom) (ts : terms) : bool :=
  short_any (fun x => short_any (atom_eqb x) xs) (projection_moving ts).

Fixpoint projection_component (fuel : nat) (cs : list constraint)
    (xs : list atom) : list atom :=
  match fuel with
  | O => xs
  | S n =>
      let ys := fold_left (fun seen k => let '(ts, _, _) := k in
        if projection_touches seen ts
        then atoms_nodup (projection_moving ts) seen else seen) cs xs in
      if Nat.eqb (length xs) (length ys) then ys
      else projection_component n cs ys
  end.

(* A fixed-only goal can still depend on a dynamic bridge. Seed only
   components whose constraints share at least two distinct fixed goal atoms;
   a single common launch parameter remains a component boundary. *)
Definition projection_fixed_atoms (ts : terms) : list atom :=
  atoms_nodup (filter (fun x => is_fixed (fst x)) (map fst ts)) [].
Definition projection_shares_two (goals : list atom) (ts : terms) : bool :=
  Nat.leb 2 (length (filter (fun x => existsb (atom_eqb x) (map fst ts)) goals)).
Definition projection_fixed_seeds (ts : terms) (cs : list constraint) : list atom :=
  let goals := projection_fixed_atoms ts in
  atoms_nodup (flat_map (fun k => let '(ks, _, _) := k in
    if projection_shares_two goals ks then projection_moving ks else []) cs) [].
Definition projection_slice (ts : terms) (cs : list constraint) : list constraint :=
  let moving := projection_moving ts in
  let initial := if null moving then projection_fixed_seeds ts cs else moving in
  let xs := projection_component (length cs) cs initial in
  filter (fun k => let '(ks, _, _) := k in
    if null (projection_moving ks) then true else projection_touches xs ks) cs.

Definition projection_members (cs : list constraint) : list atom :=
  atoms_nodup (flat_map (fun k => let '(ts, _, _) := k in map fst ts) cs) [].

(* The constraints that a projection of f can read, found by registers
   before any range is computed: every constraint without a moving register,
   and every constraint on a moving register that the goal reaches through
   shared constraints and exact forms. Atoms are finer than registers and a
   register's form links it to its form's registers in both directions, so
   the component that projection_slice keeps lies inside these registers.
   Projection then reads only a subset of the justified constraints. *)
Definition moving_regs (ts : terms) : list reg :=
 filter (fun r => negb (is_fixed r)) (map (fun t => fst (fst t)) ts).
Definition reg_set_add (s : PM.t unit) (rs : list reg) : PM.t unit :=
 fold_left (fun s r => PM.add r tt s) rs s.
Definition reg_set_any (s : PM.t unit) (rs : list reg) : bool := existsb (fun r => PM.mem r s) rs.
Definition form_links (st : astate) : list (reg * list reg) :=
 PM.fold (fun r x acc => match x with
   | FForm h _ _ => (r, moving_regs (ft h)) :: acc | _ => acc end) (amap st) [].
Fixpoint near_regs (fuel : nat) (cs : list (list reg)) (links : list (reg * list reg))
    (s : PM.t unit) : PM.t unit :=
 match fuel with
 | O => s
 | S n =>
   let s1 := fold_left (fun s rs => if reg_set_any s rs then reg_set_add s rs else s) cs s in
   let s2 := fold_left (fun s l => let '(r, rs) := l in
     if PM.mem r s then reg_set_add s rs
     else if reg_set_any s rs then PM.add r tt s else s) links s1 in
   if Nat.eqb (PM.cardinal s2) (PM.cardinal s) then s else near_regs n cs links s2
 end.
Definition near_constraints (st : astate) (cs : list constraint) (goal : terms) : list constraint :=
 let regs := map (fun k => let '(ts,_,_) := k in moving_regs ts) cs in
 let moving := moving_regs goal in
 let fixed := filter is_fixed (map (fun t => fst (fst t)) goal) in
 let seeds := match moving with
   | [] => concat (map (fun k => let '(ts,_,_) := k in
        if existsb (fun t => existsb (Pos.eqb (fst (fst t))) fixed) ts then moving_regs ts else []) cs)
   | _ => moving end in
 let links := form_links st in
 let s := near_regs (S (length cs + length links)) regs links (reg_set_add (PM.empty unit) seeds) in
 filter (fun k => let '(ts,_,_) := k in let rs := moving_regs ts in null rs || reg_set_any s rs) cs.

Definition project_range_plain (ranges : rfuns) (st : astate) (extra : list constraint) (f : form) : option range :=
  let ar0 := atom_range_with ranges in
  let ar := fun x => inter (ar0 x) (natural_range x) in
  let g := {| fm := fm f; ft := canon_terms_with ranges (ft f); flo := flo f; fhi := fhi f |} in
  let cs := map (canon_cons_with ranges) (near_constraints st (acons st ++ extra) (ft f)) in
  let xs := atoms_nodup (map fst (ft g) ++ flat_map (fun k => let '(ts, _, _) := k in map fst ts) cs) [] in
  let units := flat_map (fun x => match ar x with Some (lo, hi) => [([(x, 1)], lo, hi)] | None => [] end) xs in
  let eqs := map (canon_cons_with ranges) (flat_map (form_eq_with ranges st) xs) in
  let links := shift_links xs in
  let expanded := atoms_nodup (flat_map (fun k => let '(ts,_,_) := k in map fst ts) eqs) xs in
  let kept := projection_slice (ft g) (cs ++ units ++ eqs ++ links) in
  let members := projection_members kept in
  let elim := filter (fun x => if tmem (ft g) x then false
                              else short_any (atom_eqb x) members) expanded in
  let cs' := fold_left (fun acc x => compact_projection (fm_step x acc)) elim
                       (compact_projection kept) in
  fold_left (fun acc k => inter acc (projected_cons_range ar g k)) cs' (plain_range ar g).

(* The registers of the goal and of the constraints near it, each once:
   the registers whose ranges a projection reads for its canonical atoms
   and its atoms' ranges. *)
Definition projection_regs (st : astate) (extra : list constraint) (f : form) : list reg :=
 map fst (PM.elements (reg_set_add (PM.empty unit)
   (term_regs (ft f) ++ flat_map (fun k => let '(ts, _, _) := k in term_regs ts)
                          (near_constraints st (acons st ++ extra) (ft f))))).
(* The projection with each of these registers' ranges computed once. *)
Definition project_range_with (ranges : rfuns) (st : astate) (extra : list constraint) (f : form) : option range :=
 project_range_plain (cache_rf ranges (projection_regs st extra f)) st extra f.

(* The terms of the goal and of every constraint. A projection reads the
   ranges of their registers and of the registers of these registers'
   exact forms, which reach_regs adds. *)
Definition projection_terms (st : astate) (extra : list constraint) (f : form) : terms :=
 ft f ++ flat_map (fun k => let '(ts,_,_) := k in ts) (acons st ++ extra).

Definition project_range (st : astate) (extra : list constraint) (f : form) : option range :=
 let ranges := rft_on (cons_index (acons st)) st (term_regs (projection_terms st extra f)) in
 project_range_with ranges st extra f.

(* The window a residue form's register is known to lie in: its value
   range, or, for a signed extension of low bits, the constraint on the
   register's signed reading that the conversion's transfer stated. *)
Definition residue_window (st : astate) (r : reg) (x : fact) : option (Z * Z * bool) :=
  match x with
  | FForm _ _ (Some (lo, hi)) => Some (lo, hi, false)
  | FForm _ _ None =>
      match find (fun k => let '(ts, _, _) := k in terms_eqb ts [((r, ES), 1)]) (acons st) with
      | Some (_, lo, hi) => Some (lo, hi, true)
      | None => None
      end
  | _ => None
  end.

(* The value of a 64-bit register whose fact is a residue form modulo 2^m,
   m < 64, as an equality with the form's integer value: the value, or its
   signed reading, lies in a window, and the projected range of the form's
   value lies within 2^m of every point of the window, so the two, which
   agree modulo 2^m, are equal; a signed reading that is at least 0 is the
   value. Triton's byte address chars + (begin + j).to(int64) is such a
   conversion, with begin + j below the end offset under the guard. *)
Definition residue_eq (project : form -> option range) (st : astate) (x : atom) : list constraint :=
  let '(r, e) := x in
  match e, get st r with
  | EX, Some (FForm h _ _ as y) =>
      if Z.eqb (wd r) 64 && (0 <? fm h) && (fm h <? 64) && Z.eqb (flo h) (fhi h) && negb (null (ft h)) then
        match residue_window st r y with
        | Some (lo, hi, signed) =>
            match project {| fm := 64; ft := ft h; flo := flo h; fhi := fhi h |} with
            | Some (p, q) =>
                if (q - lo <? 2 ^ fm h) && (hi - p <? 2 ^ fm h) && (negb signed || (0 <=? p))
                then [(sorted_terms ([((r, EX), 1)] ++ tscale (-1) (ft h)), flo h, flo h)] else []
            | None => []
            end
        | None => []
        end
      else []
  | _, _ => []
  end.

Definition check_range (st : astate) (f : form) : option range :=
  project_range st (flat_map (residue_eq (project_range st []) st) (atoms_nodup (map fst (ft f)) [])) f.

(* The width bytes at rest lie inside a span of extent ext, as `inside`
   decides with the ranges of check_range. *)
Definition inside_deep (sg : astate) (rest ext : form) (width : Z) : bool :=
  match check_range sg rest with
  | None => false
  | Some (x, y) =>
      let D := fadd ext (fscale rest (-1) 64) in
      match check_range sg D, form_range_near sg ext with
      | Some (xd, yd), Some (xe, ye) =>
          (0 <=? x) && (width <=? xd) && (y + yd <? 2 ^ 64) && (0 <=? xe) && (ye <? 2 ^ 64)
      | _, _ => false
      end
  end.

(* The span that the access of width bytes at a + off, under guard g in the
   refined state sg, provably stays inside: its base parameter, the span,
   and the offset form rest. *)
Definition span_of_access (sg : astate) (g : guard) (a : reg) (off width : Z) : option (reg * span * form) :=
  let A := fshift (normalize sg (as_form sg a EX)) off off in
  if negb (Z.eqb (fm A) 64) then None else
  match base_params A with
  | [P] =>
      let rest := mk 64 (without (ft A) (P, EX)) (flo A) (fhi A) in
      match role_of P with
      | Some sp =>
          if (if inside sg (form_range_near sg rest) rest (sp_ext sp) width then true else
              if inside sg (guard_bound sg g rest (rest_align sg a off P rest)) rest (sp_ext sp) width then true else
              inside_deep sg rest (sp_ext sp) width)
          then Some (P, sp, rest) else None
      | None => None
      end
  | _ => None
  end.

(* Each local variable's register and bytes. *)
Definition local_vars : list (reg * form) := map (fun e => (fst e, fconst 64 (fst (snd e)))) (lf_locals F).

(* Whether the terms ts have atom (x, EX) with coefficient 1. *)
Definition has_unit (ts : terms) (x : reg) : bool :=
  existsb (fun t => atom_eqb (fst t) (x, EX) && Z.eqb (snd t) 1) ts.

(* A term of register u with coefficient 1 whose value is u's value. *)
Definition unit_of (u : reg) (t : atom * Z) : bool :=
  Pos.eqb (fst (fst t)) u && match snd (fst t) with EX | EZ => true | _ => false end && Z.eqb (snd t) 1.

(* The form of register a's value, through its zero extension if it is
   narrower than 64 bits, which a form of its width lifts when it cannot
   wrap. *)
Definition value_ext (a : reg) : ext := if Z.eqb (wd a) 64 then EX else EZ.

(* The width bytes at address a + off lie inside a variable of vars: the
   address is the base register's value, for a generic address in a window,
   plus a variable's address, both fixed registers, plus a rest whose range
   lies in [0, bytes - width]. The variable lies in the window's first
   window_size bytes, and the window below 2^64, so the sum does not wrap. *)
Definition access_range (st : astate) (f : form) (lo hi : Z) : option range :=
  match bounded_form_range st f lo hi with
  | Some (p,q) as found => if (lo <=? p) && (q <=? hi) then found else check_range st f
  | None => check_range st f
  end.

Definition bound_form (f : form) : bool :=
  Z.eqb (fm f) 64 && Z.eqb (flo f) (fhi f) &&
  forallb (fun t => let '((r, e), _) := t in is_fixed r && match e with EX => true | _ => false end) (ft f).

(* Shared extents are launch-fixed affine forms; locals use constant forms.
   Constant and symbolic extents have separate arithmetic decision procedures. *)
Definition var_extent (st : astate) (rest bytes : form) (upper width al : Z) : bool :=
 if null (ft bytes) then
   match access_range st rest 0 (flo bytes - width) with
   | Some rg => let '(x,y) := round al rg in (upper + flo bytes <=? window_size) && (0 <=? x) && (y + width <=? flo bytes)
   | None => false end
 else match form_range st bytes with
   | Some (_,hi) => (upper + hi <=? window_size) && inside_deep st rest bytes width
   | None => false end.

Definition private_align (st : astate) (base : option reg) (a u : reg) (off : Z) : Z :=
 Z.min 64 (Z.min (Z.min (reg_align st a) (v2 off))
   (Z.min (reg_align st u) (match base with Some w => reg_align st w | None => 64 end))).
Definition var_of (vars : list (reg * form)) (base : option reg) (sg : astate) (a : reg) (off width : Z)
    : bool :=
  let A := fshift (normalize sg (as_form sg a (value_ext a))) off off in
  let T := match base with Some w => without (ft A) (w, EX) | None => ft A end in
  Z.eqb (fm A) 64 &&
  match base with
  | Some w =>
      is_fixed w && has_unit (ft A) w &&
      match lookup (lf_bounds F) w with
      | Some (lo, hi) => (0 <=? lo) && (hi + window_size <? 2 ^ 64)
      | None => false
      end
  | None => true
  end &&
  existsb (fun e =>
             let '(u, bytes) := e in
             is_fixed u &&
             match find (unit_of u) T with
             | Some (x, _) =>
                 match lookup (lf_bounds F) u with
                 | Some (lo, hi) => (0 <=? lo) && bound_form bytes &&
                     var_extent sg (mk 64 (without T x) (flo A) (fhi A)) bytes hi width (private_align sg base a u off)
                 | None => false end
             | None => false
             end) vars.

Definition is_load (k : mkind) : bool := match k with KLoad => true | _ => false end.

(* A value whose integer value vf's form gives lies in index range ix: the
   difference of vf and each bound, projected through the constraints, has
   the right sign. *)
Definition form_in_range (st : astate) (vf : form) (ix : index) : bool :=
  let diff := fun b : form =>
    {| fm := 64; ft := sorted_terms (ft vf ++ tscale (-1) (ft b)); flo := flo vf - flo b; fhi := fhi vf - flo b |} in
  match check_range st (diff (ix_lo ix)), check_range st (diff (ix_hi ix)) with
  | Some (x, _), Some (_, y) => (0 <=? x) && (y <=? 0)
  | _, _ => false
  end.

(* The value of operand o, read as an element of index range ix, lies in
   the range. *)
Definition in_range (st : astate) (o : opnd) (ix : index) : bool :=
  let w := 8 * ix_elem ix in
  let sg := ix_signed ix in
  (* a register wider than the element, as in `st.u32 [a], %rd`, stores its
     low bits, which are its value when that lies below the element's sign
     bit; the value is then read at the register's width *)
  let '(wr, sgr, same) := match o with
                          | OReg r => if Z.eqb (wd r) w then (w, sg, true) else (wd r, false, false)
                          | OImm _ => (w, sg, true)
                          end in
  let by_form :=
    match cmp_range st o wr sgr with
    | None => false
    | Some (x0, y0) => (same || ((0 <=? x0) && (y0 <? 2 ^ (w - 1)))) && form_in_range st (cmp_form st o wr sgr) ix
    end in
  (* a register of the element's width whose bits the constraints place
     below the sign bit: its signed and its unsigned reading are one value,
     as for a 32-bit row id that a guard on its zero extension bounds *)
  let by_bits :=
    match o with
    | OReg r =>
        let bits := {| fm := 64; ft := [((r, EZ), 1)]; flo := 0; fhi := 0 |} in
        Z.eqb (wd r) w &&
        match check_range st bits with
        | Some (x0, y0) => (0 <=? x0) && (y0 <? 2 ^ (w - 1)) && form_in_range st bits ix
        | None => false
        end
    | OImm _ => false
    end in
  by_form || by_bits.

(* An access of width bytes at offset alignment al into a span with an
   index range reads or writes one whole element, and a write stores a
   value in the range; a write into a flag span stores the number 1 into
   one whole word. *)
Definition value_ok (st : astate) (sp : span) (k : mkind) (width al : Z) (v : option opnd) : bool :=
  (negb (sp_flag sp) ||
   match k, v with
   | KStore, Some (OImm z) => Z.eqb width 4 && (2 <=? al) && Z.eqb z 1
   | _, _ => false
   end) &&
  match sp_index sp, k, v with
  | None, _, _ => true
  | Some ix, KLoad, _ => Z.eqb width (ix_elem ix) && (v2 (ix_elem ix) <=? al)
  | Some ix, KStore, Some o => Z.eqb width (ix_elem ix) && (v2 (ix_elem ix) <=? al) && in_range st o ix
  | Some _, _, _ => false
  end.

(* Whether the access of kind k at address a + off and width bytes, in state
   st before an instruction with guard g, stays in its region, or, for a
   generic address, inside a variable in a window: a shared variable, or for
   a load, a local variable. A generic store or atomic in local memory must
   lie in a live variable, which the check of the instruction decides. *)
(* Whether the access of width bytes at a + off in shared or local memory,
   in state st before an instruction with guard g, lies inside a variable of
   vars. *)
Definition private_ok (st : astate) (g : guard) (vars : list (reg * form)) (a : reg) (off width : Z) : bool :=
  match refine_guard st g with
  | None => true                            (* the guard never holds *)
  | Some sg => var_of vars None sg a off width
  end.

(* Only project conditional-load predicates connected to the access guard's
   operands. The selection is a precision/cost choice; every result is checked. *)
Definition operand_registers (st : astate) (r : reg) : list reg :=
 r :: match get st r with Some (FForm f _ _) => map (fun u => fst (fst u)) (ft f) | _ => [] end.
Definition guard_registers (st : astate) (p : reg) : list reg :=
 let cs := match get st p with Some (FPred cs _) | Some (FNeg cs _) => cs | _ => [] end in
 flat_map (fun k => operand_registers st (ca k) ++
   match cb k with OReg r=>operand_registers st r | _=>[] end) cs.
Definition guard_origins (st : astate) (p : reg) : list atom :=
 let wanted := guard_registers st p in
 if null wanted then [] else
 atoms_nodup (flat_map (fun k => let '(ts,_,_) := k in
   flat_map (fun u => let '(x,c) := u in
     if Z.eqb (wd (fst x)) 1 &&
        existsb (fun v => let '(y,q) := v in
          Z.eqb q 1 && (1 <? wd (fst y)) && existsb (Pos.eqb (fst y)) wanted &&
          Z.eqb (Z.abs c) (2 ^ wd (fst y)-1)) ts
     then [(fst x,EX)] else []) ts) (acons st)) [].
Definition refine_origins (st : astate) (xs : list atom) : option astate :=
 fold_left (fun acc x => let p := fst x in match acc with None => None | Some st =>
   match check_range st (atom64 p EX) with
   | Some (lo,hi) => if 1 <=? lo then refine st p true else if hi <=? 0 then refine st p false else Some st
   | None => Some st end end) xs (Some st).
(* Reuse the already-refined ordinary guard if provenance is required. *)
Definition access_guard (st refined : astate) (g : guard) : option astate :=
 match g with None => Some refined
 | Some (_,p) => refine_origins refined (guard_origins st p)
 end.

(* One complete access obligation in a state where the guard holds. *)
Definition access_state_ok (sg : astate) (g : guard) (k : mkind) (gen : bool)
    (a : reg) (off width : Z) (v : option opnd) : bool :=
  match span_of_access sg g a off width with
  | Some (P, sp, rest) => allows (sp_dir sp) k && value_ok sg sp k width (rest_align sg a off P rest) v
  | None =>
      gen && (var_of (lf_shared F) (Some (lf_shared_window F)) sg a off width ||
              is_load k && var_of local_vars (Some (lf_local_window F)) sg a off width)
  end.

(* First discharge every access obligation using the ordinary guard. Only
   an unresolved obligation needs conditional-load provenance projection. *)
Definition access_ok (st : astate) (g : guard) (k : mkind) (gen : bool) (a : reg) (off width : Z) (v : option opnd)
    : bool :=
  match refine_guard st g with
  | None => true
  | Some sg =>
      if access_state_ok sg g k gen a off width v then true else
      match g with
      | None => false
      | Some _ => match access_guard st sg g with
                  | None => true
                  | Some refined => access_state_ok refined g k gen a off width v
                  end
      end
  end.

(* ------------------------------------------------------------------ *)
(* Constraints that follow from a state.                               *)

(* The atom a register's copy reads: a register whose fact is a copy of
   another register has that register's value, as an integer, and the atom
   of its fact reads it. *)
Definition copy_atom (st : astate) (x : atom) : atom :=
  let (r, e) := x in
  match e, copy_of r (get st r), get st r with
  | EX, Some _, Some (FForm f _ _) | EZ, Some _, Some (FForm f _ _) =>
      match ft f with [(y, _)] => y | _ => x end
  | _, _, _ => x
  end.

Fixpoint copy_atom_n (n : nat) (st : astate) (x : atom) : atom :=
  match n with O => x | S n' => copy_atom_n n' st (copy_atom st x) end.

(* The terms read through copies, with the same integer sum: a loop's slot
   loaded from a local cell that holds home_slot's result. *)
Definition copy_terms (st : astate) (ts : terms) : terms := map (fun t => (copy_atom_n 4 st (fst t), snd t)) ts.

(* Constraint k follows from a: a has one of the same terms within its
   bounds, or the range of its terms in a lies within them. Only a register
   with a fact in a can hold the terms within. The terms are read through
   copies. The range comes from the terms, or from their normal form in a
   when the two ranges lie within one 2^64 window, so that the two sums are
   equal: the entry edge of a loop bounds p < end on the 32-bit p, and the
   loop bounds its 64-bit copy. A constraint of a also holds when its terms
   and k's, both read through a's copies, are the same, as a join proposes a
   constraint read through one side's copies. *)
(* Extension changes preserve the register/coefficient projection. Reject
   incompatible constraints before computing any signed atom ranges. *)
Definition value_terms (ts : terms) : terms :=
 sorted_terms (map (fun t => let '((r,e),c) := t in
   ((r,match e with EX | EZ | ES => EX | _ => e end),c)) ts).
(* Three-register sums cover coupled cursors without projecting every
   byte-stride or lane-layout constraint at every control-flow join. *)
Definition cursor_sum (ts : terms) : bool :=
 Nat.eqb (length ts) 3 && forallb (fun t => let '((r,e),c) := t in
   Z.eqb (wd r) 64 && Z.eqb (Z.abs c) 1 && match e with EX => true | _ => false end) ts.
Definition mixed_cursor_sum (ts : terms) : bool :=
 Nat.eqb (length ts) 3 && forallb (fun t => let '((r,e),c) := t in
   Z.eqb (Z.abs c) 1 && match e with EX => Z.eqb (wd r) 64 | EZ => (Z.eqb (wd r) 32 || Z.eqb (wd r) 64) | _ => false end) ts.

(* A single cursor relative to a bounded affine launch expression.
   Weighted thread/block terms need the same projection as scalar bounds. *)
Definition affine_cursor_bound (ts : terms) : bool :=
 let moving := filter (fun t => negb (is_fixed (fst (fst t)))) ts in
 match moving with
 | [((r,e),c)] => existsb (fun t => is_fixed (fst (fst t)) && Z.eqb (wd (fst (fst t))) 64) ts && Z.eqb (wd r) 64 && Z.eqb (Z.abs c) 1 &&
   forallb (fun t => let x := fst (fst t) in
     if is_fixed x then is_some (lookup (lf_bounds F) x) else Pos.eqb x r) ts &&
   existsb (fun t => is_fixed (fst (fst t)) && (1 <? Z.abs (snd t))) ts
 | _ => false end.

(* Two registers that no launch fixes, each with coefficient 1 or -1, one of
   them at a constant in state a: a loop counter at its start value against
   the bound its exit test compares, B - j >= 1 on the entry edge, where the
   state bounds B only through a constraint on another reading of B. *)
Definition pair_cursor (a : astate) (ts : terms) : bool :=
 Nat.eqb (length ts) 2 && forallb (fun t => let '((r,e),c) := t in
   negb (is_fixed r) && Z.eqb (Z.abs c) 1 &&
   match e with EX => Z.eqb (wd r) 64 | EZ => (Z.eqb (wd r) 32 || Z.eqb (wd r) 64) | _ => false end) ts &&
 existsb (fun t => match get a (fst (fst t)) with
                   | Some (FForm f _ _) => null (ft f) && Z.eqb (flo f) (fhi f)
                   | _ => false
                   end) ts.

(* Classify a constraint using only the shallow integer range. A disjoint
   range cannot satisfy the constraint; an overlapping range still requires
   the existing relational check. No normalization drops integer wraparound. *)
Definition range_constraint (lo hi : Z) (rg : option range) : option bool :=
  match rg with
  | Some (p,q) =>
      if (lo <=? p) && (q <=? hi) then Some true else
      if (hi <? p) || (q <? lo) then Some false else None
  | None => None
  end.

Definition near_constraint (a : astate) (ts : terms) (lo hi : Z) : option bool :=
  range_constraint lo hi (form_range_near a (form_of_terms ts)).

(* One constraint classifier serves scalar and batched implication checks.
   It returns either the exact cheap result or the remaining deep query. *)
Definition cons_finish (a:astate) (ts:terms) (lo hi:Z) (rg:option range) : bool :=
 match rg with None=>false | Some(p,q)=>
 if (lo <=? p) && (q <=? hi) then true else
 match form_range_near a (normalize a (form_of_terms ts)) with
 | Some(x,y)=>(lo <=? x) && (y <=? hi) && (Z.max q y - Z.min p x <? 2^64)
 | None=>false end end.

Definition cons_prepare (a : astate) (k : constraint) : bool + constraint :=
  let '(ts0, lo, hi) := k in
  let ts := copy_terms a ts0 in
  let read := fun (_ : unit) => canon_terms a ts in
  if short_any (fun k' => let '(ts', lo', hi') := k' in
                        (if (lo <=? lo') && (hi' <=? hi) then
                           if terms_eqb ts0 ts' then true else
                           if terms_eqb (value_terms ts) (value_terms (copy_terms a ts'))
                           then terms_eqb (read tt) (canon_terms a (copy_terms a ts')) else false
                         else false)
                        && (lo <=? lo') && (hi' <=? hi)) (acons a)
  then inl true else
  if negb (forallb (fun t => is_fixed (fst (fst t)) || is_some (get a (fst (fst t)))) ts0) then inl false else
  (* Reuse this exact shallow range when classification needs its normal form. *)
  let rg := form_range_near a (form_of_terms ts) in
  match range_constraint lo hi rg with
  | Some answer => inl answer
  | None =>
  if cursor_sum ts || (Z.eqb lo hi && mixed_cursor_sum ts) || affine_cursor_bound ts || pair_cursor a ts
  then inr (ts,lo,hi)
  else inl (cons_finish a ts lo hi rg)
  end.


Definition prepared_run (a:astate) (check:form->option range) (j:bool+constraint):bool :=
 match j with inl answer=>answer | inr(ts,lo,hi)=>cons_finish a ts lo hi (check(form_of_terms ts)) end.

Definition cons_holds (a : astate) (k : constraint) : bool :=
 prepared_run a (check_range a) (cons_prepare a k).

(* ------------------------------------------------------------------ *)
(* Loads.                                                              *)

(* The constraint lo <= v - f's terms <= hi on the value v of atom x, for a
   bound form f over size registers. *)
Definition bound_cons (x : atom) (f : form) (lo hi : Z) : constraint :=
  (sorted_terms ([(x, 1)] ++ tscale (-1) (ft f)), lo, hi).

(* An unguarded load of one element of an index input, after the load has
   written d in st1: the value, read as the input's integer type, lies in
   the input's range. Each bound becomes a constraint on d's atom, and d
   gets the hull of the range. A signed read is d's sign extension at d's
   own width, and d's value in a wider d when the range is not negative. *)
(* Only a whole element of a declared index input supplies value facts.
   If no such element has this width, avoid refining guards and resolving
   spans solely to rediscover that no index facts can be derived. The
   ordinary memory-access checker remains independent of this shortcut. *)
Definition index_width (bits : Z) : bool :=
 existsb (fun pair => match sp_index (snd pair) with
   | Some ix => Z.eqb bits (8 * ix_elem ix)
   | None => false end) (lf_roles F).

Definition index_load (st st1 : astate) (bits : Z) (d a : reg) (off : Z) : astate :=
  if negb (index_width bits) then st1 else
  match span_of_access st None a off (bits / 8) with
  | Some (P, sp, rest) =>
      match sp_index sp with
      | Some ix =>
          if negb (Z.eqb bits (8 * ix_elem ix)) || (rest_align st a off P rest <? v2 (ix_elem ix))
          then st1 else
          (* the ranges of the bounds, which close the far side of each
             one-sided constraint *)
          match form_range st (ix_lo ix), form_range st (ix_hi ix) with
          | Some (xl, _), Some (_, yh) =>
              (* a register wider than the element holds it zero-extended,
                 which is its signed value when the range is not negative,
                 as ld.u32 of an int32 offset into a 64-bit register *)
              if ix_signed ix && negb (Z.eqb (wd d) bits) && (xl <? 0) then st1 else
              let sg := ix_signed ix && Z.eqb (wd d) bits in
              let e := if sg then ES else if Z.eqb (wd d) 64 then EX else EZ in
              let atoms := if sg then [(d, ES)] else if Z.eqb (wd d) 64 then [(d, EX)] else [(d, EX); (d, EZ)] in
              let st2 := fold_left (fun s x =>
                           add_cons (add_cons s (bound_cons x (ix_lo ix) (flo (ix_lo ix)) (yh - xl + flo (ix_lo ix))))
                                    (bound_cons x (ix_hi ix) (xl - yh + flo (ix_hi ix)) (flo (ix_hi ix))))
                         atoms st1 in
              match tighten_reg st2 d e xl yh with Some s => s | None => st2 end
          | _, _ => st1
          end
      | None => st1
      end
  | None => st1
  end.

(* A constraint lo <= T <= hi that holds when a load under the guard of
   one-bit register p runs, where T lies in [lo0, hi0], its range in st, when
   the load does not run. With s the value that says the load runs, p, or
   1 - p for a negated guard, T + (hi0 - hi) s <= hi0 and
   T + (lo0 - lo) s >= lo0 hold either way, and a refinement by the guard,
   which puts p's value in every constraint, gives lo <= T <= hi back. *)
Definition guarded_cons (st : astate) (neg : bool) (p : reg) (k : constraint) : list constraint :=
  let '(ts, lo, hi) := k in
  match form_range_near st (form_of_terms ts) with
  | Some (lo0, hi0) =>
      let with_s := fun (c l h : Z) =>
        if neg then (sorted_terms (ts ++ [((p, EZ), - c)]), l - c, h - c)
        else (sorted_terms (ts ++ [((p, EZ), c)]), l, h) in
      [with_s (hi0 - hi) (Z.min (lo + (hi0 - hi)) lo0) hi0; with_s (lo0 - lo) lo0 (Z.max (hi + (lo0 - lo)) hi0)]
  | None => []
  end.

(* A load of an index input under guard g, which leaves d as it was when the
   guard fails, as Triton's masked load mov r, other; @m ld r, [a]. The load
   under the guard gives d the facts of index_load. d gets the join of that
   fact and its old one, and each constraint of the load on d that d's old
   value meets too, as the bounds of a row index whose other value is 0; for
   a guard of one bit, each other constraint of the load on d holds under the
   guard through guarded_cons, so that an access under the same guard, such
   as Triton's gather col[sel[i]] with both loads masked, is bounded. *)
Definition guarded_index_load (st : astate) (g : guard) (bits : Z) (d a : reg) (off : Z) : astate :=
  if negb (index_width bits) then transfer_dest st d g None false else
  match refine_guard st g with
  | None => transfer_dest st d g None false
  | Some sg =>
      let st2 := index_load sg (transfer_dest sg d None None false) bits d a off in
      let new := match get st2 d with
                 | Some (FForm f al vr) => if mentions f d then None else Some (FForm f al vr)
                 | _ => None
                 end in
      let st3 := fold_left (fun s k => if (if cons_mentions d k then cons_holds st k else false) then add_cons s k else s)
                           (acons st2) (transfer_dest st d g new false) in
      match g with
      | Some (neg, p) =>
          if Z.eqb (wd p) 1 then
            fold_left (fun s k => if (if cons_mentions d k then negb (cons_holds st k) else false)
                                  then fold_left add_cons (guarded_cons st neg p k) s else s)
                      (acons st2) st3
          else st3
      | None => st3
      end
  end.

(* ------------------------------------------------------------------ *)
(* Local memory: rule R8.                                              *)

(* The address form of a + off. *)
Definition addr_form (st : astate) (a : reg) (off : Z) : form :=
  fshift (normalize st (as_form st a EX)) off off.

Definition is_cell (r : reg) : bool := existsb (fun k => let '(p, _, _) := k in Pos.eqb p r) (lf_cells F).

(* The local variable and the offset of a local address form: the register
   of one local variable's address and a constant, so that the address is
   the variable's address plus the constant. *)
Definition local_at (A : form) : option (reg * Z) :=
  if negb (Z.eqb (fm A) 64 && Z.eqb (flo A) (fhi A) && (0 <=? flo A) && (flo A <=? 2 ^ 32)) then None else
  match ft A with
  | [((u, EX), 1)] | [((u, EZ), 1)] => if is_some (lookup (lf_locals F) u) then Some (u, flo A) else None
  | _ => None
  end.

(* The bytes of local variable u. *)
Definition local_size (u : reg) : Z := match lookup (lf_locals F) u with Some (s, _) => s | None => 0 end.

(* The local variable and the offset range of a local address form for an
   access of width bytes: the address is the variable's address plus an
   offset in the range. A constant offset, as local_at reads it, has the
   range of that constant; otherwise the form holds the register of one
   local variable's address with coefficient 1, and the range of the rest
   of the form, a register index the compiler did not unroll, comes from
   the state, as for the local load rule. *)
Definition local_span (st : astate) (A : form) (width : Z) : option (reg * Z * Z) :=
  match local_at A with
  | Some (u, c) => Some (u, c, c)
  | None =>
      if negb (Z.eqb (fm A) 64) then None else
      match find (fun t => let '((u, e), c) := t in
                           Z.eqb c 1 && match e with EX | EZ => true | _ => false end &&
                           is_some (lookup (lf_locals F) u)) (ft A) with
      | Some (x, _) =>
          let u := fst x in
          match access_range st (mk 64 (without (ft A) x) (flo A) (fhi A)) 0 (local_size u - width) with
          | Some (lo, hi) => if (0 <=? lo) && (hi <=? 2 ^ 32) then Some (u, lo, hi) else None
          | None => None
          end
      | None => None
      end
  end.

(* The local variable and the offset range of a generic address form whose
   width bytes lie in the window of local memory: the window's base, a
   local variable's address, and an offset in the range. *)
Definition local_of_generic (st : astate) (A : form) (width : Z) : option (reg * Z * Z) :=
  let w := lf_local_window F in
  if negb (Z.eqb (tget (ft A) (w, EX)) 1) then None else
  match reg_range st w,
        local_span st {| fm := fm A; ft := without (ft A) (w, EX); flo := flo A; fhi := fhi A |} width with
  | Some (_, hw), Some (u, lo, hi) =>
      match reg_range st u with
      | Some (_, hu) =>
          if (hw + window_size <? 2 ^ 64) && (hu + hi + width <=? window_size) then Some (u, lo, hi) else None
      | None => None
      end
  | _, _ => None
  end.

Inductive cell_op := CKeep | CPut | CForget.

(* What a store of width bytes at an offset in [lo, hi] of local variable
   u, at an instruction of frame fr, does to cell k: writes it, when the
   offset is the cell's, leaves it, when the span [lo, hi + width) misses
   the cell, or may overlap it. The live variables lie in disjoint memory,
   so a store whose span lies inside its live variable leaves the cells of
   the other live variables. *)
Definition cell_op_of (fr : frame) (u : reg) (lo hi width : Z) (k : reg * reg * Z) : cell_op :=
  let '(p, u', c') := k in
  if Pos.eqb u' u then
    if Z.eqb lo hi && Z.eqb c' lo && Z.eqb (wd p) (8 * width) then CPut
    else if (hi + width <=? c') || (c' + wd p / 8 <=? lo) then CKeep else CForget
  else if live (lf_locals F) fr u && live (lf_locals F) fr u' && (hi + width <=? local_size u) then CKeep
  else CForget.

(* A store of width bytes at a local address, when loc gives its variable
   and offset range, of value v if known, under guard g, at an instruction
   of frame fr, into st1: a cell it writes takes v's fact, a cell it may
   overlap is forgotten, and every other cell keeps its facts. A store at an
   unknown address forgets every cell. *)
Definition store_cell (g : guard) (loc : option (reg * Z * Z)) (width : Z) (v : option opnd) (fr : frame)
    (s : astate) (k : reg * reg * Z) : astate :=
  let '(p, _, _) := k in
  match loc with
  | None => assign s p None None
  | Some (u, lo, hi) =>
      match cell_op_of fr u lo hi width k with
      | CKeep => s
      | CPut =>
          match g, v with
          | None, Some (OReg r) =>
              if Z.eqb (wd r) (wd p) && negb (is_cell r) then transfer_dest s p None (value s (IMov None p r)) false
              else assign s p None None
          | None, Some (OImm z) => transfer_dest s p None (value s (IImm None p z)) false
          | _, _ => assign s p None None
          end
      | CForget => assign s p None None
      end
  end.

Definition store_cells (st1 : astate) (g : guard) (loc : option (reg * Z * Z)) (width : Z) (v : option opnd)
    (fr : frame) : astate :=
  fold_left (store_cell g loc width v fr) (lf_cells F) st1.

(* A local load of bits at a + off into d, under guard g, at an instruction
   of frame fr: the value of the cell of that address and width, if a live
   variable has one. *)
Definition load_cell (st : astate) (g : guard) (bits : nat) (d a : reg) (off : Z) (fr : frame) : astate :=
  match local_at (addr_form st a off) with
  | Some (u, c) =>
      if negb (live (lf_locals F) fr u) then transfer_dest st d g None false else
      match find (fun k => let '(p, u', c') := k in Pos.eqb u' u && Z.eqb c' c && Z.eqb (wd p) (Z.of_nat bits))
                 (lf_cells F) with
      | Some (p, _, _) =>
          if Z.eqb (wd d) (wd p) then transfer_dest st d g (value st (IMov g d p)) false
          else if wd p <? wd d then transfer_dest st d g (value st (ICvt g (wdn d) (wdn p) false d p)) false
          else transfer_dest st d g None false
      | None => transfer_dest st d g None false
      end
  | None => transfer_dest st d g None false
  end.

(* Entering the frame of variables vs forgets their cells. *)
Definition frame_cells (st : astate) (vs : list reg) : astate :=
  fold_left (fun s k => let '(p, u, _) := k in if existsb (Pos.eqb u) vs then assign s p None None else s)
            (lf_cells F) st.

(* A local store stays inside a live variable, over its whole offset range:
   other local memory may hold the compiler's spilled registers and call
   state. *)
Definition local_store_ok (loc : option (reg * Z * Z)) (width : Z) (fr : frame) : bool :=
  match loc with
  | Some (u, lo, hi) => live (lf_locals F) fr u && (0 <=? lo) && (hi + width <=? local_size u)
  | None => false
  end.

(* ------------------------------------------------------------------ *)
(* Transfer of an instruction.                                         *)

(* The fact of an operand that a selp writes at width w: the fact of a
   register of width w, with its range, or a constant. *)
Definition sel_fact (st : astate) (o : opnd) (w : Z) : option fact :=
  match o with
  | OImm z => Some (FForm (fconst w (z mod 2 ^ w)) 0 None)
  | OReg s =>
      if negb (Z.eqb (wd s) w) then None else
      (* the copy of s, which normalization reads through s's own fact, so
         that s's form and the relations of s's atom hold of the result *)
      let copy := if Z.eqb w 64 then fatom 64 (s, EX) 1 else fatom w (s, EZ) 1 in
      match get st s with
      | Some (FForm f al None) => Some (FForm copy al None)
      | Some x => Some x
      | None => Some (FForm copy 0 None)
      end
  end.

(* The fact of selp d, a, b, p: a's when p always holds, b's when it never
   does, and otherwise the join of both. Numba's negative-index term
   i + (i < 0 ? shape : 0) is then i when i is known to be nonnegative. *)
Definition sel_join (st : astate) (a b : opnd) (p : reg) (w : Z) : option fact :=
  match refine st p true, refine st p false with
  | None, Some sb => sel_fact sb b w
  | Some sa, None => sel_fact sa a w
  | None, None => None
  | Some sa, Some sb => join_fact sa sb (sel_fact sa a w) (sel_fact sb b w)
  end.

(* selp d, a, b, p with two constants and a one-bit p that may hold or
   not: d = b + (a - b) * p, which a later refinement of p makes a
   constant, as for Numba's i := i + (p ? 1 : 0). Otherwise sel_join. *)
Definition sel_value (st : astate) (a b : opnd) (p : reg) (w : Z) : option fact :=
  match refine st p true, refine st p false, a, b with
  | Some _, Some _, OImm za, OImm zb =>
      if Z.eqb (wd p) 1 then Some (FForm (mk w [((p, EZ), za - zb)] zb zb) 0 None) else sel_join st a b p w
  | _, _, _, _ => sel_join st a b p w
  end.

(* Constraint k with each atom of register r read from register r'. *)
Definition subst_cons (r r' : reg) (k : constraint) : constraint :=
  let '(ts, lo, hi) := k in
  (sorted_terms (map (fun t => let '((x, e), c) := t in if Pos.eqb x r then ((r', e), c) else t) ts), lo, hi).

(* Substitute a constant through every atom interpretation using the same
   evaluator as the semantics, including signed values, shifts and bitfields. *)
Fixpoint constant_terms (r : reg) (v : Z) (ts : terms) : terms * Z :=
 match ts with
 | [] => ([],0)
 | ((x,e),c)::rest =>
   let '(tail,z) := constant_terms r v rest in
   if Pos.eqb x r then (tail,c * atom_val wdn (fun _=>v) (r,e)+z)
   else (((x,e),c)::tail,z)
 end.
Definition const_cons (r : reg) (v : Z) (k : constraint) : constraint :=
 let '(ts,lo,hi) := k in
 let '(ts',c) := constant_terms r v ts in (ts',lo-c,hi-c).

(* selp d, a, b, p of a register of d's width and another operand, unguarded:
   a constraint on the register's atoms that also holds with the other
   operand in its place, a register of d's width or a constant, holds with d
   in its place, whichever operand d takes, as the hash slot of
   tl.where(active, (slot + 1) & mask, slot) or tl.where(active, slot, 0)
   stays below the capacity. A constraint may mention d when d is the source
   being substituted: every occurrence then denotes the selected value. *)
(* One ordered job constructor supplies scalar and prepared transport. *)
Definition branch_work (st:astate) (w:Z) (d:reg) (a b:opnd):list(constraint*constraint) :=
 let move:=fun r other=>if Z.eqb(wd r)w && Z.eqb(wd d)w then
 flat_map(fun k=>if cons_mentions r k && (negb(cons_mentions d k)||Pos.eqb r d)
 then [(subst_cons r d k,other k)] else []) (acons st) else [] in
 match a,b with
 | OReg ra,OReg rb=>if Z.eqb(wd rb)w then move ra (subst_cons ra rb) else []
 | OReg ra,OImm z=>move ra (const_cons ra (z mod 2^w))
 | OImm _,_=>[] end.

Definition work_apply (holds:constraint->bool) (jobs:list(constraint*constraint))(out:astate):astate :=
 fold_left(fun s job=>if holds(snd job) then add_cons s (fst job) else s) jobs out.

Definition branch_cons1 (st opposite st1 : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
 work_apply (cons_holds opposite) (branch_work st w d a b) st1.

(* Both alternatives may carry a useful relation, including the common
   cursor update selp d, d, midpoint, p. Do not privilege operand order. *)
Definition selection_jobs (st : astate) (w : Z) (d : reg) (a b : opnd)
 : list (constraint * constraint) :=
 let jobs := fun a b => match a,b with
   | OImm _,OReg _ => branch_work st w d b a
   | _,_ => branch_work st w d a b end in
 let first := jobs a b in
 match a,b with
 | OReg r,OReg s => if Pos.eqb r s then first else first ++ jobs b a
 | _,_ => first end.

Definition sel_cons (st st1 : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
 work_apply (cons_holds st) (selection_jobs st w d a b) st1.

(* A selection's alternatives are checked under their respective predicate
   facts. This retains conditional relations without assuming both paths. *)
Definition copy_operand (st : astate) (o : opnd) : opnd :=
  match o with
  | OReg r => match copy_atom_n 4 st (r, EX) with
              | (r', EX) | (r', EZ) => if Z.eqb (wd r') (wd r) then OReg r' else o
              | _ => o
              end
  | _ => o
  end.

(* A compiler may copy a bounded midpoint before selecting it as a cursor.
   Inspect its original register as well, under the branch where it is used.
   This is specific to selections; ordinary copies already move constraints. *)
Definition branch_jobs (st other : astate) (w : Z) (d : reg) (a b : opnd)
 : list (constraint * constraint) :=
 let direct := branch_work st w d a b in
 let a' := copy_operand st a in let b' := copy_operand other b in
 let same := fun x y => match x,y with
   | OReg r,OReg s => Pos.eqb r s
   | OImm x,OImm y => Z.eqb x y
   | _,_ => false end in
 if same a a' && same b b' then direct else direct ++ branch_work st w d a' b'.

Definition branch_cons (st other out : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
 work_apply (cons_holds other) (branch_jobs st other w d a b) out.

(* A comparison can read an otherwise unknown register through its complete
   machine-type range. Keep existing facts; this does not assert an input
   contract or an exact symbolic relation. *)
Definition type_operand (st : astate) (o : opnd) (sg : bool) : astate :=
 match o with OReg r =>
 if negb (Z.eqb (wd r) 64) then st else
 match get st r with None => put st r (FForm (range_form 64 (lo_ok 64 sg) (hi_ok 64 sg)) 0 None) | _ => st end
 | _ => st end.
Definition type_comparisons (st : astate) (p : reg) : astate :=
 let cs := match get st p with Some (FPred cs _) | Some (FNeg cs _) => cs | _ => [] end in
 fold_left (fun st k => if Nat.eqb (cw k) 64 then type_operand (type_operand st (OReg (ca k)) (csg k)) (cb k) (csg k) else st) cs st.
(* Refine only load predicates connected to this selection's comparison
   operands. Data-only masked loads must not trigger unrelated projections. *)
Definition refine_selection (st : astate) (p : reg) (truth : bool) : option astate :=
 let origins := guard_origins st p in
 if null origins then refine st p truth else
 match refine (type_comparisons st p) p truth with
 | Some s => refine_origins s origins | None => None end.

(* Refine a selection through its current destination when a saved
   comparison operand is proved equal to that destination. This recovers
   in-place bounds without discarding the saved value used by later writes. *)
Definition selection_operand (st : astate) (d : reg) (o : opnd) : opnd :=
 match o with OReg r => if comparison_alias st r d then OReg d else o | _ => o end.
Definition selection_comparison_raw (st : astate) (d : reg) (k : cmpf) : cmpf :=
 {| cc := cc k; cw := cw k; csg := csg k;
    ca := if comparison_alias st (ca k) d then d else ca k;
    cb := selection_operand st d (cb k) |}.
Definition reverse_comparison (c : cmp) : cmp :=
 match c with CLt => CGt | CLe => CGe | CGt => CLt | CGe => CLe | CEq => CEq | CNe => CNe end.
Definition selection_comparison (st : astate) (d : reg) (k : cmpf) : cmpf :=
 let k := selection_comparison_raw st d k in
 match cb k with
 | OReg r => if Pos.eqb r d && negb (Pos.eqb (ca k) d) then
     {|cc:=reverse_comparison (cc k);cw:=cw k;csg:=csg k;ca:=r;cb:=OReg (ca k)|}
   else k
 | _ => k end.
Definition selection_predicate (st : astate) (d p : reg) : astate :=
 match get st p with
 | Some (FPred cs exact) => put st p (FPred (map (selection_comparison st d) cs) exact)
 | Some (FNeg cs exact) => put st p (FNeg (map (selection_comparison st d) cs) exact)
 | _ => st end.

(* Prefer the overwritten operand for in-place selections; otherwise use
   a selected register so equivalent greater/less comparisons refine the
   same source before its relation is moved to the destination. *)
Definition selection_register (d : reg) (a b : opnd) : reg :=
 match a,b with
 | OReg r,OReg s => if Pos.eqb r d || Pos.eqb s d then d else r
 | OReg r,_ => r
 | _,OReg r => r
 | _,_ => d end.

(* A selection of a register and a constant can preserve a signed ordering
   even when the result crosses the unsigned wrap boundary. Check the
   ordering only in the opposite branch, where the constant is selected;
   choosing the register satisfies either ordering by equality. *)
Definition signed_selection_bound (other out : astate) (w : Z) (d r : reg) (z : Z) : astate :=
 if negb ((0 <? w) && Z.eqb w (wd d) && Z.eqb w (wd r)) || Pos.eqb r d then out else
 match cmp_range other (OReg r) w true with
 | Some (lo,hi) =>
   let enforce := fun c =>
      match refine_cmp out {|cc:=c;cw:=Z.to_nat w;csg:=true;ca:=d;cb:=OReg r|} with
      | Some s => s | None => out end in
   let c := extend (Z.to_nat w) true z in
   if c <=? lo then enforce CLe else if hi <=? c then enforce CGe else out
 | None => out end.

(* Selection refinement and signed ordering use one control sequence;
   transport implementations differ only in how they share immutable ranges. *)
Definition sel_refined_cons_with
 (same_side : astate -> astate -> Z -> reg -> opnd -> opnd -> astate)
 (branch : astate -> astate -> astate -> Z -> reg -> opnd -> opnd -> astate)
 (st st1 : astate) (w : Z) (d : reg) (a b : opnd) (p : reg) : astate :=
  let st := selection_predicate st (selection_register d a b) p in
  let base := same_side st st1 w d a b in
  match refine_selection st p true, refine_selection st p false with
  | Some sa, Some sb =>
      let out := branch sb sa (branch sa sb base w d a b) w d b a in
      match a,b with
      | OReg r,OImm z => signed_selection_bound sb out w d r z
      | OImm z,OReg r => signed_selection_bound sa out w d r z
      | _,_ => out end
  | _, _ => base
  end.

Definition sel_refined_cons_raw (st st1 : astate) (w : Z) (d : reg) (a b : opnd) (p : reg) : astate :=
 sel_refined_cons_with sel_cons branch_cons st st1 w d a b p.

(* Both alternatives are constants: branch constraint transport cannot add
   anything. Avoid projecting their predicates before returning that same state. *)
Definition sel_refined_cons (st st1 : astate) (w : Z) (d : reg) (a b : opnd) (p : reg) : astate :=
 match a,b with
 | OImm _,OImm _ => st1
 | _,_ => sel_refined_cons_raw st st1 w d a b p
 end.

Lemma sel_refined_cons_same : forall st st1 w d a b p,
 sel_refined_cons st st1 w d a b p = sel_refined_cons_raw st st1 w d a b p.
Proof using F.
 intros st st1 w d [r|za] [r'|zb] p; try reflexivity.
 unfold sel_refined_cons, sel_refined_cons_raw, sel_refined_cons_with.
 cbn [selection_register sel_cons selection_jobs branch_work work_apply].
 destruct (refine_selection (selection_predicate st d p) p true);
 destruct (refine_selection (selection_predicate st d p) p false); try reflexivity.
 unfold branch_cons, branch_jobs, copy_operand, branch_work, work_apply. cbn. rewrite !Z.eqb_refl. reflexivity.
Qed.

(* Prepare immutable opposite-state implications before creating shared range
   evaluators. Terminal cheap decisions keep their exact Boolean result; only
   pending deep projections share recursive ranges. Job order and every added
   constraint remain identical, including the direct and copied operands. *)
Definition prepare_work (opposite:astate)(jobs:list(constraint*constraint)) :=
 map(fun job=>(fst job,cons_prepare opposite(snd job))) jobs.

Definition prepared_apply (opposite:astate)(check:form->option range)
 (jobs:list(constraint*(bool+constraint)))(out:astate):astate :=
 fold_left(fun s job=>if prepared_run opposite check(snd job) then add_cons s(fst job) else s) jobs out.

Definition prepared_pending (jobs:list(constraint*(bool+constraint))) : bool :=
 existsb(fun job=>match snd job with inl _=>false|inr _=>true end) jobs.

Definition shared_check (opposite:astate):form->option range :=
 let ix:=cons_index (acons opposite) in
 let ranges:=tighten ix opposite (rf_with_cache ix true DEPTH opposite) in
 let project:=project_range_with ranges opposite in
 fun f=>project(flat_map(residue_eq(project[])opposite)(atoms_nodup(map fst(ft f))[]))f.

Definition run_prepared (opposite:astate)(jobs:list(constraint*constraint))(out:astate):astate :=
 let prepared:=prepare_work opposite jobs in
 if prepared_pending prepared then
 let check:=shared_check opposite in prepared_apply opposite check prepared out
 else prepared_apply opposite (check_range opposite) prepared out.

Definition branch_cons1_prepared (st other out:astate)(w:Z)(d:reg)(a b:opnd):astate :=
 run_prepared other (branch_work st w d a b) out.

Definition branch_cons_prepared (st other out : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
 run_prepared other (branch_jobs st other w d a b) out.

Definition sel_cons_prepared (st out : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
 run_prepared st (selection_jobs st w d a b) out.

Definition sel_refined_cons_prepared_raw (st st1 : astate) (w : Z) (d : reg) (a b : opnd) (p : reg) : astate :=
 sel_refined_cons_with sel_cons_prepared branch_cons_prepared st st1 w d a b p.

Definition sel_refined_cons_prepared (st st1 : astate) (w : Z) (d : reg) (a b : opnd) (p : reg) : astate :=
 match a,b with
 | OImm _,OImm _ => st1
 | _,_ => sel_refined_cons_prepared_raw st st1 w d a b p
 end.


(* Rule R4's relation: d := a & m at width w is at most m, so when m's
   form is exact, d's atom is at most that form. The relation is stated
   only when it says more than the ranges do: the form has terms and a
   bound below the width, as a mask capacity - 1 has. It is not stated when
   the form mentions d, which the instruction overwrites. *)
Definition and_cons1 (st : astate) (w : Z) (d m : reg) (s : astate) : astate :=
  match cmp_range st (OReg m) w false with
  | Some (_, y) =>
      if null (ft (cmp_form st (OReg m) w false)) || (2 ^ w - 1 <=? y)
         || existsb (fun t => Pos.eqb (fst (fst t)) d) (ft (cmp_form st (OReg m) w false)) then s
      else add_cons s (bound_cons (d, if Z.eqb (wd d) 64 then EX else EZ) (cmp_form st (OReg m) w false)
                                  (flo (cmp_form st (OReg m) w false) - y) (fhi (cmp_form st (OReg m) w false)))
  | None => s
  end.

(* Preserve the rounded-down value's relation independently of its interval. *)
Definition mask_const_cons (st out : astate) (w : Z) (d a : reg) (k : Z) : astate :=
 if negb (Z.eqb w 64) then out else
 (* Exact destination values already preserve the mask result; do not
    recompute ranges or add a redundant rounded-mask constraint. *)
 if match get out d with
      | Some (FForm f _ _) => Z.eqb (fm f) 64 && null (ft f) && Z.eqb (flo f) (fhi f)
      | _ => false end then out else
 match contiguous (k mod 2^w), cmp_range st (OReg a) w false with
 | Some(s,j), Some(lo,hi) =>
   let f:=cmp_form st (OReg a) w false in
   if (0 <? s) && (0 <=? lo) && (hi <? 2^j) && negb (null (ft f)) &&
      negb (existsb (fun t=>Pos.eqb (fst(fst t)) d) (ft f))
   then add_cons out (bound_cons (d,EX) f (flo f-(2^s-1)) (fhi f)) else out
 | _,_=>out end.

Definition and_cons (st st1 : astate) (w : Z) (d a rb : reg) : astate :=
  and_cons1 st w d rb (and_cons1 st w d a st1).

(* Rule R4's relation for a remainder: d := a rem b at width w, with b at
   least 1 and a not negative, is at most b - 1, so when b's form is exact,
   d's atom lies below that form, as a Bloom filter's bit p = h % (32 words)
   lies below 32 words. As for a mask, the relation is stated only when the
   form has terms and does not mention d. *)
(* The compared value's range, through the constraints (R7) when the form's
   own range does not fit the width: nvcc divides by the 32-bit copy of
   32 words when 32 words has no high bits, and the copy's form 32 words
   fits 32 bits only through the branch's bound on words. *)
Definition cmp_deep_range (st : astate) (o : opnd) (w : Z) (sg : bool) : option range :=
  match cmp_range st o w sg with
  | Some r => Some r
  | None =>
      match check_range st (cmp_form st o w sg) with
      | Some (x, y) => if (lo_ok w sg <=? x) && (y <=? hi_ok w sg) then Some (x, y) else None
      | None => None
      end
  end.

(* A symbolic remainder bound is useful to allocation checking when its
   divisor depends on an allocation extent or an index bound. Keep ordinary
   interval facts for all divisions, but avoid coupling unrelated scalar
   parameters into every loop invariant through extra remainder relations.
   Match registers across readings: the later projection still proves every
   conversion and relation used for an access. *)
Definition allocation_bound_terms : terms :=
  flat_map (fun p => let sp := snd p in ft (sp_ext sp) ++
     match sp_index sp with Some ix => ft (ix_lo ix) ++ ft (ix_hi ix) | None => [] end)
     (lf_roles F) ++ flat_map (fun p => ft (snd p)) (lf_shared F).

Definition allocation_related (f : form) : bool :=
  existsb (fun t => existsb (fun u => Pos.eqb (fst (fst t)) (fst (fst u))) allocation_bound_terms) (ft f).

Definition rem_cons (st st1 : astate) (sg : bool) (w : Z) (d a : reg) (b : opnd) : astate :=
  if negb (allocation_related (cmp_form st b w sg)) then st1 else
  match cmp_range st (OReg a) w sg, cmp_deep_range st b w sg with
  | Some (x, _), Some (k0, k1) =>
      let f := cmp_form st b w sg in
      if negb (0 <=? x) || negb (1 <=? k0) || null (ft f)
         || existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f) then st1
      else add_cons st1 (bound_cons (d, if Z.eqb (wd d) 64 then EX else EZ) f (flo f - k1) (fhi f - 1))
  | _, _ => st1
  end.

(* Nonnegative 64-bit halving preserves the quotient/remainder relation.
   Keep it over the source's exact form so it survives a temporary register.
   Restrict the extra relational state to halving: tracking unrelated hash,
   sign-bit and tile-count shifts increases fixpoint cost without helping
   the cursor bounds this rule maintains. Existing bit/range facts remain. *)
(* Preserve useful upper bounds against fixed scalar parameters before a later
   select overwrites the cursor that justified them. The projection is proved;
   selecting only nonpositive bounds keeps the relation stable at loop joins. *)
Definition project_upper (st : astate) (d r : reg) : astate :=
  let f := form_of_terms (sorted_terms [((d, EX), 1); ((r, EX), -1)]) in
  match check_range st f with
  | Some (lo, hi) =>
      if (- 2 ^ 64 <=? lo) && (hi <=? 0)
      then let st1 := add_cons st (ft f, - 2 ^ 64, 0) in
           if hi <? 0 then add_cons st1 (ft f, - 2 ^ 64, -1) else st1
      else st
  | None => st
  end.

Definition projected_constraint (st : astate) (k : constraint) : astate :=
 let '(ts,lo,hi) := k in
 match check_range st (form_of_terms ts) with
 | Some (l,h) => if (lo <=? l) && (h <=? hi) then add_cons st k else st
 | None => st end.
(* Materialized affine launch expressions, such as tid + block * tile,
   supply stable upper-bound targets in addition to individual size symbols. *)
Definition fixed_affine_targets (st : astate) : list form :=
 PM.fold (fun r x acc => match x with
 | FForm f _ _ =>
   if Z.eqb (fm f) 64 && Z.eqb (flo f) (fhi f) && Nat.leb 2 (length (ft f)) &&
      forallb (fun t => is_fixed (fst (fst t)) && is_some (lookup (lf_bounds F) (fst (fst t))) && (0 <? snd t)) (ft f)
   then if existsb (form_eqb f) acc then acc else f::acc
   else acc
 | _ => acc end) (amap st) [].
Definition project_affine_upper (st : astate) (d : reg) (f : form) : astate :=
 let ts := sorted_terms ([((d,EX),1)] ++ tscale (-1) (ft f)) in
 projected_constraint st (ts,flo f-2^64,flo f).

Definition project_fixed (st : astate) (d : reg) : astate :=
  let out := fold_left (fun s p => let '(r, _) := p in
               if Z.eqb (wd r) 64 && is_fixed r then project_upper s d r else s)
            (lf_bounds F) st in
  fold_left (fun out f => project_affine_upper out d f) (fixed_affine_targets st) out.

(* Materialize quotient relations only for arithmetic connected to fixed
   scalar bounds. Ordinary data arithmetic and thread bit fields already
   have interval/bit facts; projecting every such shift makes unrelated
   constraints grow throughout a kernel. This relevance test only selects
   which additional facts to propose; their soundness is checked below. *)
Definition fixed_scalar_term (t : atom * Z) : bool :=
  let r := fst (fst t) in is_fixed r && Z.eqb (wd r) 64 && is_some (lookup (lf_bounds F) r).

Definition bound_related (st : astate) (f : form) : bool :=
  existsb fixed_scalar_term (ft f) ||
  existsb (fun k => let '(ts, _, _) := k in
             existsb fixed_scalar_term ts &&
             existsb (fun t => let r := fst (fst t) in negb (is_fixed r) && cons_mentions r k) (ft f))
          (acons st).

Definition shift_cons (st st1 : astate) (sg : bool) (w : Z) (d a : reg) (b : opnd) : astate :=
  if negb (Z.eqb w 64) then st1 else
  match const_of st b 32 false with
  | None => st1
  | Some k =>
      if negb (Z.eqb (Z.min (k mod 2 ^ 32) w) 1) then st1 else
      let f := cmp_form st (OReg a) w sg in
      if negb (bound_related st f) then st1 else
      match cmp_range st (OReg a) w sg with
      | None => st1
      | Some (lo, hi) =>
          let k := Z.min (k mod 2 ^ 32) w in
          if (0 <=? lo) && Z.eqb (flo f) (fhi f) &&
             negb (existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f))
          then project_fixed (add_cons st1 (sorted_terms (ft f ++ [((d, EX), - (2 ^ k))]),
                            - flo f, 2 ^ k - 1 - flo f)) d
          else st1
      end
  end.

(* A state that no register values satisfy, because one of its constraints
   bounds the empty sum, which is 0, away from 0: the invariant of a block
   that no run reaches. *)
Definition bottom_state : astate := {| amap := PM.empty fact; acons := [([], 1, 0)] |}.

Definition is_bottom (st : astate) : bool :=
  existsb (fun k => let '(ts, lo, hi) := k in null ts && ((0 <? lo) || (hi <? 0))) (acons st).

(* Keep a selected value between ordered operands, including raw-register
   relations that survive changes to the forms used to represent them. *)
Definition select_between (d : reg) (low high : opnd) (out : astate) : astate :=
     let out := match low with OReg r => add_cons out (sorted_terms [((d,EX),1);((r,EX),-1)],0,2^64-1) | _ => out end in
     let out := match high with OReg r => add_cons out (sorted_terms [((d,EX),1);((r,EX),-1)],1-2^64,0) | _ => out end in
     let lower := {|cc:=CGe;cw:=64;csg:=false;ca:=d;cb:=low|} in
     let upper := {|cc:=CLe;cw:=64;csg:=false;ca:=d;cb:=high|} in
     match refine_cmp out lower with None => bottom_state | Some s =>
       match refine_cmp s upper with Some s' => s' | None => bottom_state end end.

Definition select_order (st out : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
 if negb (match a,b with OReg _,OImm _ | OImm _,OReg _ => true | _,_ => false end) then out else
 if negb (Z.eqb w 64) then out else
 let valid := fun o => match o with OReg r => negb (Pos.eqb r d) && Z.eqb (wd r) 64 | OImm _ => true end in
 if negb (valid a && valid b) then out else
 match cmp_range st a 64 false,cmp_range st b 64 false with
 | Some (al,ah),Some (bl,bh) =>
   let bound := select_between d in
   if ah <=? bl then bound a b out else if bh <=? al then bound b a out else out
 | _,_ => out end.

(* The result of min/max is bounded by each operand even when their
   intervals overlap. Refine the post-state with those two comparisons. *)
Definition minmax_cons (st : astate) (mn sg : bool) (w : nat) (d a : reg) (b : opnd) : astate :=
  let k := {| cc := if mn then CLe else CGe; cw := w; csg := sg; ca := d; cb := OReg a |} in
  let l := {| cc := if mn then CLe else CGe; cw := w; csg := sg; ca := d; cb := b |} in
  match refine_cmp st k with
  | None => bottom_state
  | Some s => match refine_cmp s l with Some s' => s' | None => bottom_state end
  end.

(* A symbolic min keeps a lower bound shared by both operands; max keeps
   a shared upper bound. Keep only constraints that establish this order,
   whose target atoms survive the destination assignment. *)
Definition minmax_common_cons (st out : astate) (mn sg : bool) (w : nat) (d a : reg) (b : opnd) : astate :=
 if negb (Nat.eqb w 64) then out else
 match cmp_range st (OReg a) 64 sg, cmp_range st b 64 sg with
 | Some (alo,_), Some (blo,_) =>
   if negb (0 <=? (if mn then Z.min alo blo else Z.max alo blo)) then out else
   let fa := cmp_form st (OReg a) 64 sg in let fb := cmp_form st b 64 sg in
   let terms := fold_right (fun t rest => if (0 <? snd t) && negb (existsb (fun u=>atom_eqb(fst t)(fst u) && Z.eqb(snd t)(snd u)) rest) then t::rest else rest)
                 [] (ft fa ++ ft fb ++ flat_map(fun k=>let '(ts,_,_):=k in ts)(acons st)) in
   let combined := filter (fun t => is_fixed (fst (fst t)) && (0 <? snd t)) terms in
   let targets := map (fun t => [t]) terms ++ (if Nat.leb 2 (length combined) then [combined] else []) in
   fold_left (fun out target =>
     if negb (forallb (fun t => negb (Pos.eqb (fst (fst t)) d)) target) then out else
     let subtract := fun f => {| fm := 64; ft := sorted_terms (ft f ++ tscale (-1) target); flo := flo f; fhi := fhi f |} in
     match check_range st (subtract fa), check_range st (subtract fb) with
     | Some (lo,hi), Some (lo',hi') =>
       if (if mn then 0 <=? Z.min lo lo' else Z.max hi hi' <=? 0) then
       add_cons out (sorted_terms ([((d,EX),1)] ++ tscale (-1) target),
                     (if mn then Z.min lo lo' else Z.max lo lo'),
                     (if mn then Z.min hi hi' else Z.max hi hi')) else out
     | _,_ => out end) targets out
 | _,_ => out end.

(* Preserve the existing constant-clamp path. Symbolic operands use the
   common-bound rule, which checks their relations before recording them. *)
Definition minmax_relative_cons (st out : astate) (mn sg : bool) (w : nat) (d a : reg) (b : opnd) : astate :=
 if negb (Nat.eqb w 64) then out else
 match cmp_range st (OReg a) 64 sg, cmp_range st b 64 sg with
 | Some (alo,_), Some (blo,_) =>
   if negb (0 <=? (if mn then Z.min alo blo else Z.max alo blo)) then out else
   let fa := cmp_form st (OReg a) 64 sg in let fb := cmp_form st b 64 sg in
   if negb (null (ft fa) || null (ft fb)) then minmax_common_cons st out mn sg w d a b else
   let terms := sorted_terms (ft fa ++ ft fb) in
   let combined := filter (fun t => is_fixed (fst (fst t)) && (0 <? snd t)) terms in
   let targets := map (fun t => [t]) terms ++ (if Nat.leb 2 (length combined) then [combined] else []) in
   fold_left (fun out target =>
     if negb (forallb (fun t => is_fixed (fst (fst t)) && negb (Pos.eqb (fst (fst t)) d)) target) then out else
     let subtract := fun f => {| fm := 64; ft := sorted_terms (ft f ++ tscale (-1) target); flo := flo f; fhi := fhi f |} in
     match form_range st (subtract fa), form_range st (subtract fb) with
     | Some (lo,hi), Some (lo',hi') =>
       add_cons out (sorted_terms ([((d,EX),1)] ++ tscale (-1) target),
                     (if mn then Z.min lo lo' else Z.max lo lo'),
                     (if mn then Z.min hi hi' else Z.max hi hi'))
     | _,_ => out end) targets out
 | _,_ => out end.

(* A narrow source's full affine form can be recovered when relational
   projection proves that its integer sum cannot wrap before extension. *)
Definition relational_cvt (st : astate) (to from : nat) (sg : bool) (d s : reg) : option fact :=
 let original := value st (ICvt None to from sg d s) in
 if negb (Nat.eqb to 64 && Nat.eqb from 32) then original else
 match get st s with
 | Some (FForm f _ _) =>
   if Z.eqb (wd s) 32 && (32 <=? fm f) && negb (null (ft f)) then
     match check_range st f with
     | Some (lo,hi) =>
       if (0 <=? lo) && (hi <? if sg then 2 ^ 31 else 2 ^ 32) then
         Some (FForm {| fm := 64; ft := ft f; flo := flo f; fhi := fhi f |} 0 None)
       else original
     | None => original end
   else original
 | _ => original end.

(* The signed reading of a sign extension of the low 32 bits of a 64-bit
   register lies in [-2^31, 2^31 - 1]; stated where the value's fact is the
   residue modulo 2^32, so that the check can read the register as the
   sum it extends. *)
Definition cvt_signed_bound (st out : astate) (to from : nat) (sg : bool) (d s : reg) : astate :=
 if Nat.eqb to 64 && Nat.eqb from 32 && sg && Z.eqb (wd s) 64 then
   match get out d with
   | Some (FForm f _ None) =>
       if Z.eqb (fm f) 32 && negb (null (ft f)) then add_cons out ([((d, ES), 1)], - 2 ^ 31, 2 ^ 31 - 1) else out
   | _ => out
   end
 else out.

(* A projected inequality can be reused by the selection checker once its
   complete integer range has been checked. *)
(* Prefer an existing register for a proposed step, so later writes to the
   step's inputs do not erase it. Both selected alternatives are checked;
   this choice proposes constraints and is not trusted for soundness. *)
Definition materialized_form (st : astate) (f : form) (d r : reg) : form :=
 match find (fun pair => let '(x,fact) := pair in
   negb (Pos.eqb x d || Pos.eqb x r) && Z.eqb (wd x) 64 &&
   match fact with FForm g _ _ => form_eqb f g | _ => false end) (PM.elements (amap st)) with
 | Some (x,_) => fatom 64 (x,EX) 1 | None => f end.
(* For d := select(d, d + step), propose the old cursor constraints with
   step subtracted. Check the retained and advanced alternatives separately. *)
(* Share the immutable input state's projected ranges across selected-step
   obligations. Empty relevant-constraint sets do not construct the cache. *)
Definition shared_check_cached (st:astate):form->option range :=
 let ix:=cons_index (acons st) in
 let ranges:=cache_rf (tighten ix st (rf_with_cache ix true DEPTH st)) (map fst(PM.elements(amap st))) in
 let project:=project_range_with ranges st in
 fun f=>project(flat_map(residue_eq(project[])st)(atoms_nodup(map fst(ft f))[]))f.

Definition projected_cached (st:astate)(check:form->option range)(k:constraint):astate:=
 let '(ts,lo,hi):=k in match check(form_of_terms ts) with
 | Some(l,h)=>if (lo <=? l) && (h <=? hi) then add_cons st k else st
 | None=>st end.

Definition selected_step_cached (st out : astate) (w : Z) (d r : reg) : astate :=
 if negb (Z.eqb w 64 && Z.eqb (wd d) 64 && Z.eqb (wd r) 64) || Pos.eqb d r then out else
 let delta := normalize st (fadd (as_form st r EX) (fscale (as_form st d EX) (-1) 64)) in
 let delta := materialized_form st delta d r in
 if negb (Z.eqb (fm delta) 64 && Z.eqb (flo delta) (fhi delta)) || mentions delta d then out else
 if short_any (cons_mentions d) (acons st) then
 let check := shared_check_cached st in
 fold_left (fun out k => let '(ts,lo,hi) := k in
   if negb (cons_mentions d k) then out else
   let q := coef ts (d,EX) + coef ts (d,EZ) in
   let ts' := sorted_terms (ts ++ tscale (-q) (ft delta)) in
   match check (form_of_terms ts') with
   | None => out
   | Some (l,h) =>
     let candidate := (ts',Z.min l (lo+q*flo delta),Z.max h (hi+q*flo delta)) in
     branch_cons1_prepared {|amap:=amap st;acons:=[candidate]|} (projected_cached st check (subst_cons d r candidate)) out w d (OReg d) (OReg r)
   end) (acons st) out else out.

Definition selection_steps (st out : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
 match a,b with OReg r,OReg s =>
   if Pos.eqb d r then selected_step_cached st out w d s else
   if Pos.eqb d s then selected_step_cached st out w d r else out
 | _,_ => out end.

Definition known_operand (st : astate) (o : opnd) : opnd :=
 match o with
 | OReg r =>
   match reg_range st r with
   | Some (lo,hi) => if Z.eqb lo hi then OImm lo else o
   | None => o
   end
 | _ => o
 end.

(* A masked load that retains a signed-maximum sentinel when inactive
   satisfies this inequality for either predicate value. *)
Definition sentinel_load (st out : astate) (g : guard) (bits : Z) (d : reg) : astate :=
 match g with Some (false,p) =>
 if negb (Z.eqb bits 64 && Z.eqb (wd d) 64 && Z.eqb (wd p) 1) then out else
 match const_of st (OReg d) 64 false with
 | Some c => if Z.eqb c (2^63-1) then
   add_cons out (sorted_terms [((d,ES),1);((p,EX),2^64-1)],2^63-1,2^63-1+(2^64-1))
   else out
 | None => out end
 | _ => out end.

(* A guarded write leaves its exact default c when inactive. With U the
   complete destination range, these two linear inequalities hold for every
   loaded value; no property of the memory contents is assumed. *)
Definition type_default (out : astate) (d : reg) : astate :=
 match get out d with
 | None=>put out d (FForm (range_form (wd d) 0 (2^wd d-1)) 0 None)
 | _=>out end.
Definition default_load (st out : astate) (g : guard) (d : reg) : astate :=
 match g with Some (neg,p) =>
 if negb (Z.eqb (wd p) 1) || Pos.eqb p d then out else
 match const_of st (OReg d) 64 false with
 | None=>out
 | Some c=>
   let u := 2 ^ wd d-1 in
   let upper := if neg then (sorted_terms [((d,EZ),1);((p,EX),u)],0,u+c)
                else (sorted_terms [((d,EZ),1);((p,EX),-u)],-u,c) in
   let lower := if neg then (sorted_terms [((d,EZ),1);((p,EX),-u)],c-u,u)
                else (sorted_terms [((d,EZ),1);((p,EX),u)],c,2*u) in
   add_cons (add_cons (type_default out d) upper) lower
 end | None=>out end.

Definition transfer_live (st : astate) (i : instr) : astate :=
  match i with
  | IBra _ _ | IExit _ | INop | IShA _ _ _ _ _ _ => st
  | IDef _ ds => fold_left (fun s d => assign s d None None) ds st
  | IMem g k gen a off width v fr ds =>
      let st1 := fold_left (fun s d => assign s d None None) ds st in
      match k with
      | KLoad => st1
      | _ =>
          (* a generic store or atomic may write local memory *)
          if gen then store_cells st1 g (local_of_generic st (addr_form st a off) width) width
                                  (match k, ds with KStore, [] => v | _, _ => None end) fr
          else st1
      end
  | ILdL g bits d a off fr => load_cell st g bits d a off fr
  | IStL g a off width v fr => store_cells st g (local_span st (addr_form st a off) width) width v fr
  | IFrame vs => frame_cells st vs
  | _ =>
      match dest i with
      | None => st
      | Some d =>
          match i with
          | ILoad g _ bits _ a off =>
              match g with
              | None => index_load st (transfer_dest st d None None false) (Z.of_nat bits) d a off
              | Some _ => default_load st (sentinel_load st (guarded_index_load st g (Z.of_nat bits) d a off) g (Z.of_nat bits) d) g d
              end
          | IBin None BAnd w _ a (OImm k) =>
              mask_const_cons st (transfer_dest st d None (value st i) false) (Z.of_nat w) d a k
          | IBin None BAnd w _ a (OReg rb) =>
              and_cons st (transfer_dest st d None (value st i) false) (Z.of_nat w) d a rb
          | IBin None BShr w _ a b =>
              shift_cons st (transfer_dest st d None (value st i) false) false (Z.of_nat w) d a b
          | IBin None BShrs w _ a b =>
              shift_cons st (transfer_dest st d None (value st i) false) true (Z.of_nat w) d a b
          | IDiv None true sg w _ a b =>
              rem_cons st (transfer_dest st d None (value st i) false) sg (Z.of_nat w) d a b
          | ICvt None to from sg _ s =>
              cvt_signed_bound st (transfer_dest st d None (relational_cvt st to from sg d s) false) to from sg d s
          | IMinMax None mn sg w _ a b =>
              minmax_relative_cons st (minmax_cons (transfer_dest st d None (value st i) false) mn sg w d a b) mn sg w d a b
          | ISelp g w _ a b p =>
              if PM.mem d (lf_payloads F) then assign st d None None else
              let st1 := transfer_dest st d g (sel_value st a b p (Z.of_nat w)) false in
              match g with None => selection_steps st (select_order st (sel_refined_cons_prepared st st1 (Z.of_nat w) d a b p) (Z.of_nat w) d (known_operand st a) (known_operand st b)) (Z.of_nat w) d a b | Some _ => st1 end
          (* a copy has its source's constraints, and its source's range as a
             constraint, which survive when the source is overwritten next,
             as a loop head that copies the position and then advances it *)
          | IMov None _ s =>
              let st1 := sel_cons st (transfer_dest st d None (value st i) false) (wd d) d (OReg s) (OReg s) in
              if existsb (cons_mentions s) (acons st) && Z.eqb (wd d) (wd s) then
                match reg_range st s with
                | Some (lo, hi) => add_cons st1 ([((d, EX), 1)], lo, hi)
                | None => st1
                end
              else st1
          | _ => transfer_dest st d (guard_of i) (value st i) false
          end
      end
  end.

(* A state that no values satisfy stays so. *)
(* An integer update d := d + s transports each constraint by substitution.
   Check the unreduced integer sum first: modular normalization would hide
   overflow and cannot justify this substitution. Constant steps already
   have a separate transfer rule. *)
Definition affine_step_cons (st out : astate) (d : reg) (s : form) : astate :=
 if null (ft s) then out else
 if negb (existsb (fun k => let '(ts,_,_) := k in
     cons_mentions d k && Nat.leb 2 (length (filter (fun t => negb (is_fixed (fst (fst t)))) ts))) (acons st)) then out else
 if match form_range st s with Some (lo,hi) => Z.eqb lo hi | None => false end then out else
 if negb (Z.eqb (wd d) 64 && Z.eqb (fm s) 64 && Z.eqb (flo s) (fhi s)) || mentions s d then out else
 let result := {|fm:=64;ft:=sorted_terms ([((d,EX),1)] ++ ft s);flo:=flo s;fhi:=fhi s|} in
 match check_range st result with
 | Some (lo,hi) => if (0 <=? lo) && (hi <? 2^64) then
   fold_left (fun out k => let '(ts,l,h) := k in
     let ds := filter (on_reg d) ts in
     if null ds || negb (forallb (fun t => match snd (fst t) with EX | EZ => true | _ => false end) ds) then out else
     let q := fold_left (fun acc t => acc+snd t) ds 0 in
     add_cons out (sorted_terms (ts ++ tscale (-q) (ft s)),l+q*flo s,h+q*fhi s)) (acons st) out
   else out
 | None => out end.
Definition complement_cons (st out : astate) (d : reg) (k : cmpf) : astate :=
 if negb (Z.eqb (wd d) 1) then out else
 fold_left (fun out pair => let '(p,x) := pair in
   if Pos.eqb p d then out else
   match x with FPred [old] true =>
     if Z.eqb (wd p) 1 && cmpf_eqb (negate old) k then
       add_cons out (sorted_terms [((p,EX),1);((d,EX),1)],1,1) else out
   | _ => out end) (PM.elements (amap st)) out.

Definition arithmetic_relations (st out : astate) (i : instr) : astate :=
 match i with
 | ISetp None c w sg d a b => complement_cons st out d {|cc:=c;cw:=w;csg:=sg;ca:=a;cb:=b|}
 | IBin None op w d a b =>
   if negb (Nat.eqb w 64) then out else
   if negb (Pos.eqb d a) then
     match op,b with BAdd,OReg r =>
       if Pos.eqb d r && Z.eqb (wd a) 64 then affine_step_cons st out d (fatom 64 (a,EX) 1) else out
     | _,_ => out end
   else
   let step := match b with OReg r => if Z.eqb (wd r) 64 then Some (fatom 64 (r,EX) 1) else None
                         | OImm z => Some (fconst 64 (z mod 2^64)) end in
   match step with Some s =>
     match op with BAdd => affine_step_cons st out d s | BSub => affine_step_cons st out d (fscale s (-1) 64) | _ => out end
   | None => out end
 | _ => out end.

(* ------------------------------------------------------------------ *)
(* The check.                                                          *)

Definition empty_state : astate := {| amap := PM.empty fact; acons := lf_derived F |}.

(* Only comparison results need explicit folding: Boolean identities already
   propagate known operands. Ordinary variable comparisons retain their exact
   predicate facts. A constant affine difference, or compound predicate
   negation, can expose the constant flags used by compiled range loops. *)
Definition foldable_predicate (st : astate) (i : instr) : bool :=
 match i with
 | ISetp None _ w sg d a b =>
   let compound := match get st d with Some (FPred (_::_::_) true) | Some (FNeg (_::_::_) true) => true | _ => false end in
   if Nat.eqb w 1 && compound then true else
   let fa := cmp_form st (OReg a) (Z.of_nat w) sg in
   let fb := cmp_form st b (Z.of_nat w) sg in
   let D := normalize st (fadd fa (fscale fb (-1) 64)) in
   null (ft D) && Z.eqb (flo D) (fhi D)

 | _ => false end.

Definition fold_predicate (st : astate) (i : instr) : astate :=
 if negb (foldable_predicate st i) then st else
 match i with ISetp None _ _ _ d _ _ =>
 if negb (Z.eqb (wd d) 1) then st else
 match get st d with Some (FPred _ _) | Some (FNeg _ _) =>
   match refine st d true with None => put st d (FForm (fconst 1 0) 0 None) | Some _ =>
   match refine st d false with None => put st d (FForm (fconst 1 1) 0 None) | Some _ => st end end
 | _ => st end
 | _ => st end.

(* The bounded launch scalars that a constraint of st relates to one of
   the registers rs. *)
Definition related_scalars (st : astate) (rs : list reg) : list reg :=
  fold_left (fun acc k => let '(ts, _, _) := k in
    if existsb (fun t => existsb (Pos.eqb (fst (fst t))) rs) ts
    then fold_left (fun acc t => let n := fst (fst t) in
           if fixed_scalar_term t && negb (existsb (Pos.eqb n) acc) then n :: acc else acc) ts acc
    else acc) (acons st) [].

(* Machine-type bounds remove vacuous endpoints before widening. Without
   this normalization a countdown can repeatedly lower the irrelevant side
   of count - start <= 0, causing widening to discard its stable upper side. *)
Definition clip_constraint (k : constraint) : constraint :=
 let '(ts,lo,hi) := k in
 match plain_sum natural_range ts 0 0 with
 | Some (l,h) => (ts,Z.max lo l,Z.min hi h)
 | None => k end.

(* The constraint that the terms ts are at most hi, or at least lo, with
   the other bound the natural one of the terms, so that two one-sided
   constraints of the same terms compare by their one bound. *)
Definition upper_cons (ts : terms) (hi : Z) : constraint := clip_constraint (ts, - 2 ^ 66, hi).
Definition lower_cons (ts : terms) (lo : Z) : constraint := clip_constraint (ts, lo, 2 ^ 66).

(* k added unless a constraint of the same terms already lies within it. *)
Definition add_bound (st : astate) (k : constraint) : astate :=
  let '(ts, lo, hi) := k in
  if existsb (fun k' => let '(ts', lo', hi') := k' in terms_eqb ts ts' && (lo <=? lo') && (hi' <=? hi)) (acons st)
  then st else add_cons st k.

(* The bounds of d - n that the projection of st gives, as two
   constraints, one for each side, so that each side moves on its own. *)
Definition scalar_bound (st : astate) (d n : reg) : astate :=
  let ts := sorted_terms [((d, EX), 1); ((n, EX), -1)] in
  match check_range st (form_of_terms ts) with
  | Some (lo, hi) =>
      let st := if hi <? 2 ^ 63 then add_bound st (upper_cons ts hi) else st in
      if - 2 ^ 63 <? lo then add_bound st (lower_cons ts lo) else st
  | None => st
  end.

(* Register r is half of a value: a constraint gives it coefficient -2, as
   the relation f - 2 (f >> 1) in [0, 1] of rule shift_cons. *)
Definition halved (st : astate) (r : reg) : bool :=
  existsb (fun k => let '(ts, _, _) := k in Z.eqb (coef ts (r, EX)) (-2)) (acons st).

(* The bound lo <= a - n <= hi of a register a in [0, 2^63) by a launch
   scalar n, moved to d = a + c: on d's value when a + c stays in
   [0, 2^64), and on d's signed reading when it stays in the signed range
   and may be negative, as mid - 1 for mid >= 0, each within the natural
   range of its terms. *)
Definition offset_bound (a d : reg) (c : Z) (value signed : bool) (out : astate) (k : constraint) : astate :=
  let '(ts, lo, hi) := k in
  let reads := fun e => match e with EX | ES => true | _ => false end in
  let scalar := match ts with
                | [((u, eu), cu); ((v, ev), cv)] =>
                    if Pos.eqb u a && Z.eqb cu 1 && reads eu && Z.eqb cv (-1) &&
                       fixed_scalar_term ((v, ev), cv) && match ev with EX => true | _ => false end
                    then Some v
                    else if Pos.eqb v a && Z.eqb cv 1 && reads ev && Z.eqb cu (-1) &&
                            fixed_scalar_term ((u, eu), cu) && match eu with EX => true | _ => false end
                    then Some u else None
                | _ => None
                end in
  match scalar with
  | Some n =>
      let out := if value then add_bound out (clip_constraint (sorted_terms [((d, EX), 1); ((n, EX), -1)], lo + c, hi + c)) else out in
      if signed then add_bound out (clip_constraint (sorted_terms [((d, ES), 1); ((n, EX), -1)], lo + c, hi + c)) else out
  | None => out
  end.

(* Register r is a midpoint: a half, or a sum whose form has a half. *)
Definition midpoint (st : astate) (r : reg) : bool :=
  halved st r || match get st r with
                 | Some (FForm f _ _) => existsb (fun t => halved st (fst (fst t))) (ft f)
                 | _ => false
                 end.

(* A search's midpoint d := lo + half, the sum of two registers that no
   launch fixes, each with coefficient 1 in d's form, one of them a half,
   gets its bound by each bounded launch scalar that a constraint relates
   to a term, as mid - n <= -1, projected while the relations of the terms
   are known: a later write to lo, the first of the two bounds a search
   updates, forgets the relations through lo. A sum d := a + c of a
   midpoint in [0, 2^63) and a constant, as mid + 1 or mid - 1, moves each
   bound a - n of a launch scalar n to d, in d's value when a + c stays in
   [0, 2^64) and in d's signed reading when it stays in the signed range,
   and states that reading as a + c. *)
Definition sum_bounds (out : astate) (i : instr) : astate :=
  match i with
  | IBin None BAdd w d a (OReg r) =>
      if negb (Nat.eqb w 64) || Pos.eqb d a || Pos.eqb d r || negb (Z.eqb (wd d) 64) then out else
      match get out d with
      | Some (FForm f _ _) =>
          if negb (Nat.eqb (length (ft f)) 2 && forallb (fun t => let '((x, e), c) := t in
                     negb (is_fixed x) && Z.eqb c 1 && Z.eqb (wd x) 64 &&
                     match e with EX => true | _ => false end) (ft f) &&
                   existsb (fun t => halved out (fst (fst t))) (ft f)) then out else
          fold_left (fun s n => scalar_bound s d n) (related_scalars out (map (fun t => fst (fst t)) (ft f))) out
      | _ => out
      end
  | IBin None BAdd w d a (OImm c) =>
      if negb (Nat.eqb w 64) || Pos.eqb d a || is_fixed a || negb (Z.eqb (wd d) 64 && Z.eqb (wd a) 64) ||
         negb (midpoint out a) then out else
      let c := extend 64 true c in
      match reg_range out a with
      | Some (x, y) =>
          if negb ((0 <=? x) && (y <? 2 ^ 63)) then out else
          let value := (0 <=? x + c) && (y + c <? 2 ^ 64) in
          let signed := (- 2 ^ 63 <=? x + c) && (y + c <? 2 ^ 63) && (x + c <? 0) in
          let out := if signed then add_bound out (sorted_terms [((d, ES), 1); ((a, EX), -1)], c, c) else out in
          fold_left (offset_bound a d c value signed) (acons out) out
      | None => out
      end
  | _ => out
  end.

(* d := a >> 1 of a value a in [lo, hi], lo >= 0, that is the difference of
   two registers that no launch fixes, with at most the shifted sign bits
   of a compiler's rounding added, as half := (hi - lo) >> 1 of a search: a - d = a - a / 2 grows with a, so it lies in
   [lo - lo / 2, hi - hi / 2]. With hi - lo >= 1 this gives
   lo + half <= hi - 1, the midpoint below the upper end. *)
Definition half_bounds (st out : astate) (i : instr) : astate :=
  match i with
  | IBin None op w d a (OImm k) =>
      let sg := match op with BShrs => true | _ => false end in
      if negb (match op with BShr | BShrs => true | _ => false end && Nat.eqb w 64 && Z.eqb k 1 &&
               Z.eqb (wd d) 64 && Z.eqb (wd a) 64) then out else
      let f := cmp_form st (OReg a) 64 sg in
      let value := fun t : atom * Z => match snd (fst t) with EX | EZ | ES => negb (is_fixed (fst (fst t))) | _ => false end in
      let shift := fun t : atom * Z => match snd (fst t) with EShr _ => true | _ => false end in
      if negb (Nat.eqb (length (filter value (ft f))) 2 &&
               Z.eqb (fold_left (fun acc t => acc + snd t) (filter value (ft f)) 0) 0 &&
               forallb (fun t => value t || shift t) (ft f)) ||
         negb (Z.eqb (flo f) (fhi f)) || mentions f d then out else
      match cmp_range st (OReg a) 64 sg with
      | Some (lo, hi) =>
          if 0 <=? lo
          then add_bound out (sorted_terms (ft f ++ [((d, EX), -1)]), lo - lo / 2 - flo f, hi - hi / 2 - flo f)
          else out
      | None => out
      end
  | _ => out
  end.

(* The bounded launch scalars of the forms of the registers rs. *)
Definition form_scalars (st : astate) (rs : list reg) : list reg :=
  fold_left (fun acc r => match get st r with
    | Some (FForm f _ _) => fold_left (fun acc t => let n := fst (fst t) in
        if fixed_scalar_term t && negb (existsb (Pos.eqb n) acc) then n :: acc else acc) (ft f) acc
    | _ => acc end) rs [].

(* An update d := selp(a, b, p) of a loop register d that is a or b, of two
   64-bit registers, lies, by each bounded launch
   scalar n of the form of a or b, within the hull of a - n where p holds and
   b - n where it does not, each projected in the state that the branch of
   p refines, for the value and for the signed reading of the registers, one
   constraint for each side, so that each side moves on its own: the
   bound hi - n <= -1 of a closed search survives hi := selp(hi, mid - 1, p)
   when mid - n <= -1, in the signed reading, where mid - 1 may be -1. *)
Definition selection_bounds (st out : astate) (i : instr) : astate :=
  match i with
  | ISelp None w d (OReg a) (OReg b) p =>
      if negb (Nat.eqb w 64 && Z.eqb (wd d) 64 && Z.eqb (wd a) 64 && Z.eqb (wd b) 64 &&
               (Pos.eqb d a || Pos.eqb d b)) then out else
      match form_scalars st [a; b] with
      | [] => out
      | scalars =>
          match refine_selection st p true, refine_selection st p false with
          | Some sa, Some sb =>
              fold_left (fun out n => if Pos.eqb n d then out else fold_left (fun out e =>
                match check_range sa (form_of_terms (sorted_terms [((a, e), 1); ((n, EX), -1)])),
                      check_range sb (form_of_terms (sorted_terms [((b, e), 1); ((n, EX), -1)])) with
                | Some (l, h), Some (l', h') =>
                    let ts := sorted_terms [((d, e), 1); ((n, EX), -1)] in
                    let out := if Z.max h h' <? 2 ^ 63 then add_bound out (upper_cons ts (Z.max h h')) else out in
                    if - 2 ^ 63 <? Z.min l l' then add_bound out (lower_cons ts (Z.min l l')) else out
                | _, _ => out
                end) [EX; ES] out) scalars out
          | _, _ => out
          end
      end
  | _ => out
  end.

Definition transfer (st : astate) (i : instr) : astate :=
  if is_bottom st then st else
  fold_predicate (selection_bounds st (sum_bounds (half_bounds st (arithmetic_relations st (transfer_live st i) i) i) i) i) i.

(* Register widths agree with the instruction's types, and no instruction
   writes a fixed register. *)
Definition wf_instr (i : instr) : bool :=
  let fixed_ok := match i with
                  | IDef _ ds | IMem _ _ _ _ _ _ _ _ ds => forallb (fun d => negb (is_fixed d)) ds
                  | ILdL _ _ d _ _ _ => negb (is_fixed d)
                  | _ => match dest i with Some d => negb (is_fixed d) | None => true end
                  end in
  fixed_ok &&
  match i with
  | IMov _ d s => Z.eqb (wd d) (wd s)
  | IBin _ o w d a b =>
      let w := Z.of_nat w in
      Z.eqb (wd d) w && Z.eqb (wd a) w &&
      match o with BShl | BShr | BShrs => opnd_width_ok b 32 | _ => opnd_width_ok b w end
  | IMad _ w d a b c => let w := Z.of_nat w in Z.eqb (wd d) w && Z.eqb (wd a) w && opnd_width_ok b w && opnd_width_ok c w
  | IWide _ _ d a b c =>
      Z.eqb (wd d) 64 && Z.eqb (wd a) 32 && opnd_width_ok b 32 &&
      match c with Some r => Z.eqb (wd r) 64 | None => true end
  | ICvt _ to from _ d s => Z.eqb (wd d) (Z.of_nat to) && (Z.of_nat from <=? wd s)
  | IBfe _ w d a p ln => let w := Z.of_nat w in Z.eqb (wd d) w && Z.eqb (wd a) w && (Z.of_nat p <? 256) && (Z.of_nat ln <? 256)
  | ISetp _ _ w _ d a b => Z.eqb (wd d) 1 && Z.eqb (wd a) (Z.of_nat w) && opnd_width_ok b (Z.of_nat w)
  | IAndp _ d a b | IOrp _ d a b | IXorp _ d a b => Z.eqb (wd d) 1 && Z.eqb (wd a) 1 && Z.eqb (wd b) 1
  | IMinMax _ _ _ w d a b => let w := Z.of_nat w in Z.eqb (wd d) w && Z.eqb (wd a) w && opnd_width_ok b w
  | ISelp _ w d _ _ p => Z.eqb (wd d) (Z.of_nat w) && Z.eqb (wd p) 1
  | IDiv _ _ _ w d a b => let w := Z.of_nat w in Z.eqb (wd d) w && Z.eqb (wd a) w && opnd_width_ok b w
  | ILoad _ _ bits d a _ => (Z.of_nat bits <=? wd d) && Z.eqb (wd a) 64 && (8 <=? Z.of_nat bits) &&
                          Z.eqb (Z.of_nat bits mod 8) 0
  | IMem _ _ _ a _ width _ _ _ => Z.eqb (wd a) 64 && (0 <? width)
  | IStL _ _ _ width _ _ => 0 <? width
  | ILdS _ bits d a _ => (Z.of_nat bits <=? wd d) && Z.eqb (wd a) 32 && (8 <=? Z.of_nat bits) &&
                        (Z.of_nat bits <=? 64) && Z.eqb (Z.of_nat bits mod 8) 0
  | IShA _ _ _ _ width _ => 0 <? width
  | _ => true
  end &&
  match guard_of i with Some (_, p) => Z.eqb (wd p) 1 | None => true end.

(* The check that the global access of instruction i stays in its region,
   and that a local store stays in a live variable. *)
Definition region_of (st : astate) (i : instr) : bool :=
  match i with
  | ILoad g gen bits _ a off => access_ok st g KLoad gen a off (Z.of_nat bits / 8) None
  | IMem g k gen a off width v fr _ =>
      access_ok st g k gen a off width v ||
      (* a generic store or atomic in a live local variable *)
      (gen && negb (is_load k) && local_store_ok (local_of_generic st (addr_form st a off) width) width fr)
  | IStL _ a off width _ fr => local_store_ok (local_span st (addr_form st a off) width) width fr
  (* a shared access inside a shared variable, and a local load inside a
     local variable: past the memory of the block or the thread, an access
     faults *)
  | ILdS g bits _ a off => private_ok st g (lf_shared F) a off (Z.of_nat bits / 8)
  | IShA g _ a off width _ => private_ok st g (lf_shared F) a off width
  | ILdL g bits _ a off _ => private_ok st g local_vars a off (Z.of_nat bits / 8)
  | _ => true
  end.

(* The alignment the address a + off has: its form's, or the one a and off
   have in common. *)
Definition addr_align (st : astate) (a : reg) (off : Z) : Z :=
  Z.max (form_align st (addr_form st a off)) (Z.min (reg_align st a) (v2 off)).

(* The alignment an access of width bytes needs: the largest power of two
   that divides the width, up to 32 bytes, as for a scalar or vector access
   of that width. *)
Definition need (width : Z) : Z := Z.min (v2 width) 5.

(* The check that the memory access of instruction i, in any state space, is
   aligned. *)
Definition aligned_ok (st : astate) (i : instr) : bool :=
  match i with
  | ILoad _ _ bits _ a off | ILdL _ bits _ a off _ | ILdS _ bits _ a off =>
      need (Z.of_nat bits / 8) <=? addr_align st a off
  | IMem _ _ _ a off width _ _ _ | IStL _ a off width _ _ | IShA _ _ a off width _ =>
      need width <=? addr_align st a off
  | _ => true
  end.

(* The checks of the memory access of instruction i. *)
Definition access_of (st : astate) (i : instr) : bool :=
  if region_of st i then aligned_ok st i else false.

(* The effective fact of r: its own, or the bound of a fixed register. *)
Definition eff_fact (st : astate) (r : reg) : option fact :=
  match get st r with
  | Some x => Some x
  | None => if is_fixed r then
              match lookup (lf_bounds F) r with
              | Some (lo, hi) => Some (FForm (range_form (wd r) lo hi) 0 None)
              | None => None
              end
            else None
  end.

Definition branch_excludes (st : astate) (r : reg) (truth : bool) (k : cmpf) : bool :=
 match refine st r truth with None => true | Some s =>
 match refine_cmp s k with None => true | Some s' => is_bottom s' end end.
Definition predicate_fact_le (st : astate) (r : reg) (g : fact) : bool :=
 match g with FPred [k] exact =>
 branch_excludes st r true (negate k) && (negb exact || branch_excludes st r false k)
 | _ => false end.
(* A contradictory branch establishes a predicate's bit value. *)
Definition predicate_constant (st : astate) (p : reg) : option Z :=
 if negb (Z.eqb (wd p) 1) then None else
 match get st p with Some (FPred _ true) | Some (FNeg _ true) =>
   match refine st p true with None => Some 0 | Some _ =>
   match refine st p false with None => Some 1 | Some _ => None end end
 | _ => None end.
Definition register_fact_le (st : astate) (r : reg) (g : fact) : bool :=
 if fact_le_o st (eff_fact st r) (Some g) then true else
 match g with
 | FPred _ _ => predicate_fact_le st r g
 | FForm f al None =>
   if Z.eqb (wd r) 1 && Z.eqb (fm f) 1 && null (ft f) && (al <=? 0) then
     if (flo f <=? 0) && (1 <=? fhi f) then true else
     match predicate_constant st r with Some c => (flo f <=? c) && (c <=? fhi f) | None => false end else
   if Z.eqb (fm f) 64 && Z.eqb (wd r) 64 && null (ft f) && (al <=? 0) then
   match check_range st (fatom 64 (r,EX) 1) with
   | Some (lo,hi) => (flo f <=? lo) && (hi <=? fhi f)
   | None => false end else false
 | _ => false end.

(* Each constraint of b follows from a. *)
Definition cons_le (a : astate) (b : list constraint) : bool := short_all (cons_holds a) b.

(* Every fact and constraint that b claims follows from a. *)
Definition state_le (a b : astate) : bool :=
  if is_bottom a then true else
  if PM.fold (fun r g (acc : bool) => if acc then register_fact_le a r g else false) (amap b) true
  then cons_le a (acons b) else false.

Definition inv_at (inv : list astate) (b : nat) : astate := nth b inv empty_state.

Fixpoint walk (st : astate) (body : list instr) : astate :=
  match body with
  | [] => st
  | ins :: rest => walk (transfer st ins) rest
  end.

(* The state along each edge out of block b, from the state at its end. *)
Definition edge_states (prog : program) (b : nat) (out : astate) : list (nat * option astate) :=
  match last_instr (block_body prog b) with
  | Some (IBra (Some (neg, p)) t) => [(t, refine out p (negb neg)); (S b, refine out p neg)]
  | Some (IBra None t) => [(t, Some out)]
  | Some (IExit None) => []
  | Some (IExit (Some (neg, p))) => [(S b, refine out p neg)]
  | _ => [(S b, Some out)]
  end.

(* The targets of the edges out of block b, which do not depend on the state. *)
Definition edge_targets (prog : program) (b : nat) : list nat :=
  match last_instr (block_body prog b) with
  | Some (IBra (Some _) t) => [t; S b]
  | Some (IBra None t) => [t]
  | Some (IExit None) => []
  | Some (IExit (Some _)) => [S b]
  | _ => [S b]
  end.
(* A block whose edges all leave the program adds no state to any block. *)
Definition leaves_program (prog : program) (b : nat) : bool :=
  forallb (fun v => negb (Nat.ltb v (length prog))) (edge_targets prog b).

(* The launch facts that are properties of F alone, which the check decides
   rather than assumes. *)
(* A bound form: 64-bit, one constant, and terms over fixed registers read
   as their values, so that its value is the same in every state. *)


(* A span's extent and index bounds are bound forms, and its elements have
   1, 2, 4, or 8 bytes; a flag span is written only, without an index
   range. *)
Definition span_ok (sp : span) : bool :=
  bound_form (sp_ext sp) &&
  match sp_index sp with
  | Some ix => (Z.eqb (ix_elem ix) 1 || Z.eqb (ix_elem ix) 2 || Z.eqb (ix_elem ix) 4 || Z.eqb (ix_elem ix) 8) &&
               bound_form (ix_lo ix) && bound_form (ix_hi ix) && negb (sp_flag sp)
  | None => negb (sp_flag sp) || match sp_dir sp with DWrite => true | _ => false end
  end.

(* A cell: a register that is not fixed, of 4 or 8 bytes, inside a local
   variable. *)
Definition cell_ok (k : reg * reg * Z) : bool :=
  let '(p, u, c) := k in
  negb (is_fixed p) && (Z.eqb (wd p) 32 || Z.eqb (wd p) 64) &&
  match lookup (lf_locals F) u with Some (s, _) => (0 <=? c) && (c + wd p / 8 <=? s) | None => false end.

Definition lf_ok : bool :=
  forallb (fun p => Nat.leb 1 (snd p) && Nat.leb (snd p) 64) (PM.elements (lf_widths F)) &&
  forallb (fun p => is_fixed (fst p) && span_ok (snd p)) (lf_roles F) &&
  is_fixed (lf_shared_window F) && is_fixed (lf_local_window F) &&
  (* each local variable lies in the first window_size bytes of the window *)
  forallb (fun v => is_fixed (fst v) && Z.eqb (wd (fst v)) 32 &&
                    match lookup (lf_bounds F) (fst v) with
                    | Some (lo, hi) => (0 <=? lo) && (hi + fst (snd v) <=? window_size)
                    | None => false
                    end) (lf_locals F) &&
  forallb cell_ok (lf_cells F).

(* Separate invariants retain distinctions between bounded loop iterations.
   The certificate is checked against the original program. Every outgoing
   state must imply one of the target block's cases. *)
Definition cases_at (inv : list (list astate)) (b : nat) : list astate := nth b inv [].
Definition entry_case (st : astate) : bool :=
 PM.is_empty (amap st) &&
 forallb (fun k => existsb (constraint_eqb k) (lf_derived F)) (acons st).

(* ------------------------------------------------------------------ *)
(* The analysis: block invariants by iteration, with widening. Nothing
   about it is trusted; `check` decides whether its result holds.       *)

(* A bounded launch scalar often supplies a loop's limit. Retain its endpoints
   and the preceding integers when widening a strict loop guard. Without
   these thresholds, a window of at most 1024 widens straight to 2^28.
   These are invariant proposals; check_thread still validates every edge. *)
Definition THRESH : list Z :=
  flat_map (fun pair => let '(_, (lo, hi)) := pair in [lo; lo - 1; hi; hi - 1]) (lf_bounds F)
  ++ [0; 2 ^ 28; 2 ^ 29; 2 ^ 30; 2 ^ 31 - 1; 2 ^ 32 - 1; - 2 ^ 31].
Definition WIDEN_AFTER : nat := 8.

(* The constants that the program tests a register for equality with, and
   their neighbours by the steps 1 to 16: a counter that steps by c and
   leaves its loop when it equals K stops widening at K - c or K + c,
   where the test's refinement bounds it,
   independent of the launch bounds, as a Triton loop over a block of
   BLOCK elements that a smaller CTA runs. *)
Definition program_thresholds (prog : program) : list Z :=
  flat_map (fun i => match i with
    | ISetp _ CEq w sg _ _ (OImm z) | ISetp _ CNe w sg _ _ (OImm z) =>
        let v := extend w sg z in v :: flat_map (fun c => [v - c; v + c]) [1; 2; 4; 8; 16]
    | _ => [] end) (concat prog).

Definition widen (extra : list Z) (f g : fact) : option fact :=
  match f, g with
  | FForm a _ _, FForm b bl _ =>
      if null (ft a) && null (ft b) then
        let lo := if flo a <=? flo b then Some (flo b)
                  else fold_left (fun acc t => if t <=? flo b then match acc with Some x => Some (Z.max x t) | None => Some t end else acc) (THRESH ++ extra) None in
        let hi := if fhi b <=? fhi a then Some (fhi b)
                  else fold_left (fun acc t => if fhi b <=? t then match acc with Some x => Some (Z.min x t) | None => Some t end else acc) (THRESH ++ extra) None in
        match lo, hi with
        | Some lo, Some hi => Some (FForm (range_form (Z.min (fm a) (fm b)) lo hi) bl None)
        | _, _ => None
        end
      else None
  | _, _ => None
  end.

Definition const_form (x : fact) : option form :=
  match x with FForm f _ _ => if Z.eqb (flo f) (fhi f) then Some f else None | _ => None end.

(* The c with step dr = c * dp, for two 64-bit steps. *)
Definition step_ratio (dr dp : form) : option Z :=
  let c := match ft dp with
           | (x, cp) :: _ =>
               let cr := tget (ft dr) x in
               if negb (Z.eqb cp 0) && Z.eqb (Z.rem cr cp) 0 then Some (Z.quot cr cp) else None
           | [] => None
           end in
  match c with
  | Some c =>
      let sp := fscale dp c 64 in
      if negb (Z.eqb c 0) && terms_eqb (ft dr) (ft sp) &&
         Z.eqb (flo dr mod 2 ^ 64) (flo sp mod 2 ^ 64) && Z.eqb (fhi dr mod 2 ^ 64) (fhi sp mod 2 ^ 64)
      then Some c else None
  | None => None
  end.

(* Propose signed versions only where the other side of this join can cross
   unsigned zero. Validate every proposal on both incoming states. Cache the
   relevant registers once: recomputing recursive ranges for every occurrence
   in every constraint makes nested loops needlessly expensive. *)
(* Propose bounds on a pair of loop cursors from their common initial
   affine sum. Every candidate is checked on both incoming states, and the
   final thread checker validates all resulting edges. *)
(* The sum of two loop-entry cursor values can remain invariant while each
   cursor changes. Restrict proposals to three-term unit-coefficient sums;
   stride and address combinations otherwise cause quadratic projection work.
   Both incoming states must prove each retained candidate. *)
Definition pair_candidates (st other : astate) : list constraint :=
 let vars := flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match const_form x, get other r with
   | Some f, Some (FForm g _ _) =>
     if Z.eqb (fm f) 64 && negb (form_eqb f g) &&
        forallb (fun t => is_fixed (fst (fst t))) (ft f)
     then [(r,f)] else []
   | _,_ => [] end) (PM.elements (amap st)) in
 flat_map (fun pair => let '(r,f) := pair in
   flat_map (fun pair' => let '(s,g) := pair' in
     if Pos.ltb r s then
       let h := fadd f g in
       
       let ts := sorted_terms ([((r,EX),1);((s,EX),1)] ++ tscale (-1) (ft h)) in
       if negb (cursor_sum ts || (Nat.eqb (length ts) 2 && forallb (fun t => let '((r,e),c) := t in
         Z.eqb (wd r) 64 && Z.eqb c 1 && match e with EX => true | _ => false end) ts)) then [] else
       let upper := (ts,-(2^65),fhi h) in
       let lower := (ts,flo h,2^65) in
       let checks := fun k => if cons_holds st k then cons_holds other k else false in
       (* short_all_same preserves both checks while skipping an unnecessary second projection. *)
       if null (ft h) then if short_all checks [upper;lower] then [(ts,flo h,fhi h)] else []
       else filter checks [upper;lower]
     else []) vars) vars.

Definition triple_candidates (st other : astate) : list constraint :=
 let vars := flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match const_form x,get other r with Some f,Some (FForm g _ _) =>
    if Z.eqb (fm f) 64 && negb (form_eqb f g) && forallb (fun t => is_fixed (fst (fst t))) (ft f)
    then [(r,f)] else [] | _,_ => [] end) (PM.elements (amap st)) in
 flat_map (fun pair => let '(r,f) := pair in
 flat_map (fun pair' => let '(s,g) := pair' in
 if negb (Pos.ltb r s) then [] else
 let pair_ts := sorted_terms [((r,EX),1);((s,EX),1)] in
 if negb (existsb (fun k => let '(ts,_,_) := k in terms_eqb ts pair_ts) (acons other)) then [] else
 flat_map (fun third => let '(q,h) := third in
 if Pos.eqb q r || Pos.eqb q s then [] else
 let start := fadd (fadd f g) (fscale h (-1) 64) in
 let ts := sorted_terms ([((r,EX),1);((s,EX),1);((q,EX),-1)] ++ tscale (-1) (ft start)) in
 if negb (cursor_sum ts) then [] else
 let candidates := [(ts,-2^65,fhi start);(ts,flo start,2^65)] in
 if forallb (fun k => if cons_holds st k then cons_holds other k else false) candidates then [(ts,flo start,fhi start)] else []) vars) vars) vars.

Definition exit_sum_candidates (st other : astate) : list constraint :=
 let vars := flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match const_form x,get other r with Some f,Some (FForm g _ _) =>
    if Z.eqb (fm f) 64 && negb (form_eqb f g) && null (ft f)
    then [r] else [] | _,_ => [] end) (PM.elements (amap st)) in
 let fixed := flat_map (fun pair => let '(r,x) := pair in
   if negb (Z.eqb (wd r) 64) then [] else
   match const_form x,get other r with Some f,Some (FForm g _ _) =>
    if form_eqb f g then [r] else [] | _,_ => [] end) (PM.elements (amap st)) in
 flat_map (fun r => flat_map (fun s => if negb (Pos.ltb r s) then [] else
 let pair_ts := sorted_terms [((r,EX),1);((s,EX),1)] in
 if negb (existsb (fun k => let '(ts,_,_) := k in terms_eqb ts pair_ts) (acons other)) then [] else
 flat_map (fun q =>
 if Pos.eqb q r || Pos.eqb q s then [] else
 let ts := sorted_terms [((r,EX),1);((s,EX),1);((q,EX),-1)] in
 let k := (ts,0,0) in
 if cons_holds st k then if cons_holds other k then [k] else [] else []) fixed) vars) vars.

Definition signed_candidates (st other : astate) : list constraint :=
  let atoms := atoms_nodup (flat_map (fun k => let '(ts, _, _) := k in map fst ts) (acons st)) [] in
  let negative := flat_map (fun x => let '(r, e) := x in
    if (if Z.eqb (wd r) 64 then
          match get other r with
          | Some (FForm f _ _) => (flo f <? 0) || short_any (fun t => (snd t <? 0) ||
                                     match snd (fst t) with ES => true | _ => false end) (ft f)
          | _ => false
          end
        else false) then
      if match fact_range r (get other r) with
         | Some (lo, hi) => (0 <=? lo) && (hi <? 2 ^ 63)
         | None => false
         end then [] else
      match sreg_range other r with Some (lo, _) => if lo <? 0 then [r] else [] | None => [] end
    else []) atoms in
  if null negative then [] else
  flat_map (fun k => let '(ts, lo, hi) := k in
    let ts' := map (fun t => let '((r, e), c) := t in
      match e with
      | EX | EZ => if existsb (Pos.eqb r) negative then ((r, ES), c) else t
      | _ => t end) ts in
    if terms_eqb ts ts' then [] else
      let k' := (sorted_terms ts', lo, hi) in
      if cons_holds st k' then if cons_holds other k' then [k'] else [] else []) (acons st).

(* The terms of an equality m * i - j = c over launch values that
   proportion_candidates proposes: two 64-bit registers that no launch fixes,
   read as their values, and fixed registers. *)
Definition proportion_terms (ts : terms) : bool :=
 Nat.eqb (length (filter (fun t => negb (is_fixed (fst (fst t)))) ts)) 2 &&
 forallb (fun t => let '((r,e),_) := t in
   is_fixed r || (Z.eqb (wd r) 64 && match e with EX => true | _ => false end)) ts.

Definition recover_cursor_ranges (st : astate) : astate :=
 let atoms := atoms_nodup (flat_map (fun k => let '(ts,lo,hi) := k in
   if Z.eqb lo hi && (cursor_sum ts || proportion_terms ts)
   then filter (fun x => negb (is_fixed (fst x))) (map fst ts) else []) (acons st)) [] in
 fold_left (fun out a => let '(r,e) := a in
   if is_fixed r || negb (Z.eqb (wd r) 64) then out else
   let eligible := match get st r with None => true | Some (FForm f _ _) => null (ft f) | _ => false end in
   if negb eligible then out else
   match check_range st (fatom 64 (r,EX) 1) with
   | Some (lo,hi) => if (0 <=? lo) && (hi <? 2^63) then
       (* keep the alignment, and the old bounds where they are tighter *)
       let al := match get st r with Some (FForm _ al _) => al | _ => 0 end in
       let '(lo,hi) := match fact_range r (get st r) with
                       | Some (p,q) => (Z.max lo p, Z.min hi q) | None => (lo,hi) end in
       {|amap:=PM.add r (FForm (range_form 64 lo hi) al None) (amap out);acons:=acons out|} else out
   | None => out end) atoms st.

(* An advancing counter and an increasing lower bound on a fixed affine
   expression suggest a relative bound. Check both incoming states before
   proposing it. Restrict proposals to changing bounds to keep their number
   small; the final thread checker independently validates every invariant. *)
Definition moving_bound_candidates (st other : astate) : list constraint :=
 flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match const_form x,get other r with Some f,Some (FForm g _ _) =>
   if negb (Z.eqb (fm f) 64 && Z.eqb (fm g) 64 && null (ft f) && null (ft g) && Z.eqb (flo g) (fhi g) && (flo f <? flo g)) then [] else
   flat_map (fun k => let '(ts,lo,hi) := k in
     if null ts || negb (forallb (fun t => is_fixed (fst (fst t))) ts) then [] else
     if negb (existsb (fun k' => let '(ts',lo',hi') := k' in terms_eqb ts ts' && (lo <? lo')) (acons other)) then [] else
     let c := (sorted_terms ([((r,EX),1)] ++ tscale (-1) ts),-(2^64),flo f-lo) in
     if cons_holds st c then if cons_holds other c then [c] else [] else []) (acons st)
   | _,_ => [] end) (PM.elements (amap st)).

(* Propose a common constant only when the ordinary join loses a predicate
   fact or its exactness. Eagerly evaluating every predicate at every join
   is unnecessarily expensive. The edge checker validates the proposed bit. *)
Definition join_bit (st : astate) (p : reg) : option Z :=
 match known_bit st p with Some b => Some (if b then 1 else 0) | None => predicate_constant st p end.
(* A loop that Triton counts by a trip count exits by an inequality:
   j := 0; body; j := j + 1; if j <> B goto head. The bound B - j >= 1 is
   proposed at the head from the test's predicate on the back edge, for a
   64-bit counter that steps by 1 from a constant, in the orientation of
   the test's difference; both incoming states must prove it. The final
   checker validates the invariant on every edge. *)
(* This is the exact predicate-fact shape consumed by exit-bound
   proposal recovery. Incoming liveness uses the same classification. *)
Definition exit_bound_witness (f : fact) : option (reg * reg) :=
 match f with
 | FPred [k] _ =>
     match cc k, Nat.eqb (cw k) 64, cb k with
     | (CEq | CNe), true, OReg rb => Some (rb, ca k)
     | _, _, _ => None
     end
 | _ => None
 end.

Definition exit_bound_candidates (st other : astate) : list constraint :=
 let counter := fun (j b : reg) (sign : Z) =>
   if is_fixed j || negb (Z.eqb (wd j) 64) || negb (Z.eqb (wd b) 64) then [] else
   match get st j, get other j with
   | Some (FForm a _ _), Some (FForm a' _ _) =>
       if Z.eqb (fm a) 64 && Z.eqb (fm a') 64 && terms_eqb (ft a) (ft a') &&
          Z.eqb (flo a' - flo a) 1 && Z.eqb (fhi a' - fhi a) 1 then
         let fb := cmp_form other (OReg b) 64 false in
         let ts := sorted_terms (tscale sign (ft fb) ++ [((j, EX), - sign)]) in
         let c := if Z.eqb sign 1 then (ts, 1 - fhi fb, 2 ^ 64 - 1 - flo fb)
                  else (ts, flo fb - (2 ^ 64 - 1), fhi fb - 1) in
         if cons_holds st c then if cons_holds other c then [c] else [] else []
       else []
   | _, _ => []
   end in
 flat_map (fun pair => let '(_, x) := pair in
   match exit_bound_witness x with
   | Some (a,b) => counter a b 1 ++ counter b a (-1)
   | None => []
   end) (PM.elements (amap other)).

(* A constant-step counter that starts at a fixed affine expression keeps
   an upper bound by that start when descending, or a lower bound when
   ascending. Both incoming states must establish the proposed relation;
   the final edge checker independently validates it across every iteration. *)
Definition start_bound_candidates (st other : astate) : list constraint :=
 flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match const_form x,get other r with Some a,Some (FForm b _ _) =>
     if negb (Z.eqb (fm a) 64 && Z.eqb (fm b) 64 &&
          negb (null (ft a)) && terms_eqb (ft a) (ft b) &&
          Z.eqb (flo b) (fhi b) && forallb (fun t=>is_fixed (fst (fst t))) (ft a))
     then [] else
     let delta := rep 64 (flo b-flo a) in
     if Z.eqb delta 0 then [] else
     let ts := sorted_terms ([((r,EX),1)] ++ tscale (-1) (ft a)) in
     let c := if delta <? 0 then (ts,flo a-(2^64-1),flo a)
              else (ts,flo a,flo a+(2^64-1)) in
     if cons_holds st c then if cons_holds other c then [c] else [] else []
   | _,_ => [] end) (PM.elements (amap st)).

(* Loop registers that start at affine forms of launch values and that the
   loop sets by selections or branches, as the bounds lo and hi of a binary
   search: each register of st with such a form that other does not give an
   affine form of launch values, as a search's update from its midpoint. *)
Definition relation_vars (st other : astate) : list (reg * form) :=
 flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match const_form x with
   | Some f =>
     if Z.eqb (fm f) 64 && forallb (fun t => is_fixed (fst (fst t))) (ft f) &&
        negb (match get other r with
              | Some (FForm g _ _) => Z.eqb (flo g) (fhi g) && forallb (fun t => is_fixed (fst (fst t))) (ft g)
              | _ => false end)
     then [(r,f)] else []
   | None => [] end) (PM.elements (amap st)).

(* A constraint of st relates register r to another register that no launch
   fixes, as the loop test lo < hi relates the bounds of a search. *)
Definition paired (st : astate) (r : reg) : bool :=
 existsb (fun k => let '(ts,_,_) := k in
   cons_mentions r k && existsb (fun t => negb (Pos.eqb (fst (fst t)) r) && negb (is_fixed (fst (fst t)))) ts)
   (acons st).

(* The first of the constraints ks that both states prove. *)
Definition first_held (st other : astate) (ks : list constraint) : list constraint :=
 match find (fun k => if cons_holds st k then cons_holds other k else false) ks with
 | Some k => [k] | None => [] end.

(* Such registers keep their start bounds, hi - n <= 0 for hi starting at
   n, of the value and of the signed reading, as the update hi := mid - 1 of
   a closed search can make hi -1. Propose each bound that both incoming
   states prove, at a loop head for a register that the incoming state
   relates to another loop register, and where a branch of a search sets the
   register from a midpoint; the final checker validates every edge of the
   invariant. *)
Definition bound_candidates (st other : astate) (loop_edge : bool) : list constraint :=
 flat_map (fun pair => let '(r,f) := pair in
   if null (ft f) || negb ((loop_edge && paired other r) || match get other r with
       | Some (FForm g _ _) => existsb (fun t => midpoint other (fst (fst t))) (ft g)
       | _ => false end) then [] else
   flat_map (fun e =>
     let ts := sorted_terms ([((r,e),1)] ++ tscale (-1) (ft f)) in
     first_held st other [upper_cons ts (fhi f)] ++ first_held st other [lower_cons ts (flo f)]) [EX; ES])
   (relation_vars st other).

(* At a loop head, two such registers that a constraint of the incoming
   state relates, as the loop test compares them, keep their order,
   lo - hi <= -1 or <= 0, of their values and of their signed readings; the
   strict order is proposed before the weak one. *)
Definition order_candidates (st other : astate) : list constraint :=
 let vars := relation_vars st other in
 flat_map (fun pair => let '(r,_) := pair in
   flat_map (fun pair' => let '(s,_) := pair' in
     if Pos.ltb r s && existsb (fun k => cons_mentions r k && cons_mentions s k) (acons other) then
       flat_map (fun e =>
         let ts := sorted_terms [((r,e),1);((s,e),-1)] in
         first_held st other [upper_cons ts (-1); upper_cons ts 0] ++
         first_held st other [lower_cons ts 1; lower_cons ts 0]) [EX; ES]
     else []) vars) vars.

(* Two loop registers that step by constants, one step m times the other,
   as a counter i stepped by 1 and its byte offset stepped by 8, or a
   pointer and an index: m * i - j keeps its value at the loop's entry,
   where one starts at a constant and the other at an affine form of
   registers the loop does not write, as the block's first element
   ctaid * BLOCK, so that the join's pivot cannot read one as the other's
   form. Propose that
   equality when both incoming states prove it; with the bound of j that
   its exit test gives, it bounds i, which the loop does not test. *)
Definition proportion_candidates (st other : astate) : list constraint :=
 let steps := flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match const_form x, get other r with
   | Some f, Some (FForm g _ _) =>
     if Z.eqb (fm f) 64 && Z.eqb (fm g) 64 && Z.eqb (flo g) (fhi g) && terms_eqb (ft f) (ft g) &&
        negb (Z.eqb (flo f) (flo g)) && negb (mentions f r)
     then [(r, f, flo g - flo f)] else []
   | _, _ => [] end) (PM.elements (amap st)) in
 flat_map (fun a => let '(r, f, dr) := a in
   flat_map (fun b => let '(s, g, ds) := b in
     if Pos.eqb r s || negb (Z.eqb (Z.rem ds dr) 0) || (Z.eqb ds dr && negb (Pos.ltb r s)) ||
        mentions f s || mentions g r || Bool.eqb (null (ft f)) (null (ft g)) then [] else
     let m := Z.quot ds dr in
     let ts := sorted_terms ([((r,EX),m); ((s,EX),-1)] ++ tscale (-m) (ft f) ++ ft g) in
     let v := m * flo f - flo g in
     let k := (ts, v, v) in
     if cons_holds st k then if cons_holds other k then [k] else [] else []) steps) steps.

(* A register that one incoming state may set to -1, as a search's result
   when it finds nothing, and the other to an index, as the position where
   it found the key: propose the bound of its signed reading by each
   bounded launch scalar, r - n <= -1 or <= 0, that both incoming states
   prove, so that the index bound survives the merge and the test r >= 0
   completes it. *)
Definition sentinel_candidates (st other : astate) : list constraint :=
 let scalars := flat_map (fun pair => let '(n,_) := pair in
   if is_fixed n && Z.eqb (wd n) 64 then [n] else []) (lf_bounds F) in
 let deep := fun s k => let '(ts,lo,hi) := k in
   match check_range s (form_of_terms ts) with
   | Some (p,q) => (lo <=? p) && (q <=? hi) | None => false end in
 let checks := fun k => if deep st k then deep other k else false in
 flat_map (fun pair => let '(r,x) := pair in
   if is_fixed r || negb (Z.eqb (wd r) 64) then [] else
   match x, get other r with
   | FForm f _ _, Some (FForm g _ _) =>
     if null (ft f) && Z.eqb (flo f) (-1) && negb (form_eqb f g) then
       flat_map (fun n =>
         let ts := sorted_terms [((r,ES),1);((n,EX),-1)] in
         match find checks [upper_cons ts (-1); upper_cons ts 0] with Some k => [k] | None => [] end) scalars
     else []
   | _, _ => [] end) (PM.elements (amap st)).

Definition clip_join_state (st : astate) : astate :=
 {|amap:=amap st;acons:=map clip_constraint (acons st)|}.

(* Batch implication work for each immutable side. Exact predicate eligibility
   avoids refining one side when the other cannot supply a constant bit. *)
Definition keep_job (widen_now:bool)(k:constraint)(s:astate):list constraint + (constraint*(bool+constraint)) :=
 let '(ts,lo,hi):=k in
 match filter(fun k'=>let '(ts',_,_):=k' in terms_eqb ts ts')(acons s) with
 | (_,lo0,hi0)::more=>
   let '(lo',hi'):=fold_left(fun acc k'=>let '(l,h):=acc in let '(_,l',h'):=k' in (Z.max l l',Z.min h h'))more(lo0,hi0) in
   inl(if widen_now && ((lo' <? lo)||(hi <? hi')) then [] else [(ts,Z.min lo lo',Z.max hi hi')])
 | []=>inr(k,cons_prepare s k) end.

Definition keep_run(s:astate)(check:form->option range)(job:list constraint+(constraint*(bool+constraint))):list constraint :=
 match job with inl result=>result|inr(k,prep)=>if prepared_run s check prep then [k] else [] end.

Definition keep_pending (jobs:list(list constraint+(constraint*(bool+constraint)))):bool :=
 existsb(fun job=>match job with inr(_,inr _)=>true|_=>false end)jobs.

Definition keep_apply(s:astate)(check:form->option range)(jobs:list(list constraint+(constraint*(bool+constraint)))):=flat_map(keep_run s check)jobs.

Definition prepared_keep_two(widen_now:bool)(s:astate)(left right:list constraint):list constraint*list constraint :=
 let left:=map(fun k=>keep_job widen_now k s)left in
 let right:=map(fun k=>keep_job widen_now k s)right in
 if keep_pending left || keep_pending right then
   let check:=shared_check_cached s in
   (keep_apply s check left,keep_apply s check right)
 else (keep_apply s (check_range s) left,keep_apply s (check_range s) right).

Definition moved_jobs(s:astate)(ks:list constraint):list constraint :=
 flat_map(fun k=>let '(ts,lo,hi):=k in let ts':=sorted_terms(copy_terms s ts) in
 if terms_eqb ts ts' then [] else [(ts',lo,hi)])ks.

Definition novel_jobs(old:astate)(ks:list constraint):list constraint :=
 filter(fun k=>let '(ts,_,_):=k in negb(existsb(fun k'=>let '(ts',_,_):=k' in terms_eqb ts ts')(acons old)))ks.

Definition join_kept_prepared(old out:astate)(widen_now loop_edge:bool):list constraint :=
 let old:=if widen_now then clip_join_state old else old in
 let out:=if widen_now then clip_join_state out else out in
 let left:=prepared_keep_two widen_now out (acons old)(if widen_now then [] else moved_jobs old(acons old))in
 let right:=prepared_keep_two widen_now old (if widen_now then [] else novel_jobs old(acons out))(if widen_now then [] else moved_jobs out(acons out))in
 fst left++fst right++
 (if widen_now then [] else snd left++snd right++
 pair_candidates old out++pair_candidates out old++
 (if loop_edge then [] else exit_sum_candidates old out++exit_sum_candidates out old++sentinel_candidates old out++sentinel_candidates out old)++
 (if loop_edge then signed_candidates old out++signed_candidates out old++triple_candidates old out++triple_candidates out old++moving_bound_candidates old out++moving_bound_candidates out old++order_candidates old out else[])++bound_candidates old out loop_edge++proportion_candidates old out++
 exit_bound_candidates old out++start_bound_candidates old out).

Definition join_bit_possible(st:astate)(p:reg):bool :=
 match known_bit st p with Some _=>true |None=>
 if negb(Z.eqb(wd p)1) then false else
 match get st p with Some(FPred _ true)|Some(FNeg _ true)=>true|_=>false end end.

Definition join_predicate_bits_ready(old out:astate)(p:reg)(j:option fact):option fact :=
 if negb(Z.eqb(wd p)1) then j else
 let loses:=match j with None|Some(FPred _ false)|Some(FNeg _ false)=>true|_=>false end in
 if negb loses then j else
 if join_bit_possible old p && join_bit_possible out p then
 match join_bit old p,join_bit out p with Some c,Some c'=>
 if Z.eqb c c' then Some(FForm(fconst 1 c)0 None)else j |_,_=>j end
 else j.

Definition join_states_with (extra : list Z) (old out : astate) (widen_now loop_edge : bool) : astate :=
  let kept := join_kept_prepared old out widen_now loop_edge in
  let old := if widen_now then clip_join_state old else old in
  let out := if widen_now then clip_join_state out else out in
  (* registers that step by a constant, and the others *)
  let both := PM.fold (fun r f acc => match PM.find r (amap out) with Some g => (r, f, g) :: acc | None => acc end)
                      (amap old) [] in
  (* one side's state implies the other side's fact, as when a merge point
     meets the first pass and the generalized loop: join_fact keeps the
     weaker fact, and no step is proposed *)
  let implied := fun (f g : fact) => fact_le out g f || fact_le old f g in
  (* widening from an old fact that is a form with terms starts from the
     range of the form, as when an inner loop head, visited often in the
     first pass, meets the generalized outer loop *)
  let widen_old := fun (f jf : fact) =>
    match f with
    | FForm a al _ =>
        if null (ft a) then widen extra f jf else
        match form_range old a with
        | Some (p, q) =>
            match jf with
            | FForm b bl _ =>
                match form_range old b, form_range out b with
                | Some (l,h), Some (l',h') =>
                    widen extra (FForm (range_form (fm a) p q) al None)
                          (FForm (range_form (fm b) (Z.min l l') (Z.max h h')) bl None)
                | _,_ => None end
            | _ => None end
        | None => None
        end
    | _ => widen extra f jf
    end in
  let is_step := fun '(r, f, g) =>
    match const_form f, const_form g with
    | Some a, Some b =>
        if Z.eqb (fm a) (fm b) && terms_eqb (ft a) (ft b) && negb (Z.eqb (flo a) (flo b)) &&
           (negb (null (ft a)) || Z.eqb (fm a) 64)
        then negb (implied f g) else false
    | _, _ => false
    end in
  let joined := fold_left (fun m '(r, f, g) =>
                  if is_step (r, f, g) then m else
                  let j := join_fact out old (Some g) (Some f) in
                  let j := join_predicate_bits_ready old out r j in
                  let j := match j with None =>
                    if predicate_fact_le out r f then Some f else
                    if predicate_fact_le old r g then Some g else None
                    | _ => j end in
                  (* two forms of different terms: the range of both *)
                  let j := match j, f, g with
                           | None, FForm a _ _, FForm b _ _ =>
                               match form_range old a, form_range out b with
                               | Some (p, q), Some (p', q') =>
                                   Some (FForm (range_form (Z.max 0 (Z.min (fm a) (fm b))) (Z.min p p') (Z.max q q')) 0 None)
                               | _, _ => None
                               end
                           | _, _, _ => j
                           end in
                  let j := match j with
                           | Some (FForm _ _ _ as jf) => if widen_now && negb (fact_le old jf f) then widen_old f jf else j
                           | Some _ => j
                           | None => None
                           end in
                  match j with Some jf => PM.add r jf m | None => m end)
                  both (PM.empty fact) in
  let steps := flat_map (fun '(r, f, g) =>
                  if is_step (r, f, g) then
                    match const_form f, const_form g with
                    | Some a, Some b => [(r, a, b, rep (fm a) (flo b - flo a))]
                    | _, _ => []
                    end
                  else []) both in
  (* Prefer 64-bit pivots with fewer symbolic start terms, then the least
     step. A tile cursor starting at tid keeps a tighter range than a global
     cursor starting at block * tile + tid - window. Their affine relation
     still describes both pointers, and check_thread validates the result. *)
  let key := fun '(r, a, _, s) => (Z.of_nat (length (ft a)), (if Z.eqb (fm a) 64 then 0 else 1), Z.abs s, r) in
  let lt := fun x y => let '(a1, n1, b1, c1) := key x in let '(a2, n2, b2, c2) := key y in
                       (a1 <? a2) || (Z.eqb a1 a2 && ((n1 <? n2) || (Z.eqb n1 n2 &&
                         ((b1 <? b2) || (Z.eqb b1 b2 && Pos.ltb c1 c2))))) in
  let ins := fix ins x l := match l with [] => [x] | y :: l' => if lt x y then x :: l else y :: ins x l' end in
  let can_pivot := fun a b =>
    if Z.eqb (fm a) 64 then true else
    if negb (Z.eqb (fm a) 32) then false else
    match form_range_near old a, form_range_near out b with
    | Some (lo,hi),Some (lo',hi') => (0 <=? lo) && (hi <? 2^31) && (0 <=? lo') && (hi' <? 2^31)
    | _,_ => false end in
  (* A pointer with no numeric range must not become the pivot of a bounded
     countdown merely because its register number is smaller. Evaluate each
     step's ranges once, then retain the established ordering within groups. *)
  let '(bounded_steps, unbounded_steps) := partition (fun st4 =>
    let '(_, a, b, _) := st4 in
    is_some (form_range old a) && is_some (form_range out b)) steps in
  let sorted := fold_right ins [] bounded_steps ++ fold_right ins [] unbounded_steps in
  let '(m, _) :=
    fold_left (fun acc st4 =>
      let '(m, pivots) := acc in
      let '(r, a, b, s) := st4 in
      match find (fun p => let '(_, _, _, sp) := p in Z.eqb (Z.rem s sp) 0) pivots with
      | Some (y, ay, _, sy) =>
          let c := Z.quot s sy in
          let ay := {|fm:=64;ft:=ft ay;flo:=flo ay;fhi:=fhi ay|} in
          (PM.add r (FForm (fat (fadd (fadd a (fscale ay (- c) 64)) (fatom 64 (y, if Z.eqb (wd y) 32 then EZ else EX) c)) (fm a)) 0 None) m, pivots)
      | None =>
          if negb (can_pivot a b) then
            let j := join_fact out old (Some (FForm b 0 None)) (Some (FForm a 0 None)) in
            let j := match j with
                     | Some (FForm _ _ _ as jf) => if widen_now && negb (fact_le old jf (FForm a 0 None)) then widen_old (FForm a 0 None) jf else j
                     | Some _ => j
                     | None => None
                     end in
            (match j with Some jf => PM.add r jf m | None => m end, pivots)
          else
            let k := Z.min (form_align old a) (v2 s) in
            (* the pivot keeps the hull of its ranges, as a loop index j that
               starts at lane + start keeps j >= 0 *)
            let x := match form_range old a, form_range out b with
                     | Some (p, q), Some (p', q') =>
                         let f := FForm (range_form (fm a) (Z.min p p') (Z.max q q')) k None in
                         if widen_now then
                           match widen extra (FForm (range_form (fm a) p q) 0 None) f with
                           | Some (FForm g _ _) => Some (FForm g k None)
                           | _ => if 0 <? k then Some (FAlign k) else None
                           end
                         else Some f
                     | _, _ => if 0 <? k then Some (FAlign k) else None
                     end in
            (match x with Some y => PM.add r y m | None => m end, pivots ++ [st4])
      end) sorted (joined, []) in
  (* 64-bit registers that step by a form, such as a grid-stride index
     i := i + ntid * nctaid and the pointers the compiler advances with it:
     a register whose step is c times a pivot's is its old form less c times
     the pivot's old form, plus c times the pivot *)
  let fsteps := flat_map (fun '(r, f, g) =>
                  match const_form f, const_form g with
                  | Some a, Some b =>
                      if (if Z.eqb (fm a) 64 && Z.eqb (fm b) 64 && negb (terms_eqb (ft a) (ft b))
                          then negb (implied f g) else false) then
                        let D := fadd b (fscale a (-1) 64) in
                        if null (ft D) then [] else [(r, a, D)]
                      else []
                  | _, _ => []
                  end) both in
  (* the pivots first: a register that a constraint mentions, then the least
     step. A range loop keeps its position and the next position, and the
     loop test bounds the position, so the next position is read through it *)
  let bound := fun r => existsb (cons_mentions r) (acons old ++ acons out) in
  let fkey := fun '(r, _, D) => ((if bound r then 0 else 1), match ft D with (_, c) :: _ => Z.abs c | [] => 0 end, r) in
  let flt := fun x y => let '(a1, b1, c1) := fkey x in let '(a2, b2, c2) := fkey y in
                        (a1 <? a2) || (Z.eqb a1 a2 && ((b1 <? b2) || (Z.eqb b1 b2 && Pos.ltb c1 c2))) in
  let fins := fix fins x l := match l with [] => [x] | y :: l' => if flt x y then x :: l else y :: fins x l' end in
  let '(m, _) :=
    fold_left (fun acc st3 =>
      let '(m, pivots) := acc in
      let '(r, a, D) := st3 in
      match find (fun p => let '(_, _, Dp) := p in is_some (step_ratio D Dp)) pivots with
      | Some (y, ay, Dy) =>
          match step_ratio D Dy with
          | Some c => (PM.add r (FForm (fadd (fadd a (fscale ay (- c) 64)) (fatom 64 (y, EX) c)) 0 None) m, pivots)
          | None => (m, pivots)
          end
      | None => (m, pivots ++ [st3])
      end) (fold_right fins [] fsteps) (m, []) in
  (* each constraint once: a loop's joins would otherwise gather copies *)
  let same := fun (k l : constraint) =>
    let '(ts, lo, hi) := k in let '(ts', lo', hi') := l in terms_eqb ts ts' && Z.eqb lo lo' && Z.eqb hi hi' in
  recover_cursor_ranges ({| amap := m; acons := fold_right (fun k acc => if existsb (same k) acc then acc else k :: acc) [] kept |}).

Definition join_states (old out : astate) (widen_now loop_edge : bool) : astate :=
  join_states_with [] old out widen_now loop_edge.

Definition fact_eqb (x y : fact) : bool :=
  match x, y with
  | FForm f al vr, FForm g bl vr' =>
      form_eqb f g && Z.eqb al bl &&
      match vr, vr' with Some (a, b), Some (c, d) => Z.eqb a c && Z.eqb b d | None, None => true | _, _ => false end
  | FAlign k, FAlign k' => Z.eqb k k'
  | FPred cs e, FPred cs' e' | FNeg cs e, FNeg cs' e' =>
      Bool.eqb e e' && Nat.eqb (length cs) (length cs') && forallb (fun p => cmpf_eqb (fst p) (snd p)) (combine cs cs')
  | _, _ => false
  end.

Definition state_eqb (a b : astate) : bool :=
  Nat.eqb (PM.cardinal (amap a)) (PM.cardinal (amap b)) &&
  PM.fold (fun r x acc => acc && match PM.find r (amap b) with Some y => fact_eqb x y | None => false end) (amap a) true &&
  Nat.eqb (length (acons a)) (length (acons b)) &&
  forallb (fun p => let '((t, lo, hi), (t', lo', hi')) := p in terms_eqb t t' && Z.eqb lo lo' && Z.eqb hi hi') (combine (acons a) (acons b)).

Definition LIMIT : nat := 80.

Fixpoint set_nth {A} (l : list A) (n : nat) (x : A) : list A :=
  match l, n with
  | [], _ => []
  | _ :: l', O => x :: l'
  | y :: l', S n' => y :: set_nth l' n' x
  end.

(* The worklist in block order, without repeats. Taking the first block
   first lets an inner loop reach its invariant before the blocks after it
   see its exit state. *)
Fixpoint insert_block (v : nat) (l : list nat) : list nat :=
  match l with
  | [] => [v]
  | x :: l' => if Nat.eqb v x then l else if Nat.ltb v x then v :: l else x :: insert_block v l'
  end.

(* Backward CFG liveness, computed once per analysis and once per check of
   a certificate's cases. Every instruction read is a root even when its
   result is unused. *)
Definition union_register_marks (a b : PM.t unit) : PM.t unit :=
  PM.fold (fun r _ m => PM.add r tt m) a b.

Definition live_cell_roots : list reg :=
  flat_map (fun c => let '(p, u, _) := c in [p; u]) (lf_cells F).

Definition live_before (i : instr) (after : PM.t unit) : PM.t unit :=
  mark_regs (instruction_inputs i)
    (fold_left (fun m r => PM.remove r m) (instruction_definite_writes i) after).

Definition live_successors (prog : program) (b : nat) : list nat :=
  match last_instr (block_body prog b) with
  | Some (IBra None t) => [t]
  | Some (IBra (Some _) t) => [t; S b]
  | Some (IExit None) => []
  | _ => [S b]
  end.

(* With no internal CFG edge, no incoming state can reach the boundary
   transform. The exact fuel/worklist result below avoids discarded
   transfer work; the certificate checker still checks every instruction. *)
Definition no_internal_edges (prog : program) : bool :=
 forallb (fun b => forallb (fun v => negb (Nat.ltb v (length prog)))
   (live_successors prog b)) (seq 0 (length prog)).

(* The registers live at the end of block b, from the registers live at
   the start of each block. *)
Definition live_out (prog : program) (table : list (PM.t unit)) (b : nat) : PM.t unit :=
  mark_regs live_cell_roots
    (fold_left (fun m t => union_register_marks (nth t table (PM.empty unit)) m)
      (live_successors prog b) (PM.empty unit)).

Definition live_round (prog : program) (table : list (PM.t unit)) : list (PM.t unit) :=
  map (fun b => fold_right live_before (live_out prog table b) (block_body prog b))
    (seq 0 (length prog)).

Fixpoint live_rounds (fuel : nat) (prog : program) (table : list (PM.t unit))
    : list (PM.t unit) :=
  match fuel with
  | O => table
  | S fuel =>
      let next := live_round prog table in
      if forallb (fun pair => register_marks_eqb (fst pair) (snd pair)) (combine table next)
      then table else live_rounds fuel prog next
  end.

Definition cfg_live_program (prog : program) : list (PM.t unit) :=
  live_rounds (S (length prog)) prog (repeat (PM.empty unit) (length prog)).

(* Abstract normalization can erase the operands of a constant. Later
   joins still need their old values to recover a relation. Keep the inputs
   of every possible definition of a live register, including definitions
   from earlier loop iterations. This only selects whole facts to retain. *)
Definition instruction_support_outputs (i : instr) : list reg :=
  let cells := map (fun c => let '(p, _, _) := c in p) (lf_cells F) in
  match i with
  | IStL _ _ _ _ _ _ => cells
  | IMem _ KLoad _ _ _ _ _ _ _ => instruction_outputs i
  | IMem _ _ true _ _ _ _ _ _ => instruction_outputs i ++ cells
  | _ => instruction_outputs i
  end.

Definition instruction_support_graph (prog : program) : PM.t (list reg) :=
  fold_left (fun graph i =>
    fold_left (fun graph d =>
      let previous := match PM.find d graph with Some rs => rs | None => [] end in
      PM.add d (instruction_inputs i ++ previous) graph)
      (instruction_support_outputs i) graph) (concat prog) (PM.empty (list reg)).

(* Record incoming control-predicate IDs at their actual successors. The
   boundary later selects only current facts consumed by join recovery.
   Predicated exits contribute only their fallthrough edge. *)
Definition live_incoming_controls (prog : program) : list (PM.t unit) :=
  fold_left (fun table b =>
    let guard := match last_instr (block_body prog b) with
      | Some (IBra (Some (_,p)) _) | Some (IExit (Some (_,p))) => Some p
      | _ => None end in
    match guard with
    | None => table
    | Some p => fold_left (fun table target =>
        set_nth table target (PM.add p tt (nth target table (PM.empty unit))))
        (live_successors prog b) table
    end) (seq 0 (length prog)) (repeat (PM.empty unit) (length prog)).

Definition live_program (prog : program) : list (PM.t unit) :=
  let close := register_closure (instruction_support_graph prog) in
  (* Guard-refined facts about fixed launch registers remain useful to
     contract bounds after the corresponding runtime test becomes dead. *)
  map (fun keep => mark_regs (lf_fixed F) (close (map fst (PM.elements keep))))
    (cfg_live_program prog).

Definition fact_inputs (f : fact) : list reg :=
  match f with
  | FForm x _ _ => map (fun term => fst (fst term)) (ft x)
  | FPred cs _ | FNeg cs _ => flat_map (fun c => ca c :: opnd_inputs (cb c)) cs
  | FAlign _ => []
  end.

Definition constraint_inputs (k : constraint) : list reg :=
  let '(ts, _, _) := k in map (fun term => fst (fst term)) ts.

Definition live_constraint_touches (keep : PM.t unit) (k : constraint) : bool :=
  existsb (fun r => negb (is_fixed r) && PM.mem r keep) (constraint_inputs k).

Definition live_dependency_round (st : astate) (keep : PM.t unit) : PM.t unit :=
  let by_facts := PM.fold (fun r f m =>
    if PM.mem r keep then mark_regs (fact_inputs f) m else m) (amap st) keep in
  fold_left (fun m k => if live_constraint_touches by_facts k
    then mark_regs (constraint_inputs k) m else m) (acons st) by_facts.

Fixpoint live_dependency_closure (fuel : nat) (st : astate) (keep : PM.t unit)
    : PM.t unit :=
  match fuel with
  | O => keep
  | S fuel =>
      let next := live_dependency_round st keep in
      if register_marks_eqb keep next then keep else live_dependency_closure fuel st next
  end.

Definition trim_live_seed (seed : PM.t unit) (st : astate) : astate :=
  let keep := live_dependency_closure
    (S (PM.cardinal (amap st) + length (acons st))) st seed in
  whole_state_filter keep
    (fun k => forallb (fun r => is_fixed r || PM.mem r keep) (constraint_inputs k)) st.

Definition trim_live_state (table : list (PM.t unit)) (b : nat) (st : astate) : astate :=
  trim_live_seed (nth b table (PM.empty unit)) st.

(* Execution dependencies already have static definition-history support.
   Additional join witnesses use only their current whole fact and current
   dynamic dependencies, not all past definitions of a dead predicate. *)
Definition live_join_seed (table incoming : list (PM.t unit)) (b : nat) (st : astate)
    : PM.t unit :=
  PM.fold (fun p _ keep =>
    match get st p with
    | Some f => match exit_bound_witness f with
      | Some _ => PM.add p tt keep
      | None => keep
      end
    | None => keep
    end) (nth b incoming (PM.empty unit)) (nth b table (PM.empty unit)).

(* An interval temporary can mediate a bound between fixed launch inputs
   after its last executable use. Retain that whole fact and its original
   constraints when it has a checked-domain affine launch-bound shape.
   No historical definitions or other dead dynamic components are added. *)
Definition affine_bridge_candidate (st retained : astate) (r : reg) : bool :=
 if is_fixed r || is_some (get retained r) then false else
 match get st r with
 | Some (FForm f _ None) =>
   if null (ft f) && negb (Z.eqb (flo f) (fhi f)) then
   existsb (fun k => let '(ts,_,_) := k in
     affine_cursor_bound ts && cons_mentions r k &&
     forallb (fun t => is_fixed (fst (fst t)) || Pos.eqb r (fst (fst t))) ts)
     (acons st)
   else false
 | _ => false end.

Definition retain_affine_bridges (st retained : astate) : astate :=
 let extras := filter (affine_bridge_candidate st retained) (map fst (PM.elements (amap st))) in
 match extras with
 | [] => retained
 | _ => let keep := mark_regs extras (PM.map (fun _ => tt) (amap retained)) in
   whole_state_filter keep
     (fun k => forallb (fun r => is_fixed r || PM.mem r keep) (constraint_inputs k)) st
 end.

Definition trim_join_live_state (table incoming : list (PM.t unit)) (b : nat) (st : astate)
    : astate :=
  retain_affine_bridges st (trim_live_seed (live_join_seed table incoming b st) st).

(* The changes of each block's entry state that the iteration allows:
   LIMIT for each edge into the block, since each edge can change the state
   of a block that joins several. *)
Definition revisit_limits (prog : program) : list nat :=
  map (fun d => (LIMIT * Nat.max 1 d)%nat)
    (fold_left (fun acc v => if Nat.ltb v (length prog) then set_nth acc v (S (nth v acc O)) else acc)
       (flat_map (edge_targets prog) (seq 0 (length prog))) (repeat O (length prog))).

(* A worklist iteration from the entry block. Returns None if some block's
   entry state changes more often than its limit allows. *)
Fixpoint iterate_with_boundary (boundary : nat -> astate -> astate) (fuel : nat) (prog : program) (limits : list nat) (entry : list (option astate)) (visits : list nat) (work : list nat)
    : option (list (option astate)) :=
  match fuel with
  | O => None
  | S fuel' =>
      match work with
      | [] => Some entry
      | b :: work' =>
          let st := match nth b entry None with Some s => s | None => empty_state end in
          let out := walk st (block_body prog b) in
          let '(entry, visits, work, ok) :=
            fold_left (fun acc e =>
              let '(entry, visits, work, ok) := acc in
              let '(v, s) := e in
              match s with
              | None => acc
              | Some s =>
                  if negb (Nat.ltb v (length prog)) then acc else
                  let s := boundary v s in
                  let old := nth v entry None in
                  let new := match old with
                             | None => s
                             | Some o => join_states_with (program_thresholds prog) o s (Nat.ltb WIDEN_AFTER (nth v visits O)) (Nat.leb v b)
                             end in
                  let changed := match old with None => true | Some o => negb (state_eqb new o) end in
                  if changed then
                    let n := S (nth v visits O) in
                    (set_nth entry v (Some new), set_nth visits v n, insert_block v work, ok && Nat.leb n (nth v limits O))
                  else acc
              end) (edge_states prog b out) (entry, visits, work', true) in
          if ok then iterate_with_boundary boundary fuel' prog limits entry visits work else None
      end
  end.

(* The boundary transform is selected once. Disabling trimming supplies
   the identity to this same iterator; it does not retry a failed analysis. *)
Definition iterate (fuel : nat) (prog : program) (entry : list (option astate))
    (visits : list nat) (work : list nat) : option (list (option astate)) :=
  if no_internal_edges prog then
    if Nat.ltb (length work) fuel then Some entry else None
  else
  let incoming := live_incoming_controls prog in
  let lives := live_program prog in
  iterate_with_boundary (trim_join_live_state lives incoming) fuel prog (revisit_limits prog) entry visits work.

(* Candidate recovery across cycles whose instructions leave a relation's
   registers unchanged. The final checker still validates every edge. *)
Definition relation_unchanged (k : constraint) (i : instr) : bool :=
 match i with
 | IDef _ ds | IMem _ _ _ _ _ _ _ _ ds => forallb (fun r => negb (cons_mentions r k)) ds
 | ILdL _ _ _ _ _ _ | IStL _ _ _ _ _ _ | IFrame _ => false
 | _ => match dest i with Some r => negb (cons_mentions r k) | None => true end
 end.
Definition recover_once (prog : program) (inv : list astate) : list astate :=
 let n := length prog in
 let incoming := fold_left (fun incoming b =>
   let st := inv_at inv b in
   if is_bottom st || leaves_program prog b then incoming else
   fold_left (fun incoming edge => let '(v,s) := edge in
     match s with None => incoming | Some s =>
       if Nat.ltb v n then set_nth incoming v ((b,s) :: nth v incoming []) else incoming end)
     (edge_states prog b (walk st (block_body prog b))) incoming)
   (seq 0 n) ([(0%nat,empty_state)] :: repeat [] (n - 1)) in
 map (fun v => let old := inv_at inv v in let sources := nth v incoming [] in
   let new := filter (fun k => negb (existsb (fun k' =>
      let '(ts,lo,hi) := k in let '(ts',lo',hi') := k' in
      terms_eqb ts ts') (acons old)))
      (flat_map (fun pair => acons (snd pair)) sources) in
   let checked := filter (fun k => short_all (fun pair => let '(b,st) := pair in
      if cons_holds st k then true else
      if Nat.leb v b && negb (Nat.eqb v O) then
        short_all (fun idx => short_all (relation_unchanged k) (block_body prog idx)) (seq v (S b - v))
      else false) sources) new in
   {| amap := match sources with
               | [(b,st)] => if Nat.ltb b v then amap st else amap old
               | _ => amap old end;
      acons := acons old ++ checked |}) (seq 0 n).
Fixpoint recover_n (fuel : nat) (prog : program) (inv : list astate) : list astate :=
 match fuel with O => inv | S fuel => let next := recover_once prog inv in
  if forallb (fun pair => state_eqb (fst pair) (snd pair)) (combine inv next)
  then inv else recover_n fuel prog next end.

(* Drop unsupported facts after recovering relational invariants. The
   complete checker still proves all remaining facts and every access. *)
Definition prune_once (prog : program) (inv : list astate) : list astate :=
 let n := length prog in
 let incoming := fold_left (fun incoming b =>
   let st := inv_at inv b in if is_bottom st || leaves_program prog b then incoming else
   fold_left (fun incoming edge => let '(v,s) := edge in
     match s with None => incoming | Some s =>
       if Nat.ltb v n then set_nth incoming v (s :: nth v incoming []) else incoming end)
     (edge_states prog b (walk st (block_body prog b))) incoming)
   (seq 0 n) ([empty_state] :: repeat [] (n - 1)) in
 map (fun pair => let '(old,sources) := pair in
   {| amap := PM.fold (fun r g m =>
       if short_all (fun st => register_fact_le st r g) sources
       then PM.add r g m else m) (amap old) (PM.empty fact);
      acons := filter (fun k => short_all (fun st => cons_holds st k) sources) (acons old) |})
   (combine inv incoming).
Fixpoint prune_n (fuel : nat) (prog : program) (inv : list astate) : list astate :=
 match fuel with O => inv | S fuel => let next := prune_once prog inv in
  if forallb (fun pair => state_eqb (fst pair) (snd pair)) (combine inv next)
  then inv else prune_n fuel prog next end.

(* Recover ordinary loop relations, including loops that stage and search
   launch-sized shared arrays. Shared loads remain arbitrary values; only
   register relations validated on every edge survive. Static shared layouts
   and local cells still require separate summaries. *)
Definition ordinary_loop_recovery (prog : program) : bool :=
 let body := concat prog in
 existsb (fun i => match i with ICvt _ 64 32 _ _ _ => true | _ => false end) body &&
 forallb (fun i => match i with
   | ILdS _ _ _ _ _ | IShA _ _ _ _ _ _ => existsb (fun p => negb (null (ft (snd p)))) (lf_shared F)
   | ILdL _ _ _ _ _ _ | IStL _ _ _ _ _ _ | IFrame _ => false
   | _ => true end) body.

Definition analyze (prog : program) : option (list astate) :=
  let n := length prog in
  match iterate (S (fold_left Nat.add (revisit_limits prog) O)) prog (Some empty_state :: repeat None (n - 1)) (repeat O n) [O] with
  (* a block that no edge reaches is never reached *)
  | Some entry => let inv := map (fun s => match s with Some s => s | None => bottom_state end) entry in
      Some (if ordinary_loop_recovery prog then prune_n 4 prog (recover_n 4 prog inv) else inv)
  | None => None
  end.

(* For diagnosis: the positions of accesses the check rejects. *)
Fixpoint walk_fail (st : astate) (b i : nat) (body : list instr) : list (nat * nat) :=
  match body with
  | [] => []
  | ins :: rest =>
      (if wf_instr ins && access_of st ins then [] else [(b, i)]) ++ walk_fail (transfer st ins) b (S i) rest
  end.


(* ------------------------------------------------------------------ *)
(* The check of a certificate's cases.                                 *)

(* The walk of a long block keeps a fact for every register it writes,
   and most of those registers are not read again. After every TRIM_EVERY
   instructions the checked walk keeps the facts of a set of registers and
   of the registers they depend on (trim_live_seed): the registers live
   after the instruction, those that the cases of the block's successors
   name, and the fixed registers. Dropping facts only weakens a state, so
   every set is sound (trim_live_seed_sound); the set decides which checks
   can succeed. *)
Definition TRIM_EVERY : nat := 16.

(* A block body with the registers that the walk keeps after each
   instruction, or None where it keeps every fact. *)
Definition walk_plan : Type := list (instr * option (PM.t unit)).

Definition trim_after (keep : option (PM.t unit)) (st : astate) : astate :=
  match keep with Some k => trim_live_seed k st | None => st end.

Fixpoint plan_walk (st : astate) (plan : walk_plan) : astate :=
  match plan with
  | [] => st
  | (ins, keep) :: rest => plan_walk (trim_after keep (transfer st ins)) rest
  end.

(* Check each instruction while computing its successor state once. The
   resulting state is reused for the outgoing-edge check. *)
Fixpoint checked_walk (st : astate) (plan : walk_plan) : option astate :=
  match plan with
  | [] => Some st
  | (ins, keep) :: rest =>
      if wf_instr ins then
        if access_of st ins then checked_walk (trim_after keep (transfer st ins)) rest else None
      else None
  end.

(* The registers that a state names: those it has facts of, and those
   that its facts and constraints read. *)
Definition state_regs (st : astate) : list reg :=
  PM.fold (fun r f acc => r :: fact_inputs f ++ acc) (amap st) [] ++
  flat_map constraint_inputs (acons st).

(* The registers that the walk of block b keeps besides the live ones:
   those that the cases of its successors name, which its edge checks
   compare with, and the fixed registers. *)
Definition plan_keep (prog : program) (inv : list (list astate)) (b : nat) : PM.t unit :=
  fold_left (fun m v => fold_left (fun m st => mark_regs (state_regs st) m) (cases_at inv v) m)
    (edge_targets prog b) (mark_regs (lf_fixed F) (PM.empty unit)).

(* The plan of a body that ends with the registers out live: after every
   TRIM_EVERY-th instruction, keep and the registers live after it; with
   the registers live at the body's start. left counts the instructions to
   the next trim. The walk keeps every fact after the last instruction,
   whose state the edge checks refine by its guard. *)
Fixpoint plan_from (left : nat) (keep out : PM.t unit) (body : list instr)
    : PM.t unit * walk_plan :=
  match body with
  | [] => (out, [])
  | ins :: rest =>
      let '(after, plan) :=
        plan_from (match left with S (S l) => S l | _ => TRIM_EVERY end) keep out rest in
      (live_before ins after,
       (ins, match left, rest with
             | S (S _), _ | _, [] => None
             | _, _ :: _ => Some (union_register_marks keep after)
             end) :: plan)
  end.

Definition block_plan (table : list (PM.t unit)) (prog : program) (inv : list (list astate))
    (b : nat) : walk_plan :=
  snd (plan_from TRIM_EVERY (plan_keep prog inv b) (live_out prog table b) (block_body prog b)).

(* The plan of each block of the program, from its CFG liveness. *)
Definition case_plan (prog : program) (inv : list (list astate)) : nat -> walk_plan :=
  block_plan (cfg_live_program prog) prog inv.

(* Every state at the end of block b implies one of the target block's
   cases. *)
Definition case_edges_ok (prog : program) (inv : list (list astate)) (b : nat) (out : astate) : bool :=
 forallb (fun e => let '(v,s) := e in match s with
 | None => true
 | Some s => if Nat.ltb v (length prog) then existsb (state_le s) (cases_at inv v) else true
 end) (edge_states prog b out).
Definition case_block_ok (prog : program) (inv : list (list astate)) (plan : walk_plan) (b : nat)
    (st : astate) : bool :=
 if is_bottom st then forallb wf_instr (block_body prog b)
 else match checked_walk st plan with
 | None => false
 | Some out => case_edges_ok prog inv b out
 end.
Definition case_facts_ok (prog : program) (inv : list (list astate)) (plan : walk_plan) (b : nat)
    (st : astate) : bool :=
 forallb wf_instr (block_body prog b) && (is_bottom st || case_edges_ok prog inv b (plan_walk st plan)).
Definition check_cases (prog : program) (inv : list (list astate)) : bool :=
 lf_ok && structured prog && existsb entry_case (cases_at inv 0) &&
 (let plan := case_plan prog inv in
  short_all (fun b => short_all (case_block_ok prog inv (plan b) b) (cases_at inv b)) (seq 0 (length prog))).
Definition check_case_facts (prog : program) (inv : list (list astate)) : bool :=
 lf_ok && structured prog && existsb entry_case (cases_at inv 0) &&
 (let plan := case_plan prog inv in
  short_all (fun b => short_all (case_facts_ok prog inv (plan b) b) (cases_at inv b)) (seq 0 (length prog))).
Definition check_thread (prog : program) (inv : list astate) : bool :=
 check_cases prog (map (fun st => [st]) inv).
Definition check_facts (prog : program) (inv : list astate) : bool :=
 check_case_facts prog (map (fun st => [st]) inv).

End Domain.
(* Pointwise equality preserves the original exit-bound proposal policy. *)
Lemma exit_bound_witness_exact : forall (counter : reg -> reg -> Z -> list constraint) x,
 (match exit_bound_witness x with
  | Some (a,b) => counter a b 1 ++ counter b a (-1)
  | None => [] end) =
 (match x with
  | FPred [k] _ =>
      match cc k, Nat.eqb (cw k) 64, cb k with
      | (CEq | CNe), true, OReg rb => counter rb (ca k) 1 ++ counter (ca k) rb (-1)
      | _, _, _ => [] end
  | _ => [] end).
Proof.
 intros counter x. destruct x as [f al vr|al|cs exact|cs exact];try reflexivity.
 destruct cs as [|k cs];[reflexivity|]. destruct cs;[|reflexivity].
 unfold exit_bound_witness. destruct (cc k),(Nat.eqb (cw k) 64),(cb k);reflexivity.
Qed.


Lemma edge_targets_exact : forall F prog b out,
 map fst (edge_states F prog b out) = live_successors prog b.
Proof.
 intros. unfold edge_states,live_successors.
 destruct (last_instr (block_body prog b)) as [i|]; [|reflexivity].
 destruct i; try reflexivity; destruct g as [[pol p]|]; reflexivity.
Qed.
Lemma no_internal_edges_outside : forall F prog,
 no_internal_edges prog = true -> forall b out v s,
 In (v,s) (edge_states F prog b out) -> Nat.ltb v (length prog) = false.
Proof.
 intros F prog Hnone b out v s Hin.
 destruct (Nat.lt_ge_cases b (length prog)) as [Hvalid|Hinvalid].
 - unfold no_internal_edges in Hnone.
   rewrite forallb_forall in Hnone.
   specialize (Hnone b ltac:(apply in_seq;lia)).
   rewrite forallb_forall in Hnone.
   assert (Hv: In v (live_successors prog b)).
   { rewrite <- (edge_targets_exact F prog b out).
     apply in_map_iff. exists (v,s). split;[reflexivity|assumption]. }
   apply negb_true_iff. apply Hnone. exact Hv.
 - assert (Hbody:block_body prog b=[]).
   { unfold block_body. rewrite (proj2 (nth_error_None prog b) Hinvalid). reflexivity. }
   unfold edge_states in Hin. rewrite Hbody in Hin. cbn [last_instr] in Hin.
   destruct Hin as [Hin|[]]. injection Hin as <- _.
   apply Nat.ltb_ge. lia.
Qed.
Lemma fold_left_noop : forall A B (f:A->B->A) xs,
 (forall acc x, In x xs -> f acc x=acc) -> forall acc, fold_left f xs acc=acc.
Proof.
 intros A B f xs. induction xs as [|x xs IH];intros H acc;[reflexivity|].
 cbn. rewrite (H acc x (or_introl eq_refl)). apply IH.
 intros a y Hy. apply H. right. exact Hy.
Qed.
Lemma no_internal_edges_iteration : forall F prog,
 no_internal_edges prog=true -> forall boundary fuel limits entry visits work,
 iterate_with_boundary F boundary fuel prog limits entry visits work =
 if Nat.ltb (length work) fuel then Some entry else None.
Proof.
 intros F prog Hnone boundary fuel limits. induction fuel as [|fuel IH];intros entry visits work.
 - reflexivity.
 - destruct work as [|b work];[reflexivity|].
   cbn [iterate_with_boundary].
   match goal with
   | |- context [fold_left ?f ?xs ?acc] =>
     assert (Hfold:fold_left f xs acc=acc)
       by (apply fold_left_noop;intros [[[es vs] ws] ok] [v [s|]] Hin;
           [cbn beta iota zeta;rewrite (no_internal_edges_outside F prog Hnone b _ v (Some s) Hin);reflexivity|reflexivity]);
     rewrite Hfold
   end.
   cbn. apply IH.
Qed.
Theorem no_internal_edges_boundary_independent : forall F prog,
 no_internal_edges prog=true -> forall first second fuel limits entry visits work,
 iterate_with_boundary F first fuel prog limits entry visits work =
 iterate_with_boundary F second fuel prog limits entry visits work.
Proof.
 intros. rewrite !no_internal_edges_iteration by assumption. reflexivity.
Qed.
Theorem iterate_no_internal_edges_same : forall F fuel prog entry visits work,
 iterate F fuel prog entry visits work =
 iterate_with_boundary F
   (trim_join_live_state F (live_program F prog) (live_incoming_controls prog))
   fuel prog (revisit_limits prog) entry visits work.
Proof.
 intros. unfold iterate. destruct (no_internal_edges prog) eqn:Hnone; [|reflexivity].
 symmetry. apply no_internal_edges_iteration. assumption.
Qed.


(* These proposals are untrusted. The checker validates their invariants
   against the original instructions and every original control-flow edge. *)
Definition immediate_loop_plan (prog : program) : list (nat * nat * nat) :=
 flat_map (fun tail =>
 let body := block_body prog tail in
 match last_instr body with
 | Some (IBra (Some (neg,p)) head) =>
   if Nat.leb head tail then
   match find (fun ins => match ins with
     | ISetp None CNe _ _ d _ (OImm _) => Pos.eqb d p && negb neg
     | ISetp None CEq _ _ d _ (OImm _) => Pos.eqb d p && neg
     | _ => false end) (rev body) with
   | Some (ISetp _ _ _ _ _ r (OImm limit)) =>
     match find (fun ins => match ins with
       | IBin None BAdd _ d a (OImm _) => Pos.eqb d r && Pos.eqb a r
       | _ => false end) (rev body),
       find (fun ins => match ins with IImm None d _ => Pos.eqb d r | _ => false end)
         (rev (concat (firstn head prog))) with
     | Some (IBin _ _ _ _ _ (OImm step)), Some (IImm _ _ initial) =>
       let count := (limit-initial)/step in
       if negb (Z.eqb step 0) && Z.eqb ((limit-initial) mod step) 0 && (1 <=? count) && (count <=? 8)
       then [(head,tail,Z.to_nat count)] else []
     | _,_ => [] end
   | _ => [] end else []
 | _ => [] end) (rev (seq 0 (length prog))).

(* Discover small compiler-lowered range loops whose increment is selected
   by the continuation predicate. These are only case proposals: every
   original block and edge is validated by check_cases. *)
Definition prefix_constants (body : list instr) : PM.t Z :=
 fold_left (fun m i => match i with
 | IImm None d z => PM.add d z m
 | IMov None d s => match PM.find s m with Some z => PM.add d z m | None => PM.remove d m end
 | _ => match dest i with Some d => PM.remove d m | None => m end end) body (PM.empty Z).
Definition conditional_loop_plan (prog : program) : list (nat * nat * nat) :=
 flat_map (fun tail => let body := block_body prog tail in
 match last_instr body with
 | Some (IBra (Some (false,p)) head) =>
   if negb (Nat.leb head tail) then [] else
   let loop_body := concat (firstn (S (tail-head)) (skipn head prog)) in
   let predicates := p :: flat_map (fun i => match i with
     | IAndp None d a b => if Pos.eqb d p then [a;b] else [] | _ => [] end) body in
   let bounds := flat_map (fun i => match i with
     | ISetp None CLt 64 true d r (OImm z) => if existsb (Pos.eqb d) predicates then [(r,z)] else []
     | ISetp None CLe 64 true d r (OImm z) => if existsb (Pos.eqb d) predicates then [(r,z+1)] else []
     | _ => [] end) body in
   let constants := prefix_constants (concat (firstn head prog)) in
   flat_map (fun bound => let '(r,limit) := bound in
     let roots := r :: flat_map (fun i => match i with
       | IMov None d s => if Pos.eqb d r then [s] else [] | _ => [] end) loop_body in
     flat_map (fun counter =>
       let steps := flat_map (fun i => match i with
         | IBin None BAdd 64 d a (OReg delta) =>
           if Pos.eqb d counter && Pos.eqb a counter then [delta] else [] | _ => [] end) body in
       if negb (existsb (fun i => match i with
         | ISelp None 64 d (OImm 1) (OImm 0) q => existsb (Pos.eqb d) steps && existsb (Pos.eqb q) predicates
         | _ => false end) body) then [] else
       match PM.find counter constants with
       | Some initial => let count := limit-initial+1 in
         if (1 <=? count) && (count <=? 8) then [(head,tail,Z.to_nat count)] else []
       | None => [] end) roots) bounds
 | _ => [] end) (rev (seq 0 (length prog))).
(* Unfold disjoint regions from the last one backwards, so expanding a
   region cannot change the original indices of any region still to visit.
   Nested or duplicate proposals do not justify extra cases on their own. *)
Definition independent_plan (proposed : list (nat * nat * nat)) : list (nat * nat * nat) :=
 let later := fun p q => let '(_,tail,_) := p in let '(_,tail',_) := q in Nat.ltb tail' tail in
 let insert := fix insert p xs := match xs with
   | [] => [p] | q::rest => if later p q then p::xs else q::insert p rest end in
 let ordered := fold_right insert [] proposed in
 fold_left (fun chosen p => let '(head,tail,_) := p in
   if existsb (fun q => let '(head',tail',_) := q in
     negb (Nat.ltb tail head' || Nat.ltb tail' head)) chosen
   then chosen else chosen ++ [p]) ordered [].
Definition loop_plan (prog : program) : list (nat * nat * nat) :=
 independent_plan (immediate_loop_plan prog ++ conditional_loop_plan prog).

Definition unfold_region (prog : program) (head tail count : nat) : program :=
 let width := S (tail-head) in
 let target := fun level t => if Nat.ltb t head then t else
   if Nat.ltb tail t then (t + count * width)%nat else (head + level * width + (t-head))%nat in
 let ordinary := fun level ins => match ins with IBra g t => IBra g (target level t) | _ => ins end in
 let copies := map (fun level =>
   map (fun pair => let '(b,body) := pair in map (fun ins =>
      match ins with
      | IBra (Some (neg,p)) t =>
          if Nat.eqb b tail && Nat.eqb t head && Nat.ltb level count
          then IBra (Some (negb neg,p)) (target level (S tail)) else ordinary level ins
      | _ => ordinary level ins end) body)
     (combine (seq head width) (firstn width (skipn head prog)))) (seq 0 (S count)) in
 map (map (ordinary O)) (firstn head prog) ++ concat copies ++ map (map (ordinary O)) (skipn (S tail) prog).

Definition unfold_plan (prog : program) (plan : list (nat * nat * nat)) : program * list nat :=
 fold_left (fun pair item => let '(p,ids) := pair in let '(head,tail,count) := item in
   (unfold_region p head tail count,
    firstn head ids ++ concat (repeat (firstn (S (tail-head)) (skipn head ids)) (S count)) ++ skipn (S tail) ids))
   plan (prog,seq 0 (length prog)).

Definition propose_cases (F : lfacts) (prog : program) (plan : list (nat * nat * nat)) : option (list (list astate)) :=
 match plan with
 | [] => match analyze F prog with Some inv => Some (map (fun st => [st]) inv) | None => None end
 | _ => let '(p,ids) := unfold_plan prog plan in match analyze F p with
   | None => None
   | Some inv => Some (map (fun b => map snd (filter (fun pair => Nat.eqb (fst pair) b) (combine ids inv)))
                       (seq 0 (length prog))) end
 end.

Definition copy_root (copies : PM.t reg) (r : reg) : reg := match PM.find r copies with Some u => u | None => r end.

(* Select a launch-fixed comparison that chooses between distinct shared
   allocations. Splitting at its threshold preserves each allocation base. *)
Fixpoint selection_split (F : lfacts) (body : list instr) (copies : PM.t reg)
  (cmps : PM.t (reg * Z)) : option (reg * Z) :=
 match body with
 | [] => None
 | ins::rest =>
   let next := fun copies cmps => selection_split F rest copies cmps in
   match ins with
   | IMov None d s => next (PM.add d (copy_root copies s) copies) (PM.remove d cmps)
   | ISetp None c _ _ d a (OImm z) =>
     let r := copy_root copies a in
     let cut := match c with CLt | CGe => Some z | CLe | CGt => Some (z+1) | _ => None end in
     next (PM.remove d copies)
       (match cut, lookup (lf_bounds F) r with
        | Some z, Some (lo,hi) => if is_fixed F r && (lo <? z) && (z <=? hi)
            then PM.add d (r,z) cmps else PM.remove d cmps
        | _,_ => PM.remove d cmps end)
   | ISelp _ _ d (OReg a) (OReg b) p =>
     let a := copy_root copies a in let b := copy_root copies b in
     if negb (Pos.eqb a b) && is_some (lookup (lf_shared F) a) && is_some (lookup (lf_shared F) b)
     then match PM.find p cmps with Some key => Some key
          | None => next (PM.remove d copies) (PM.remove d cmps) end
     else next (PM.remove d copies) (PM.remove d cmps)
   | _ => match dest ins with
     | Some d => next (PM.remove d copies) (PM.remove d cmps)
     | None => next copies cmps end
   end
 end.
Definition split_key (F : lfacts) (prog : program) : option (reg * Z) :=
 selection_split F (concat prog) (PM.empty reg) (PM.empty (reg * Z)).

Definition split_bound (r : reg) (cut : Z) (low : bool) (p : reg * range) : reg * range :=
 let '(u,(lo,hi)) := p in
 if Pos.eqb u r then (u,if low then (lo,Z.min hi (cut-1)) else (Z.max lo cut,hi)) else p.
Definition split_launch (F : lfacts) (r : reg) (cut : Z) (low : bool) : lfacts :=
 {| lf_widths := lf_widths F; lf_fixed := lf_fixed F;
    lf_bounds := map (split_bound r cut low) (lf_bounds F);
    lf_aligns := lf_aligns F; lf_roles := lf_roles F;
    lf_shared_window := lf_shared_window F; lf_local_window := lf_local_window F;
    lf_shared := lf_shared F; lf_locals := lf_locals F; lf_cells := lf_cells F;
    lf_payloads := lf_payloads F; lf_derived := lf_derived F; lf_names := lf_names F |}.

Definition check_partition (F : lfacts) (prog : program) (plan : list (nat * nat * nat)) : bool :=
 match propose_cases F prog plan with Some inv => check_cases F prog inv | None => false end.

Definition check_block (F : lfacts) (prog : program) : bool :=
 if lf_ok F then match split_key F prog with
 | None => check_partition F prog (independent_plan (conditional_loop_plan prog))
 | Some (r,cut) => let plan := loop_plan prog in
     if check_partition (split_launch F r cut true) prog plan
     then check_partition (split_launch F r cut false) prog plan else false
 end else false.

Definition partition_failures (F : lfacts) (prog : program) (plan : list (nat * nat * nat)) : option (list (nat * nat)) :=
 match propose_cases F prog plan with
 | None => None
 | Some inv => Some (flat_map (fun b => flat_map (fun st => if is_bottom st then []
     else walk_fail F st b 0 (block_body prog b)) (cases_at inv b)) (seq 0 (length prog)))
 end.
Definition failures (F : lfacts) (prog : program) : option (list (nat * nat)) :=
 match split_key F prog with
 | None => partition_failures F prog (independent_plan (conditional_loop_plan prog))
 | Some (r,cut) => let plan := loop_plan prog in
   match partition_failures (split_launch F r cut true) prog plan,
         partition_failures (split_launch F r cut false) prog plan with
   | Some a, Some b => Some (a++b) | _,_ => None end
 end.


(* ------------------------------------------------------------------ *)
(* Explicit, untrusted invariant certificates. Proposal is evaluated once;
   acceptance checks this value against the independently reconstructed
   program and launch facts. No equality with the proposal is required. *)
Definition case_certificate := list (list astate).
Inductive block_certificate : Type :=
| CertSingle (cases : option case_certificate)
| CertSplit (r : reg) (cut : Z)
    (low high : option case_certificate).

Definition check_case_certificate (F : lfacts) (prog : program)
    (cases : option case_certificate) : bool :=
 match cases with Some inv => check_cases F prog inv | None => false end.

Definition check_block_certificate (F : lfacts) (prog : program)
    (cert : block_certificate) : bool :=
 lf_ok F && match cert with
 | CertSingle cases => check_case_certificate F prog cases
 | CertSplit r cut low high =>
     check_case_certificate (split_launch F r cut true) prog low &&
     check_case_certificate (split_launch F r cut false) prog high
 end.

Definition propose_block_certificate (F : lfacts) (prog : program) : block_certificate :=
 if lf_ok F then match split_key F prog with
 | None => CertSingle (propose_cases F prog (independent_plan (conditional_loop_plan prog)))
 | Some (r,cut) =>
     let plan := loop_plan prog in
     let low := propose_cases (split_launch F r cut true) prog plan in
     (* A failed low case already refutes this proposal. Do not spend the
        remaining deadline proposing the other half. The checker still
        requires both halves whenever it accepts a split certificate. *)
     let high := if check_case_certificate (split_launch F r cut true) prog low
                 then propose_cases (split_launch F r cut false) prog plan else None in
     CertSplit r cut low high
 end else CertSingle None.
