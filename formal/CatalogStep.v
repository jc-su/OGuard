(* CatalogStep.v: the step check of the catalog, and the invariants it keeps.

   The catalog records the pool's size, each live buffer (its id, offset,
   bytes, owner, and status), the grants, each freed region with its last
   owner, each tenant's reservation, and each verified operator with its
   binding. A buffer's id is its owner and a number that the owner's steps
   choose, so that the ids of two tenants never meet. A tenant reserves one
   region of the pool when its session opens, and each of its buffers lies
   in that region, so that where its buffers lie, and whether an
   allocation fits, depend only on its own steps. A step is a tenant
   and a sequence of actions; `check_step` checks the actions in order,
   each against the state the previous ones leave, and gives the next
   catalog with the effects the executor performs, or the reason for a
   refusal. The Python `check_step` in `opgate/catalog` is the
   transcription of this definition, and `tests/formal` requires the two
   to agree on generated steps; `show_decision` prints a decision as the
   test compares it.

   A write carries its bytes and, for checked data, the type of its
   elements, whose range the check computes. Numbers are Z: offsets and
   extents are byte counts, and a unary nat would be as large as they are.

   Proved: `check_step_inv`, an accepted step keeps every buffer inside the
   pool, aligned to 256 bytes, with a distinct id, disjoint from every other
   buffer, and every grant on a live buffer of another owner; and
   `check_launch_sound`, an accepted launch has the facts that the gate's
   launch semantics assumes of each argument (design section 6.1). *)

From Coq Require Import ZArith.ZArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Strings.String.
From Coq Require Import Lia.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateDomain.
Require Import OpGate.GateFront.
Require Import OpGate.GateParse.

Local Open Scope Z_scope.
Local Open Scope string_scope.

(* ------------------------------------------------------------------ *)
(* The catalog and the actions of a step.                             *)

(* A buffer's id: its owner, and a number that the owner chooses. *)
Definition bid := (Z * Z)%type.

Definition bid_eqb (x y : bid) : bool := Z.eqb (fst x) (fst y) && Z.eqb (snd x) (snd y).

Lemma bid_eqb_eq : forall x y, bid_eqb x y = true <-> x = y.
Proof.
  intros [a b] [c d]. unfold bid_eqb. cbn [fst snd]. rewrite andb_true_iff, !Z.eqb_eq.
  split; [intros [-> ->]; reflexivity | intros H; injection H as -> ->; split; reflexivity].
Qed.

Lemma bid_eqb_refl : forall x, bid_eqb x x = true.
Proof. intros x. apply bid_eqb_eq. reflexivity. Qed.

Lemma bid_eqb_neq : forall x y, bid_eqb x y = false <-> x <> y.
Proof.
  intros x y. split.
  - intros H E. apply bid_eqb_eq in E. congruence.
  - intros H. destruct (bid_eqb x y) eqn:E; [exfalso; apply H; apply bid_eqb_eq; exact E | reflexivity].
Qed.

Lemma bid_eqb_spec : forall x y, reflect (x = y) (bid_eqb x y).
Proof. intros x y. apply iff_reflect. symmetry. apply bid_eqb_eq. Qed.

Definition bid_eq_dec (x y : bid) : {x = y} + {x <> y}.
Proof.
  destruct (bid_eqb x y) eqn:E; [left; apply bid_eqb_eq; exact E | right; apply bid_eqb_neq; exact E].
Defined.

(* What the catalog knows of a buffer's values: nothing, or that every
   element, of elem bytes read signed or not, lies in [lo, hi]. *)
Inductive status := SRaw | SChecked (elem : Z) (signed : bool) (lo hi : Z).

Record buffer := { bf_id : bid; bf_offset : Z; bf_bytes : Z; bf_owner : Z; bf_status : status }.

(* A freed region of the pool and its last owner. *)
Record region := { rg_offset : Z; rg_bytes : Z; rg_owner : Z }.

(* A tenant's reservation: the region of the pool that holds its buffers. *)
Record reservation := { rs_tenant : Z; rs_offset : Z; rs_bytes : Z }.

Record catalog := {
  c_pool : Z;                          (* the bytes of the pool *)
  c_buffers : list buffer;
  c_grants : list (bid * Z);           (* (buffer, tenant): the tenant may read the buffer *)
  c_freed : list region;
  c_reserved : list reservation;       (* at most one for each tenant *)
  c_operators : list (Z * binding)     (* each verified operator's binding *)
}.

Inductive action :=
| Allocate (id : bid) (offset bytes : Z)
| Free (id : bid)
| Grant (id : bid) (tenant : Z)
(* the host writes the bytes of data at the buffer's start; with an element
   type, of elem bytes read signed or not, the data fills the buffer *)
| Write (id : bid) (data : list Z) (element : option (Z * bool))
| Fill (id : bid) (elem : Z) (signed : bool) (value : Z)
(* with guarded, the elements of an index input that the catalog has not
   checked are checked when the launch runs, and the kernel runs only if
   each lies in the range (design section 6.4); each buffer argument is a
   view, a buffer id and the byte offset of the view's start in the buffer *)
| Launch (op gx gy gz : Z) (buffers : list (bid * Z)) (sizes scalars : list Z) (guarded : bool)
(* the tenant reserves the bytes [offset, offset + bytes) as its session opens *)
| Reserve (offset bytes : Z)
(* the tenant, holding no live buffer, ends its reservation and drops its
   freed regions, so that their bytes are cleared before any later use *)
| Unreserve.

Record step := { s_tenant : Z; s_actions : list action }.

(* What a PTX parameter of a checked launch holds: the address of a byte
   of the pool, or a value modulo 2 to the parameter's width. *)
Inductive arg := At (offset : Z) | Val (v : Z).

(* A check when a kernel runs: every whole element of the bytes at the
   pool's offset, of elem bytes read signed or not, lies in [lo, hi]. *)
Record check := { ck_offset : Z; ck_bytes : Z; ck_elem : Z; ck_signed : bool; ck_lo : Z; ck_hi : Z }.

Inductive effect :=
| Clear (offset bytes : Z)
| Upload (offset : Z) (data : list Z)
| Memset (offset bytes elem : Z) (signed : bool) (value : Z)
(* a checked launch, which runs only if every check holds *)
| Kernel (op gx gy gz shared : Z) (args : list arg) (checks : list check).

Inductive decision := Accepted (c : catalog) (effects : list effect) | Refused (reason : string).

Definition alignment : Z := 256.

(* ------------------------------------------------------------------ *)
(* Queries.                                                           *)

Definition find_buffer (c : catalog) (id : bid) : option buffer :=
  find (fun b => bid_eqb (bf_id b) id) (c_buffers c).

Definition find_reservation (c : catalog) (t : Z) : option reservation :=
  find (fun r => Z.eqb (rs_tenant r) t) (c_reserved c).

Definition find_binding (c : catalog) (op : Z) : option binding :=
  match find (fun e => Z.eqb (fst e) op) (c_operators c) with Some (_, b) => Some b | None => None end.

Definition granted (c : catalog) (id : bid) : bool := existsb (fun g => bid_eqb (fst g) id) (c_grants c).

Definition has_grant (c : catalog) (id : bid) (t : Z) : bool :=
  existsb (fun g => bid_eqb (fst g) id && Z.eqb (snd g) t) (c_grants c).

Definition readable (c : catalog) (t : Z) (id : bid) : bool :=
  match find_buffer c id with Some b => Z.eqb (bf_owner b) t || has_grant c id t | None => false end.

Definition writable (c : catalog) (t : Z) (id : bid) : bool :=
  match find_buffer c id with Some b => Z.eqb (bf_owner b) t && negb (granted c id) | None => false end.

Definition set_status (c : catalog) (id : bid) (st : status) : catalog :=
  {| c_pool := c_pool c;
     c_buffers := map (fun b => if bid_eqb (bf_id b) id
                                then {| bf_id := bf_id b; bf_offset := bf_offset b; bf_bytes := bf_bytes b;
                                        bf_owner := bf_owner b; bf_status := st |}
                                else b) (c_buffers c);
     c_grants := c_grants c; c_freed := c_freed c; c_reserved := c_reserved c;
     c_operators := c_operators c |}.

Definition overlap (ao ab bo bb : Z) : bool := Z.ltb ao (bo + bb) && Z.ltb bo (ao + ab).

(* The bytes [o, o + n) lie in tenant t's reservation. *)
Definition in_reservation (c : catalog) (t o n : Z) : bool :=
  match find_reservation c t with
  | Some r => Z.leb (rs_offset r) o && Z.leb (o + n) (rs_offset r + rs_bytes r)
  | None => false
  end.

(* The freed regions without the bytes [o, o + n). *)
Definition carve (freed : list region) (o n : Z) : list region :=
  flat_map (fun r =>
              if negb (overlap (rg_offset r) (rg_bytes r) o n) then [r]
              else app (if Z.ltb (rg_offset r) o
                        then [{| rg_offset := rg_offset r; rg_bytes := o - rg_offset r; rg_owner := rg_owner r |}]
                        else [])
                       (if Z.ltb (o + n) (rg_offset r + rg_bytes r)
                        then [{| rg_offset := o + n; rg_bytes := rg_offset r + rg_bytes r - o - n;
                                 rg_owner := rg_owner r |}] else [])) freed.

(* A freed region touches the bytes [o, o + n) of tenant t if t last owned
   it and it ends at o or starts at o + n. *)
Definition touches (o n t : Z) (r : region) : bool :=
  Z.eqb (rg_owner r) t && (Z.eqb (rg_offset r + rg_bytes r) o || Z.eqb (rg_offset r) (o + n)).

(* The freed regions after tenant t frees the bytes [o, o + n): the bytes and
   the regions that touch them become one region of t, which comes last, and
   the other regions keep their order. Regions of other owners are never
   merged. An allocation that spans adjacent frees of one tenant then lies
   in one region of that tenant, and keeps its bytes. *)
Definition release (freed : list region) (o n t : Z) : list region :=
  let near := filter (touches o n t) freed in
  let lo := fold_left Z.min (map rg_offset near) o in
  let hi := fold_left Z.max (map (fun r => rg_offset r + rg_bytes r) near) (o + n) in
  filter (fun r => negb (touches o n t r)) freed ++ [{| rg_offset := lo; rg_bytes := hi - lo; rg_owner := t |}].

(* An element is 1, 2, 4, or 8 bytes, as the Python type of an element
   states. *)
Definition width_ok (elem : Z) : bool := Z.eqb elem 1 || Z.eqb elem 2 || Z.eqb elem 4 || Z.eqb elem 8.

Definition element_range (elem : Z) (signed : bool) : Z * Z :=
  if signed then (- 2 ^ (8 * elem - 1), 2 ^ (8 * elem - 1) - 1) else (0, 2 ^ (8 * elem) - 1).

(* The value of bytes, least significant first, each taken modulo 256. *)
Fixpoint bytes_value (bs : list Z) : Z :=
  match bs with [] => 0 | x :: rest => x mod 256 + 256 * bytes_value rest end.

(* The whole elements of data, elem bytes each, read signed or not. *)
Definition elements (elem : Z) (signed : bool) (data : list Z) : list Z :=
  let n := Z.to_nat elem in
  map (fun i => extend (Z.to_nat (8 * elem)) signed (bytes_value (firstn n (skipn (i * n) data))))
      (seq 0 (List.length data / n)).

(* The least and the greatest element of data, or (0, 0) if it has none. *)
Definition data_range (elem : Z) (signed : bool) (data : list Z) : Z * Z :=
  match elements elem signed data with
  | [] => (0, 0)
  | x :: rest => (fold_left Z.min rest x, fold_left Z.max rest x)
  end.

(* The value of an affine at the launch's sizes. *)
Definition eval (a : affine) (sizes : list Z) : Z :=
  af_const a + fold_left (fun acc t => acc + snd t * nth (Z.to_nat (fst t)) sizes 0) (af_terms a) 0.

(* ------------------------------------------------------------------ *)
(* The launch check.                                                  *)

Definition spans (b : binding) : list bspan :=
  flat_map (fun p => match snd p with PBase sp => [sp] | _ => [] end) (b_params b).

Definition bounds (b : binding) : list (Z * Z) :=
  flat_map (fun p => match snd p with PBounded lo hi => [(lo, hi)] | _ => [] end) (b_params b).

Definition reads (sp : bspan) : bool := match bs_dir sp with DRead | DReadWrite => true | DWrite => false end.
Definition writes (sp : bspan) : bool := match bs_dir sp with DWrite | DReadWrite => true | DRead => false end.

(* The buffer is checked inside the index range at the sizes. *)
Definition index_ok (b : buffer) (ix : ixrange) (sizes : list Z) : bool :=
  match bf_status b with
  | SChecked e s lo hi =>
      Z.eqb e (ir_elem ix) && Bool.eqb s (ir_signed ix) &&
      Z.leb (eval (ir_lo ix) sizes) lo && Z.leb hi (eval (ir_hi ix) sizes)
  | SRaw => false
  end.

(* The launch checks an index input when it runs: the launch is guarded, the
   span does not write the buffer, and the catalog has not checked it. *)
Definition checked_at_run (guarded : bool) (b : buffer) (sp : bspan) (ix : ixrange) (sizes : list Z) : bool :=
  guarded && negb (writes sp) && negb (index_ok b ix sizes).

(* A view starts at a multiple of 256 bytes into its buffer; the view of a
   span with an index range starts at the buffer's start, since the catalog
   records an index range for a whole buffer. *)
Definition view_ok (off : Z) (sp : bspan) : bool :=
  Z.leb 0 off && Z.eqb (off mod alignment) 0 &&
  match bs_index sp with Some _ => Z.eqb off 0 | None => true end.

(* No argument of the launch passes buffer id with an index range or as a
   flag buffer. *)
Definition plain_views (pairs : list ((bid * Z) * bspan)) (id : bid) : bool :=
  forallb (fun y => negb (bid_eqb (fst (fst y)) id) ||
                    (match bs_index (snd y) with Some _ => false | None => true end && negb (bs_flag (snd y))))
          pairs.

(* Why the view of buffer id from off may not be the argument of span sp, or
   None. A buffer that the launch writes is passed once, or only through
   spans without an index range and without the flag: the catalog records
   the status of a whole buffer, the elements of an index span must not
   change under it, and the words of a flag buffer may only be set to 1.
   A buffer that the tenant may not read is refused first, for the same
   reason whether it is live or not, so that a refusal shows the tenant
   nothing of another tenant's buffers. *)
Definition arg_reason (c : catalog) (t : Z) (guarded : bool) (buffers : list (bid * Z)) (sps : list bspan)
    (sizes : list Z) (x : (bid * Z) * bspan) : option string :=
  let '((id, off), sp) := x in
  match find_buffer c id with
  | None => Some "the tenant may not read a buffer argument"
  | Some b =>
      if negb (readable c t id) then Some "the tenant may not read a buffer argument"
      else if negb (view_ok off sp) then Some "a view is not at a multiple of 256 bytes, or an index span's view does not start its buffer"
      else if Z.ltb (bf_bytes b) (off + Z.max 1 (eval (bs_ext sp) sizes)) then Some "a buffer is smaller than its extent"
      else if writes sp && negb (writable c t id && (Nat.eqb (count_occ bid_eq_dec (map fst buffers) id) 1 ||
                                                     plain_views (combine buffers sps) id))
      then Some "the tenant may not write a buffer argument, or passes a written buffer twice with an index range or a flag"
      else match bs_index sp with
           | Some ix => if index_ok b ix sizes || checked_at_run guarded b sp ix sizes then None
                        else Some "an index buffer is not checked inside its declared range"
           | None => None
           end
  end.

(* The checks of the index inputs that a guarded launch checks when it runs:
   each whole element of the buffer lies in the declared range. *)
Definition checks_of (c : catalog) (guarded : bool) (sizes : list Z) (pairs : list ((bid * Z) * bspan))
    : list check :=
  flat_map (fun x => let '((id, _), sp) := x in
              match find_buffer c id, bs_index sp with
              | Some b, Some ix =>
                  if checked_at_run guarded b sp ix sizes
                  then [{| ck_offset := bf_offset b; ck_bytes := bf_bytes b; ck_elem := ir_elem ix;
                           ck_signed := ir_signed ix; ck_lo := eval (ir_lo ix) sizes; ck_hi := eval (ir_hi ix) sizes |}]
                  else []
              | _, _ => []
              end) pairs.

Fixpoint first_reason {A : Type} (f : A -> option string) (l : list A) : option string :=
  match l with
  | [] => None
  | x :: rest => match f x with Some r => Some r | None => first_reason f rest end
  end.

(* What each PTX parameter holds, with the buffer arguments and the scalars
   in order. *)
Fixpoint args_of (c : catalog) (sizes : list Z) (ps : list (string * holding)) (buffers : list (bid * Z))
    (scalars : list Z) : list arg :=
  match ps with
  | [] => []
  | (_, p) :: rest =>
      match p with
      | PBase _ =>
          match buffers with
          | (id, off) :: bs => At (match find_buffer c id with Some b => bf_offset b + off | None => 0 end)
                          :: args_of c sizes rest bs scalars
          | [] => []
          end
      | PSize a => Val (eval a sizes) :: args_of c sizes rest buffers scalars
      | PConst z => Val z :: args_of c sizes rest buffers scalars
      | PBounded _ _ =>
          match scalars with
          | v :: vs => Val v :: args_of c sizes rest buffers vs
          | [] => []
          end
      | PZero => Val 0 :: args_of c sizes rest buffers scalars
      end
  end.

(* The status a launch leaves in the buffer of span sp: its declared range
   for an index output, raw for any other written buffer. *)
Definition written (c : catalog) (sizes : list Z) (x : (bid * Z) * bspan) : catalog :=
  let '((id, _), sp) := x in
  if negb (writes sp) then c else
  match bs_index sp with
  | Some ix => set_status c id (SChecked (ir_elem ix) (ir_signed ix) (eval (ir_lo ix) sizes) (eval (ir_hi ix) sizes))
  | None => set_status c id SRaw
  end.

Definition derived_holds (sizes : list Z) (d : derived_size) : Prop :=
  0 < ds_divisor d /\
  nth (Z.to_nat (ds_target d)) sizes 0 = eval (ds_numerator d) sizes / ds_divisor d.
Definition derived_check (sizes : list Z) (d : derived_size) : bool :=
  Z.ltb 0 (ds_divisor d) &&
  Z.eqb (nth (Z.to_nat (ds_target d)) sizes 0) (eval (ds_numerator d) sizes / ds_divisor d).

Definition check_launch (c : catalog) (t op gx gy gz : Z) (buffers : list (bid * Z)) (sizes scalars : list Z)
    (guarded : bool)
    : catalog * effect + string :=
  match find_binding c op with
  | None => inr "the operator is not verified"
  | Some b =>
      let '(lx, ly, lz) := grid_limit in
      let '(mx, my, mz) := b_grid b in
      if negb (Z.leb 1 gx && Z.leb gx lx && Z.leb 1 gy && Z.leb gy ly && Z.leb 1 gz && Z.leb gz lz)
      then inr "the grid is outside the limits the gate assumes"
      else if negb (Z.leb gx mx && Z.leb gy my && Z.leb gz mz)
      then inr "the grid is larger than the binding's maxima"
      else if negb (Nat.eqb (List.length sizes) (List.length (b_sizes b))) then inr "a size is missing or extra"
      else if negb (forallb (fun e => Z.leb (fst (fst e)) (snd e) && Z.leb (snd e) (snd (fst e))) (combine (b_sizes b) sizes))
      then inr "a size lies outside its declared bounds"
      else if negb (forallb (derived_check sizes) (b_derived b))
      then inr "a derived size does not equal its declared quotient"
      else if negb (Z.leb 0 (eval (b_shared b) sizes) && Z.leb (eval (b_shared b) sizes) shared_max)
      then inr "dynamic shared memory lies outside [0, 48 KiB]"
      else if negb (Nat.eqb (List.length buffers) (List.length (spans b))) then inr "a buffer argument is missing or extra"
      else if negb (Nat.eqb (List.length scalars) (List.length (bounds b))) then inr "a scalar is missing or extra"
      else if negb (forallb (fun e => Z.leb (fst (fst e)) (snd e) && Z.leb (snd e) (snd (fst e)))
                            (combine (bounds b) scalars))
      then inr "a scalar lies outside its bounds"
      else match first_reason (arg_reason c t guarded buffers (spans b) sizes) (combine buffers (spans b)) with
           | Some r => inr r
           | None =>
               inl (fold_left (fun acc x => written acc sizes x) (combine buffers (spans b)) c,
                    Kernel op gx gy gz (eval (b_shared b) sizes) (args_of c sizes (b_params b) buffers scalars)
                           (checks_of c guarded sizes (combine buffers (spans b))))
           end
  end.

(* ------------------------------------------------------------------ *)
(* The step check.                                                    *)

Definition check_action (c : catalog) (t : Z) (a : action) : catalog * list effect + string :=
  match a with
  | Allocate id o n =>
      if negb (Z.eqb (fst id) t) then inr "the buffer id is not the tenant's" else
      match find_buffer c id with
      | Some _ => inr "the buffer id is in use"
      | None =>
          if Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool c) (o + n) then inr "the region is empty or outside the pool"
          else if negb (Z.eqb (o mod alignment) 0) then inr "the region is not aligned to 256 bytes"
          else if negb (in_reservation c t o n) then inr "the region lies outside the tenant's reservation"
          else if existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers c)
          then inr "the region overlaps a live buffer"
          else
            (* the region keeps its contents only if the tenant last owned all of it *)
            let kept := existsb (fun r => Z.eqb (rg_owner r) t && Z.leb (rg_offset r) o &&
                                          Z.leb (o + n) (rg_offset r + rg_bytes r)) (c_freed c) in
            inl ({| c_pool := c_pool c;
                    c_buffers := app (c_buffers c) [{| bf_id := id; bf_offset := o; bf_bytes := n;
                                                      bf_owner := t; bf_status := SRaw |}];
                    c_grants := c_grants c; c_freed := carve (c_freed c) o n;
                    c_reserved := c_reserved c; c_operators := c_operators c |},
                 if kept then [] else [Clear o n])
      end
  | Free id =>
      match find_buffer c id with
      | Some b =>
          if negb (Z.eqb (bf_owner b) t) then inr "the tenant does not own the buffer" else
          inl ({| c_pool := c_pool c;
                  c_buffers := filter (fun x => negb (bid_eqb (bf_id x) id)) (c_buffers c);
                  c_grants := filter (fun g => negb (bid_eqb (fst g) id)) (c_grants c);
                  c_freed := release (c_freed c) (bf_offset b) (bf_bytes b) t;
                  c_reserved := c_reserved c; c_operators := c_operators c |}, [])
      | None => inr "the tenant does not own the buffer"
      end
  | Grant id u =>
      match find_buffer c id with
      | Some b =>
          if negb (Z.eqb (bf_owner b) t) || Z.eqb u t
          then inr "the tenant does not own the buffer, or grants it to itself"
          else if has_grant c id u then inr "the grant exists"
          else inl ({| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := app (c_grants c) [(id, u)];
                       c_freed := c_freed c; c_reserved := c_reserved c; c_operators := c_operators c |}, [])
      | None => inr "the tenant does not own the buffer, or grants it to itself"
      end
  | Write id data element =>
      match find_buffer c id with
      | Some b =>
          if negb (writable c t id) then inr "the tenant may not write the buffer" else
          let n := Z.of_nat (List.length data) in
          match element with
          | None =>
              if Z.leb n (bf_bytes b) then inl (set_status c id SRaw, [Upload (bf_offset b) data])
              else inr "the data is larger than the buffer"
          | Some (e, s) =>
              if width_ok e && Z.eqb n (bf_bytes b) && Z.eqb (bf_bytes b mod e) 0
              then let '(lo, hi) := data_range e s data in
                   inl (set_status c id (SChecked e s lo hi), [Upload (bf_offset b) data])
              else inr "checked data must fill the buffer with whole elements"
          end
      | None => inr "the tenant may not write the buffer"
      end
  | Fill id e s v =>
      match find_buffer c id with
      | Some b =>
          if negb (writable c t id) then inr "the tenant may not write the buffer" else
          let '(lo, hi) := element_range e s in
          if width_ok e && Z.leb lo v && Z.leb v hi && Z.eqb (bf_bytes b mod e) 0
          then inl (set_status c id (SChecked e s v v), [Memset (bf_offset b) (bf_bytes b) e s v])
          else inr "the value is not an element, or the buffer is not whole elements"
      | None => inr "the tenant may not write the buffer"
      end
  | Launch op gx gy gz buffers sizes scalars guarded =>
      match check_launch c t op gx gy gz buffers sizes scalars guarded with
      | inl (c', k) => inl (c', [k])
      | inr r => inr r
      end
  | Reserve o n =>
      match find_reservation c t with
      | Some _ => inr "the tenant holds a reservation"
      | None =>
          if Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool c) (o + n) then inr "the reservation is empty or outside the pool"
          else if existsb (fun r => overlap (rs_offset r) (rs_bytes r) o n) (c_reserved c)
          then inr "the reservation overlaps another"
          else inl ({| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c; c_freed := c_freed c;
                       c_reserved := app (c_reserved c) [{| rs_tenant := t; rs_offset := o; rs_bytes := n |}];
                       c_operators := c_operators c |}, [])
      end
  | Unreserve =>
      match find_reservation c t with
      | None => inr "the tenant holds no reservation"
      | Some _ =>
          if existsb (fun b => Z.eqb (bf_owner b) t) (c_buffers c) then inr "the tenant holds a live buffer"
          else inl ({| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c;
                       c_freed := filter (fun r => negb (Z.eqb (rg_owner r) t)) (c_freed c);
                       c_reserved := filter (fun r => negb (Z.eqb (rs_tenant r) t)) (c_reserved c);
                       c_operators := c_operators c |}, [])
      end
  end.

Fixpoint check_actions (c : catalog) (t : Z) (position : Z) (actions : list action) (done : list effect)
    : decision :=
  match actions with
  | [] => Accepted c done
  | a :: rest =>
      match check_action c t a with
      | inl (c', es) => check_actions c' t (position + 1) rest (app done es)
      | inr r => Refused ("action " ++ zstr position ++ ": " ++ r)
      end
  end.

Definition check_step (c : catalog) (s : step) : decision := check_actions c (s_tenant s) 0 (s_actions s) [].

(* ------------------------------------------------------------------ *)
(* A decision as the conformance test compares it.                    *)

Definition show_status (st : status) : string :=
  match st with
  | SRaw => "raw"
  | SChecked e s lo hi =>
      "checked(" ++ zstr e ++ "," ++ (if s then "s" else "u") ++ "," ++ zstr lo ++ "," ++ zstr hi ++ ")"
  end.

Definition show_bid (id : bid) : string := "(" ++ zstr (fst id) ++ "," ++ zstr (snd id) ++ ")".

Definition show_buffer (b : buffer) : string :=
  "(" ++ show_bid (bf_id b) ++ "," ++ zstr (bf_offset b) ++ "," ++ zstr (bf_bytes b) ++ "," ++
  zstr (bf_owner b) ++ "," ++ show_status (bf_status b) ++ ")".

Definition show_region (r : region) : string :=
  "(" ++ zstr (rg_offset r) ++ "," ++ zstr (rg_bytes r) ++ "," ++ zstr (rg_owner r) ++ ")".

Definition show_grant (g : bid * Z) : string := "(" ++ show_bid (fst g) ++ "," ++ zstr (snd g) ++ ")".

Definition show_reservation (r : reservation) : string :=
  "(" ++ zstr (rs_tenant r) ++ "," ++ zstr (rs_offset r) ++ "," ++ zstr (rs_bytes r) ++ ")".

Fixpoint join_semicolon (l : list string) : string :=
  match l with [] => "" | [x] => x | x :: rest => x ++ ";" ++ join_semicolon rest end.

Definition show_list {A : Type} (f : A -> string) (l : list A) : string :=
  "[" ++ join_semicolon (map f l) ++ "]".

Definition show_catalog (c : catalog) : string :=
  "pool=" ++ zstr (c_pool c) ++ " buffers=" ++ show_list show_buffer (c_buffers c) ++
  " grants=" ++ show_list show_grant (c_grants c) ++ " freed=" ++ show_list show_region (c_freed c) ++
  " reserved=" ++ show_list show_reservation (c_reserved c).

Definition show_arg (a : arg) : string :=
  match a with At o => "at(" ++ zstr o ++ ")" | Val v => "val(" ++ zstr v ++ ")" end.

Definition show_check (k : check) : string :=
  "(" ++ zstr (ck_offset k) ++ "," ++ zstr (ck_bytes k) ++ "," ++ zstr (ck_elem k) ++ "," ++
  (if ck_signed k then "s" else "u") ++ "," ++ zstr (ck_lo k) ++ "," ++ zstr (ck_hi k) ++ ")".

Definition show_effect (e : effect) : string :=
  match e with
  | Clear o n => "clear(" ++ zstr o ++ "," ++ zstr n ++ ")"
  | Upload o data => "upload(" ++ zstr o ++ "," ++ zstr (Z.of_nat (List.length data)) ++ ")"
  | Memset o n e s v =>
      "memset(" ++ zstr o ++ "," ++ zstr n ++ "," ++ zstr e ++ "," ++ (if s then "s" else "u") ++ "," ++
      zstr v ++ ")"
  | Kernel op gx gy gz shared args checks =>
      "kernel(" ++ zstr op ++ "," ++ zstr gx ++ "," ++ zstr gy ++ "," ++ zstr gz ++ "," ++
      zstr shared ++ "," ++ show_list show_arg args ++ "," ++ show_list show_check checks ++ ")"
  end.

Definition show_decision (d : decision) : string :=
  match d with
  | Accepted c es => "accepted " ++ show_catalog c ++ " effects=" ++ show_list show_effect es
  | Refused r => "refused " ++ r
  end.

(* ------------------------------------------------------------------ *)
(* An accepted launch has the facts of each argument.                 *)

(* The view of buffer id from off may be the argument of span sp: the
   buffer is live, the view starts at a multiple of 256 bytes, at the
   buffer's start for an index span, and covers the extent; the buffer is
   readable and writable as the span needs, passed once if written or else
   never with an index range or as a flag buffer, and holds an index range
   inside the declared one, or is an index input that a guarded launch
   checks when it runs. *)
Definition arg_ok (c : catalog) (t : Z) (guarded : bool) (buffers : list (bid * Z)) (sps : list bspan)
    (sizes : list Z) (x : (bid * Z) * bspan) : Prop :=
  let '((id, off), sp) := x in
  exists b, find_buffer c id = Some b /\
    0 <= off /\ off mod alignment = 0 /\ (bs_index sp <> None -> off = 0) /\
    off + Z.max 1 (eval (bs_ext sp) sizes) <= bf_bytes b /\
    (reads sp = true -> readable c t id = true) /\
    (writes sp = true -> writable c t id = true /\
       (count_occ bid_eq_dec (map fst buffers) id = 1%nat \/
        forall off' sp', In ((id, off'), sp') (combine buffers sps) -> bs_index sp' = None /\ bs_flag sp' = false)) /\
    (forall ix, bs_index sp = Some ix -> index_ok b ix sizes = true \/ checked_at_run guarded b sp ix sizes = true).

Lemma first_reason_none : forall (A : Type) (f : A -> option string) l,
  first_reason f l = None -> Forall (fun x => f x = None) l.
Proof.
  intros A f l. induction l as [| x l IH]; cbn; intros H; [constructor |].
  destruct (f x) eqn:E; [discriminate H |]. constructor; [exact E | exact (IH H)].
Qed.

Lemma arg_reason_ok : forall c t guarded buffers sps sizes x,
  arg_reason c t guarded buffers sps sizes x = None -> arg_ok c t guarded buffers sps sizes x.
Proof.
  intros c t guarded buffers sps sizes [[id off] sp]. unfold arg_reason, arg_ok.
  destruct (find_buffer c id) as [b |]; [| intros H; discriminate H].
  destruct (negb (readable c t id)) eqn:E3; [intros H; discriminate H |].
  destruct (negb (view_ok off sp)) eqn:E0; [intros H; discriminate H |].
  destruct (Z.ltb (bf_bytes b) (off + Z.max 1 (eval (bs_ext sp) sizes))) eqn:E1; [intros H; discriminate H |].
  destruct (writes sp && negb (writable c t id && (Nat.eqb (count_occ bid_eq_dec (map fst buffers) id) 1 ||
                                                   plain_views (combine buffers sps) id))) eqn:E4;
    [intros H; discriminate H |].
  intros H. exists b. split; [reflexivity |].
  apply negb_false_iff in E0. unfold view_ok in E0. rewrite !andb_true_iff in E0.
  destruct E0 as [[E0a E0b] E0c]. apply Z.leb_le in E0a. apply Z.eqb_eq in E0b.
  split; [exact E0a |]. split; [exact E0b |].
  split; [intros Hix; destruct (bs_index sp) as [ix |];
           [apply Z.eqb_eq; exact E0c | exfalso; apply Hix; reflexivity] |].
  apply Z.ltb_ge in E1.
  split; [exact E1 |]. split.
  { intros _. apply negb_false_iff in E3. exact E3. }
  split.
  { intros W. rewrite W in E4. cbn in E4. apply negb_false_iff, andb_true_iff in E4.
    destruct E4 as [E4 E5]. split; [exact E4 |]. apply orb_true_iff in E5. destruct E5 as [E5 | E5].
    - left. apply Nat.eqb_eq in E5. exact E5.
    - right. intros off' sp' Hin. unfold plain_views in E5. rewrite forallb_forall in E5.
      specialize (E5 _ Hin). cbn [fst snd] in E5. rewrite bid_eqb_refl in E5. cbn in E5.
      destruct (bs_index sp'); cbn in E5; [discriminate E5 |]. apply negb_true_iff in E5.
      split; [reflexivity | exact E5]. }
  intros ix Hix. rewrite Hix in H. apply orb_true_iff.
  destruct (index_ok b ix sizes || checked_at_run guarded b sp ix sizes); [reflexivity | discriminate H].
Qed.

(* Case on each condition of H, and close each case in which H states that
   a refusal is an acceptance. *)
Ltac refusals H :=
  repeat match type of H with
         | context [if ?cond then _ else _] =>
             let E := fresh "E" in destruct cond eqn:E; cbn beta iota in H; try discriminate H
         end.

Theorem check_launch_derived_sound : forall c t op gx gy gz buffers sizes scalars guarded r b,
  find_binding c op = Some b ->
  check_launch c t op gx gy gz buffers sizes scalars guarded = inl r ->
  Forall (derived_holds sizes) (b_derived b).
Proof.
  intros c t op gx gy gz buffers sizes scalars guarded r b Hb H.
  unfold check_launch in H. rewrite Hb in H.
  destruct grid_limit as [[lx ly] lz]. destruct (b_grid b) as [[mx my] mz].
  refusals H.
  match goal with E : negb (forallb (derived_check sizes) _) = false |- _ =>
    apply negb_false_iff in E; rewrite forallb_forall in E;
    apply Forall_forall; intros d Hd; specialize (E d Hd);
    unfold derived_check in E; rewrite andb_true_iff, Z.ltb_lt, Z.eqb_eq in E;
    exact E end.
Qed.

(* The dynamic shared memory of an accepted launch lies in [0, shared_max]. *)
Theorem check_launch_shared_sound : forall c t op gx gy gz buffers sizes scalars guarded r b,
  find_binding c op = Some b ->
  check_launch c t op gx gy gz buffers sizes scalars guarded = inl r ->
  0 <= eval (b_shared b) sizes <= shared_max.
Proof.
  intros c t op gx gy gz buffers sizes scalars guarded r b Hb H.
  unfold check_launch in H. rewrite Hb in H.
  destruct grid_limit as [[lx ly] lz]. destruct (b_grid b) as [[mx my] mz].
  refusals H.
  match goal with E : negb (Z.leb 0 (eval (b_shared b) sizes) && _) = false |- _ =>
    apply negb_false_iff in E; rewrite andb_true_iff, !Z.leb_le in E; exact E end.
Qed.

Theorem check_launch_sound : forall c t op gx gy gz buffers sizes scalars guarded r,
  check_launch c t op gx gy gz buffers sizes scalars guarded = inl r ->
  exists b, find_binding c op = Some b /\
    (1 <= gx <= 2 ^ 31 - 1 /\ 1 <= gy <= 65535 /\ 1 <= gz <= 65535) /\
    (let '(mx, my, mz) := b_grid b in gx <= mx /\ gy <= my /\ gz <= mz) /\
    List.length sizes = List.length (b_sizes b) /\
    Forall (fun e => fst (fst e) <= snd e <= snd (fst e)) (combine (b_sizes b) sizes) /\
    List.length buffers = List.length (spans b) /\
    List.length scalars = List.length (bounds b) /\
    Forall (fun e => fst (fst e) <= snd e <= snd (fst e)) (combine (bounds b) scalars) /\
    Forall (arg_ok c t guarded buffers (spans b) sizes) (combine buffers (spans b)).
Proof.
  intros c t op gx gy gz buffers sizes scalars guarded r H. unfold check_launch, grid_limit in H.
  destruct (find_binding c op) as [b |]; [| discriminate H]. exists b. split; [reflexivity |].
  destruct (b_grid b) as [[mx my] mz]. cbn beta iota zeta.
  cbn beta iota in H. refusals H.
  destruct (first_reason (arg_reason c t guarded buffers (spans b) sizes) (combine buffers (spans b))) eqn:F;
    [discriminate H |].
  repeat match goal with E : negb _ = false |- _ => apply negb_false_iff in E end.
  repeat match goal with E : (_ && _) = true |- _ => apply andb_true_iff in E; destruct E end.
  repeat match goal with E : Z.leb _ _ = true |- _ => apply Z.leb_le in E end.
  repeat match goal with E : Nat.eqb _ _ = true |- _ => apply Nat.eqb_eq in E end.
  split; [lia |]. split; [lia |]. split; [assumption |]. split.
  { apply Forall_forall. intros e Hin.
    match goal with E : forallb _ (combine (b_sizes b) sizes) = true |- _ =>
      pose proof (proj1 (forallb_forall _ _) E e Hin) as He end.
    apply andb_true_iff in He. destruct He as [He1 He2]. apply Z.leb_le in He1, He2. lia. }
  split; [assumption |]. split; [assumption |]. split.
  { apply Forall_forall. intros e Hin.
    match goal with E : forallb _ (combine (bounds b) scalars) = true |- _ =>
      pose proof (proj1 (forallb_forall _ _) E e Hin) as He end.
    apply andb_true_iff in He. destruct He as [He1 He2]. apply Z.leb_le in He1, He2. lia. }
  eapply Forall_impl; [| exact (first_reason_none _ _ _ F)]. intros x Hx. exact (arg_reason_ok _ _ _ _ _ _ _ Hx).
Qed.

(* ------------------------------------------------------------------ *)
(* An accepted step keeps the catalog's invariants.                   *)

Definition placed (pool : Z) (b : buffer) : Prop :=
  0 <= bf_offset b /\ 1 <= bf_bytes b /\ bf_offset b + bf_bytes b <= pool /\ bf_offset b mod alignment = 0.

Definition apart (a b : buffer) : Prop :=
  bf_offset a + bf_bytes a <= bf_offset b \/ bf_offset b + bf_bytes b <= bf_offset a.

(* Every buffer lies in the pool at a multiple of 256 bytes, has its own id,
   and is disjoint from every other buffer; every grant names a live buffer
   and a tenant other than its owner. *)
Definition inv (c : catalog) : Prop :=
  (forall b, In b (c_buffers c) -> placed (c_pool c) b) /\
  NoDup (map bf_id (c_buffers c)) /\
  (forall a b, In a (c_buffers c) -> In b (c_buffers c) -> bf_id a <> bf_id b -> apart a b) /\
  (forall g, In g (c_grants c) -> exists b, In b (c_buffers c) /\ bf_id b = fst g /\ bf_owner b <> snd g).

Lemma set_status_same : forall c id st b, In b (c_buffers (set_status c id st)) ->
  exists b0, In b0 (c_buffers c) /\ bf_id b = bf_id b0 /\ bf_offset b = bf_offset b0 /\
             bf_bytes b = bf_bytes b0 /\ bf_owner b = bf_owner b0.
Proof.
  intros c id st b H. unfold set_status in H. cbn [c_buffers] in H. apply in_map_iff in H.
  destruct H as [b0 [Hb H]]. subst b. exists b0. split; [exact H |].
  destruct (bid_eqb (bf_id b0) id); cbn; auto.
Qed.

Lemma set_status_keeps : forall c id st b0, In b0 (c_buffers c) ->
  exists b, In b (c_buffers (set_status c id st)) /\ bf_id b = bf_id b0 /\ bf_owner b = bf_owner b0.
Proof.
  intros c id st b0 H. unfold set_status. cbn [c_buffers].
  eexists. split; [apply in_map_iff; exists b0; split; [reflexivity | exact H] |].
  destruct (bid_eqb (bf_id b0) id); cbn; auto.
Qed.

Lemma set_status_inv : forall c id st, inv c -> inv (set_status c id st).
Proof.
  intros c id st [Hp [Hn [Ha Hg]]]. split; [| split; [| split]].
  - intros b Hb. destruct (set_status_same _ _ _ _ Hb) as [b0 [H0 [_ [Ho [Hy _]]]]].
    unfold placed. rewrite Ho, Hy. exact (Hp _ H0).
  - unfold set_status. cbn [c_buffers]. rewrite map_map.
    replace (map (fun x => bf_id (if bid_eqb (bf_id x) id then _ else x)) (c_buffers c))
      with (map bf_id (c_buffers c)); [exact Hn |].
    apply map_ext. intros b. destruct (bid_eqb (bf_id b) id); reflexivity.
  - intros a b HA HB Hd.
    destruct (set_status_same _ _ _ _ HA) as [a0 [A0 [Ai [Ao [Ay _]]]]].
    destruct (set_status_same _ _ _ _ HB) as [b0 [B0 [Bi [Bo [By _]]]]].
    unfold apart. rewrite Ao, Ay, Bo, By. apply Ha; [exact A0 | exact B0 | congruence].
  - intros g Hin. destruct (Hg g Hin) as [b0 [H0 [Hi Ho]]].
    destruct (set_status_keeps c id st b0 H0) as [b [Hb [Hbi Hbo]]].
    exists b. split; [exact Hb |]. split; congruence.
Qed.

Lemma written_inv : forall sizes l c, inv c -> inv (fold_left (fun acc x => written acc sizes x) l c).
Proof.
  intros sizes l. induction l as [| [[id off] sp] l IH]; intros c H; cbn; [exact H |].
  apply IH. unfold written. destruct (negb (writes sp)); [exact H |].
  destruct (bs_index sp); apply set_status_inv; exact H.
Qed.

Lemma check_launch_inv : forall c t op gx gy gz buffers sizes scalars guarded c' k,
  inv c -> check_launch c t op gx gy gz buffers sizes scalars guarded = inl (c', k) -> inv c'.
Proof.
  intros c t op gx gy gz buffers sizes scalars guarded c' k Hinv H. unfold check_launch, grid_limit in H.
  destruct (find_binding c op) as [b |]; [| discriminate H].
  destruct (b_grid b) as [[mx my] mz]. cbn beta iota in H. refusals H.
  destruct (first_reason (arg_reason c t guarded buffers (spans b) sizes) (combine buffers (spans b))); [discriminate H |].
  injection H as <- _. apply written_inv. exact Hinv.
Qed.

Lemma find_buffer_in : forall c id b, find_buffer c id = Some b -> In b (c_buffers c) /\ bf_id b = id.
Proof.
  intros c id b H. unfold find_buffer in H. apply find_some in H. destruct H as [H E].
  apply bid_eqb_eq in E. split; assumption.
Qed.

Lemma find_buffer_none : forall c id, find_buffer c id = None -> ~ In id (map bf_id (c_buffers c)).
Proof.
  intros c id H Hin. apply in_map_iff in Hin. destruct Hin as [b [Hb Hin]].
  pose proof (find_none _ _ H b Hin) as E. cbn in E. rewrite Hb, bid_eqb_refl in E. discriminate E.
Qed.

Lemma existsb_false : forall (A : Type) (f : A -> bool) l,
  existsb f l = false -> forall x, In x l -> f x = false.
Proof.
  intros A f l H x Hx. destruct (f x) eqn:E; [| reflexivity].
  rewrite (proj2 (existsb_exists f l) (ex_intro _ x (conj Hx E))) in H. discriminate H.
Qed.

Lemma NoDup_snoc : forall (A : Type) (l : list A) a, NoDup l -> ~ In a l -> NoDup (l ++ [a]).
Proof.
  intros A l a. induction l as [| x l IH]; intros Hn Ha; cbn; [constructor; [intros [] | constructor] |].
  inversion Hn as [| ? ? Hx Hl]; subst. constructor.
  - intros Hin. apply in_app_or in Hin. destruct Hin as [Hin | [Hin | []]]; [contradiction |].
    subst. apply Ha. left. reflexivity.
  - apply IH; [exact Hl |]. intros Hin. apply Ha. right. exact Hin.
Qed.

Lemma NoDup_filter_map : forall (p : buffer -> bool) l,
  NoDup (map bf_id l) -> NoDup (map bf_id (filter p l)).
Proof.
  intros p l. induction l as [| x l IH]; intros Hn; cbn; [constructor |].
  inversion Hn as [| ? ? Hx Hl]; subst. destruct (p x); [| exact (IH Hl)].
  cbn. constructor; [| exact (IH Hl)].
  intros Hin. apply Hx. apply in_map_iff in Hin. destruct Hin as [y [Hy Hin]].
  apply filter_In in Hin. apply in_map_iff. exists y. split; [exact Hy | exact (proj1 Hin)].
Qed.

Lemma check_action_inv : forall c t a c' es, inv c -> check_action c t a = inl (c', es) -> inv c'.
Proof.
  intros c t a c' es Hinv H. destruct Hinv as [Hp [Hn [Ha Hg]]].
  destruct a as [id o n | id | id u | id data element | id e s v | op gx gy gz buffers sizes scalars guarded
                | o n |]; cbn [check_action] in H.
  - (* an allocation *)
    destruct (negb (Z.eqb (fst id) t)); [discriminate H |].
    destruct (find_buffer c id) eqn:F; [discriminate H |].
    destruct (Z.ltb n 1 || Z.ltb o 0 || Z.ltb (c_pool c) (o + n)) eqn:E1; [discriminate H |].
    destruct (negb (Z.eqb (o mod alignment) 0)) eqn:E2; [discriminate H |].
    destruct (negb (in_reservation c t o n)); [discriminate H |].
    destruct (existsb (fun b => overlap (bf_offset b) (bf_bytes b) o n) (c_buffers c)) eqn:E3; [discriminate H |].
    cbv zeta in H. injection H as <- _.
    apply orb_false_iff in E1. destruct E1 as [E1 E0]. apply orb_false_iff in E1. destruct E1 as [E1 E1'].
    apply Z.ltb_ge in E0, E1, E1'. apply negb_false_iff, Z.eqb_eq in E2.
    assert (Hfar : forall b, In b (c_buffers c) -> bf_offset b + bf_bytes b <= o \/ o + n <= bf_offset b).
    { intros b Hb. pose proof (existsb_false _ _ _ E3 b Hb) as Ho. cbn beta in Ho. unfold overlap in Ho.
      apply andb_false_iff in Ho. destruct Ho as [Ho | Ho]; apply Z.ltb_ge in Ho; lia. }
    split; [| split; [| split]]; cbn [c_buffers c_pool c_grants].
    + intros b Hb. apply in_app_or in Hb. destruct Hb as [Hb | [Hb | []]]; [exact (Hp b Hb) |].
      subst b. unfold placed. cbn. repeat split; lia.
    + rewrite map_app. apply NoDup_snoc; [exact Hn | exact (find_buffer_none _ _ F)].
    + intros a b HA HB Hd. apply in_app_or in HA, HB.
      destruct HA as [HA | [HA | []]], HB as [HB | [HB | []]].
      * exact (Ha a b HA HB Hd).
      * subst b. unfold apart. cbn. destruct (Hfar a HA); lia.
      * subst a. unfold apart. cbn. destruct (Hfar b HB); lia.
      * subst a b. contradiction.
    + intros g Hin. destruct (Hg g Hin) as [b [Hb Hbg]]. exists b. split; [| exact Hbg].
      apply in_or_app. left. exact Hb.
  - (* a free *)
    destruct (find_buffer c id) as [fb |] eqn:F; [| discriminate H].
    refusals H. injection H as <- _.
    split; [| split; [| split]]; cbn [c_buffers c_pool c_grants].
    + intros b Hb. apply filter_In in Hb. exact (Hp b (proj1 Hb)).
    + apply NoDup_filter_map. exact Hn.
    + intros a b HA HB Hd. apply filter_In in HA, HB. exact (Ha a b (proj1 HA) (proj1 HB) Hd).
    + intros g Hin. apply filter_In in Hin. destruct Hin as [Hin Hne].
      destruct (Hg g Hin) as [b [Hb [Hbi Hbo]]]. exists b. split; [| split; assumption].
      apply filter_In. split; [exact Hb |]. rewrite Hbi. exact Hne.
  - (* a grant *)
    destruct (find_buffer c id) as [fb |] eqn:F; [| discriminate H].
    refusals H. injection H as <- _.
    destruct (find_buffer_in _ _ _ F) as [Hfb Hid].
    match goal with E : (negb _ || _) = false |- _ =>
      apply orb_false_iff in E; destruct E as [Eo Eu]; apply negb_false_iff, Z.eqb_eq in Eo;
      apply Z.eqb_neq in Eu end.
    split; [| split; [| split]]; cbn [c_buffers c_pool c_grants]; [exact Hp | exact Hn | exact Ha |].
    intros g Hin. apply in_app_or in Hin. destruct Hin as [Hin | [Hin | []]]; [exact (Hg g Hin) |].
    subst g. exists fb. cbn. split; [exact Hfb |]. split; [exact Hid |]. congruence.
  - (* a write *)
    destruct (find_buffer c id) as [fb |] eqn:F; [| discriminate H].
    destruct element as [[e s] |]; refusals H; [destruct (data_range e s data) as [lo hi] |];
      injection H as <- _; apply set_status_inv; exact (conj Hp (conj Hn (conj Ha Hg))).
  - (* a fill *)
    destruct (find_buffer c id) as [fb |] eqn:F; [| discriminate H].
    refusals H. destruct (element_range e s) as [lo hi]. refusals H. injection H as <- _.
    apply set_status_inv. exact (conj Hp (conj Hn (conj Ha Hg))).
  - (* a launch *)
    destruct (check_launch c t op gx gy gz buffers sizes scalars guarded) as [[c'' k] |] eqn:L; [| discriminate H].
    injection H as <- _. eapply check_launch_inv; [| exact L]. exact (conj Hp (conj Hn (conj Ha Hg))).
  - (* a reservation: the buffers and grants stay *)
    destruct (find_reservation c t); [discriminate H |]. refusals H. injection H as <- _.
    exact (conj Hp (conj Hn (conj Ha Hg))).
  - (* the end of a reservation *)
    destruct (find_reservation c t); [| discriminate H]. refusals H. injection H as <- _.
    exact (conj Hp (conj Hn (conj Ha Hg))).
Qed.

Lemma check_actions_inv : forall actions c t position done c' es,
  inv c -> check_actions c t position actions done = Accepted c' es -> inv c'.
Proof.
  intros actions. induction actions as [| a actions IH]; intros c t position done c' es Hinv H; cbn in H.
  - injection H as <- _. exact Hinv.
  - destruct (check_action c t a) as [[c1 es1] |] eqn:E; [| discriminate H].
    exact (IH _ _ _ _ _ _ (check_action_inv _ _ _ _ _ Hinv E) H).
Qed.

Lemma inv_empty : forall pool ops,
  inv {| c_pool := pool; c_buffers := []; c_grants := []; c_freed := []; c_reserved := []; c_operators := ops |}.
Proof.
  intros pool ops. split; [| split; [| split]]; cbn; [intros b [] | constructor | intros a b [] | intros g []].
Qed.

Theorem check_step_inv : forall c s c' es, inv c -> check_step c s = Accepted c' es -> inv c'.
Proof. intros c s c' es Hinv H. exact (check_actions_inv _ _ _ _ _ _ _ Hinv H). Qed.
