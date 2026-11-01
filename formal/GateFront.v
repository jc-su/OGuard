(*
  Trusted lowering of parsed PTX into GateIsa instructions and launch facts.
  The parser and this translation remain part of the trusted boundary: the
  proved checker establishes safety of the resulting instruction set.

  Known arithmetic, comparisons, conversions, selections and parameter
  loads retain their exact models where available. Only the documented
  register-result families in pure_register_op may otherwise define their
  destination registers arbitrarily. Unknown mnemonics and effect classes
  are refused; arbitrary result values do not abstract memory or control
  effects.

  Global, shared and local memory handlers emit checked accesses for their
  complete supported footprints. Unsupported local-load shapes, asynchronous
  copies and completion-memory effects, tensor-map accesses, and matrix
  collectives whose participation is not proved are refused. Ordinary
  fences only order memory, and change none of the values tracked here;
  shared loads remain arbitrary values. Termination is proved separately
  (GateTerm.v).

  Generic addresses may name global memory or the fixed private windows.
  Local stores must be inside live declared local variables; dynamic stack
  changes are refused. Module-defined device functions are inlined with
  distinct local frames. Indirect and external calls remain unsupported.
  Indexed branches require their target list and a checked in-range index.
  Trap-like behavior is checked through an impossible access under its guard.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import Lia.
From Coq Require Import PArith.PArith.
From Coq Require Import NArith.NArith.
From Coq Require Import Arith.PeanoNat.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Strings.String.
From Coq Require Import Strings.Ascii.
From Coq Require Import FSets.FMapPositive.
From Coq Require Import MSets.MSetPositive.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.
Require Import OpGate.GateDomain.
Require Import OpGate.GateMeaning.
Require Import OpGate.GateThread.
Require Import OpGate.GateInterval.
Require Import OpGate.GateCertificate.
Require Import OpGate.GateBarrier.

(* Comparisons are written with their functions: string_scope and Z_scope
   both define =? and <?. *)
Local Open Scope Z_scope.
Local Open Scope string_scope.

(* ------------------------------------------------------------------ *)
(* The parsed kernel.                                                  *)

(* An operand as the parser classifies it. *)
Inductive popnd :=
| PReg (r : string)                  (* a register, %r1 *)
| PNot (r : string)                  (* a negated predicate, !%p1 *)
| PVec (xs : list popnd)             (* a vector, {a, b}, of its elements *)
| PImm (z : Z)                       (* an exact integer *)
| PFloat (literal : string)          (* a floating literal, retaining its exact spelling *)
| PAddr (r : option string) (off : Z)
                                     (* [r + off], or None for another address form *)
| PSym (s : string) (off : Z)        (* [s + off], a parameter or a variable *)
| PTensor (m : string) (cs : list popnd)
                                     (* [m, {c0, c1}], a tensor map in register m
                                        and its coordinates, innermost first *)
| PLabel (s : string)
| PText (s : string).                (* any other operand, such as %tid.x *)

(* An instruction: its guard (negated, predicate register), mnemonic,
   suffixes in order, and operands. *)
Inductive pinstr := PI (pred : option (bool * string)) (op : string) (sfx : list string)
                       (args : list popnd).

(* A device function of the module, .func, with a body. *)
Record pfunc := {
  f_name : string;
  f_params : list string;                       (* its parameters, in order *)
  f_rets : list string;                         (* its return values, in order *)
  f_regs : list (string * nat * option Z);
  f_locals : list (string * Z * Z);             (* its local variables: name, alignment, bytes *)
  f_labels : list (string * N);
  f_branch_targets : list (string * list string);
  f_insts : list pinstr
}.

Record pkernel := {
  k_address_size : Z;
  k_target : option Z;                         (* explicitly parsed target architecture *)
  k_ptx_version : option (Z * Z);              (* explicitly parsed ISA version *)
  k_params : list (string * string);            (* name, type *)
  k_regs : list (string * nat * option Z);      (* register declarations: name, width, and
                                                   the count of an array *)
  k_shared : list (string * Z * Z);             (* shared variables, their alignments, and bytes *)
  k_locals : list (string * Z * Z * frame);     (* local variables: name, alignment, bytes, and the
                                                   frame of the call they belong to *)
  k_reqntid : option (Z * Z * Z);
  k_cluster : bool;                             (* the kernel declares a cluster that may hold
                                                   more than one CTA *)
  k_labels : list (string * N);                 (* a label and the instruction it precedes *)
  k_branch_targets : list (string * list string); (* each .branchtargets list and its labels *)
  k_funcs : list pfunc;                         (* the module's device functions *)
  k_insts : list pinstr
}.

(* ------------------------------------------------------------------ *)
(* The binding.                                                       *)

(* A constant plus a sum of coefficient times size symbol; a symbol is its
   index below b_sizes. *)
Record affine := { af_const : Z; af_terms : list (Z * Z) }.

(* The elements of an index input or output: elem bytes each, read as
   signed or unsigned integers, each in [lo, hi]. *)
Record ixrange := { ir_elem : Z; ir_signed : bool; ir_lo : affine; ir_hi : affine }.

(* A buffer parameter: the base of ext bytes, a multiple of 2^align, that
   the launch may access in the direction, an index range for an index
   input (read) or output (write), and whether it is a flag buffer, whose
   words of 4 bytes a launch may only set to 1. *)
Record bspan := { bs_dir : dir; bs_ext : affine; bs_align : Z; bs_index : option ixrange; bs_flag : bool }.

(* What one PTX parameter holds: a buffer's base, an affine of the size
   symbols, a constant, an integer with bounds, or zero. *)
Inductive holding :=
| PBase (sp : bspan)
| PSize (a : affine)
| PConst (z : Z)
| PBounded (lo hi : Z)
| PZero.

(* A target size is exactly floor(numerator / positive divisor). *)
Record derived_size := { ds_target : Z; ds_numerator : affine; ds_divisor : Z }.
Definition derived_affine (d : derived_size) : affine :=
 {| af_const := af_const (ds_numerator d);
    af_terms := app (af_terms (ds_numerator d)) [(ds_target d, - ds_divisor d)] |}.

(* What the runtime establishes at each launch of a kernel. Size symbol k
   takes a value within its declared minimum and maximum at each launch, the executor
   launches every block with the binding's size, and no launch has a grid
   larger than the binding's maxima in any dimension. This record is the one
   the Python binding mirrors field by field. *)
Record binding := {
  b_sizes : list (Z * Z);                         (* each size symbol's inclusive minimum and maximum, in order *)
  b_params : list (string * holding);       (* each PTX parameter by name, in order *)
  b_block : Z * Z * Z;                      (* the block size of every launch *)
  b_grid : Z * Z * Z;                       (* the largest grid of any launch, in blocks *)
  b_shared : affine;                             (* affine bytes of dynamic shared memory *)
  b_derived : list derived_size                   (* exact quotient relations enforced at launch *)
}.

(* The largest grid of every GPU, in blocks: a binding's grid maxima lie
   within it. *)
Definition grid_limit : Z * Z * Z := (2 ^ 31 - 1, 65535, 65535).

(* The largest dynamic shared memory of a launch: the 48 KiB that a launch
   gets without opting in to more. *)
Definition shared_max : Z := 49152.

(* The largest value of a size symbol: the row limit of a 32-bit signed
   count. *)
Definition size_max : Z := 2 ^ 31 - 1.

(* ------------------------------------------------------------------ *)
(* Translation state and errors.                                       *)

Inductive res (A : Type) :=
| Ok (a : A)
| Err (msg : string) (inst : option N).      (* the reason, and the instruction's index *)
Arguments Ok {A}.
Arguments Err {A}.

(* A name's key: its bytes, eight bits each, so distinct names have
   distinct keys. *)
Definition push (b : bool) (p : positive) : positive := if b then xI p else xO p.

Fixpoint key (s : string) : positive :=
  match s with
  | EmptyString => xH
  | String (Ascii a0 a1 a2 a3 a4 a5 a6 a7) t =>
      push a0 (push a1 (push a2 (push a3 (push a4 (push a5 (push a6 (push a7 (key t))))))))
  end.

(* The registers are numbered from 1 in the order the translation meets
   them. A parameter is the register `param:<name>`, which the parameter's
   loads read and no instruction writes, and a shared variable is the
   register `shared:<name>`, its address. *)
Record tstate := {
  ts_ids : PM.t reg;                  (* the register of each name, by the name's key *)
  ts_names : list (string * reg);     (* the numbered names, the last first *)
  ts_widths : PM.t nat;
  ts_next : reg;
  ts_at : option N;                   (* the instruction being translated *)
  ts_fr : frame                       (* the frame of the call it belongs to *)
}.

Definition Tr (A : Type) := tstate -> res (A * tstate).

Definition ret {A} (a : A) : Tr A := fun s => Ok (a, s).

Definition bind {A B} (m : Tr A) (f : A -> Tr B) : Tr B :=
  fun s => match m s with Ok (a, s') => f a s' | Err e i => Err e i end.

Definition refuse {A} (msg : string) : Tr A := fun s => Err msg (ts_at s).

Notation "x <- m ;; k" := (bind m (fun x => k)) (at level 61, m at next level, right associativity).

Definition get_state : Tr tstate := fun s => Ok (s, s).

Definition set_at (n : option N) (fr : frame) : Tr unit :=
  fun s => Ok (tt, {| ts_ids := ts_ids s; ts_names := ts_names s; ts_widths := ts_widths s;
                     ts_next := ts_next s; ts_at := n; ts_fr := fr |}).

Definition cur_frame : Tr frame := fun s => Ok (ts_fr s, s).

Fixpoint mapT {A B} (f : A -> Tr B) (l : list A) : Tr (list B) :=
  match l with
  | [] => ret []
  | x :: t => y <- f x ;; ys <- mapT f t ;; ret (y :: ys)
  end.

(* ------------------------------------------------------------------ *)
(* Strings.                                                            *)

Definition sin (x : string) (l : list string) : bool := existsb (String.eqb x) l.

(* A number as its decimal numeral. *)
Fixpoint digits_of (fuel : nat) (n : N) (acc : string) : string :=
  match fuel with
  | O => acc
  | S f =>
      let c := String (ascii_of_N (48 + N.modulo n 10)) acc in
      if N.ltb n 10 then c else digits_of f (N.div n 10) c
  end.

Definition nstr (n : N) : string := digits_of 24 n "".

(* No string occurs twice in l. *)
Fixpoint distinct (l : list string) : bool :=
  match l with [] => true | x :: t => negb (sin x t) && distinct t end.

Fixpoint list_eqs (a b : list string) : bool :=
  match a, b with
  | [], [] => true
  | x :: a', y :: b' => String.eqb x y && list_eqs a' b'
  | _, _ => false
  end.

Definition first (l : list string) : string := match l with x :: _ => x | [] => "" end.

Definition defined {A} (o : option A) : bool := match o with Some _ => true | None => false end.

Fixpoint has_char (c : ascii) (s : string) : bool :=
  match s with EmptyString => false | String d t => Ascii.eqb c d || has_char c t end.

Definition starts (c : ascii) (s : string) : bool :=
  match s with String d _ => Ascii.eqb c d | EmptyString => false end.

Definition starts_in (cs : list ascii) (s : string) : bool :=
  match s with String d _ => existsb (Ascii.eqb d) cs | EmptyString => false end.

Definition drop (n : nat) (s : string) : string := substring n (String.length s - n) s.

Fixpoint digits (s : string) (acc : Z) : option Z :=
  match s with
  | EmptyString => Some acc
  | String c t =>
      let d := Z.of_nat (nat_of_ascii c) - 48 in
      if Z.leb 0 d && Z.leb d 9 then digits t (10 * acc + d) else None
  end.

(* A decimal numeral without leading zeros. *)
Definition decimal (s : string) : option Z :=
  match s with
  | EmptyString => None
  | String c t => if Ascii.eqb c "0"%char then match t with EmptyString => Some 0 | _ => None end
                  else digits s 0
  end.

(* The width of a type suffix: pred, or b, s, u, or f with 8, 16, 32, or 64
   bits. *)
Definition type_width (t : string) : option nat :=
  if String.eqb t "pred" then Some 1%nat else
  match t with
  | String c n =>
      if existsb (Ascii.eqb c) ["b"; "s"; "u"; "f"]%char then
        if String.eqb n "8" then Some 8%nat else if String.eqb n "16" then Some 16%nat
        else if String.eqb n "32" then Some 32%nat else if String.eqb n "64" then Some 64%nat
        else None
      else None
  | EmptyString => None
  end.

Definition types (sf : list string) : list string := filter (fun t => defined (type_width t)) sf.

Definition bsu (t : string) : bool := starts_in ["b"; "s"; "u"]%char t.
Definition bu (t : string) : bool := starts_in ["b"; "u"]%char t.
Definition is_b (t : string) : bool := starts "b"%char t.
Definition is_s (t : string) : bool := starts "s"%char t.

(* The bytes of a memory type suffix. *)
Definition size_of (t : string) : option Z :=
  if String.eqb t "bf16" then Some 2 else
  if String.eqb t "f16x2" || String.eqb t "bf16x2" then Some 4 else
  match t with
  | String c n =>
      if existsb (Ascii.eqb c) ["b"; "s"; "u"; "f"]%char then
        if String.eqb n "8" then Some 1 else if String.eqb n "16" then Some 2
        else if String.eqb n "32" then Some 4 else if String.eqb n "64" then Some 8
        else if String.eqb n "128" then Some 16 else None
      else None
  | EmptyString => None
  end.

(* v followed by digits: a vector suffix, of which v2, v4, and v8 are
   supported. *)
Definition vec_suffix (t : string) : bool :=
  match t with
  | String c (String _ _ as n) => Ascii.eqb c "v"%char && defined (digits n 0)
  | _ => false
  end.

Definition lanes (t : string) : option Z :=
  if String.eqb t "v2" then Some 2 else if String.eqb t "v4" then Some 4
  else if String.eqb t "v8" then Some 8 else None.

Definition vector (sf : list string) : bool := existsb (fun s => defined (lanes s)) sf.

Definition specials := ["%tid.x"; "%tid.y"; "%tid.z"; "%ntid.x"; "%ntid.y"; "%ntid.z";
                        "%ctaid.x"; "%ctaid.y"; "%ctaid.z"; "%nctaid.x"; "%nctaid.y"; "%nctaid.z";
                        "%laneid"; "%clusterid.x"; "%clusterid.y"; "%clusterid.z"].

(* Z.log2 of the lowest set bit: a multiple of m is a multiple of 2 to this. *)
Definition low_log2 (m : Z) : Z := Z.log2 (Z.land m (- m)).

(* ------------------------------------------------------------------ *)
(* Instruction classes.                                                *)

(* The state spaces an instruction names, in order. The parser reads
   .shared::cta as .shared. *)
Definition spaces (sf : list string) : list string :=
  filter (fun s => sin s ["global"; "shared"; "shared::cluster"; "local"; "param"; "param::entry";
                          "param::func"; "const"]) sf.

Definition generic_space (sf : list string) : bool := match spaces sf with [] => true | _ => false end.

Definition is_shared (s : string) : bool := String.eqb s "shared" || String.prefix "shared::" s.

(* An instruction that may access global memory. A memory instruction without
   a state space uses a generic address, which may name global memory. *)
Definition global_op (mn : string) (sf : list string) : bool :=
  if sin mn ["ld"; "ldu"; "st"; "atom"; "red"] then
    sin "global" sf || generic_space sf
  else false.

Definition loads_mem (mn : string) (sf : list string) : bool :=
  sin mn ["ld"; "ldu"; "atom"; "red"].

Definition stores_mem (mn : string) (sf : list string) : bool :=
  sin mn ["st"; "atom"; "red"].

Definition address_operand (mn : string) (sf : list string) (ops : list popnd) : option popnd :=
  nth_error ops (if stores_mem mn sf && negb (String.eqb mn "atom") then 0 else 1)%nat.

(* The bytes of a supported scalar/vector memory instruction. *)
Definition access_bytes (mn : string) (sf : list string) (ops : list popnd) : option Z :=
  if negb (sin mn ["ld"; "ldu"; "st"; "atom"; "red"]) then None
  else match filter (fun t => defined (size_of t)) sf, filter vec_suffix sf with
       | [t], [] => size_of t
       | [t], [v] => match size_of t, lanes v with Some b, Some n => Some (b * n) | _, _ => None end
       | _, _ => None
       end.

(* Parameter loads are lowered from the launch binding rather than from
   memory-region accesses. No other bracketed effect bypasses dispatch. *)
Definition parameter_load (mn : string) (sf : list string) : bool :=
  String.eqb mn "ld" &&
  match filter (fun s => sin s ["param"; "param::entry"; "param::func"]) sf with
  | [_] => true
  | _ => false
  end.

Definition is_addr (o : popnd) : bool :=
  match o with PAddr _ _ | PSym _ _ | PTensor _ _ => true | PText s => starts "["%char s | _ => false end.

(* An operand that may name memory. *)
Definition brackets (o : popnd) : bool :=
  match o with
  | PAddr _ _ | PSym _ _ | PTensor _ _ => true
  | PVec xs => existsb (fun x => match x with
                                 | PAddr _ _ | PSym _ _ | PVec _ => true
                                 | PText s => has_char "["%char s
                                 | _ => false
                                 end) xs
  | PText s => has_char "["%char s
  | _ => false
  end.

(* The one register an operand names, alone or as a vector of one. *)
Definition single (o : popnd) : option string :=
  match o with PReg r => Some r | PVec [PReg r] => Some r | _ => None end.

Definition cmp_of (s : string) : option cmp :=
  if String.eqb s "lt" then Some CLt else if String.eqb s "le" then Some CLe
  else if String.eqb s "gt" then Some CGt else if String.eqb s "ge" then Some CGe
  else if String.eqb s "eq" then Some CEq else if String.eqb s "ne" then Some CNe else None.

Definition binop_of (mn : string) (signed : bool) : option binop :=
  if String.eqb mn "add" then Some BAdd else if String.eqb mn "sub" then Some BSub
  else if String.eqb mn "mul" then Some BMul else if String.eqb mn "shl" then Some BShl
  else if String.eqb mn "shr" then Some (if signed then BShrs else BShr)
  else if String.eqb mn "and" then Some BAnd else if String.eqb mn "or" then Some BOr
  else if String.eqb mn "xor" then Some BXor else None.

(* An immediate first operand of a commutative operation goes second. *)
Definition swap (a b : option opnd) : option opnd * option opnd :=
  match a with Some (OImm _) => (b, a) | _ => (a, b) end.

(* ------------------------------------------------------------------ *)
(* Translation of one kernel.                                          *)

Section Translate.

Variable K : pkernel.
Variable Mf : binding.

(* The CTA runs in a cluster that may hold other CTAs. *)
Definition clustered : bool := k_cluster K.

(* A register the declaration names: the declared name, or for an array of
   c registers, the name followed by a numeral below c. *)
Definition names (d : string * nat * option Z) (r : string) : bool :=
  let '(name, _, count) := d in
  match count with
  | None => String.eqb r name
  | Some c => String.prefix name r &&
              match decimal (drop (String.length name) r) with Some v => Z.ltb v c | None => false end
  end.

Definition decls (r : string) := filter (fun d => names d r) (k_regs K).

(* Whether r is a declared register. A register with several declarations
   needs scope resolution, which the translation does not model. *)
Definition declared (r : string) : Tr bool :=
  match decls r with
  | [] => ret false
  | [_] => ret true
  | _ => refuse (r ++ " has several declarations")
  end.

(* The register of a name, numbered at its first use, of the given width or
   its declared one. *)
Definition reg_of (name : string) (w : option nat) : Tr reg :=
  fun s => match PM.find (key name) (ts_ids s) with
  | Some r => Ok (r, s)
  | None =>
      let w' := match w with
                | Some n => Some n
                | None => match decls name with [(_, n, _)] => Some n | _ => None end
                end in
      match w' with
      | None => Err (name ++ " has no declared width") (ts_at s)
      | Some n =>
          let r := ts_next s in
          Ok (r, {| ts_ids := PM.add (key name) r (ts_ids s); ts_names := (name, r) :: ts_names s;
                    ts_widths := PM.add r n (ts_widths s); ts_next := Pos.succ r; ts_at := ts_at s;
                    ts_fr := ts_fr s |})
      end
  end.

Definition width_of (r : reg) : Tr nat :=
  fun s => match PM.find r (ts_widths s) with
           | Some n => Ok (n, s)
           | None => Err "a register without a width" (ts_at s)
           end.

Definition param (name : string) : Tr reg :=
  match find (fun p => String.eqb (fst p) name) (k_params K) with
  | None => refuse (name ++ " is not a parameter of the kernel")
  | Some (_, ty) =>
      match type_width ty with
      | None => refuse ("parameter " ++ name ++ " has type " ++ ty)
      | Some w => reg_of ("param:" ++ name) (Some w)
      end
  end.

(* The register of size symbol k, fixed at launch. *)
Definition size_reg (k : Z) : Tr reg :=
  if Z.ltb k 0 || Z.leb (Z.of_nat (List.length (b_sizes Mf))) k then refuse "a size symbol the binding does not declare"
  else reg_of ("size:" ++ nstr (Z.to_N k)) (Some 64%nat).

(* What the binding says parameter name holds. *)
Definition param_of (name : string) : option holding :=
  match find (fun p => String.eqb (fst p) name) (b_params Mf) with
  | Some (_, p) => Some p
  | None => None
  end.

Definition operand (o : popnd) : Tr (option opnd) :=
  match o with
  (* a special register without a dot, such as %laneid, reads as a name *)
  | PReg r => if sin r specials then (x <- reg_of r (Some 32%nat) ;; ret (Some (OReg x))) else
              d <- declared r ;;
              if d then (x <- reg_of r None ;; ret (Some (OReg x))) else ret None
  | PText s => if sin s specials then (x <- reg_of s (Some 32%nat) ;; ret (Some (OReg x)))
               else ret None
  | PImm z => ret (Some (OImm z))
  | _ => ret None
  end.

Definition guard_reg (p : option (bool * string)) : Tr guard :=
  match p with
  | None => ret None
  | Some (neg, r) =>
      d <- declared r ;;
      if d then (x <- reg_of r None ;; ret (Some (neg, x)))
      else refuse ("guard " ++ r ++ " is not a declared register")
  end.

(* The parts of s between the characters c. *)
Fixpoint split_on (c : ascii) (s acc : string) : list string :=
  match s with
  | EmptyString => [acc]
  | String d t => if Ascii.eqb c d then acc :: split_on c t "" else split_on c t (acc ++ String d "")
  end.

(* The registers an operand writes when it is a destination. A pair p|q, as
   setp and elect write, names two. *)
Definition atom_names (o : popnd) : Tr (list string) :=
  match o with
  | PReg r | PNot r => ret [r]
  | PAddr (Some r) _ => ret [r]
  | PImm _ | PFloat _ | PLabel _ | PSym _ _ => ret []
  | PText s =>
      if has_char "|"%char s then
        let parts := split_on "|"%char s "" in
        if forallb (starts "%"%char) parts then ret parts
        else refuse ("operand " ++ s ++ " is not a pair of registers")
      else if has_char "%"%char s then refuse ("operand " ++ s ++ " is not a register")
      else ret []
  | PAddr None _ | PVec _ | PTensor _ _ => refuse "an operand the translation does not model"
  end.

Definition dest_names (o : popnd) : Tr (list string) :=
  match o with
  | PVec xs => fold_right (fun x acc => a <- atom_names x ;; b <- acc ;; ret (app a b)) (ret []) xs
  | _ => atom_names o
  end.

Definition regs_of (ns : list string) : Tr (list reg) := mapT (fun n => reg_of n None) ns.

Definition all_declared (ns : list string) : Tr unit :=
  _ <- mapT (fun n => d <- declared n ;;
                      if d then ret tt else refuse (n ++ " is not a declared register")) ns ;;
  ret tt.

(* An access of kind k and width bytes at the address that operand o names. *)
Definition mem_at (g : guard) (k : mkind) (o : popnd) (w : Z) (ds : list reg) : Tr instr :=
  match o with
  | PAddr (Some r) off =>
      d <- declared r ;;
      if negb d then refuse (r ++ " is not a declared register")
      else (a <- reg_of r None ;; fr <- cur_frame ;; ret (IMem g k false a off w None fr ds))
  | _ => refuse "address is not a register and an offset"
  end.

(* A global memory instruction: its address register and offset, its width
   in bytes, and whether it loads, stores, or both. One without a state space
   is a global access at its generic address. *)
(* A register that no PTX name denotes, of width w. *)
Definition fresh (w : nat) : Tr reg :=
  fun s => let r := ts_next s in
    Ok (r, {| ts_ids := ts_ids s; ts_names := ts_names s; ts_widths := PM.add r w (ts_widths s);
              ts_next := Pos.succ r; ts_at := ts_at s; ts_fr := ts_fr s |}).

(* The value of a store of one register or number; a vector of one
   register, as Triton writes st.global.b32 [a], { %r }, is that register. *)
Definition store_value (mn : string) (sf : list string) (ops : list popnd) : Tr (option opnd) :=
  if String.eqb mn "st" && negb (vector sf) then
    match ops with
    | [_; o1] => operand (match single o1 with Some r => PReg r | None => o1 end)
    | _ => ret None
    end
  else ret None.

Definition global_access (g : guard) (mn : string) (sf : list string) (ops : list popnd) : Tr (list instr) :=
  (* the warp executes it as one access, whose behavior is undefined unless
     every thread of a whole warp gives it the same operands *)
  if sin "aligned" sf then refuse "warp-collective global access"
  else match address_operand mn sf ops with
  | Some (PAddr (Some r) off) =>
      d <- declared r ;;
      if negb d then refuse (r ++ " is not a declared register") else
      (a <- reg_of r None ;;
       match access_bytes mn sf ops with
       | None => refuse "access without a width"
       | Some w =>
           let k := if loads_mem mn sf && stores_mem mn sf then KRmw
                    else if stores_mem mn sf then KStore else KLoad in
           let o0 := match ops with o :: _ => o | [] => PVec [] end in
           ds <- (if sin mn ["st"; "red"] then ret []
                  else (ns <- dest_names o0 ;; regs_of ns)) ;;
           let gen := generic_space sf in
           sv <- store_value mn sf ops ;;
           fr <- cur_frame ;;
           let mem := IMem g k gen a off w sv fr ds in
           match k, types sf, single o0 with
           | KLoad, [t], Some dn =>
               if sin mn ["ld"; "ldu"] && negb (vector sf) then
                 (dest <- reg_of dn None ;; wd <- width_of dest ;;
                  match type_width t with
                  | Some bits =>
                      if bu t || Nat.eqb wd bits then ret [ILoad g gen bits dest a off]
                      (* a signed load narrower than its register is the load
                         of its bits and their sign extension *)
                      else (v <- fresh bits ;; ret [ILoad g gen bits v a off; ICvt g wd bits true dest v])
                  | None => ret [mem]
                  end)
               else ret [mem]
           | _, _, _ => ret [mem]
           end
       end)
  | _ => refuse "address is not a register and an offset"
  end.

(* A shared memory load of one register. None for any other shared memory
   instruction, which the generic translation gives arbitrary destinations:
   shared memory holds what any thread of the block may have written, so a
   store to it or a copy into it changes no register. *)
Definition shared_name (o : popnd) : option string :=
  match o with
  | PText s | PLabel s =>
      if existsb (fun v => let '(x, _, _) := v in String.eqb x s) (k_shared K) then Some s else None
  | _ => None
  end.

(* The register and the offset of a shared memory address: a declared
   register, or the address of a shared variable, a register fixed at launch,
   as Triton's [global_smem+16]. *)
Definition shared_base (o : popnd) : Tr (option (reg * Z)) :=
  match o with
  | PAddr (Some r) off =>
      dr <- declared r ;;
      if negb dr then ret None else (a <- reg_of r None ;; ret (Some (a, off)))
  | PSym s off =>
      match shared_name (PText s) with
      | Some v => a <- reg_of ("shared:" ++ v) (Some 32%nat) ;; ret (Some (a, off))
      | None => ret None
      end
  | _ => ret None
  end.

Definition shared_load (g : guard) (mn : string) (sf : list string) (ops : list popnd)
    : Tr (option (list instr)) :=
  if negb (String.eqb mn "ld") || vector sf then ret None else
  match ops, types sf with
  | [o0; oa], [t] =>
      match single o0 with
      | Some dn =>
          dd <- declared dn ;;
          if negb dd then ret None else
          (b <- shared_base oa ;;
           match b with
           | None => ret None
           | Some (a, off) =>
               dest <- reg_of dn None ;; wa <- width_of a ;; wd <- width_of dest ;;
               match type_width t with
               | Some bits =>
                   if negb (Nat.eqb wa 32) then ret None
                   else if bu t || Nat.eqb wd bits then ret (Some [ILdS g bits dest a off])
                   (* a signed load narrower than its register: the load of its
                      bits and their sign extension *)
                   else (v <- fresh bits ;; ret (Some [ILdS g bits v a off; ICvt g wd bits true dest v]))
               | None => ret None
               end
           end)
      | None => ret None
      end
  | _, _ => ret None
  end.

(* bfe.s<w> d, a, p, n with 1 <= n and p + n <= w, as the shift of the field
   to the top bits and the arithmetic shift back, which sign-extends it. *)
Definition signed_field (g : guard) (mn : string) (sf : list string) (ops : list popnd) (d : reg)
    : Tr (option (list instr)) :=
  match ops, sf with
  | [_; oa; op; on], [t] =>
      if String.eqb mn "bfe" && (String.eqb t "s32" || String.eqb t "s64") then
        (a <- operand oa ;; p <- operand op ;; n <- operand on ;;
         match type_width t, a, p, n with
         | Some w, Some (OReg a), Some (OImm p), Some (OImm n) =>
             let wz := Z.of_nat w in
             ret (if Z.leb 1 n && Z.leb 0 p && Z.leb (p + n) wz
                  then Some [IBin g BShl w d a (OImm (wz - p - n)); IBin g BShrs w d d (OImm (wz - n))]
                  else None)
         | _, _, _, _ => ret None
         end)
      else ret None
  | _, _ => ret None
  end.

(* abs.s<w> d, a as the larger of a and -a, signed, at width w: -a of the
   least value wraps to itself, which abs gives for it, as the GPU does
   (tests/gpu). *)
Definition absolute (g : guard) (mn : string) (sf : list string) (ops : list popnd) (d : reg)
    : Tr (option (list instr)) :=
  match ops, sf with
  | [_; oa], [t] =>
      if String.eqb mn "abs" && (String.eqb t "s16" || String.eqb t "s32" || String.eqb t "s64") then
        (a <- operand oa ;;
         match type_width t, a with
         | Some w, Some (OReg a) =>
             v <- fresh w ;; ret (Some [IMad g w v a (OImm (-1)) (OImm 0); IMinMax g false true w d a (OReg v)])
         | _, _ => ret None
         end)
      else ret None
  | _, _ => ret None
  end.

Definition local_name (o : popnd) : option string :=
  match o with
  | PText s | PLabel s =>
      if existsb (fun v => let '(x, _, _, _) := v in String.eqb x s) (k_locals K) then Some s else None
  | _ => None
  end.

(* The access of kind k and width bytes to shared memory at operand o: a
   register and an offset, or a shared variable and an offset; sv is the
   value of a store of one register or number. *)
Definition shared_at (g : guard) (k : mkind) (o : popnd) (w : Z) (sv : option opnd) : Tr instr :=
  match o with
  | PAddr (Some r) off =>
      d <- declared r ;;
      if negb d then refuse (r ++ " is not a declared register") else
      (a <- reg_of r None ;; ret (IShA g k a off w sv))
  | PSym s off =>
      match shared_name (PText s) with
      | Some v => a <- reg_of ("shared:" ++ v) (Some 32%nat) ;; ret (IShA g k a off w sv)
      | None => refuse "a shared memory address that is not a register or a shared variable"
      end
  | _ => refuse "a shared memory address that is not a register or a shared variable"
  end.

(* The register that holds the base of the window of state space sp,
   "shared" or "local", in the generic address space, fixed at launch. *)
Definition window_reg (sp : string) : Tr reg := reg_of ("window:" ++ sp) (Some 64%nat).

(* d := the address of a shared or local variable, a 32-bit register fixed
   at launch, at d's width: 32 bits as it is, 64 bits zero-extended. *)
Definition var_address (g : guard) (d : reg) (wd : nat) (name : string) (ts : list string) : Tr (option instr) :=
  x <- reg_of name (Some 32%nat) ;;
  if Nat.eqb wd 32 && (list_eqs ts ["b32"] || list_eqs ts ["u32"]) then ret (Some (IMov g d x))
  else if Nat.eqb wd 64 && (list_eqs ts ["b64"] || list_eqs ts ["u64"]) then ret (Some (ICvt g 64 32 false d x))
  else ret None.

(* An integer instruction writing register d, with exactly the modifiers it
   names: a floating-point type or another modifier (.sat, .cc, .relu, .hi,
   a rounding mode) computes something else. *)
Definition integer (g : guard) (mn : string) (sf : list string) (ops : list popnd) (d : reg)
    : Tr (option instr) :=
  match types sf with
  | [t] =>
    match type_width t with
    | Some w =>
      if negb (bsu t) then ret None else
      let sg := is_s t in
      let mods := filter (fun s => negb (String.eqb s t)) sf in
      let plain := match mods with [] => true | _ => false end in
      let wide := list_eqs mods ["wide"] && Nat.eqb w 32 in
      match ops with
      | [_; o1] =>
          (* -a is a * -1 + 0 and ~a is a * -1 - 1 at the register's width *)
          let c := if String.eqb mn "neg" && plain && sg then Some 0
                   else if String.eqb mn "not" && plain && is_b t then Some (-1) else None in
          match c with
          | Some c =>
              a <- operand o1 ;;
              ret (match a with Some (OReg a) => Some (IMad g w d a (OImm (-1)) (OImm c)) | _ => None end)
          | None => ret None
          end
      | [_; o1; o2] =>
          let two := fun f => a <- operand o1 ;; b <- operand o2 ;; ret (f a b) in
          if sin mn ["min"; "max"; "div"; "rem"] && plain && negb (is_b t) then
            two (fun a b => match a, b with
                            | Some (OReg a), Some b =>
                                Some (if sin mn ["min"; "max"] then IMinMax g (String.eqb mn "min") sg w d a b
                                      else IDiv g (String.eqb mn "rem") sg w d a b)
                            | _, _ => None
                            end)
          else if String.eqb mn "mul" && wide then
            two (fun a b => match swap a b with
                            | (Some (OReg a), Some b) => Some (IWide g sg d a b None)
                            | _ => None
                            end)
          else match binop_of mn sg with
               | Some o =>
                   if list_eqs mods (if String.eqb mn "mul" then ["lo"] else []) then
                     two (fun a b => match (if sin mn ["add"; "mul"; "and"; "or"; "xor"] then swap a b else (a, b)) with
                                     | (Some (OReg a), Some b) => Some (IBin g o w d a b)
                                     | (Some (OImm z), Some (OReg b)) =>
                                         (* z - b is (-1) * b + z modulo the instruction width. *)
                                         match o with
                                         | BSub => Some (IMad g w d b (OImm (-1)) (OImm z))
                                         | _ => None
                                         end
                                     | _ => None
                                     end)
                   else ret None
               | None => ret None
               end
      | [_; o1; o2; o3] =>
          let three := fun f => a <- operand o1 ;; b <- operand o2 ;; c <- operand o3 ;; ret (f a b c) in
          if String.eqb mn "selp" && plain then
            three (fun a b c => match a, b, c with
                                | Some a, Some b, Some (OReg p) => Some (ISelp g w d a b p)
                                | _, _, _ => None
                                end)
          else if String.eqb mn "mad" && list_eqs mods ["lo"] then
            three (fun a b c => match swap a b, c with
                                | (Some (OReg a), Some b), Some c => Some (IMad g w d a b c)
                                | _, _ => None
                                end)
          else if String.eqb mn "mad" && wide then
            three (fun a b c => match swap a b, c with
                                | (Some (OReg a), Some b), Some (OReg c) => Some (IWide g sg d a b (Some c))
                                | _, _ => None
                                end)
          else if String.eqb mn "bfe" && plain && bu t then
            three (fun a b c => match a, b, c with
                                | Some (OReg a), Some (OImm p), Some (OImm n) =>
                                    if Z.leb 0 p && Z.leb 0 n then Some (IBfe g w d a (Z.to_nat p) (Z.to_nat n))
                                    else None
                                | _, _, _ => None
                                end)
          else ret None
      | _ => ret None
      end
    | None => ret None
    end
  | _ => ret None
  end.

(* The instruction writing register d, if the instruction set models it. *)
Definition exact (g : guard) (mn : string) (sf : list string) (ops : list popnd) (d : reg)
    : Tr (option instr) :=
  wd <- width_of d ;;
  let ts := types sf in
  if String.eqb mn "mov" then
    match ops with
    | [_; o1] =>
        match shared_name o1, local_name o1 with
        (* the shared memory address of a shared variable, which every
           thread of the block shares, or the local memory address of a
           local variable of the thread *)
        | Some v, _ => var_address g d wd ("shared:" ++ v) ts
        | None, Some v => var_address g d wd ("local:" ++ v) ts
        | None, None =>
            o <- operand o1 ;;
            ret (match o with
                 | Some (OReg x) => Some (IMov g d x)
                 | Some (OImm z) => Some (IImm g d z)
                 | None => None
                 end)
        end
    | _ => ret None
    end
  else if String.eqb mn "ld" && sin "param" sf then
    match ops with
    | [_; PSym name off] =>
        p <- param name ;; wp <- width_of p ;;
        (* a load at the parameter's start reads its low w bits; one at an
           offset or wider than the parameter reads other bytes of the
           parameter space *)
        ret (match ts with
             | [t] =>
                 match type_width t with
                 | Some w =>
                     if negb (Z.eqb off 0) || Nat.ltb wp w then None
                     else if Nat.eqb w wd && Nat.eqb wd wp then Some (IMov g d p)
                     (* the low w bits, extended to the register's width *)
                     else Some (ICvt g wd w (is_s t) d p)
                 | None => None
                 end
             | _ => None
             end)
    | _ => ret None
    end
  (* A predicate is one bit, so bitwise negation is equality with zero. *)
  else if String.eqb mn "not" && list_eqs sf ["pred"] && Nat.eqb wd 1 then
    match ops with
    | [_;o1] => a <- operand o1 ;;
      ret (match a with Some (OReg a) => Some (ISetp g CEq 1 false d a (OImm 0)) | _ => None end)
    | _ => ret None end
  else if (String.eqb mn "and" || String.eqb mn "or" || String.eqb mn "xor") && sin "pred" sf then
    match ops with
    | [_; o1; o2] =>
        a <- operand o1 ;; b <- operand o2 ;;
        ret (match a, b with
             | Some (OReg a), Some (OReg b) =>
                 Some (if String.eqb mn "and" then IAndp g d a b
                       else if String.eqb mn "or" then IOrp g d a b else IXorp g d a b)
             | _, _ => None
             end)
    | _ => ret None
    end
  else if String.eqb mn "setp" then
    match ops, sf with
    | [_; o1; o2], [c; t] =>
        match cmp_of c with
        | Some cm =>
            if bsu t then
              (a <- operand o1 ;; b <- operand o2 ;;
               ret (match a, b, type_width t with
                    | Some (OReg a), Some b, Some w => Some (ISetp g cm w (is_s t) d a b)
                    | _, _, _ => None
                    end))
            else ret None
        | None => ret None
        end
    | _, _ => ret None
    end
  else if String.eqb mn "cvta" then
    (* a global address is its generic address; a shared or local address
       is its offset in the window of its state space *)
    match ops with
    | [_; o1] =>
        o <- operand o1 ;;
        match o with
        | Some (OReg x) =>
            wx <- width_of x ;;
            if negb (Nat.eqb wd 64 && Nat.eqb wx 64) then ret None
            else if list_eqs sf ["to"; "global"; "u64"] || list_eqs sf ["global"; "u64"] then ret (Some (IMov g d x))
            else if list_eqs sf ["shared"; "u64"] then (w <- window_reg "shared" ;; ret (Some (IBin g BAdd 64 d w (OReg x))))
            else if list_eqs sf ["local"; "u64"] then (w <- window_reg "local" ;; ret (Some (IBin g BAdd 64 d w (OReg x))))
            else if list_eqs sf ["to"; "shared"; "u64"] then (w <- window_reg "shared" ;; ret (Some (IBin g BSub 64 d x (OReg w))))
            else if list_eqs sf ["to"; "local"; "u64"] then (w <- window_reg "local" ;; ret (Some (IBin g BSub 64 d x (OReg w))))
            else ret None
        | _ => ret None
        end
    | _ => ret None
    end
  else if String.eqb mn "cvt" then
    match ops, sf with
    | [_; o1], [t1; t2] =>
        if bsu t1 && bsu t2 && defined (type_width t1) && defined (type_width t2) then
          (src <- operand o1 ;;
           ret (match src, type_width t1, type_width t2 with
                | Some (OReg x), Some to, Some from => Some (ICvt g to from (is_s t2) d x)
                | _, _, _ => None
                end))
        else ret None
    | _, _ => ret None
    end
  else integer g mn sf ops d.

(* Only these documented register-result instruction families can lose their
   value precision to IDef. This is an effect classification, not a corpus
   inventory. Synchronization/collective, memory, descriptor and control
   families require their own complete lowering and are absent here.
   NVIDIA PTX ISA: Integer Arithmetic; Extended-Precision Integer Arithmetic;
   Floating-Point; Comparison/Selection; Logic/Shift; Video; and the scalar
   Data Movement/Conversion sections. activemask only observes active lanes.
   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html *)
Definition pure_register_op (mn : string) : bool :=
 sin mn ["abs"; "add"; "addc"; "sub"; "subc"; "mul"; "mad"; "madc";
         "fma"; "neg"; "min"; "max"; "div"; "rem"; "sad"; "dp2a"; "dp4a";
         "and"; "or"; "xor"; "not"; "cnot"; "lop3"; "shl"; "shr"; "shf";
         "bfe"; "bfi"; "bfind"; "bmsk"; "brev"; "clz"; "popc"; "fns"; "prmt";
         "szext"; "set"; "setp"; "selp"; "slct"; "mov"; "cvt"; "cvta";
         "isspacep"; "activemask"; "rcp"; "sqrt"; "rsqrt"; "sin"; "cos";
         "lg2"; "ex2"; "tanh"; "testp"; "copysign";
         "vadd"; "vadd2"; "vadd4"; "vsub"; "vsub2"; "vsub4"; "vmad";
         "vavrg2"; "vavrg4"; "vabsdiff"; "vabsdiff2"; "vabsdiff4";
         "vmin"; "vmin2"; "vmin4"; "vmax"; "vmax2"; "vmax4";
         "vset"; "vset2"; "vset4"; "vshl"; "vshr"].

(* After the memory footprint has independently been checked, a load or
   atomic can define arbitrary result registers. Stores/reductions have no
   register result. No unknown mnemonic can use this helper. *)
Definition memory_results (g : guard) (mn : string) (ops : list popnd) : Tr (list instr) :=
 if sin mn ["st"; "red"] then ret [] else
 if sin mn ["ld"; "ldu"; "atom"] then
 match ops with
 | o :: _ => ns <- dest_names o ;;
             _ <- (match ns with [] => refuse "a memory instruction without register results" | _ => ret tt end) ;;
             _ <- all_declared ns ;;
             ds <- regs_of ns ;; ret [IDef g ds]
 | [] => refuse "a memory instruction without its result"
 end else refuse "unmodeled memory instruction effects".

(* The load of a parameter that holds an affine of the sizes: d gets the
   affine's value, computed from the size registers, which the launch check
   establishes to be the parameter's value. A narrower d gets its low bits,
   and a load of the parameter's low w bits, as Triton's ld.param.b32 of a
   64-bit capacity, gets those bits extended to d's width. *)
Definition size_param_load (g : guard) (mn : string) (sf : list string) (ops : list popnd) (d : reg)
    : Tr (option (list instr)) :=
  if negb (String.eqb mn "ld" && sin "param" sf) then ret None else
  match ops, types sf with
  | [_; PSym name off], [t] =>
      match param_of name, type_width t with
      | Some (PSize a), Some w =>
          p <- param name ;; wp <- width_of p ;; wd <- width_of d ;;
          if negb (Z.eqb off 0) || Nat.ltb wp w then ret None else
          v <- (if Nat.eqb wd 64 && Nat.eqb w wp then ret d else fresh 64%nat) ;;
          terms <- mapT (fun x => let '(k, c) := x in s <- size_reg k ;; ret (IMad g 64 v s (OImm c) (OReg v)))
                        (af_terms a) ;;
          ret (Some (IImm g v (af_const a) :: app terms
                       (if negb (Nat.eqb w wp) then [ICvt g wd w (is_s t) d v]
                        else if Nat.eqb wd 64 then [] else [ICvt g wd 64 false d v])))
      | _, _ => ret None
      end
  | _, _ => ret None
  end.

(* Each scalar destination receives one low-to-high field from the original
   source. Width checks keep the source distinct from every destination;
   intermediate extracts use fresh registers. Sinks never create a result. *)
Fixpoint unpack_fields (g : guard) (whole part : nat) (source : reg)
    (offset : nat) (fields : list (option reg)) : Tr (list instr) :=
  match fields with
  | [] => ret []
  | None :: rest => unpack_fields g whole part source (offset + part)%nat rest
  | Some d :: rest =>
      piece <- (if Nat.eqb offset 0 then ret [ICvt g part whole false d source]
                else scratch <- fresh whole ;;
                     ret [IBfe g whole scratch source offset part;
                          ICvt g part whole false d scratch]) ;;
      tail <- unpack_fields g whole part source (offset + part)%nat rest ;;
      ret (app piece tail)
  end.

Definition unpack_move (g : guard) (sf : list string) (ops : list popnd) : Tr (list instr) :=
  match sf, ops with
  | [ty], [PVec xs; PReg source_name] =>
    match type_width ty with
    | Some whole =>
      let count := List.length xs in
      let part := Nat.div whole count in
      if negb (is_b ty && (Nat.eqb whole 16 || Nat.eqb whole 32 || Nat.eqb whole 64) &&
               (Nat.eqb count 2 || Nat.eqb count 4) && Nat.leb 8 part &&
               Nat.eqb (part * count) whole) then
        refuse "an unpack move requires two or four equal-width byte-sized fields" else
      names <- mapT (fun x => match x with
                  | PReg r => ret (Some r)
                  | PText s => if String.eqb s "_" then ret None
                               else refuse "an unpack destination must be a register or sink"
                  | _ => refuse "an unpack destination must be a register or sink" end) xs ;;
      let written := flat_map (fun x => match x with Some r => [r] | None => [] end) names in
      if null written || negb (distinct written) then
        refuse "an unpack move requires distinct register destinations and at least one result" else
      _ <- all_declared (source_name :: written) ;;
      source <- reg_of source_name None ;; source_width <- width_of source ;;
      if negb (Nat.eqb source_width whole) then refuse "an unpack source has the wrong width" else
      fields <- mapT (fun x => match x with
                 | None => ret None
                 | Some name => d <- reg_of name None ;; width <- width_of d ;;
                     if Nat.eqb width part then ret (Some d)
                     else refuse "an unpack destination has the wrong width" end) names ;;
      unpack_fields g whole part source 0 fields
    | None => refuse "an unpack move has an unsupported bit width"
    end
  | _, _ => refuse "an unpack move requires one bit type and a scalar register source"
  end.

Definition generic (g : guard) (mn : string) (sf : list string) (ops : list popnd) : Tr (list instr) :=
  if negb (pure_register_op mn || parameter_load mn sf) then
    refuse ("unmodeled instruction effects: " ++ mn) else
  if String.eqb mn "mov" && (match ops with PVec _ :: _ => true | _ => false end)
  then unpack_move g sf ops else
  match ops with
  | [] => refuse "an instruction without modeled effects or a result"
  | o0 :: _ =>
      if is_addr o0 then refuse "an unmodeled memory effect" else
      (ns <- dest_names o0 ;;
       _ <- (match ns with [] => refuse "an instruction without register results" | _ => ret tt end) ;;
       _ <- all_declared ns ;;
       let defs := (if pure_register_op mn then
                      ds <- regs_of ns ;; ret [IDef g ds]
                    else refuse ("unmodeled instruction effects: " ++ mn)) in
       match single o0 with
       | Some r =>
           d <- reg_of r None ;;
           sz <- size_param_load g mn sf ops d ;;
           match sz with
           | Some xs => ret xs
           | None =>
           f <- signed_field g mn sf ops d ;;
           match f with
           | Some xs => ret xs
           | None =>
           b <- absolute g mn sf ops d ;;
           match b with
           | Some xs => ret xs
           | None => e <- exact g mn sf ops d ;;
                     match e with Some x => ret [x] | None => defs end
           end
           end
           end
       | None => defs
       end)
  end.

(* The address of a local access: a register and an offset, or a local
   variable and an offset, with the instructions that put the variable's
   address in a register. *)
Definition local_addr (o : popnd) : Tr (option (list instr * reg * Z)) :=
  match o with
  | PAddr (Some r) off =>
      d <- declared r ;;
      if negb d then refuse (r ++ " is not a declared register") else
      (a <- reg_of r None ;; ret (Some ([], a, off)))
  | PSym s off =>
      match local_name (PText s) with
      | Some v => x <- reg_of ("local:" ++ v) (Some 32%nat) ;; a <- fresh 64 ;;
                  ret (Some ([ICvt None 64 32 false a x], a, off))
      | None => ret None
      end
  | _ => ret None
  end.

(* A load or store of local memory, at the frame of the instruction: a store
   of one register or number writes its value, a vector store unknown
   values. Scalar loads retain tracked values; vector loads check their
   entire footprint and alignment before assigning arbitrary results. *)
Definition local_access (g : guard) (mn : string) (sf : list string) (ops : list popnd) : Tr (list instr) :=
  fr <- cur_frame ;;
  if String.eqb mn "st" then
    match ops with
    | [o; src] =>
        la <- local_addr o ;;
        match la, access_bytes mn sf ops with
        | Some (pre, a, off), Some w =>
            v <- (if vector sf then ret None else operand src) ;;
            ret (app pre [IStL g a off w v fr])
        | None, _ => refuse "a local store at an address the translation does not model"
        | _, None => refuse "a local store without a width"
        end
    | _ => refuse "a local store without an address and a value"
    end
  else if vector sf then
    match ops, types sf, filter vec_suffix sf with
    | [PVec xs; o], [t], [v] =>
      match type_width t, lanes v, access_bytes mn sf ops with
      | Some bits, Some count, Some bytes =>
        if negb (Z.eqb (Z.of_nat (List.length xs)) count) then refuse "a local vector load with the wrong result count" else
        ns <- mapT (fun x => match x with
                            | PReg r => ret r
                            | _ => refuse "a local vector load without register results"
                            end) xs ;;
        _ <- all_declared ns ;;
        ds <- regs_of ns ;;
        widths <- mapT width_of ds ;;
        if negb (forallb (fun w => Nat.leb bits w) widths) then
          refuse "a local vector load whose result registers are too narrow" else
        la <- local_addr o ;;
        match la with
        | Some (pre, a, off) =>
          (* One read covers the entire vector, including its alignment.
             The fresh value is unused; each actual result is arbitrary. *)
          scratch <- fresh 64 ;;
          ret (app pre [ILdL g (Z.to_nat (8 * bytes)) scratch a off fr; IDef g ds])
        | None => refuse "a local vector load at an address the translation does not model"
        end
      | _, _, _ => refuse "a local vector load without a supported width"
      end
    | _, _, _ => refuse "a local vector load without its register vector and address"
    end
  else
    match ops, types sf with
    | [o0; o], [t] =>
        la <- local_addr o ;;
        match la, single o0, type_width t with
        | Some (pre, a, off), Some dn, Some bits =>
            (dest <- reg_of dn None ;; wd <- width_of dest ;;
             if bu t || Nat.eqb wd bits then ret (app pre [ILdL g bits dest a off fr])
             (* a signed load narrower than its register is the load of its
                bits and their sign extension *)
             else (v <- fresh bits ;; ret (app pre [ILdL g bits v a off fr; ICvt g wd bits true dest v])))
        | _, _, _ => generic g mn sf ops
        end
    | _, _ => generic g mn sf ops
    end.

(* The entry of a call's frame, which the inlining of the call marks with
   the pseudo-instruction frame$ and the call's local variables. *)
Definition frame_entry (ops : list popnd) : Tr (list instr) :=
  vs <- mapT (fun o => match local_name o with
                       | Some v => reg_of ("local:" ++ v) (Some 32%nat)
                       | None => refuse "a frame entry of a name that is not a local variable"
                       end) ops ;;
  ret [IFrame vs].

(* What must not happen under guard g: behavior that PTX leaves undefined,
   or a trap or breakpoint, which ends the launch with an error for every
   tenant of the context. It becomes a one-byte store to address 0 under g.
   No region contains address 0, so the checker accepts the kernel only if
   it proves that g never holds there. *)
Definition never_under (g : guard) : Tr (list instr) :=
  z <- fresh 64 ;; fr <- cur_frame ;; ret [IImm None z 0; IMem g KStore false z 0 1 None fr []].

Definition undefined_if (q : reg) : Tr (list instr) := never_under (Some (false, q)).

(* Undefined unless a <c> b holds, unsigned, for the w-bit register a. *)
Definition require (w : nat) (c : cmp) (a : reg) (b : Z) : Tr (list instr) :=
  q <- fresh 1 ;; u <- undefined_if q ;;
  let neg := match c with CLt => CGe | CGe => CLt | CLe => CGt | CGt => CLe | CEq => CNe | CNe => CEq end in
  ret (ISetp None neg w false q a (OImm b) :: u).

(* Undefined unless the w-bit register a is a multiple of m, a power of
   two. *)
Definition require_multiple (w : nat) (a : reg) (m : Z) : Tr (list instr) :=
  if Z.leb m 1 then ret [] else
  u <- fresh w ;; q <- require w CEq u 0 ;; ret (IBin None BAnd w u a (OImm (m - 1)) :: q).

(* Undefined unless the address that operand o names is a multiple of m, a
   power of two, as PTX requires of the instruction's address. *)
Definition require_aligned (o : popnd) (m : Z) : Tr (list instr) :=
  match o with
  | PAddr (Some r) off =>
      d <- declared r ;;
      if negb d then refuse (r ++ " is not a declared register") else
      a <- reg_of r None ;; t <- fresh 64 ;; x <- require_multiple 64 t m ;;
      ret (IBin None BAdd 64 t a (OImm off) :: x)
  | _ => refuse "address is not a register and an offset"
  end.

(* Full-mask .sync shuffle/vote/match and integer redux on supported targets.
   NVIDIA PTX distinguishes an undefined shuffle RESULT from an undefined
   executing-thread behavior. The full mask includes every executing lane;
   sm_70+ removes the older target's convergence precondition. Waiting is
   erased only as a spatial-safety overapproximation, with no progress or
   transported-value guarantee. These are not generic pure-register ops. *)
Definition sync_target_ok : bool :=
 match k_target K, k_ptx_version K with
 | Some target, Some (major, minor) =>
   Z.leb 70 target && Z.leb 6 major && Z.leb 0 minor &&
   (Z.ltb major 9 || (Z.eqb major 9 && Z.leb minor 4))
 | _, _ => false
 end.

Definition sync_register (bits : nat) (name : string) : Tr reg :=
 match decls name with
 | [(_, width, _)] =>
   if Nat.eqb width bits then
     r <- reg_of name None ;; actual <- width_of r ;;
     if Nat.eqb actual bits then ret r else refuse "a collective register width disagrees"
   else refuse "a collective register has the wrong width"
 | _ => refuse "a collective operand is not one declared register"
 end.

(* Exact literal fractions. Decimal constants first have binary64 semantics
   in PTX. We only accept decimal values that are exactly representable as
   binary32 too, so no rounding or double-rounding assumption is needed.
   Canonical decimal m / 10^k is dyadic exactly when 5^k divides m. After
   removing decimal trailing zeroes, divisibility by 5^k makes the quotient
   odd for k>0. Thus it has at most 24 significant binary digits precisely
   when it is below 2^24. For k=0, the <=1 bound forces quotient=1. k <= 149
   includes the positive binary32 subnormals; quotient <= 2^k bounds it by
   one. Other floating literals remain available for arbitrary results. *)
Definition literal_digit (a : ascii) : option Z :=
 let n := Z.of_nat (nat_of_ascii a) in
 if Z.leb 48 n && Z.leb n 57 then Some (n-48) else
 if Z.leb 65 n && Z.leb n 70 then Some (n-55) else
 if Z.leb 97 n && Z.leb n 102 then Some (n-87) else None.

Fixpoint bounded_digits (base limit acc : Z) (xs : list ascii) : option Z :=
 match xs with
 | [] => Some acc
 | a :: rest =>
   match literal_digit a with
   | Some d => let next := base*acc+d in
       if Z.ltb d base && Z.leb next limit then bounded_digits base limit next rest else None
   | None => None
   end
 end.
Definition literal_unsigned (base limit : Z) (xs : list ascii) : option Z :=
 match xs with [] => None | _ => bounded_digits base limit 0 xs end.

Definition literal_exponent (limit : Z) (xs : list ascii) : option Z :=
 let '(negative,xs) := match xs with
   | "+"%char :: rest => (false,rest)
   | "-"%char :: rest => (true,rest)
   | xs => (false,xs)
   end in
 option_map (fun n => if negative then -n else n) (literal_unsigned 10 limit xs).

Fixpoint skip_decimal_zeroes (xs : list ascii) : list ascii :=
 match xs with
 | "0"%char :: rest => skip_decimal_zeroes rest
 | _ => xs
 end.

Fixpoint divide_fives (steps : nat) (n : Z) : option Z :=
 match steps with
 | O => Some n
 | S rest => if Z.eqb (Z.rem n 5) 0 then divide_fives rest (Z.quot n 5) else None
 end.

Lemma divide_fives_exact : forall steps n q,
 divide_fives steps n = Some q -> n = q * 5^(Z.of_nat steps).
Proof.
 induction steps as [|steps IH]; intros n q H; cbn [divide_fives] in H.
 - injection H as <-. cbn. lia.
 - destruct (Z.eqb (Z.rem n 5) 0) eqn:Er; [|discriminate].
   apply Z.eqb_eq in Er. specialize (IH _ _ H).
   pose proof (Z.quot_rem n 5 ltac:(lia)) as Hdivide.
   rewrite Nat2Z.inj_succ, Z.pow_succ_r by lia. nia.
Qed.

(* Linear splitting avoids repeated string appends for long literals. *)
Fixpoint literal_split_go (separator : ascii) (xs acc : list ascii)
    (parts : list (list ascii)) : list (list ascii) :=
 match xs with
 | [] => rev (rev acc :: parts)
 | a :: rest => if Ascii.eqb a separator then literal_split_go separator rest [] (rev acc :: parts)
                else literal_split_go separator rest (a :: acc) parts
 end.
Definition literal_split (separator : ascii) (xs : list ascii) : list (list ascii) :=
 literal_split_go separator xs [] [].

Definition decimal_fraction (s : string) : bool :=
 let s := match s with String "+"%char rest => rest | _ => s end in
 let normal := map (fun a => if Ascii.eqb a "E"%char then "e"%char else a) (list_ascii_of_string s) in
 let parts := match literal_split "e"%char normal with
   | [mantissa] => Some (mantissa,0)
   | [mantissa; exponent] =>
     (* A larger exponent cannot be cancelled by the available digits into
        the supported binary32 range. The bound avoids huge arithmetic. *)
     option_map (fun e => (mantissa,e))
       (literal_exponent (Z.of_nat (String.length s)+149) exponent)
   | _ => None
   end in
 match parts with
 | None => false
 | Some (mantissa,exponent) =>
   let digits := match literal_split "."%char mantissa with
     | [whole] => Some (whole,0)
     | [whole; fraction] => Some (List.app whole fraction,Z.of_nat (List.length fraction))
     | _ => None
     end in
   match digits with
   | None => false
   | Some (digits,places) =>
     let digits := skip_decimal_zeroes digits in
     let short := rev (skip_decimal_zeroes (rev digits)) in
     let removed := Z.of_nat (List.length digits-List.length short) in
     let scale := places-exponent-removed in
     (* A canonical positive fraction <=1 with scale<=149 has at most150
        decimal digits; this bounds parsing work without a numeric guess. *)
     if negb (Nat.leb (List.length short) 150 && Z.leb 0 scale && Z.leb scale 149) then false else
     match literal_unsigned 10 (10^150) short with
     | None => false
     | Some numerator =>
       match divide_fives (Z.to_nat scale) numerator with
       | None => false
       | Some q => Z.ltb 0 q && Z.ltb q (2^24) && Z.leb q (2^scale)
       end
     end
   end
 end.

Definition fraction_literal (s : string) : bool :=
 if String.prefix "0f" s || String.prefix "0F" s then
   if Nat.eqb (String.length s) 10 then
     match literal_unsigned 16 (2^32-1) (list_ascii_of_string (drop 2 s)) with
     | Some bits => Z.ltb 0 bits && Z.leb bits 1065353216
     | None => false
     end
   else false
 else if String.prefix "0d" s || String.prefix "0D" s then false
 else decimal_fraction s.

(* This has only an opaque register result. Fractional policy creation
   names no address and changes no memory. Range/cvt policies are separate
   effect classes and are not covered by this rule. *)
Definition fractional_policy (g : guard) (sf : list string) (ops : list popnd)
    : Tr (list instr) :=
 let target_ok := match k_target K,k_ptx_version K with
   | Some target,Some (major,minor) =>
     Z.leb 80 target && (Z.ltb 7 major || (Z.eqb major 7 && Z.leb 4 minor)) &&
     (Z.ltb major 9 || (Z.eqb major 9 && Z.leb minor 4)) && Z.leb 0 minor
   | _,_ => false
   end in
 if negb target_ok then refuse "a fractional cache policy requires explicit PTX 7.4 through 9.4 and target sm_80+" else
 let priorities := match sf with
   | ["fractional";primary;"b64"] => Some (primary,"L2::evict_unchanged")
   | ["fractional";primary;secondary;"b64"] => Some (primary,secondary)
   | _ => None
   end in
 match priorities with
 | None => refuse "an unsupported fractional cache-policy shape"
 | Some (primary,secondary) =>
   if negb (sin primary ["L2::evict_last";"L2::evict_normal";"L2::evict_first";"L2::evict_unchanged"] &&
            sin secondary ["L2::evict_first";"L2::evict_unchanged"])
   then refuse "an unsupported fractional cache-policy priority" else
   let destination := match ops with
     | [PReg d] => Some d
     | [PReg d;PFloat literal] => if fraction_literal literal then Some d else None
     | _ => None
     end in
   match destination with
   | None => refuse "a cache fraction is not an exactly represented binary32 value in (0,1]"
   | Some d =>
     _ <- (match g with None => ret tt | Some (_,p) =>
             bits <- width_of p ;; if Nat.eqb bits 1 then ret tt else refuse "a cache-policy guard is not a predicate" end) ;;
     r <- sync_register 64 d ;; ret [IDef g [r]]
   end
 end.

Definition sync_word (o : popnd) : Tr unit :=
 match o with
 | PReg r => _ <- sync_register 32 r ;; ret tt
 | PImm z => if Z.leb (-2147483648) z && Z.leb z 4294967295
                    then ret tt else refuse "a collective word literal is out of range"
 | _ => refuse "an unsupported collective word operand"
 end.

Definition sync_predicate (o : popnd) : Tr unit :=
 match o with
 | PReg r | PNot r => _ <- sync_register 1 r ;; ret tt
 | _ => refuse "a collective source is not a predicate"
 end.

Definition sync_mask (o : popnd) : bool :=
 match o with PImm z => Z.eqb z 4294967295 || Z.eqb z (-1) | _ => false end.

(* A register mask is never accepted from its width alone. Its full-mask
   equality is an independent checker obligation. This obligation is
   deliberately unconditional, so a predicated-off collective may still
   refuse when its mask cannot be established. *)
Definition sync_mask_requirement (o : popnd) : Tr (list instr) :=
 match o with
 | PReg r => a <- sync_register 32 r ;; require 32 CEq a 4294967295
 | _ => if sync_mask o then ret [] else refuse "a collective requires a full 32-bit mask"
 end.

Definition sync_shuffle_dest (o : popnd) : Tr (list reg) :=
 match o with
 | PReg d => r <- sync_register 32 d ;; ret [r]
 | PText names =>
   match split_on "|"%char names "" with
   | [d; p] => r <- sync_register 32 d ;; q <- sync_register 1 p ;; ret [r; q]
   | _ => refuse "a shuffle destination is not a register or register-predicate pair"
   end
 | _ => refuse "an unsupported shuffle destination"
 end.

Definition sync_collective (g : guard) (mn : string) (sf : list string) (ops : list popnd)
    : Tr (list instr) :=
 if negb sync_target_ok then refuse "a collective requires explicit PTX 6.0 through 9.4 and target sm_70+" else
 _ <- (match g with
       | None => ret tt
       | Some (_, p) => bits <- width_of p ;;
                         if Nat.eqb bits 1 then ret tt else refuse "a collective guard is not a predicate"
       end) ;;
 if String.eqb mn "shfl" then
   match sf, ops with
   | ["sync"; mode; "b32"], [dst; PReg src; b; c; mask] =>
     if sin mode ["up"; "down"; "bfly"; "idx"] then
       checks <- sync_mask_requirement mask ;;
       ds <- sync_shuffle_dest dst ;;
       _ <- sync_register 32 src ;; _ <- sync_word b ;; _ <- sync_word c ;;
       ret (app checks [IDef g ds])
     else refuse "a shuffle requires a recognized mode"
   | _, _ => refuse "an unsupported synchronous shuffle shape"
   end
 else if String.eqb mn "vote" then
   match sf, ops with
   | ["sync"; mode; ty], [PReg dst; src; mask] =>
     let width := if String.eqb mode "ballot" && String.eqb ty "b32" then Some 32%nat
                  else if sin mode ["all"; "any"; "uni"] && String.eqb ty "pred" then Some 1%nat
                  else None in
     match width with
     | Some bits => checks <- sync_mask_requirement mask ;;
         d <- sync_register bits dst ;; _ <- sync_predicate src ;; ret (app checks [IDef g [d]])
     | None => refuse "an unsupported synchronous vote type or mode"
     end
   | _, _ => refuse "an unsupported synchronous vote shape"
   end
 else if String.eqb mn "match" then
   match sf, ops with
   | [mode; "sync"; ty], [dst; PReg src; mask] =>
     let bits := if String.eqb ty "b32" then Some 32%nat
                 else if String.eqb ty "b64" then Some 64%nat else None in
     match bits with
     | Some w =>
       checks <- sync_mask_requirement mask ;;
       _ <- sync_register w src ;;
       ds <- (if String.eqb mode "all" then sync_shuffle_dest dst
              else if String.eqb mode "any" then
                match dst with PReg d => r <- sync_register 32 d ;; ret [r]
                | _ => refuse "a match.any destination must be one 32-bit register" end
              else refuse "an unsupported synchronous match mode") ;;
       ret (app checks [IDef g ds])
     | None => refuse "an unsupported synchronous match type"
     end
   | _, _ => refuse "an unsupported synchronous match shape"
   end
 else if String.eqb mn "redux" then
   (* Integer reductions were introduced in PTX 7.0 on sm_80. Their
      arbitrary 32-bit result is sufficient only for spatial safety; no
      reduction bound or relation to another lane is asserted. *)
   if negb (match k_target K, k_ptx_version K with
            | Some target, Some (major, _) => Z.leb 80 target && Z.leb 7 major
            | _, _ => false end)
   then refuse "an integer reduction requires PTX 7.0+ and target sm_80+" else
   match sf, ops with
   | ["sync"; mode; ty], [PReg dst; PReg src; mask] =>
     if (sin mode ["add"; "min"; "max"] && sin ty ["s32"; "u32"]) ||
        (sin mode ["and"; "or"; "xor"] && String.eqb ty "b32") then
       checks <- sync_mask_requirement mask ;;
       d <- sync_register 32 dst ;; _ <- sync_register 32 src ;;
       ret (app checks [IDef g [d]])
     else refuse "an unsupported synchronous integer reduction type or mode"
   | _, _ => refuse "an unsupported synchronous integer reduction shape"
   end
 else refuse "an unsupported synchronous collective".

(* A shared memory instruction: its access, which the check must place
   inside a shared variable, since an access past the block's shared memory
   faults, its kind, the value of a store of one register or number, and
   its destinations. A scalar load is a shared load, whose check is its
   access's. *)
Definition shared_access (g : guard) (mn : string) (sf : list string) (ops : list popnd) : Tr (list instr) :=
  if negb (existsb brackets ops) then generic g mn sf ops else
  l <- shared_load g mn sf ops ;;
  match l with
  | Some xs => ret xs
  | None =>
      match address_operand mn sf ops, access_bytes mn sf ops with
      | Some o, Some w =>
          let k := if loads_mem mn sf && stores_mem mn sf then KRmw
                   else if stores_mem mn sf then KStore else KLoad in
          sv <- store_value mn sf ops ;;
          x <- shared_at g k o w sv ;; ds <- memory_results g mn ops ;; ret (x :: ds)
      | _, _ => refuse "a shared memory access whose address or width the translation does not model"
      end
  end.

Definition branch_list (ops : list popnd) : option (list string) :=
  match ops with
  | [_; PLabel t] => option_map snd (find (fun e => String.eqb (fst e) t) (k_branch_targets K))
  | _ => None
  end.

(* brx.idx r, list in block b: a comparison of r and a branch for each label
   of the list but the last, and a branch to the last, after the requirement
   that r is below the list's length, where PTX leaves the target undefined.
   A guarded one first branches to the next block where its guard fails. *)
Definition branch_indexed (labels : PM.t nat) (b : nat) (g : guard) (ops : list popnd) : Tr (list instr) :=
  match branch_list ops, ops with
  | Some ls, o :: _ =>
      x <- operand o ;;
      match x with
      | Some (OReg r) =>
          wr <- width_of r ;;
          if negb (Nat.eqb wr 32) then refuse "a branch index that is not a 32-bit register" else
          blks <- mapT (fun l => match PM.find (key l) labels with
                                 | Some n => ret n
                                 | None => refuse ("branch target " ++ l ++ " is not a label")
                                 end) ls ;;
          match rev blks with
          | [] => refuse "an empty list of branch targets"
          | last :: _ =>
              below <- require 32 CLt r (Z.of_nat (List.length blks)) ;;
              tests <- mapT (fun kb => let '(k, blk) := kb in
                               p <- fresh 1 ;;
                               ret [ISetp None CEq 32 false p r (OImm (Z.of_nat k)); IBra (Some (false, p)) blk])
                            (combine (seq 0 (List.length blks)) (removelast blks)) ;;
              let skip := match g with None => [] | Some (neg, p) => [IBra (Some (negb neg, p)) (S b)] end in
              ret (app skip (app below (app (List.concat tests) [IBra None last])))
          end
      | _ => refuse "a branch index that is not a register"
      end
  | _, _ => refuse "a branch through a register without its .branchtargets list"
  end.

(* These operand-free fences only constrain memory order. The per-thread
   model already permits every shared-load value and makes no progress claim.
   Resource/proxy fences with additional operands or prerequisites require
   a separate model rather than inheriting this classification. *)
Definition ordering_fence (mn : string) (sf : list string) (ops : list popnd) : bool :=
 match ops with
 | [] =>
   if String.eqb mn "membar" then
     match sf with [scope] => sin scope ["cta"; "gl"; "sys"] | _ => false end
   else if String.eqb mn "fence" then
     match sf with
     | [scope] => sin scope ["cta"; "cluster"; "gpu"; "sys"]
     | [sem; scope] => sin sem ["sc"; "acq_rel"; "acquire"; "release"] &&
                       sin scope ["cta"; "cluster"; "gpu"; "sys"]
     | _ => false
     end
   else false
 | _ => false
 end.

(* A bar.sync is an aligned CTA barrier. Absence of @p alone does not
   establish participation. The complete frontend checks either a finite
   straight-line event order or the single-event progress certificate below.
   Syntax checking here is only the first step; front_core validates the
   resulting control flow before returning the lowered program. Neither
   mechanism infers values communicated through shared memory. *)
Definition simple_barrier (i : pinstr) : bool :=
 match i with
 | PI None "bar" ["sync"] [PImm id] => Z.leb 0 id && Z.ltb id 16
 | _ => false
 end.

Fixpoint linear_barrier_body (xs : list pinstr) : bool :=
 match xs with
 | [PI None "ret" [] []] => true
 | [] => false
 | (PI _ mn _ _ as i) :: rest =>
   if sin mn ["bra"; "brx"; "call"; "ret"; "exit"; "trap"; "brkpt"] then false else
   if String.eqb mn "bar" then simple_barrier i && linear_barrier_body rest
   else if sin mn ["barrier"; "mbarrier"] then false
   else linear_barrier_body rest
 end.

Definition checked_simple_barrier (i : pinstr) : Tr (list instr) :=
 if sync_target_ok && simple_barrier i
 then ret [INop]
 else refuse "an aligned barrier requires a supported target and exact simple form".

(* The instructions that model one PTX instruction, the first at index j of
   block b, which a shared memory load takes as its id. *)
Definition translate_inst (labels : PM.t nat) (b : nat) (i : pinstr) : Tr (list instr) :=
  let '(PI p mn sf ops) := i in
  g <- guard_reg p ;;
  if String.eqb mn "bra" then
    match ops with
    | [PLabel t] => match PM.find (key t) labels with
                    | Some n => ret [IBra g n]
                    | None => refuse "branch without a label target"
                    end
    | _ => refuse "branch without a label target"
    end
  else if sin mn ["ret"; "exit"] then ret [IExit g]
  else if String.eqb mn "trap" then (xs <- never_under g ;; ret (app xs [IExit g]))
  else if String.eqb mn "brkpt" then never_under g
  else if String.eqb mn "brx" then branch_indexed labels b g ops
  else if String.eqb mn "call" then refuse "a call"
  (* a frame whose address changes while its call runs, which the local
     memory model does not have *)
  else if sin mn ["alloca"; "stacksave"; "stackrestore"] then refuse "a dynamic stack allocation"
  else if String.eqb mn "createpolicy" then fractional_policy g sf ops
  else if String.eqb mn "bar" then checked_simple_barrier i
  else if sin mn ["shfl"; "vote"; "match"; "redux"] then sync_collective g mn sf ops
  else if ordering_fence mn sf ops then ret [INop]
  else if sin mn ["fence"; "membar"] then refuse "a fence with unmodeled prerequisites"
  else if String.eqb mn "wmma" then refuse "warp-collective participation is not modeled"
  else if String.eqb mn "cp" then
    refuse "asynchronous-copy effects and prerequisites require a complete model"
  (* a weak write of indeterminate values to [a, a + size), at an address
     that PTX requires to be a multiple of 128 *)
  else if String.eqb mn "discard" then
    match ops with
    | [o; PImm z] => al <- require_aligned o 128 ;; x <- mem_at g KStore o z [] ;; ret (app al [x])
    | _ => refuse "a discard without an address and a size"
    end
  (* A tensor-map update/copy has memory and collective preconditions that
     the current lowering does not completely establish. *)
  else if String.eqb mn "tensormap" then refuse "tensor-map memory effects are not modeled"
  else if sin mn ["txq"; "suq"; "tex"; "tld4"; "suld"; "sust"; "sured"] then
    refuse "a texture or surface resource effect"
  else if String.eqb mn "frame$" then frame_entry ops
  else if sin "async" sf then
    refuse "asynchronous instruction effects are not modeled"
  else if sin mn ["ld"; "st"] && sin "local" sf then local_access g mn sf ops
  else
    if global_op mn sf then global_access g mn sf ops
    else if existsb is_shared sf then shared_access g mn sf ops
    (* any other access, such as a multimem access of the memory of several
       GPUs, whose memory effect the instruction set does not model *)
    else if existsb brackets ops && negb (parameter_load mn sf)
    then refuse "memory instruction the translation does not model"
    else generic g mn sf ops.

(* Blocks: a block ends at a jump or before a branch target. *)

(* Instruction indices are binary, N: a unary nat term of a large index is
   as large as the index. *)
Definition nkey (n : N) : positive := N.succ_pos n.

Definition is_jump_op (i : pinstr) : bool :=
  let '(PI _ mn _ _) := i in sin mn ["bra"; "brx"; "ret"; "exit"; "trap"].

Definition targets : PM.t unit :=
  fold_left (fun m i => let '(PI _ mn _ ops) := i in
                        if String.eqb mn "bra"
                        then fold_left (fun m o => match o with PLabel t => PM.add (key t) tt m | _ => m end) ops m
                        else if String.eqb mn "brx"
                        then match branch_list ops with
                             | Some ls => fold_left (fun m t => PM.add (key t) tt m) ls m
                             | None => m
                             end
                        else m) (k_insts K) (PM.empty unit).

(* An instruction's place: its origin, the instruction of the kernel as the
   parser read it that it comes from, and the frame of the call it belongs
   to. *)
Definition place := (N * frame)%type.

(* Each instruction with its index, its place, and whether a block starts
   there. *)
Fixpoint mark (tp : PM.t unit) (n : N) (after_jump : bool) (l : list (place * pinstr))
    : list (N * place * pinstr * bool) :=
  match l with
  | [] => []
  | (o, i) :: t => (n, o, i, N.eqb n 0 || after_jump || defined (PM.find (nkey n) tp))
                     :: mark tp (N.succ n) (is_jump_op i) t
  end.

Definition group (l : list (N * place * pinstr * bool)) : list (list (N * place * pinstr)) :=
  rev (map (@rev (N * place * pinstr))
           (fold_left (fun (acc : list (list (N * place * pinstr))) (x : N * place * pinstr * bool) =>
    let '(n, o, i, lead) := x in
    if lead then [(n, o, i)] :: acc
    else match acc with blk :: rest => ((n, o, i) :: blk) :: rest | [] => [[(n, o, i)]] end) l [])).

(* The blocks, of instructions with their index and place. *)
Definition blocks (tg : PM.t unit) (org : list place) : list (list (N * place * pinstr)) :=
  let tp := fold_left (fun m x => let '(name, n) := x in
                                  if defined (PM.find (key name) tg) then PM.add (nkey n) tt m else m)
                      (k_labels K) (PM.empty unit) in
  group (mark tp 0 false (combine org (k_insts K))).

(* The block of each branch target; a label past the last instruction names
   no block. A target defined twice is refused. *)
Definition label_blocks (tg : PM.t unit) (bs : list (list (N * place * pinstr))) : Tr (PM.t nat) :=
  let starts := snd (fold_left (fun acc blk => let '(b, m) := acc in
                         (S b, match blk with (n, _, _) :: _ => PM.add (nkey n) b m | [] => m end))
                       bs (0%nat, PM.empty nat)) in
  let past := List.length bs in
  fold_left (fun acc x => let '(name, n) := x in
               m <- acc ;;
               if defined (PM.find (key name) tg) then
                 if defined (PM.find (key name) m) then refuse ("label " ++ name ++ " is defined twice")
                 else ret (PM.add (key name) (match PM.find (nkey n) starts with
                                              | Some b => b
                                              | None => past
                                              end) m)
               else ret m) (k_labels K) (ret (PM.empty nat)).

(* A lowered instruction tagged with the unique index of its expanded PTX
   source occurrence. Raw caller origins remain in the inlining place list
   and are restored for diagnostics only. *)
Definition placed := (instr * N)%type.

Definition translate_block (labels : PM.t nat) (b : nat) (blk : list (N * place * pinstr)) : Tr (list placed) :=
  r <- fold_left (fun acc x => let '(occurrence, (org, fr), i) := x in
                   out <- acc ;;
                   _ <- set_at (Some org) fr ;;
                   xs <- translate_inst labels b i ;;
                   ret (rev_append (map (fun y => (y, occurrence)) xs) out)) blk (ret []) ;;
  ret (rev r).

Fixpoint translate_blocks (labels : PM.t nat) (b : nat) (bs : list (list (N * place * pinstr)))
    : Tr (list (list placed)) :=
  match bs with
  | [] => ret []
  | blk :: t => x <- translate_block labels b blk ;; xs <- translate_blocks labels (S b) t ;; ret (x :: xs)
  end.

(* A block cut after each jump: a PTX instruction that becomes several
   branches continues in a new block after each. *)
Fixpoint cut_at_jumps (body cur : list placed) : list (list placed) :=
  match body with
  | [] => match cur with [] => [] | _ => [rev cur] end
  | i :: t => if is_jump (fst i) then rev (i :: cur) :: cut_at_jumps t [] else cut_at_jumps t (i :: cur)
  end.

(* The blocks cut after each jump, with branch targets renumbered to the new
   blocks. A branch to a block goes to the first of its pieces. *)
Definition split_blocks (prog : list (list placed)) : list (list placed) :=
  let pieces := map (fun body => cut_at_jumps body []) prog in
  let starts := rev (snd (fold_left (fun (acc : nat * list nat) ps =>
                                       let '(n, l) := acc in ((n + List.length ps)%nat, n :: l))
                                    pieces (0%nat, []))) in
  let total := fold_left (fun n ps => (n + List.length ps)%nat) pieces 0%nat in
  map (fun body => map (fun '(i, o) => (match i with
                                        | IBra g t => IBra g (nth t starts total)
                                        | _ => i
                                        end, o)) body)
      (List.concat pieces).

(* Retain the values used by comparisons when the block later overwrites
   their operands. These same-width copies use fresh, unnamed registers;
   they do not change a PTX register, guard, branch, or memory access.
   Annotating suffix writes once avoids rescanning long basic blocks. *)
Definition comparison_writes := instruction_outputs.

Definition with_future_writes (body : list placed) : list (placed * PM.t unit) :=
  fst (fold_left (fun acc p => let '(out, written) := acc in
       ((p, written) :: out,
        fold_left (fun m r => PM.add r tt m) (comparison_writes (fst p)) written))
       (rev body) ([], PM.empty unit)).

Definition snapshot_operand (written : PM.t unit) (org : N) (r : reg)
    : Tr (list placed * reg) :=
  match PM.find r written with
  | None => ret ([], r)
  | Some _ => w <- width_of r ;; tmp <- fresh w ;;
              ret ([(IMov None tmp r, org)], tmp)
  end.

Definition snapshot_comparison (x : placed * PM.t unit) : Tr (list placed) :=
  let '((i, org), written) := x in
  match i with
  | ISetp g c w signed d a b =>
      left <- snapshot_operand written org a ;;
      let '(pre, a') := left in
      right <- (match b with
                | OReg r => copied <- snapshot_operand written org r ;;
                            ret (fst copied, OReg (snd copied))
                | OImm _ => ret ([], b)
                end) ;;
      ret (List.app pre (List.app (fst right) [(ISetp g c w signed d a' (snd right), org)]))
  | _ => ret [(i, org)]
  end.

Definition snapshot_block (body : list placed) : Tr (list placed) :=
  parts <- mapT snapshot_comparison (with_future_writes body) ;;
  ret (List.concat parts).

(* The kernel's lowered instructions carry unique expanded-source positions.
   org retains each expanded instruction's raw diagnostic origin and frame. *)
Definition translate_placed (org : list place) : Tr (list (list placed)) :=
  (* the instruction set's addresses are 64-bit *)
  if negb (Z.eqb (k_address_size K) 64) then refuse "the module's addresses are not 64-bit" else
  let tg := targets in
  let bs := blocks tg org in
  (labels <- label_blocks tg bs ;; p <- translate_blocks labels 0 bs ;; _ <- set_at None [] ;;
   mapT snapshot_block (split_blocks p)).

Definition translate (org : list place) : Tr program := p <- translate_placed org ;; ret (map (map fst) p).

(* ------------------------------------------------------------------ *)
(* Launch facts.                                                       *)

(* The direction and extent of buffer parameter name. *)
Definition zmin (l : list Z) (default : Z) : Z :=
  match l with [] => default | x :: t => fold_left Z.min t x end.

(* An affine bound as a form over the size registers, with one constant. *)
Definition affine_form (a : affine) : Tr form :=
  ts <- mapT (fun x => let '(k, c) := x in r <- size_reg k ;; ret ((r, EX), c)) (af_terms a) ;;
  ret (mk 64 ts (af_const a) (af_const a)).

Definition index_of (ir : ixrange) : Tr index :=
  if negb (Z.eqb (ir_elem ir) 1 || Z.eqb (ir_elem ir) 2 || Z.eqb (ir_elem ir) 4 || Z.eqb (ir_elem ir) 8)
  then refuse "an index range whose elements are not 1, 2, 4, or 8 bytes" else
  lo <- affine_form (ir_lo ir) ;; hi <- affine_form (ir_hi ir) ;;
  ret {| ix_elem := ir_elem ir; ix_signed := ir_signed ir; ix_lo := lo; ix_hi := hi |}.

(* Whether a register of w bits can hold every value in [lo, hi]: a launch
   fact outside it would be false, and a checker that assumes a false fact
   proves anything. *)
Definition fits (lo hi : Z) (w : nat) : bool := Z.leb 0 lo && Z.leb lo hi && Z.ltb hi (2 ^ Z.of_nat w).

(* The facts the binding declares, which the runtime establishes at launch. *)
Definition facts : Tr lfacts :=
  (* each parameter holds one thing *)
  if negb (distinct (map fst (b_params Mf))) then refuse "the binding names a parameter twice" else
  (* the executor launches every block with the binding's size, and a grid of
     at most the binding's maxima, which lie within the limit of every GPU *)
  let '(bx, by_, bz) := b_block Mf in
  if match k_reqntid K with
     | Some (x, y, z) => negb (Z.eqb x bx && Z.eqb y by_ && Z.eqb z bz)
     | None => false
     end
  then refuse "the binding's block size is not the kernel's .reqntid" else
  let '(gx, gy, gz) := b_grid Mf in
  let '(lx, ly, lz) := grid_limit in
  if negb (Z.leb 1 gx && Z.leb gx lx && Z.leb 1 gy && Z.leb gy ly && Z.leb 1 gz && Z.leb gz lz)
  then refuse "the binding's grid maxima lie outside [1, the GPU's limit]" else
  tx <- reg_of "%tid.x" (Some 32%nat) ;; ty <- reg_of "%tid.y" (Some 32%nat) ;;
  tz <- reg_of "%tid.z" (Some 32%nat) ;;
  nx <- reg_of "%ntid.x" (Some 32%nat) ;; ny <- reg_of "%ntid.y" (Some 32%nat) ;;
  nz <- reg_of "%ntid.z" (Some 32%nat) ;;
  cx <- reg_of "%ctaid.x" (Some 32%nat) ;; cy <- reg_of "%ctaid.y" (Some 32%nat) ;;
  cz <- reg_of "%ctaid.z" (Some 32%nat) ;;
  mx <- reg_of "%nctaid.x" (Some 32%nat) ;; my <- reg_of "%nctaid.y" (Some 32%nat) ;;
  mz <- reg_of "%nctaid.z" (Some 32%nat) ;;
  lane <- reg_of "%laneid" (Some 32%nat) ;;
  (* every size interval is nonnegative and below the global size limit *)
  sizes <- mapT (fun x => let '(k, (m, h)) := x in
                  if negb (Z.leb 0 m && Z.leb m h && Z.leb h size_max) then refuse "size bounds outside [0, size_max]" else
                  r <- size_reg (Z.of_nat k) ;; ret (r, (m, h)))
                (combine (seq 0 (List.length (b_sizes Mf))) (b_sizes Mf)) ;;
  derived <- mapT (fun d =>
    if negb (Z.ltb 0 (ds_divisor d)) then refuse "a derived size divisor is not positive" else
    f <- affine_form (derived_affine d) ;;
    ret (ft f, - flo f, ds_divisor d - 1 - flo f)) (b_derived Mf) ;;
  (* what each parameter holds: bounds for a constant, a bounded integer, or
     zero; a span for a buffer's base; nothing for a size, which the
     translation reads as its size register *)
  decls <- mapT (fun x => let '(name, pm) := x in
                  match pm with
                  | PBase sp =>
                      p <- param name ;; ext <- affine_form (bs_ext sp) ;;
                      ix <- (match bs_index sp with Some ir => i <- index_of ir ;; ret (Some i) | None => ret None end) ;;
                      ret ([], [(p, {| sp_dir := bs_dir sp; sp_ext := ext; sp_index := ix; sp_flag := bs_flag sp |})],
                           [(p, bs_align sp)])
                  | PSize a => _ <- affine_form a ;; ret ([], [], [])
                  | PConst z =>
                      p <- param name ;; w <- width_of p ;;
                      if negb (fits z z w) then refuse ("parameter " ++ name ++ " holds a constant its register cannot hold")
                      else ret ([(p, (z, z))], [], [(p, v2 z)])
                  | PBounded lo hi =>
                      p <- param name ;; w <- width_of p ;;
                      if negb (fits lo hi w) then refuse ("parameter " ++ name ++ " has bounds its register cannot hold")
                      else ret ([(p, (lo, hi))], [], [])
                  | PZero => p <- param name ;; ret ([(p, (0, 0))], [], [(p, 64)])
                  end) (b_params Mf) ;;
  let bounds := app [(tx, (0, bx - 1)); (ty, (0, by_ - 1)); (tz, (0, bz - 1));
                     (nx, (bx, bx)); (ny, (by_, by_)); (nz, (bz, bz));
                     (cx, (0, gx - 1)); (cy, (0, gy - 1)); (cz, (0, gz - 1));
                     (mx, (1, gx)); (my, (1, gy)); (mz, (1, gz)); (lane, (0, 31))]
                    (app sizes (flat_map (fun '(b, _, _) => b) decls)) in
  let roles := flat_map (fun '(_, r, _) => r) decls in
  let mults := flat_map (fun '(_, _, m) => m) decls in
  (* the bases of the windows of shared and local memory in the generic
     address space; every GPU keeps a window of at least window_size bytes
     below 2^64, at a multiple of window_size *)
  ws <- window_reg "shared" ;; wl <- window_reg "local" ;;
  s <- get_state ;;
  (* a shared or local variable's address is a multiple of its declared
     alignment, the least one if it is declared more than once; a shared
     variable's address lies below 2^20, since no GPU has 1 MiB of shared
     memory for a block *)
  let local_decls := map (fun '(x, a, _, _) => (x, a)) (k_locals K) in
  let shared_decls := map (fun '(x, a, _) => (x, a)) (k_shared K) in
  let vars := flat_map (fun x => let '(name, r) := x in
                 if String.prefix "shared:" name then [(r, drop 7 name, shared_decls)]
                 else if String.prefix "local:" name then [(r, drop 6 name, local_decls)]
                 else []) (ts_names s) in
  (* each shared variable's bytes: an array of unknown size, the dynamic
     shared memory, has the bytes of dynamic shared memory that every launch
     passes *)
  dynamic <- affine_form (b_shared Mf) ;;
  let shared := flat_map (fun x => let '(name, r) := x in
                   if String.prefix "shared:" name then
                     match find (fun v => let '(y, _, _) := v in String.eqb y (drop 7 name)) (k_shared K) with
                     | Some (_, _, bytes) => [(r, if Z.eqb bytes 0 then dynamic else fconst 64 bytes)]
                     | None => []
                     end
                   else []) (ts_names s) in
  let var_aligns := map (fun '(r, v, decls) =>
                      (r, zmin (map (fun e => low_log2 (snd e)) (filter (fun e => String.eqb (fst e) v) decls)) 0))
                    vars in
  let shared_bounds := flat_map (fun x => let '(name, r) := x in
                          if String.prefix "shared:" name then [(r, (0, 2 ^ 20 - 1))] else []) (ts_names s) in
  let window_bounds := [(ws, (0, 2 ^ 64 - window_size - 1)); (wl, (0, 2 ^ 64 - window_size - 1))] in
  (* each local variable's bytes and the frame of its call *)
  let locals := flat_map (fun x => let '(name, r) := x in
                   if String.prefix "local:" name then
                     match find (fun v => let '(y, _, _, _) := v in String.eqb y (drop 6 name)) (k_locals K) with
                     | Some (_, _, bytes, fr) => [(r, (bytes, fr))]
                     | None => []
                     end
                   else []) (ts_names s) in
  (* a local variable lies inside the window of local memory, whose top the
     stack of calls starts from, so its address is at most window_size less
     its bytes *)
  if negb (forallb (fun e => let '(_, (bytes, _)) := e in Z.leb 0 bytes && Z.leb bytes window_size) locals)
  then refuse "a local variable larger than the window of local memory" else
  let local_bounds := map (fun e => let '(r, (bytes, _)) := e in (r, (0, window_size - bytes))) locals in
  (* the cells of a local variable of at most 256 bytes: 8 bytes at each
     multiple of 8, and 4 bytes at each multiple of 4 *)
  cells <- mapT (fun (e : reg * (Z * frame)) =>
             let '(r, (bytes, _)) := e in
             if Z.leb bytes 256 then
               c8 <- mapT (fun i => p <- fresh 64 ;; ret (p, r, 8 * Z.of_nat i)) (seq 0 (Z.to_nat (bytes / 8))) ;;
               c4 <- mapT (fun i => p <- fresh 32 ;; ret (p, r, 4 * Z.of_nat i)) (seq 0 (Z.to_nat (bytes / 4))) ;;
               ret (app c8 c4)
             else ret []) locals ;;
  s2 <- get_state ;;
  let named := filter (fun x => let '(name, _) := x in
                        String.prefix "param:" name || String.prefix "shared:" name
                        || String.prefix "local:" name || String.prefix "window:" name
                        || String.prefix "size:" name || sin name specials) (ts_names s) in
  let fixed := rev (map snd named) in
  ret {| lf_widths := ts_widths s2;
         lf_fixed := fixed;
         lf_bounds := app bounds (app shared_bounds (app local_bounds window_bounds));
         lf_aligns := app mults (app var_aligns [(ws, 24); (wl, 24)]);
         lf_roles := roles;
         lf_shared_window := ws;
         lf_local_window := wl;
         lf_shared := shared;
         lf_locals := locals;
         lf_cells := List.concat cells;
         lf_payloads := PM.empty unit;
         lf_derived := derived;
         lf_names := named |}.

End Translate.

(* ------------------------------------------------------------------ *)
(* Calls.                                                              *)

(* The translation reads a kernel with each call of a device function of
   the module replaced by the function's body. The body's registers, labels,
   and own parameter variables get names of the call's number, which no PTX
   name has; each parameter or return value of the function is the variable
   of the caller that the call names; and a return branches to a label after
   the body. A parameter variable holds what st.param stores and ld.param
   loads: each of its w-bit values at byte offset off is one register, which
   a store copies into and a load copies out of. A call of a function that
   the module does not define, a call through a register, and calls nested
   deeper than inline_depth, which a recursive call is, are refused.

   Each call has its own frame: its local variables get names of the call's
   number, the frame is the frames of the calls that lead to it and the
   call's number, and the body starts with the pseudo-instruction frame$ of
   its local variables, whose memory holds nothing the call wrote before. *)

(* A body as labels and instructions in order: the labels of position n come
   just before instruction n, and those at the end after the last one. Each
   instruction carries its origin, the kernel instruction it comes from, and
   the frame of the call it belongs to. *)
Inductive item := Lbl (name : string) | Ins (origin : N) (fr : frame) (i : pinstr).

(* The names of the labels at each position, in their order in the list. *)
Definition label_index (labels : list (string * N)) : PositiveMap.t (list string) :=
  fold_right (fun e m => let k := N.succ_pos (snd e) in
    PositiveMap.add k (fst e :: match PositiveMap.find k m with Some l => l | None => [] end) m)
    (PositiveMap.empty (list string)) labels.

(* The items of l from position n: the labels at each position, then its
   instruction. The index gives each position's labels in the order that a
   filter of the label list gives them, without a pass over every label. *)
Fixpoint items_at (ix : PositiveMap.t (list string)) (origin : N -> N) (fr : frame) (n : N) (l : list pinstr)
    : list item :=
  let here := map Lbl (match PositiveMap.find (N.succ_pos n) ix with Some ls => ls | None => [] end) in
  match l with
  | [] => here
  | i :: t => app here (Ins (origin n) fr i :: items_at ix origin fr (N.succ n) t)
  end.

Definition items_from (labels : list (string * N)) (origin : N -> N) (fr : frame) (n : N) (l : list pinstr)
    : list item :=
  items_at (label_index labels) origin fr n l.

(* The scope of a body: the prefix of its own names, "" at the kernel, its
   register declarations and local variables, the caller's variable that each
   of its parameters and return values is, the label its returns branch to,
   at the kernel the kernel's parameters, and its frame. *)
Record scope := {
  sc_pre : string;
  sc_regs : list (string * nat * option Z);
  sc_locals : list string;
  sc_params : list (string * string);
  sc_ret : option string;
  sc_kparams : list string;
  sc_frame : frame
}.

Definition prefix_reg (pre r : string) : string :=
  match r with
  | String c t => if Ascii.eqb c "%"%char then String c (pre ++ t) else pre ++ r
  | EmptyString => pre
  end.

(* A register of the scope under its name in the kernel. *)
Definition rename_reg (sc : scope) (r : string) : string :=
  if String.eqb (sc_pre sc) "" then r
  else if existsb (fun d => names d r) (sc_regs sc) then prefix_reg (sc_pre sc) r
  else r.

Definition rename_label (sc : scope) (l : string) : string := sc_pre sc ++ l.

Fixpoint join (sep : string) (l : list string) : string :=
  match l with [] => "" | [x] => x | x :: t => x ++ sep ++ join sep t end.

Fixpoint rename_opnd (sc : scope) (o : popnd) : popnd :=
  match o with
  | PReg r => PReg (rename_reg sc r)
  | PNot r => PNot (rename_reg sc r)
  | PVec xs => PVec (map (rename_opnd sc) xs)
  | PTensor m cs => PTensor (rename_reg sc m) (map (rename_opnd sc) cs)
  | PAddr (Some r) off => PAddr (Some (rename_reg sc r)) off
  | PLabel l => PLabel (rename_label sc l)
  | PText s => if existsb (String.eqb s) (sc_locals sc) then PText (sc_pre sc ++ s)
               else PText (join "|" (map (rename_reg sc) (split_on "|"%char s "")))
  | PSym s off => if existsb (String.eqb s) (sc_locals sc) then PSym (sc_pre sc ++ s) off else o
  | _ => o
  end.

Definition rename_guard (sc : scope) (p : option (bool * string)) : option (bool * string) :=
  option_map (fun '(neg, r) => (neg, rename_reg sc r)) p.

(* The variable of the kernel that a parameter variable of the scope is: the
   caller's variable that the call binds to a parameter or return value, or
   else the scope's own variable. *)
Definition variable (sc : scope) (s : string) : string :=
  match find (fun e => String.eqb (fst e) s) (sc_params sc) with
  | Some (_, v) => v
  | None => sc_pre sc ++ s
  end.

(* The register that holds the w-bit value of parameter variable v at byte
   offset off. *)
Definition cell (v : string) (off : Z) (w : nat) : string :=
  "%param$" ++ v ++ "$" ++ nstr (Z.to_N off) ++ "$" ++ nstr (N.of_nat w).

Definition reg_width (sc : scope) (r : string) : option nat :=
  match filter (fun d => names d r) (sc_regs sc) with [(_, w, _)] => Some w | _ => None end.

Definition int_sfx (signed : bool) (w : nat) : string := (if signed then "s" else "u") ++ nstr (N.of_nat w).

Definition bits_sfx (w : nat) : string := "b" ++ nstr (N.of_nat w).

Definition decl := (string * nat * option Z)%type.

(* ld.param.<t> d, [s + off], or a vector of destinations: each element
   copies its cell, extended to its register's width. *)
Definition cell_loads (sc : scope) (o : N) (p : option (bool * string)) (sf : list string)
    (dst : popnd) (s : string) (off : Z) : res (list item * list decl) :=
  match types sf with
  | [t] =>
      match type_width t with
      | Some w =>
          let one := fun (k : nat) (x : popnd) =>
            match x with
            | PReg r =>
                let c := cell (variable sc s) (off + Z.of_nat k * (Z.of_nat w / 8)) w in
                let r' := rename_reg sc r in
                let i := match reg_width sc r with
                         | Some wr => if Nat.eqb wr w then PI p "mov" [bits_sfx w] [PReg r'; PReg c]
                                      else PI p "cvt" [int_sfx (is_s t) wr; int_sfx (is_s t) w] [PReg r'; PReg c]
                         | None => PI p "mov" [bits_sfx w] [PReg r'; PReg c]
                         end in
                Ok (Ins o (sc_frame sc) i, (c, w, None))
            | _ => Err "a parameter load into an operand that is not a register" (Some o)
            end in
          let xs := match dst with PVec ys => ys | y => [y] end in
          fold_right (fun kx acc => match acc, one (fst kx) (snd kx) with
                                    | Ok (ins, ds), Ok (i, d) => Ok (i :: ins, d :: ds)
                                    | Err e a, _ => Err e a
                                    | _, Err e a => Err e a
                                    end) (Ok ([], [])) (combine (seq 0 (List.length xs)) xs)
      | None => Err "a parameter load of a type without a width" (Some o)
      end
  | _ => Err "a parameter load without one type" (Some o)
  end.

(* st.param.<t> [s + off], v, or a vector of sources: each element sets its
   cell, truncated to the type's width, and the cell of each narrower width
   at the same offset, which holds the element's low bytes: a load of fewer
   bits from that offset, as of a byte that the caller passes in 32 bits,
   reads them. *)
Definition cell_stores (sc : scope) (o : N) (p : option (bool * string)) (sf : list string)
    (src : popnd) (s : string) (off : Z) : res (list item * list decl) :=
  match types sf with
  | [t] =>
      match type_width t with
      | Some w =>
          let one := fun (k : nat) (x : popnd) =>
            let at_ := off + Z.of_nat k * (Z.of_nat w / 8) in
            let c := cell (variable sc s) at_ w in
            let low := map (fun n => let c' := cell (variable sc s) at_ n in
                                     (Ins o (sc_frame sc) (PI p "cvt" [int_sfx false n; int_sfx false w] [PReg c'; PReg c]),
                                      (c', n, None)))
                           (filter (fun n => Nat.ltb n w) [8; 16; 32]%nat) in
            match x with
            | PReg r =>
                let r' := rename_reg sc r in
                let i := match reg_width sc r with
                         | Some wr => if Nat.eqb wr w then PI p "mov" [bits_sfx w] [PReg c; PReg r']
                                      else PI p "cvt" [int_sfx false w; int_sfx false wr] [PReg c; PReg r']
                         | None => PI p "mov" [bits_sfx w] [PReg c; PReg r']
                         end in
                Ok ((Ins o (sc_frame sc) i, (c, w, None)) :: low)
            | PImm _ | PFloat _ => Ok ((Ins o (sc_frame sc) (PI p "mov" [bits_sfx w] [PReg c; x]), (c, w, None)) :: low)
            | _ => Err "a parameter store of an operand that is not a register or a number" (Some o)
            end in
          let xs := match src with PVec ys => ys | y => [y] end in
          fold_right (fun kx acc => match acc, one (fst kx) (snd kx) with
                                    | Ok (ins, ds), Ok l => Ok (app (map fst l) ins, app (map snd l) ds)
                                    | Err e a, _ => Err e a
                                    | _, Err e a => Err e a
                                    end) (Ok ([], [])) (combine (seq 0 (List.length xs)) xs)
      | None => Err "a parameter store of a type without a width" (Some o)
      end
  | _ => Err "a parameter store without one type" (Some o)
  end.

(* An instruction of the scope other than a call, under its names in the
   kernel: a return of a function branches to its return label, and a load
   or store of a parameter variable that is not the kernel's copies a cell. *)
Definition rewrite (sc : scope) (o : N) (i : pinstr) : res (list item * list decl) :=
  let '(PI p mn sf ops) := i in
  let p' := rename_guard sc p in
  let same := Ok ([Ins o (sc_frame sc) (PI p' mn sf (map (rename_opnd sc) ops))], []) in
  if String.eqb mn "ret" then
    match sc_ret sc with
    | Some l => Ok ([Ins o (sc_frame sc) (PI p' "bra" [] [PLabel l])], [])
    | None => same
    end
  else if existsb (fun x => sin x ["param"; "param::func"]) sf then
    match ops with
    | [dst; PSym s off] =>
        if String.eqb mn "ld" && negb (sin s (sc_kparams sc)) then cell_loads sc o p' sf dst s off else same
    | [PSym s off; src] => if String.eqb mn "st" then cell_stores sc o p' sf src s off else same
    | _ => same
    end
  else same.

(* The variables that a call's operand lists, and the name of its function. *)
Definition name_list (o : popnd) : option (list string) :=
  match o with
  | PVec xs => fold_right (fun x acc => match x, acc with
                                         | PText s, Some l | PLabel s, Some l => Some (s :: l)
                                         | _, _ => None
                                         end) (Some []) xs
  | _ => None
  end.

Definition fname (o : popnd) : option string := match o with PText s | PLabel s => Some s | _ => None end.

(* A call's return variables, function, and argument variables:
   call (r...), f, (a...); call (r...), f; call f, (a...); or call f. *)
Definition call_parts (ops : list popnd) : option (list string * string * list string) :=
  match ops with
  | [r; f; a] => match name_list r, fname f, name_list a with
                 | Some rs, Some n, Some xs => Some (rs, n, xs)
                 | _, _, _ => None
                 end
  | [PVec _ as r; f] => match name_list r, fname f with Some rs, Some n => Some (rs, n, []) | _, _ => None end
  | [f; a] => match fname f, name_list a with Some n, Some xs => Some ([], n, xs) | _, _ => None end
  | [f] => match fname f with Some n => Some ([], n, []) | None => None end
  | _ => None
  end.

(* The state of inlining: the next call's number, the items so far, the last
   first, and the declarations and .branchtargets lists of the inlined bodies
   and of the cells. *)
Record istate := {
  i_next : N;
  i_out : list item;
  i_regs : list decl;
  i_targets : list (string * list string);
  i_locals : list (string * Z * Z * frame)
}.

Definition emit (s : istate) (xs : list item) (ds : list decl) : istate :=
  {| i_next := i_next s; i_out := rev_append xs (i_out s);
     i_regs := fold_left (fun acc d => let '(name, _, _) := d in
                            if existsb (fun e => let '(n, _, _) := e in String.eqb n name) acc then acc
                            else d :: acc) ds (i_regs s);
     i_targets := i_targets s; i_locals := i_locals s |}.

Definition inline_depth : nat := 16.

(* PTX uniform control qualifiers are promises about all active warp
   threads, not value hints. A fixed direct target and no predicate establish
   the source shape of an unconditional bra.uni/call.uni. An indexed target,
   a predicate, or ret.uni's return-convergence promise requires a separate
   uniformity proof. Check before inlining removes calls/return qualifiers,
   and only when expanding a body so uncalled functions stay irrelevant.
   NVIDIA PTX ISA 9.4 sections 9.7.14.3--9.7.14.6. *)
Definition source_control_ok (i : pinstr) : bool :=
 let 'PI pred mn sf _ := i in
 if sin "uni" sf then
   if sin mn ["brx"; "ret"] then false else
   if sin mn ["bra"; "call"] then
     match pred with None => true | Some _ => false end
   else true
 else true.

Lemma source_control_indirect : forall pred sf ops,
 sin "uni" sf = true -> source_control_ok (PI pred "brx" sf ops) = false.
Proof. intros pred sf ops H. unfold source_control_ok; rewrite H; reflexivity. Qed.
Lemma source_control_return : forall pred sf ops,
 sin "uni" sf = true -> source_control_ok (PI pred "ret" sf ops) = false.
Proof. intros pred sf ops H. unfold source_control_ok; rewrite H; reflexivity. Qed.
Lemma source_control_predicated_direct : forall pred mn sf ops,
 sin "uni" sf = true -> sin mn ["bra";"call"] = true ->
 source_control_ok (PI (Some pred) mn sf ops) = false.
Proof.
 intros pred mn sf ops Hu Hm. unfold source_control_ok; rewrite Hu.
 destruct (sin mn ["brx";"ret"]); [reflexivity|]. rewrite Hm; reflexivity.
Qed.

(* The items of a body with each call replaced by the called function's
   body, in scope sc. *)
Fixpoint inline_items (fuel : nat) (funcs : list pfunc) (sc : scope) (body : list item) (s : istate)
    {struct fuel} : res istate :=
  let fix go (l : list item) (s : istate) : res istate :=
    match l with
    | [] => Ok s
    | Lbl x :: t => go t (emit s [Lbl (rename_label sc x)] [])
    | Ins o _ i :: t =>
        let '(PI p mn sf ops) := i in
        if negb (source_control_ok i) then
          Err "a uniform control qualifier requires a checked warp-uniformity proof" (Some o)
        else if negb (String.eqb mn "call") then
          match rewrite sc o i with Ok (xs, ds) => go t (emit s xs ds) | Err e a => Err e a end
        else if match call_parts ops with Some (_, n, _) => String.eqb n "__assertfail" | None => false end then
          (* a failed assertion traps *)
          go t (emit s [Ins o (sc_frame sc) (PI (rename_guard sc p) "trap" [] [])] [])
        else
          match fuel, call_parts ops with
          | O, _ => Err "a call nested too deep, or a recursive call" (Some o)
          | _, None => Err "a call through a register, or without its function and variables" (Some o)
          | S fuel', Some (rs, n, args) =>
              match find (fun f => String.eqb (f_name f) n) funcs with
              | None => Err ("a call of " ++ n ++ ", which the module does not define") (Some o)
              | Some f =>
                  if negb (Nat.eqb (List.length args) (List.length (f_params f)) &&
                           Nat.eqb (List.length rs) (List.length (f_rets f)))
                  then Err ("a call of " ++ n ++ " with other parameters than it declares") (Some o) else
                  let pre := "c" ++ nstr (i_next s) ++ "$" in
                  let back := pre ++ "return" in
                  let fr := app (sc_frame sc) [i_next s] in
                  let sc' := {| sc_pre := pre; sc_regs := f_regs f;
                                sc_locals := map (fun '(x, _, _) => x) (f_locals f);
                                sc_params := app (combine (f_params f) (map (variable sc) args))
                                                 (combine (f_rets f) (map (variable sc) rs));
                                sc_ret := Some back; sc_kparams := []; sc_frame := fr |} in
                  (* a call whose guard fails continues after it *)
                  let skip := match rename_guard sc p with
                              | Some (neg, r) => [Ins o (sc_frame sc) (PI (Some (negb neg, r)) "bra" [] [PLabel back])]
                              | None => []
                              end in
                  (* the call's frame, of its local variables *)
                  let enter := match f_locals f with
                               | [] => []
                               | ls => [Ins o fr (PI None "frame$" [] (map (fun '(x, _, _) => PLabel (pre ++ x)) ls))]
                               end in
                  let s1 := {| i_next := N.succ (i_next s); i_out := rev_append (app skip enter) (i_out s);
                               i_regs := app (rev (map (fun d => let '(name, w, c) := d in (prefix_reg pre name, w, c))
                                                       (f_regs f))) (i_regs s);
                               i_targets := app (map (fun e => (rename_label sc' (fst e), map (rename_label sc') (snd e)))
                                                     (f_branch_targets f)) (i_targets s);
                               i_locals := app (i_locals s) (map (fun '(x, a, b) => (pre ++ x, a, b, fr)) (f_locals f)) |} in
                  match inline_items fuel' funcs sc' (items_from (f_labels f) (fun _ => o) fr 0 (f_insts f)) s1 with
                  | Ok s2 => go t (emit s2 [Lbl back] [])
                  | Err e a => Err e a
                  end
              end
          end
    end in go body s.

(* The kernel with each call replaced by the called function's body, and the
   place of each of its instructions: its origin, the kernel instruction it
   is or the call it comes from, and its frame. *)
Definition inline (K : pkernel) : res (pkernel * list place) :=
  let sc := {| sc_pre := ""; sc_regs := k_regs K; sc_locals := []; sc_params := []; sc_ret := None;
               sc_kparams := map fst (k_params K); sc_frame := [] |} in
  let s0 := {| i_next := 0%N; i_out := []; i_regs := []; i_targets := []; i_locals := [] |} in
  match inline_items inline_depth (k_funcs K) sc (items_from (k_labels K) (fun n => n) [] 0 (k_insts K)) s0 with
  | Err e a => Err e a
  | Ok s =>
      let '(_, labels, insts) :=
        fold_left (fun (acc : N * list (string * N) * list (place * pinstr)) x =>
                     let '(n, ls, xs) := acc in
                     match x with
                     | Lbl l => (n, (l, n) :: ls, xs)
                     | Ins o fr i => (N.succ n, ls, ((o, fr), i) :: xs)
                     end) (rev (i_out s)) (0%N, [], []) in
      let insts := rev insts in
      Ok ({| k_address_size := k_address_size K; k_target := k_target K;
             k_ptx_version := k_ptx_version K; k_params := k_params K;
             k_regs := app (k_regs K) (rev (i_regs s)); k_shared := k_shared K;
             k_locals := app (k_locals K) (i_locals s);
             k_reqntid := k_reqntid K; k_cluster := k_cluster K; k_labels := rev labels;
             k_branch_targets := app (k_branch_targets K) (i_targets s); k_funcs := [];
             k_insts := map snd insts |}, map fst insts)
  end.

(* ------------------------------------------------------------------ *)
(* The zeroing prologue.                                               *)

(* Shared memory and local memory hold, when a kernel starts, what the
   kernels that ran before it on the multiprocessor left there. The gate
   refuses a kernel whose first instructions are not exactly the prologue
   that `prologue` gives (opgate/gate/prologue.py writes it): they write
   zero to every byte of every shared variable of the module, by all
   threads of the block, then wait at a barrier, and then write zero to
   every byte of every local variable of the kernel. A shared variable of
   fixed bytes, or a dynamic one whose bytes do not depend on the sizes,
   is cleared in rounds unrolled to its bytes. A dynamic one whose bytes
   depend on the sizes is cleared by a loop: thread t stores the elements
   t, t + N, t + 2N, ... of the N threads of the block below the count of
   elements, which the loop computes from the binding's size parameters
   at run time. Each device function of the module that declares local
   variables begins with the stores of its local variables, which a call
   runs when it enters its frame. GateZero.v proves that each thread's run
   of the loop stores zero at each of its elements before it reaches the
   barrier, and that the threads of a block store every byte of the
   dynamic shared memory of the launch. Trusted: that every byte of shared
   or local memory that a run of a kernel that begins with these
   instructions reads holds zero or a value that the run wrote, which
   rests on the barrier making the stores of the block's threads visible
   to their later loads, and which StoreLaunch.v states by starting both
   memories at zero. *)

(* The prologue's registers: the block's linear thread index, an address,
   zero, and whether a thread stores in a variable's last round. *)
Definition zero_index : string := "%opgate_z0".
Definition zero_address : string := "%opgate_z1".
Definition zero_value : string := "%opgate_z2".
Definition zero_predicate : string := "%opgate_p".
(* The registers of the loop that clears dynamic shared memory: the count
   of its elements, and the element that a thread stores. *)
Definition zero_count : string := "%opgate_d0".
Definition zero_element : string := "%opgate_d1".

(* The labels of the loop of the j-th dynamic shared variable: its head, and
   the instruction after it. *)
Definition zero_head (j : nat) : string := "$opgate_zero_" ++ nstr (N.of_nat j).
Definition zero_exit (j : nat) : string := "$opgate_zeroed_" ++ nstr (N.of_nat j).

(* The bytes of each store to a local variable: the largest power of two up
   to 16 that divides the variable's alignment and its bytes. *)
Definition zero_width (align bytes : Z) : Z :=
  if Z.eqb (align mod 16) 0 && Z.eqb (bytes mod 16) 0 then 16
  else if Z.eqb (align mod 8) 0 && Z.eqb (bytes mod 8) 0 then 8
  else if Z.eqb (align mod 4) 0 && Z.eqb (bytes mod 4) 0 then 4
  else if Z.eqb (align mod 2) 0 && Z.eqb (bytes mod 2) 0 then 2
  else 1.

(* A store of w zero bytes in state space sp at the address operand a. *)
Definition zero_store (g : option (bool * string)) (sp : string) (w : Z) (a : popnd) : pinstr :=
  if Z.eqb w 16 then PI g "st" [sp; "v4"; "u32"] [a; PVec (repeat (PReg zero_value) 4)]
  else if Z.eqb w 8 then PI g "st" [sp; "v2"; "u32"] [a; PVec (repeat (PReg zero_value) 2)]
  else PI g "st" [sp; "u" ++ nstr (Z.to_N (8 * w))] [a; PReg zero_value].

(* A variable's name as the parser reads it as an operand: a label when it
   looks like one. *)
Definition name_operand (name : string) : popnd :=
  if (String.prefix "$" name && Nat.ltb 1 (String.length name)) || String.prefix "L_" name ||
     String.prefix "LBB" name
  then PLabel name else PText name.

(* The bytes of each store to a shared variable: the largest of 4, 2, and 1
   that divides the alignment and the bytes of every shared variable vs. The
   stores are then stores of the number 0 of one width, which the shared
   value analysis takes as ordered writes (`writes_ordered_at` in
   GateShared.v). *)
Definition shared_width (vs : list (string * Z * Z)) : Z :=
  let divides := fun w => forallb (fun v => let '(_, align, bytes) := v in
                                            Z.eqb (align mod w) 0 && Z.eqb (bytes mod w) 0) vs in
  if divides 4 then 4 else if divides 2 then 2 else 1.

(* The stores of w bytes of a shared variable of the given bytes, by
   `threads` threads: thread t stores element t + k * threads in round k. *)
Definition zero_shared (threads w : Z) (name : string) (bytes : Z) : list pinstr :=
  let elements := bytes / w in
  let full := elements / threads in
  let left := elements mod threads in
  [PI None "mov" ["u32"] [PReg zero_address; name_operand name];
   PI None "mad" ["lo"; "u32"] [PReg zero_address; PReg zero_index; PImm w; PReg zero_address]] ++
  map (fun k => zero_store None "shared" w (PAddr (Some zero_address) (w * threads * Z.of_nat k)))
      (seq 0 (Z.to_nat full)) ++
  (if Z.eqb left 0 then [] else
   [PI None "setp" ["lt"; "u32"] [PReg zero_predicate; PReg zero_index; PImm left];
    zero_store (Some (false, zero_predicate)) "shared" w (PAddr (Some zero_address) (w * threads * full))]).

(* The stores of a local variable, by each thread. *)
Definition zero_local (name : string) (align bytes : Z) : list pinstr :=
  let w := zero_width align bytes in
  map (fun k => zero_store None "local" w (PSym name (w * Z.of_nat k))) (seq 0 (Z.to_nat (bytes / w))).

(* The prologue of a device function: the stores of its local variables. *)
Definition function_prologue (f : pfunc) : list pinstr :=
  match filter (fun v => let '(_, _, bytes) := v in Z.ltb 0 bytes) (f_locals f) with
  | [] => []
  | locals =>
      PI None "mov" ["u32"] [PReg zero_value; PImm 0] ::
      flat_map (fun v => let '(name, align, bytes) := v in zero_local name align bytes) locals
  end.

(* Whether the bytes of dynamic shared memory depend on the sizes. *)
Definition sized_shared (Mf : binding) : bool := negb (null (af_terms (b_shared Mf))).

(* Each shared variable of the module that rounds unrolled to its bytes
   clear, with its bytes, when it has any: a variable of fixed bytes, and an
   array of unknown size, the dynamic shared memory, when its bytes, the
   binding's, do not depend on the sizes. *)
Definition zeroed_shared (K : pkernel) (Mf : binding) : list (string * Z * Z) :=
  flat_map (fun v => let '(name, align, bytes) := v in
              if Z.eqb bytes 0 && sized_shared Mf then [] else
              let size := if Z.eqb bytes 0 then af_const (b_shared Mf) else bytes in
              if Z.ltb 0 size then [(name, align, size)] else []) (k_shared K).

(* Each array of unknown size with its alignment, when the bytes of
   dynamic shared memory depend on the sizes: a loop clears each. *)
Definition zeroed_dynamic (K : pkernel) (Mf : binding) : list (string * Z) :=
  if sized_shared Mf then
    flat_map (fun v => let '(name, align, bytes) := v in
                if Z.eqb bytes 0 then [(name, align)] else []) (k_shared K)
  else [].

(* The alignments and byte counts that the width of the shared stores must
   divide: those of each variable that rounds clear, and for each that a
   loop clears, its alignment with the constant and each coefficient of the
   bytes of dynamic shared memory. *)
Definition width_entries (K : pkernel) (Mf : binding) : list (string * Z * Z) :=
  zeroed_shared K Mf ++
  flat_map (fun v => let '(name, align) := v in
              map (fun c => (name, align, c)) (af_const (b_shared Mf) :: map snd (af_terms (b_shared Mf))))
           (zeroed_dynamic K Mf).

(* The first parameter of the binding that holds size symbol k itself. *)
Definition size_parameter (Mf : binding) (k : Z) : option string :=
  match find (fun p => match snd p with
                       | PSize a => Z.eqb (af_const a) 0 &&
                                    match af_terms a with
                                    | [(k', c)] => Z.eqb k' k && Z.eqb c 1
                                    | _ => false
                                    end
                       | _ => false
                       end) (b_params Mf) with
  | Some (name, _) => Some name
  | None => None
  end.

(* The parameter that holds the size of each term of the bytes of dynamic
   shared memory, if every one has one. *)
Definition size_parameters (Mf : binding) : option (list string) :=
  fold_right (fun x acc => match acc, size_parameter Mf (fst x) with
                           | Some ps, Some p => Some (p :: ps)
                           | _, _ => None
                           end) (Some []) (af_terms (b_shared Mf)).

(* The loop that clears the j-th dynamic shared variable, name, in stores of
   w bytes by `threads` threads. Its count of elements is the constant plus
   each coefficient times its size, read from the size's parameter, all
   divided by w, at run time. Thread t, the block's linear thread index,
   stores the elements t, t + threads, t + 2 threads, ... below the count. *)
Definition zero_dynamic (threads w : Z) (a : affine) (params : list string) (j : nat) (name : string)
    : list pinstr :=
  PI None "mov" ["u32"] [PReg zero_count; PImm (af_const a / w)] ::
  flat_map (fun e => let '((_, c), p) := e in
              [PI None "ld" ["param"; "u32"] [PReg zero_element; PSym p 0];
               PI None "mad" ["lo"; "u32"] [PReg zero_count; PReg zero_element; PImm (c / w); PReg zero_count]])
           (combine (af_terms a) params) ++
  [PI None "mov" ["u32"] [PReg zero_element; PReg zero_index];
   PI None "setp" ["ge"; "u32"] [PReg zero_predicate; PReg zero_element; PReg zero_count];
   PI (Some (false, zero_predicate)) "bra" [] [PLabel (zero_exit j)];
   PI None "mov" ["u32"] [PReg zero_address; name_operand name];
   PI None "mad" ["lo"; "u32"] [PReg zero_address; PReg zero_element; PImm w; PReg zero_address];
   zero_store None "shared" w (PAddr (Some zero_address) 0);
   PI None "add" ["u32"] [PReg zero_element; PReg zero_element; PImm threads];
   PI None "bra" [] [PLabel (zero_head j)]].

(* The instructions of one loop, for terms of the bytes of dynamic shared
   memory. *)
Definition dynamic_length (terms : nat) : nat := 9 + 2 * terms.

(* The prologue's instructions before its first loop: zero, and the block's
   linear thread index. *)
Definition index_length : nat := 6.

(* The labels of the loops, and the instruction that each precedes: the
   head of loop j at its comparison, its exit after its last instruction. *)
Definition prologue_labels (K : pkernel) (Mf : binding) : list (string * N) :=
  let n := dynamic_length (List.length (af_terms (b_shared Mf))) in
  flat_map (fun j => let start := (index_length + j * n)%nat in
                     [(zero_head j, N.of_nat (start + 2 + 2 * List.length (af_terms (b_shared Mf))));
                      (zero_exit j, N.of_nat (start + n))])
           (seq 0 (List.length (zeroed_dynamic K Mf))).

Definition prologue (K : pkernel) (Mf : binding) : list pinstr :=
  let '(bx, by_, bz) := b_block Mf in
  let shared := zeroed_shared K Mf in
  let dynamic := zeroed_dynamic K Mf in
  let w := shared_width (width_entries K Mf) in
  let params := match size_parameters Mf with Some ps => ps | None => [] end in
  let locals := filter (fun v => let '(_, _, bytes, _) := v in Z.ltb 0 bytes) (k_locals K) in
  if null shared && null dynamic && null locals then [] else
      PI None "mov" ["u32"] [PReg zero_value; PImm 0] ::
      (if null shared && null dynamic then [] else
              [PI None "mov" ["u32"] [PReg zero_index; PText "%tid.z"];
               PI None "mov" ["u32"] [PReg zero_address; PText "%tid.y"];
               PI None "mad" ["lo"; "u32"] [PReg zero_index; PReg zero_index; PImm by_; PReg zero_address];
               PI None "mov" ["u32"] [PReg zero_address; PText "%tid.x"];
               PI None "mad" ["lo"; "u32"] [PReg zero_index; PReg zero_index; PImm bx; PReg zero_address]] ++
              flat_map (fun e => let '(j, (name, _)) := e in
                                 zero_dynamic (bx * by_ * bz) w (b_shared Mf) params j name)
                       (combine (seq 0 (List.length dynamic)) dynamic) ++
              flat_map (fun v => let '(name, _, size) := v in
                                 zero_shared (bx * by_ * bz) w name size) shared ++
              [PI None "bar" ["sync"] [PImm 0]]) ++
      flat_map (fun v => let '(name, align, bytes, _) := v in zero_local name align bytes) locals.

Fixpoint popnd_eqb (a b : popnd) {struct a} : bool :=
  let fix all (xs ys : list popnd) : bool :=
    match xs, ys with
    | [], [] => true
    | x :: xs', y :: ys' => popnd_eqb x y && all xs' ys'
    | _, _ => false
    end in
  match a, b with
  | PReg x, PReg y | PNot x, PNot y | PFloat x, PFloat y | PLabel x, PLabel y | PText x, PText y =>
      String.eqb x y
  | PVec xs, PVec ys => all xs ys
  | PImm x, PImm y => Z.eqb x y
  | PAddr None x, PAddr None y => Z.eqb x y
  | PAddr (Some r) x, PAddr (Some s) y => String.eqb r s && Z.eqb x y
  | PSym r x, PSym s y => String.eqb r s && Z.eqb x y
  | PTensor m xs, PTensor n ys => String.eqb m n && all xs ys
  | _, _ => false
  end.

Definition pinstr_eqb (a b : pinstr) : bool :=
  let '(PI p m s xs) := a in
  let '(PI q n t ys) := b in
  match p, q with
  | None, None => true
  | Some (u, r), Some (v, w) => Bool.eqb u v && String.eqb r w
  | _, _ => false
  end && String.eqb m n && list_eqs s t &&
  Nat.eqb (List.length xs) (List.length ys) && forallb (fun e => popnd_eqb (fst e) (snd e)) (combine xs ys).

(* The first position at which the instructions differ from the prologue,
   if any. *)
Fixpoint prologue_mismatch (n : N) (expected actual : list pinstr) : option N :=
  match expected, actual with
  | [], _ => None
  | _ :: _, [] => Some n
  | e :: es, a :: rest => if pinstr_eqb e a then prologue_mismatch (N.succ n) es rest else Some n
  end.

(* The one declaration of a register, if exactly one names it. *)
Definition only_declaration (regs : list (string * nat * option Z)) (r : string)
    : option (string * nat * option Z) :=
  match filter (fun d => names d r) regs with [d] => Some d | _ => None end.

Definition declaration_eqb (d e : string * nat * option Z) : bool :=
  let '(x, w, c) := d in
  let '(y, v, b) := e in
  String.eqb x y && Nat.eqb w v &&
  match c, b with None, None => true | Some m, Some n => Z.eqb m n | _, _ => false end.

Definition zero_registers : string * nat * option Z := ("%opgate_z", 32%nat, Some 3).

Definition registers_ok (regs : list (string * nat * option Z)) (expected : list (string * (string * nat * option Z)))
    : bool :=
  forallb (fun e => match only_declaration regs (fst e) with
                    | Some d => declaration_eqb d (snd e)
                    | None => false
                    end) expected.

(* A device function that declares local variables begins with their stores,
   and a branch does not enter them. *)
Definition function_zeroed (f : pfunc) : bool :=
  let p := function_prologue f in
  match p with
  | [] => true
  | _ =>
      match prologue_mismatch 0 p (f_insts f) with
      | Some _ => false
      | None =>
          registers_ok (f_regs f) [(zero_value, zero_registers)] &&
          negb (existsb (fun l => N.ltb (snd l) (N.of_nat (List.length p))) (f_labels f))
      end
  end.

Definition dynamic_registers : string * nat * option Z := ("%opgate_d", 32%nat, Some 2).

Definition label_eqb (l m : string * N) : bool := String.eqb (fst l) (fst m) && N.eqb (snd l) (snd m).

(* The labels before the n-th instruction are exactly the loops' own. *)
Definition prologue_labels_ok (K : pkernel) (Mf : binding) (n : nat) : bool :=
  let inside := filter (fun l => N.ltb (snd l) (N.of_nat n)) (k_labels K) in
  let expected := prologue_labels K Mf in
  Nat.eqb (List.length inside) (List.length expected) &&
  forallb (fun l => existsb (label_eqb l) inside) expected.

(* No branch after the first n instructions, and no list of branch
   targets, names a label of the loops. *)
Definition prologue_entered (K : pkernel) (Mf : binding) (n : nat) : bool :=
  let names := map fst (prologue_labels K Mf) in
  existsb (fun i => let '(PI _ mn _ ops) := i in
                    String.eqb mn "bra" &&
                    existsb (fun o => match o with PLabel l => sin l names | _ => false end) ops)
          (skipn n (k_insts K)) ||
  existsb (fun t => existsb (fun l => sin l names) (snd t)) (k_branch_targets K).

(* The kernel begins with its zeroing prologue, and a branch does not enter
   it; so does each device function that declares local variables. The
   bytes of dynamic shared memory that a loop clears must be a constant and
   sizes with positive coefficients, each held by a parameter. *)
Definition zeroing_check (K : pkernel) (Mf : binding) : res unit :=
  let dynamic := zeroed_dynamic K Mf in
  if negb (null dynamic) &&
     negb (Z.leb 0 (af_const (b_shared Mf)) && forallb (fun x => Z.ltb 0 (snd x)) (af_terms (b_shared Mf))) then
    Err "the zeroing prologue needs dynamic shared memory whose bytes have a constant and coefficients that are not negative" None
  else if negb (null dynamic) && negb (defined (size_parameters Mf)) then
    Err "the zeroing prologue needs a parameter that holds each size of the dynamic shared memory" None
  else match find (fun f => negb (function_zeroed f)) (k_funcs K) with
  | Some f =>
      Err ("the device function " ++ f_name f ++
           " does not begin with the zeroing prologue of its local variables, with %opgate_z<3> declared once") None
  | None =>
    let p := prologue K Mf in
    match p with
    | [] => Ok tt
    | _ =>
        match prologue_mismatch 0 p (k_insts K) with
        | Some n => Err "the kernel does not begin with the zeroing prologue of its shared and local memory" (Some n)
        | None =>
            if negb (registers_ok (k_regs K)
                       ([(zero_index, zero_registers); (zero_address, zero_registers);
                         (zero_value, zero_registers); (zero_predicate, ("%opgate_p", 1%nat, None))] ++
                        if null dynamic then []
                        else [(zero_count, dynamic_registers); (zero_element, dynamic_registers)])) then
              Err "the zeroing prologue's registers are not declared once, %opgate_z<3> of 32 bits, the predicate %opgate_p, and for its loops %opgate_d<2> of 32 bits" None
            else if negb (prologue_labels_ok K Mf (List.length p)) then
              Err "a label inside the zeroing prologue other than those of its loops" None
            else if prologue_entered K Mf (List.length p) then
              Err "a branch into the zeroing prologue" None
            else Ok tt
        end
    end
  end.

(* ------------------------------------------------------------------ *)
(* Registers are written before they are read.                         *)

(* A register that an instruction has not written holds what the hardware
   left in it, which may be another tenant's data. The gate refuses a
   kernel, with its device functions inlined, in which an instruction may
   read a declared register that an instruction does not write on every
   path from the kernel's start: a guarded instruction may leave its
   destinations as they were. The carry flag, which .cc instructions write
   and addc, subc and madc read, counts as a register. Trusted: that the
   instructions that read a register are those that name it, as a source
   operand, as an address's base, or as a guard. StoreLink.v then gives each
   register that a launch value does not fill the value 0 at launch. *)

Definition carry_flag : string := "carry flag".

(* The names an operand gives: a register, a negated predicate, the
   elements of a vector, an address's base, a tensor map and its
   coordinates, and each name of a pair p|q. *)
Fixpoint operand_names (o : popnd) : list string :=
  match o with
  | PReg r | PNot r => [r]
  | PVec xs => flat_map operand_names xs
  | PAddr (Some r) _ => [r]
  | PTensor m cs => m :: flat_map operand_names cs
  | PText s => split_on "|"%char s ""
  | _ => []
  end.

(* The instructions whose operands are all read. *)
Definition writes_none (mn : string) : bool :=
  sin mn ["bra"; "brx"; "ret"; "exit"; "trap"; "brkpt"; "bar"; "barrier"; "membar"; "fence";
          "discard"; "st"; "red"; "frame$"; "prefetch"; "prefetchu"; "call"].

Definition read_names (i : pinstr) : list string :=
  let '(PI p mn sf ops) := i in
  (match p with Some (_, r) => [r] | None => [] end) ++
  (if writes_none mn then flat_map operand_names ops
   else match ops with
        | (PAddr _ _ as o) :: rest => flat_map operand_names (o :: rest)
        | _ :: rest => flat_map operand_names rest
        | [] => []
        end) ++
  (if sin mn ["addc"; "subc"; "madc"] then [carry_flag] else []).

(* The names an instruction writes on every run of it. *)
Definition written_names (i : pinstr) : list string :=
  let '(PI p mn sf ops) := i in
  match p with
  | Some _ => []
  | None =>
      (if writes_none mn then []
       else match ops with
            | PAddr _ _ :: _ | [] => []
            | o :: _ => operand_names o
            end) ++
      (if sin "cc" sf then [carry_flag] else [])
  end.

Definition label_position (K : pkernel) (l : string) : list nat :=
  match find (fun e => String.eqb (fst e) l) (k_labels K) with
  | Some (_, n) => [N.to_nat n]
  | None => []
  end.

(* The positions control may reach after instruction i at position j; the
   position past the last instruction is the kernel's end. *)
Definition next_positions (K : pkernel) (j : nat) (i : pinstr) : list nat :=
  let '(PI p mn _ ops) := i in
  let fall := match p with Some _ => [S j] | None => [] end in
  if String.eqb mn "bra" then
    match ops with [PLabel l] => label_position K l ++ fall | _ => fall end
  else if String.eqb mn "brx" then
    match ops with
    | [_; PLabel l] =>
        match find (fun e => String.eqb (fst e) l) (k_branch_targets K) with
        | Some (_, ls) => flat_map (label_position K) ls ++ fall
        | None => fall
        end
    | _ => fall
    end
  else if sin mn ["ret"; "exit"; "trap"] then fall
  else [S j].

Module RS := MSetPositive.PositiveSet.

(* The names that the analysis counts, numbered from 1 in the order in which
   the instructions name them: the declared registers and the carry flag. A
   name that is not counted, such as a special register, maps to None. *)
Definition register_numbers (K : pkernel) : PM.t (option positive) :=
  fst (fold_left (fun acc r =>
         let '(m, next) := acc in
         match PM.find (key r) m with
         | Some _ => acc
         | None =>
             if String.eqb r carry_flag || existsb (fun d => names d r) (k_regs K)
             then (PM.add (key r) (Some next) m, Pos.succ next)
             else (PM.add (key r) None m, next)
         end)
       (flat_map (fun i => app (read_names i) (written_names i)) (k_insts K)) (PM.empty _, 1%positive)).

Definition numbered (numbers : PM.t (option positive)) (rs : list string) : list positive :=
  flat_map (fun r => match PM.find (key r) numbers with Some (Some k) => [k] | _ => []
                     end) rs.

(* An instruction that ends a block: control may leave it for a block start. *)
Definition transfers (i : pinstr) : bool :=
  let '(PI _ mn _ _) := i in sin mn ["bra"; "brx"; "ret"; "exit"; "trap"].

(* The positions where a block starts: the start, every label, and the
   position after each instruction that transfers control. *)
Definition block_starts (K : pkernel) : RS.t :=
  fst (fold_left (fun acc i =>
         let '(s, j) := acc in
         (if transfers i then RS.add (Pos.of_succ_nat (S j)) s else s, S j))
       (k_insts K)
       (fold_left (fun s l => RS.add (Pos.of_succ_nat (N.to_nat (snd l))) s) (k_labels K)
                  (RS.singleton 1%positive), 0%nat)).

(* A block: its start, the numbers that each of its instructions reads and
   writes with the instruction's position, the numbers that the block
   writes, and the positions that control may reach after it. *)
Record rblock := {
  rb_start : nat;
  rb_steps : list (nat * list positive * list positive);
  rb_writes : RS.t;
  rb_next : list nat
}.

(* The blocks in order. *)
Definition register_blocks (K : pkernel) : list rblock :=
  let numbers := register_numbers K in
  let starts := block_starts K in
  let n := List.length (k_insts K) in
  let close := fun (start : nat) (steps : list (nat * list positive * list positive)) (next : list nat) =>
    {| rb_start := start; rb_steps := rev steps;
       rb_writes := fold_left (fun s e => fold_left (fun s k => RS.add k s) (snd e) s) steps RS.empty;
       rb_next := next |} in
  let '(blocks, _, _, _) :=
    fold_left (fun acc i =>
      let '(blocks, start, steps, j) := acc in
      let step := (j, numbered numbers (read_names i), numbered numbers (written_names i)) in
      if RS.mem (Pos.of_succ_nat (S j)) starts || Nat.eqb (S j) n
      then (close start (step :: steps) (next_positions K j i) :: blocks, S j, [], S j)
      else (blocks, start, step :: steps, S j))
    (k_insts K) ([], 0%nat, [], 0%nat) in
  rev blocks.

(* One pass over the blocks in order: each block that a path from the start
   reaches passes the numbers written before it and in it to the blocks that
   follow it, which keep those that every path gives them; and whether a
   block's numbers changed. *)
Definition written_pass (blocks : list rblock) (before : PM.t RS.t) : PM.t RS.t * bool :=
  fold_left (fun acc b =>
    let '(m, changed) := acc in
    match PM.find (Pos.of_succ_nat (rb_start b)) m with
    | None => acc
    | Some here =>
        let after := RS.union here (rb_writes b) in
        fold_left (fun acc k =>
          let '(m, changed) := acc in
          match PM.find (Pos.of_succ_nat k) m with
          | None => (PM.add (Pos.of_succ_nat k) after m, true)
          | Some old =>
              let joined := RS.inter old after in
              if RS.equal joined old then acc else (PM.add (Pos.of_succ_nat k) joined m, true)
          end) (rb_next b) (m, changed)
    end) blocks (before, false).

(* The numbers written on every path to each block that a path from the
   start reaches, after passes until none changes. *)
Fixpoint written_before (fuel : nat) (blocks : list rblock) (before : PM.t RS.t) : option (PM.t RS.t) :=
  match fuel with
  | O => None
  | S fuel =>
      let '(after, changed) := written_pass blocks before in
      if changed then written_before fuel blocks after else Some after
  end.

(* The position of the first instruction of a block that reads a number that
   the instructions before it do not write, from the numbers written before
   the block, and the number. *)
Fixpoint first_unwritten (here : RS.t) (steps : list (nat * list positive * list positive))
    : option (nat * positive) :=
  match steps with
  | [] => None
  | (j, reads, writes) :: rest =>
      match find (fun k => negb (RS.mem k here)) reads with
      | Some k => Some (j, k)
      | None => first_unwritten (fold_left (fun s k => RS.add k s) writes here) rest
      end
  end.

(* Every declared register that an instruction reads, and the carry flag,
   an instruction writes before it on every path from the start. A gen and
   kill problem reaches its fixed point within one pass more than the
   blocks on a path, so a pass for each block and two more suffice. A
   refusal names the register, under its name in the inlined kernel, and
   the instruction. *)
Definition defined_check (K : pkernel) (org : list place) : res unit :=
  let blocks := register_blocks K in
  match written_before (List.length blocks + 2) blocks (PM.add 1%positive RS.empty (PM.empty _)) with
  | None => Err "the analysis of the registers written before each instruction does not reach a fixed point" None
  | Some before =>
      match fold_left (fun found b =>
              match found with
              | Some _ => found
              | None =>
                  match PM.find (Pos.of_succ_nat (rb_start b)) before with
                  | Some here => first_unwritten here (rb_steps b)
                  | None => None
                  end
              end) blocks None with
      | Some (j, k) =>
          let numbers := register_numbers K in
          let named := fun r => match PM.find (key r) numbers with
                                | Some (Some k') => Pos.eqb k k'
                                | _ => false
                                end in
          let name := match find named (read_names (nth j (k_insts K) (PI None "" [] []))) with
                      | Some r => r
                      | None => ""
                      end in
          Err ("an instruction may read the register " ++ name ++ " before an instruction writes it")
              (Some (fst (nth j org (0%N, []))))
      | None => Ok tt
      end
  end.

(* ------------------------------------------------------------------ *)
(* Verification.                                                         *)

Definition init : tstate :=
  {| ts_ids := PM.empty reg; ts_names := []; ts_widths := PM.empty nat; ts_next := 1%positive;
     ts_at := None; ts_fr := [] |}.

(* Pure selection results outside the safety dependencies do not need value
   precision when no writable binding declares indexed contents. This includes values
   used only in ordinary store payloads and unused modeled results.
   Backward dependency closures retain address/guard/local uses,
   and selection registers with several definitions. This only selects facts to
   forget: transfer soundness does not assume this classification correct. *)
Definition payload_op_regs := opnd_inputs.
Definition payload_required_reads := phase_required_reads.

(* Dependencies of a pure result. Definitions of the same register are
   combined: overwriting a reduction accumulator does not turn its data
   into a safety obligation, while every definition remains reachable. *)
Definition payload_dependencies (i : instr) : list reg :=
  match i with
  | IMov _ _ _ | IImm _ _ _ | IBin _ _ _ _ _ _ | IMad _ _ _ _ _ _
  | IWide _ _ _ _ _ _ | ICvt _ _ _ _ _ _ | IBfe _ _ _ _ _ _
  | IDiv _ _ _ _ _ _ _ | ISetp _ _ _ _ _ _ _ | IAndp _ _ _ _ | IOrp _ _ _ _
  | IXorp _ _ _ _ | IMinMax _ _ _ _ _ _ _ | ISelp _ _ _ _ _ _ =>
      payload_required_reads i
  | _ => []
  end.

(* These uses always retain precision, independently of whether an output
   value is needed. Ordinary direct stores do not require a payload bound;
   local cells and all other memory operations remain conservative roots. *)
Definition payload_safety_roots (i : instr) : list reg :=
  let inputs := match i with
    | ILoad _ _ _ _ a _ | ILdS _ _ _ a _
    | ILdL _ _ _ a _ _ | IShA _ _ a _ _ _ => [a]
    | IMem _ KStore false a _ _ _ _ [] => [a]
    | IMem _ _ _ a _ _ v _ _ | IStL _ a _ _ v _ =>
        a :: match v with Some o => payload_op_regs o | None => [] end
    | _ => []
    end in
  match guard_of i with Some (_, p) => p :: inputs | None => inputs end.

Definition payload_reach := register_reach.
Definition payload_closure := register_closure.

Definition payload_candidates (F : lfacts) (prog : program) : PM.t unit :=
  if negb (short_any (fun i => match i with ISelp _ _ _ _ _ _ => true | _ => false end)
                    (List.concat prog)) then PM.empty unit else
  if existsb (fun x => match sp_index (snd x), sp_dir (snd x) with
             | Some _, DWrite | Some _, DReadWrite => true
             | _, _ => false
             end) (lf_roles F)
  then PM.empty unit else
  let '(required, graph, written, repeated, selected) :=
    fold_left (fun acc i =>
      let '(required, graph, written, repeated, selected) := acc in
      let required := List.app (payload_safety_roots i) required in
      let graph := match dest i with
        | Some d =>
            let old := match PM.find d graph with Some rs => rs | None => [] end in
            PM.add d (List.app (payload_dependencies i) old) graph
        | None => graph
        end in
      let selected := match i with
        | ISelp _ _ d _ _ _ => PM.add d tt selected
        | _ => selected
        end in
      let '(written, repeated) :=
        fold_left (fun acc r =>
          let '(written, repeated) := acc in
          (PM.add r tt written,
           if PM.mem r written then PM.add r tt repeated else repeated))
          (comparison_writes i) (written, repeated) in
      (required, graph, written, repeated, selected))
      (List.concat prog)
      ([], PM.empty (list reg), PM.empty unit, PM.empty unit, PM.empty unit) in
  let required := payload_closure graph required in
  PM.fold (fun r _ out =>
    if PM.mem r required || PM.mem r repeated ||
       negb (PM.mem r written)
    then out else PM.add r tt out) selected (PM.empty unit).

Definition payload_facts (F : lfacts) (prog : program) : lfacts :=
  {| lf_widths := lf_widths F; lf_fixed := lf_fixed F; lf_bounds := lf_bounds F;
     lf_aligns := lf_aligns F; lf_roles := lf_roles F;
     lf_shared_window := lf_shared_window F; lf_local_window := lf_local_window F;
     lf_shared := lf_shared F; lf_locals := lf_locals F; lf_cells := lf_cells F;
     lf_payloads := payload_candidates F prog;
     lf_derived := lf_derived F; lf_names := lf_names F |}.

Require Import OpGate.GateLoop.
Require Import OpGate.GateContext.
Require Import OpGate.GateEnds.
Require Import OpGate.GateShared.

(* Expanded positions distinguish instructions in the same call frame and
   repeated expansions of a helper. Frames and raw diagnostic origins are
   unchanged. This is metadata: no instruction or precondition is removed. *)
Fixpoint expanded_places_from (n : N) (org : list place) : list place :=
 match org with
 | [] => []
 | (_, fr) :: rest => (n, fr) :: expanded_places_from (N.succ n) rest
 end.
Definition expanded_places (org : list place) : list place :=
 expanded_places_from 0 org.

Lemma expanded_places_length : forall n org,
 List.length (expanded_places_from n org) = List.length org.
Proof. intros n org; revert n; induction org as [|[o fr] rest IH]; intros n; cbn; auto. Qed.

Lemma expanded_places_nth : forall org n j o fr,
 nth_error org j = Some (o,fr) ->
 nth_error (expanded_places_from n org) j = Some ((n + N.of_nat j)%N,fr).
Proof.
 intros org; induction org as [|[head frame] rest IH]; intros n [|j] o fr H; cbn in H; try discriminate.
 - injection H as <- <-. cbn. rewrite N.add_0_r. reflexivity.
 - cbn. rewrite (IH (N.succ n) j o fr H). f_equal. f_equal. lia.
Qed.

Lemma expanded_places_in_bounds : forall org n o fr,
 In (o,fr) (expanded_places_from n org) -> (n <= o)%N.
Proof.
 intros org; induction org as [|[head frame] rest IH]; intros n o fr H; cbn in H.
 - contradiction.
 - destruct H as [H|H]; [injection H as <- <-; lia|].
   specialize (IH (N.succ n) o fr H). lia.
Qed.

Lemma expanded_places_unique : forall org n,
 NoDup (map fst (expanded_places_from n org)).
Proof.
 intros org; induction org as [|[head frame] rest IH]; intros n; cbn.
 - constructor.
 - constructor; [|apply IH]. intro H.
   apply in_map_iff in H. destruct H as [[o fr] [He Hin]]. cbn in He. subst o.
   pose proof (expanded_places_in_bounds rest (N.succ n) n fr Hin). lia.
Qed.

(* Each blocking event keeps one exact expanded-source origin. A literal
   mask has one result instruction; a register mask retains its complete
   independently checked precondition span. Ambiguous origins refuse. *)
Definition source_barriers (K:pkernel) (org:list place) : list N :=
 map (fun x => fst (fst x))
 (filter (fun x => let 'PI _ mn _ _ := snd x in String.eqb mn "bar")
   (combine org (k_insts K))).
Definition source_events (K:pkernel) (org:list place) : list N :=
 map (fun x => fst (fst x))
 (filter (fun x => let 'PI _ mn _ _ := snd x in
   sin mn ["bar";"shfl";"vote";"match";"redux"])
   (combine org (k_insts K))).
Definition checked_sync_event_source (i:pinstr) : bool :=
 let 'PI pred mn _ args := i in
 match pred with
 | Some _=>false
 | None=>sin mn ["shfl";"vote";"match";"redux"] &&
   match rev args with PReg _::_=>true|mask::_=>sync_mask mask|_=>false end
 end.

Inductive event_request :=
| EBarrier | ELiteral | ERegister (mask:reg) (fr:frame).

(* Resolve a register mask from the actual translator map, and require one
   exact expanded source instruction for this origin. *)
(* An injective origin key indexes complete ordered lists. Processing the tail
   first retains original order and every duplicate, including ambiguous origins. *)
Definition origin_lookup {A:Type} (table:PM.t (list A)) (origin:N) : list A :=
 match PM.find (nkey origin) table with Some xs=>xs|None=>[] end.
Fixpoint origin_index {A:Type} (items:list (N*A)) : PM.t (list A) :=
 match items with
 | []=>PM.empty (list A)
 | (origin,x)::rest=>let table:=origin_index rest in
   PM.add (nkey origin) (x::origin_lookup table origin) table
 end.
Lemma origin_nkey_injective : forall a b,nkey a=nkey b -> a=b.
Proof.
 intros a b H. unfold nkey in H.
 apply (f_equal N.pos) in H. rewrite !N.succ_pos_spec in H. lia.
Qed.
Theorem origin_index_exact : forall A (items:list(N*A)) origin,
 origin_lookup (origin_index items) origin =
 flat_map (fun x=>if N.eqb (fst x) origin then [snd x] else []) items.
Proof.
 intros A items. induction items as [|[o x] rest IH];intros origin.
 - unfold origin_lookup;cbn [origin_index];rewrite PM.gempty;reflexivity.
 - cbn [origin_index flat_map fst snd]. unfold origin_lookup at 1.
   destruct (N.eqb o origin) eqn:H.
   + apply N.eqb_eq in H;subst o. rewrite PM.gss. cbn. rewrite IH. reflexivity.
   + rewrite PM.gso.
     * exact (IH origin).
     * intro He;apply origin_nkey_injective in He. subst o. rewrite N.eqb_refl in H;discriminate.
Qed.
Lemma origin_index_keyed : forall A (keyof:A->N) xs origin,
 origin_lookup (origin_index (map (fun x=>(keyof x,x)) xs)) origin =
 filter (fun x=>N.eqb (keyof x) origin) xs.
Proof.
 intros A keyof xs origin. rewrite origin_index_exact.
 induction xs as [|x xs IH];cbn;[reflexivity|].
 destruct (N.eqb (keyof x) origin);cbn;rewrite IH;reflexivity.
Qed.

Definition source_request_items (ids:PM.t reg) (items:list(place*pinstr)) : option event_request :=
 match items with
 | [((_,fr),PI None mn _ args)]=>
   if String.eqb mn "bar" then Some EBarrier else
   if sin mn ["shfl";"vote";"match";"redux"] then
     match rev args with
     | PReg name::_=>match PM.find (key name) ids with
                     | Some r=>Some (ERegister r fr)|None=>None end
     | mask::_=>if sync_mask mask then Some ELiteral else None
     | []=>None end
   else None
 | _=>None end.

Definition source_event_request (K:pkernel) (org:list place) (ids:PM.t reg)
 (origin:N) : option event_request :=
 source_request_items ids
  (filter (fun x=>N.eqb (fst(fst x)) origin) (combine org (k_insts K))).

(* Enumerate blocks and instructions once per origin. The exact theorem
   retains the former indexed-lookup list, including order and duplicates. *)
Lemma origin_enumeration_exact : forall (A:Type) (xs:list A) (d:A),
 combine (seq 0 (List.length xs)) xs =
 map (fun i=>(i,nth i xs d)) (seq 0 (List.length xs)).
Proof.
 intros A xs d.
 apply nth_ext with (d:=(0%nat,d)) (d':=(0%nat,nth 0 xs d)).
 - rewrite combine_length,map_length,!seq_length,Nat.min_id. reflexivity.
 - intros n Hn. rewrite combine_length,seq_length,Nat.min_id in Hn.
   rewrite combine_nth by apply seq_length.
   rewrite (map_nth (fun i=>(i,nth i xs d)) (seq 0 (List.length xs)) 0%nat n).
   rewrite seq_nth by exact Hn. reflexivity.
Qed.

Definition origin_hits (pp:list (list placed)) (origin:N) : list (pc*instr) :=
 flat_map (fun block=>let '(b,body):=block in
  flat_map (fun item=>let '(i,(ins,o)):=item in
   if N.eqb o origin then [((b,i),ins)] else [])
   (combine (seq 0 (List.length body)) body))
  (combine (seq 0 (List.length pp)) pp).

Theorem origin_hits_exact : forall pp origin,
 origin_hits pp origin =
 flat_map (fun b => let body:=nth b pp [] in
   flat_map (fun i=>match nth_error body i with
    | Some (ins,o)=>if N.eqb o origin then [((b,i),ins)] else []
    | None=>[] end) (seq 0 (List.length body))) (seq 0 (List.length pp)).
Proof.
 intros pp origin. unfold origin_hits.
 rewrite (origin_enumeration_exact _ pp []).
 rewrite !flat_map_concat_map,map_map. f_equal.
 apply map_ext_in. intros b Hb. cbn.
 pose proof (origin_enumeration_exact placed (nth b pp []) (INop,0%N)) as Henum.
 unfold placed in *. rewrite Henum.
 rewrite !flat_map_concat_map,map_map. f_equal.
 apply map_ext_in. intros i Hi. cbn.
 apply in_seq in Hi.
 rewrite (nth_error_nth' (nth b pp []) (INop,0%N)) by lia.
 destruct (nth i (nth b pp []) (INop,0%N));reflexivity.
Qed.

Definition origin_items (pp:list(list placed)) : list(N*(pc*instr)) :=
 flat_map (fun block=>let '(b,body):=block in
  map (fun item=>let '(i,(ins,o)):=item in (o,((b,i),ins)))
   (combine (seq 0 (List.length body)) body))
  (combine (seq 0 (List.length pp)) pp).
Lemma origin_flat_map_compose : forall A B C (f:B->list C) (g:A->list B) xs,
 flat_map f (flat_map g xs)=flat_map (fun x=>flat_map f (g x)) xs.
Proof. intros A B C f g xs;induction xs;cbn;[reflexivity|]. rewrite flat_map_app,IHxs;reflexivity. Qed.
Lemma origin_flat_map_map : forall A B C (f:B->list C) (g:A->B) xs,
 flat_map f (map g xs)=flat_map (fun x=>f(g x)) xs.
Proof. intros A B C f g xs;induction xs;cbn;[reflexivity|]. rewrite IHxs;reflexivity. Qed.
Theorem origin_hits_indexed_exact : forall pp origin,
 origin_lookup (origin_index (origin_items pp)) origin = origin_hits pp origin.
Proof.
 intros pp origin. rewrite origin_index_exact. unfold origin_items,origin_hits.
 rewrite origin_flat_map_compose.
 apply flat_map_ext. intros [b body]. cbn.
 rewrite origin_flat_map_map. apply flat_map_ext.
 intros [i [ins o]]. reflexivity.
Qed.


Definition span_next (a b:pc) : bool :=
 Nat.eqb (fst a) (fst b) && Nat.eqb (S (snd a)) (snd b).
Fixpoint event_frame_eq (a b:frame) : bool :=
 match a,b with [],[]=>true|x::xs,y::ys=>N.eqb x y && event_frame_eq xs ys|_,_=>false end.

(* This is exactly require32 CEq mask full: the inequality predicate,
   its zero-address witness, its original guarded impossibility check,
   and the untouched collective result. No check is removed or replaced. *)
Definition checked_mask_span (mask:reg) (fr:frame) (hits:list (pc*instr)) : option pc :=
 match hits with
 | [(p0,ISetp None CNe w false q a (OImm full));
    (p1,IImm None z zero);
    (p2,IMem (Some (false,q')) KStore false z' off width None fr' []);
    (event,IDef None _)]=>
   if Nat.eqb w 32 && Pos.eqb mask a && Z.eqb full 4294967295 &&
      Z.eqb zero 0 && Pos.eqb q q' && Pos.eqb z z' &&
      Z.eqb off 0 && Z.eqb width 1 && event_frame_eq fr fr' &&
      negb (Pos.eqb q z || Pos.eqb q mask || Pos.eqb z mask) &&
      span_next p0 p1 && span_next p1 p2 && span_next p2 event
   then Some event else None
 | _=>None end.

Definition placed_event_hits (request:event_request) (hits:list(pc*instr)) : option pc :=
 match request with
 | EBarrier=>match hits with [(pos,INop)]=>Some pos|_=>None end
 | ELiteral=>match hits with [(pos,IDef None _)]=>Some pos|_=>None end
 | ERegister mask fr=>
   match hits with
   (* snapshot_comparison may preserve an overwritten source operand first.
      Recognize its exact one-copy shape and original mask; never skip it. *)
   | (p,IMov None saved original)::(next,ins)::rest=>
       if Pos.eqb original mask && negb (Pos.eqb saved mask) && span_next p next
       then checked_mask_span saved fr ((next,ins)::rest) else None
   | _=>checked_mask_span mask fr hits
   end
 end.

Definition placed_event (request:event_request) (pp:list(list placed)) (origin:N)
 : option pc := placed_event_hits request (origin_hits pp origin).
Definition source_placed_event (K:pkernel) (org:list place) (ids:PM.t reg)
 (pp:list (list placed)) (origin:N) : option pc :=
 match source_event_request K org ids origin with
 | Some request=>placed_event request pp origin|None=>None end.
Fixpoint origin_collect_events (lookup:N->option pc) (origins:list N) : option(list pc) :=
 match origins with
 | []=>Some []
 | origin::rest=>match lookup origin,origin_collect_events lookup rest with
   | Some event,Some events=>Some(event::events)|_,_=>None end
 end.
Lemma origin_collect_events_ext : forall f g origins,
 (forall origin,f origin=g origin) -> origin_collect_events f origins=origin_collect_events g origins.
Proof.
 intros f g origins H;induction origins;cbn;[reflexivity|]. rewrite H,IHorigins;reflexivity.
Qed.
Definition placed_events (K:pkernel) (org:list place) (ids:PM.t reg)
 (pp:list(list placed)) (origins:list N) : option(list pc) :=
 match origins with
 | []=>Some []
 | _=>let sources:=origin_index (map(fun x=>(fst(fst x),x)) (combine org (k_insts K))) in
      let hits:=origin_index (origin_items pp) in
      origin_collect_events (fun origin=>
        match source_request_items ids (origin_lookup sources origin) with
        | Some request=>placed_event_hits request (origin_lookup hits origin)
        | None=>None end) origins
 end.
Theorem placed_events_exact : forall K org ids pp origins,
 placed_events K org ids pp origins =
 origin_collect_events (source_placed_event K org ids pp) origins.
Proof.
 intros K org ids pp origins. destruct origins as [|origin origins];[reflexivity|].
 unfold placed_events. apply origin_collect_events_ext. intros o.
 rewrite origin_index_keyed,origin_hits_indexed_exact.
 unfold source_placed_event,source_event_request,placed_event. reflexivity.
Qed.
Definition phase_source_ok (K:pkernel) (org:list place) (origin:N) : bool :=
 Nat.eqb (List.length org) (List.length (k_insts K)) &&
 Nat.eqb (List.length (filter (fun p=>N.eqb (fst p) origin) org)) 1 &&
 forallb (fun i => let 'PI pred mn sf _ := i in
   negb (sin mn ["barrier";"mbarrier"])
   && (if sin mn ["shfl";"vote";"match";"redux"] then checked_sync_event_source i else true)
   && negb (sin "uni" sf && match pred with Some _=>true|None=>false end))
   (k_insts K).
(* The first source check establishes the shared instruction conditions.
   Remaining sites need only their own unique source origin. Empty site lists
   retain short_all's true result without imposing any source conditions. *)
Definition phase_sources_ok (K:pkernel) (org:list place) (origins:list N) : bool :=
 match origins with
 | []=>true
 | origin::rest=>if phase_source_ok K org origin then
   short_all (fun other=>Nat.eqb
     (List.length (filter (fun p=>N.eqb (fst p) other) org)) 1) rest
   else false
 end.
Lemma phase_source_unique : forall K org first other,
 phase_source_ok K org first=true ->
 phase_source_ok K org other =
 Nat.eqb (List.length (filter (fun p=>N.eqb (fst p) other) org)) 1.
Proof.
 intros K org first other H. unfold phase_source_ok in *.
 repeat rewrite andb_true_iff in H. destruct H as [[Hlength Hfirst] Hbody].
 rewrite Hlength,Hbody. cbn. apply andb_true_r.
Qed.
Theorem phase_sources_exact : forall K org origins,
 phase_sources_ok K org origins = short_all (phase_source_ok K org) origins.
Proof.
 intros K org [|first rest]; [reflexivity|].
 cbn [phase_sources_ok short_all].
 destruct (phase_source_ok K org first) eqn:Hfirst; [|reflexivity].
 cbn. induction rest as [|other rest IH]; [reflexivity|].
 cbn [short_all]. rewrite (phase_source_unique K org first other Hfirst),IH.
 destruct (Nat.eqb (List.length (filter (fun p=>N.eqb (fst p) other) org)) 1);
 reflexivity.
Qed.

Inductive category :=
| CSize (k : nat)
| CParam (i : nat) (h : holding)
| CTid (d : nat) | CNtid (d : nat) | CCtaid (d : nat) | CNctaid (d : nat)
| CLane
| CPlatform.

Definition dimension (n prefix : string) : option nat :=
  if String.eqb n (prefix ++ ".x") then Some 0%nat
  else if String.eqb n (prefix ++ ".y") then Some 1%nat
  else if String.eqb n (prefix ++ ".z") then Some 2%nat
  else None.

Definition category_of (Mf : binding) (n : string) : option category :=
  match find (fun k => String.eqb n ("size:" ++ nstr (N.of_nat k))) (seq 0 (List.length (b_sizes Mf))) with
  | Some k => Some (CSize k)
  | None =>
  match find (fun x => String.eqb n ("param:" ++ fst (snd x)))
             (combine (seq 0 (List.length (b_params Mf))) (b_params Mf)) with
  | Some (i, (_, h)) => Some (CParam i h)
  | None =>
  match dimension n "%tid", dimension n "%ntid", dimension n "%ctaid", dimension n "%nctaid" with
  | Some d, _, _, _ => Some (CTid d)
  | _, Some d, _, _ => Some (CNtid d)
  | _, _, Some d, _ => Some (CCtaid d)
  | _, _, _, Some d => Some (CNctaid d)
  | None, None, None, None =>
      if String.eqb n "%laneid" then Some CLane
      else if String.prefix "window:" n || String.prefix "shared:" n || String.prefix "local:" n ||
              String.prefix "%clusterid." n
      then Some CPlatform else None
  end end end.

Definition name_of (F : lfacts) (r : reg) : option string :=
  match find (fun e => Pos.eqb (snd e) r) (lf_names F) with Some (n, _) => Some n | None => None end.

Definition reg_named (F : lfacts) (n : string) : option reg :=
  match find (fun e => String.eqb (fst e) n) (lf_names F) with Some (_, r) => Some r | None => None end.

Definition category_reg (F : lfacts) (Mf : binding) (r : reg) : option category :=
  match name_of F r with Some n => category_of Mf n | None => None end.

Definition block_dim (Mf : binding) (d : nat) : Z :=
  let '(x, y, z) := b_block Mf in match d with 0%nat => x | 1%nat => y | _ => z end.

(* The binding's grid maximum in dimension d. *)
Definition grid_dim (Mf : binding) (d : nat) : Z :=
  let '(x, y, z) := b_grid Mf in match d with 0%nat => x | 1%nat => y | _ => z end.

(* The scope selector is a checked physical shape and an actual-IR control
   shape. It selects one proof procedure before analysis; a failed selected
   proof never retries another procedure. *)
Definition warp_launch_shape (F:lfacts) (Mf:binding) : option(reg*nat) :=
 let '(bx,by_,bz):=b_block Mf in
 if Z.ltb 0 bx && Z.leb bx 1024 && Z.eqb by_ 1 && Z.eqb bz 1 &&
    Z.eqb (bx mod 32) 0 then
 match reg_named F "%tid.x" with
 | Some r=>match category_reg F Mf r,lookup (lf_bounds F) r with
   | Some (CTid O),Some (lo,hi)=>
     if is_fixed F r && Z.eqb lo 0 && Z.eqb hi (bx-1) && fits 0 (bx-1) (wdn F r)
     then Some(r,Z.to_nat(bx/32)) else None
   | _,_=>None end
 | None=>None end
 else None.

Lemma warp_launch_shape_sound : forall F Mf r count,
 warp_launch_shape F Mf=Some(r,count) ->
 0<block_dim Mf 0 /\ block_dim Mf 0<=1024 /\
 block_dim Mf 1=1 /\ block_dim Mf 2=1 /\ block_dim Mf 0 mod 32=0 /\
 count=Z.to_nat(block_dim Mf 0/32) /\
 category_reg F Mf r=Some(CTid 0) /\
 fits 0 (block_dim Mf 0-1) (wdn F r)=true.
Proof.
 intros F Mf r count H. unfold warp_launch_shape in H.
 destruct (b_block Mf) as [[bx by_] bz] eqn:Hblock.
 destruct (Z.ltb 0 bx && Z.leb bx 1024 && Z.eqb by_ 1 && Z.eqb bz 1 &&
   Z.eqb (bx mod 32) 0) eqn:Hshape;[|discriminate].
 destruct (reg_named F "%tid.x") as [r'|];[|discriminate].
 destruct (category_reg F Mf r') as [cat|] eqn:Hcat;[|discriminate].
 destruct cat;try discriminate. destruct d;try discriminate.
 destruct (lookup (lf_bounds F) r') as [[lo hi]|];[|discriminate].
 destruct (is_fixed F r' && Z.eqb lo 0 && Z.eqb hi (bx-1) && fits 0 (bx-1) (wdn F r')) eqn:Hrange;[|discriminate].
 injection H as <- <-. unfold block_dim. rewrite Hblock. cbn.
 rewrite !andb_true_iff,!Z.ltb_lt,!Z.leb_le,!Z.eqb_eq in Hshape.
 rewrite !andb_true_iff in Hrange. destruct Hshape as [[[[Hpos Hmax] Hy] Hz] Hmod].
 destruct Hrange as [[[Hfixed Hlo] Hhi] Hfits]. repeat split;assumption || reflexivity.
Qed.

Definition warp_optional_case (pp:list(list placed)) (events cta:list pc) : bool :=
 existsb (fun item=>let '(_,(pos,ins)):=item in
   match ins with
   | IBra (Some _) target=>existsb (fun event=>
       negb (occurrence_member cta event) && phase_after pos event &&
       phase_after event (target,0%nat)) events
   | _=>false end) (origin_items pp).

(* CTA agreement comes only from these named launch categories. Fixed
   thread/lane/platform registers are deliberately not CTA-uniform. *)
Definition cta_uniform_key F Mf r :=
 match category_reg F Mf r with
 |Some(CSize _)|Some(CParam _ _)|Some(CNtid _)|Some(CCtaid _)|Some(CNctaid _)=>true
 |_=>false end.

(* In a kernel without CTA barriers, an entry branch that leaves before every
   event selects the warp exit proof; any other optional warp event selects
   the per-warp words. *)
Definition front_fixed_event_check (Mf:binding) (K:pkernel) (org:list place)
 (ids:PM.t reg) (pp:list(list placed)) (F:lfacts) (events:list pc) : bool :=
 match warp_launch_shape F Mf with
 | Some(r,count)=>
   match placed_events K org ids pp (source_barriers K org) with
   | Some cta=>if warp_optional_case pp events cta then
       if warp_exit_selected (map(map fst)pp) events cta
       then warp_exit_check F (cta_uniform_key F Mf) (map(map fst)pp) events cta r count
       else match warp_scoped_check F (map(map fst)pp) events cta r count with
         | Some _=>true | None=>false end
     else fixed_progress_check F (map(map fst)pp) events
   | None=>false end
 | None=>fixed_progress_check F (map(map fst)pp) events
 end.

Definition front_fixed_events_safe (Mf:binding) (K:pkernel) (org:list place)
 (ids:PM.t reg) (pp:list(list placed)) (F:lfacts) (events:list pc) : Prop :=
 fixed_progress_safe F (map(map fst)pp) events \/
 (exists cta r count,
   placed_events K org ids pp (source_barriers K org)=Some cta /\
   warp_launch_shape F Mf=Some(r,count) /\
   scoped_schedule_safe F (map(map fst)pp) events cta r count) \/
 exists cta r count,
   placed_events K org ids pp (source_barriers K org)=Some cta /\
   warp_launch_shape F Mf=Some(r,count) /\
   warp_exit_schedule_safe F (cta_uniform_key F Mf) (map(map fst)pp) events cta r count.

Theorem front_fixed_event_check_sound : forall Mf K org ids pp F events,
 front_fixed_event_check Mf K org ids pp F events=true ->
 front_fixed_events_safe Mf K org ids pp F events.
Proof.
 intros Mf K org ids pp F events H. unfold front_fixed_event_check in H.
 unfold front_fixed_events_safe.
 destruct (warp_launch_shape F Mf) as [[r count]|] eqn:Hshape.
 - destruct (placed_events K org ids pp (source_barriers K org)) as [cta|] eqn:Hcta;[|discriminate].
   destruct (warp_optional_case pp events cta).
   + destruct (warp_exit_selected (map(map fst)pp) events cta).
     * right. right. exists cta,r,count. repeat split;try assumption.
       eapply warp_exit_check_sound;exact H.
     * destruct (warp_scoped_check F (map(map fst)pp) events cta r count) as [words|] eqn:Hwords;[|discriminate].
       right. left. exists cta,r,count. repeat split;try assumption.
       eapply warp_scoped_check_sound;exact Hwords.
   + left. apply fixed_progress_check_sound;exact H.
 - left. apply fixed_progress_check_sound;exact H.
Qed.

(* This selects a structural route before proposing/checking its certificate.
   Only an all-CTA source alphabet and a unique controller shape qualify. *)
Definition front_context_shape K org ids pp events :=
 match placed_events K org ids pp(source_barriers K org)with
 |Some cta=>if event_word_eqb cta events
   then context_shape_select(map(map fst)pp)events else None
 |None=>None end.
(* A mixed controller observes the full alphabet; the CTA subset is retained
   only as checked source correspondence. It is never used to filter a word. *)
Definition front_mixed_context_shape K org ids pp events :=
 match placed_events K org ids pp(source_barriers K org)with
 |Some cta=>if negb(Nat.eqb(List.length cta)0)&&negb(event_word_eqb cta events)
   then mixed_context_shape_select(map(map fst)pp)events else None
 |None=>None end.
(* These are selection conditions, not a claim that a lexical zero write
   executes on every path. The unchanged entry checker proves actual head
   counter zero, uniformity, ranges and complete event progress. *)
Definition front_mixed_zero_source prog shape:=
 let loop:=ms_loop shape in
 let writes:=filter(fun item=>phase_after(fst item)(ml_head loop,0%nat)&&
   existsb(Pos.eqb(ml_counter loop))(loop_writes(snd item)))(context_items prog)in
 negb(Nat.eqb(List.length writes)0)&&short_all(fun item=>
  match snd item with IImm None r z=>Pos.eqb r(ml_counter loop)&&Z.eqb z 0|_=>false end)writes.
Definition front_mixed_eligible F uniform prog events shape:=
 let loop:=ms_loop shape in
 if front_mixed_zero_source prog shape&&
    short_all(fun event=>negb(phase_after(ml_branch loop)event))events then
 let header:=uniform_states_run F(ms_header shape)(empty_state F)in
 let marks:=uniform_marks_run F(ms_header shape)uniform in
 match get header(ms_predicate shape)with
 |Some(FPred[k]true)=>
  if mixed_comparison_marked marks k&&uniform_marks_run F(ms_ready shape)marks(ml_limit loop)then
  match refine F(transfer F header(IBra(Some(ms_negated shape,ms_predicate shape))(ms_skip shape)))
    (ms_predicate shape)(ms_negated shape)with
  |Some entry=>let ready:=uniform_states_run F(ms_ready shape)entry in
   match cmp_range F ready(OReg(ml_limit loop))(Z.of_nat(ml_width loop))true with
   |Some(lo,hi)=>negb(Z.eqb lo hi)|None=>false end
  |None=>false end else false
 |_=>false end else false.
Definition front_context_route F uniform K org ids pp events
 :option(context_shape+mixed_context_shape):=
 match front_context_shape K org ids pp events with
 |Some shape=>Some(inl shape)
 |None=>match front_mixed_context_shape K org ids pp events with
   |Some shape=>if front_mixed_eligible F uniform(map(map fst)pp)events shape
      then Some(inr shape)else None
   |None=>None end
 end.
Definition front_event_check (Mf:binding) (K:pkernel) (org:list place)
 (ids:PM.t reg) (pp:list(list placed)) (F:lfacts) (events:list pc) : bool :=
 match front_context_route F(cta_uniform_key F Mf)K org ids pp events with
 |Some(inl shape)=>context_check F(cta_uniform_key F Mf)(map(map fst)pp)events shape
 |Some(inr shape)=>mixed_context_check F(cta_uniform_key F Mf)(map(map fst)pp)events shape
 |None=>front_fixed_event_check Mf K org ids pp F events
 end.
Definition front_legacy_events_safe (Mf:binding) (K:pkernel) (org:list place)
 (ids:PM.t reg) (pp:list(list placed)) (F:lfacts) (events:list pc) : Prop :=
 front_fixed_events_safe Mf K org ids pp F events \/
 exists cta,
 placed_events K org ids pp(source_barriers K org)=Some cta /\ cta=events /\
 context_schedule_safe F(cta_uniform_key F Mf)(map(map fst)pp)events.
Definition front_events_safe (Mf:binding) (K:pkernel) (org:list place)
 (ids:PM.t reg) (pp:list(list placed)) (F:lfacts) (events:list pc) : Prop :=
 front_legacy_events_safe Mf K org ids pp F events \/
 exists cta,placed_events K org ids pp(source_barriers K org)=Some cta /\
 cta<>[] /\event_word_eqb cta events=false /\
 context_schedule_safe F(cta_uniform_key F Mf)(map(map fst)pp)events.
Theorem front_event_check_sound : forall Mf K org ids pp F events,
 front_event_check Mf K org ids pp F events=true ->
 front_events_safe Mf K org ids pp F events.
Proof.
 intros Mf K org ids pp F events H. unfold front_event_check,front_context_route in H.
 destruct(front_context_shape K org ids pp events)as[shape|]eqn:Hshape.
 - left. right. unfold front_context_shape in Hshape.
   destruct(placed_events K org ids pp(source_barriers K org))as[cta|]eqn:Hcta;[|discriminate].
   destruct(event_word_eqb cta events)eqn:Hsame;[|discriminate].
   exists cta. split;[reflexivity|]. split.
   + now apply event_word_eqb_eq.
   + eapply context_check_sound;exact H.
 - destruct(front_mixed_context_shape K org ids pp events)as[shape|]eqn:Hmixed.
   + destruct(front_mixed_eligible F(cta_uniform_key F Mf)(map(map fst)pp)events shape)eqn:Heligible;
     [|left;left;now apply front_fixed_event_check_sound].
     right. unfold front_mixed_context_shape in Hmixed.
     destruct(placed_events K org ids pp(source_barriers K org))as[cta|]eqn:Hcta;[|discriminate].
     destruct(negb(Nat.eqb(List.length cta)0)&&negb(event_word_eqb cta events))eqn:Hgate;[|discriminate].
     apply andb_true_iff in Hgate. destruct Hgate as[Hnonempty Hdiff].
     apply negb_true_iff in Hdiff.
     exists cta. split;[reflexivity|]. split;[destruct cta;discriminate|].
     split;[exact Hdiff|]. eapply mixed_context_check_sound;exact H.
   + left. left. now apply front_fixed_event_check_sound.
Qed.
Lemma front_fixed_route_preserved : forall Mf K org ids pp F events,
 front_context_route F(cta_uniform_key F Mf)K org ids pp events=None ->
 front_event_check Mf K org ids pp F events=front_fixed_event_check Mf K org ids pp F events.
Proof. intros. unfold front_event_check. now rewrite H. Qed.
Lemma front_old_context_route_preserved : forall F uniform K org ids pp events shape,
 front_context_shape K org ids pp events=Some shape ->
 front_context_route F uniform K org ids pp events=Some(inl shape).
Proof. intros. unfold front_context_route. now rewrite H. Qed.
Lemma front_without_mixed_shape_preserved : forall Mf K org ids pp F events,
 front_mixed_context_shape K org ids pp events=None ->
 front_event_check Mf K org ids pp F events=
 match front_context_shape K org ids pp events with
 |Some shape=>context_check F(cta_uniform_key F Mf)(map(map fst)pp)events shape
 |None=>front_fixed_event_check Mf K org ids pp F events end.
Proof.
 intros. unfold front_event_check,front_context_route.
 destruct(front_context_shape K org ids pp events);[reflexivity|]. now rewrite H.
Qed.
Lemma front_without_mixed_capability_preserved : forall Mf K org ids pp F events,
 mixed_context_shape_select(map(map fst)pp)events=None ->
 front_event_check Mf K org ids pp F events=
 match front_context_shape K org ids pp events with
 |Some shape=>context_check F(cta_uniform_key F Mf)(map(map fst)pp)events shape
 |None=>front_fixed_event_check Mf K org ids pp F events end.
Proof.
 intros. apply front_without_mixed_shape_preserved. unfold front_mixed_context_shape. rewrite H.
 destruct(placed_events K org ids pp(source_barriers K org));[|reflexivity].
 destruct(negb(Nat.eqb(List.length l)0)&&negb(event_word_eqb l events));reflexivity.
Qed.
Lemma front_event_word_reflexive:forall events,event_word_eqb events events=true.
Proof.
 induction events as[|x rest IH];cbn[event_word_eqb];[reflexivity|].
 assert(H:phase_event x x=true)by(apply phase_event_eq;reflexivity). now rewrite H.
Qed.
Lemma front_all_cta_route_preserved : forall Mf K org ids pp F events,
 placed_events K org ids pp(source_barriers K org)=Some events ->
 front_event_check Mf K org ids pp F events=
 match front_context_shape K org ids pp events with
 |Some shape=>context_check F(cta_uniform_key F Mf)(map(map fst)pp)events shape
 |None=>front_fixed_event_check Mf K org ids pp F events end.
Proof.
 intros. apply front_without_mixed_shape_preserved. unfold front_mixed_context_shape.
 rewrite H,front_event_word_reflexive. now rewrite andb_false_r.
Qed.
Lemma front_no_cta_route_preserved : forall Mf K org ids pp F events,
 placed_events K org ids pp(source_barriers K org)=Some [] ->
 front_event_check Mf K org ids pp F events=
 match front_context_shape K org ids pp events with
 |Some shape=>context_check F(cta_uniform_key F Mf)(map(map fst)pp)events shape
 |None=>front_fixed_event_check Mf K org ids pp F events end.
Proof. intros. apply front_without_mixed_shape_preserved. unfold front_mixed_context_shape. now rewrite H. Qed.
Lemma front_mixed_ineligible_preserved : forall Mf K org ids pp F events shape,
 front_mixed_context_shape K org ids pp events=Some shape ->
 front_mixed_eligible F(cta_uniform_key F Mf)(map(map fst)pp)events shape=false ->
 front_event_check Mf K org ids pp F events=
 match front_context_shape K org ids pp events with
 |Some old=>context_check F(cta_uniform_key F Mf)(map(map fst)pp)events old
 |None=>front_fixed_event_check Mf K org ids pp F events end.
Proof.
 intros. unfold front_event_check,front_context_route.
 destruct(front_context_shape K org ids pp events);[reflexivity|]. now rewrite H,H0.
Qed.
Lemma front_mixed_refusal_final : forall Mf K org ids pp F events shape,
 front_context_route F(cta_uniform_key F Mf)K org ids pp events=Some(inr shape) ->
 mixed_context_check F(cta_uniform_key F Mf)(map(map fst)pp)events shape=false ->
 front_event_check Mf K org ids pp F events=false.
Proof. intros. unfold front_event_check. now rewrite H,H0. Qed.
(* Barrier-only straight-line bodies retain their finite structural check.
   Every mixed body, including a syntactically straight-line one, checks all
   waiting sites through the same ordered-event boundary. *)
Definition barrier_linear_case (K:pkernel) : bool :=
 linear_barrier_body (k_insts K) &&
 negb (existsb (fun i => let 'PI _ mn _ _ := i in
   sin mn ["shfl";"vote";"match";"redux"]) (k_insts K)).
(* The full-mask reductions and butterfly shuffles of the source, at the
   positions of their lowered results, with their source registers and the
   register of a member mask that is not a literal. The shared value check
   requires such a register to hold the full mask. *)
Definition collective_mask (ids:PM.t reg) (mask:popnd) : option (option reg) :=
 match mask with
 | PReg name => match PM.find (key name) ids with Some r => Some (Some r) | None => None end
 | _ => if sync_mask mask then Some None else None
 end.
Definition collective_of_source (ids:PM.t reg) (d:reg) (i:pinstr) : option collective :=
 match i with
 | PI None mn sf args =>
   if String.eqb mn "redux" then
     match sf, args with
     | ["sync"; _; _], [PReg _; PReg src; mask] =>
       match PM.find (key src) ids, collective_mask ids mask with
       | Some s, Some m => Some (CRedux d s m) | _, _ => None end
     | _, _ => None end
   else if String.eqb mn "shfl" then
     match sf, args with
     | ["sync"; mode; "b32"], [PReg _; PReg src; PImm lane; PImm c; mask] =>
       if String.eqb mode "bfly" && Z.eqb c 31 then
         match PM.find (key src) ids, collective_mask ids mask with
         | Some s, Some m => Some (CBfly d s lane m) | _, _ => None end
       else None
     | _, _ => None end
   else None
 | _ => None
 end.
Definition source_collectives (K:pkernel) (org:list place) (ids:PM.t reg) (pp:list(list placed))
 : list (pc * collective) :=
 let sources := origin_index (map (fun x => (fst (fst x), x)) (combine org (k_insts K))) in
 let hits := origin_index (origin_items pp) in
 flat_map (fun x => let '((origin, _), i) := x in
   let 'PI _ mn _ _ := i in
   if sin mn ["redux"; "shfl"] then
     match source_request_items ids (origin_lookup sources origin) with
     | Some request =>
       match placed_event_hits request (origin_lookup hits origin) with
       | Some p =>
         match fetch (map (map fst) pp) (fst p) (snd p) with
         | Some (IDef None [d]) =>
           match collective_of_source ids d i with Some c => [(p, c)] | None => [] end
         | _ => []
         end
       | None => []
       end
     | None => []
     end
   else []) (combine org (k_insts K)).

(* A shared variable's address is the same in every thread of a block. *)
Definition shared_variable_key (F:lfacts) (r:reg) : bool :=
 match name_of F r with Some nm => String.prefix "shared:" nm | None => false end.
Definition shared_uniform (F:lfacts) (Mf:binding) (r:reg) : bool :=
 cta_uniform_key F Mf r || shared_variable_key F r.

(* Every thread of a block of whole warps executes the same sequence of
   waiting sites, by the shared value check (GateShared.v). It is tried
   after the routes that need no fact about shared memory or collectives. *)
Definition front_shared_check (Mf:binding) (K:pkernel) (org:list place) (ids:PM.t reg)
 (pp:list(list placed)) (F:lfacts) (events:list pc) : bool :=
 match warp_launch_shape F Mf, placed_events K org ids pp (source_barriers K org) with
 | Some (tidr, count), Some cta =>
   shared_check F (shared_uniform F Mf) tidr (32 * count) (map (map fst) pp) cta events
     (source_collectives K org ids pp)
 | _, _ => false
 end.
Definition front_shared_safe (Mf:binding) (K:pkernel) (org:list place) (ids:PM.t reg)
 (pp:list(list placed)) (F:lfacts) (events:list pc) : Prop :=
 exists cta tidr count,
 placed_events K org ids pp (source_barriers K org) = Some cta /\
 warp_launch_shape F Mf = Some (tidr, count) /\
 shared_schedule_safe F (shared_uniform F Mf) tidr (32 * count) (map (map fst) pp) cta events
   (source_collectives K org ids pp).
Lemma front_shared_check_sound : forall Mf K org ids pp F events,
 front_shared_check Mf K org ids pp F events = true -> front_shared_safe Mf K org ids pp F events.
Proof.
 intros Mf K org ids pp F events H. unfold front_shared_check in H.
 destruct (warp_launch_shape F Mf) as [[tidr count]|] eqn:Hshape; [|discriminate].
 destruct (placed_events K org ids pp (source_barriers K org)) as [cta|] eqn:Hcta; [|discriminate].
 exists cta, tidr, count. split; [exact Hcta|]. split; [exact Hshape|].
 apply shared_check_sound. exact H.
Qed.

(*
   Non-linear kernels check one finite phase or an ordered event sequence
   with ranked loop summaries, or a finite occurrence word when actual IR
   can return across a static event. The latter proves a common complete
   word and an event-free suffix. These are structural cases, not retries.
   Every waiting site is an event: a CTA barrier, and a warp collective
   (shfl, vote, match, redux) also in a kernel without a CTA barrier, so
   that every thread that its member mask names reaches it. *)
Definition barrier_boundary (Mf:binding) (K:pkernel) (org:list place) (ids:PM.t reg)
 (pp:list (list placed)) (F:lfacts) : bool :=
 match source_events K org with
 | []=>true
 | [origin] => if barrier_linear_case K then true else
   if phase_source_ok K org origin then
   match source_placed_event K org ids pp origin with
   | Some event=>single_phase_check F (map (map fst) pp) event
   | None=>false end
   else false
 | origins => if barrier_linear_case K then true else
   if phase_sources_ok K org origins then
   match placed_events K org ids pp origins with
   | Some events=>if front_event_check Mf K org ids pp F events then true
                  else front_shared_check Mf K org ids pp F events
   | None=>false end else false
 end.

(* The refusal names the waiting sites it concerns. *)
Definition participation_refusal (K:pkernel) (org:list place) : string :=
 match source_barriers K org with
 | []=>"warp collective participation is not proved by the finite phase checker"
 | _=>"CTA barrier participation is not proved by the finite phase checker"
 end.

(* ------------------------------------------------------------------ *)
(* The lowered loops of the zeroing prologue.                          *)

(* The lowering of a kernel whose zeroing prologue clears dynamic shared
   memory with loops (zero_dynamic) must have the form below, of which
   GateZero.v proves that each thread stores zero at each of its elements
   before it leaves the loops. Block 0 begins with zero and the thread's
   linear index and ends with the count of the first loop; loop j has its
   head in block 3j + 1 and its body in block 3j + 2, and block 3j + 3
   begins after it, with the count of the next loop or the rest of the
   prologue. *)

(* The registers of the lowered prologue: zero, the thread's linear index, an
   address, the count, the element, and the predicate. *)
Record zero_regs := {
  zr_zero : reg; zr_index : reg; zr_address : reg;
  zr_count : reg; zr_element : reg; zr_pred : reg
}.

(* What the lowered loops must be: their registers, the registers of the
   thread's index in each dimension, the block, the width of a store, the
   count's constant and each size's register with its coefficient, both in
   elements of that width, and the register of each dynamic variable. *)
Record zero_plan := {
  zp_regs : zero_regs;
  zp_tx : reg; zp_ty : reg; zp_tz : reg;
  zp_bx : Z; zp_by : Z; zp_bz : Z;
  zp_width : Z;
  zp_const : Z;
  zp_terms : list (reg * Z);
  zp_bases : list reg
}.

Definition zp_threads (zp : zero_plan) : Z := zp_bx zp * zp_by zp * zp_bz zp.

Definition zero_index_ir (zp : zero_plan) : list instr :=
  let zr := zp_regs zp in
  [IImm None (zr_zero zr) 0; IMov None (zr_index zr) (zp_tz zp); IMov None (zr_address zr) (zp_ty zp);
   IMad None 32 (zr_index zr) (zr_index zr) (OImm (zp_by zp)) (OReg (zr_address zr));
   IMov None (zr_address zr) (zp_tx zp);
   IMad None 32 (zr_index zr) (zr_index zr) (OImm (zp_bx zp)) (OReg (zr_address zr))].

(* The load of a size s into the element through the 64-bit register v, as
   size_param_load lowers it, and its multiple added to the count. *)
Definition zero_term_ir (zr : zero_regs) (s : reg) (c : Z) (v : reg) (from : nat) : list instr :=
  [IImm None v 0; IMad None 64 v s (OImm 1) (OReg v); ICvt None 32 from false (zr_element zr) v;
   IMad None 32 (zr_count zr) (zr_element zr) (OImm c) (OReg (zr_count zr))].

(* The count of a loop, with the 64-bit register and the bits converted of
   each load, then the first element, the thread's index. *)
Definition zero_setup_ir (zp : zero_plan) (temps : list (reg * nat)) : list instr :=
  let zr := zp_regs zp in
  IImm None (zr_count zr) (zp_const zp) ::
  app (flat_map (fun x => let '((s, c), (v, from)) := x in zero_term_ir zr s c v from)
                (combine (zp_terms zp) temps))
      [IMov None (zr_element zr) (zr_index zr)].

Definition zero_head_ir (zr : zero_regs) (exit : nat) : list instr :=
  [ISetp None CGe 32 false (zr_pred zr) (zr_element zr) (OReg (zr_count zr));
   IBra (Some (false, zr_pred zr)) exit].

Definition zero_body_ir (zp : zero_plan) (base : reg) (head : nat) : list instr :=
  let zr := zp_regs zp in
  [IMov None (zr_address zr) base;
   IMad None 32 (zr_address zr) (zr_element zr) (OImm (zp_width zp)) (OReg (zr_address zr));
   IShA None KStore (zr_address zr) 0 (zp_width zp) (Some (OReg (zr_zero zr)));
   IBin None BAdd 32 (zr_element zr) (zr_element zr) (OImm (zp_threads zp));
   IBra None head].

Definition guard_eqb (g h : guard) : bool :=
  match g, h with
  | None, None => true
  | Some (a, p), Some (b, q) => Bool.eqb a b && Pos.eqb p q
  | _, _ => false
  end.

Definition opnd_eqb (a b : opnd) : bool :=
  match a, b with
  | OReg r, OReg s => Pos.eqb r s
  | OImm x, OImm y => Z.eqb x y
  | _, _ => false
  end.

Definition option_opnd_eqb (a b : option opnd) : bool :=
  match a, b with
  | None, None => true
  | Some x, Some y => opnd_eqb x y
  | _, _ => false
  end.

(* Equality of the instructions of the lowered loops: an instruction of
   another kind is equal to none. *)
Definition zero_instr_eqb (a b : instr) : bool :=
  match a, b with
  | IImm g d z, IImm h d' z' => guard_eqb g h && Pos.eqb d d' && Z.eqb z z'
  | IMov g d s, IMov h d' s' => guard_eqb g h && Pos.eqb d d' && Pos.eqb s s'
  | IMad g w d x y z, IMad h w' d' x' y' z' =>
      guard_eqb g h && Nat.eqb w w' && Pos.eqb d d' && Pos.eqb x x' && opnd_eqb y y' && opnd_eqb z z'
  | ICvt g t f s d x, ICvt h t' f' s' d' x' =>
      guard_eqb g h && Nat.eqb t t' && Nat.eqb f f' && Bool.eqb s s' && Pos.eqb d d' && Pos.eqb x x'
  | ISetp g CGe w s d x y, ISetp h CGe w' s' d' x' y' =>
      guard_eqb g h && Nat.eqb w w' && Bool.eqb s s' && Pos.eqb d d' && Pos.eqb x x' && opnd_eqb y y'
  | IBin g BAdd w d x y, IBin h BAdd w' d' x' y' =>
      guard_eqb g h && Nat.eqb w w' && Pos.eqb d d' && Pos.eqb x x' && opnd_eqb y y'
  | IShA g KStore x off w v, IShA h KStore x' off' w' v' =>
      guard_eqb g h && Pos.eqb x x' && Z.eqb off off' && Z.eqb w w' && option_opnd_eqb v v'
  | IBra g t, IBra h t' => guard_eqb g h && Nat.eqb t t'
  | _, _ => false
  end.

Fixpoint zero_instrs_eqb (l m : list instr) : bool :=
  match l, m with
  | [], [] => true
  | a :: l', b :: m' => zero_instr_eqb a b && zero_instrs_eqb l' m'
  | _, _ => false
  end.

(* The 64-bit register and the bits converted of each load of a count, as
   the lowering placed them in the count's instructions. *)
Definition zero_temps (setup : list instr) (terms : nat) : list (reg * nat) :=
  map (fun i => (match nth (1 + 4 * i) setup INop with IImm _ v _ => v | _ => 1%positive end,
                 match nth (3 + 4 * i) setup INop with ICvt _ _ from _ _ _ => from | _ => O end))
      (seq 0 terms).

(* The instructions of the count of loop j: block 0 after the thread's
   index for the first loop, the whole block 3j for the others. *)
Definition zero_setup_block (prog : program) (j : nat) : list instr :=
  match j with
  | O => skipn 6 (block_body prog 0)
  | _ => block_body prog (3 * j)
  end.

Definition pos_nodup (l : list reg) : bool :=
  fst (fold_left (fun acc r => let '(ok, seen) := acc in
                    (ok && negb (existsb (Pos.eqb r) seen), r :: seen)) l (true, [])).

Definition pos_apart (l m : list reg) : bool := forallb (fun r => negb (existsb (Pos.eqb r) m)) l.

Definition zero_regs_list (zr : zero_regs) : list reg :=
  [zr_zero zr; zr_index zr; zr_address zr; zr_count zr; zr_element zr; zr_pred zr].

(* The lowered loops have the form of zp, with registers of their widths:
   those that the loops write are distinct, and apart from those that they
   only read, the thread's index, the sizes, and the variables. *)
Definition zero_program_ok (wd : reg -> nat) (zp : zero_plan) (prog : program) : bool :=
  let zr := zp_regs zp in
  let m := List.length (zp_bases zp) in
  let temps := fun j => zero_temps (zero_setup_block prog j) (List.length (zp_terms zp)) in
  let written := zero_regs_list zr in
  let read := app [zp_tx zp; zp_ty zp; zp_tz zp] (app (map fst (zp_terms zp)) (zp_bases zp)) in
  Nat.ltb (3 * m) (List.length prog) &&
  zero_instrs_eqb (firstn 6 (block_body prog 0)) (zero_index_ir zp) &&
  forallb (fun x => let '(j, base) := x in
             zero_instrs_eqb (zero_setup_block prog j) (zero_setup_ir zp (temps j)) &&
             zero_instrs_eqb (block_body prog (3 * j + 1)) (zero_head_ir zr (3 * j + 3)) &&
             zero_instrs_eqb (block_body prog (3 * j + 2)) (zero_body_ir zp base (3 * j + 1)) &&
             forallb (fun t => let '(v, from) := t in
                        Nat.eqb (wd v) 64 && Nat.leb 32 from && negb (existsb (Pos.eqb v) (app written read)))
                     (temps j))
          (combine (seq 0 m) (zp_bases zp)) &&
  pos_nodup written && pos_apart written read &&
  forallb (fun r => Nat.eqb (wd r) 32) [zr_zero zr; zr_index zr; zr_address zr; zr_count zr; zr_element zr] &&
  Nat.eqb (wd (zr_pred zr)) 1 &&
  Z.ltb 0 (zp_bx zp) && Z.ltb 0 (zp_by zp) && Z.ltb 0 (zp_bz zp) && Z.ltb (zp_threads zp) (2 ^ 31) &&
  Z.ltb 0 (zp_width zp) && Z.ltb (zp_width zp) (2 ^ 31).

(* The values of a list of options, if every one has one. *)
Fixpoint option_all {A} (l : list (option A)) : option (list A) :=
  match l with
  | [] => Some []
  | Some x :: l' => match option_all l' with Some xs => Some (x :: xs) | None => None end
  | None :: _ => None
  end.

(* The plan of a kernel's lowered loops, from the binding and the names of
   the launch facts: the thread's index, each size, and each dynamic
   variable is the register that the launch facts name for it, and the
   lowered prologue's own registers are those of the first loop. *)
(* The name of the register that holds size symbol k (size_reg). *)
Definition size_name (k : Z) : string := "size:" ++ nstr (Z.to_N k).

Definition zero_terms_of (Mf : binding) (F : lfacts) (w : Z) : option (list (reg * Z)) :=
  option_all (map (fun x => match reg_named F (size_name (fst x)) with
                            | Some s => Some (s, snd x / w)
                            | None => None
                            end) (af_terms (b_shared Mf))).

(* The lowered prologue's own registers, from its first instructions and
   the head of its first loop. *)
Definition zero_regs_of (prog : program) : option zero_regs :=
  match block_body prog 0, block_body prog 1 with
  | IImm None z _ :: IMov None t _ :: IMov None ad _ :: _, [ISetp None _ _ _ p e (OReg n); _] =>
      Some {| zr_zero := z; zr_index := t; zr_address := ad; zr_count := n; zr_element := e; zr_pred := p |}
  | _, _ => None
  end.

Definition zero_plan_of (K : pkernel) (Mf : binding) (F : lfacts) (prog : program) : option zero_plan :=
  let '(bx, by_, bz) := b_block Mf in
  let w := shared_width (width_entries K Mf) in
  match reg_named F "%tid.x", reg_named F "%tid.y", reg_named F "%tid.z", zero_terms_of Mf F w,
        option_all (map (fun v => reg_named F ("shared:" ++ fst v)) (zeroed_dynamic K Mf)),
        zero_regs_of prog with
  | Some tx, Some ty, Some tz, Some terms, Some bases, Some zr =>
      Some {| zp_regs := zr; zp_tx := tx; zp_ty := ty; zp_tz := tz; zp_bx := bx; zp_by := by_; zp_bz := bz;
              zp_width := w; zp_const := af_const (b_shared Mf) / w; zp_terms := terms; zp_bases := bases |}
  | _, _, _, _, _, _ => None
  end.

(* A kernel whose prologue has loops has lowered loops of their plan's form. *)
Definition zero_program_check (K : pkernel) (Mf : binding) (F : lfacts) (prog : program) : res unit :=
  if null (zeroed_dynamic K Mf) then Ok tt else
  match zero_plan_of K Mf F prog with
  | Some zp => if zero_program_ok (wdn F) zp prog then Ok tt
               else Err "the lowered loops of the zeroing prologue do not have the form that GateZero.v proves" None
  | None => Err "the lowered loops of the zeroing prologue do not have the form that GateZero.v proves" None
  end.

(* The lowering of front before its participation check: the inlined
   kernel, its places, the placed program, the translator state, and the
   launch facts. *)
Definition front_lower (K:pkernel) (Mf:binding)
    : res (pkernel * list place * list (list placed) * tstate * lfacts) :=
 match zeroing_check K Mf with
 | Err e i=>Err e i
 | Ok _=>
 match inline K with
 | Err e i=>Err e i
 | Ok (K',org)=>
   match defined_check K' org with
   | Err e i=>Err e i
   | Ok _=>
   match (p <- translate_placed K' Mf org ;; F <- facts K' Mf ;; ret (p,F)) init with
   | Err e i=>Err e i
   | Ok ((pp,raw),ts)=>
     match zero_program_check K Mf (payload_facts raw (map (map fst) pp)) (map (map fst) pp) with
     | Err e i=>Err e i
     | Ok _=>Ok (K',org,pp,ts,payload_facts raw (map (map fst) pp))
     end
   end
   end
 end
 end.

Definition front_core (K:pkernel) (Mf:binding) : res (list place * list (list placed) * lfacts) :=
 match front_lower K Mf with
 | Err e i=>Err e i
 | Ok (K',org,pp,ts,F)=>
   if barrier_boundary Mf K' (expanded_places org) (ts_ids ts) pp F then Ok (org,pp,F)
   else Err (participation_refusal K' (expanded_places org)) None
 end.
(* The kernel's exact checked program and launch facts. *)
Definition front (K:pkernel) (Mf:binding) : res (program*lfacts) :=
 match front_core K Mf with
 | Ok (org,pp,F)=>Ok (map (map fst) pp,F)
 | Err e i=>Err e i end.

(* The program and launch facts that front gives when its participation
   check holds. The gate's query computes them for each entry kernel
   outside the kernel, and the kernel checks that they are what front
   gives before the proposal reads them, so the participation check runs
   once. *)
Definition front_program (K:pkernel) (Mf:binding) : res (program*lfacts) :=
 match front_lower K Mf with
 | Ok (_,_,pp,_,F)=>Ok (map (map fst) pp,F)
 | Err e i=>Err e i end.

Lemma barrier_boundary_phase_sound : forall Mf K org ids pp F origin,
 source_events K org = [origin] -> barrier_linear_case K = false ->
 barrier_boundary Mf K org ids pp F = true ->
 exists event, source_placed_event K org ids pp origin = Some event /\
 single_phase_safe F (map (map fst) pp) event.
Proof.
 intros Mf K org ids pp F origin Hsource Hlinear H.
 unfold barrier_boundary in H. rewrite Hsource,Hlinear in H.
 destruct (phase_source_ok K org origin); [|discriminate].
 destruct (source_placed_event K org ids pp origin) as [event|] eqn:He; [|discriminate].
 exists event. split; [reflexivity|]. apply single_phase_check_sound. exact H.
Qed.
(* This corollary ties the progress proof to the actual accepted frontend
   result, rather than to a separately supplied program or launch facts. *)
Local Opaque barrier_boundary inline translate_placed facts payload_facts single_phase_check phase_check.
Theorem accepted_front_phase : forall K Mf prog F K' org origin,
 front K Mf = Ok (prog,F) -> inline K = Ok (K',org) ->
 source_events K' (expanded_places org) = [origin] ->
 barrier_linear_case K' = false ->
 exists pp raw ts event,
 ((p <- translate_placed K' Mf org ;; F <- facts K' Mf ;; ret (p,F)) init)=Ok ((pp,raw),ts) /\
 prog = map (map fst) pp /\ F=payload_facts raw prog /\
 source_placed_event K' (expanded_places org) (ts_ids ts) pp origin = Some event /\ single_phase_safe F prog event.
Proof.
 intros K Mf prog F K' org origin Hfront Hinline Hsource Hlinear.
 unfold front,front_core,front_lower in Hfront.
 destruct (zeroing_check K Mf); [|discriminate]. rewrite Hinline in Hfront.
 destruct (defined_check K' org); [|discriminate].
 destruct ((p <- translate_placed K' Mf org ;; F <- facts K' Mf ;; ret (p,F)) init)
   as [[[pp raw] ts]|msg where_] eqn:Hlower; [|discriminate].
 destruct (zero_program_check K Mf (payload_facts raw (map (map fst) pp)) (map (map fst) pp)); [|discriminate].
 set (F' := payload_facts raw (map (map fst) pp)) in *.
 destruct (barrier_boundary Mf K' (expanded_places org) (ts_ids ts) pp F') eqn:Hboundary; [|discriminate].
 injection Hfront as <- <-.
 destruct (barrier_boundary_phase_sound Mf K' (expanded_places org) (ts_ids ts) pp F' origin Hsource Hlinear Hboundary)
   as [event [He Hsafe]].
 exists pp,raw,ts,event. split;[reflexivity|].
 split;[reflexivity|]. split;[reflexivity|]. split;assumption.
Qed.
Local Transparent barrier_boundary inline translate_placed facts payload_facts single_phase_check phase_check.

Lemma barrier_boundary_events_sound : forall Mf K org ids pp F origins,
 source_events K org=origins -> (2<=List.length origins)%nat ->
 barrier_linear_case K=false -> barrier_boundary Mf K org ids pp F=true ->
 exists events,placed_events K org ids pp origins=Some events /\
 (front_events_safe Mf K org ids pp F events \/ front_shared_safe Mf K org ids pp F events).
Proof.
 intros Mf K org ids pp F origins Hsource Hcount Hlinear H.
 unfold barrier_boundary in H. rewrite Hsource in H.
 destruct origins as [|first [|second rest]];cbn [List.length] in Hcount;try lia.
 rewrite Hlinear in H.
 rewrite phase_sources_exact in H.
 destruct (short_all (phase_source_ok K org) (first::second::rest));[|discriminate].
 destruct (placed_events K org ids pp (first::second::rest)) as [events|] eqn:He;[|discriminate].
 exists events. split;[reflexivity|].
 destruct (front_event_check Mf K org ids pp F events) eqn:Hold.
 - left. apply front_event_check_sound. exact Hold.
 - right. apply front_shared_check_sound. exact H.
Qed.
Local Opaque barrier_boundary inline translate_placed facts payload_facts event_progress_check.
Theorem accepted_front_events : forall K Mf prog F K' org,
 front K Mf=Ok (prog,F) -> inline K=Ok (K',org) ->
 (2<=List.length (source_events K' (expanded_places org)))%nat -> barrier_linear_case K'=false ->
 exists pp raw ts events,
 ((p <- translate_placed K' Mf org ;; F <- facts K' Mf ;; ret (p,F)) init)=Ok ((pp,raw),ts) /\
 prog=map (map fst) pp /\ F=payload_facts raw prog /\
 placed_events K' (expanded_places org) (ts_ids ts) pp (source_events K' (expanded_places org))=Some events /\
 (front_events_safe Mf K' (expanded_places org) (ts_ids ts) pp F events \/
  front_shared_safe Mf K' (expanded_places org) (ts_ids ts) pp F events).
Proof.
 intros K Mf prog F K' org Hfront Hinline Hcount Hlinear.
 unfold front,front_core,front_lower in Hfront.
 destruct (zeroing_check K Mf); [|discriminate]. rewrite Hinline in Hfront.
 destruct (defined_check K' org); [|discriminate].
 destruct ((p <- translate_placed K' Mf org ;; F <- facts K' Mf ;; ret (p,F)) init)
 as [[[pp raw] ts]|msg where_] eqn:Hlower;[|discriminate].
 destruct (zero_program_check K Mf (payload_facts raw (map (map fst) pp)) (map (map fst) pp)); [|discriminate].
 set (F':=payload_facts raw (map (map fst) pp)) in *.
 destruct (barrier_boundary Mf K' (expanded_places org) (ts_ids ts) pp F') eqn:Hboundary;[|discriminate].
 injection Hfront as <- <-.
 destruct (barrier_boundary_events_sound Mf K' (expanded_places org) (ts_ids ts) pp F' (source_events K' (expanded_places org)) eq_refl Hcount Hlinear Hboundary)
 as [events [He Hsafe]]. exists pp,raw,ts,events. split;[reflexivity|].
 split;[reflexivity|]. split;[reflexivity|]. split;assumption.
Qed.
Local Transparent barrier_boundary inline translate_placed facts payload_facts event_progress_check.

(* ------------------------------------------------------------------ *)
(* The link between the launch facts and the binding.                  *)

(* What the launch puts in a named register: size symbol k, parameter i of
   the binding, a thread register of dimension d, the lane, or a register of
   the platform (a window's base, a variable's address, a cluster's index).
   Of any other register the facts may state nothing. *)
(* The values the launch puts in a register of category c, if the facts may
   bound it: a buffer's base and a size parameter they may not, and the
   platform's registers have the platform's facts. *)
Definition value_range (Mf : binding) (c : category) : option (Z * Z) :=
  match c with
  | CSize k => Some (nth k (b_sizes Mf) (0, 0))
  | CParam _ (PConst z) => Some (z, z)
  | CParam _ (PBounded lo hi) => Some (lo, hi)
  | CParam _ PZero => Some (0, 0)
  | CParam _ (PBase _) | CParam _ (PSize _) | CPlatform => None
  | CTid d => Some (0, block_dim Mf d - 1)
  | CNtid d => Some (block_dim Mf d, block_dim Mf d)
  | CCtaid d => Some (0, grid_dim Mf d - 1)
  | CNctaid d => Some (1, grid_dim Mf d)
  | CLane => Some (0, 31)
  end.

(* A coefficient that a form of 64 bits keeps as it is. *)
Definition exact64 (c : Z) : bool := Z.ltb (- 2 ^ 63) c && Z.leb c (2 ^ 63).

(* The terms of an affine over the size registers: size symbol k is the
   register named size:k, which holds the size. *)
Fixpoint size_terms (F : lfacts) (Mf : binding) (l : list (Z * Z)) : option terms :=
  match l with
  | [] => Some []
  | (k, c) :: rest =>
      match reg_named F ("size:" ++ nstr (Z.to_N k)), size_terms F Mf rest with
      | Some r, Some ts =>
          match category_reg F Mf r with
          | Some (CSize j) =>
              if Z.leb 0 k && Nat.eqb j (Z.to_nat k) && fits (fst (nth j (b_sizes Mf) (0, 0))) (snd (nth j (b_sizes Mf) (0, 0))) (wdn F r)
              then Some (((r, EX), c) :: ts) else None
          | _ => None
          end
      | _, _ => None
      end
  end.

(* Form f is affine a over the size registers, and keeps its coefficients
   and constant, so its value at a launch is the affine's at the sizes. *)
Definition form_is (F : lfacts) (Mf : binding) (f : form) (a : affine) : bool :=
  match size_terms F Mf (af_terms a) with
  | Some ts => forallb (fun t => exact64 (snd t)) (tsort ts) && exact64 (af_const a) &&
               form_eqb (mk 64 ts (af_const a) (af_const a)) f
  | None => false
  end.

Definition dir_eqb (a b : dir) : bool :=
  match a, b with DRead, DRead | DWrite, DWrite | DReadWrite, DReadWrite => true | _, _ => false end.

(* Each bound the facts state of a fixed register holds of the value the
   launch puts in it, which the register holds as it is. *)
Definition bound_ok (F : lfacts) (Mf : binding) (e : reg * range) : bool :=
  let '(r, (lo, hi)) := e in
  negb (is_fixed F r) ||
  match category_reg F Mf r with
  | Some CPlatform => true
  | Some c => match value_range Mf c with
              | Some (l, h) => fits l h (wdn F r) && Z.leb lo l && Z.leb h hi
              | None => false
              end
  | None => false
  end.

(* Each alignment the facts state of a fixed register holds of the value the
   launch puts in it: a buffer's base is the pool's base plus the buffer's
   offset, both multiples of 2^8 bytes. *)
Definition align_ok (F : lfacts) (Mf : binding) (e : reg * Z) : bool :=
  let '(r, k) := e in
  negb (is_fixed F r) || Z.leb k 0 ||
  match category_reg F Mf r with
  | Some (CParam _ (PBase _)) => Z.leb k 8 && Nat.eqb (wdn F r) 64
  | Some (CParam _ (PConst z)) => fits z z (wdn F r) && Z.eqb (z mod 2 ^ k) 0
  | Some (CParam _ PZero) | Some CPlatform => true
  | _ => false
  end.

(* A role of the facts is the span of a buffer parameter of the binding,
   whose 64-bit register holds the buffer's address. *)
Definition role_ok (F : lfacts) (Mf : binding) (e : reg * span) : bool :=
  let '(P, sp) := e in
  match category_reg F Mf P with
  | Some (CParam _ (PBase bsp)) =>
      Nat.eqb (wdn F P) 64 && dir_eqb (sp_dir sp) (bs_dir bsp) && form_is F Mf (sp_ext sp) (bs_ext bsp) &&
      Bool.eqb (sp_flag sp) (bs_flag bsp) &&
      match sp_index sp, bs_index bsp with
      | None, None => true
      | Some ix, Some ir =>
          Z.eqb (ix_elem ix) (ir_elem ir) && Bool.eqb (ix_signed ix) (ir_signed ir) &&
          form_is F Mf (ix_lo ix) (ir_lo ir) && form_is F Mf (ix_hi ix) (ir_hi ir)
      | _, _ => false
      end
  | _ => false
  end.

(* The launch facts of a kernel hold of what a launch of its binding puts in
   each register: every bound, alignment, and role the facts state follows
   from the register's name (`link_sound` in StoreLink.v). *)
Definition derived_ok (F : lfacts) (Mf : binding) (k : constraint) : bool :=
 let '(ts,lo,hi) := k in
 existsb (fun d =>
   let a := derived_affine d in let c := af_const a in
   Z.ltb 0 (ds_divisor d) && Z.eqb lo (-c) && Z.eqb hi (ds_divisor d - 1 - c) &&
   form_is F Mf {| fm := 64; ft := ts; flo := c; fhi := c |} a) (b_derived Mf).

Definition link_ok (F : lfacts) (Mf : binding) : bool :=
  forallb (bound_ok F Mf) (lf_bounds F) && forallb (align_ok F Mf) (lf_aligns F) &&
  forallb (role_ok F Mf) (lf_roles F) && forallb (derived_ok F Mf) (lf_derived F).

Definition verify_ptx (K : pkernel) (Mf : binding) : bool :=
  match front K Mf with Ok (prog, F) => check_scalar F prog && link_ok F Mf | Err _ _ => false end.

(* The kernel's conversion of verify_ptx unfolds verify_ptx and keeps the
   translation folded: unfolding the translation of an unknown kernel
   unfolds the inlining at every depth. *)
Strategy opaque [front].

Theorem verify_ptx_sound : forall K Mf, verify_ptx K Mf = true ->
    exists prog F, front K Mf = Ok (prog, F) /\ check_scalar F prog = true /\ kernel_safe F prog /\
      link_ok F Mf = true.
Proof.
  intros K Mf H. unfold verify_ptx in H.
  destruct (front K Mf) as [[prog F] | e i]; [| discriminate].
  apply andb_true_iff in H. destruct H as [H Hl].
  exists prog, F. split; [reflexivity | split; [exact H | split; [exact (check_scalar_sound F prog H) | exact Hl]]].
Qed.

(* ------------------------------------------------------------------ *)
(* Why a kernel is refused.                                            *)

(* These functions name the check that fails and the PTX instruction it
   fails at. They decide nothing: `verify_ptx` accepts or refuses. *)

(* The kernel's program with each instruction's origin, and its launch
   facts: the program of `front` with origins. *)
(* Build the raw-origin lookup once. A missing or duplicated occurrence
   refuses diagnosis rather than selecting an arbitrary caller origin. *)
Definition diagnostic_origins (org : list place) : PM.t (list N) :=
 origin_index (map (fun '(expanded, raw) => (fst expanded, fst raw))
   (combine (expanded_places org) org)).
Fixpoint diagnostic_block (origins : PM.t (list N)) (body : list placed)
 : option (list placed) :=
 match body with
 | [] => Some []
 | (i,n) :: rest =>
   match origin_lookup origins n, diagnostic_block origins rest with
   | [raw], Some tail => Some ((i,raw) :: tail)
   | _,_ => None
   end
 end.
Fixpoint diagnostic_blocks (origins : PM.t (list N)) (pp : list (list placed))
 : option (list (list placed)) :=
 match pp with
 | [] => Some []
 | body :: rest =>
   match diagnostic_block origins body, diagnostic_blocks origins rest with
   | Some block, Some tail => Some (block :: tail)
   | _,_ => None
   end
 end.
Definition diagnostic_program (org : list place) (pp : list (list placed))
 : option (list (list placed)) := diagnostic_blocks (diagnostic_origins org) pp.
Lemma diagnostic_block_instructions : forall origins body restored,
 diagnostic_block origins body = Some restored -> map fst restored = map fst body.
Proof.
 intros origins body; induction body as [|[i n] rest IH]; intros restored H; cbn in H.
 - injection H as <-. reflexivity.
 - destruct (origin_lookup origins n) as [|raw [|extra remaining]]; try discriminate.
   destruct (diagnostic_block origins rest) as [tail|] eqn:Ht; [|discriminate].
   injection H as <-. cbn. f_equal. apply IH. reflexivity.
Qed.
Lemma diagnostic_blocks_instructions : forall origins pp restored,
 diagnostic_blocks origins pp = Some restored ->
 map (map fst) restored = map (map fst) pp.
Proof.
 intros origins pp; induction pp as [|body rest IH]; intros restored H; cbn in H.
 - injection H as <-. reflexivity.
 - destruct (diagnostic_block origins body) as [block|] eqn:Hb; [|discriminate].
   destruct (diagnostic_blocks origins rest) as [tail|] eqn:Ht; [|discriminate].
   injection H as <-. cbn. rewrite (diagnostic_block_instructions origins body block Hb).
   f_equal. apply IH. reflexivity.
Qed.
Lemma diagnostic_program_instructions : forall org pp restored,
 diagnostic_program org pp = Some restored ->
 map (map fst) restored = map (map fst) pp.
Proof. intros org pp restored H; exact (diagnostic_blocks_instructions _ _ _ H). Qed.
Definition front_placed (K : pkernel) (Mf : binding) : res (list (list placed) * lfacts) :=
 match front_core K Mf with
 | Err e i => Err e i
 | Ok (org,pp,F) =>
   match diagnostic_program org pp with
   | Some restored => Ok (restored,F)
   | None => Err "expanded source correspondence is inconsistent" None
   end
 end.

(* Why the check refuses the global access at a + off of kind k and width
   bytes under guard g, in state st. *)
Definition global_reason (F : lfacts) (st : astate) (g : guard) (k : mkind) (a : reg) (off width : Z) : string :=
  match refine_guard F st g with
  | None => "the access is refused under a guard that never holds"
  | Some sg =>
      match span_of_access F sg g a off width with
      | Some (_, sp, _) =>
          if negb (allows (sp_dir sp) k) then "the buffer's direction does not allow the access"
          else if sp_flag sp then "the write to the flag buffer is not a store of the number 1 into one whole word"
          else if is_load k then "the load does not read one whole element of the index buffer"
          else "the stored value is not proved to be one element in the index range"
      | None => "the access is not proved inside its buffer, a shared variable, or a local variable"
      end
  end.

(* Why the check refuses the memory access of instruction i in state st. *)
Definition access_reason (F : lfacts) (st : astate) (i : instr) : string :=
  if negb (aligned_ok F st i) then "the access is not proved aligned to its width" else
  match i with
  | ILoad g _ bits _ a off => global_reason F st g KLoad a off (Z.of_nat bits / 8)
  | IMem g k _ a off width _ _ _ =>
      let A := addr_form F st a off in
      (* the store to address 0 of never_under *)
      if null (ft A) && Z.eqb (flo A) 0 && Z.eqb (fhi A) 0
      then "the instruction is not proved unreachable: it traps, or PTX leaves its behavior undefined here"
      else global_reason F st g k a off width
  | IStL _ _ _ _ _ _ => "the local store is not proved inside a live local variable"
  | ILdS _ _ _ _ _ | IShA _ _ _ _ _ _ => "the shared memory access is not proved inside a shared variable"
  | ILdL _ _ _ _ _ _ => "the local load is not proved inside a local variable"
  | _ => "the access is not proved"
  end.

(* How the diagnostic walk of a block ends: at the first access that it
   refuses, with the accesses after it that it also refuses, by index and
   reason; at the index of an instruction whose register widths do not
   agree with it; or after the body, in a state. *)
Inductive walk_end :=
  | WalkRefused (first : nat * string) (later : list (nat * string))
  | WalkWidths (n : nat)
  | WalkDone (out : astate).

(* The accesses that a walk refuses. *)
Definition walk_refused (e : walk_end) : list (nat * string) :=
  match e with WalkRefused first later => first :: later | WalkWidths _ | WalkDone _ => [] end.

(* The walk of a plan, which lists at most S more refused accesses. The
   walk continues past a refused access with the same transfer and trims
   as checked_walk, so one walk lists the accesses that a series of checks
   would name one at a time, and it stops at the last access that it
   lists. *)
Fixpoint walk_failures (F : lfacts) (more : nat) (st : astate) (plan : walk_plan) (n : nat)
    : walk_end :=
  match plan with
  | [] => WalkDone st
  | (ins, keep) :: rest =>
      if negb (wf_instr F ins) then WalkWidths n
      else if access_of F st ins then walk_failures F more (trim_after F keep (transfer F st ins)) rest (S n)
      else WalkRefused (n, access_reason F st ins) (match more with
        | O => []
        | S more' => walk_refused (walk_failures F more' (trim_after F keep (transfer F st ins)) rest (S n))
        end)
  end.

(* At most S more accesses that the walk of case st of block b refuses, and
   the first check of case_block_ok that fails, at an index of the block,
   and why. *)
Definition case_failures (F : lfacts) (prog : program) (inv : list (list astate))
    (plan : walk_plan) (b : nat) (st : astate) (more : nat) : list (nat * string) * option (nat * string) :=
  let body := block_body prog b in
  if is_bottom st then
    ([], if forallb (wf_instr F) body then None
         else Some (0%nat, "the register widths do not agree with an instruction"))
  else match walk_failures F more st plan 0 with
    | WalkRefused first later => (first :: later, Some first)
    | WalkWidths n => ([], Some (n, "the register widths do not agree with the instruction"))
    | WalkDone out => ([], if case_edges_ok F prog inv b out then None
       else Some (pred (List.length body), "the analysis's invariant does not follow after this instruction"))
    end.

(* The first check of check_block that fails: at a block and instruction,
   or for the whole kernel, and why. *)
Definition partition_failure (F : lfacts) (prog : program) (plan : list (nat * nat * nat))
 : option (option (nat * nat) * string) :=
 match propose_cases F prog plan with
 | None => Some (None, "the analysis reaches no fixpoint")
 | Some inv =>
   if negb (lf_ok F) then Some (None, "the launch case is not consistent") else
   if negb (existsb (entry_case F) (cases_at inv 0))
   then Some (None, "the analysis states facts at the kernel's entry")
   else let plans := case_plan F prog inv in
   fold_left (fun acc b => let plan := plans b in fold_left (fun acc st => match acc with
     | Some _ => acc
     | None => option_map (fun p => (Some (b, fst p), snd p)) (snd (case_failures F prog inv plan b st 0))
     end) (cases_at inv b) acc) (seq 0 (List.length prog)) None
 end.

Definition block_check_failure (F : lfacts) (prog : program) : option (option (nat * nat) * string) :=
 if negb (lf_ok F) then Some (None, "the launch facts are not consistent")
 else if negb (structured prog) then Some (None, "a jump in the middle of a block")
 else match split_key F prog with
 | None => partition_failure F prog (independent_plan (conditional_loop_plan prog))
 | Some (r,cut) => let plan := loop_plan prog in
   match partition_failure (split_launch F r cut true) prog plan with
   | Some reason => Some reason
   | None => partition_failure (split_launch F r cut false) prog plan end
 end.

Definition check_failure (F : lfacts) (prog : program)
 : option (option (nat * nat) * string) :=
 match interval_prepare F prog with
 | Some prepared => match propose_prepared_interval F prog prepared with
   | None => Some (None, "the interval analysis cannot construct a complete certificate")
   | Some cert => if check_cached_certificate F prog cert then None
       else Some (None, "the interval certificate fails an instruction, access, or control-flow obligation")
   end
 | None => block_check_failure F prog
 end.

(* Why the translation or the check refuses the kernel, and the origin of
   the instruction where it does. *)
Definition refusal (K : pkernel) (Mf : binding) : option (string * option N) :=
  match front_placed K Mf with
  | Err e i => Some (e, i)
  | Ok (pp, F) =>
      match check_failure F (map (map fst) pp) with
      | None => None
      | Some (None, r) => Some (r, None)
      | Some (Some (b, i), r) => Some (r, option_map snd (nth_error (nth b pp []) i))
      end
  end.
