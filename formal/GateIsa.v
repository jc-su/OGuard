(*
  The instruction set and one-thread semantics that the checker of
  GateThread.v is proved against.

  A kernel is exported as basic blocks. Integer instructions have their exact
  PTX semantics: every register holds an unsigned value below 2^width, and
  each result is reduced modulo its destination's width. Loads that the
  checker tracks read memory; every other global memory instruction and every
  other instruction writes its destination registers with arbitrary values.
  Memory is an arbitrary input that may change at every step. A shared memory
  load reads the thread's view of its block's shared memory at that step,
  which is also an arbitrary input, so the other threads of the block may
  have written any value. A register that no instruction has written holds
  its launch value.

  A thread's local memory is private to it. Each local variable belongs to
  the frame of one function call, which its frame path names: the calls
  from the kernel to that call. A variable is live at an instruction when
  its frame path is a prefix of the instruction's: the variables of the
  frames on the call chain, which lie in disjoint memory. Each cell of the
  program's cell list is a register that holds the bytes of local memory at
  the cell's address, at the cell register's width. A local load reads the
  cell of a live variable with its address and width, if there is one, and
  otherwise gives an arbitrary value. A store of a cell's address and width
  writes the cell; a store that overlaps a cell otherwise gives it an
  arbitrary value, as does a store of an unknown value. Entering a frame
  gives the cells of its variables arbitrary values. Cell registers are
  fresh: no other instruction names them.

  `step` runs one instruction of one thread. `access_at` names the global
  memory access that an instruction makes, if its guard holds: its kind, its
  address modulo 2^64, and its width in bytes. A memory instruction without a
  state space uses a generic address: when all its bytes lie in the window of
  shared or local memory, which `win` decides, it accesses that memory and no
  global memory, so it makes no global access.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import PArith.PArith.
From Coq Require Import Arith.PeanoNat.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
From Coq Require Import NArith.NArith.
Import ListNotations.

Definition reg := positive.

(* The frame of a function call: the numbers of the calls from the kernel
   to it, [] for the kernel's own frame. *)
Definition frame := list N.

(* Whether frame a is frame b or one of the frames that called it. *)
Fixpoint prefix (a b : frame) : bool :=
  match a, b with
  | [], _ => true
  | x :: a', y :: b' => N.eqb x y && prefix a' b'
  | _ :: _, [] => false
  end.

Fixpoint lookup {A} (l : list (reg * A)) (r : reg) : option A :=
  match l with
  | [] => None
  | (x, v) :: l' => if Pos.eqb x r then Some v else lookup l' r
  end.

(* Whether local variable u, in the list of local variables with their
   bytes and frames, is live at an instruction of frame fr. *)
Definition live (locals : list (reg * (Z * frame))) (fr : frame) (u : reg) : bool :=
  match lookup locals u with Some (_, f) => prefix f fr | None => false end.

(* An instruction's guard: none, or a predicate register and whether it is
   negated. *)
Definition guard := option (bool * reg).

Inductive opnd := OReg (r : reg) | OImm (z : Z).

Inductive binop := BAdd | BSub | BMul | BShl | BShr | BShrs | BAnd | BOr | BXor.

Inductive cmp := CLt | CLe | CGt | CGe | CEq | CNe.

Inductive mkind := KLoad | KStore | KRmw.

Inductive instr :=
| IMov (g : guard) (d s : reg)                                  (* d := s *)
| IImm (g : guard) (d : reg) (z : Z)                            (* d := z *)
| IBin (g : guard) (o : binop) (w : nat) (d a : reg) (b : opnd) (* d := a o b at width w *)
| IMad (g : guard) (w : nat) (d a : reg) (b c : opnd)           (* mad.lo *)
| IWide (g : guard) (signed : bool) (d a : reg) (b : opnd) (c : option reg)
                                                                (* mul.wide / mad.wide from 32 bits *)
| ICvt (g : guard) (to from : nat) (signed : bool) (d s : reg)  (* integer conversion *)
| IBfe (g : guard) (w : nat) (d a : reg) (p ln : nat)           (* unsigned bit-field extract *)
| ISetp (g : guard) (c : cmp) (w : nat) (signed : bool) (d a : reg) (b : opnd)
| IAndp (g : guard) (d a b : reg)                               (* predicate conjunction *)
| IOrp (g : guard) (d a b : reg)                                (* predicate disjunction *)
| IMinMax (g : guard) (is_min signed : bool) (w : nat) (d a : reg) (b : opnd)
| ISelp (g : guard) (w : nat) (d : reg) (a b : opnd) (p : reg)  (* d := p ? a : b *)
| ILoad (g : guard) (gen : bool) (bits : nat) (d a : reg) (off : Z)
                                                                (* d := mem[a + off] mod 2^bits; gen:
                                                                   the address is generic *)
| IMem (g : guard) (k : mkind) (gen : bool) (a : reg) (off width : Z) (v : option opnd) (fr : frame)
       (ds : list reg)                                          (* any other global access; v: the
                                                                   value of a store, if known *)
| IDef (g : guard) (ds : list reg)                              (* any other instruction *)
| IBra (g : guard) (target : nat)
| IExit (g : guard)
| INop
| IDiv (g : guard) (rem signed : bool) (w : nat) (d a : reg) (b : opnd)
                                                                (* d := a / b, or a rem b, at width w *)
| ILdS (g : guard) (bits : nat) (d a : reg) (off : Z)          (* d := shared[a + off] mod 2^bits *)
| ILdL (g : guard) (bits : nat) (d a : reg) (off : Z) (fr : frame)
                                                                (* d := local[a + off] mod 2^bits *)
| IStL (g : guard) (a : reg) (off width : Z) (v : option opnd) (fr : frame)
                                                                (* local[a + off] := v, width bytes *)
| IFrame (vs : list reg)                                        (* a call's frame, of variables vs *)
| IShA (g : guard) (k : mkind) (a : reg) (off width : Z) (v : option opnd)
                                                                (* an access of kind k of width
                                                                   bytes to shared[a + off] that
                                                                   changes no register the model
                                                                   holds; v: the value of a store
                                                                   of one register or number *)
| IXorp (g : guard) (d a b : reg).                              (* predicate exclusive or *)

Definition program := list (list instr).

Definition is_jump (i : instr) : bool :=
  match i with IBra _ _ | IExit _ => true | _ => false end.

(* ------------------------------------------------------------------ *)
(* Integer operations.                                                 *)

Definition trunc (n : nat) (v : Z) : Z := Z.modulo v (2 ^ Z.of_nat n).

(* v's low n bits read as a signed integer. *)
Definition sext (n : nat) (v : Z) : Z :=
  let u := trunc n v in
  if Z.ltb u (2 ^ (Z.of_nat n - 1)) then u else u - 2 ^ Z.of_nat n.

Definition extend (n : nat) (signed : bool) (v : Z) : Z := if signed then sext n v else trunc n v.

(* A shift amount: the operand's low 32 bits, clamped to the width. *)
Definition shamt (w : nat) (b : Z) : Z := Z.min (trunc 32 b) (Z.of_nat w).

Definition binop_eval (o : binop) (w : nat) (a b : Z) : Z :=
  match o with
  | BAdd => a + b
  | BSub => a - b
  | BMul => a * b
  | BShl => a * 2 ^ shamt w b
  | BShr => trunc w a / 2 ^ shamt w b
  | BShrs => sext w a / 2 ^ shamt w b
  | BAnd => Z.land (trunc w a) (trunc w b)
  | BOr => Z.lor (trunc w a) (trunc w b)
  | BXor => Z.lxor (trunc w a) (trunc w b)
  end.

Definition cmp_eval (c : cmp) (a b : Z) : bool :=
  match c with
  | CLt => Z.ltb a b
  | CLe => Z.leb a b
  | CGt => Z.ltb b a
  | CGe => Z.leb b a
  | CEq => Z.eqb a b
  | CNe => negb (Z.eqb a b)
  end.

Definition bool_z (b : bool) : Z := if b then 1 else 0.

(* The quotient, rounded toward zero, or the remainder, with the sign of the
   dividend, of a and b read at width w. PTX leaves the result unspecified
   for a zero divisor and for the one signed quotient that overflows. *)
Definition div_eval (rem signed : bool) (w : nat) (a b : Z) : option Z :=
  let x := extend w signed a in
  let y := extend w signed b in
  if Z.eqb y 0 || (signed && Z.eqb y (-1) && Z.eqb x (- 2 ^ (Z.of_nat w - 1))) then None
  else Some (if rem then Z.rem x y else Z.quot x y).

(* The bytes of a window: every GPU the executor supports maps at least 2^24
   generic addresses from each window's base to its state space. *)
Definition window_size : Z := 2 ^ 24.

Section Semantics.

Variable prog : program.
Variable wd : reg -> nat.               (* each register's declared width *)
Variable mem : nat -> Z -> Z.           (* memory at each step *)
Variable arb : nat -> reg -> Z.         (* results of the other instructions *)
Variable shm : nat -> Z -> Z.           (* the shared memory word at each address, at each step *)
Variable init : reg -> Z.               (* registers at launch *)
Variable cells : list (reg * reg * Z).  (* each cell register, the register that holds its local
                                           variable's address, and its offset in the variable *)
Variable locals : list (reg * (Z * frame)).
                                        (* each local variable's register, bytes, and frame *)
Variable lwin : reg.                    (* holds the generic address of local address 0 *)

Record state := { blk : nat; idx : nat; regs : reg -> Z }.

Definition upd (rs : reg -> Z) (r : reg) (v : Z) : reg -> Z :=
  fun x => if Pos.eqb x r then trunc (wd r) v else rs x.

Definition holds (g : guard) (rs : reg -> Z) : bool :=
  match g with
  | None => true
  | Some (neg, p) => xorb neg (negb (Z.eqb (rs p) 0))
  end.

Definition opv (rs : reg -> Z) (o : opnd) : Z :=
  match o with OReg r => rs r | OImm z => z end.

Definition block_body (b : nat) : list instr :=
  match nth_error prog b with Some body => body | None => [] end.

Definition fetch (b i : nat) : option instr := nth_error (block_body b) i.

Definition block_len (b : nat) : nat := length (block_body b).

(* The position after instruction i of block b when control falls through. *)
Definition next_pos (b i : nat) : nat * nat :=
  if Nat.ltb (S i) (block_len b) then (b, S i) else (S b, 0).

Definition goto (p : nat * nat) (rs : reg -> Z) : state :=
  {| blk := fst p; idx := snd p; regs := rs |}.

Fixpoint write_all (rs : reg -> Z) (t : nat) (ds : list reg) : reg -> Z :=
  match ds with
  | [] => rs
  | d :: ds' => write_all (upd rs d (arb t d)) t ds'
  end.

(* The cell of bits at local address x, at an instruction of frame fr: the
   first cell register of that width, of a live variable, whose address is
   x. *)
Definition cell_at (fr : frame) (x : Z) (bits : nat) : option reg :=
  match find (fun k => let '(p, u, c) := k in
                       live locals fr u && Z.eqb (init u + c)%Z x && Nat.eqb (wd p) bits) cells with
  | Some (p, _, _) => Some p
  | None => None
  end.

(* A local load of bits at local address x into d, at an instruction of
   frame fr. *)
Definition lload (t : nat) (rs : reg -> Z) (d : reg) (x : Z) (bits : nat) (fr : frame) : Z :=
  match cell_at fr x bits with Some p => rs p | None => arb t d end.

(* Entering the frame of variables vs: their cells get arbitrary values. *)
Definition lframe_cell (t : nat) (vs : list reg) (r : reg -> Z) (k : reg * reg * Z) : reg -> Z :=
  let '(p, u, _) := k in if existsb (Pos.eqb u) vs then upd r p (arb t p) else r.

Definition lframe (t : nat) (rs : reg -> Z) (vs : list reg) : reg -> Z := fold_left (lframe_cell t vs) cells rs.

(* A local store of width bytes at local address x, of value v if known, on
   the register of cell k: the value if the store has the cell's address and
   width, an arbitrary value if it overlaps the cell otherwise. *)
Definition lstore_cell (t : nat) (x width : Z) (v : option Z) (r : reg -> Z) (k : reg * reg * Z) : reg -> Z :=
  let '(p, u, c) := k in
  let y := (init u + c)%Z in
  if Z.eqb y x && Z.eqb (Z.of_nat (wd p)) (8 * width)%Z
  then upd r p (match v with Some z => z | None => arb t p end)
  else if (y <? x + width)%Z && (x <? y + Z.of_nat (wd p) / 8)%Z then upd r p (arb t p)
  else r.

Definition lstore (t : nat) (rs : reg -> Z) (x width : Z) (v : option Z) : reg -> Z :=
  fold_left (lstore_cell t x width v) cells rs.

(* A generic store of width bytes at x: a local store when all of them lie
   in the window of local memory. *)
Definition gstore (t : nat) (rs : reg -> Z) (x width : Z) (v : option Z) : reg -> Z :=
  if (init lwin <=? x)%Z && (x + width <=? init lwin + window_size)%Z then lstore t rs (x - init lwin)%Z width v
  else rs.

(* The value an exact instruction writes to its destination. *)
Definition result (t : nat) (rs : reg -> Z) (i : instr) : Z :=
  match i with
  | IMov _ _ s => rs s
  | IImm _ _ z => z
  | IBin _ o w _ a b => binop_eval o w (rs a) (opv rs b)
  | IMad _ w _ a b c => rs a * opv rs b + opv rs c
  | IWide _ signed _ a b c =>
      extend 32 signed (rs a) * extend 32 signed (opv rs b) +
      match c with Some r => rs r | None => 0 end
  | ICvt _ _ from signed _ s => extend from signed (rs s)
  | IBfe _ w _ a p ln => trunc ln (trunc w (rs a) / 2 ^ Z.of_nat p)
  | ISetp _ c w signed _ a b => bool_z (cmp_eval c (extend w signed (rs a)) (extend w signed (opv rs b)))
  | IAndp _ _ a b => bool_z (negb (Z.eqb (rs a) 0) && negb (Z.eqb (rs b) 0))
  | IOrp _ _ a b => bool_z (negb (Z.eqb (rs a) 0) || negb (Z.eqb (rs b) 0))
  | IXorp _ _ a b => bool_z (xorb (negb (Z.eqb (rs a) 0)) (negb (Z.eqb (rs b) 0)))
  | IMinMax _ is_min signed w _ a b =>
      let x := extend w signed (rs a) in
      let y := extend w signed (opv rs b) in
      if is_min then Z.min x y else Z.max x y
  | ISelp _ _ _ a b p => if Z.eqb (rs p) 0 then opv rs b else opv rs a
  | ILoad _ _ bits _ a off => trunc bits (mem t (trunc 64 (rs a + off)))
  | IDiv _ rem signed w d a b =>
      match div_eval rem signed w (rs a) (opv rs b) with Some v => v | None => arb t d end
  | ILdS _ bits _ a off => trunc bits (shm t (trunc 32 (rs a + off)))
  | _ => 0
  end.

(* The register an exact instruction writes. *)
Definition dest (i : instr) : option reg :=
  match i with
  | IMov _ d _ | IImm _ d _ | IBin _ _ _ d _ _ | IMad _ _ d _ _ _ | IWide _ _ d _ _ _
  | ICvt _ _ _ _ d _ | IBfe _ _ d _ _ _ | ISetp _ _ _ _ d _ _ | IAndp _ d _ _ | IOrp _ d _ _
  | IMinMax _ _ _ _ d _ _ | ISelp _ _ d _ _ _ | ILoad _ _ _ d _ _ | IDiv _ _ _ _ d _ _
  | ILdS _ _ d _ _ | IXorp _ d _ _ => Some d
  | _ => None
  end.

Definition guard_of (i : instr) : guard :=
  match i with
  | IMov g _ _ | IImm g _ _ | IBin g _ _ _ _ _ | IMad g _ _ _ _ _ | IWide g _ _ _ _ _
  | ICvt g _ _ _ _ _ | IBfe g _ _ _ _ _ | ISetp g _ _ _ _ _ _ | IAndp g _ _ _ | IOrp g _ _ _
  | IMinMax g _ _ _ _ _ _ | ISelp g _ _ _ _ _ | ILoad g _ _ _ _ _ | IMem g _ _ _ _ _ _ _ _
  | IDef g _ | IBra g _ | IExit g | IDiv g _ _ _ _ _ _
  | ILdS g _ _ _ _ | ILdL g _ _ _ _ _ | IStL g _ _ _ _ _ | IShA g _ _ _ _ _ | IXorp g _ _ _ => g
  | INop | IFrame _ => None
  end.

(* Explicit register inputs, including guards and every modeled payload.
   Local-memory cell registers and their bases depend on launch facts; users
   that track such cells must add those roots separately. *)
Definition opnd_inputs (o : GateIsa.opnd) : list reg :=
  match o with OReg r => [r] | _ => [] end.

Definition instruction_inputs (i : instr) : list reg :=
  let inputs := match i with
    | IMov _ _ s => [s]
    | IBin _ _ _ _ a b | ISetp _ _ _ _ _ a b
    | IMinMax _ _ _ _ _ a b | IDiv _ _ _ _ _ a b => a :: opnd_inputs b
    | IMad _ _ _ a b c => a :: opnd_inputs b ++ opnd_inputs c
    | IWide _ _ _ a b c =>
        a :: opnd_inputs b ++ match c with Some r => [r] | _ => [] end
    | ICvt _ _ _ _ _ r | IBfe _ _ _ r _ _ => [r]
    | IAndp _ _ a b | IOrp _ _ a b | IXorp _ _ a b => [a; b]
    | ISelp _ _ _ a b p => p :: opnd_inputs a ++ opnd_inputs b
    | ILoad _ _ _ _ a _ | ILdS _ _ _ a _
    | ILdL _ _ _ a _ _ | IShA _ _ a _ _ _ => [a]
    | IMem _ _ _ a _ _ v _ _ | IStL _ a _ _ v _ =>
        a :: match v with Some o => opnd_inputs o | _ => [] end
    | _ => []
    end in
  match guard_of i with Some (_, p) => p :: inputs | _ => inputs end.

(* All explicit destinations, including conditional and multiple writes.
   The analysis separately accounts for hidden local-cell writes. *)
Definition instruction_outputs (i : instr) : list reg :=
  match i with
  | IDef _ ds | IMem _ _ _ _ _ _ _ _ ds => ds
  | ILdL _ _ d _ _ _ => [d]
  | _ => match dest i with Some d => [d] | None => [] end
  end.

(* A guarded instruction may leave every destination unchanged. *)
Definition instruction_definite_writes (i : instr) : list reg :=
  match guard_of i with Some _ => [] | None => instruction_outputs i end.

Definition step (t : nat) (s : state) : option state :=
  match fetch (blk s) (idx s) with
  | None => None
  | Some i =>
      let rs := regs s in
      let n := next_pos (blk s) (idx s) in
      Some (if negb (holds (guard_of i) rs) then goto n rs else
            match i with
            | IDef _ ds => goto n (write_all rs t ds)
            | IMem _ k gen a off width v _ ds =>
                let rs1 := write_all rs t ds in
                goto n (match k with
                        | KLoad => rs1
                        | _ => if gen
                               then gstore t rs1 (trunc 64 (rs a + off)) width
                                           (match k, v with KStore, Some o => Some (opv rs o) | _, _ => None end)
                               else rs1
                        end)
            | ILdL _ bits d a off fr => goto n (upd rs d (lload t rs d (trunc 64 (rs a + off)) bits fr))
            | IStL _ a off width v _ => goto n (lstore t rs (trunc 64 (rs a + off)) width (option_map (opv rs) v))
            | IFrame vs => goto n (lframe t rs vs)
            | IBra _ target => goto (target, 0) rs
            | IExit _ => goto (length prog, 0) rs
            | INop => goto n rs
            | _ => match dest i with
                   | Some d => goto n (upd rs d (result t rs i))
                   | None => goto n rs
                   end
            end)
  end.

Fixpoint run (t : nat) : option state :=
  match t with
  | 0 => Some {| blk := 0; idx := 0; regs := init |}
  | S t' => match run t' with Some s => step t' s | None => None end
  end.

(* The global memory access instruction i makes in register state rs, if its
   guard holds: its kind, its address, its width in bytes, and the value a
   store of one register or number writes. `win x w` holds when the w bytes
   at generic address x lie in the window of shared or local memory. *)
Definition access_at (win : Z -> Z -> bool) (rs : reg -> Z) (i : instr) : option (mkind * Z * Z * option Z) :=
  if negb (holds (guard_of i) rs) then None else
  match i with
  | ILoad _ gen bits _ a off =>
      let x := trunc 64 (rs a + off) in
      if gen && win x (Z.of_nat bits / 8)%Z then None else Some (KLoad, x, (Z.of_nat bits / 8)%Z, None)
  | IMem _ k gen a off width v _ _ =>
      let x := trunc 64 (rs a + off) in
      if gen && win x width then None
      else Some (k, x, width, match k, v with KStore, Some o => Some (opv rs o) | _, _ => None end)
  | _ => None
  end.

(* The address and the width in bytes of the memory access instruction i
   makes in register state rs, in any state space, if its guard holds. *)
Definition mem_access (rs : reg -> Z) (i : instr) : option (Z * Z) :=
  if negb (holds (guard_of i) rs) then None else
  match i with
  | ILoad _ _ bits _ a off | ILdL _ bits _ a off _ => Some (trunc 64 (rs a + off), (Z.of_nat bits / 8)%Z)
  | IMem _ _ _ a off width _ _ _ | IStL _ a off width _ _ => Some (trunc 64 (rs a + off), width)
  | ILdS _ bits _ a off => Some (trunc 32 (rs a + off), (Z.of_nat bits / 8)%Z)
  | IShA _ _ a off width _ => Some (trunc 32 (rs a + off), width)
  | _ => None
  end.

(* An access to shared or local memory: at an address of that memory, or at
   a generic address in a window. *)
Inductive private := PShared (x width : Z) | PLocal (x width : Z) | PGeneric (x width : Z).

(* The access to shared or local memory that instruction i makes in
   register state rs, if its guard holds; a generic access is one when win
   places its address in a window. *)
Definition private_access (win : Z -> Z -> bool) (rs : reg -> Z) (i : instr) : option private :=
  if negb (holds (guard_of i) rs) then None else
  match i with
  | ILdS _ bits _ a off => Some (PShared (trunc 32 (rs a + off)) (Z.of_nat bits / 8)%Z)
  | IShA _ _ a off width _ => Some (PShared (trunc 32 (rs a + off)) width)
  | ILdL _ bits _ a off _ => Some (PLocal (trunc 64 (rs a + off)) (Z.of_nat bits / 8)%Z)
  | IStL _ a off width _ _ => Some (PLocal (trunc 64 (rs a + off)) width)
  | ILoad _ gen bits _ a off =>
      let x := trunc 64 (rs a + off) in
      if gen && win x (Z.of_nat bits / 8)%Z then Some (PGeneric x (Z.of_nat bits / 8)%Z) else None
  | IMem _ _ gen a off width _ _ _ =>
      let x := trunc 64 (rs a + off) in
      if gen && win x width then Some (PGeneric x width) else None
  | _ => None
  end.

End Semantics.

(* Same-width fresh copies preserve comparisons and every original register.
   The front end uses these equations when retaining overwritten operands. *)
Local Open Scope Z_scope.

Lemma comparison_copy_value wd rs tmp src :
  0 <= rs src < 2 ^ Z.of_nat (wd tmp) ->
  upd wd rs tmp (rs src) tmp = rs src.
Proof. intros H. unfold upd. rewrite Pos.eqb_refl. unfold trunc. apply Z.mod_small. exact H. Qed.

Lemma comparison_copy_other wd rs tmp src r :
  r <> tmp -> upd wd rs tmp (rs src) r = rs r.
Proof. intros H. unfold upd. apply Pos.eqb_neq in H. rewrite H. reflexivity. Qed.

Lemma comparison_copy_guard wd rs tmp src g :
  (forall neg p, g = Some (neg, p) -> p <> tmp) ->
  holds g (upd wd rs tmp (rs src)) = holds g rs.
Proof.
  intros H. destruct g as [[neg p]|]; [|reflexivity].
  unfold holds. rewrite comparison_copy_other; [reflexivity|eapply H; reflexivity].
Qed.

Lemma comparison_snapshot_left wd mem arb shm t rs g c w signed d a b tmp :
  0 <= rs a < 2 ^ Z.of_nat (wd tmp) ->
  (match b with OReg r => r <> tmp | OImm _ => True end) ->
  result mem arb shm t (upd wd rs tmp (rs a)) (ISetp g c w signed d tmp b) =
  result mem arb shm t rs (ISetp g c w signed d a b).
Proof.
  intros Hwidth Hother. cbn [result]. rewrite comparison_copy_value by exact Hwidth.
  destruct b; cbn [opv]; [rewrite comparison_copy_other by exact Hother|]; reflexivity.
Qed.

Lemma comparison_snapshot_right wd mem arb shm t rs g c w signed d a b tmp :
  0 <= rs b < 2 ^ Z.of_nat (wd tmp) -> a <> tmp ->
  result mem arb shm t (upd wd rs tmp (rs b)) (ISetp g c w signed d a (OReg tmp)) =
  result mem arb shm t rs (ISetp g c w signed d a (OReg b)).
Proof.
  intros Hwidth Hother. cbn [result opv].
  rewrite comparison_copy_value by exact Hwidth.
  rewrite comparison_copy_other by exact Hother. reflexivity.
Qed.

Lemma comparison_snapshot_state wd rs tmp src g d value r :
  (forall neg p, g = Some (neg, p) -> p <> tmp) -> r <> tmp ->
  (if holds g (upd wd rs tmp (rs src))
   then upd wd (upd wd rs tmp (rs src)) d value else upd wd rs tmp (rs src)) r =
  (if holds g rs then upd wd rs d value else rs) r.
Proof.
  intros Hguard Hfresh. rewrite comparison_copy_guard by exact Hguard.
  destruct (holds g rs); [unfold upd at 1 3; destruct (Pos.eqb r d); [reflexivity|]|].
  all: apply comparison_copy_other; exact Hfresh.
Qed.


Local Close Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Structure.                                                          *)

Definition last_instr (body : list instr) : option instr :=
  match rev body with [] => None | i :: _ => Some i end.

(* Blocks are non-empty, and a branch or exit only ends a block. *)
Definition structured (prog : program) : bool :=
  forallb (fun body =>
    negb (Nat.eqb (length body) 0) && forallb (fun i => negb (is_jump i)) (removelast body)) prog.
