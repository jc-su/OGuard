(*
  The PTX parser: the text of a PTX module to the kernels that verification
  checks.

  Verification gives Rocq the exact text that the driver loads, eight 7-bit
  characters in each element of a primitive array. `verify_text` reads the
  module from it, requires a binding for each entry kernel and an entry
  kernel for each binding, and runs `verify_ptx` (GateFront.v) on each entry
  kernel with the module's device functions. `verify_text_sound`: each entry
  kernel of the text is accepted with its binding, so `verify_ptx_sound`
  gives the regions of its accesses.

  The parser and lowering remain trusted and must agree with PTX semantics.
  tests/gate/test_parser.py checks corpus entry-name parsing against the
  exported entry names and that those modules assemble with ptxas; it does
  not prove instruction-effect fidelity. tests/gate/test_effect_boundary.py
  checks fail-closed lowering of benign unknown/malformed instructions and
  the supported register-only classification. The reading rules:

    - The text holds tabs, line breaks and the printable ASCII characters.
    - A comment is removed, but a comment marker inside a string is text, as
      it is to `ptxas`. A string with an escape or a line break, and a `#`
      outside comments and strings, which `ptxas` reads as a preprocessor
      line, are refused.
    - Each string's text is blanked before the structure is read, so a brace,
      a semicolon or a declaration inside a string is no structure.
    - A kernel is `.entry` or `.func` with its parameter list and a body in
      braces or a semicolon. A body is statements: `{ }` scopes, register
      declarations, directives, labels, `.branchtargets` lists and
      instructions. A directive the grammar does not model is refused.
    - A name refers to its declaration in the innermost enclosing `{ }`
      scope, and a name a nested scope declares is made distinct: `%t` of
      scope 3 becomes `%s3$t`, which no PTX name is.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import NArith.NArith.
From Coq Require Import Arith.PeanoNat.
From Coq Require Import Bool.Bool.
From Coq Require Import Lists.List.
From Coq Require Import Strings.String.
From Coq Require Import Strings.Ascii.
From Coq Require Import Numbers.Cyclic.Int63.Uint63.
From Coq Require Import Array.PArray.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateDomain.
Require Import OpGate.GateThread.
Require Import OpGate.GateFront.
Require Import OpGate.GateInterval.
Require Export OpGate.GateCertificate.
Require Import OpGate.GateTerm.
Require Import OpGate.GatePrologue.

(* The integer comparisons are uint63's: that scope opens last. *)
Local Open Scope string_scope.
Local Open Scope uint63_scope.

(* ------------------------------------------------------------------ *)
(* Characters.                                                         *)

Definition code (a : ascii) : int :=
  let '(Ascii b0 b1 b2 b3 b4 b5 b6 b7) := a in
  (if b0 then 1 else 0) + (if b1 then 2 else 0) + (if b2 then 4 else 0) + (if b3 then 8 else 0) +
  (if b4 then 16 else 0) + (if b5 then 32 else 0) + (if b6 then 64 else 0) +
  (if b7 then 128 else 0).

Definition bit (c k : int) : bool := (c >> k) land 1 =? 1.

Definition char (c : int) : ascii :=
  Ascii (bit c 0) (bit c 1) (bit c 2) (bit c 3) (bit c 4) (bit c 5) (bit c 6) (bit c 7).

Definition space (c : int) : bool := ((9 <=? c) && (c <=? 13)) || (c =? 32).
Definition alpha (c : int) : bool := ((65 <=? c) && (c <=? 90)) || ((97 <=? c) && (c <=? 122)).
Definition digit (c : int) : bool := (48 <=? c) && (c <=? 57).
Definition hexdigit (c : int) : bool :=
  digit c || ((65 <=? c) && (c <=? 70)) || ((97 <=? c) && (c <=? 102)).
(* A word character, `\w` of a regular expression. *)
Definition word (c : int) : bool := alpha c || digit c || (c =? 95).
Definition ident_start (c : int) : bool := alpha c || (c =? 95) || (c =? 36).
Definition ident_cont (c : int) : bool := word c || (c =? 36).
(* A character the parser reads: tab to carriage return, and printable ASCII. *)
Definition readable (c : int) : bool := ((9 <=? c) && (c <=? 13)) || ((32 <=? c) && (c <=? 126)).
Definition lower (c : int) : int := if (65 <=? c) && (c <=? 90) then c + 32 else c.

Definition slen (s : string) : int := of_Z (Z.of_nat (String.length s)).

Fixpoint codes (s : string) : list int :=
  match s with EmptyString => [] | String a r => code a :: codes r end.

(* More steps than any scan of n characters takes. *)
Definition fuel_of (n : int) : nat := S (Z.to_nat (to_Z n)).

(* ------------------------------------------------------------------ *)
(* The text.                                                           *)

(* The n characters of the packed text, one in each element: character i is
   bits 7 * (i mod 8) to 7 * (i mod 8) + 6 of element i / 8. *)
Definition unpack (a : array int) (n : int) : array int :=
  (fix go (k : nat) (i : int) (out : array int) :=
     match k with
     | O => out
     | S k => if i <? n then go k (i + 1) (set out i ((a.[i / 8] >> (7 * (i mod 8))) land 127))
              else out
     end) (fuel_of n) 0 (PArray.make n 0).

(* The end of the string that opens at i: the index of its closing quote, or
   None for a string with an escape, a line break or no end. *)
Definition string_end (F : nat) (a : array int) (n i : int) : option int :=
  (fix go (k : nat) (j : int) :=
     match k with
     | O => None
     | S k => if j <? n then
                let c := a.[j] in
                if c =? 34 then Some j else if (c =? 92) || (c =? 10) then None else go k (j + 1)
              else None
     end) F (i + 1).

(* The text without comments, with each string's text blanked. A line comment
   leaves its line break, a block comment a blank for each character and its
   line breaks. As in the reference parser, an unterminated block comment ends
   before the last character, which `ptxas` refuses anyway. *)
Fixpoint strip_go (k : nat) (F : nat) (a : array int) (n i : int) (o : array int) (m : int)
    : res (array int * int) :=
  match k with
  | O => Err "the PTX text is longer than the parser reads" None
  | S k =>
      if i <? n then
        let c := a.[i] in
        if negb (readable c) then Err "PTX text with a character the parser does not read" None
        else if c =? 35 then Err "PTX preprocessor line, which the parser does not model" None
        else if c =? 34 then
          match string_end F a n i with
          | None => Err "PTX string literal with an escape, a line break or no end" None
          | Some e =>
              let o := (fix blank (k : nat) (j : int) (o : array int) :=
                          match k with
                          | O => o
                          | S k => if j <? e then blank k (j + 1) (set o (m + (j - i)) 32) else o
                          end) F (i + 1) (set o m 34) in
              strip_go k F a n (e + 1) (set o (m + (e - i)) 34) (m + (e - i) + 1)
          end
        else if (c =? 47) && (i + 1 <? n) && (a.[i + 1] =? 47) then
          let e := (fix line (k : nat) (j : int) :=
                      match k with
                      | O => j
                      | S k => if (j <? n) && negb (a.[j] =? 10) then line k (j + 1) else j
                      end) F (i + 2) in
          if e <? n then strip_go k F a n (e + 1) (set o m 10) (m + 1) else Ok (o, m)
        else if (c =? 47) && (i + 1 <? n) && (a.[i + 1] =? 42) then
          let '(j, o, m) := (fix block (k : nat) (j : int) (o : array int) (m : int) :=
                               match k with
                               | O => (j, o, m)
                               | S k =>
                                   if (j + 1 <? n) && negb ((a.[j] =? 42) && (a.[j + 1] =? 47))
                                   then block k (j + 1) (set o m (if a.[j] =? 10 then 10 else 32)) (m + 1)
                                   else (j, o, m)
                               end) F (i + 2) (set (set o m 32) (m + 1) 32) (m + 2) in
          if j + 1 <? n then strip_go k F a n (j + 2) (set (set o m 32) (m + 1) 32) (m + 2)
          else strip_go k F a n j o m
        else strip_go k F a n (i + 1) (set o m c) (m + 1)
      else Ok (o, m)
  end.

(* ------------------------------------------------------------------ *)
(* Scanning the stripped text T.                                       *)

Section Scan.

Variable T : array int.     (* the stripped text, one character in each element *)
Variable F : nat.           (* more steps than any scan of the text takes *)

(* The character at i of a window that ends at hi, or 0 past it. *)
Definition ch (hi i : int) : int := if i <? hi then T.[i] else 0.

Definition skip_while (p : int -> bool) (hi i : int) : int :=
  (fix go (k : nat) (i : int) :=
     match k with O => i | S k => if (i <? hi) && p T.[i] then go k (i + 1) else i end) F i.

Definition skip_ws (hi i : int) : int := skip_while space hi i.

(* The window [lo, hi) without its leading and trailing whitespace. *)
Definition trim (lo hi : int) : int * int :=
  let a := skip_ws hi lo in
  let b := (fix go (k : nat) (j : int) :=
              match k with
              | O => j
              | S k => if (a <? j) && space T.[j - 1] then go k (j - 1) else j
              end) F hi in
  (a, b).

(* The characters of [i, j) as a string. *)
Definition sub (i j : int) : string :=
  (fix go (k : nat) (p : int) (acc : string) :=
     match k with
     | O => acc
     | S k => if i <? p then go k (p - 1) (String (char T.[p - 1]) acc) else acc
     end) F j EmptyString.

(* Whether the characters l are at i, inside a window that ends at hi. The
   scans are written with `if` rather than `&&`, which evaluates both of its
   arguments. *)
Fixpoint starts_codes (hi i : int) (l : list int) : bool :=
  match l with
  | [] => true
  | c :: r => if i <? hi then (if T.[i] =? c then starts_codes hi (i + 1) r else false) else false
  end.

Definition starts (hi i : int) (s : string) : bool := starts_codes hi i (codes s).

(* The same, ignoring the case of letters in the text; s is lower case. *)
Fixpoint starts_ci (hi i : int) (s : string) : bool :=
  match s with
  | EmptyString => true
  | String a r => if i <? hi then (if lower T.[i] =? code a then starts_ci hi (i + 1) r else false)
                  else false
  end.

(* The token of characters l, of length len, at i, not inside a longer name:
   the window [lo, hi) holds no name character just before i or just after
   the token. *)
Definition token_codes (lo hi i : int) (l : list int) (len : int) : bool :=
  if starts_codes hi i l then
    if (lo <? i) && ident_cont T.[i - 1] then false else negb (ident_cont (ch hi (i + len)))
  else false.

Definition token_at (lo hi i : int) (s : string) : bool := token_codes lo hi i (codes s) (slen s).

Definition find_token (lo hi i : int) (s : string) : option int :=
  let l := codes s in
  let len := slen s in
  (fix go (k : nat) (i : int) :=
     match k with
     | O => None
     | S k => if i <? hi then (if token_codes lo hi i l len then Some i else go k (i + 1)) else None
     end) F i.

Definition find_char (hi i c : int) : option int :=
  (fix go (k : nat) (i : int) :=
     match k with
     | O => None
     | S k => if i <? hi then (if T.[i] =? c then Some i else go k (i + 1)) else None
     end) F i.

(* The delimiter that closes the one at start. *)
Definition match_delim (hi start o c : int) : option int :=
  if (start <? hi) && (T.[start] =? o) then
    (fix go (k : nat) (i depth : int) :=
       match k with
       | O => None
       | S k =>
           if i <? hi then
             let x := T.[i] in
             if x =? o then go k (i + 1) (depth + 1)
             else if x =? c then (if depth =? 1 then Some i else go k (i + 1) (depth - 1))
             else go k (i + 1) depth
           else None
       end) F start 0
  else None.

Definition ident (hi i : int) : option (string * int) :=
  if (i <? hi) && ident_start T.[i] then
    let j := skip_while ident_cont hi (i + 1) in Some (sub i j, j)
  else None.

Definition is_ident (a b : int) : bool :=
  match ident b a with Some (_, e) => e =? b | None => false end.

(* The parts of [lo, hi) between the commas outside brackets. *)
Definition split_commas (lo hi : int) : list (int * int) :=
  (fix go (k : nat) (i depth start : int) (acc : list (int * int)) :=
     match k with
     | O => rev ((start, hi) :: acc)
     | S k =>
         if i <? hi then
           let x := T.[i] in
           if (x =? 91) || (x =? 40) || (x =? 123) then go k (i + 1) (depth + 1) start acc
           else if (x =? 93) || (x =? 41) || (x =? 125) then
             go k (i + 1) (if depth =? 0 then 0 else depth - 1) start acc
           else if (x =? 44) && (depth =? 0) then go k (i + 1) depth (i + 1) ((start, i) :: acc)
           else go k (i + 1) depth start acc
         else rev ((start, hi) :: acc)
     end) F lo 0 lo [].

(* The parts of [lo, hi) between the characters c. *)
Definition split_at (c lo hi : int) : list (int * int) :=
  (fix go (k : nat) (i start : int) (acc : list (int * int)) :=
     match k with
     | O => rev ((start, hi) :: acc)
     | S k =>
         if i <? hi then (if T.[i] =? c then go k (i + 1) (i + 1) ((start, i) :: acc)
                          else go k (i + 1) start acc)
         else rev ((start, hi) :: acc)
     end) F lo lo [].

(* ------------------------------------------------------------------ *)
(* Numbers.                                                            *)

Definition all_digits (a b : int) : bool := (a <? b) && (skip_while digit b a =? b).
Definition all_hex (a b : int) : bool := (a <? b) && (skip_while hexdigit b a =? b).

Definition value (base : Z) (a b : int) : Z :=
  (fix go (k : nat) (i : int) (acc : Z) :=
     match k with
     | O => acc
     | S k =>
         if i <? b then
           let c := T.[i] in
           let d := if digit c then (to_Z c - 48)%Z else (to_Z (lower c) - 87)%Z in
           go k (i + 1) (acc * base + d)%Z
         else acc
     end) F a 0%Z.

(* An integer literal: decimal, or hexadecimal after 0x, with an optional
   minus sign and an optional trailing U. *)
Definition parse_int (lo hi : int) : option Z :=
  let '(a, b) := trim lo hi in
  if b <=? a then None else
  let b := if (1 <? b - a) && ((T.[b - 1] =? 117) || (T.[b - 1] =? 85))
              && (let '(a', b') := trim a (b - 1) in a' <? b') then b - 1 else b in
  if starts_ci b a "0x" || starts_ci b a "-0x" then
    let neg := T.[a] =? 45 in
    let d := if neg then a + 3 else a + 2 in
    let '(_, e) := trim d b in
    if all_hex d e then Some (if neg then (- value 16 d e)%Z else value 16 d e) else None
  else if T.[a] =? 45 then (if all_digits (a + 1) b then Some (- value 10 (a + 1) b)%Z else None)
  else if all_digits a b then Some (value 10 a b)
  else None.

(* A decimal floating-point literal: a sign, digits with at most one point,
   and an exponent. *)
Definition float_literal (a b : int) : bool :=
  let i := if (a <? b) && ((T.[a] =? 43) || (T.[a] =? 45)) then a + 1 else a in
  let d1 := skip_while digit b i in
  let point := (d1 <? b) && (T.[d1] =? 46) in
  let d2 := if point then skip_while digit b (d1 + 1) else d1 in
  let mantissa := (i <? d1) || (point && (d1 + 1 <? d2)) in
  let rest :=
    if d2 <? b then
      if (T.[d2] =? 101) || (T.[d2] =? 69) then
        let s := if (d2 + 1 <? b) && ((T.[d2 + 1] =? 43) || (T.[d2 + 1] =? 45)) then d2 + 2
                 else d2 + 1 in
        all_digits s b
      else false
    else true in
  mantissa && rest.

Definition looks_float (a b : int) : bool :=
  let only := skip_while (fun c => digit c || (c =? 43) || (c =? 45) || (c =? 46) || (c =? 101)
                                   || (c =? 69)) b a =? b in
  let point_or_e := (fix go (k : nat) (i : int) :=
                       match k with
                       | O => false
                       | S k => if i <? b then (let c := T.[i] in
                                                (c =? 46) || (c =? 101) || (c =? 69) || go k (i + 1))
                                else false
                       end) F a in
  only && point_or_e.

(* Retain a floating literal's exact spelling instead of erasing its value. *)
Definition immediate (lo hi : int) : option (Z + string) :=
  let '(a, b) := trim lo hi in
  if b <=? a then None
  else if (starts_ci b a "0f" || starts_ci b a "0d") && all_hex (a + 2) b then Some (inr (sub a b))
  else match parse_int a b with
       | Some z => Some (inl z)
       | None => if looks_float a b && float_literal a b then Some (inr (sub a b)) else None
       end.

(* ------------------------------------------------------------------ *)
(* Operands.                                                           *)

(* An operand as the parser classifies it; names are resolved later. *)
Inductive sop :=
| SReg (neg : bool) (r : string)
| SVec (xs : list sop)
| SImm (z : Z)
| SFloat (literal : string)
| SAddr (r : option string) (off : Z)
| STensor (m : string) (cs : list sop)
| SSym (s : string) (off : Z)
| SLabel (s : string)
| SText (s : string).

(* The end of the register name at i: % and a letter or underscore, then
   letters, digits and underscores. *)
Definition reg_end (hi i : int) : option int :=
  if (i + 1 <? hi) && (T.[i] =? 37) && ((T.[i + 1] =? 95) || alpha T.[i + 1]) then
    Some (skip_while (fun c => (c =? 95) || alpha c || digit c) hi (i + 2))
  else None.

Definition is_reg (a b : int) : bool :=
  match reg_end b a with Some e => e =? b | None => false end.

(* An offset suffix, + n or - n, from p: its value and its end. *)
Definition signed_offset (hi p : int) : option (Z * int) :=
  let p := skip_ws hi p in
  if (p <? hi) && ((T.[p] =? 43) || (T.[p] =? 45)) then
    let s := skip_ws hi (p + 1) in
    let e := skip_while (fun c => negb (space c)) hi s in
    match parse_int s e with
    | Some v => Some (if T.[p] =? 45 then (- v)%Z else v, e)
    | None => None
    end
  else None.

(* A name at the start of [a, b), with an optional offset suffix to b. *)
Definition name_offset (name_end : option int) (a b : int) : option (string * Z) :=
  match name_end with
  | None => None
  | Some e =>
      let p := skip_ws b e in
      if p =? b then Some (sub a e, 0%Z)
      else match signed_offset b p with
           | Some (off, q) => if skip_ws b q =? b then Some (sub a e, off) else None
           | None => None
           end
  end.

Definition label_like (a b : int) : bool :=
  ((T.[a] =? 36) && (a + 1 <? b) && (skip_while ident_cont b (a + 1) =? b))
  || starts b a "L_" || starts b a "LBB".

Definition trimmed_parts (lo hi : int) : list (int * int) :=
  filter (fun w => fst w <? snd w) (map (fun w => trim (fst w) (snd w)) (split_commas lo hi)).

Fixpoint classify (k : nat) (a b : int) : sop :=
  match k with
  | O => SText (sub a b)
  | S k =>
      if is_reg a b then SReg false (sub a b)
      else if (T.[a] =? 33) && is_reg (a + 1) b then SReg true (sub (a + 1) b)
      else if (T.[a] =? 123) && (T.[b - 1] =? 125) then
        SVec (map (fun w => classify k (fst w) (snd w)) (trimmed_parts (a + 1) (b - 1)))
      else if (T.[a] =? 40) && (T.[b - 1] =? 41) then
        SVec (map (fun w => classify k (fst w) (snd w)) (trimmed_parts (a + 1) (b - 1)))
      else if (T.[a] =? 91) && (T.[b - 1] =? 93) then
        let '(ia, ib) := trim (a + 1) (b - 1) in
        let other :=
          match name_offset (reg_end ib ia) ia ib with
          | Some (r, off) => SAddr (Some r) off
          | None =>
              match name_offset (match ident ib ia with Some (_, e) => Some e | None => None end)
                                ia ib with
              | Some (s, off) => SSym s off
              | None => SAddr None 0%Z
              end
          end in
        match map (fun w => trim (fst w) (snd w)) (split_commas ia ib) with
        | [(m0, m1); (c0, c1)] =>
            if is_reg m0 m1 && (c0 <? c1) && (T.[c0] =? 123) then
              match classify k c0 c1 with SVec cs => STensor (sub m0 m1) cs | _ => other end
            else other
        | _ => other
        end
      else match immediate a b with
           | Some (inl z) => SImm z
           | Some (inr literal) => SFloat literal
           | None => if label_like a b then SLabel (sub a b) else SText (sub a b)
           end
  end.

(* ------------------------------------------------------------------ *)
(* Instructions.                                                       *)

Record sinst := {
  si_pred : option (bool * string);
  si_op : string;
  si_sfx : list string;
  si_args : list sop
}.

(* A guard, @p or @!p, at a: the guard and where the instruction starts. *)
Definition guard (a b : int) : option (option (bool * string) * int) :=
  let i := a + 1 in
  let neg := (i <? b) && (T.[i] =? 33) in
  let i := if neg then i + 1 else i in
  let e := match reg_end b i with
           | Some e => Some e
           | None => match ident b i with Some (_, e) => Some e | None => None end
           end in
  match e with
  | None => None
  | Some e => if (e <? b) && negb (space T.[e]) then None
              else Some (Some (neg, sub i e), skip_ws b e)
  end.

Definition suffix (w : int * int) : string :=
  let s := sub (fst w) (snd w) in if String.eqb s "shared::cta" then "shared" else s.

(* The instruction in [lo, hi), its semicolon excluded; None for an empty
   statement. *)
Definition instruction (lo hi : int) : res (option sinst) :=
  let '(a, b) := trim lo hi in
  if b <=? a then Ok None
  else if T.[a] =? 46 then Err "PTX directive was not consumed by the body grammar" None
  else
    match (if T.[a] =? 64 then guard a b else Some (None, a)) with
    | None => Err "cannot parse instruction predicate" None
    | Some (pred, r) =>
        let h := skip_while (fun c => negb (space c)) b r in
        if h =? r then Ok None else
        let parts := split_at 46 r h in
        let op := match parts with w :: _ => sub (fst w) (snd w) | [] => EmptyString end in
        let sfx := map suffix (tl parts) in
        let args := trimmed_parts (skip_ws b h) b in
        let ops := map (fun w => classify F (fst w) (snd w)) args in
        let ops :=
          match args, ops with
          | [(x0, x1)], [_] =>
              if String.eqb op "bra" && is_ident x0 x1 then [SLabel (sub x0 x1)] else ops
          | [_; (x0, x1)], [o0; _] =>
              if String.eqb op "brx" && is_ident x0 x1 then [o0; SLabel (sub x0 x1)] else ops
          | _, _ => ops
          end in
        Ok (Some {| si_pred := pred; si_op := op; si_sfx := sfx; si_args := ops |})
    end.

(* ------------------------------------------------------------------ *)
(* Register declarations.                                              *)

Record sreg := { r_name : string; r_width : nat; r_count : option Z; r_scope : N }.

(* The type of a register declaration at p: pred, or b, u, s or f with 8,
   16, 32, 64 or 128 bits, and its end. *)
Definition reg_type (hi p : int) : option (nat * int) :=
  if starts hi p "pred" then Some (1%nat, p + 4)
  else let c := ch hi p in
  if (c =? 98) || (c =? 117) || (c =? 115) || (c =? 102) then
    let q := p + 1 in
    if starts hi q "8" then Some (8%nat, q + 1)
    else if starts hi q "16" then Some (16%nat, q + 2)
    else if starts hi q "32" then Some (32%nat, q + 2)
    else if starts hi q "64" then Some (64%nat, q + 2)
    else if starts hi q "128" then Some (128%nat, q + 3)
    else None
  else None.

(* One register or register array of a declaration: %?name or %?name<count>. *)
Definition reg_item (width : nat) (scope : N) (w : int * int) : res sreg :=
  let '(a, b) := trim (fst w) (snd w) in
  let s := if (a <? b) && (T.[a] =? 37) then a + 1 else a in
  if negb ((s <? b) && ident_start T.[s]) then Err "invalid PTX register declaration" None else
  let e := skip_while ident_cont b (s + 1) in
  let name := sub a e in
  if e =? b then Ok {| r_name := name; r_width := width; r_count := None; r_scope := scope |}
  else if T.[e] =? 60 then
    let d := skip_while digit b (e + 1) in
    if (e + 1 <? d) && (d + 1 =? b) && (T.[d] =? 62) then
      let c := value 10 (e + 1) d in
      if Z.eqb c 0 then Err "empty PTX register array" None
      else Ok {| r_name := name; r_width := width; r_count := Some c; r_scope := scope |}
    else Err "invalid PTX register declaration" None
  else Err "invalid PTX register declaration" None.

Fixpoint all_res {A} (l : list (res A)) : res (list A) :=
  match l with
  | [] => Ok []
  | Ok x :: t => match all_res t with Ok xs => Ok (x :: xs) | Err e i => Err e i end
  | Err e i :: _ => Err e i
  end.

(* The registers that `.reg` declares in [lo, hi), which follows `.reg`. *)
Definition reg_decl (lo hi : int) (scope : N) : res (list sreg) :=
  let p := skip_ws hi lo in
  if negb (ch hi p =? 46) then Err "unmodeled PTX register declaration" None else
  match reg_type hi (p + 1) with
  | None => Err "unmodeled PTX register declaration" None
  | Some (width, e) =>
      let q := skip_ws hi e in
      if (q =? e) || (hi <=? q) then Err "unmodeled PTX register declaration" None
      else all_res (map (reg_item width scope) (split_at 44 q hi))
  end.

(* ------------------------------------------------------------------ *)
(* Directives of a body.                                               *)

Definition ws1 (hi p : int) : option int :=
  let e := skip_ws hi p in if p <? e then Some e else None.

Definition digits1 (hi p : int) : option int :=
  let e := skip_while digit hi p in if p <? e then Some e else None.

(* No word character and no point at e. *)
Definition ends_number (hi e : int) : bool := negb (word (ch hi e) || (ch hi e =? 46)).

Definition then_ {A B} (x : option A) (f : A -> option B) : option B :=
  match x with Some a => f a | None => None end.

(* `<space> .align a .bw name[count];` at p, for the space `.shared` or
   `.local` of `len` characters: the variable, its alignment, its bytes, and
   the end. An array without a count, which an extern shared array is, has
   0 bytes, and a scalar `name;`, as nvcc declares a demoted `__shared__`
   variable, has one element. *)
Definition var_at (hi : int) (space : string) (len : int) (p : int) : option (string * Z * Z * int) :=
  if negb (ch hi p =? 46) then None else
  if negb (starts hi p space) then None else
  then_ (ws1 hi (p + len)) (fun p =>
  if negb (starts hi p ".align") then None else
  then_ (ws1 hi (p + 6)) (fun p =>
  then_ (digits1 hi p) (fun e =>
  let align := value 10 p e in
  then_ (ws1 hi e) (fun p =>
  let c := ch hi (p + 1) in
  if negb ((ch hi p =? 46) && ((c =? 98) || (c =? 117) || (c =? 115) || (c =? 102))) then None else
  let q := p + 2 in
  then_ (if starts hi q "8" then Some (1%Z, q + 1) else if starts hi q "16" then Some (2%Z, q + 2)
         else if starts hi q "32" then Some (4%Z, q + 2) else if starts hi q "64" then Some (8%Z, q + 2)
         else None) (fun '(bytes, p) =>
  then_ (ws1 hi p) (fun p =>
  then_ (ident hi p) (fun '(name, p) =>
  let p := skip_ws hi p in
  if ch hi p =? 59 then Some (name, align, bytes, p + 1) else
  if negb (ch hi p =? 91) then None else
  let p := skip_ws hi (p + 1) in
  let e := skip_while digit hi p in
  let count := if p <? e then value 10 p e else 0%Z in
  let p := skip_ws hi e in
  if negb (ch hi p =? 93) then None else
  let p := skip_ws hi (p + 1) in
  if ch hi p =? 59 then Some (name, align, (bytes * count)%Z, p + 1) else None))))))).

(* Three numbers, each after whitespace, and the end of the last. *)
Definition three_numbers (hi p : int) : option int :=
  then_ (ws1 hi p) (fun p => then_ (digits1 hi p) (fun p =>
  then_ (ws1 hi p) (fun p => then_ (digits1 hi p) (fun p =>
  then_ (ws1 hi p) (fun p => then_ (digits1 hi p) (fun p =>
  if ends_number hi p then Some p else None)))))).

(* `, function_name f [+ offset], inlined_at a b c` after a location. *)
Definition inlined (hi p : int) : option int :=
  let p := skip_ws hi p in
  if negb (ch hi p =? 44) then None else
  let p := skip_ws hi (p + 1) in
  if negb (starts hi p "function_name") then None else
  then_ (ws1 hi (p + 13)) (fun p =>
  then_ (match ident hi p with Some (_, e) => Some e | None => None end) (fun p =>
  let q := skip_ws hi p in
  then_ (if ch hi q =? 43 then
           let s := skip_ws hi (q + 1) in
           if (ch hi s =? 48) && ((ch hi (s + 1) =? 120) || (ch hi (s + 1) =? 88))
              && hexdigit (ch hi (s + 2))
           then Some (skip_while hexdigit hi (s + 2))
           else digits1 hi s
         else Some p) (fun p =>
  let p := skip_ws hi p in
  if negb (ch hi p =? 44) then None else
  let p := skip_ws hi (p + 1) in
  if negb (starts hi p "inlined_at") then None else
  three_numbers hi (p + 10)))).

(* A string, from its opening quote at p, and the index after its closing
   quote. *)
Definition quoted (hi p : int) : option int :=
  if ch hi p =? 34 then then_ (find_char hi (p + 1) 34) (fun e => Some (e + 1)) else None.

(* `"s" [, "s"]* ;` from p. *)
Definition pragma_args (hi p : int) : option int :=
  then_ (quoted hi (skip_ws hi p)) (fun p =>
  (fix go (k : nat) (p : int) :=
     match k with
     | O => None
     | S k =>
         let q := skip_ws hi p in
         if ch hi q =? 44 then then_ (quoted hi (skip_ws hi (q + 1))) (go k)
         else if ch hi q =? 59 then Some (q + 1)
         else None
     end) F p).

(* `n [, n [, n]]` after whitespace, as far as a number may end. *)
Definition dims_args (hi p : int) : option int :=
  then_ (ws1 hi p) (fun p => then_ (digits1 hi p) (fun e0 =>
  let more := fun e => let s := skip_ws hi e in
                       if ch hi s =? 44 then digits1 hi (skip_ws hi (s + 1)) else None in
  let e1 := more e0 in
  let e2 := then_ e1 more in
  let fits := fun e => match e with Some e => ends_number hi e | None => false end in
  if fits e2 then e2 else if fits e1 then e1 else if ends_number hi e0 then Some e0 else None)).

(* The end of the body directive at pos, which starts with a point. *)
Definition directive (hi pos : int) : res int :=
  let q := skip_while word hi (pos + 1) in
  let name := sub (pos + 1) q in
  let unmodeled := Err (String.append "unmodeled PTX body directive: ." name) None in
  if sin name ["loc"; "loc_intermediate"] then
    match three_numbers hi q with
    | None => unmodeled
    | Some e =>
        if ch hi (skip_ws hi e) =? 44 then
          match inlined hi e with
          | Some e' => if String.eqb name "loc" then Ok e'
                       else Err "invalid inline PTX location directive" None
          | None => Err "invalid inline PTX location directive" None
          end
        else Ok e
    end
  else if String.eqb name "pragma" then
    match pragma_args hi q with Some e => Ok e | None => unmodeled end
  else if sin name ["reg"; "local"; "shared"; "param"; "const"; "global"; "align"] then
    match find_char hi q 59 with
    | Some e => Ok (e + 1)
    | None => Err "unterminated PTX variable declaration" None
    end
  else if sin name ["reqntid"; "maxntid"; "reqnctapercluster"] then
    match dims_args hi q with Some e => Ok e | None => unmodeled end
  else if sin name ["maxnreg"; "minnctapersm"; "maxnctapersm"; "maxclusterrank";
                      "abi_preserve"; "abi_preserve_control"] then
    match then_ (ws1 hi q) (fun p => then_ (digits1 hi p) (fun e =>
            if ends_number hi e then Some e else None)) with
    | Some e => Ok e
    | None => unmodeled
    end
  else if sin name ["explicitcluster"; "blocksareclusters"; "noreturn"] then Ok q
  else unmodeled.

(* ------------------------------------------------------------------ *)
(* Bodies.                                                             *)

Record sbody := {
  sb_insts : list (sinst * list N);        (* each instruction and its scopes, outermost first *)
  sb_labels : list (string * N * N);       (* each label, the instruction it precedes, its scope *)
  sb_targets : list (string * list string * list N);
                                           (* each .branchtargets list, its labels, its scopes *)
  sb_regs : list sreg;
  sb_locals : list (string * Z * Z)        (* each local variable, its alignment, and its bytes *)
}.

Definition empty_body : sbody :=
  {| sb_insts := []; sb_labels := []; sb_targets := []; sb_regs := []; sb_locals := [] |}.

Fixpoint names_distinct (l : list (string * Z * Z)) : bool :=
  match l with
  | [] => true
  | (x, _, _) :: t => negb (existsb (fun e => let '(y, _, _) := e in String.eqb x y) t) && names_distinct t
  end.

(* The statements of the body [lo, hi). `scopes` holds the enclosing scopes,
   innermost first; each `{` opens scope `next`. The lists accumulate in
   reverse. *)
Fixpoint body_go (k : nat) (lo hi pos : int) (scopes : list N) (next count : N) (acc : sbody)
    : res sbody :=
  match k with
  | O => Err "the PTX body is longer than the parser reads" None
  | S k =>
      let pos := skip_ws hi pos in
      let scope := hd 0%N scopes in
      let path := rev scopes in
      if hi <=? pos then
        match scopes with
        | [_] => if names_distinct (sb_locals acc) then
                   Ok {| sb_insts := rev (sb_insts acc); sb_labels := rev (sb_labels acc);
                         sb_targets := rev (sb_targets acc); sb_regs := sb_regs acc;
                         sb_locals := rev (sb_locals acc) |}
                 else Err "a local variable declared twice in one function" None
        | _ => Err "unbalanced PTX instruction scope" None
        end
      else
        let c := T.[pos] in
        if c =? 123 then body_go k lo hi (pos + 1) (next :: scopes) (N.succ next) count acc
        else if c =? 125 then
          match scopes with
          | _ :: (_ :: _) as outer => body_go k lo hi (pos + 1) outer next count acc
          | _ => Err "unbalanced PTX instruction scope" None
          end
        else if c =? 59 then body_go k lo hi (pos + 1) scopes next count acc
        else if token_at lo hi pos ".reg" then
          match find_char hi pos 59 with
          | None => Err "unterminated PTX register declaration" None
          | Some e =>
              match reg_decl (pos + 4) e scope with
              | Err m i => Err m i
              | Ok regs =>
                  body_go k lo hi (e + 1) scopes next count
                    {| sb_insts := sb_insts acc; sb_labels := sb_labels acc;
                       sb_targets := sb_targets acc; sb_regs := app (sb_regs acc) regs;
                       sb_locals := sb_locals acc |}
              end
          end
        else if c =? 46 then
          match var_at hi ".local" 6 pos with
          | Some (name, al, bytes, e) =>
              body_go k lo hi e scopes next count
                {| sb_insts := sb_insts acc; sb_labels := sb_labels acc;
                   sb_targets := sb_targets acc; sb_regs := sb_regs acc;
                   sb_locals := (name, al, bytes) :: sb_locals acc |}
          | None =>
              match directive hi pos with
              | Err m i => Err m i
              | Ok e => body_go k lo hi e scopes next count acc
              end
          end
        else
          let label :=
            match ident hi pos with
            | Some (name, e) =>
                let colon := skip_ws hi e in
                if (ch hi colon =? 58) && negb (ch hi (colon + 1) =? 58) then Some (name, colon)
                else None
            | None => None
            end in
          match label with
          | Some (name, colon) =>
              let after := skip_ws hi (colon + 1) in
              if token_at lo hi after ".branchtargets" then
                match find_char hi after 59 with
                | None => Err "unterminated .branchtargets list" None
                | Some close =>
                    let ts := map (fun w => trim (fst w) (snd w)) (split_at 44 (after + 14) close) in
                    if forallb (fun w => is_ident (fst w) (snd w)) ts then
                      body_go k lo hi (close + 1) scopes next count
                        {| sb_insts := sb_insts acc; sb_labels := sb_labels acc;
                           sb_targets := (name, map (fun w => sub (fst w) (snd w)) ts, path)
                                         :: sb_targets acc;
                           sb_regs := sb_regs acc; sb_locals := sb_locals acc |}
                    else Err "invalid .branchtargets list" None
                end
              else
                body_go k lo hi (colon + 1) scopes next count
                  {| sb_insts := sb_insts acc; sb_labels := (name, count, scope) :: sb_labels acc;
                     sb_targets := sb_targets acc; sb_regs := sb_regs acc; sb_locals := sb_locals acc |}
          | None =>
              match find_char hi pos 59 with
              | None => Err "unterminated PTX instruction" None
              | Some e =>
                  match instruction pos e with
                  | Err m i => Err m i
                  | Ok None => body_go k lo hi (e + 1) scopes next count acc
                  | Ok (Some i) =>
                      body_go k lo hi (e + 1) scopes next (N.succ count)
                        {| sb_insts := (i, path) :: sb_insts acc; sb_labels := sb_labels acc;
                           sb_targets := sb_targets acc; sb_regs := sb_regs acc; sb_locals := sb_locals acc |}
                  end
              end
          end
  end.

Definition body (lo hi : int) : res sbody := body_go F lo hi lo [0%N] 1%N 0%N empty_body.

(* ------------------------------------------------------------------ *)
(* Kernel directives.                                                  *)

(* The one to three integers after the directive in [lo, hi), the missing
   ones 1: up to the end of the directive's line, without a trailing
   semicolon, as far as they are integers. *)
Definition dims (lo hi : int) (d : string) : option (Z * Z * Z) :=
  match find_token lo hi lo d with
  | None => None
  | Some pos =>
      let i := skip_ws hi (pos + slen d) in
      let le := match find_char hi i 10 with Some e => e | None => hi end in
      let '(a, b) := trim i le in
      let b := (fix go (k : nat) (j : int) :=
                  match k with
                  | O => j
                  | S k => if (a <? j) && (T.[j - 1] =? 59) then go k (j - 1) else j
                  end) F b in
      let vals := (fix take (l : list (int * int)) :=
                     match l with
                     | [] => []
                     | w :: t => match parse_int (fst w) (snd w) with
                                 | Some v => v :: take t
                                 | None => []
                                 end
                     end) (split_commas a b) in
      match vals with
      | [] => None
      | v0 :: rest => Some (v0, nth 0 rest 1%Z, nth 1 rest 1%Z)
      end
  end.

(* Whether the directives in [lo, hi) allow a cluster of more than one CTA. *)
Definition cluster (lo hi : int) : bool :=
  let d := dims lo hi ".reqnctapercluster" in
  let r := dims lo hi ".maxclusterrank" in
  let explicit := match find_token lo hi lo ".explicitcluster" with Some _ => true | None => false end in
  match d with Some (x, y, z) => Z.ltb 1 (x * y * z) | None => false end
  || match r with Some (x, _, _) => Z.ltb 1 x | None => false end
  || (explicit && match d with None => true | Some _ => false end).

(* ------------------------------------------------------------------ *)
(* Parameters.                                                         *)

(* The tokens of [lo, hi) between spaces, tabs and line breaks. *)
Definition tokens (lo hi : int) : list string :=
  let sep := fun c => (c =? 32) || (c =? 10) || (c =? 9) in
  (fix go (k : nat) (i : int) (acc : list string) :=
     match k with
     | O => rev acc
     | S k =>
         let i := skip_while sep hi i in
         if i <? hi then let e := skip_while (fun c => negb (sep c)) hi i in go k e (sub i e :: acc)
         else rev acc
     end) F lo [].

End Scan.

(* The name and type of a parameter declaration, as the reference parser
   reads them from its tokens. *)
Definition all_digits_s (s : string) : bool :=
  match s with
  | EmptyString => false
  | _ => forallb (fun a => digit (code a)) (list_ascii_of_string s)
  end.

Definition ident_s (s : string) : bool :=
  match list_ascii_of_string s with
  | [] => false
  | a :: t => ident_start (code a) && forallb (fun a => ident_cont (code a)) t
  end.

Fixpoint before_bracket (s : string) : string :=
  match s with
  | EmptyString => EmptyString
  | String a t => if Ascii.eqb a "["%char then EmptyString else String a (before_bracket t)
  end.

(* A parameter's name without its array suffix, name[n]. *)
Definition strip_array (s : string) : string :=
  let l := list_ascii_of_string s in
  if existsb (fun a => Ascii.eqb a "["%char) l
     && match rev l with a :: _ => Ascii.eqb a "]"%char | [] => false end
  then before_bracket s else s.

Definition dotted (s : string) : option string :=
  match s with String a t => if Ascii.eqb a "."%char then Some t else None | EmptyString => None end.

Definition param_decl (toks : list string) : option (string * string) :=
  if negb (sin ".param" toks) then None else
  let first := (fix go (l : list string) :=
                  match l with
                  | [] => None
                  | t :: r => match dotted t with
                              | Some w => if sin w ["param"; "ptr"; "align"; "global"; "shared";
                                                       "const"; "local"; "generic"]
                                             || all_digits_s w then go r else Some w
                              | None => go r
                              end
                  end) toks in
  let second := (fix go (l : list string) :=
                   match l with
                   | [] => None
                   | t :: r => match dotted t with
                               | Some w => if sin t [".param"; ".ptr"; ".align"]
                                              || sin w ["global"; "shared"; "const"; "local";
                                                          "param"; "generic"] then go r else Some w
                               | None => go r
                               end
                   end) toks in
  let ty := match first with
            | Some w => if String.eqb w "" then second else Some w
            | None => second
            end in
  match ty with
  | None => None
  | Some ty =>
      if String.eqb ty "" then None else
      match find (fun t => negb (String.prefix "." t) && ident_s (strip_array t)) (rev toks) with
      | Some t => Some (strip_array t, ty)
      | None => None
      end
  end.

(* ------------------------------------------------------------------ *)
(* Modules.                                                            *)

Record skernel := {
  sk_name : string;
  sk_entry : bool;
  sk_has_body : bool;
  sk_params : list (string * string);
  sk_rets : list string;
  sk_reqntid : option (Z * Z * Z);
  sk_cluster : bool;
  sk_body : sbody
}.

Record smodule := {
  sm_address_size : Z;
  sm_target : option Z;
  sm_ptx_version : option (Z * Z);
  sm_shared : list (string * Z * Z);       (* shared variables, their alignments, and their bytes *)
  sm_kernels : list skernel
}.

(* Recognized NVIDIA target spellings, independent of input kernels. Keep
   unsupported/future targets fail closed rather than interpreting a prefix.
   compute_ is the documented synonym; architecture/family variants are
   recognized only for the architectures that define them. *)
Definition target_architecture (name : string) : option Z :=
 let suffix := if String.prefix "sm_" name then Some (drop 3 name)
               else if String.prefix "compute_" name then Some (drop 8 name)
               else None in
 match suffix with
 | None => None
 | Some suffix =>
   let parts := match rev (list_ascii_of_string suffix) with
                | "a"%char :: rest => (string_of_list_ascii (rev rest), 1%nat)
                | "f"%char :: rest => (string_of_list_ascii (rev rest), 2%nat)
                | _ => (suffix, 0%nat)
                end in
   match decimal (fst parts) with
   | None => None
   | Some arch =>
     let known := match snd parts with
       | O => existsb (Z.eqb arch) [10;11;12;13;20;21;30;32;35;37;50;52;53;60;61;62;
                                      70;72;75;80;86;87;89;90;100;101;103;107;110;120;121]%Z
       | S O => existsb (Z.eqb arch) [90;100;101;103;107;110;120;121]%Z
       | _ => existsb (Z.eqb arch) [100;101;103;107;110;120;121]%Z
       end in
     if known then Some arch else None
   end
 end.

Definition ptx_version (s : string) : option (Z * Z) :=
 match split_on "."%char s "" with
 | [major; minor] =>
   match decimal major, decimal minor with
   | Some a, Some b => if Z.ltb 0 a && Z.leb 0 b && Z.leb b 9 then Some (a,b) else None
   | _, _ => None
   end
 | _ => None
 end.

Section Module.

Variable T : array int.
Variable F : nat.
Variable n : int.

Definition params_of (lo hi : int) : list (string * string) :=
  flat_map (fun w => match param_decl (tokens T F (fst w) (snd w)) with
                     | Some p => [p]
                     | None => []
                     end) (split_commas T F lo hi).

(* A directive's value: the text after its first occurrence up to
   whitespace or punctuation. *)
Definition directive_value (d : string) : option (int * int) :=
  match find_token T F 0 n 0 d with
  | None => None
  | Some p =>
      let i := skip_ws T F n (p + slen d) in
      let j := skip_while T F (fun c => negb (space c || (c =? 44) || (c =? 59) || (c =? 123)
                                            || (c =? 125) || (c =? 40) || (c =? 41))) n i in
      if i <? j then Some (i, j) else None
  end.

(* Retargeting is legal PTX, but this narrow module-wide model requires
   exactly one initial version/target pair. Comments and quoted text have
   already been removed from structural consideration by strip_go. *)
Definition module_architecture : res (option Z * option (Z * Z)) :=
 match find_token T F 0 n 0 ".version", find_token T F 0 n 0 ".target" with
 | None, None => Ok (None, None)
 | Some vp, Some tp =>
   match find_token T F 0 n (vp + slen ".version") ".version",
         find_token T F 0 n (tp + slen ".target") ".target" with
   | None, None =>
     let ve := match find_char T F n vp 10 with Some e => e | None => n end in
     let te := match find_char T F n tp 10 with Some e => e | None => n end in
     if negb ((skip_ws T F n 0 =? vp) && (skip_ws T F n ve =? tp)) then
       Err "version and target must be the initial module directives" None else
     match tokens T F (vp + slen ".version") ve,
           map (fun '(a,b) => let '(a,b) := trim T F a b in sub T F a b)
               (split_commas T F (tp + slen ".target") te) with
     | [version], arch :: options =>
       if negb (forallb (fun o => sin o ["texmode_unified"; "texmode_independent";
                                         "map_f64_to_f32"; "debug"]) options) then
         Err "an unsupported target option" None else
       if (existsb (String.eqb "texmode_unified") options &&
           existsb (String.eqb "texmode_independent") options) ||
          negb (names_distinct (map (fun x => (x,0%Z,0%Z)) options)) then
         Err "duplicate or conflicting target options" None else
       match ptx_version version, target_architecture arch with
       | Some version, Some arch => Ok (Some arch, Some version)
       | _, _ => Err "an invalid PTX version or target architecture" None
       end
     | _, _ => Err "an incomplete PTX version or target directive" None
     end
   | _, _ => Err "multiple PTX version or target directives are unsupported" None
   end
 | _, _ => Err "PTX version and target must be specified together" None
 end.

(* The shared variables of the module and their alignments. *)
(* Each shared variable of the module with its alignment and bytes; an
   array of unknown size, such as dynamic shared memory, has none. *)
Definition shared_vars : list (string * Z * Z) :=
  (fix go (k : nat) (p : int) (acc : list (string * Z * Z)) :=
     match k with
     | O => rev acc
     | S k => if p <? n then
                match var_at T F n ".shared" 7 p with
                | Some (name, a, bytes, e) => go k e ((name, a, bytes) :: acc)
                | None => go k (p + 1) acc
                end
              else rev acc
     end) F 0 [].

(* The declaration of a kernel at start: its name, the parentheses of its
   parameter list, and a device function's return list. A function without
   parameters has the empty window before its body or semicolon. *)
Definition decl_at (start : int) (entry : bool) : option (string * int * int * option (int * int)) :=
  let i := skip_ws T F n (start + (if entry then 6 else 5)) in
  let rets := if negb entry && (ch T n i =? 40) then
                match match_delim T F n i 40 41 with
                | Some e => Some (Some (i + 1, e), skip_ws T F n (e + 1))
                | None => None
                end
              else Some (None, i) in
  then_ rets (fun '(rets, i) =>
  then_ (ident T F n i) (fun '(name, j) =>
  let i := skip_ws T F n j in
  if ch T n i =? 40 then
    then_ (match_delim T F n i 40 41) (fun e => Some (name, i, e, rets))
  else if negb entry && (i <? n) && ((ch T n i =? 123) || (ch T n i =? 59) || (ch T n i =? 46)) then
    Some (name, i - 1, i - 1, rets)
  else then_ (find_char T F n i 40) (fun i =>
       then_ (match_delim T F n i 40 41) (fun e => Some (name, i, e, rets))))).

(* The next kernel declaration from pos. *)
Definition next_decl (pos : int) : option (int * bool * (string * int * int * option (int * int))) :=
  (fix go (k : nat) (pos : int) :=
     match k with
     | O => None
     | S k =>
         if pos <? n then
           let e := find_token T F 0 n pos ".entry" in
           let f := find_token T F 0 n pos ".func" in
           let first := match e, f with
                        | Some e, Some f => if e <? f then Some (e, true) else Some (f, false)
                        | Some e, None => Some (e, true)
                        | None, Some f => Some (f, false)
                        | None, None => None
                        end in
           match first with
           | None => None
           | Some (start, entry) =>
               match decl_at start entry with
               | Some d => Some (start, entry, d)
               | None => go k (start + 1)
               end
           end
         else None
     end) F pos.

Fixpoint kernels_go (k : nat) (pos : int) (acc : list skernel) : res (list skernel) :=
  match k with
  | O => Err "the PTX module is longer than the parser reads" None
  | S k =>
      if pos <? n then
        match next_decl pos with
        | None => Ok (rev acc)
        | Some (start, entry, (name, ps, pe, rets)) =>
            let params := params_of (ps + 1) pe in
            let rets := match rets with
                        | Some (a, b) => map fst (params_of a b)
                        | None => []
                        end in
            let semi := find_char T F n (pe + 1) 59 in
            let brace := find_char T F n (pe + 1) 123 in
            let decl := match semi, brace with
                        | Some s, Some b => s <? b
                        | Some _, None => true
                        | None, _ => false
                        end in
            let bend := if decl then None
                        else match brace with Some b => match_delim T F n b 123 125 | None => None end in
            let kernel := fun reqntid cl bd =>
              {| sk_name := name; sk_entry := entry; sk_has_body := negb decl; sk_params := params;
                 sk_rets := rets; sk_reqntid := reqntid; sk_cluster := cl; sk_body := bd |} in
            let next := fun pos' => if pos' <=? start then start + 1 else pos' in
            match brace, bend with
            | Some b, Some e =>
                match body T F (b + 1) e with
                | Err m i => Err m i
                | Ok bd =>
                    let reqntid := match dims T F (pe + 1) b ".reqntid" with
                                   | Some d => Some d
                                   | None => dims T F (b + 1) e ".reqntid"
                                   end in
                    let cl := cluster T F (pe + 1) b || cluster T F (b + 1) e in
                    kernels_go k (next (e + 1)) (kernel reqntid cl bd :: acc)
                end
            | _, _ =>
                let after := match decl, semi with true, Some s => s | _, _ => pe end in
                kernels_go k (next (after + 1)) (kernel None false empty_body :: acc)
            end
        end
      else Ok (rev acc)
  end.

Definition parse_module : res smodule :=
  match module_architecture with
  | Err e i => Err e i
  | Ok (target, version) =>
  let size := match directive_value ".address_size" with
              | Some (a, b) => match parse_int T F a b with
                               | Some z => if Z.eqb z 0 then 32%Z else z
                               | None => 32%Z
                               end
              | None => 32%Z
              end in
  (* one register holds each shared variable's address, so two variables
     must not share a name *)
  if negb (names_distinct shared_vars) then Err "a shared variable name is declared twice" None else
  match kernels_go F 0 [] with
  | Err m i => Err m i
  | Ok ks => Ok {| sm_address_size := size; sm_target := target; sm_ptx_version := version;
                         sm_shared := shared_vars; sm_kernels := ks |}
  end
  end.

End Module.

(* ------------------------------------------------------------------ *)
(* Names and kernels.                                                  *)

(* A name that scope s declares, made distinct from every PTX name. *)
Definition scoped (name : string) (s : N) : string :=
  if N.eqb s 0 then name
  else match name with
       | String a t => if Ascii.eqb a "%"%char then String.append "%s" (String.append (nstr s) (String.append "$" t))
                       else String.append "s" (String.append (nstr s) (String.append "$" name))
       | EmptyString => String.append "s" (String.append (nstr s) "$")
       end.

Section Names.

Variable regs : list sreg.
Variable labels : list (string * N).          (* each label or .branchtargets name and its scope *)

(* Whether declaration d names register r, as `names` of GateFront.v
   decides, testing the prefix first. *)
Definition declares (d : sreg) (r : string) : bool :=
  match r_count d with
  | None => String.eqb r (r_name d)
  | Some c => if String.prefix (r_name d) r then
                match decimal (drop (String.length (r_name d)) r) with
                | Some v => Z.ltb v c
                | None => false
                end
              else false
  end.

(* The innermost scope of the path that declares register r. *)
Definition reg_scope (r : string) (path : list N) : option N :=
  find (fun s => existsb (fun d => if N.eqb (r_scope d) s then declares d r else false) regs)
       (rev path).

Definition rename (r : string) (path : list N) : string :=
  match reg_scope r path with Some s => scoped r s | None => r end.

Definition rename_label (l : string) (path : list N) : string :=
  match find (fun s => existsb (fun e => if N.eqb (snd e) s then String.eqb (fst e) l else false)
                               labels) (rev path) with
  | Some s => scoped l s
  | None => l
  end.

Fixpoint split_bar (s : string) (acc : string) : list string :=
  match s with
  | EmptyString => [acc]
  | String a t => if Ascii.eqb a "|"%char then acc :: split_bar t EmptyString
                  else split_bar t (String.append acc (String a EmptyString))
  end.

Fixpoint join_bar (l : list string) : string :=
  match l with [] => "" | [x] => x | x :: t => String.append x (String.append "|" (join_bar t)) end.

Fixpoint opnd (path : list N) (o : sop) : popnd :=
  match o with
  | SReg neg r => if neg then PNot (rename r path) else PReg (rename r path)
  | SVec xs => PVec (map (opnd path) xs)
  | SImm z => PImm z
  | SFloat literal => PFloat literal
  | SAddr r off => PAddr (option_map (fun r => rename r path) r) off
  | STensor m cs => PTensor (rename m path) (map (opnd path) cs)
  | SSym s off => PSym s off
  | SLabel s => match reg_scope s path with
                | Some _ => PReg (rename s path)
                | None => PLabel (rename_label s path)
                end
  | SText s => match reg_scope s path with
               | Some _ => PReg (rename s path)
               | None => PText (join_bar (map (fun p => rename p path) (split_bar s EmptyString)))
               end
  end.

Definition pinstr_of (x : sinst * list N) : pinstr :=
  let '(i, path) := x in
  PI (option_map (fun '(neg, r) => (neg, rename r path)) (si_pred i)) (si_op i) (si_sfx i)
     (map (opnd path) (si_args i)).

End Names.

Definition body_labels (b : sbody) : list (string * N) :=
  app (map (fun '(name, _, s) => (name, s)) (sb_labels b))
      (map (fun '(name, _, path) => (name, last path 0%N)) (sb_targets b)).

Definition regs_of (b : sbody) : list (string * nat * option Z) :=
  map (fun d => (scoped (r_name d) (r_scope d), r_width d, r_count d)) (sb_regs b).

Definition labels_of (b : sbody) : list (string * N) :=
  map (fun '(name, i, s) => (scoped name s, i)) (sb_labels b).

Definition targets_of (b : sbody) : list (string * list string) :=
  map (fun '(name, ts, path) =>
         (scoped name (last path 0%N), map (fun t => rename_label (body_labels b) t path) ts))
      (sb_targets b).

Definition insts_of (b : sbody) : list pinstr :=
  map (pinstr_of (sb_regs b) (body_labels b)) (sb_insts b).

Definition pfunc_of (k : skernel) : pfunc :=
  let b := sk_body k in
  {| f_name := sk_name k; f_params := map fst (sk_params k); f_rets := sk_rets k;
     f_regs := regs_of b; f_locals := sb_locals b; f_labels := labels_of b; f_branch_targets := targets_of b;
     f_insts := insts_of b |}.

Definition pkernel_of (m : smodule) (k : skernel) : pkernel :=
  let b := sk_body k in
  {| k_address_size := sm_address_size m; k_target := sm_target m;
     k_ptx_version := sm_ptx_version m; k_params := sk_params k; k_regs := regs_of b;
     k_shared := sm_shared m; k_locals := map (fun '(x, a, s) => (x, a, s, [])) (sb_locals b);
     k_reqntid := sk_reqntid k; k_cluster := sk_cluster k;
     k_labels := labels_of b; k_branch_targets := targets_of b;
     k_funcs := map pfunc_of (filter (fun f => negb (sk_entry f) && sk_has_body f) (sm_kernels m));
     k_insts := insts_of b |}.

(* The kernels of a module: the names of its entries and of its functions
   with a body, and each entry kernel with the module's device functions. *)
Record kernels := {
  ks_names : list string;
  ks_entries : list (string * pkernel)
}.

Definition kernels_of (m : smodule) : kernels :=
  {| ks_names := map sk_name (filter (fun k => sk_entry k || sk_has_body k) (sm_kernels m));
     ks_entries := map (fun k => (sk_name k, pkernel_of m k)) (filter sk_entry (sm_kernels m)) |}.

(* The kernels of the packed text of n characters. *)
Definition parse_text (a : array int) (n : int) : res kernels :=
  if max_length <? n then Err "the PTX text is longer than the parser reads" None else
  let F := fuel_of n in
  match strip_go F F (unpack a n) n 0 (PArray.make n 0) 0 with
  | Err e i => Err e i
  | Ok (T, m) =>
      match parse_module T (fuel_of m) m with
      | Err e i => Err e i
      | Ok md => Ok (kernels_of md)
      end
  end.

(* ------------------------------------------------------------------ *)
(* Verification.                                                         *)

Definition lookup (mfs : list (string * binding)) (name : string) : option binding :=
  match find (fun e => String.eqb (fst e) name) mfs with Some (_, mf) => Some mf | None => None end.

Fixpoint join (sep : string) (l : list string) : string :=
  match l with [] => "" | [x] => x | x :: t => String.append x (String.append sep (join sep t)) end.

(* Each entry kernel accepted by `verify_ptx` with its binding. *)
Fixpoint verify_entries (mfs : list (string * binding)) (l : list (string * pkernel)) : res unit :=
  match l with
  | [] => Ok tt
  | (name, K) :: t =>
      match lookup mfs name with
      | None => Err (String.append "kernel '" (String.append name "' has no binding")) None
      | Some mf => if verify_ptx K mf then verify_entries mfs t
                   else Err (String.append name ": Rocq does not accept the kernel") None
      end
  end.

(* Accepts the module of the packed text with its bindings, one for each
   entry kernel: each entry kernel is accepted by `verify_ptx` with its
   binding. *)
Definition verify_text (a : array int) (n : int) (mfs : list (string * binding)) : res unit :=
  match parse_text a n with
  | Err e i => Err e i
  | Ok ks =>
      if negb (distinct (ks_names ks)) then
        Err (String.append "the module defines a kernel twice: " (join ", " (ks_names ks))) None
      else
        let missing := filter (fun e => negb (sin (fst e) (map fst (ks_entries ks)))) mfs in
        match missing with
        | _ :: _ => Err (String.append "the module lacks the kernels " (join ", " (map fst missing))) None
        | [] => verify_entries mfs (ks_entries ks)
        end
  end.

Definition verify_text_ok (a : array int) (n : int) (mfs : list (string * binding)) : bool :=
  match verify_text a n mfs with Ok _ => true | Err _ _ => false end.

Lemma verify_entries_ok : forall mfs l, verify_entries mfs l = Ok tt ->
    forall name K, In (name, K) l -> exists mf, lookup mfs name = Some mf /\ verify_ptx K mf = true.
Proof.
  intros mfs l. induction l as [| [name0 K0] t IH]; intros H name K Hin; [destruct Hin |].
  simpl in H. destruct (lookup mfs name0) as [mf |] eqn:Hl; [| discriminate].
  destruct (verify_ptx K0 mf) eqn:Ha; [| discriminate].
  destruct Hin as [E | Hin].
  - inversion E; subst. exists mf. split; assumption.
  - exact (IH H name K Hin).
Qed.

(* Each entry kernel of an accepted text is accepted with its binding, so
   every global access of every thread of it stays in its region, and its
   launch facts hold of what a launch of the binding puts in its registers. *)
Theorem verify_text_sound : forall a n mfs, verify_text_ok a n mfs = true ->
    exists ks, parse_text a n = Ok ks /\
      forall name K, In (name, K) (ks_entries ks) ->
        exists mf, lookup mfs name = Some mf /\
          exists prog F, front K mf = Ok (prog, F) /\ check_scalar F prog = true /\ kernel_safe F prog /\
            link_ok F mf = true.
Proof.
  intros a n mfs H. unfold verify_text_ok, verify_text in H.
  destruct (parse_text a n) as [ks | e i]; [| discriminate].
  exists ks. split; [reflexivity |].
  destruct (negb (distinct (ks_names ks))); [discriminate |].
  destruct (filter _ mfs); [| discriminate].
  destruct (verify_entries mfs (ks_entries ks)) as [[] | e i] eqn:E; [| discriminate].
  intros name K Hin.
  destruct (verify_entries_ok mfs (ks_entries ks) E name K Hin) as [mf [Hl Ha]].
  exists mf. split; [exact Hl | exact (verify_ptx_sound K mf Ha)].
Qed.

(* ------------------------------------------------------------------ *)
(* Diagnostics.                                                        *)

Definition zstr (z : Z) : string :=
  if Z.ltb z 0 then String.append "-" (nstr (Z.to_N (- z))) else nstr (Z.to_N z).

Fixpoint show_opnd (o : popnd) : string :=
  match o with
  | PReg r => r
  | PNot r => String.append "!" r
  | PVec xs => String.append "{" (String.append (join ", " (map show_opnd xs)) "}")
  | PImm z => zstr z
  | PFloat literal => literal
  | PAddr (Some r) off => String.append "[" (String.append r (String.append "+" (String.append (zstr off) "]")))
  | PAddr None _ => "[an address]"
  | PSym s off => String.append "[" (String.append s (String.append "+" (String.append (zstr off) "]")))
  | PTensor m cs =>
      String.append "[" (String.append m (String.append ", {" (String.append (join ", " (map show_opnd cs)) "}]")))
  | PLabel s | PText s => s
  end.

(* An instruction as the parser read it. *)
Definition show_pinstr (i : pinstr) : string :=
  let '(PI pred op sfx args) := i in
  let guard := match pred with
               | Some (neg, r) => String.append "@" (String.append (if neg then "!" else "") (String.append r " "))
               | None => ""
               end in
  String.append guard (String.append op (String.append
    (fold_right (fun s acc => String.append "." (String.append s acc)) "" sfx)
    (String.append " " (join ", " (map show_opnd args))))).

(* The diagnosis of an entry kernel with its binding: the kernel, why the
   translation or the check refuses it, and the instruction where it does. *)
Definition kernel_refusal (mfs : list (string * binding)) (x : string * pkernel) : option string :=
  let (name, K) := x in
  match lookup mfs name with
  | Some mf =>
      match refusal K mf with
      | Some (r, Some i) =>
          let at_ := match nth_error (k_insts K) (N.to_nat i) with
                     | Some ins => show_pinstr ins
                     | None => String.append "instruction " (nstr i)
                     end in
          Some (String.append name (String.append ": " (String.append r (String.append ": " at_))))
      | Some (r, None) => Some (String.append name (String.append ": " r))
      | None => None
      end
  | None => None
  end.

(* Why the text is not accepted, for diagnostics: the diagnosis of the first
   entry kernel that it refuses, which analyzes each kernel once, or else the
   reason of verify_text, as for a kernel defined twice or a link the
   binding does not give. None when the text is accepted. *)
Definition text_refusal (a : array int) (n : int) (mfs : list (string * binding)) : option string :=
  match parse_text a n with
  | Err e _ => Some e
  | Ok ks =>
      match fold_left (fun acc x => match acc with Some _ => acc | None => kernel_refusal mfs x end)
                      (ks_entries ks) None with
      | Some s => Some s
      | None => match verify_text a n mfs with Ok _ => None | Err e _ => Some e end
      end
  end.


(* ------------------------------------------------------------------ *)
(* Explicit certificates: proposal, independent checking, and diagnostics.
   The parser, frontend and binding link are recomputed by the checker.
   Entry names and certificate positions must cover the module exactly. *)
Definition text_certificate := list (string * scalar_certificate).

(* A missing side cannot satisfy the block checker, independently of the
   front end. Inspect that certificate shape before the link, prologue,
   and termination checks. *)
Definition certificate_has_cases := scalar_certificate_present.
Lemma missing_certificate_cases : forall F prog cert,
 certificate_has_cases cert = false -> check_scalar_certificate F prog cert = false.
Proof. exact missing_scalar_certificate. Qed.

Definition certificate_module_error (ks : kernels) (mfs : list (string * binding)) : option string :=
 if negb (distinct (ks_names ks)) then
   Some (String.append "the module defines a kernel twice: " (join ", " (ks_names ks)))
 else match filter (fun e => negb (sin (fst e) (map fst (ks_entries ks)))) mfs with
 | _::_ as missing => Some (String.append "the module lacks the kernels " (join ", " (map fst missing)))
 | [] => match find (fun e => match lookup mfs (fst e) with None => true | Some _ => false end) (ks_entries ks) with
   | Some (name,_) => Some (String.append "kernel '" (String.append name "' has no binding"))
   | None => None end
 end.

(* The front end's result for each entry kernel of the text, in the
   module's order: the kernel's name, and its binding with the program and
   launch facts that lower gives it, or None when no binding names it. The
   check reads text_front front. The gate's query computes
   text_front front_program outside the kernel and has the kernel check
   that it equals text_front front, and the proposal and the check of the
   certificate read that one result. *)
Definition entry_front : Type := string * option (binding * res (program * lfacts)).

Definition text_front (lower : pkernel -> binding -> res (program * lfacts)) (a : array int) (n : int)
    (mfs : list (string * binding)) : res (list entry_front) :=
 match parse_text a n with
 | Err e i => Err e i
 | Ok ks => match certificate_module_error ks mfs with
   | Some e => Err e None
   | None => Ok (map (fun x => let '(name,K) := x in
       (name, option_map (fun mf => (mf, lower K mf)) (lookup mfs name))) (ks_entries ks))
   end
 end.

Definition propose_entry_certificate (x : entry_front) : string * scalar_certificate :=
 let '(name,r) := x in
 (name, match r with
 | Some (_, Ok (prog,F)) => propose_scalar_certificate F prog
 | Some (_, Err _ _) | None => ScalarMissing
 end).

Definition propose_front (fr : res (list entry_front)) : text_certificate :=
 match fr with
 | Err _ _ => []
 | Ok entries => map propose_entry_certificate entries
 end.

Definition propose_text_certificate (a : array int) (n : int)
    (mfs : list (string * binding)) : text_certificate :=
 propose_front (text_front front a n mfs).

(* The check of an entry kernel's certificate on the front end's result r
   for the kernel with binding mf. *)
Definition check_entry_certificate (mf : binding) (r : res (program * lfacts))
    (cert : scalar_certificate) : bool :=
 if certificate_has_cases cert then
 match r with
 | Err _ _ => false
 | Ok (prog,F) => link_ok F mf && prologue_ok F mf prog && check_scalar_certificate F prog cert && term_check F prog cert
 end else false.

Lemma check_entry_certificate_same : forall mf r cert,
 check_entry_certificate mf r cert =
 match r with
 | Err _ _ => false
 | Ok (prog,F) => link_ok F mf && prologue_ok F mf prog && check_scalar_certificate F prog cert && term_check F prog cert
 end.
Proof.
 intros mf r cert. unfold check_entry_certificate.
 destruct (certificate_has_cases cert) eqn:Hcases; [reflexivity|].
 destruct r as [[prog F]|e i]; [|reflexivity].
 rewrite (missing_certificate_cases F prog cert Hcases),andb_false_r. reflexivity.
Qed.

Fixpoint check_entries_certificate (entries : list entry_front) (certs : text_certificate) : res unit :=
 match entries, certs with
 | [], [] => Ok tt
 | (name,r)::rest, (cname,cert)::tail =>
   if String.eqb name cname then
     match r with
     | None => Err (String.append "kernel '" (String.append name "' has no binding")) None
     | Some (mf,lowered) => if check_entry_certificate mf lowered cert
       then check_entries_certificate rest tail
       else Err (String.append name ": Rocq does not accept the certificate") None
     end
   else Err "the certificate entries do not match the module" None
 | _, _ => Err "the certificate does not cover exactly the module entries" None
 end.

(* The check of a text's certificate on the front end's result fr. *)
Definition check_front (fr : res (list entry_front)) (certs : text_certificate) : res unit :=
 match fr with
 | Err e i => Err e i
 | Ok entries => check_entries_certificate entries certs
 end.

Definition check_front_ok (fr : res (list entry_front)) (certs : text_certificate) : bool :=
 match check_front fr certs with Ok _ => true | Err _ _ => false end.

Definition check_text_certificate_ok (a : array int) (n : int)
    (mfs : list (string * binding)) (certs : text_certificate) : bool :=
 check_front_ok (text_front front a n mfs) certs.

(* The check reads the front end's result of the text: the gate's query
   rewrites it with the result that the kernel checks. *)
Lemma check_text_front : forall a n mfs certs,
 check_text_certificate_ok a n mfs certs = check_front_ok (text_front front a n mfs) certs.
Proof. reflexivity. Qed.

Lemma check_entry_certificate_sound : forall K mf cert,
 check_entry_certificate mf (front K mf) cert = true ->
 exists prog F, front K mf = Ok (prog,F) /\
 check_scalar_certificate F prog cert = true /\ kernel_safe F prog /\ kernel_ends F prog /\
 link_ok F mf = true /\ prologue_ok F mf prog = true.
Proof.
 intros K mf cert H. rewrite check_entry_certificate_same in H.
 destruct (front K mf) as [[prog F]|e i]; [|discriminate].
 rewrite !andb_true_iff in H. destruct H as [[[Hl Hp] Hc] Ht].
 exists prog,F. split; [reflexivity|]. split; [exact Hc|].
 split; [exact (check_scalar_certificate_sound F prog cert Hc)|].
 split; [exact (term_check_sound F prog cert Hc Ht)|]. split; [exact Hl|exact Hp].
Qed.

Lemma check_entries_certificate_ok : forall entries certs,
 check_entries_certificate entries certs = Ok tt ->
 forall name r, In (name,r) entries ->
 exists mf lowered cert, r = Some (mf,lowered) /\ check_entry_certificate mf lowered cert = true.
Proof.
 intros entries. induction entries as [|[name0 r0] rest IH]; intros certs H name r Hin.
 - destruct Hin.
 - destruct certs as [|[cname cert] tail]; [discriminate|].
   cbn [check_entries_certificate] in H.
   destruct (String.eqb name0 cname); [|discriminate].
   destruct r0 as [[mf lowered]|]; [|discriminate].
   destruct (check_entry_certificate mf lowered cert) eqn:Hc; [|discriminate].
   destruct Hin as [Heq|Hin].
   + inversion Heq; subst. exists mf,lowered,cert. split; [reflexivity|exact Hc].
   + exact (IH tail H name r Hin).
Qed.

(* The guarantee is independent of the untrusted proposal. Every entry in
   the actual text is checked with an actual supplied binding; every block
   case and outgoing edge is checked by check_cases. Every thread of every
   entry kernel also ends (GateTerm.v). *)
Theorem verify_certificate_sound : forall a n mfs certs,
 check_text_certificate_ok a n mfs certs = true ->
 exists ks, parse_text a n = Ok ks /\
   forall name K, In (name,K) (ks_entries ks) ->
   exists mf cert, lookup mfs name = Some mf /\
     exists prog F, front K mf = Ok (prog,F) /\
       check_scalar_certificate F prog cert = true /\ kernel_safe F prog /\ kernel_ends F prog /\
       link_ok F mf = true /\ prologue_ok F mf prog = true.
Proof.
 intros a n mfs certs H. unfold check_text_certificate_ok, check_front_ok, check_front, text_front in H.
 destruct (parse_text a n) as [ks|e i]; [|discriminate].
 exists ks. split; [reflexivity|].
 destruct (certificate_module_error ks mfs); [discriminate|].
 match type of H with
 | context [check_entries_certificate ?entries certs] =>
   destruct (check_entries_certificate entries certs) as [[]|e i] eqn:Hc; [|discriminate]
 end.
 intros name K Hin.
 destruct (check_entries_certificate_ok _ certs Hc name
   (option_map (fun mf => (mf, front K mf)) (lookup mfs name))
   (in_map (fun x => let '(name,K) := x in
      (name, option_map (fun mf => (mf, front K mf)) (lookup mfs name))) _ _ Hin))
   as [mf [lowered [cert [Hr He]]]].
 destruct (lookup mfs name) as [mf'|] eqn:Hm; [|discriminate].
 injection Hr as Hmf Hlow. subst mf lowered.
 exists mf',cert. split; [reflexivity|exact (check_entry_certificate_sound K mf' cert He)].
Qed.

(* Diagnostics inspect the same explicit cases and never call proposal.
   They are explanations only: a successful checked theorem is required
   even if an untrusted diagnostic reports no error. One walk of each case
   gives both diagnostics: the first check that fails and the accesses
   that the walk refuses (case_failures), at most LISTED of them for the
   text. The walks stop after the LISTED-th refused access; the first
   failing check comes at or before it, so the reason is the one of the
   walks without the limit. *)
Definition LISTED : positive := 32%positive.

(* The first check of case_block_ok that fails in the cases, each a block
   and one of its states, at a block and instruction, and why; with at
   most left accesses that their walks refuse, by block, index and
   reason. *)
Fixpoint cases_failures (F : lfacts) (prog : program) (inv : list (list astate))
    (plans : nat -> walk_plan) (cases : list (nat * astate)) (left : nat)
    : option (option (nat * nat) * string) * list (nat * nat * string) :=
 match cases, left with
 | [], _ | _, O => (None, [])
 | (b,st) :: rest, S more =>
   let '(accesses, failure) := case_failures F prog inv (plans b) b st more in
   let '(first, refused) := cases_failures F prog inv plans rest (Nat.sub left (List.length accesses)) in
   (match failure with Some (i,r) => Some (Some (b,i), r) | None => first end,
    List.app (map (fun p => (b, fst p, snd p)) accesses) refused)
 end.

(* The first check of check_cases that fails, at a block and instruction
   or for the whole kernel, and why; with at most left accesses that the
   walks of the cases refuse, by block, index and reason. *)
Definition cases_diagnosis (F : lfacts) (prog : program) (cases : option case_certificate) (left : nat)
    : option (option (nat * nat) * string) * list (nat * nat * string) :=
 match cases with
 | None => (Some (None,"the analysis reaches no fixpoint"), [])
 | Some inv =>
   let '(first, refused) := cases_failures F prog inv (case_plan F prog inv)
     (flat_map (fun b => map (fun st => (b,st)) (cases_at inv b)) (seq 0 (List.length prog))) left in
   (if negb (lf_ok F) then Some (None,"the launch case is not consistent") else
    if negb (structured prog) then Some (None,"a jump in the middle of a block") else
    if negb (existsb (entry_case F) (cases_at inv 0))
    then Some (None,"the analysis states facts at the kernel's entry")
    else first, refused)
 end.

Definition certificate_diagnosis (F : lfacts) (prog : program) (cert : scalar_certificate) (left : nat)
    : option (option (nat * nat) * string) * list (nat * nat * string) :=
 let '(failure, refused) := match cert with
 | ScalarMissing => (Some (None,"the analysis cannot construct a complete certificate"), [])
 | ScalarCached cached => (if check_cached_certificate F prog cached then None
     else Some (None,"the interval certificate fails an instruction, access, or control-flow obligation"), [])
 | ScalarBlocks (CertSingle cases) => cases_diagnosis F prog cases left
 | ScalarBlocks (CertSplit r cut low high) =>
   let '(low_failure, low_refused) := cases_diagnosis (split_launch F r cut true) prog low left in
   let '(high_failure, high_refused) :=
     cases_diagnosis (split_launch F r cut false) prog high (Nat.sub left (List.length low_refused)) in
   (match low_failure with Some _ => low_failure | None => high_failure end, List.app low_refused high_refused)
 end in
 (if negb (lf_ok F) then Some (None,"the launch facts are not consistent") else failure, refused).

(* Why the termination check refuses: the first loop it does not prove. *)
Definition term_refusal (F : lfacts) (prog : program) (cert : scalar_certificate)
 : option (string * option (nat * nat)) :=
 if negb (term_shape prog) then
   Some ("the kernel is not proved to end: its loops are not nested or start in the first block", None)
 else match find (fun e => negb (term_loop_ok F prog cert e)) (term_edges prog) with
 | Some (l,_) => Some ("the kernel is not proved to end: the check does not prove that the loop of this back edge ends",
     Some (l, Nat.pred (block_len prog l)))
 | None => None
 end.

(* A reason with the instruction of the kernel's text where it holds. *)
Definition refusal_at (K : pkernel) (r : string) (at_ : option N) : string :=
 match at_ with
 | None => r
 | Some i => String.append r (String.append ": " (match nth_error (k_insts K) (N.to_nat i) with
   | Some ins => show_pinstr ins | None => String.append "instruction " (nstr i) end))
 end.

(* The diagnosis of an entry kernel: why the check refuses it, when reason
   is true, and at most left accesses that the walk of its certificate
   refuses, as the reason and the instruction. *)
Definition kernel_diagnosis (name : string) (K : pkernel) (mf : binding)
    (cert : scalar_certificate) (reason : bool) (left : nat) : option string * list string :=
 match front_placed K mf with
 | Err e i => (if reason then Some (String.append name (String.append ": " (refusal_at K e i))) else None, [])
 | Ok (pp,F) =>
   let prog := map (map fst) pp in
   let '(failure, refused) := certificate_diagnosis F prog cert left in
   let found := if negb reason then None else
     match failure with
     | Some (None,r) => Some (r,None)
     | Some (Some (b,i),r) => Some (r,option_map snd (nth_error (nth b pp []) i))
     | None => if negb (link_ok F mf) then Some ("the launch facts do not follow from the binding",None)
       else if negb (prologue_ok F mf prog) then
         Some ("the zeroing prologue of the shared variables is not in the form the launch model recognizes",None)
       else
         match term_refusal F prog cert with
         | Some (r,None) => Some (r,None)
         | Some (r,Some (b,i)) => Some (r,option_map snd (nth_error (nth b pp []) i))
         | None => None
         end
     end in
   (option_map (fun x => let '(r,at_) := x in String.append name (String.append ": " (refusal_at K r at_))) found,
    map (fun x => let '(b,i,r) := x in refusal_at K r (option_map snd (nth_error (nth b pp []) i))) refused)
 end.

(* The first refusal of the entries, after found, and at most left accesses
   that the walks of their certificates refuse. The reasons of the entries
   after the first refusal are not computed, and no entry is diagnosed
   once left accesses are listed: the first refusal is then found. *)
Fixpoint entries_diagnosis (mfs : list (string * binding)) (entries : list (string * pkernel))
    (certs : text_certificate) (found : option string) (left : nat) : option string * list string :=
 let first := fun reason => match found with Some _ => found | None => Some reason end in
 match left with O => (found, []) | S _ =>
 match entries,certs with
 | [],[] => (found, [])
 | (name,K)::rest,(cname,cert)::tail =>
   if String.eqb name cname then match lookup mfs name with
   | None => entries_diagnosis mfs rest tail (first (String.append "kernel '" (String.append name "' has no binding"))) left
   | Some mf =>
     let '(reason, refused) := kernel_diagnosis name K mf cert (match found with Some _ => false | None => true end) left in
     let '(later, more) := entries_diagnosis mfs rest tail (match found with Some _ => found | None => reason end)
       (Nat.sub left (List.length refused)) in
     (later, List.app refused more)
   end else (first "the certificate entries do not match the module", [])
 | _,_ => (first "the certificate does not cover exactly the module entries", [])
 end end.

(* Why the text is not accepted with the certificate, and the first LISTED
   refused accesses of its entry kernels, as the reason and the
   instruction. *)
Definition text_diagnosis (a : array int) (n : int)
    (mfs : list (string * binding)) (certs : text_certificate) : option string * list string :=
 match parse_text a n with
 | Err e _ => (Some e, [])
 | Ok ks => entries_diagnosis mfs (ks_entries ks) certs (certificate_module_error ks mfs) (Pos.to_nat LISTED)
 end.
