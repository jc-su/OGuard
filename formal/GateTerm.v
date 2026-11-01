(* Termination of every thread of an accepted kernel.

   A back edge is a branch at the end of block l to a block h <= l; its loop
   is the block interval [h, l]. The check requires loops that are nested or
   disjoint and a head after block 0; a head may have several back edges, and
   a thread may enter a loop past its head. A kernel without back edges ends
   because every step moves to a later position. A loop ends when the
   certificate's states show that its back edges are never taken, or when it
   has a ranking value that every iteration that returns to the head strictly
   decreases: a counter register with one checked update in the head or the
   latch block, a counter incremented by 1 or by a power of two that a test
   after the increment keeps from one value while the loop continues, or the
   distance between the two bounds of a binary search. In a search, the new
   bounds are evaluated as functions of the head's lower bound and distance
   along every path through the loop. The ranges that the checks use come
   from the safety certificate's states, an interval of the counter at the
   head that the check proves inductive, the bounds of a search that the
   loop keeps from its entry, and the back-edge condition; the counter's
   walk keeps the exact signed interval of an addition and of a comparison
   with a constant, which the domain reads as unsigned for a narrow
   register. Descriptors are proposed and then checked; a failed check
   refuses. *)
From Coq Require Import List ZArith Bool Arith Lia.
Require Import OpGate.GateIsa OpGate.GateForm OpGate.GateDomain OpGate.GateMeaning
 OpGate.GateSoundRange OpGate.GateSoundNorm OpGate.GateSoundOp OpGate.GateSoundRefine OpGate.GateSoundTransfer OpGate.GateThread
 OpGate.GateInterval OpGate.GateCertificate OpGate.GateBarrier OpGate.GateLoop.
Import ListNotations.
Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Loops.                                                              *)

Definition term_edge_at (prog : program) (b : nat) : option nat :=
 match last_instr (block_body prog b) with
 | Some (IBra _ t) => if Nat.leb t b then Some t else None
 | _ => None end.

(* Each back edge as (latch, head). *)
Definition term_edges (prog : program) : list (nat * nat) :=
 flat_map (fun b => match term_edge_at prog b with Some t => [(b,t)] | None => [] end)
   (seq 0 (List.length prog)).

Definition term_in (h l b : nat) : bool := Nat.leb h b && Nat.leb b l.

Definition term_nested (e1 e2 : nat * nat) : bool :=
 let '(l1,h1) := e1 in let '(l2,h2) := e2 in
 Nat.ltb l1 h2 || Nat.ltb l2 h1 || (Nat.leb h1 h2 && Nat.leb l2 l1) ||
 (Nat.leb h2 h1 && Nat.leb l1 l2).

Definition term_entered (prog : program) (h l : nat) : bool :=
 forallb (fun b => term_in h l b ||
   forallb (fun t => negb (Nat.ltb h t && Nat.leb t l)) (edge_targets prog b))
 (seq 0 (List.length prog)).

(* The latches of head h inside [h, l]: the blocks whose back edge goes to h. *)
Definition term_latches (prog : program) (h l : nat) : list nat :=
 map fst (filter (fun e => Nat.eqb (snd e) h && Nat.leb h (fst e) && Nat.leb (fst e) l) (term_edges prog)).

Definition term_shape (prog : program) : bool :=
 let edges := term_edges prog in
 structured prog &&
 forallb (fun e1 => forallb (term_nested e1) edges) edges &&
 forallb (fun e => let '(l,h) := e in Nat.ltb 0 h) edges.

(* ------------------------------------------------------------------ *)
(* Instructions of blocks.                                             *)

Definition term_writes (r : reg) (ins : instr) : bool := existsb (Pos.eqb r) (loop_writes ins).

Definition term_indexed (prog : program) (b : nat) : list (nat * instr) :=
 combine (seq 0 (block_len prog b)) (block_body prog b).

Definition term_last_opt {A : Type} (xs : list A) : option A :=
 match rev xs with x :: _ => Some x | [] => None end.

(* The last write of r before index i of block b. *)
Definition term_last_write (prog : program) (b i : nat) (r : reg) : option (nat * instr) :=
 term_last_opt (filter (fun x => Nat.ltb (fst x) i && term_writes r (snd x)) (term_indexed prog b)).

(* Every write of r in the blocks h to l. *)
Definition term_region_writes (prog : program) (h l : nat) (r : reg) : list (nat * nat * instr) :=
 flat_map (fun b => map (fun x => (b, fst x, snd x))
   (filter (fun x => term_writes r (snd x)) (term_indexed prog b))) (seq h (S l - h)).

(* r is not written at indices [x, y) of block b. *)
Definition term_kept (prog : program) (b x y : nat) (r : reg) : bool :=
 forallb (fun z => negb (Nat.leb x (fst z) && Nat.ltb (fst z) y && term_writes r (snd z)))
   (term_indexed prog b).

Definition term_slice (prog : program) (b x y : nat) : list instr :=
 firstn (y - x) (skipn x (block_body prog b)).

(* ------------------------------------------------------------------ *)
(* States of the safety certificate.                                   *)

(* Every abstract state the certificate gives before instruction i of
   block b, with the launch facts it holds under. *)
Definition term_states (F : lfacts) (prog : program) (cert : scalar_certificate) (b i : nat)
 : list (lfacts * astate) :=
 let before := firstn i (block_body prog b) in
 match cert with
 | ScalarBlocks (CertSingle (Some inv)) =>
   map (fun st => (F, walk F st before)) (cases_at inv b)
 | ScalarBlocks (CertSplit r cut (Some low) (Some high)) =>
   let Fl := split_launch F r cut true in let Fh := split_launch F r cut false in
   map (fun st => (Fl, walk Fl st before)) (cases_at low b) ++
   map (fun st => (Fh, walk Fh st before)) (cases_at high b)
 | ScalarCached c => map (fun k => (F, cached_at k i)) (cached_cases (cert_blocks c) b)
 | _ => []
 end.

(* ------------------------------------------------------------------ *)
(* Counters.                                                           *)

Inductive term_kind := KUp | KDown | KHalve | KDouble.

Definition term_signed (k : term_kind) : bool :=
 match k with KUp | KDown => true | KHalve | KDouble => false end.

(* The update's arithmetic strictly moves the counter of value range [x, y]
   in the kind's direction, without wrapping. *)
Definition term_step_ok (F : lfacts) (st : astate) (k : term_kind) (step : instr) (x y : Z) : bool :=
 match k, step with
 | KUp, IBin None BAdd w _ _ o => match cmp_range F st o (Z.of_nat w) true with
   | Some (sl,sh) => (1 <=? sl) && (y + sh <? 2 ^ (Z.of_nat w - 1)) | None => false end
 | KDown, IBin None BAdd w _ _ o => match cmp_range F st o (Z.of_nat w) true with
   | Some (sl,sh) => (sh <=? -1) && (- 2 ^ (Z.of_nat w - 1) <=? x + sl) | None => false end
 | KDown, IBin None BSub w _ _ o => match cmp_range F st o (Z.of_nat w) true with
   | Some (sl,sh) => (1 <=? sl) && (- 2 ^ (Z.of_nat w - 1) <=? x - sh) | None => false end
 | KHalve, IBin None BShr _ _ _ (OImm k) => (1 <=? k) && (k <? 2 ^ 32) && (1 <=? x)
 | KHalve, IBin None BShrs w _ _ (OImm k) =>
   (1 <=? k) && (k <? 2 ^ 32) && (1 <=? x) && (y <? 2 ^ (Z.of_nat w - 1))
 | KDouble, IBin None BShl w _ _ (OImm k) =>
   (1 <=? k) && (k <=? 64) && (1 <=? x) && (y * 2 ^ k <? 2 ^ Z.of_nat w)
 | _, _ => false end.

(* A counter descriptor: register c, its one write in the loop at index i
   of block b, the arithmetic step at index j, either the write itself or
   t := a op o followed by the copy c := t; a holds c's value: a = c, a copy
   in block b at index m, or the only write of a in the loop, a copy in the
   head block. With tc_shift = Some m, j = i and the copy c := t in the
   latch reads t := c op o, the only write of t in the loop, at index m of
   the head. The interval [lo, hi] of c's value at the head is checked. *)
Inductive term_alias := ASame | ABlock (m : nat) | AHead (m : nat).
Record term_counter := {
 tc_c : reg; tc_b : nat; tc_i : nat; tc_j : nat; tc_step : instr; tc_alias : term_alias;
 tc_shift : option nat; tc_kind : term_kind; tc_lo : Z; tc_hi : Z }.

Definition term_step_regs (d : term_counter) : option (binop * nat * reg * reg * GateIsa.opnd) :=
 match tc_step d with IBin None op w t a o => Some (op,w,t,a,o) | _ => None end.

(* r's value read at width w, as a signed integer if sg, assumed in [lo, hi];
   the interval keeps r's alignment and is rounded inward to it. *)
Definition term_assume (F : lfacts) (w : nat) (sg : bool) (r : reg) (lo hi : Z) (st : astate) : astate :=
 let al := Z.max 0 (Z.min (reg_align F st r) (Z.of_nat w)) in
 let '(x,y) := match cmp_range F st (OReg r) (Z.of_nat w) sg with
   | Some (u,v) => (Z.max lo u, Z.min hi v) | None => (lo,hi) end in
 let '(x',y') := round al (x,y) in
 if y' <? x' then bottom_state else put st r (FForm (range_form (Z.of_nat w) x' y') al None).

(* The transfer, with the exact signed interval of an addition's or a
   subtraction's result when its operands' signed intervals keep it inside
   the width: the domain reads a narrow register's interval as unsigned, so
   a counter that crosses 0 would lose its interval. *)
Definition term_xfer (F : lfacts) (st : astate) (ins : instr) : astate :=
 let st' := transfer F st ins in
 if is_bottom st' then st' else
 match ins with
 | IBin None op w d a o =>
   let ranged := match op with
     | BAdd => match cmp_range F st (OReg a) (Z.of_nat w) true, cmp_range F st o (Z.of_nat w) true with
       | Some (x,y), Some (u,v) => Some (x + u, y + v) | _, _ => None end
     | BSub => match cmp_range F st (OReg a) (Z.of_nat w) true, cmp_range F st o (Z.of_nat w) true with
       | Some (x,y), Some (u,v) => Some (x - v, y - u) | _, _ => None end
     | _ => None end in
   match ranged with
   | Some (lo,hi) =>
     if Nat.eqb (wdn F d) w && Nat.eqb (wdn F a) w && Nat.leb 1 w && Nat.leb w 64 &&
        match o with OReg r => Nat.eqb (wdn F r) w | OImm _ => true end &&
        (- 2 ^ (Z.of_nat w - 1) <=? lo) && (hi <? 2 ^ (Z.of_nat w - 1)) &&
        negb (match cmp_range F st' (OReg d) (Z.of_nat w) true with
              | Some (x',y') => (lo <=? x') && (y' <=? hi) | None => false end)
     then term_assume F w true d lo hi st' else st'
   | None => st' end
 | _ => st' end.

Definition term_walk (F : lfacts) (st : astate) (body : list instr) : astate :=
 fold_left (term_xfer F) body st.

(* States restricted to the registers that decide some registers: the
   walk of a long block is cheaper over fewer facts. *)

(* The registers whose values in block b decide the values of roots. *)
Definition term_slice_regs (prog : program) (b : nat) (roots : list reg) : list reg :=
 fold_right (fun ins acc => if existsb (fun r => existsb (Pos.eqb r) acc) (loop_writes ins)
   then instruction_inputs ins ++ acc else acc) roots (block_body prog b).

Definition term_keep (prog : program) (b : nat) (roots : list reg) : PM.t unit :=
 fold_left (fun m r => PM.add r tt m) (term_slice_regs prog b roots) (PM.empty unit).

(* The facts of keep and of the registers they depend on. *)
Definition term_cut (F : lfacts) (keep : PM.t unit) (st : astate) : astate := trim_live_seed F keep st.

(* The certificate's states at the start of block b, restricted to keep and
   walked to index i. *)
Definition term_cut_states (keep : PM.t unit) (F : lfacts) (prog : program) (cert : scalar_certificate)
 (b i : nat) : list (lfacts * astate) :=
 map (fun x => let '(F',st) := x in (F', term_walk F' (term_cut F' keep st) (term_slice prog b 0 i)))
   (term_states F prog cert b 0).

(* The instructions of block b are well formed under every certificate
   state at its start that is not bottom. *)
Definition term_block_wf (F : lfacts) (prog : program) (cert : scalar_certificate) (b : nat) : bool :=
 short_all (fun x => let '(F',st) := x in is_bottom st || forallb (wf_instr F') (block_body prog b))
   (term_states F prog cert b 0).

(* The registers a walk of block v keeps for descriptor d: those that
   decide the counter, its update, and the branch that ends v. *)
Definition term_counter_keep (prog : program) (v : nat) (d : term_counter) : PM.t unit :=
 term_keep prog v (tc_c d :: (match tc_step d with IBin _ _ _ t a o => t :: a :: opnd_inputs o | _ => [] end) ++
   match last_instr (block_body prog v) with Some ins => instruction_inputs ins | None => [] end).

(* The signed interval [x, y] of a register narrowed by comparison k with a
   constant, when k holds (or fails, if holds is false). *)
Definition term_cmp_bounds (k : cmpf) (holds : bool) (x y : Z) : option (Z * Z) :=
 match cb k with
 | OImm z =>
   if csg k then
     let c0 := sext (cw k) z in
     Some match (if holds then cc k else flip (cc k)) with
     | CEq => (Z.max x c0, Z.min y c0)
     | CNe => (if Z.eqb x c0 then x + 1 else x, if Z.eqb y c0 then y - 1 else y)
     | CLt => (x, Z.min y (c0 - 1))
     | CLe => (x, Z.min y c0)
     | CGt => (Z.max x (c0 + 1), y)
     | CGe => (Z.max x c0, y)
     end
   else None
 | OReg _ => None end.

Definition term_cmp_assume (F : lfacts) (st s : astate) (holds : bool) (k : cmpf) : astate :=
 let r := ca k in let w := cw k in
 if Nat.eqb (wdn F r) w && Nat.leb 1 w && Nat.leb w 64 then
   match cmp_range F st (OReg r) (Z.of_nat w) true with
   | Some (x,y) => match term_cmp_bounds k holds x y with
     | Some (lo,hi) => term_assume F w true r lo hi s
     | None => s end
   | None => s end
 else s.

(* The refinement by predicate p, followed by the signed intervals that p's
   comparisons with constants give their registers. *)
Definition term_refine (F : lfacts) (st : astate) (p : reg) (b : bool) : option astate :=
 match refine F st p b with
 | None => None
 | Some s => if is_bottom s then Some s else
   Some match get st p with
   | Some (FPred cs exact) =>
     if b then fold_left (fun acc k => term_cmp_assume F st acc true k) cs s
     else match cs, exact with [k], true => term_cmp_assume F st s false k | _, _ => s end
   | Some (FNeg cs exact) =>
     if b then match cs, exact with [k], true => term_cmp_assume F st s false k | _, _ => s end
     else fold_left (fun acc k => term_cmp_assume F st acc true k) cs s
   | _ => s end
 end.

(* The guard of the back edge: its last write in the latch, plus one. *)
Definition term_guard_from (prog : program) (l : nat) (p : reg) : nat :=
 match term_last_write prog l (block_len prog l) p with Some (k,_) => S k | None => O end.

(* The state on the back edge from l to h, from state st at index start
   of l; None if the back edge cannot be taken. The guard is refined right
   after its last write, since it is unchanged from there to the branch. *)
Definition term_latch (F : lfacts) (prog : program) (l start : nat) (st : astate) : option astate :=
 let body := block_body prog l in
 match last_instr body with
 | Some (IBra (Some (neg,p)) _) =>
   let r := Nat.max (term_guard_from prog l p) start in
   match term_refine F (term_walk F st (term_slice prog l start r)) p (negb neg) with
   | None => None
   | Some s => Some (term_walk F s (skipn r body))
   end
 | _ => Some (term_walk F st (skipn start body))
 end.

(* The state at index i of latch l, refined by the back-edge guard when the
   guard's last write precedes i. *)
Definition term_guard_at (F : lfacts) (prog : program) (l i : nat) (st : astate) : option astate :=
 match last_instr (block_body prog l) with
 | Some (IBra (Some (neg,p)) _) =>
   if Nat.leb (term_guard_from prog l p) i then term_refine F st p (negb neg) else Some st
 | _ => Some st
 end.

Definition term_range_hull (rs : list (option (Z * Z))) : option (option (Z * Z)) :=
 fold_right (fun r acc => match r, acc with
   | Some (x,y), Some None => Some (Some (x,y))
   | Some (x,y), Some (Some (u,v)) => Some (Some (Z.min x u, Z.max y v))
   | _, _ => None end) (Some None) rs.

Definition term_within (lo hi : Z) (r : option (Z * Z)) : bool :=
 match r with Some (x,y) => (lo <=? x) && (y <=? hi) | None => false end.

(* The ranges of c on every edge into [h, l] from a block outside it. *)
Definition term_entry_ranges (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (c : reg) (w : nat) (sg : bool) : list (option (Z * Z)) :=
 flat_map (fun v =>
   if term_in h l v then [] else
   if negb (existsb (term_in h l) (edge_targets prog v)) then [] else
   match last_instr (block_body prog v) with
   | None => [None]
   | Some last =>
     flat_map (fun x => let '(F',st) := x in
       if is_bottom st then [] else
       if negb (wf_instr F' last) then [None] else
       flat_map (fun e => let '(u,os) := e in
         if term_in h l u then match os with
           | Some s => if is_bottom s then [] else [cmp_range F' s (OReg c) (Z.of_nat w) sg]
           | None => [] end else [])
         (edge_states F' prog v (transfer F' st last)))
       (term_states F prog cert v (Nat.pred (block_len prog v)))
   end) (seq 0 (List.length prog)).

(* The states on the edges into the loop from outside it, once per loop. *)
Definition term_entry_states (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) : list (option (lfacts * astate)) :=
 flat_map (fun v =>
   if term_in h l v then [] else
   if negb (existsb (term_in h l) (edge_targets prog v)) then [] else
   match last_instr (block_body prog v) with
   | None => [None]
   | Some last =>
     flat_map (fun x => let '(F',st) := x in
       if is_bottom st then [] else
       if negb (wf_instr F' last) then [None] else
       flat_map (fun e => let '(u,os) := e in
         if term_in h l u then match os with
           | Some s => if is_bottom s then [] else [Some (F',s)]
           | None => [] end else [])
         (edge_states F' prog v (transfer F' st last)))
       (term_states F prog cert v (Nat.pred (block_len prog v)))
   end) (seq 0 (List.length prog)).

Definition term_wf_from (F : lfacts) (prog : program) (b x : nat) : bool :=
 forallb (wf_instr F) (skipn x (block_body prog b)).

(* The states with the counter, and its copy, assumed in [lo, hi]. *)
Definition term_assume_all (F : lfacts) (d : term_counter) (w : nat) (a : reg) (st : astate) : astate :=
 let sg := term_signed (tc_kind d) in
 let s1 := term_assume F w sg (tc_c d) (tc_lo d) (tc_hi d) st in
 if Pos.eqb a (tc_c d) then s1 else term_assume F w sg a (tc_lo d) (tc_hi d) s1.

(* The descriptor's facts that are decided without an abstract state. *)
Definition term_counter_static (F : lfacts) (prog : program) (h l : nat) (d : term_counter)
 (op : binop) (w : nat) (t a : reg) (o : GateIsa.opnd) : bool :=
 let c := tc_c d in let b := tc_b d in let i := tc_i d in let j := tc_j d in
 (* entered only at the head, or the update does not read a value made in the head *)
 (term_entered prog h l || match tc_alias d, tc_shift d with AHead _, _ | _, Some _ => false | _, _ => true end) &&
 (Nat.eqb b h || Nat.eqb b l) && Nat.leb j i && (tc_lo d <=? tc_hi d) &&
 negb (is_cell F c) && negb (is_cell F a) && negb (is_cell F t) &&
 Nat.eqb (wdn F c) w && Nat.eqb (wdn F a) w && Nat.eqb (wdn F t) w &&
 (* the one write of c in the loop *)
 match term_region_writes prog h l c with
 | [(b',i',ins)] => Nat.eqb b' b && Nat.eqb i' i &&
   (if Nat.eqb j i then
      match tc_shift d with
      | None => Pos.eqb t c && match fetch prog b i with
        | Some s => if loop_instr_eq_dec s (tc_step d) then true else false | None => false end
      | Some m =>
        Nat.eqb b l && negb (Nat.eqb h l) && negb (Pos.eqb t c) && Pos.eqb a c && wf_instr F (tc_step d) &&
        match ins with IMov None c' t' => Pos.eqb c' c && Pos.eqb t' t | _ => false end &&
        match term_region_writes prog h l t with
        | [(b2,m2,s2)] => Nat.eqb b2 h && Nat.eqb m2 m &&
          (if loop_instr_eq_dec s2 (tc_step d) then true else false)
        | _ => false end &&
        match o with
        | OReg r => negb (is_cell F r) &&
          forallb (fun z => let '(b2,m2,_) := z in Nat.eqb b2 h && Nat.ltb m2 m) (term_region_writes prog h l r)
        | OImm _ => true end
      end
    else
      match tc_shift d with None => true | Some _ => false end &&
      match ins with IMov None c' t' => Pos.eqb c' c && Pos.eqb t' t | _ => false end &&
      negb (Pos.eqb t c) &&
      match fetch prog b j with
      | Some s => if loop_instr_eq_dec s (tc_step d) then true else false | None => false end &&
      term_kept prog b (S j) i t &&
      match o with OReg r => negb (is_cell F r) && term_kept prog b j i r | OImm _ => true end)
 | _ => false end &&
 (* a holds c's value at the step *)
 match tc_alias d with
 | ASame => Pos.eqb a c
 | ABlock m => negb (Pos.eqb a c) &&
   match term_last_write prog b j a with
   | Some (m', IMov None a' c') => Nat.eqb m' m && Pos.eqb a' a && Pos.eqb c' c
   | _ => false end
 | AHead m => negb (Pos.eqb a c) && Nat.eqb b l && negb (Nat.eqb h l) &&
   match term_region_writes prog h l a with
   | [(b',m',IMov None a' c')] => Nat.eqb b' h && Nat.eqb m' m && Pos.eqb a' a && Pos.eqb c' c
   | _ => false end
 end &&
 (* the guard of the back edge is not a cell *)
 match last_instr (block_body prog l) with
 | Some (IBra (Some (_,p)) _) => negb (is_cell F p)
 | Some (IBra None _) => true
 | _ => false end.

(* The value that register r has at index i of latch l when the back edge
   is taken, when r's last write before i is the selection r := p ? x1 : x2
   on the back edge's guard p, after p's last write. *)
Definition term_guard_sel (prog : program) (l i : nat) (r : reg) : option Z :=
 match last_instr (block_body prog l) with
 | Some (IBra (Some (neg,p)) _) =>
   match term_last_write prog l i r with
   | Some (m, ISelp None _ r' (OImm x1) (OImm x2) p') =>
     if Pos.eqb p' p && negb (Pos.eqb r p) && Nat.leb (term_guard_from prog l p) m
     then Some (if neg then x2 else x1) else None
   | _ => None end
 | _ => None end.

(* When the update writes c itself: c has its copy a's interval, and an
   operand that a selection on the back edge's guard gives has its value. *)
Definition term_step_focus (F : lfacts) (prog : program) (l : nat) (d : term_counter) (w : nat) (a : reg)
 (s : astate) : astate :=
 if negb (Nat.eqb (tc_j d) (tc_i d)) then s else
 let sg := term_signed (tc_kind d) in
 let s1 := if Pos.eqb a (tc_c d) then s else
   match cmp_range F s (OReg a) (Z.of_nat w) sg with
   | Some (x,y) => term_assume F w sg (tc_c d) x y s
   | None => s end in
 match tc_step d with
 | IBin _ _ _ _ _ (OReg r) =>
   if Nat.eqb (tc_b d) l && Nat.eqb (wdn F r) w && negb (is_cell F r) then
     match term_guard_sel prog l (tc_i d) r with
     | Some v => term_assume F w true r (extend w true v) (extend w true v) s1
     | None => s1 end
   else s1
 | _ => s1 end.

(* The update strictly moves c on every iteration that takes the back edge. *)
Definition term_counter_step_with (S : list (lfacts * astate)) (F : lfacts) (prog : program)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 let c := tc_c d in let b := tc_b d in let i := tc_i d in let j := tc_j d in
 let sg := term_signed (tc_kind d) in
 short_all (fun x => let '(F',st) := x in
   if is_bottom st then true else
   if negb (term_wf_from F' prog b j) then false else
   let s0 := term_assume_all F' d w a st in
   let si := term_walk F' s0 (term_slice prog b j i) in
   match (if Nat.eqb b l then term_guard_at F' prog l i si else Some si) with
   | None => true
   | Some s => if is_bottom s then true else
     let s' := term_step_focus F' prog l d w a s in
     match cmp_range F' s' (OReg c) (Z.of_nat w) sg with
     | Some (x,y) => term_step_ok F' s' (tc_kind d) (tc_step d) x y | None => false end
   end) S.

Definition term_counter_step (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 term_counter_step_with (term_cut_states (term_counter_keep prog (tc_b d) d) F prog cert (tc_b d) (tc_j d))
   F prog h l d w a.

(* The range of c on the edges from the head block into the rest of the
   loop, after its update in the head, over the head's states. *)
Definition term_counter_hull_with (S : list (lfacts * astate)) (F : lfacts) (prog : program)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : option (option (Z * Z)) :=
 term_range_hull (flat_map (fun x => let '(F',st) := x in
   if is_bottom st then [] else
   if negb (term_wf_from F' prog h (tc_j d)) then [None] else
   let s := term_walk F' (term_assume_all F' d w a st) (term_slice prog h (tc_j d) (Nat.pred (block_len prog h))) in
   if is_bottom s then [] else
   match last_instr (block_body prog h) with
   | None => [None]
   | Some last =>
     flat_map (fun e => let '(u,os) := e in
       if Nat.ltb h u && Nat.leb u l then match os with
         | Some q => if is_bottom q then [] else [cmp_range F' q (OReg (tc_c d)) (Z.of_nat w) (term_signed (tc_kind d))]
         | None => [] end else [])
       (edge_states F' prog h (transfer F' s last))
   end) S).

Definition term_counter_hull (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : option (option (Z * Z)) :=
 term_counter_hull_with (term_cut_states (term_counter_keep prog h d) F prog cert h (tc_j d)) F prog h l d w a.

(* Where the latch's walk starts, and the state it starts from. *)
Definition term_latch_start (d : term_counter) (l : nat) : nat :=
 if Nat.eqb (tc_b d) l then tc_j d else O.

Definition term_latch_entry_with (hull : option (option (Z * Z)))
 (l : nat) (d : term_counter) (w : nat) (a : reg) (F' : lfacts) (st : astate) : astate :=
 if Nat.eqb (tc_b d) l then term_assume_all F' d w a st else
 match hull with
 | Some (Some (x,y)) => term_assume F' w (term_signed (tc_kind d)) (tc_c d) x y st
 | _ => st end.

Definition term_latch_entry (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) (F' : lfacts) (st : astate) : astate :=
 term_latch_entry_with (term_counter_hull F prog cert h l d w a) l d w a F' st.

(* The interval holds on the back edge. *)
Definition term_counter_latch_with (hull : option (option (Z * Z))) (S : list (lfacts * astate))
 (prog : program) (l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 short_all (fun x => let '(F',st) := x in
   if is_bottom st then true else
   if negb (term_wf_from F' prog l (term_latch_start d l)) then false else
   match term_latch F' prog l (term_latch_start d l) (term_latch_entry_with hull l d w a F' st) with
   | None => true
   | Some s => if is_bottom s then true else
     term_within (tc_lo d) (tc_hi d) (cmp_range F' s (OReg (tc_c d)) (Z.of_nat w) (term_signed (tc_kind d)))
   end) S.

Definition term_counter_latch (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 term_counter_latch_with (term_counter_hull F prog cert h l d w a)
   (term_cut_states (term_counter_keep prog l d) F prog cert l (term_latch_start d l)) prog l d w a.

Definition term_counter_back (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 if Nat.eqb (tc_b d) l then term_counter_latch F prog cert h l d w a else
 match term_counter_hull F prog cert h l d w a with
 | None => false
 | Some None => true
 | Some (Some _) => term_counter_latch F prog cert h l d w a
 end.

(* The back-edge check with the states at the update shared with the step
   check: S holds the restricted states before index j of block b. *)
Definition term_counter_back_with (S : list (lfacts * astate)) (F : lfacts) (prog : program)
 (cert : scalar_certificate) (h l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 if Nat.eqb (tc_b d) l then term_counter_latch_with None S prog l d w a else
 let hull := term_counter_hull_with S F prog h l d w a in
 match hull with
 | None => false
 | Some None => true
 | Some (Some _) => term_counter_latch_with hull (term_cut_states (term_counter_keep prog l d) F prog cert l 0) prog l d w a
 end.

(* With several latches, the update lies in the head, every latch follows
   it, and the counter's range on the head's edges into the loop lies in
   [lo, hi]: the counter keeps that value on every back edge. *)
Definition term_counter_multi_with (S : list (lfacts * astate)) (F : lfacts) (prog : program)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 Nat.eqb (tc_b d) h && negb (Nat.eqb h l) && forallb (Nat.ltb h) (term_latches prog h l) &&
 match term_counter_hull_with S F prog h l d w a with
 | Some None => true
 | Some (Some (x,y)) => (tc_lo d <=? x) && (y <=? tc_hi d)
 | None => false end.

Definition term_counter_multi (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_counter) (w : nat) (a : reg) : bool :=
 term_counter_multi_with (term_cut_states (term_counter_keep prog h d) F prog cert h (tc_j d)) F prog h l d w a.

Definition term_counter_ok (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_counter) : bool :=
 match term_step_regs d with
 | None => false
 | Some (op,w,t,a,o) =>
   if negb (term_counter_static F prog h l d op w t a o) then false else
   if negb (short_all (term_within (tc_lo d) (tc_hi d))
     (term_entry_ranges F prog cert h l (tc_c d) w (term_signed (tc_kind d)))) then false else
   if negb (term_block_wf F prog cert (tc_b d) && term_block_wf F prog cert l) then false else
   let S := term_cut_states (term_counter_keep prog (tc_b d) d) F prog cert (tc_b d) (tc_j d) in
   if negb (term_counter_step_with S F prog h l d w a) then false else
   if forallb (Nat.eqb l) (term_latches prog h l) then term_counter_back_with S F prog cert h l d w a
   else term_counter_multi_with S F prog h l d w a
 end.

(* ------------------------------------------------------------------ *)
(* Binary searches.                                                    *)

(* A value as a function of the head's lower bound L and distance D = H - L:
   k + p*L + q*D + r*floor((D + j)/2). *)
Record bs_val := { bv_k : Z; bv_p : Z; bv_q : Z; bv_r : Z; bv_j : Z }.

Definition bs_eval (v : bs_val) (L D : Z) : Z :=
 bv_k v + bv_p v * L + bv_q v * D + bv_r v * ((D + bv_j v) / 2).

Definition bs_lin (k p q : Z) : bs_val := {| bv_k := k; bv_p := p; bv_q := q; bv_r := 0; bv_j := 0 |}.

Definition bs_add (u v : bs_val) : option bs_val :=
 let k := bv_k u + bv_k v in let p := bv_p u + bv_p v in let q := bv_q u + bv_q v in
 if Z.eqb (bv_r u) 0 then Some {| bv_k := k; bv_p := p; bv_q := q; bv_r := bv_r v; bv_j := bv_j v |}
 else if Z.eqb (bv_r v) 0 then Some {| bv_k := k; bv_p := p; bv_q := q; bv_r := bv_r u; bv_j := bv_j u |}
 else if Z.eqb (bv_j u) (bv_j v) then Some {| bv_k := k; bv_p := p; bv_q := q; bv_r := bv_r u + bv_r v; bv_j := bv_j u |}
 else None.

Definition bs_neg (v : bs_val) : bs_val :=
 {| bv_k := - bv_k v; bv_p := - bv_p v; bv_q := - bv_q v; bv_r := - bv_r v; bv_j := bv_j v |}.

(* floor(v/2), for v = k + 2m*L + D or v = k + 2m*L. *)
Definition bs_half (v : bs_val) : option bs_val :=
 if Z.eqb (bv_r v) 0 && Z.even (bv_p v) then
   if Z.eqb (bv_q v) 1 then Some {| bv_k := 0; bv_p := bv_p v / 2; bv_q := 0; bv_r := 1; bv_j := bv_k v |}
   else if Z.eqb (bv_q v) 0 then Some (bs_lin (bv_k v / 2) (bv_p v / 2) 0)
   else None
 else None.

Definition bs_mono (q r j d : Z) : Z := q * d + r * ((d + j) / 2).

(* Bounds of q*d + r*floor((d + j)/2) for d in [d1, d2]: with d + j = 2f + m
   and m in {0, 1}, the value is (2q + r)f + qm - qj, for f between
   floor((d1 + j)/2) and floor((d2 + j)/2). *)
Definition bs_split (q r j d1 d2 : Z) : Z * Z :=
 let c := 2 * q + r in let f1 := (d1 + j) / 2 in let f2 := (d2 + j) / 2 in
 (Z.min (c * f1) (c * f2) + Z.min 0 q - q * j, Z.max (c * f1) (c * f2) + Z.max 0 q - q * j).

(* Bounds of v for L in [l1, l2] and D in [d1, d2]; the part in D is also
   bounded by its ends when it is monotone. *)
Definition bs_bounds (v : bs_val) (l1 l2 d1 d2 : Z) : Z * Z :=
 let q := bv_q v in let r := bv_r v in let j := bv_j v in
 let pl := if 0 <=? bv_p v then (bv_p v * l1, bv_p v * l2) else (bv_p v * l2, bv_p v * l1) in
 let sp := bs_split q r j d1 d2 in
 let g := if ((0 <=? q) && (0 <=? r)) || ((r <? 0) && (0 <=? q + r))
   then (Z.max (bs_mono q r j d1) (fst sp), Z.min (bs_mono q r j d2) (snd sp))
   else if ((q <=? 0) && (r <=? 0)) || ((0 <? r) && (q + r <=? 0))
   then (Z.max (bs_mono q r j d2) (fst sp), Z.min (bs_mono q r j d1) (snd sp))
   else sp in
 (bv_k v + fst pl + fst g, bv_k v + snd pl + snd g).

(* Bounds of a L + b D for L in [l1, l2], D in [d1, d2] and L + D in
   [h1, h2]: a L + b D = (a - g) L + (b - g) D + g (L + D), each term
   bounded by its interval, for g in {0, a, b}. *)
Definition bs_lin_one (a b l1 l2 d1 d2 h1 h2 g : Z) : Z * Z :=
 (Z.min ((a - g) * l1) ((a - g) * l2) + Z.min ((b - g) * d1) ((b - g) * d2) + Z.min (g * h1) (g * h2),
  Z.max ((a - g) * l1) ((a - g) * l2) + Z.max ((b - g) * d1) ((b - g) * d2) + Z.max (g * h1) (g * h2)).

Definition bs_lin_bounds (a b l1 l2 d1 d2 h1 h2 : Z) : Z * Z :=
 let '(x0,y0) := bs_lin_one a b l1 l2 d1 d2 h1 h2 0 in
 let '(x1,y1) := bs_lin_one a b l1 l2 d1 d2 h1 h2 a in
 let '(x2,y2) := bs_lin_one a b l1 l2 d1 d2 h1 h2 b in
 (Z.max x0 (Z.max x1 x2), Z.min y0 (Z.min y1 y2)).

(* Bounds of v when also L + D lies in [h1, h2]: 2v is 2k + rj + 2pL +
   (2q + r)D less r times the low bit of D + j. *)
Definition bs_hbounds (v : bs_val) (l1 l2 d1 d2 h1 h2 : Z) : Z * Z :=
 let '(x,y) := bs_bounds v l1 l2 d1 d2 in
 let r := bv_r v in
 let c := 2 * bv_k v + r * bv_j v in
 let '(u,t) := bs_lin_bounds (2 * bv_p v) (2 * bv_q v + r) l1 l2 d1 d2 h1 h2 in
 (Z.max x (- ((- (c + u - Z.max r 0)) / 2)), Z.min y ((c + t - Z.min r 0) / 2)).

(* The values of L, D, and L + D that a walk assumes. *)
Record bs_box := { bx_w : nat; bx_l1 : Z; bx_l2 : Z; bx_d1 : Z; bx_d2 : Z; bx_h1 : Z; bx_h2 : Z }.

Definition bs_box_bounds (bx : bs_box) (v : bs_val) : Z * Z :=
 bs_hbounds v (bx_l1 bx) (bx_l2 bx) (bx_d1 bx) (bx_d2 bx) (bx_h1 bx) (bx_h2 bx).

Definition bs_fits (bx : bs_box) (v : bs_val) : bool :=
 let '(x,y) := bs_box_bounds bx v in
 (- 2 ^ (Z.of_nat (bx_w bx) - 1) <=? x) && (y <? 2 ^ (Z.of_nat (bx_w bx) - 1)).

(* 2v, the value of v with its low bit cleared when v is 2x. *)
Definition bs_double (v : bs_val) : bs_val :=
 {| bv_k := 2 * bv_k v; bv_p := 2 * bv_p v; bv_q := 2 * bv_q v; bv_r := 2 * bv_r v; bv_j := bv_j v |}.

(* Whether v is at least 0 in the box. *)
Definition bs_nonneg (bx : bs_box) (v : bs_val) : bool :=
 0 <=? fst (bs_box_bounds bx v).

Definition bs_opnd (bx : bs_box) (env : PM.t bs_val) (o : GateIsa.opnd) : option bs_val :=
 match o with OReg r => PM.find r env | OImm z => Some (bs_lin (sext (bx_w bx) z) 0 0) end.

Definition bs_forget (env : PM.t bs_val) (ins : instr) : PM.t bs_val :=
 fold_left (fun e r => PM.remove r e) (loop_writes ins) env.

(* The exact value of each register of width w that the step determines. *)
Definition bs_step (F : lfacts) (bx : bs_box) (env : PM.t bs_val) (ins : instr) : PM.t bs_val :=
 let w := bx_w bx in
 let rest := bs_forget env ins in
 let keep := fun d v => if Nat.eqb (wdn F d) w && negb (is_cell F d) && bs_fits bx v then PM.add d v rest else rest in
 match ins with
 | IMov None d r => match PM.find r env with Some v => keep d v | None => rest end
 | IImm None d z => keep d (bs_lin (sext w z) 0 0)
 | IBin None BAdd w' d a o => if Nat.eqb w' w then
     match PM.find a env, bs_opnd bx env o with
     | Some u, Some v => match bs_add u v with Some x => keep d x | None => rest end
     | _, _ => rest end else rest
 | IBin None BSub w' d a o => if Nat.eqb w' w then
     match PM.find a env, bs_opnd bx env o with
     | Some u, Some v => match bs_add u (bs_neg v) with Some x => keep d x | None => rest end
     | _, _ => rest end else rest
 | IBin None BShrs w' d a (OImm k) => if Nat.eqb w' w && Nat.leb 2 w then
     match PM.find a env with
     | Some u =>
       if Z.eqb k 1 then match bs_half u with Some x => keep d x | None => rest end
       else if Z.eqb k (Z.of_nat w - 1) && bs_nonneg bx u then keep d (bs_lin 0 0 0) else rest
     | None => rest end else rest
 | IBin None BShr w' d a (OImm k) => if Nat.eqb w' w && Nat.leb 2 w then
     match PM.find a env with
     | Some u => if bs_nonneg bx u then
         if Z.eqb k 1 then match bs_half u with Some x => keep d x | None => rest end
         else if Z.eqb k (Z.of_nat w - 1) then keep d (bs_lin 0 0 0) else rest
       else rest
     | None => rest end else rest
 | IBin None BAnd w' d a (OImm z) => if Nat.eqb w' w && Nat.leb 2 w && Z.eqb (trunc w z) (2 ^ Z.of_nat w - 2) then
     match PM.find a env with
     | Some u => match bs_half u with Some x => keep d (bs_double x) | None => rest end
     | None => rest end else rest
 | _ => rest
 end.

(* The instructions from index 0 of block h up to index i of block b. *)
Definition bs_prefix (prog : program) (h b i : nat) : list instr :=
 List.concat (map (block_body prog) (seq h (b - h))) ++ firstn i (block_body prog b).

Definition bs_start (lo hi : reg) : PM.t bs_val :=
 PM.add hi (bs_lin 0 1 1) (PM.add lo (bs_lin 0 1 0) (PM.empty bs_val)).

Definition bs_env_at (F : lfacts) (prog : program) (bx : bs_box) (h b i : nat) (lo hi : reg) : PM.t bs_val :=
 fold_left (bs_step F bx) (bs_prefix prog h b i) (bs_start lo hi).

(* A search: lo and hi, each written once in the loop by a selection on one
   predicate q, so that when q holds lo takes x and hi keeps its value,
   and otherwise lo keeps its value and hi takes y (or the reverse). *)
Record term_search := {
 ts_lo : reg; ts_hi : reg; ts_w : nat; ts_q : reg; ts_x : reg; ts_y : reg; ts_swap : bool;
 ts_b1 : nat; ts_i1 : nat; ts_b2 : nat; ts_i2 : nat; ts_dmin : Z }.

Definition ts_sel_lo (d : term_search) : instr :=
 if ts_swap d then ISelp None (ts_w d) (ts_lo d) (OReg (ts_lo d)) (OReg (ts_x d)) (ts_q d)
 else ISelp None (ts_w d) (ts_lo d) (OReg (ts_x d)) (OReg (ts_lo d)) (ts_q d).
Definition ts_sel_hi (d : term_search) : instr :=
 if ts_swap d then ISelp None (ts_w d) (ts_hi d) (OReg (ts_y d)) (OReg (ts_hi d)) (ts_q d)
 else ISelp None (ts_w d) (ts_hi d) (OReg (ts_hi d)) (OReg (ts_y d)) (ts_q d).

Definition pos_ltb (b1 i1 b2 i2 : nat) : bool := Nat.ltb b1 b2 || (Nat.eqb b1 b2 && Nat.ltb i1 i2).

(* The least distance the back edge's comparison of lo and hi proves. *)
Definition ts_guard_min (c : cmp) (ra rb lo hi : reg) : option Z :=
 if Pos.eqb ra lo && Pos.eqb rb hi then
   match c with CLt => Some 1 | CLe => Some 0 | _ => None end
 else if Pos.eqb ra hi && Pos.eqb rb lo then
   match c with CGt => Some 1 | CGe => Some 0 | _ => None end
 else None.

Definition ts_head_range (F : lfacts) (prog : program) (cert : scalar_certificate) (h : nat) (r : reg) (w : nat)
 : option (option (Z * Z)) :=
 term_range_hull (map (fun x => let '(F',st) := x in
   if is_bottom st then Some (0,0) else cmp_range F' st (OReg r) (Z.of_nat w) true) (term_states F prog cert h 0)).

(* The box of L in [l1, l2] and H = L + D in [h1, h2], with D at least m. *)
Definition ts_box_of (w : nat) (m l1 l2 h1 h2 : Z) : bs_box :=
 {| bx_w := w; bx_l1 := l1; bx_l2 := l2; bx_d1 := Z.max m (h1 - l2); bx_d2 := h2 - l1; bx_h1 := h1; bx_h2 := h2 |}.

(* In the box, the new lo is at least L + 1 and the new hi at most L + D - 1. *)
Definition ts_eval_ok (F : lfacts) (prog : program) (bx : bs_box) (h : nat) (d : term_search) : bool :=
 match PM.find (ts_x d) (bs_env_at F prog bx h (ts_b1 d) (ts_i1 d) (ts_lo d) (ts_hi d)),
       PM.find (ts_y d) (bs_env_at F prog bx h (ts_b2 d) (ts_i2 d) (ts_lo d) (ts_hi d)) with
 | Some fx, Some fy =>
   match bs_add fx (bs_lin 0 (-1) 0), bs_add (bs_lin 0 1 1) (bs_neg fy) with
   | Some ex, Some ey => (1 <=? fst (bs_box_bounds bx ex)) && (1 <=? fst (bs_box_bounds bx ey))
   | _, _ => false end
 | _, _ => false end.

(* The box from the certificate's ranges of lo and hi at the head. *)
Definition ts_cert_ok (F : lfacts) (prog : program) (cert : scalar_certificate) (h : nat) (d : term_search) : bool :=
 let w := ts_w d in let m := ts_dmin d in
 match ts_head_range F prog cert h (ts_lo d) w, ts_head_range F prog cert h (ts_hi d) w with
 | Some (Some (l1,l2)), Some (Some (h1,h2)) =>
   if Z.max m (h1 - l2) <=? h2 - l1 then ts_eval_ok F prog (ts_box_of w m l1 l2 h1 h2) h d else true
 | Some None, Some _ | Some (Some _), Some None => true
 | _, _ => false end.

(* The least lo and the largest hi on the edges into the loop, with hi - lo
   at least m on each. *)
Definition ts_entry_box (F : lfacts) (prog : program) (cert : scalar_certificate) (h l : nat)
 (lo hi : reg) (w : nat) (m : Z) : option (Z * Z) :=
 fold_right (fun e acc => match e, acc with
   | Some (F',st), Some (A,B) =>
     match cmp_range F' st (OReg lo) (Z.of_nat w) true, cmp_range F' st (OReg hi) (Z.of_nat w) true with
     | Some (a1,a2), Some (b1,b2) => if m <=? b1 - a2 then Some (Z.min A a1, Z.max B b2) else None
     | _, _ => None end
   | _, _ => None end) (Some (2 ^ 64, - 2 ^ 64)) (term_entry_states F prog cert h l).

(* The box that the loop keeps from its entry, when it is entered only at
   its head: lo at least A, hi at most B, and hi - lo at least m at each
   arrival at the head, since each iteration raises lo, lowers hi, and
   takes the back edge only when hi - lo is at least m. *)
Definition ts_entry_ok (F : lfacts) (prog : program) (cert : scalar_certificate) (h l : nat) (d : term_search) : bool :=
 let w := ts_w d in let m := ts_dmin d in
 if negb (term_entered prog h l) then false else
 match ts_entry_box F prog cert h l (ts_lo d) (ts_hi d) w m with
 | Some (A,B) => if A + m <=? B then ts_eval_ok F prog (ts_box_of w m A (B - m) (A + m) B) h d else false
 | None => false end.

Definition term_search_ok (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_search) : bool :=
 let lo := ts_lo d in let hi := ts_hi d in let w := ts_w d in let q := ts_q d in
 (* the loop runs straight from its head to its latch *)
 short_all (fun v => short_all (fun t0 => Nat.eqb t0 (S v) || Nat.ltb l t0) (edge_targets prog v)) (seq h (l - h)) &&
 negb (Pos.eqb lo hi) && negb (Pos.eqb q lo) && negb (Pos.eqb q hi) &&
 Nat.eqb (wdn F lo) w && Nat.eqb (wdn F hi) w && Nat.leb 2 w && Nat.leb w 64 &&
 negb (is_cell F lo) && negb (is_cell F hi) && negb (is_cell F q) &&
 (* the two selections, the only writes of lo and hi in the loop *)
 match term_region_writes prog h l lo, term_region_writes prog h l hi with
 | [(b1,i1,s1)], [(b2,i2,s2)] =>
   Nat.eqb b1 (ts_b1 d) && Nat.eqb i1 (ts_i1 d) && Nat.eqb b2 (ts_b2 d) && Nat.eqb i2 (ts_i2 d) &&
   (if loop_instr_eq_dec s1 (ts_sel_lo d) then true else false) &&
   (if loop_instr_eq_dec s2 (ts_sel_hi d) then true else false) &&
   (* q keeps its value from one selection to the other *)
   forallb (fun z => let '(b',i',ins) := z in
     negb (term_writes q ins && ((pos_ltb b1 i1 b' i' && pos_ltb b' i' b2 i2) || (pos_ltb b2 i2 b' i' && pos_ltb b' i' b1 i1))))
     (flat_map (fun v => map (fun x => (v, fst x, snd x)) (term_indexed prog v)) (seq h (S l - h)))
 | _, _ => false end &&
 (* the back edge's guard compares the new bounds *)
 match last_instr (block_body prog l) with
 | Some (IBra (Some (neg,p)) _) =>
   negb (is_cell F p) && negb (Pos.eqb p lo) && negb (Pos.eqb p hi) &&
   match term_last_write prog l (block_len prog l) p with
   | Some (k0, ISetp None c w' true p' ra (OReg rb)) =>
     Nat.eqb w' w && Pos.eqb p' p && pos_ltb (ts_b1 d) (ts_i1 d) l k0 && pos_ltb (ts_b2 d) (ts_i2 d) l k0 &&
     match ts_guard_min (if neg then flip c else c) ra rb lo hi with
     | Some m => Z.eqb m (ts_dmin d) | None => false end
   | _ => false end
 | _ => false end &&
 (* the new distance is smaller whenever the old one is at least dmin *)
 if ts_cert_ok F prog cert h d then true else ts_entry_ok F prog cert h l d.

Definition term_propose_searches (prog : program) (h l : nat) : list term_search :=
 flat_map (fun b => flat_map (fun x => match snd x with
   | ISelp None w lo (OReg a) (OReg c) q =>
     let swap := Pos.eqb a lo in
     if negb (Pos.eqb a lo || Pos.eqb c lo) then [] else
     let xr := if swap then c else a in
     flat_map (fun v => flat_map (fun y => match snd y with
       | ISelp None w2 hi (OReg a2) (OReg c2) q2 =>
         if Pos.eqb q2 q && negb (Pos.eqb hi lo) && Nat.eqb w2 w &&
            (if swap then Pos.eqb c2 hi else Pos.eqb a2 hi) then
           let yr := if swap then a2 else c2 in
           match last_instr (block_body prog l) with
           | Some (IBra (Some (neg,p)) _) =>
             match term_last_write prog l (block_len prog l) p with
             | Some (_, ISetp None cc _ true _ ra (OReg rb)) =>
               match ts_guard_min (if neg then flip cc else cc) ra rb lo hi with
               | Some m => [{| ts_lo := lo; ts_hi := hi; ts_w := w; ts_q := q; ts_x := xr; ts_y := yr;
                    ts_swap := swap; ts_b1 := b; ts_i1 := fst x; ts_b2 := v; ts_i2 := fst y; ts_dmin := m |}]
               | None => [] end
             | _ => [] end
           | _ => [] end
         else []
       | _ => [] end) (term_indexed prog v)) (seq h (S l - h))
   | _ => [] end) (term_indexed prog b)) (seq h (S l - h)).

(* ------------------------------------------------------------------ *)
(* Searches over a loop body that branches.                            *)

(* A case of the symbolic walk of an iteration: the values of registers of
   width w as functions of the head's lower bound and distance, and the
   values of the predicates that a selection split on. *)
Record ds_case := { dc_env : PM.t bs_val; dc_preds : list (reg * bool) }.

Definition ds_pred (cs : ds_case) (p : reg) : option bool :=
 match find (fun x => Pos.eqb (fst x) p) (dc_preds cs) with Some (_,v) => Some v | None => None end.

(* The register d that a selection writes, from the value of its operand o. *)
Definition ds_pick (F : lfacts) (bx : bs_box) (env rest : PM.t bs_val) (d : reg) (o : GateIsa.opnd) : PM.t bs_val :=
 match bs_opnd bx env o with
 | Some v => if Nat.eqb (wdn F d) (bx_w bx) && negb (is_cell F d) then PM.add d v rest else rest
 | None => rest end.

(* A step of a case: a selection on a predicate of unknown value gives a
   case for each value. *)
Definition ds_step (F : lfacts) (bx : bs_box) (cs : ds_case) (ins : instr) : list ds_case :=
 let ps := filter (fun x => negb (term_writes (fst x) ins)) (dc_preds cs) in
 match ins with
 | ISelp None _ d a b p =>
   let env := dc_env cs in
   let rest := bs_forget env ins in
   match ds_pred cs p with
   | Some true => [{| dc_env := ds_pick F bx env rest d a; dc_preds := ps |}]
   | Some false => [{| dc_env := ds_pick F bx env rest d b; dc_preds := ps |}]
   | None =>
     if Pos.eqb p d || is_cell F p then [{| dc_env := rest; dc_preds := ps |}]
     else [{| dc_env := ds_pick F bx env rest d a; dc_preds := (p,true) :: ps |};
           {| dc_env := ds_pick F bx env rest d b; dc_preds := (p,false) :: ps |}]
   end
 | _ => [{| dc_env := bs_step F bx (dc_env cs) ins; dc_preds := ps |}]
 end.

Definition ds_walk (F : lfacts) (bx : bs_box) (cases : list ds_case) (body : list instr) : list ds_case :=
 fold_left (fun cs ins => flat_map (fun c => ds_step F bx c ins) cs) body cases.

(* The cases at the start of block b of the loop with head h: the start case
   at the head, otherwise the cases at the end of each earlier block of the
   loop with an edge to b; tbl holds the cases at the start of h, h+1, ... *)
Definition ds_entry_of (F : lfacts) (bx : bs_box) (prog : program) (h : nat) (start : ds_case)
 (tbl : list (list ds_case)) (b : nat) : list ds_case :=
 if Nat.eqb b h then [start] else
 flat_map (fun v => if existsb (Nat.eqb b) (edge_targets prog v)
   then ds_walk F bx (nth (v - h) tbl []) (block_body prog v) else []) (seq h (b - h)).

Fixpoint ds_build (F : lfacts) (bx : bs_box) (prog : program) (h : nat) (start : ds_case) (n : nat)
 : list (list ds_case) :=
 match n with
 | O => []
 | S m => let tbl := ds_build F bx prog h start m in tbl ++ [ds_entry_of F bx prog h start tbl (h + m)]
 end.

(* A search over a loop body that branches forward: the registers lo and
   hi of the head. *)
Record term_dsearch := { dsr_lo : reg; dsr_hi : reg }.

(* At the end of latch l, y holds the value that x has after index k0. *)
Definition ds_ends (prog : program) (l k0 : nat) (y x : reg) : bool :=
 let after := fun r => filter (fun z => Nat.ltb k0 (fst z) && term_writes r (snd z)) (term_indexed prog l) in
 match after x with
 | [] => if Pos.eqb y x then true else
   match after y with [(_, IMov None y' x')] => Pos.eqb y' y && Pos.eqb x' x | _ => false end
 | _ => false end.

(* The least distance hi - lo at the end of the latch that the back edge's
   guard proves when the back edge is taken. *)
Definition ds_guard_min (F : lfacts) (prog : program) (l : nat) (lo hi : reg) (w : nat) : option Z :=
 match last_instr (block_body prog l) with
 | Some (IBra (Some (neg,p)) _) =>
   match term_last_write prog l (block_len prog l) p with
   | Some (k0, ISetp None c w' true p' ra (OReg rb)) =>
     if Nat.eqb w' w && Pos.eqb p' p && negb (Pos.eqb ra p) && negb (Pos.eqb rb p) &&
        negb (is_cell F p) && negb (is_cell F ra) && negb (is_cell F rb) then
       let c' := if neg then flip c else c in
       if ds_ends prog l k0 lo ra && ds_ends prog l k0 hi rb then ts_guard_min c' lo hi lo hi
       else if ds_ends prog l k0 hi ra && ds_ends prog l k0 lo rb then ts_guard_min c' hi lo lo hi
       else None
     else None
   | _ => None end
 | _ => None end.

(* The case's lo and hi before the latch's branch give a smaller distance. *)
Definition ds_case_ok (bx : bs_box) (neg : bool) (p lo hi : reg) (cs : ds_case) : bool :=
 match ds_pred cs p with Some bv => Bool.eqb bv neg | None => false end ||
 match PM.find lo (dc_env cs), PM.find hi (dc_env cs) with
 | Some vl, Some vh =>
   match bs_add vh (bs_neg vl) with
   | Some e => match bs_add e (bs_lin 0 0 (-1)) with
     | Some e' => snd (bs_box_bounds bx e') <=? -1
     | None => false end
   | None => false end
 | _, _ => false end.

Definition term_dsearch_ok (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) (d : term_dsearch) : bool :=
 let lo := dsr_lo d in let hi := dsr_hi d in let w := wdn F lo in
 forallb (Nat.eqb l) (term_latches prog h l) &&
 (* inside the loop, edges go forward, but the back edge *)
 forallb (fun v => forallb (fun t0 => Nat.ltb v t0 || (Nat.eqb v l && Nat.eqb t0 h)) (edge_targets prog v))
   (seq h (S l - h)) &&
 negb (Pos.eqb lo hi) && Nat.eqb (wdn F hi) w && Nat.leb 2 w && Nat.leb w 64 &&
 negb (is_cell F lo) && negb (is_cell F hi) &&
 match ds_guard_min F prog l lo hi w, last_instr (block_body prog l) with
 | Some m, Some (IBra (Some (neg,p)) _) =>
   match ts_head_range F prog cert h lo w, ts_head_range F prog cert h hi w with
   | Some (Some (l1,l2)), Some (Some (h1,h2)) =>
     let bx := {| bx_w := w; bx_l1 := l1; bx_l2 := l2; bx_d1 := Z.max m (h1 - l2); bx_d2 := h2 - l1;
                  bx_h1 := h1; bx_h2 := h2 |} in
     if Z.max m (h1 - l2) <=? h2 - l1 then
       let start := {| dc_env := bs_start lo hi; dc_preds := [] |} in
       let tbl := ds_build F bx prog h start (S l - h) in
       forallb (ds_case_ok bx neg p lo hi)
         (ds_walk F bx (nth (l - h) tbl []) (term_slice prog l 0 (Nat.pred (block_len prog l))))
     else true
   | Some None, Some _ | Some (Some _), Some None => true
   | _, _ => false end
 | _, _ => false end.

Definition term_propose_dsearches (prog : program) (h l : nat) : list term_dsearch :=
 match last_instr (block_body prog l) with
 | Some (IBra (Some (_,p)) _) =>
   match term_last_write prog l (block_len prog l) p with
   | Some (k0, ISetp None _ _ true _ ra (OReg rb)) =>
     let fin := fun x => match filter (fun z => Nat.ltb k0 (fst z) &&
         match snd z with IMov None _ x' => Pos.eqb x' x | _ => false end) (term_indexed prog l) with
       | [(_, IMov None y _)] => y | _ => x end in
     [{| dsr_lo := fin ra; dsr_hi := fin rb |}; {| dsr_lo := fin rb; dsr_hi := fin ra |}]
   | _ => [] end
 | _ => [] end.

(* ------------------------------------------------------------------ *)
(* Counters that a test keeps from one value.                          *)

(* A counter c that block b of the loop increments at index i, its only
   write in the loop: c := a + 1, where a is c, or, when b is the latch,
   a := cvt y with y := cvt c, the only write of y in the loop, in the head;
   or c := c + 2^j, when c is a multiple of 2^j on entry to the largest
   loop with the same head and has no other write there.
   The branch that ends b compares c, or a conversion of c made after i,
   and on the edge that stays in the loop the comparison keeps c from one
   value k: from the other operand's interval, its reading at the
   comparison's width or the certificate's states at the branch. The blocks
   of the loop before b branch to b or before it or leave the loop, and
   every latch is b or follows it. The rank, k minus the step minus c modulo
   2^w, divided by the step, falls by 1 at each update after which the loop
   continues, so c needs no interval. *)
Record term_eqexit := { te_c : reg; te_b : nat; te_i : nat }.

(* The source a of the update c := a + 1 at index i of block b: None when a
   is c, or y, its width W, the signedness of y := cvt c, and y's index in
   the head, for a := cvt y. *)
Definition te_alias (F : lfacts) (prog : program) (h l b i : nat) (c a : reg)
 : option (option (reg * nat * bool * nat)) :=
 if Pos.eqb a c then Some None else
 if negb (Nat.eqb b l && Nat.ltb h l && negb (is_cell F a) && Nat.eqb (wdn F a) (wdn F c)) then None else
 match term_last_write prog b i a with
 | Some (_, ICvt None _ W _ a' y) =>
   if Pos.eqb a' a && Nat.leb (wdn F c) W && Nat.eqb (wdn F y) W && negb (Pos.eqb y c) && negb (is_cell F y) then
     match term_region_writes prog h l y with
     | [(b2, my, ICvt None _ w2 sg2 y' c')] =>
       if Nat.eqb b2 h && Nat.eqb w2 (wdn F c) && Pos.eqb y' y && Pos.eqb c' c
       then Some (Some (y, W, sg2, my)) else None
     | _ => None end
   else None
 | _ => None end.

(* The comparison with its operands exchanged. *)
Definition te_swap (c : cmp) : cmp :=
 match c with CLt => CGt | CGt => CLt | CLe => CGe | CGe => CLe | CEq => CEq | CNe => CNe end.

(* The value that x cannot have when x cc y holds and y lies in [lo, hi]. *)
Definition te_excluded (cc : cmp) (lo hi : Z) : option Z :=
 match cc with
 | CLt => Some hi | CLe => Some (hi + 1) | CGt => Some lo | CGe => Some (lo - 1)
 | CNe => if Z.eqb lo hi then Some lo else None
 | CEq => None end.

(* The comparison at index kk of block b reads x at width w' (signed if sg)
   as c at c's width w, signed if the result is true: x is c, or x := cvt c
   at index m, after i, and x is not written after m in b. *)
Definition te_reading (F : lfacts) (prog : program) (b i kk : nat) (c x : reg) (w' : nat) (sg : bool) : option bool :=
 let w := wdn F c in
 if Pos.eqb x c then (if Nat.eqb w' w then Some sg else None) else
 if is_cell F x then None else
 match term_last_write prog b kk x with
 | Some (m, ICvt None _ w2 sgx x' c') =>
   if Pos.eqb x' x && Pos.eqb c' c && Nat.eqb w2 w && Nat.eqb (wdn F x) w' && Nat.ltb i m &&
      term_kept prog b (S m) (block_len prog b) x then
     if sgx then (if sg && Nat.leb w w' then Some true else None)
     else if Nat.ltb w w' || (negb sg && Nat.leb w w') then Some false else None
   else None
 | _ => None end.

(* The value k that x cannot have when x cc y holds at width w' (signed if
   sg), from y's interval at the branch that ends block b: y's reading at
   that width, or the certificate's states there. k must be a value of c
   read at c's width w, signed if s. *)
Definition te_value (F : lfacts) (prog : program) (cert : scalar_certificate) (b w : nat) (s : bool)
 (cc : cmp) (w' : nat) (sg : bool) (y : opnd) : option Z :=
 if negb (Nat.leb 1 w' && Nat.leb w' 64 && match y with OReg r => Nat.eqb (wdn F r) w' | OImm _ => true end)
 then None else
 let fits k := if Z.eqb (extend w s k) k then Some k else None in
 let wz := Z.of_nat w' in
 let rd := match y with OImm z => (extend w' sg z, extend w' sg z) | OReg _ => (lo_ok wz sg, hi_ok wz sg) end in
 match (match te_excluded cc (fst rd) (snd rd) with Some k => fits k | None => None end) with
 | Some k => Some k
 | None =>
   match term_range_hull (flat_map (fun x => let '(F',st) := x in
       if is_bottom st then [] else [cmp_range F' st y wz sg])
       (term_states F prog cert b (Nat.pred (block_len prog b)))) with
   | Some (Some (lo,hi)) => match te_excluded cc lo hi with Some k => fits k | None => None end
   | _ => None end
 end.

(* The value that the test at the end of block b keeps c from on the edge
   that stays in the loop: the taken branch when b is the latch, the fall
   through when the branch leaves the loop. *)
Definition te_test (F : lfacts) (prog : program) (cert : scalar_certificate) (h l b i : nat) (c : reg) : option Z :=
 match last_instr (block_body prog b) with
 | Some (IBra (Some (neg,p)) tg) =>
   let stay := if term_in h l tg then (if Nat.eqb b l then Some (negb neg) else None)
               else if Nat.leb (S b) l then Some neg else None in
   match stay, term_last_write prog b (block_len prog b) p with
   | Some v, Some (kk, ISetp None cc w' sg p' ra ob) =>
     if negb (Pos.eqb p' p && negb (is_cell F p) && Nat.ltb i kk &&
        forallb (fun r => negb (Pos.eqb r p) && negb (is_cell F r) && term_kept prog b (S kk) (block_len prog b) r)
          (ra :: opnd_inputs ob))
     then None else
     let cv := if v then cc else flip cc in
     match te_reading F prog b i kk c ra w' sg with
     | Some s => te_value F prog cert b (wdn F c) s cv w' sg ob
     | None => match ob with
       | OReg x => match te_reading F prog b i kk c x w' sg with
         | Some s => te_value F prog cert b (wdn F c) s (te_swap cv) w' sg (OReg ra) | None => None end
       | OImm _ => None end
     end
   | _, _ => None end
 | _ => None end.

(* The alignments of c on every edge into [h, l] from a block outside it. *)
Definition te_entry_aligns (F : lfacts) (prog : program) (cert : scalar_certificate) (h l : nat) (c : reg)
 : list (option Z) :=
 flat_map (fun v =>
   if term_in h l v then [] else
   if negb (existsb (term_in h l) (edge_targets prog v)) then [] else
   match last_instr (block_body prog v) with
   | None => [None]
   | Some last =>
     flat_map (fun x => let '(F',st) := x in
       if is_bottom st then [] else
       if negb (wf_instr F' last) then [None] else
       flat_map (fun e => let '(u,os) := e in
         if term_in h l u then match os with
           | Some q => if is_bottom q then [] else [Some (reg_align F' q c)]
           | None => [] end else [])
         (edge_states F' prog v (transfer F' st last)))
       (term_states F prog cert v (Nat.pred (block_len prog v)))
   end) (seq 0 (List.length prog)).

(* The last block of the largest loop with head h. *)
Definition te_outer (prog : program) (h l : nat) : nat :=
 fold_right Nat.max l (term_latches prog h (List.length prog)).

(* The step z of the update c := a + z: 1, or 2^j for a counter c := c + z
   that is a multiple of 2^j on every edge into the blocks h to L, the
   largest loop with head h, and has no other write there, with the
   excluded value k a multiple of 2^j, so that c is a multiple of 2^j in
   those blocks. *)
Definition te_step_ok (F : lfacts) (prog : program) (cert : scalar_certificate) (h l b i : nat) (c a : reg) (z k : Z) : bool :=
 if Z.eqb z 1 then true else
 let L := te_outer prog h l in
 if Pos.eqb a c && Z.ltb 1 z && Z.eqb (2 ^ Z.log2 z) z && Z.ltb (Z.log2 z) (Z.of_nat (wdn F c)) && Z.eqb (k mod z) 0 &&
    match term_region_writes prog h L c with [(b', i', _)] => Nat.eqb b' b && Nat.eqb i' i | _ => false end
 then short_all (fun o => match o with Some x => Z.log2 z <=? x | None => false end) (te_entry_aligns F prog cert h L c)
 else false.

Definition term_eqexit_ok (F : lfacts) (prog : program) (cert : scalar_certificate) (h l : nat) (d : term_eqexit) : bool :=
 let c := te_c d in let b := te_b d in let i := te_i d in
 if negb (Nat.leb h b && Nat.leb b l && negb (is_cell F c) &&
   forallb (fun v => forallb (fun t => Nat.leb t b || negb (term_in h l t)) (edge_targets prog v)) (seq h (b - h)) &&
   forallb (Nat.leb b) (term_latches prog h l)) then false else
 match term_region_writes prog h l c with
 | [(b', i', IBin None BAdd w' c1 a (OImm z))] =>
   if negb (Nat.eqb b' b && Nat.eqb i' i && Nat.eqb w' (wdn F c) && Pos.eqb c1 c &&
     match te_alias F prog h l b i c a with Some _ => true | None => false end) then false else
   match te_test F prog cert h l b i c with
   | Some k => te_step_ok F prog cert h l b i c a z k
   | None => false end
 | _ => false end.

(* For each branch of the loop, the counters its comparison reads: an
   operand, or the source of its last conversion in the block. *)
Definition term_propose_eqexits (prog : program) (h l : nat) : list term_eqexit :=
 flat_map (fun b => match last_instr (block_body prog b) with
   | Some (IBra (Some (_,p)) _) =>
     match term_last_write prog b (block_len prog b) p with
     | Some (kk, ISetp None _ _ _ _ ra ob) =>
       flat_map (fun x =>
         flat_map (fun c => match term_region_writes prog h l c with
           | [(b', i, _)] => if Nat.eqb b' b then [{| te_c := c; te_b := b; te_i := i |}] else []
           | _ => [] end)
         (x :: match term_last_write prog b kk x with Some (_, ICvt None _ _ _ _ c) => [c] | _ => [] end))
       (ra :: opnd_inputs ob)
     | _ => [] end
   | _ => [] end) (seq h (S l - h)).

(* ------------------------------------------------------------------ *)
(* Proposals: untrusted; every descriptor is checked.                  *)

Definition prod_eq_dec (x y : Z * Z) : {x = y} + {x <> y}.
Proof. decide equality; apply Z.eq_dec. Defined.

Definition term_kinds (step : instr) : list term_kind :=
 match step with
 | IBin None BAdd _ _ _ _ => [KUp; KDown]
 | IBin None BSub _ _ _ _ => [KDown]
 | IBin None (BShr|BShrs) _ _ _ (OImm _) => [KHalve]
 | IBin None BShl _ _ _ (OImm _) => [KDouble]
 | _ => [] end.

(* The comparisons that decide a conditional branch at the end of a block of
   the loop, through conjunctions and disjunctions of predicates. *)
Fixpoint term_pred_defs (prog : program) (b i : nat) (p : reg) (fuel : nat) : list nat :=
 match fuel with
 | O => []
 | S f =>
   match term_last_write prog b i p with
   | Some (k, ISetp _ _ _ _ _ _ _) => [k]
   | Some (k, IAndp _ _ p1 p2) | Some (k, IOrp _ _ p1 p2) =>
     term_pred_defs prog b k p1 f ++ term_pred_defs prog b k p2 f
   | _ => [] end
 end.

(* A state at each comparison that a branch of the loop decides on, with the
   comparison's width and operands: the certificate's first state at the
   block's start with only the facts of the registers that decide the
   operands, walked to the comparison. *)
Definition term_guard_states (F : lfacts) (prog : program) (cert : scalar_certificate) (h l : nat)
 : list (lfacts * astate * nat * reg * GateIsa.opnd) :=
 flat_map (fun b =>
   match last_instr (block_body prog b) with
   | Some (IBra (Some (_,p)) _) =>
     flat_map (fun k => match fetch prog b k with
       | Some (ISetp _ _ w _ _ a o) =>
         let keep := term_keep prog b (a :: opnd_inputs o) in
         match term_states F prog cert b 0 with
         | (F0,st0)::_ => [(F0, term_walk F0 (whole_state_filter keep (fun _ => false) st0) (term_slice prog b 0 k), w, a, o)]
         | [] => [] end
       | _ => [] end) (term_pred_defs prog b (block_len prog b) p 4)
   | _ => [] end) (seq h (S l - h)).

(* The values that those comparisons compare with, nearest the bound first. *)
Definition term_limits (gs : list (lfacts * astate * nat * reg * GateIsa.opnd)) (sg : bool) : list Z :=
 flat_map (fun x => let '(F0,st0,w,a,o) := x in
   flat_map (fun y => match cmp_range F0 st0 y (Z.of_nat w) sg with
     | Some (u,v) => [v-1; v; u+1; u; u-1; v+1] | None => [] end) [o; OReg a]) gs.

Definition term_alias_of (prog : program) (h l b j : nat) (c a : reg) : option term_alias :=
 if Pos.eqb a c then Some ASame else
 match term_last_write prog b j a with
 | Some (m, IMov None _ c') => if Pos.eqb c' c then Some (ABlock m) else None
 | Some _ => None
 | None => if Nat.eqb b l && negb (Nat.eqb h l) then
     match term_region_writes prog h l a with
     | [(b',m,IMov None _ c')] => if Nat.eqb b' h && Pos.eqb c' c then Some (AHead m) else None
     | _ => None end else None
 end.

Definition term_propose_counters (F : lfacts) (prog : program) (cert : scalar_certificate)
 (h l : nat) : list term_counter :=
 let gs := term_guard_states F prog cert h l in
 let limits_s := nodup Z.eq_dec (term_limits gs true) in
 let limits_u := nodup Z.eq_dec (term_limits gs false) in
 let entry := term_entry_states F prog cert h l in
 let cands := flat_map (fun b => flat_map (fun x => match snd x with
   | IBin None _ _ d _ _ | IMov None d _ => [d] | _ => [] end) (term_indexed prog b))
   (if Nat.eqb h l then [h] else [h; l]) in
 flat_map (fun c =>
   match term_region_writes prog h l c with
   | [(b,i,ins)] =>
     let step := match ins with
       | IMov None _ t => match term_last_write prog b i t with
         | Some (j, (IBin None _ _ _ _ _ as s)) => Some (j,s,None)
         | Some _ => None
         | None => if Nat.eqb b l && negb (Nat.eqb h l) then
             match term_region_writes prog h l t with
             | [(b2,m,(IBin None _ _ _ _ _ as s))] => if Nat.eqb b2 h then Some (i,s,Some m) else None
             | _ => None end else None end
       | IBin None _ _ _ _ _ => Some (i,ins,None)
       | _ => None end in
     match step with
     | Some (j, (IBin None op w t a o as s), sh) =>
       match (match sh with Some _ => if Pos.eqb a c then Some ASame else None | None => term_alias_of prog h l b j c a end) with
       | Some al => flat_map (fun k =>
           let sg := term_signed k in
           let ranges := match term_range_hull (map (fun x => match x with
               | Some (F',st) => cmp_range F' st (OReg c) (Z.of_nat w) sg | None => None end) entry) with
             | Some (Some (elo,ehi)) =>
               map (fun z => match k with
                 | KUp | KDouble => (elo, Z.max ehi z) | KDown | KHalve => (Z.min elo z, ehi) end)
                 (if sg then limits_s else limits_u) ++ [(elo,ehi)]
             | Some None => [(0,0)]
             | None => [] end in
           map (fun r => {| tc_c := c; tc_b := b; tc_i := i; tc_j := j; tc_step := s; tc_alias := al;
                tc_shift := sh; tc_kind := k; tc_lo := fst r; tc_hi := snd r |})
             (nodup (fun x y => prod_eq_dec x y) ranges))
           (match op, o with
            | BAdd, OImm z => if Z.ltb z 0 then [KDown] else [KUp]
            | _, _ => term_kinds s end)
       | None => [] end
     | _ => [] end
   | _ => [] end) (nodup Pos.eq_dec cands).

(* The back edges are never taken: from the certificate's states at each
   latch's branch, the edge to the head has no state. *)
Definition term_dead_ok (F : lfacts) (prog : program) (cert : scalar_certificate) (h l : nat) : bool :=
 forallb (fun v => match last_instr (block_body prog v) with
   | None => false
   | Some last =>
     short_all (fun x => let '(F',st) := x in
       if is_bottom st then true else
       wf_instr F' last &&
       forallb (fun e => let '(u,os) := e in
         if Nat.eqb u h then match os with Some s => is_bottom s | None => true end else true)
         (edge_states F' prog v (transfer F' st last)))
       (term_states F prog cert v (Nat.pred (block_len prog v)))
   end) (term_latches prog h l).

Definition term_loop_ok (F : lfacts) (prog : program) (cert : scalar_certificate) (e : nat * nat) : bool :=
 let '(l,h) := e in
 if short_any (term_eqexit_ok F prog cert h l) (term_propose_eqexits prog h l) then true
 else if short_any (term_counter_ok F prog cert h l) (term_propose_counters F prog cert h l) then true
 else if short_any (term_search_ok F prog cert h l) (term_propose_searches prog h l) then true
 else if short_any (term_dsearch_ok F prog cert h l) (term_propose_dsearches prog h l) then true
 else term_dead_ok F prog cert h l.

Definition term_check (F : lfacts) (prog : program) (cert : scalar_certificate) : bool :=
 if term_shape prog then short_all (term_loop_ok F prog cert) (term_edges prog) else false.

(* ------------------------------------------------------------------ *)
(* Ending.                                                             *)

(* A thread ends when it reaches a position without an instruction; its
   next step does not exist. This holds for every launch that establishes
   the facts. *)
Definition kernel_ends (F : lfacts) (prog : program) : Prop :=
 forall E, launch_ok F E ->
 exists t s, run' F prog E t = Some s /\ fetch prog (blk s) (idx s) = None.

(* The facts a certificate's state holds under: the launch facts or one
   side of a split, which only narrows a fixed register's bounds. *)
Definition term_same (F F' : lfacts) : Prop :=
 F' = F \/ exists r cut low, F' = split_launch F r cut low.

Lemma term_same_step : forall F F' prog E, term_same F F' -> phase_step F' prog E = phase_step F prog E.
Proof. intros F F' prog E [->|[r[cut[low ->]]]]; reflexivity. Qed.
Lemma term_same_wf : forall F F' i, term_same F F' -> wf_instr F' i = wf_instr F i.
Proof. intros F F' i [->|[r[cut[low ->]]]]; reflexivity. Qed.
Lemma term_same_wdn : forall F F' r, term_same F F' -> wdn F' r = wdn F r.
Proof. intros F F' r [->|[x[cut[low ->]]]]; reflexivity. Qed.
Lemma term_same_cell : forall F F' r, term_same F F' -> is_cell F' r = is_cell F r.
Proof. intros F F' r [->|[x[cut[low ->]]]]; reflexivity. Qed.
Lemma term_same_regs : forall F F' E rs, term_same F F' -> regs_ok F' E rs <-> regs_ok F E rs.
Proof. intros F F' E rs [->|[x[cut[low ->]]]]; reflexivity. Qed.
Lemma term_same_run : forall F F' prog E t, term_same F F' -> run' F' prog E t = run' F prog E t.
Proof. intros F F' prog E t [->|[x[cut[low ->]]]]; reflexivity. Qed.

Lemma term_run_succ : forall F prog E t,
 run' F prog E (S t) = match run' F prog E t with Some s => phase_step F prog E t s | None => None end.
Proof. reflexivity. Qed.

(* Every reachable state inside the program agrees with one of the
   certificate's states at its position. *)
Theorem term_states_sound : forall F prog cert E t s,
 check_scalar_certificate F prog cert = true -> launch_ok F E ->
 run' F prog E t = Some s -> (blk s < List.length prog)%nat ->
 (idx s < block_len prog (blk s))%nat /\
 exists F' st, In (F',st) (term_states F prog cert (blk s) (idx s)) /\ term_same F F' /\
   env_ok F' E /\ regs_ok F' E (regs s) /\ state_ok F' t st (regs s).
Proof.
 intros F prog cert E t s Hcert HE Hrun Hb.
 destruct cert as [[cases|r cut low high]|cached|]; cbn [check_scalar_certificate] in Hcert.
 - unfold check_block_certificate in Hcert. apply andb_true_iff in Hcert. destruct Hcert as [Hlf Hc].
   destruct cases as [inv|]; [|discriminate]. cbn [check_case_certificate] in Hc.
   destruct (run_inv F prog E inv (check_facts_of_check_cases F prog E inv Hc) HE t s Hrun)
     as [Hrs Hinv].
   destruct (Hinv Hb) as [Hi [st [Hin Hall]]]. split; [exact Hi|].
   pose proof (inv_walk F prog E st (blk s) (idx s) t (regs s) Hall) as Hst.
   exists F,(walk F st (firstn (idx s) (block_body prog (blk s)))). split.
   + cbn [term_states]. apply in_map_iff. exists st. split; [reflexivity|exact Hin].
   + split; [left; reflexivity|]. split; [exact (env_of_launch F prog E Hlf HE)|].
     split; assumption.
 - unfold check_block_certificate in Hcert. rewrite !andb_true_iff in Hcert.
   destruct Hcert as [Hlf [Hl Hh]].
   destruct low as [lo|]; [|discriminate]. destruct high as [hi|]; [|discriminate].
   cbn [check_case_certificate] in Hl, Hh.
   destruct (Z.lt_ge_cases (e_init E r) cut) as [Hside|Hside].
   + pose proof (launch_split F E r cut true HE Hside) as HE'.
     set (F' := split_launch F r cut true) in *.
     assert (Hrun' : run' F' prog E t = Some s) by exact Hrun.
     destruct (run_inv F' prog E lo (check_facts_of_check_cases F' prog E lo Hl) HE' t s Hrun')
       as [Hrs Hinv].
     destruct (Hinv Hb) as [Hi [st [Hin Hall]]]. split; [exact Hi|].
     pose proof (inv_walk F' prog E st (blk s) (idx s) t (regs s) Hall) as Hst.
     exists F',(walk F' st (firstn (idx s) (block_body prog (blk s)))). split.
     * cbn [term_states]. apply in_or_app. left. apply in_map_iff. exists st. split; [reflexivity|exact Hin].
     * split; [right; exists r,cut,true; reflexivity|].
       assert (Hlf' : lf_ok F' = true).
       { unfold check_cases in Hl. rewrite !andb_true_iff in Hl. tauto. }
       split; [exact (env_of_launch F' prog E Hlf' HE')|]. split; assumption.
   + pose proof (launch_split F E r cut false HE Hside) as HE'.
     set (F' := split_launch F r cut false) in *.
     assert (Hrun' : run' F' prog E t = Some s) by exact Hrun.
     destruct (run_inv F' prog E hi (check_facts_of_check_cases F' prog E hi Hh) HE' t s Hrun')
       as [Hrs Hinv].
     destruct (Hinv Hb) as [Hi [st [Hin Hall]]]. split; [exact Hi|].
     pose proof (inv_walk F' prog E st (blk s) (idx s) t (regs s) Hall) as Hst.
     exists F',(walk F' st (firstn (idx s) (block_body prog (blk s)))). split.
     * cbn [term_states]. apply in_or_app. right. apply in_map_iff. exists st. split; [reflexivity|exact Hin].
     * split; [right; exists r,cut,false; reflexivity|].
       assert (Hlf' : lf_ok F' = true).
       { unfold check_cases in Hh. rewrite !andb_true_iff in Hh. tauto. }
       split; [exact (env_of_launch F' prog E Hlf' HE')|]. split; assumption.
 - pose proof (check_cached_certificate_lf_ok F prog cached Hcert) as Hlf.
   unfold check_cached_certificate in Hcert. apply indexed_check_implies in Hcert.
   pose proof (env_of_launch F prog E Hlf HE) as Henv.
   assert (HT : env_ok F E -> forall t s s' st i,
     fetch prog (blk s) (idx s) = Some i ->
     step prog (wdn F) (e_mem E) (e_arb E) (e_shm E) (e_init E) (lf_cells F) (lf_locals F) (lf_local_window F) t s = Some s' ->
     state_ok F t st (regs s) -> regs_ok F E (regs s) -> wf_instr F i = true ->
     state_ok F (S t) (coherent_transfer F (cert_written cached) (cert_goals cached) (cert_descriptor cached) st i) (regs s') /\
     regs_ok F E (regs s')).
   { intros HE0 t0 s0 s0' st i Hf Hs Hst Hrs Hwf.
     exact (coherent_transfer_sound F prog E HE0 t0 s0 s0' st _ _ _ i Hf Hs Hst Hrs Hwf). }
   destruct (cached_run_inv F prog E _ HT _ Hcert HE t s Hrun) as [Hrs Hinv].
   destruct (Hinv Hb) as [Hi [k [Hin Hst]]]. split; [exact Hi|].
   exists F,(cached_at k (idx s)). split.
   + cbn [term_states]. apply in_map_iff. exists k. split; [reflexivity|exact Hin].
   + split; [left; reflexivity|]. split; [exact Henv|]. split; assumption.
 - discriminate.
Qed.

(* ------------------------------------------------------------------ *)
(* Steps between positions.                                            *)

(* A step either moves to the next instruction of its block, or leaves
   the block from its last instruction for index 0 of an edge's target or
   for the end of the program. *)
Lemma term_step_shape : forall F prog E t s s' ins,
 structured prog = true ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 (blk s' = blk s /\ idx s' = S (idx s) /\ (S (idx s) < block_len prog (blk s))%nat) \/
 (idx s' = 0%nat /\ S (idx s) = block_len prog (blk s) /\
  last_instr (block_body prog (blk s)) = Some ins /\
  (blk s' = List.length prog \/ In (blk s') (edge_targets prog (blk s)))).
Proof.
 intros F prog E t s s' ins Hstr Hf Hs.
 destruct (fetch_block F prog E _ _ _ Hf) as [Hb Hi].
 pose proof (structured_block F prog E (blk s) Hstr Hb) as [_ Hjump].
 assert (Hnth : nth_error (block_body prog (blk s)) (idx s) = Some ins) by exact Hf.
 assert (Hlast : S (idx s) = block_len prog (blk s) -> last_instr (block_body prog (blk s)) = Some ins).
 { intros Hl. exact (last_of_nth F prog E _ _ _ Hnth Hl). }
 destruct (step_pos F prog E t s s' ins Hf Hs) as [[Hn [Hbra Hexit]]|[[g [tg [-> [Hg Hp]]]]|[g [-> [Hg Hp]]]]].
 - unfold next_pos in Hn. destruct (Nat.ltb_spec (S (idx s)) (block_len prog (blk s))) as [Hlt|Hge].
   + left. injection Hn as Hb' Hi'. split; [exact Hb'|]. split; [exact Hi'|exact Hlt].
   + right. injection Hn as Hb' Hi'. assert (Hl : S (idx s) = block_len prog (blk s)) by lia.
     split; [exact Hi'|]. split; [exact Hl|]. specialize (Hlast Hl). split; [exact Hlast|].
     right. unfold edge_targets. rewrite Hlast. rewrite Hb'.
     destruct ins as [| | | | | | | | | | | | | | |g tg|g| | | | | | | |];
       try (left; reflexivity).
     * destruct g as [[neg p]|]; [right; left; reflexivity|].
       specialize (Hbra None tg eq_refl). cbn in Hbra. discriminate.
     * destruct g as [[neg p]|]; [left; reflexivity|].
       specialize (Hexit None eq_refl). cbn in Hexit. discriminate.
 - right. injection Hp as Hb' Hi'. assert (Hl : S (idx s) = block_len prog (blk s)) by
     (unfold block_len; exact (Hjump (idx s) _ Hnth eq_refl)).
   split; [exact Hi'|]. split; [exact Hl|]. specialize (Hlast Hl). split; [exact Hlast|].
   right. unfold edge_targets. rewrite Hlast. rewrite Hb'.
   destruct g as [[neg p]|]; left; reflexivity.
 - right. injection Hp as Hb' Hi'. assert (Hl : S (idx s) = block_len prog (blk s)) by
     (unfold block_len; exact (Hjump (idx s) _ Hnth eq_refl)).
   split; [exact Hi'|]. split; [exact Hl|]. split; [exact (Hlast Hl)|]. left; exact Hb'.
Qed.

(* A step to an earlier or the same block is a back edge's branch. *)
Lemma term_step_back : forall F prog E t s s' ins,
 structured prog = true ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 (blk s' <= blk s)%nat -> idx s' = 0%nat ->
 In (blk s, blk s') (term_edges prog).
Proof.
 intros F prog E t s s' ins Hstr Hf Hs Hle Hi0.
 destruct (fetch_block F prog E _ _ _ Hf) as [Hb Hi].
 pose proof (structured_block F prog E (blk s) Hstr Hb) as [_ Hjump].
 assert (Hnth : nth_error (block_body prog (blk s)) (idx s) = Some ins) by exact Hf.
 unfold term_edges. apply in_flat_map. exists (blk s). split; [apply in_seq; lia|].
 destruct (step_pos F prog E t s s' ins Hf Hs) as [[Hn [Hbra Hexit]]|[[g [tg [-> [Hg Hp]]]]|[g [-> [Hg Hp]]]]].
 - unfold next_pos in Hn. destruct (Nat.ltb (S (idx s)) (block_len prog (blk s))); injection Hn as Hb' Hi'; lia.
 - injection Hp as Hb' Hi'.
   assert (Hl : S (idx s) = length (block_body prog (blk s))) by exact (Hjump (idx s) _ Hnth eq_refl).
   unfold term_edge_at. rewrite (last_of_nth F prog E _ _ _ Hnth Hl).
   rewrite <- Hb'. destruct (Nat.leb_spec (blk s') (blk s)) as [_|Hc]; [left; reflexivity|lia].
 - injection Hp as Hb' Hi'. pose proof (fetch_block F prog E _ _ _ Hf). lia.
Qed.

Lemma term_edges_spec : forall prog l h, In (l,h) (term_edges prog) ->
 (l < List.length prog)%nat /\ (h <= l)%nat /\
 exists g, last_instr (block_body prog l) = Some (IBra g h).
Proof.
 intros prog l h H. unfold term_edges in H. apply in_flat_map in H. destruct H as [b [Hb H]].
 apply in_seq in Hb. unfold term_edge_at in H.
 destruct (last_instr (block_body prog b)) as [[| | | | | | | | | | | | | | |g t|g| | | | | | | |]|] eqn:Hl;
   cbn in H; try contradiction.
 destruct (Nat.leb_spec t b) as [Ht|Ht]; cbn in H; [|contradiction].
 destruct H as [H|H]; [|contradiction]. injection H as <- <-.
 split; [lia|]. split; [exact Ht|]. exists g. exact Hl.
Qed.

(* One back edge leaves each block. *)
Lemma term_edges_block : forall prog l h1 h2,
 In (l,h1) (term_edges prog) -> In (l,h2) (term_edges prog) -> h1 = h2.
Proof.
 intros prog l h1 h2 H1 H2.
 destruct (term_edges_spec prog l h1 H1) as [_ [_ [g1 E1]]].
 destruct (term_edges_spec prog l h2 H2) as [_ [_ [g2 E2]]].
 rewrite E1 in E2. injection E2 as _ ->. reflexivity.
Qed.

Section Shape.
Variable prog : program.
Hypothesis Hshape : term_shape prog = true.

Lemma term_shape_structured : structured prog = true.
Proof. unfold term_shape in Hshape. rewrite !andb_true_iff in Hshape. tauto. Qed.

Lemma term_shape_nested : forall e1 e2, In e1 (term_edges prog) -> In e2 (term_edges prog) ->
 term_nested e1 e2 = true.
Proof.
 intros e1 e2 H1 H2. unfold term_shape in Hshape. rewrite !andb_true_iff in Hshape.
 destruct Hshape as [[_ Hn] _]. rewrite forallb_forall in Hn. specialize (Hn e1 H1).
 rewrite forallb_forall in Hn. exact (Hn e2 H2).
Qed.

Lemma term_shape_edge : forall l h, In (l,h) (term_edges prog) -> (0 < h)%nat.
Proof.
 intros l h H. unfold term_shape in Hshape. rewrite !andb_true_iff in Hshape.
 destruct Hshape as [_ He]. rewrite forallb_forall in He. specialize (He (l,h) H).
 cbv beta iota in He. apply Nat.ltb_lt in He. exact He.
Qed.

(* Two loops that share a block are nested. *)
Lemma term_laminar : forall l1 h1 l2 h2 b,
 In (l1,h1) (term_edges prog) -> In (l2,h2) (term_edges prog) ->
 (h1 <= b <= l1)%nat -> (h2 <= b <= l2)%nat ->
 ((h1 <= h2)%nat /\ (l2 <= l1)%nat) \/ ((h2 <= h1)%nat /\ (l1 <= l2)%nat).
Proof.
 intros l1 h1 l2 h2 b H1 H2 B1 B2. pose proof (term_shape_nested _ _ H1 H2) as Hn.
 unfold term_nested in Hn. rewrite !orb_true_iff, !andb_true_iff, !Nat.ltb_lt, !Nat.leb_le in Hn. lia.
Qed.

Lemma term_latches_in : forall h l v, In (v,h) (term_edges prog) -> (h <= v <= l)%nat ->
 In v (term_latches prog h l).
Proof.
 intros h l v He Hv. unfold term_latches. apply in_map_iff. exists (v,h). split; [reflexivity|].
 apply filter_In. split; [exact He|]. cbn [fst snd]. rewrite Nat.eqb_refl.
 rewrite (proj2 (Nat.leb_le h v) ltac:(lia)), (proj2 (Nat.leb_le v l) ltac:(lia)). reflexivity.
Qed.

(* When term_entered holds, a thread enters the loop from outside only at its head. *)
Lemma term_entered_head : forall l h b v,
 term_entered prog h l = true -> (b < List.length prog)%nat -> term_in h l b = false ->
 In v (edge_targets prog b) -> term_in h l v = true -> v = h.
Proof.
 intros l h b v Hent Hb Hout Hv Hin.
 unfold term_entered in Hent. rewrite forallb_forall in Hent.
 specialize (Hent b ltac:(apply in_seq; lia)). rewrite Hout in Hent. cbn [orb] in Hent.
 rewrite forallb_forall in Hent. specialize (Hent v Hv). unfold term_in in Hin.
 rewrite andb_true_iff, !Nat.leb_le in Hin.
 destruct (Nat.ltb_spec h v), (Nat.leb_spec v l); cbn in Hent; try discriminate; lia.
Qed.
End Shape.

(* ------------------------------------------------------------------ *)
(* Leaving loops.                                                      *)

(* The run from a state leaves the blocks h to l. *)
Inductive term_leaves (F : lfacts) (prog : program) (E : env) (h l : nat) : nat -> state -> Prop :=
| TL_out : forall t s, term_in h l (blk s) = false -> term_leaves F prog E h l t s
| TL_step : forall t s s', term_in h l (blk s) = true -> phase_step F prog E t s = Some s' ->
    term_leaves F prog E h l (S t) s' -> term_leaves F prog E h l t s.

(* The run from a state leaves the blocks h to l, or reaches the back edge
   from l to h, without taking it before. *)
Inductive term_seg (F : lfacts) (prog : program) (E : env) (h l : nat) : nat -> state -> Prop :=
| TS_out : forall t s, term_in h l (blk s) = false -> term_seg F prog E h l t s
| TS_back : forall t s s', blk s = l -> phase_step F prog E t s = Some s' ->
    blk s' = h -> idx s' = 0%nat -> term_seg F prog E h l t s
| TS_step : forall t s s', term_in h l (blk s) = true -> phase_step F prog E t s = Some s' ->
    ~ (blk s = l /\ blk s' = h /\ idx s' = 0%nat) -> term_seg F prog E h l (S t) s' ->
    term_seg F prog E h l t s.

(* What a loop's check proves: a ranking value that strictly decreases
   between consecutive arrivals at the head over a back edge from inside the
   loop, through an invariant of each iteration, from its start s1. *)
Definition term_loop_proof (F : lfacts) (prog : program) (E : env) (h l : nat) : Prop :=
 exists (rank : (reg -> Z) -> nat) (inv : state -> nat -> state -> Prop),
 (forall t s, run' F prog E t = Some s -> blk s = h -> idx s = 0%nat -> inv s t s) /\
 (forall s1 t s s', inv s1 t s -> run' F prog E t = Some s -> term_in h l (blk s) = true ->
   phase_step F prog E t s = Some s' -> term_in h l (blk s') = true -> ~ (blk s' = h /\ idx s' = 0%nat) ->
   inv s1 (S t) s') /\
 (forall s1 t s s', inv s1 t s -> run' F prog E t = Some s -> term_in h l (blk s) = true ->
   phase_step F prog E t s = Some s' -> blk s' = h -> idx s' = 0%nat ->
   (rank (regs s') < rank (regs s1))%nat).

Definition term_sub (prog : program) (h l : nat) (e : nat * nat) : Prop :=
 In e (term_edges prog) /\ (h <= snd e)%nat /\ (fst e <= l)%nat /\ (h < snd e \/ fst e < l)%nat.

Definition term_wide (prog : program) : nat := S (fold_right Nat.max 0%nat (map (@List.length instr) prog)).

Definition term_measure (prog : program) (b i : nat) : nat :=
 ((List.length prog - b) * term_wide prog + (term_wide prog - i))%nat.

Lemma term_len_wide : forall prog b, (block_len prog b < term_wide prog)%nat.
Proof.
 intros prog b. unfold term_wide, block_len, block_body.
 destruct (nth_error prog b) as [body|] eqn:Hb; [|cbn; lia].
 apply Nat.lt_succ_r. revert b Hb. induction prog as [|x rest IH]; intros b Hb; [destruct b; discriminate|].
 destruct b as [|b]; cbn [nth_error] in Hb.
 - injection Hb as ->. cbn. lia.
 - cbn. specialize (IH b Hb). lia.
Qed.

Section Exit.
Variable F : lfacts.
Variable prog : program.
Variable E : env.
Hypothesis Hshape : term_shape prog = true.
Hypothesis Hidx : forall t s, run' F prog E t = Some s -> (blk s < List.length prog)%nat ->
 (idx s < block_len prog (blk s))%nat.
Local Notation reach t s := (run' F prog E t = Some s).
Local Notation edges := (term_edges prog).

Lemma term_reach_step : forall t s s', reach t s -> phase_step F prog E t s = Some s' -> reach (S t) s'.
Proof. intros t s s' H Hs. rewrite term_run_succ, H. exact Hs. Qed.

Lemma term_can_step : forall t s, reach t s -> (blk s < List.length prog)%nat ->
 exists ins s', fetch prog (blk s) (idx s) = Some ins /\ phase_step F prog E t s = Some s'.
Proof.
 intros t s H Hb. pose proof (Hidx t s H Hb) as Hi.
 destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
 - exists ins. unfold phase_step, step. rewrite Hf. eexists. split; reflexivity.
 - unfold fetch, block_len in *. apply nth_error_None in Hf. lia.
Qed.

(* Leaving a loop moves past its latch, or ends the program. *)
Lemma term_sub_exit : forall l' h' t s s' ins,
 In (l',h') edges -> fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 term_in h' l' (blk s) = true -> term_in h' l' (blk s') = false -> (l' < blk s')%nat.
Proof.
 intros l' h' t s s' ins He Hf Hs Hin Hout.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_edges_spec prog l' h' He) as [Hl' [Hhl _]].
 unfold term_in in Hin, Hout. apply andb_true_iff in Hin. destruct Hin as [H1 H2].
 apply Nat.leb_le in H1, H2.
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hb _]|[Hi0 [_ [_ [Hend|Htg]]]]].
 - rewrite Hb in Hout. rewrite (proj2 (Nat.leb_le _ _) H1), (proj2 (Nat.leb_le _ _) H2) in Hout. discriminate.
 - lia.
 - destruct (Nat.leb_spec (blk s') (blk s)) as [Hle|Hgt].
   + pose proof (term_step_back F prog E t s s' ins Hstr Hf Hs Hle Hi0) as Hback.
     destruct (term_edges_spec prog _ _ Hback) as [_ [Hle' _]].
     destruct (term_laminar prog Hshape (blk s) (blk s') l' h' (blk s) Hback He ltac:(lia) ltac:(lia))
       as [[A B]|[A B]].
     * assert (Hsl : blk s = l') by lia. rewrite Hsl in Hback.
       pose proof (term_edges_block prog _ _ _ Hback He) as Hh.
       destruct (Nat.leb_spec h' (blk s')), (Nat.leb_spec (blk s') l'); cbn in Hout; try discriminate; lia.
     * destruct (Nat.leb_spec h' (blk s')), (Nat.leb_spec (blk s') l'); cbn in Hout; try discriminate; lia.
   + destruct (Nat.leb_spec h' (blk s')), (Nat.leb_spec (blk s') l'); cbn in Hout; try discriminate; lia.
Qed.

(* A step from a loop to its head's index 0 is the back edge. *)
Lemma term_step_head : forall l h t s s' ins,
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 term_in h l (blk s) = true -> blk s' = h -> idx s' = 0%nat -> In (blk s, h) (term_edges prog).
Proof.
 intros l h t s s' ins Hf Hs Hin Hb Hi.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 unfold term_in in Hin. apply andb_true_iff in Hin. destruct Hin as [H1 H2]. apply Nat.leb_le in H1, H2.
 pose proof (term_step_back F prog E t s s' ins Hstr Hf Hs ltac:(lia) Hi) as Hback.
 rewrite Hb in Hback. exact Hback.
Qed.

Lemma term_measure_lt : forall b i b' i',
 (b <= List.length prog)%nat -> (i < term_wide prog)%nat -> (i' < term_wide prog)%nat ->
 ((b < b')%nat \/ (b = b' /\ i < i')%nat) -> (b' <= List.length prog)%nat ->
 (term_measure prog b' i' < term_measure prog b i)%nat.
Proof.
 intros b i b' i' Hb Hi Hi' Hord Hb'. unfold term_measure.
 set (W := term_wide prog) in *. set (L := List.length prog) in *.
 destruct Hord as [Hlt|[-> Hlt]].
 - assert (Hk : (L - b' + 1 <= L - b)%nat) by lia.
   assert (Hm : ((L - b' + 1) * W <= (L - b) * W)%nat) by (apply Nat.mul_le_mono_r; exact Hk).
   rewrite Nat.mul_add_distr_r in Hm. lia.
 - lia.
Qed.

(* Inside a region [h, l] whose back edges are the region's own or those
   of loops strictly inside it, which every run leaves, the run leaves the
   region or reaches its back edge. *)
Lemma term_seg_all : forall h l,
 (h <= l)%nat -> (l < List.length prog)%nat ->
 (forall b tg, In (b,tg) edges -> (h <= b <= l)%nat -> term_sub prog h l (b,tg) \/ (b = l /\ tg = h)) ->
 (forall l' h', term_sub prog h l (l',h') ->
   forall t s, reach t s -> term_in h' l' (blk s) = true -> term_leaves F prog E h' l' t s) ->
 forall t s, reach t s -> term_in h l (blk s) = true -> term_seg F prog E h l t s.
Proof.
 intros h l Hhl Hl Hback Hsub.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 assert (Hwf : forall n t s, term_measure prog (blk s) (idx s) = n -> reach t s ->
   term_in h l (blk s) = true -> term_seg F prog E h l t s).
 2:{ intros t s H Hin. exact (Hwf _ t s eq_refl H Hin). }
 intro n. induction n as [n IH] using lt_wf_ind. intros t s Hn Hreach Hin.
 assert (HinR : (h <= blk s <= l)%nat).
 { unfold term_in in Hin. apply andb_true_iff in Hin. destruct Hin as [A B]. apply Nat.leb_le in A, B. lia. }
 assert (Hbs : (blk s < List.length prog)%nat) by lia.
 pose proof (Hidx t s Hreach Hbs) as Hidxs.
 pose proof (term_len_wide prog (blk s)) as Hw.
 (* a state after s at a later position, inside the region *)
 assert (Hlater : forall t' s', reach t' s' -> term_in h l (blk s') = true ->
   ((blk s < blk s')%nat \/ (blk s = blk s' /\ idx s < idx s')%nat) -> term_seg F prog E h l t' s').
 { intros t' s' Hr' Hin' Hord. unfold term_in in Hin'. apply andb_true_iff in Hin'.
   destruct Hin' as [A B]. apply Nat.leb_le in A, B.
   pose proof (Hidx t' s' Hr' ltac:(lia)) as Hi'. pose proof (term_len_wide prog (blk s')).
   eapply IH; [|reflexivity|exact Hr'|unfold term_in; rewrite (proj2 (Nat.leb_le _ _) A), (proj2 (Nat.leb_le _ _) B); reflexivity].
   rewrite <- Hn. apply term_measure_lt; lia. }
 destruct (find (fun e => Nat.leb h (snd e) && Nat.leb (fst e) l && (Nat.ltb h (snd e) || Nat.ltb (fst e) l)
   && term_in (snd e) (fst e) (blk s)) edges) as [[l' h']|] eqn:Hfind.
 - (* s is inside a loop strictly inside the region *)
   apply find_some in Hfind. destruct Hfind as [He Hc]. cbn [fst snd] in Hc.
   rewrite !andb_true_iff, orb_true_iff, !Nat.leb_le, !Nat.ltb_lt in Hc.
   destruct Hc as [[[A B] C] Hin'].
   assert (Hs : term_sub prog h l (l',h')) by (repeat split; cbn; tauto).
   pose proof (Hsub l' h' Hs t s Hreach Hin') as Hleave.
   destruct (term_edges_spec prog l' h' He) as [Hl' [Hhl' _]].
   (* walk the run inside the inner loop *)
   assert (Hwalk : forall t0 x, term_leaves F prog E h' l' t0 x -> reach t0 x ->
     term_in h' l' (blk x) = true -> term_seg F prog E h l t0 x).
   { intros t0 x Hx. induction Hx as [t0 x Hout|t0 x x' Hin0 Hstep Hrest IHx]; intros Hr Hinx.
     - rewrite Hinx in Hout. discriminate.
     - assert (HinxR : term_in h l (blk x) = true).
       { unfold term_in in Hinx |- *. apply andb_true_iff in Hinx. destruct Hinx as [P Q].
         apply Nat.leb_le in P, Q. apply andb_true_iff. split; apply Nat.leb_le; lia. }
       pose proof (term_reach_step t0 x x' Hr Hstep) as Hr'.
       assert (Hnotback : ~ (blk x = l /\ blk x' = h /\ idx x' = 0%nat)).
       { intros [Hxl [Hxh Hxi]]. pose proof Hinx as Hinx'. unfold term_in in Hinx'.
         apply andb_true_iff in Hinx'. destruct Hinx' as [P Q].
         apply Nat.leb_le in P, Q. destruct Hs as [_ [S1 [S2 S3]]]. cbn in S1, S2, S3.
         destruct (term_can_step t0 x Hr ltac:(lia)) as [ins [x'' [Hf Hs'']]].
         rewrite Hstep in Hs''. injection Hs'' as <-.
         pose proof (term_step_back F prog E t0 x x' ins Hstr Hf Hstep ltac:(lia) Hxi) as Hb.
         assert (Hlx : blk x = l') by lia. rewrite Hlx in Hb.
         pose proof (term_edges_block prog _ _ _ Hb He) as Hh. lia. }
       apply (TS_step F prog E h l t0 x x' HinxR Hstep Hnotback).
       destruct (term_in h' l' (blk x')) eqn:Hin'x.
       + exact (IHx Hr' eq_refl).
       + destruct (term_in h l (blk x')) eqn:HinR'.
         * destruct (term_can_step t0 x Hr ltac:(unfold term_in in Hinx; apply andb_true_iff in Hinx; destruct Hinx as [P Q]; apply Nat.leb_le in Q; lia))
             as [ins [x'' [Hf Hs'']]].
           rewrite Hstep in Hs''. injection Hs'' as <-.
           pose proof (term_sub_exit l' h' t0 x x' ins He Hf Hstep Hinx Hin'x) as Hpast.
           apply Hlater; [exact Hr'|exact HinR'|].
           left. unfold term_in in Hin'. apply andb_true_iff in Hin'. destruct Hin' as [P Q].
           apply Nat.leb_le in Q. lia.
         * apply TS_out. exact HinR'. }
   exact (Hwalk t s Hleave Hreach Hin').
 - (* s is outside every inner loop: one step *)
   destruct (term_can_step t s Hreach Hbs) as [ins [s' [Hf Hs]]].
   destruct (Nat.eq_dec (blk s) l) as [Hbl|Hbl];
   [destruct (Nat.eq_dec (blk s') h) as [Hbh|Hbh];
    [destruct (Nat.eq_dec (idx s') 0) as [Hi0|Hi0]|]|].
   + exact (TS_back F prog E h l t s s' Hbl Hs Hbh Hi0).
   + all: idtac.
     assert (Hnb : ~ (blk s = l /\ blk s' = h /\ idx s' = 0%nat)) by tauto.
     apply (TS_step F prog E h l t s s' Hin Hs Hnb).
     destruct (term_in h l (blk s')) eqn:Hin'; [|apply TS_out; exact Hin'].
     apply Hlater; [exact (term_reach_step t s s' Hreach Hs)|exact Hin'|].
     destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[B [I _]]|[I _]]; [right; lia|lia].
   + assert (Hnb : ~ (blk s = l /\ blk s' = h /\ idx s' = 0%nat)) by tauto.
     apply (TS_step F prog E h l t s s' Hin Hs Hnb).
     destruct (term_in h l (blk s')) eqn:Hin'; [|apply TS_out; exact Hin'].
     apply Hlater; [exact (term_reach_step t s s' Hreach Hs)|exact Hin'|].
     destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[B [I _]]|[I [_ [_ Htg]]]]; [right; lia|].
     destruct (Nat.leb_spec (blk s') (blk s)) as [Hle|Hgt]; [|left; exact Hgt].
     pose proof (term_step_back F prog E t s s' ins Hstr Hf Hs Hle I) as Hb.
     exfalso. destruct (Hback _ _ Hb HinR) as [Hsb|[_ Htg']]; [|exact (Hbh Htg')].
     destruct (term_edges_spec prog _ _ Hb) as [_ [Hle' _]].
     assert (Hfound : find (fun e => Nat.leb h (snd e) && Nat.leb (fst e) l && (Nat.ltb h (snd e) || Nat.ltb (fst e) l)
       && term_in (snd e) (fst e) (blk s)) edges <> None).
     { intros Hnone. eapply find_none in Hnone; [|exact Hb]. cbn [fst snd] in Hnone.
       destruct Hsb as [_ [S1 [S2 S3]]]. cbn [fst snd] in S1, S2, S3.
       unfold term_in in Hnone. rewrite (proj2 (Nat.leb_le _ _) S1), (proj2 (Nat.leb_le _ _) S2),
         (proj2 (Nat.leb_le _ _) Hle), (proj2 (Nat.leb_le (blk s) (blk s)) (le_n _)) in Hnone.
       destruct S3 as [S3|S3]; [rewrite (proj2 (Nat.ltb_lt _ _) S3) in Hnone|rewrite (proj2 (Nat.ltb_lt _ _) S3), orb_true_r in Hnone];
       discriminate. }
     rewrite Hfind in Hfound. exact (Hfound eq_refl).
   + assert (Hnb : ~ (blk s = l /\ blk s' = h /\ idx s' = 0%nat)) by tauto.
     apply (TS_step F prog E h l t s s' Hin Hs Hnb).
     destruct (term_in h l (blk s')) eqn:Hin'; [|apply TS_out; exact Hin'].
     apply Hlater; [exact (term_reach_step t s s' Hreach Hs)|exact Hin'|].
     destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[B [I _]]|[I [_ [_ Htg]]]]; [right; lia|].
     destruct (Nat.leb_spec (blk s') (blk s)) as [Hle|Hgt]; [|left; exact Hgt].
     pose proof (term_step_back F prog E t s s' ins Hstr Hf Hs Hle I) as Hb.
     exfalso. destruct (Hback _ _ Hb HinR) as [Hsb|[Hbl' _]]; [|exact (Hbl Hbl')].
     destruct (term_edges_spec prog _ _ Hb) as [_ [Hle' _]].
     assert (Hfound : find (fun e => Nat.leb h (snd e) && Nat.leb (fst e) l && (Nat.ltb h (snd e) || Nat.ltb (fst e) l)
       && term_in (snd e) (fst e) (blk s)) edges <> None).
     { intros Hnone. eapply find_none in Hnone; [|exact Hb]. cbn [fst snd] in Hnone.
       destruct Hsb as [_ [S1 [S2 S3]]]. cbn [fst snd] in S1, S2, S3.
       unfold term_in in Hnone. rewrite (proj2 (Nat.leb_le _ _) S1), (proj2 (Nat.leb_le _ _) S2),
         (proj2 (Nat.leb_le _ _) Hle), (proj2 (Nat.leb_le (blk s) (blk s)) (le_n _)) in Hnone.
       destruct S3 as [S3|S3]; [rewrite (proj2 (Nat.ltb_lt _ _) S3) in Hnone|rewrite (proj2 (Nat.ltb_lt _ _) S3), orb_true_r in Hnone];
       discriminate. }
     rewrite Hfind in Hfound. exact (Hfound eq_refl).
Qed.
(* A loop whose check holds is left by every run that enters it. *)
Lemma term_loop_exit : forall l h, In (l,h) edges -> term_loop_proof F prog E h l ->
 (forall t s, reach t s -> term_in h l (blk s) = true -> term_seg F prog E h l t s) ->
 forall t s, reach t s -> term_in h l (blk s) = true -> term_leaves F prog E h l t s.
Proof.
 intros l h He [rank [inv [Hstart [Hstep Hback]]]] Hseg.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 assert (Hinh : term_in h l h = true).
 { unfold term_in. rewrite (proj2 (Nat.leb_le h h) (le_n _)), (proj2 (Nat.leb_le h l) Hhl). reflexivity. }
 assert (Hiter : forall r t1 s1, rank (regs s1) = r -> reach t1 s1 -> blk s1 = h -> idx s1 = 0%nat ->
   term_leaves F prog E h l t1 s1).
 { intro r. induction r as [r IH] using lt_wf_ind. intros t1 s1 Hr Hreach1 Hb1 Hi1.
   assert (Hgo : forall t x, term_seg F prog E h l t x -> reach t x -> inv s1 t x ->
     term_leaves F prog E h l t x).
   { intros t x Hx. induction Hx as [t x Hout|t x x' Hxl Hs Hxh Hxi|t x x' Hin Hs Hnb Hrest IHx];
       intros Hrx Hinv.
     - apply TL_out. exact Hout.
     - assert (Hinx : term_in h l (blk x) = true) by (rewrite Hxl; unfold term_in;
         rewrite (proj2 (Nat.leb_le h l) Hhl), (proj2 (Nat.leb_le l l) (le_n _)); reflexivity).
       apply (TL_step F prog E h l t x x' Hinx Hs).
       pose proof (Hback s1 t x x' Hinv Hrx Hinx Hs Hxh Hxi) as Hlt. rewrite Hr in Hlt.
       exact (IH _ Hlt (S t) x' eq_refl (term_reach_step t x x' Hrx Hs) Hxh Hxi).
     - apply (TL_step F prog E h l t x x' Hin Hs).
       destruct (term_in h l (blk x')) eqn:Hin'; [|apply TL_out; exact Hin'].
       destruct (Nat.eq_dec (blk x') h) as [Hb'|Hb']; [destruct (Nat.eq_dec (idx x') 0) as [Hi'|Hi']|].
       + (* another back edge to the head starts a new iteration *)
         pose proof (Hback s1 t x x' Hinv Hrx Hin Hs Hb' Hi') as Hlt. rewrite Hr in Hlt.
         exact (IH _ Hlt (S t) x' eq_refl (term_reach_step t x x' Hrx Hs) Hb' Hi').
       + apply IHx; [exact (term_reach_step t x x' Hrx Hs)|].
         exact (Hstep s1 t x x' Hinv Hrx Hin Hs Hin' ltac:(tauto)).
       + apply IHx; [exact (term_reach_step t x x' Hrx Hs)|].
         exact (Hstep s1 t x x' Hinv Hrx Hin Hs Hin' ltac:(tauto)). }
   apply Hgo; [apply Hseg; [exact Hreach1|rewrite Hb1; exact Hinh]|exact Hreach1|].
   exact (Hstart t1 s1 Hreach1 Hb1 Hi1). }
 intros t s Hr Hin. specialize (Hseg t s Hr Hin). clear Hin.
 induction Hseg as [t x Hout|t x x' Hxl Hs Hxh Hxi|t x x' Hin Hs Hnb Hrest IHx].
 - apply TL_out. exact Hout.
 - assert (Hinx : term_in h l (blk x) = true) by (rewrite Hxl; unfold term_in;
     rewrite (proj2 (Nat.leb_le h l) Hhl), (proj2 (Nat.leb_le l l) (le_n _)); reflexivity).
   apply (TL_step F prog E h l t x x' Hinx Hs).
   exact (Hiter _ (S t) x' eq_refl (term_reach_step t x x' Hr Hs) Hxh Hxi).
 - apply (TL_step F prog E h l t x x' Hin Hs). exact (IHx (term_reach_step t x x' Hr Hs)).
Qed.

(* Every loop whose check holds, from the innermost out, is left. *)
Theorem term_loops_exit : (forall l h, In (l,h) edges -> term_loop_proof F prog E h l) ->
 forall n l h, In (l,h) edges -> (l - h <= n)%nat ->
 forall t s, reach t s -> term_in h l (blk s) = true -> term_leaves F prog E h l t s.
Proof.
 intros Hproofs n. induction n as [n IH] using lt_wf_ind. intros l h He Hn.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 apply term_loop_exit; [exact He|exact (Hproofs l h He)|].
 apply term_seg_all; [exact Hhl|exact Hl| |].
 - intros b tg Hb Hbin. destruct (term_edges_spec prog b tg Hb) as [_ [Htb _]].
   destruct (term_laminar prog Hshape b tg l h b Hb He ltac:(lia) ltac:(lia)) as [[A B]|[A B]].
   + right. assert (b = l) by lia. subst b. split; [reflexivity|].
     exact (term_edges_block prog _ _ _ Hb He).
   + destruct (Nat.eq_dec tg h) as [->|Hne]; [destruct (Nat.eq_dec b l) as [->|Hbl]|].
     * right. split; reflexivity.
     * left. repeat split; cbn; [exact Hb|lia|lia|right; lia].
     * left. repeat split; cbn; [exact Hb|lia|lia|left; lia].
 - intros l' h' [Hs [S1 [S2 S3]]]. cbn in S1, S2, S3.
   destruct (term_edges_spec prog l' h' Hs) as [_ [Hhl' _]].
   apply (IH (l' - h')%nat ltac:(lia) l' h' Hs (le_n _)).
Qed.

(* The run of every thread ends. *)
Theorem term_runs_end : (forall l h, In (l,h) edges -> term_loop_proof F prog E h l) ->
 exists t s, reach t s /\ fetch prog (blk s) (idx s) = None.
Proof.
 intros Hproofs.
 set (s0 := {| blk := 0%nat; idx := 0%nat; regs := e_init E |}).
 assert (H0 : reach 0%nat s0) by reflexivity.
 destruct (Nat.eq_dec (List.length prog) 0) as [Hlen|Hlen].
 { exists 0%nat,s0. split; [exact H0|]. unfold fetch, block_body. cbn [blk idx].
   destruct prog; [reflexivity|discriminate]. }
 pose proof (term_shape_structured prog Hshape) as Hstr.
 assert (Hseg : term_seg F prog E 0 (List.length prog - 1) 0 s0).
 { apply term_seg_all; [lia|lia| | |exact H0|reflexivity].
   - intros b tg Hb Hbin. left. pose proof (term_shape_edge prog Hshape b tg Hb) as Hpos.
     destruct (term_edges_spec prog b tg Hb) as [_ [Htb _]].
     repeat split; cbn; [exact Hb|lia|lia|left; lia].
   - intros l' h' [Hs _]. destruct (term_edges_spec prog l' h' Hs) as [_ [Hhl' _]].
     exact (term_loops_exit Hproofs (l' - h') l' h' Hs (le_n _)). }
 assert (Hend : forall t s, term_seg F prog E 0 (List.length prog - 1) t s -> reach t s ->
   exists t' s', reach t' s' /\ fetch prog (blk s') (idx s') = None).
 { intros t s Hx. induction Hx as [t x Hout|t x x' Hxl Hs Hxh Hxi|t x x' Hin Hs Hnb Hrest IHx]; intros Hrx.
   - exists t,x. split; [exact Hrx|]. unfold term_in in Hout.
     destruct (Nat.leb_spec (blk x) (List.length prog - 1)); [cbn in Hout; discriminate|].
     unfold fetch, block_body. rewrite (proj2 (nth_error_None prog (blk x)) ltac:(lia)).
     destruct (idx x); reflexivity.
   - exfalso. destruct (term_can_step t x Hrx ltac:(lia)) as [ins [x'' [Hf Hs'']]].
     rewrite Hs in Hs''. injection Hs'' as <-.
     pose proof (term_step_back F prog E t x x' ins Hstr Hf Hs ltac:(lia) Hxi) as Hb.
     rewrite Hxh in Hb. pose proof (term_shape_edge prog Hshape _ _ Hb) as Hpos. lia.
   - exact (IHx (term_reach_step t x x' Hrx Hs)). }
 exact (Hend 0%nat s0 Hseg H0).
Qed.

End Exit.

(* ------------------------------------------------------------------ *)
(* Lists of instructions.                                              *)

Lemma term_indexed_spec : forall prog b k x,
 In (k,x) (term_indexed prog b) <-> fetch prog b k = Some x.
Proof.
 intros prog b k x. unfold term_indexed, fetch, block_len.
 generalize (block_body prog b) as body. intros body.
 assert (Hgen : forall n, In (k,x) (combine (seq n (length body)) body) <-> (n <= k)%nat /\ nth_error body (k - n) = Some x).
 { induction body as [|y body IH]; intros n; cbn [length seq combine].
   - split; [intros []|]. intros [Hn H]. destruct (k - n)%nat; discriminate.
   - cbn [In]. rewrite IH. split.
     + intros [H|[Hn H]].
       * injection H as <- <-. split; [lia|]. rewrite Nat.sub_diag. reflexivity.
       * split; [lia|]. replace (k - n)%nat with (S (k - S n)) by lia. exact H.
     + intros [Hn H]. destruct (Nat.eq_dec k n) as [->|Hne].
       * left. rewrite Nat.sub_diag in H. injection H as ->. reflexivity.
       * right. split; [lia|]. replace (k - n)%nat with (S (k - S n)) in H by lia. exact H. }
 rewrite Hgen. rewrite Nat.sub_0_r. split; [tauto|intros H; split; [lia|exact H]].
Qed.

Lemma term_fetch_len : forall prog b i x, fetch prog b i = Some x -> (i < block_len prog b)%nat.
Proof. intros prog b i x H. unfold fetch, block_len in *. apply nth_error_Some. rewrite H. discriminate. Qed.

Lemma term_region_writes_spec : forall prog h l r b i ins,
 In (b,i,ins) (term_region_writes prog h l r) <->
 (h <= b <= l)%nat /\ fetch prog b i = Some ins /\ term_writes r ins = true.
Proof.
 intros prog h l r b i ins. unfold term_region_writes. rewrite in_flat_map. split.
 - intros [b' [Hb' Hin]]. apply in_seq in Hb'. apply in_map_iff in Hin.
   destruct Hin as [[k x] [Heq Hin]]. cbn [fst snd] in Heq. injection Heq as <- <- <-.
   apply filter_In in Hin. destruct Hin as [Hin Hw]. apply term_indexed_spec in Hin.
   split; [lia|]. split; assumption.
 - intros [Hb [Hf Hw]]. exists b. split; [apply in_seq; lia|]. apply in_map_iff.
   exists (i,ins). split; [reflexivity|]. apply filter_In. split; [apply term_indexed_spec; exact Hf|exact Hw].
Qed.

Lemma term_last_opt_spec : forall A (xs : list A) x, term_last_opt xs = Some x ->
 exists pre, xs = pre ++ [x].
Proof.
 intros A xs x H. unfold term_last_opt in H. destruct (rev xs) as [|y ys] eqn:Hr; [discriminate|].
 injection H as ->. exists (rev ys). rewrite <- (rev_involutive xs), Hr. reflexivity.
Qed.

Lemma term_last_opt_none : forall A (xs : list A), term_last_opt xs = None -> xs = [].
Proof.
 intros A xs H. unfold term_last_opt in H. destruct (rev xs) eqn:Hr; [|discriminate].
 rewrite <- (rev_involutive xs), Hr. reflexivity.
Qed.

(* The indexed list increases strictly in its indices. *)
Lemma term_indexed_sorted : forall prog b pre x y post,
 term_indexed prog b = pre ++ x :: post -> In y post -> (fst x < fst y)%nat.
Proof.
 intros prog b. unfold term_indexed, block_len. generalize (block_body prog b) as body. intros body.
 generalize 0%nat as n. revert body.
 induction body as [|z body IH]; intros n pre x y post H Hy; cbn [length seq combine] in H.
 - destruct pre; discriminate.
 - destruct pre as [|w pre]; cbn [app] in H; injection H as Hx Hrest.
   + subst x. cbn [fst].
     assert (Hgen : forall m k v, In (k,v) (combine (seq m (length body)) body) -> (m <= k)%nat).
     { clear. induction body as [|u body IH]; intros m k v Hin; cbn in Hin; [contradiction|].
       destruct Hin as [Hin|Hin]; [injection Hin as <- <-; lia|]. specialize (IH (S m) k v Hin). lia. }
     destruct y as [k v]. rewrite <- Hrest in Hy. cbn [fst]. specialize (Hgen _ _ _ Hy). lia.
   + exact (IH (S n) pre x y post Hrest Hy).
Qed.

Lemma term_last_write_some : forall prog b i r k ins,
 term_last_write prog b i r = Some (k,ins) ->
 (k < i)%nat /\ fetch prog b k = Some ins /\ term_writes r ins = true /\
 forall k' ins', (k < k' < i)%nat -> fetch prog b k' = Some ins' -> term_writes r ins' = false.
Proof.
 intros prog b i r k ins H. unfold term_last_write in H. apply term_last_opt_spec in H.
 destruct H as [pre Hpre].
 assert (Hin : In (k,ins) (filter (fun x => Nat.ltb (fst x) i && term_writes r (snd x)) (term_indexed prog b)))
   by (rewrite Hpre; apply in_or_app; right; left; reflexivity).
 apply filter_In in Hin. destruct Hin as [Hin Hc]. cbn [fst snd] in Hc. apply andb_true_iff in Hc.
 destruct Hc as [Hk Hw]. apply Nat.ltb_lt in Hk. apply term_indexed_spec in Hin.
 split; [exact Hk|]. split; [exact Hin|]. split; [exact Hw|].
 intros k' ins' Hk' Hf'. destruct (term_writes r ins') eqn:Hw'; [|reflexivity]. exfalso.
 assert (Hin' : In (k',ins') (filter (fun x => Nat.ltb (fst x) i && term_writes r (snd x)) (term_indexed prog b))).
 { apply filter_In. split; [apply term_indexed_spec; exact Hf'|]. cbn. rewrite Hw', andb_true_r.
   apply Nat.ltb_lt. lia. }
 rewrite Hpre in Hin'. apply in_app_or in Hin'. destruct Hin' as [Hin'|[Heq|[]]].
 - (* earlier in the filtered list, so before k in the indexed list *)
   assert (Hsub : forall l1 l2 (f : nat * instr -> bool) x, filter f l1 = l2 ++ [x] -> forall y, In y l2 ->
     exists p q, l1 = p ++ y :: q /\ In x q).
   { clear. intros l1. induction l1 as [|u l1 IH]; intros l2 f x Hf y Hy; cbn in Hf.
     - destruct l2; discriminate.
     - destruct (f u) eqn:Hu.
       + destruct l2 as [|v l2]; cbn in Hf; injection Hf as Hux Hrest.
         * contradiction.
         * destruct Hy as [<-|Hy].
           -- subst v. exists [],l1. split; [reflexivity|].
              assert (Hx : In x (filter f l1)) by (rewrite Hrest; apply in_or_app; right; left; reflexivity).
              apply filter_In in Hx. exact (proj1 Hx).
           -- destruct (IH l2 f x Hrest y Hy) as [p [q [Hp Hq]]]. exists (u::p),q. rewrite Hp. split; [reflexivity|exact Hq].
       + destruct (IH l2 f x Hf y Hy) as [p [q [Hp Hq]]]. exists (u::p),q. rewrite Hp. split; [reflexivity|exact Hq]. }
   destruct (Hsub _ _ _ _ Hpre _ Hin') as [p [q [Hp Hq]]].
   pose proof (term_indexed_sorted prog b p (k',ins') (k,ins) q Hp Hq) as Hlt. cbn in Hlt. lia.
 - injection Heq as -> ->. lia.
Qed.

Lemma term_last_write_none : forall prog b i r k ins,
 term_last_write prog b i r = None -> (k < i)%nat -> fetch prog b k = Some ins -> term_writes r ins = false.
Proof.
 intros prog b i r k ins H Hk Hf. unfold term_last_write in H. apply term_last_opt_none in H.
 destruct (term_writes r ins) eqn:Hw; [|reflexivity]. exfalso.
 assert (Hin : In (k,ins) (filter (fun x => Nat.ltb (fst x) i && term_writes r (snd x)) (term_indexed prog b))).
 { apply filter_In. split; [apply term_indexed_spec; exact Hf|]. cbn. rewrite Hw, andb_true_r. apply Nat.ltb_lt. exact Hk. }
 rewrite H in Hin. exact Hin.
Qed.

Lemma term_kept_spec : forall prog b x y r k ins,
 term_kept prog b x y r = true -> (x <= k < y)%nat -> fetch prog b k = Some ins -> term_writes r ins = false.
Proof.
 intros prog b x y r k ins H Hk Hf. unfold term_kept in H. rewrite forallb_forall in H.
 specialize (H (k,ins) (proj2 (term_indexed_spec prog b k ins) Hf)). cbn [fst snd] in H.
 rewrite (proj2 (Nat.leb_le _ _) (proj1 Hk)), (proj2 (Nat.ltb_lt _ _) (proj2 Hk)) in H.
 cbn in H. apply negb_true_iff in H. exact H.
Qed.

Lemma term_walk_app : forall F st l1 l2, walk F st (l1 ++ l2) = walk F (walk F st l1) l2.
Proof. intros F st l1. revert st. induction l1 as [|x l1 IH]; intros st l2; [reflexivity|]. cbn. apply IH. Qed.

Lemma term_nth_skipn : forall A (l : list A) x n, nth_error (skipn x l) n = nth_error l (x + n).
Proof. intros A l x. revert l. induction x as [|x IH]; intros l n; [reflexivity|]. destruct l; [destruct n; reflexivity|]. apply IH. Qed.

Lemma term_slice_snoc : forall prog b x y ins, (x <= y)%nat -> fetch prog b y = Some ins ->
 term_slice prog b x (S y) = term_slice prog b x y ++ [ins].
Proof.
 intros prog b x y ins Hxy Hf. unfold term_slice, fetch in *.
 replace (S y - x)%nat with (S (y - x)) by lia.
 assert (Hn : nth_error (skipn x (block_body prog b)) (y - x) = Some ins).
 { rewrite term_nth_skipn. replace (x + (y - x))%nat with y by lia. exact Hf. }
 revert Hn. generalize (skipn x (block_body prog b)) as l. generalize (y - x)%nat as n. clear.
 intros n. induction n as [|n IH]; intros l Hn; destruct l as [|z l]; try discriminate.
 - cbn in Hn. injection Hn as ->. reflexivity.
 - cbn [nth_error] in Hn. change (z :: firstn (S n) l = (z :: firstn n l) ++ [ins]).
   rewrite (IH l Hn). reflexivity.
Qed.

Lemma term_slice_all : forall prog b x, term_slice prog b x (block_len prog b) = skipn x (block_body prog b).
Proof.
 intros prog b x. unfold term_slice, block_len. rewrite firstn_all2; [reflexivity|].
 rewrite skipn_length. lia.
Qed.

Lemma term_slice_empty : forall prog b x, term_slice prog b x x = [].
Proof. intros prog b x. unfold term_slice. rewrite Nat.sub_diag. reflexivity. Qed.

(* ------------------------------------------------------------------ *)
(* Ranks of counters.                                                  *)

Definition term_rank (k : term_kind) (w : nat) (v : Z) : nat :=
 match k with
 | KUp => Z.to_nat (2 ^ (Z.of_nat w - 1) - sext w v)
 | KDown => Z.to_nat (sext w v + 2 ^ (Z.of_nat w - 1))
 | KHalve => Z.to_nat v
 | KDouble => Z.to_nat (2 ^ Z.of_nat w - v)
 end.

Lemma term_sext_bounds : forall n v, (1 <= n)%nat -> - 2 ^ (Z.of_nat n - 1) <= sext n v < 2 ^ (Z.of_nat n - 1).
Proof.
 intros n v Hn. unfold sext. pose proof (trunc_range n v) as Ht.
 assert (Hp : 2 ^ Z.of_nat n = 2 * 2 ^ (Z.of_nat n - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
 destruct (Z.ltb_spec (trunc n v) (2 ^ (Z.of_nat n - 1))); lia.
Qed.

(* The sum of two values read as signed integers of width w, inside the
   signed range, is the signed reading of their wrapped sum. *)
Lemma term_sext_sum : forall w a b z,
 (1 <= w)%nat -> - 2 ^ (Z.of_nat w - 1) <= z < 2 ^ (Z.of_nat w - 1) ->
 eqm (Z.of_nat w) (a + b) z -> sext w (trunc w (a + b)) = z.
Proof.
 intros w a b z Hw Hz H.
 apply (eqm_sext_eq (Z.of_nat w) w (trunc w (a + b)) z Hw (Z.le_refl _) (trunc_range w (a + b)) Hz).
 eapply eqm_trans; [apply trunc_eqm|exact H].
Qed.

Lemma term_step_rank : forall F E t st rs k op w d a o x y cv,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs ->
 (1 <= w <= 64)%nat -> (forall r, o = OReg r -> wd F r = Z.of_nat w) ->
 term_step_ok F st k (IBin None op w d a o) x y = true ->
 0 <= cv < 2 ^ Z.of_nat w -> x <= extend w (term_signed k) cv <= y ->
 (term_rank k w (trunc w (binop_eval op w cv (opv rs o))) < term_rank k w cv)%nat.
Proof.
 intros F E t st rs k op w d a o x y cv HE Hrs Hst Hw Ho Hok Hcv Hxy.
 assert (Hrange : forall sg sl sh, cmp_range F st o (Z.of_nat w) sg = Some (sl,sh) ->
   sl <= extend w sg (opv rs o) <= sh).
 { intros sg sl sh Hc. pose proof (cmp_range_sound F E HE t rs Hrs st o (Z.of_nat w) sg sl sh Hst
     ltac:(lia) Ho Hc) as Hb. unfold cval in Hb. rewrite Nat2Z.id in Hb. exact Hb. }
 assert (Hpw : 2 ^ Z.of_nat w = 2 * 2 ^ (Z.of_nat w - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
 assert (Hp1 : 0 < 2 ^ (Z.of_nat w - 1)) by (apply Z.pow_pos_nonneg; lia).
 pose proof (term_sext_bounds w cv ltac:(lia)) as Hsc.
 destruct k, op; cbn [term_step_ok] in Hok; try discriminate.
 - (* up by an addition *)
   destruct (cmp_range F st o (Z.of_nat w) true) as [[sl sh]|] eqn:Hc; [|discriminate].
   rewrite andb_true_iff, Z.leb_le, Z.ltb_lt in Hok. destruct Hok as [Hsl Hsh].
   pose proof (Hrange true sl sh Hc) as Hov. cbn [term_signed extend] in Hov, Hxy.
   pose proof (term_sext_bounds w (opv rs o) ltac:(lia)) as Hso.
   cbn [binop_eval term_rank].
   rewrite (term_sext_sum w cv (opv rs o) (sext w cv + sext w (opv rs o)) ltac:(lia) ltac:(lia)).
   + apply Z2Nat.inj_lt; lia.
   + apply eqm_add; [lia|apply eqm_sym, sext_eqm|apply eqm_sym, sext_eqm].
 - (* down by an addition *)
   destruct (cmp_range F st o (Z.of_nat w) true) as [[sl sh]|] eqn:Hc; [|discriminate].
   rewrite andb_true_iff, !Z.leb_le in Hok. destruct Hok as [Hsh Hsl].
   pose proof (Hrange true sl sh Hc) as Hov. cbn [term_signed extend] in Hov, Hxy.
   cbn [binop_eval term_rank].
   rewrite (term_sext_sum w cv (opv rs o) (sext w cv + sext w (opv rs o)) ltac:(lia) ltac:(lia)).
   + apply Z2Nat.inj_lt; lia.
   + apply eqm_add; [lia|apply eqm_sym, sext_eqm|apply eqm_sym, sext_eqm].
 - (* down by a subtraction *)
   destruct (cmp_range F st o (Z.of_nat w) true) as [[sl sh]|] eqn:Hc; [|discriminate].
   rewrite andb_true_iff, !Z.leb_le in Hok. destruct Hok as [Hsl Hsh].
   pose proof (Hrange true sl sh Hc) as Hov. cbn [term_signed extend] in Hov, Hxy.
   cbn [binop_eval term_rank].
   replace (cv - opv rs o) with (cv + - opv rs o) by ring.
   rewrite (term_sext_sum w cv (- opv rs o) (sext w cv - sext w (opv rs o)) ltac:(lia) ltac:(lia)).
   + apply Z2Nat.inj_lt; lia.
   + replace (sext w cv - sext w (opv rs o)) with (sext w cv + - sext w (opv rs o)) by ring.
     apply eqm_add; [lia|apply eqm_sym, sext_eqm|apply eqm_opp; [lia|apply eqm_sym, sext_eqm]].
 - (* halving by a logical shift *)
   destruct o as [r|kk]; [discriminate|].
   rewrite !andb_true_iff, !Z.leb_le, !Z.ltb_lt in Hok. destruct Hok as [[Hk1 Hk2] Hx1].
   cbn [term_signed extend] in Hxy. rewrite trunc_small in Hxy by lia.
   cbn [binop_eval opv term_rank]. rewrite (trunc_small w cv) by lia.
   assert (Hs : 1 <= shamt w kk) by (unfold shamt; rewrite trunc_small by (cbn; lia); lia).
   assert (Hq : cv / 2 ^ shamt w kk < cv).
   { apply Z.div_lt; [lia|]. apply (Z.lt_le_trans _ (2 ^ 1)); [cbn; lia|apply Z.pow_le_mono_r; lia]. }
   assert (Hq0 : 0 <= cv / 2 ^ shamt w kk) by (apply Z.div_pos; [lia|apply Z.pow_pos_nonneg; lia]).
   rewrite trunc_small by lia. apply Z2Nat.inj_lt; lia.
 - (* halving by an arithmetic shift of a nonnegative value *)
   destruct o as [r|kk]; [discriminate|].
   rewrite !andb_true_iff, !Z.leb_le, !Z.ltb_lt in Hok. destruct Hok as [[[Hk1 Hk2] Hx1] Hy].
   cbn [term_signed extend] in Hxy. rewrite trunc_small in Hxy by lia.
   cbn [binop_eval opv term_rank]. rewrite (sext_small w cv) by lia.
   assert (Hs : 1 <= shamt w kk) by (unfold shamt; rewrite trunc_small by (cbn; lia); lia).
   assert (Hq : cv / 2 ^ shamt w kk < cv).
   { apply Z.div_lt; [lia|]. apply (Z.lt_le_trans _ (2 ^ 1)); [cbn; lia|apply Z.pow_le_mono_r; lia]. }
   assert (Hq0 : 0 <= cv / 2 ^ shamt w kk) by (apply Z.div_pos; [lia|apply Z.pow_pos_nonneg; lia]).
   rewrite trunc_small by lia. apply Z2Nat.inj_lt; lia.
 - (* doubling *)
   destruct o as [r|kk]; [discriminate|].
   rewrite !andb_true_iff, !Z.leb_le, !Z.ltb_lt in Hok. destruct Hok as [[[Hk1 Hk2] Hx1] Hy].
   cbn [term_signed extend] in Hxy. rewrite trunc_small in Hxy by lia.
   cbn [binop_eval opv term_rank].
   assert (Hkw : kk < Z.of_nat w).
   { destruct (Z.lt_ge_cases kk (Z.of_nat w)) as [H|H]; [exact H|].
     assert (2 ^ Z.of_nat w <= 2 ^ kk) by (apply Z.pow_le_mono_r; lia). nia. }
   assert (Hs : shamt w kk = kk) by (unfold shamt; rewrite trunc_small by (cbn; lia); lia).
   rewrite Hs. assert (Hk2p : 2 <= 2 ^ kk) by (apply (Z.le_trans _ (2 ^ 1)); [cbn; lia|apply Z.pow_le_mono_r; lia]).
   assert (Hm : cv * 2 ^ kk < 2 ^ Z.of_nat w) by nia.
   rewrite trunc_small by nia. apply Z2Nat.inj_lt; nia.
Qed.

(* Assuming a register's value range that holds keeps a state sound. *)
Lemma term_pow2_divide : forall a b, 0 <= a <= b -> (2 ^ a | 2 ^ b).
Proof. intros a b H. exists (2 ^ (b - a)). rewrite <- Z.pow_add_r by lia. f_equal. lia. Qed.

Lemma term_assume_sound : forall F E t st rs w sg r lo hi,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs -> wdn F r = w -> (1 <= w <= 64)%nat ->
 lo <= extend w sg (rs r) <= hi -> state_ok F t (term_assume F w sg r lo hi st) rs.
Proof.
 intros F E t st rs w sg r lo hi HE Hrs Hst Hw Hw64 Hv. unfold term_assume.
 set (al := Z.max 0 (Z.min (reg_align F st r) (Z.of_nat w))).
 assert (Hal : aligned al (rs r)).
 { pose proof (reg_align_sound F E HE t st rs Hst Hrs r) as Ha. unfold aligned in *.
   eapply Z.divide_trans; [|exact Ha]. apply term_pow2_divide. unfold al. lia. }
 assert (Hext : aligned al (extend w sg (rs r))).
 { pose proof (rs_range F E rs Hrs r) as Hb. rewrite Hw in Hb.
   assert (Hw2 : (2 ^ Z.max 0 al | 2 ^ Z.of_nat w)) by (apply term_pow2_divide; unfold al; lia).
   unfold aligned in *. unfold extend. destruct sg.
   - unfold sext. rewrite trunc_small by exact Hb.
     destruct (Z.ltb (rs r) (2 ^ (Z.of_nat w - 1))); [exact Hal|apply Z.divide_sub_r; assumption].
   - rewrite trunc_small by exact Hb. exact Hal. }
 assert (Hrange : exists x y, (x <= extend w sg (rs r) <= y) /\
   (let '(x',y') := match cmp_range F st (OReg r) (Z.of_nat w) sg with
      | Some (u,v) => (Z.max lo u, Z.min hi v) | None => (lo,hi) end in x' = x /\ y' = y)).
 { destruct (cmp_range F st (OReg r) (Z.of_nat w) sg) as [[u v]|] eqn:Hc.
   - pose proof (cmp_range_sound F E HE t rs Hrs st (OReg r) (Z.of_nat w) sg u v Hst ltac:(lia)
       ltac:(intros r' Hr'; injection Hr' as <-; unfold wd; rewrite Hw; reflexivity) Hc) as Hb.
     unfold cval in Hb. rewrite Nat2Z.id in Hb. cbn [opv] in Hb.
     exists (Z.max lo u),(Z.min hi v). split; [lia|split; reflexivity].
   - exists lo,hi. split; [exact Hv|split; reflexivity]. }
 destruct Hrange as [x [y [Hxy Heq]]].
 destruct (match cmp_range F st (OReg r) (Z.of_nat w) sg with
      | Some (u,v) => (Z.max lo u, Z.min hi v) | None => (lo,hi) end) as [x0 y0].
 destruct Heq as [-> ->].
 pose proof (round_in al x y (extend w sg (rs r)) Hext Hxy) as Hround.
 destruct (round al (x,y)) as [x' y'] eqn:Hr. cbn [fst snd] in Hround.
 destruct (Z.ltb_spec y' x') as [Hlt|_]; [lia|].
 apply (put_ok F E HE t rs Hrs st r _ Hst). unfold fact_ok, fact_okv.
 split; [unfold wf, range_form; cbn [fm]; lia|].
 split.
 - exists (extend w sg (rs r)). unfold range_form. cbn [fm ft flo fhi tsum]. split; [exact Hround|].
   rewrite Z.add_0_l. apply eqm_sym, extend_eqm.
 - split; [exact Hal|]. intros lo' hi' Hn. discriminate.
Qed.

(* ------------------------------------------------------------------ *)
(* Values written by single steps.                                     *)

Lemma term_step_bin : forall F prog E t s s' op w d a o,
 fetch prog (blk s) (idx s) = Some (IBin None op w d a o) -> phase_step F prog E t s = Some s' ->
 regs s' d = trunc (wdn F d) (binop_eval op w (regs s a) (opv (regs s) o)).
Proof.
 intros F prog E t s s' op w d a o Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs.
 injection Hs as <-. cbn [holds guard_of negb goto regs dest result]. unfold upd. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma term_step_imm : forall F prog E t s s' d z,
 fetch prog (blk s) (idx s) = Some (IImm None d z) -> phase_step F prog E t s = Some s' ->
 regs s' d = trunc (wdn F d) z.
Proof.
 intros F prog E t s s' d z Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs.
 injection Hs as <-. cbn [holds guard_of negb goto regs dest result]. unfold upd. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma term_step_mov : forall F prog E t s s' d r,
 fetch prog (blk s) (idx s) = Some (IMov None d r) -> phase_step F prog E t s = Some s' ->
 regs s' d = trunc (wdn F d) (regs s r).
Proof.
 intros F prog E t s s' d r Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs.
 injection Hs as <-. cbn [holds guard_of negb goto regs dest result]. unfold upd. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma term_step_keeps : forall F prog E t s s' ins r,
 is_cell F r = false -> fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 term_writes r ins = false -> regs s' r = regs s r.
Proof.
 intros F prog E t s s' ins r Hc Hf Hs Hw. apply (loop_step_preserves F prog E t s s' ins r Hc Hf Hs).
 unfold loop_preserves. unfold term_writes in Hw. rewrite Hw. reflexivity.
Qed.

(* The state on each edge out of a block, from a state at its last
   instruction, covers the step taking that edge. *)
Lemma term_edge_sound : forall F prog E tm s s' st last,
 env_ok F E -> regs_ok F E (regs s) -> state_ok F tm st (regs s) -> wf_instr F last = true ->
 fetch prog (blk s) (idx s) = Some last -> S (idx s) = block_len prog (blk s) ->
 phase_step F prog E tm s = Some s' -> idx s' = 0%nat -> (blk s' < List.length prog)%nat ->
 exists os, In (blk s', Some os) (edge_states F prog (blk s) (transfer F st last)) /\
   state_ok F (S tm) os (regs s') /\ regs_ok F E (regs s').
Proof.
 intros F prog E tm s s' st last HE Hrs Hst Hwf Hf Hlen Hs Hi0 Hb'.
 destruct (transfer_sound F prog E HE tm s s' st last Hf Hs Hst Hrs Hwf) as [Hout Hrs'].
 assert (Hnth : nth_error (block_body prog (blk s)) (idx s) = Some last) by exact Hf.
 pose proof (last_of_nth F prog E _ _ _ Hnth Hlen) as Hlast.
 unfold edge_states. rewrite Hlast.
 destruct (step_pos F prog E tm s s' last Hf Hs) as [[Hn [Hbra Hexit]]|[[g [tg [-> [Hg Hp]]]]|[g [-> [Hg Hp]]]]].
 - unfold next_pos in Hn. destruct (Nat.ltb_spec (S (idx s)) (block_len prog (blk s))) as [Hlt|_]; [lia|].
   injection Hn as Hb'' Hi''. rewrite Hb''.
   destruct last as [| | | | | | | | | | | | | | |g tg|g| | | | | | | |];
     try (eexists; split; [left; reflexivity|split; eassumption]).
   + destruct g as [[neg p]|]; [|specialize (Hbra None tg eq_refl); cbn in Hbra; discriminate].
     specialize (Hbra (Some (neg,p)) tg eq_refl). cbn [holds] in Hbra.
     pose proof (jump_regs F prog E tm s s' _ Hf eq_refl Hs) as Hregs.
     destruct (refine_sound F E HE (S tm) (regs s') Hrs' _ p neg Hout) as [os [Hr Hos]].
     { rewrite Hregs. destruct neg, (Z.eqb (regs s p) 0); cbn in Hbra |- *; congruence. }
     exists os. split; [right; left; rewrite Hr; reflexivity|split; assumption].
   + destruct g as [[neg p]|]; [|specialize (Hexit None eq_refl); cbn in Hexit; discriminate].
     specialize (Hexit (Some (neg,p)) eq_refl). cbn [holds] in Hexit.
     pose proof (jump_regs F prog E tm s s' _ Hf eq_refl Hs) as Hregs.
     destruct (refine_sound F E HE (S tm) (regs s') Hrs' _ p neg Hout) as [os [Hr Hos]].
     { rewrite Hregs. destruct neg, (Z.eqb (regs s p) 0); cbn in Hexit |- *; congruence. }
     exists os. split; [left; rewrite Hr; reflexivity|split; assumption].
 - injection Hp as Hb'' Hi''. rewrite Hb''.
   pose proof (jump_regs F prog E tm s s' _ Hf eq_refl Hs) as Hregs.
   destruct g as [[neg p]|].
   + cbn [holds] in Hg.
     destruct (refine_sound F E HE (S tm) (regs s') Hrs' _ p (negb neg) Hout) as [os [Hr Hos]].
     { rewrite Hregs. destruct neg, (Z.eqb (regs s p) 0); cbn in Hg |- *; congruence. }
     exists os. split; [left; rewrite Hr; reflexivity|split; assumption].
   + eexists. split; [left; reflexivity|split; eassumption].
 - injection Hp as Hb'' _. lia.
Qed.

(* ------------------------------------------------------------------ *)
(* Positions in the order of the program.                              *)

Definition term_pos_le (b1 i1 b2 i2 : nat) : Prop := (b1 < b2 \/ (b1 = b2 /\ i1 <= i2))%nat.
Definition term_pos_lt (b1 i1 b2 i2 : nat) : Prop := (b1 < b2 \/ (b1 = b2 /\ i1 < i2))%nat.

Section LoopSteps.
Variable F : lfacts.
Variable prog : program.
Variable E : env.
Variables l h : nat.
Hypothesis Hshape : term_shape prog = true.
Hypothesis He : In (l,h) (term_edges prog).

Lemma term_in_iff : forall x, term_in h l x = true <-> (h <= x <= l)%nat.
Proof. intros x. unfold term_in. rewrite andb_true_iff, !Nat.leb_le. reflexivity. Qed.

(* A back edge from inside the loop, other than the loop's back edge, goes to a later head. *)
Lemma term_inner_head : forall b v, In (b,v) (term_edges prog) -> (h <= b <= l)%nat -> v <> h -> (h < v)%nat.
Proof.
 intros b v Hb Hbin Hv. destruct (term_edges_spec prog b v Hb) as [_ [Hvb _]].
 destruct (term_laminar prog Hshape b v l h b Hb He ltac:(lia) ltac:(lia)) as [[A B]|[A B]].
 - assert (b = l) by lia. subst b. pose proof (term_edges_block prog _ _ _ Hb He). contradiction.
 - lia.
Qed.

(* For a point P in the head or the latch block, a step inside the loop
   that is not the back edge passes P only by executing it. *)
Lemma term_step_order : forall t s s' ins pb pi,
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 term_in h l (blk s) = true -> term_in h l (blk s') = true -> ~ (blk s' = h /\ idx s' = 0%nat) ->
 (pb = h \/ pb = l) -> (pi < block_len prog pb)%nat ->
 (term_pos_le (blk s') (idx s') pb pi -> term_pos_lt (blk s) (idx s) pb pi) /\
 (term_pos_lt pb pi (blk s') (idx s') -> term_pos_le pb pi (blk s) (idx s)).
Proof.
 intros t s s' ins pb pi Hf Hs Hin Hin' Hnh Hpb Hpi.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 apply term_in_iff in Hin, Hin'. destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 unfold term_pos_le, term_pos_lt.
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hb [Hi _]]|[Hi0 [Hlen [_ [Hend|Htg]]]]].
 - rewrite Hb, Hi. split; intros H; lia.
 - lia.
 - destruct (Nat.leb_spec (blk s') (blk s)) as [Hle|Hgt].
   + pose proof (term_step_back F prog E t s s' ins Hstr Hf Hs Hle Hi0) as Hback.
     assert (Hv : blk s' <> h) by (intros Hv; apply Hnh; split; assumption).
     pose proof (term_inner_head _ _ Hback ltac:(lia) Hv) as Hhv.
     assert (Hbl : blk s <> l).
     { intros Hbl. rewrite Hbl in Hback. pose proof (term_edges_block prog _ _ _ Hback He). contradiction. }
     split; intros H; destruct Hpb as [->| ->]; lia.
   + assert (Hlast : blk s = pb -> (pi <= idx s)%nat).
     { intros Hq. rewrite <- Hq in Hpi. lia. }
     split; intros H; destruct Hpb as [->| ->]; lia.
Qed.

(* A step to a position after index 0 of a block comes from the previous index. *)
Lemma term_step_prev : forall t s s' ins,
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' -> (0 < idx s')%nat ->
 blk s' = blk s /\ idx s' = S (idx s).
Proof.
 intros t s s' ins Hf Hs Hi. pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hb [Hi' _]]|[Hi0 _]]; [|lia].
 split; assumption.
Qed.
End LoopSteps.

(* ------------------------------------------------------------------ *)
(* A checked counter ends its loop.                                    *)

Lemma term_short_all_in : forall A (p : A -> bool) xs x, short_all p xs = true -> In x xs -> p x = true.
Proof. intros A p xs x H Hx. rewrite short_all_same, forallb_forall in H. exact (H x Hx). Qed.

Lemma term_hull_none : forall rs, In None rs -> term_range_hull rs = None.
Proof.
 intros rs Hin. induction rs as [|r rs IH]; [contradiction|]. cbn [term_range_hull fold_right].
 destruct Hin as [->|Hin]; [reflexivity|]. fold (term_range_hull rs). rewrite (IH Hin).
 destruct r as [[x y]|]; reflexivity.
Qed.

Lemma term_hull_empty : forall rs, term_range_hull rs = Some None -> rs = [].
Proof.
 intros rs H. destruct rs as [|r rs]; [reflexivity|]. cbn [term_range_hull fold_right] in H.
 fold (term_range_hull rs) in H. destruct r as [[x y]|]; [|discriminate].
 destruct (term_range_hull rs) as [[[? ?]|]|]; discriminate.
Qed.

Lemma term_hull_some : forall rs x y u v, term_range_hull rs = Some (Some (x,y)) -> In (Some (u,v)) rs ->
 x <= u /\ v <= y.
Proof.
 intros rs. induction rs as [|r rs IH]; intros x y u v H Hin; [contradiction|].
 cbn [term_range_hull fold_right] in H. fold (term_range_hull rs) in H.
 destruct r as [[p q]|]; [|discriminate].
 destruct (term_range_hull rs) as [[[m n]|]|] eqn:Hr; try discriminate.
 - injection H as <- <-. destruct Hin as [Hin|Hin].
   + injection Hin as -> ->. lia.
   + destruct (IH m n u v eq_refl Hin). lia.
 - injection H as <- <-. destruct Hin as [Hin|Hin].
   + injection Hin as -> ->. lia.
   + rewrite (term_hull_empty rs Hr) in Hin. contradiction.
Qed.

Lemma term_wf_at : forall F prog b x z ins, term_wf_from F prog b x = true -> (x <= z)%nat ->
 fetch prog b z = Some ins -> wf_instr F ins = true.
Proof.
 intros F prog b x z ins H Hz Hf. unfold term_wf_from in H. rewrite forallb_forall in H. apply H.
 unfold fetch in Hf. apply (nth_error_In _ (z - x)). rewrite term_nth_skipn.
 replace (x + (z - x))%nat with z by lia. exact Hf.
Qed.

Lemma term_walk_step : forall F prog b x z ins st, (x <= z)%nat -> fetch prog b z = Some ins ->
 walk F st (term_slice prog b x (S z)) = transfer F (walk F st (term_slice prog b x z)) ins.
Proof.
 intros F prog b x z ins st Hz Hf. rewrite (term_slice_snoc prog b x z ins Hz Hf), term_walk_app. reflexivity.
Qed.

Lemma term_wd_range : forall F E r, env_ok F E -> (1 <= wdn F r <= 64)%nat.
Proof.
 intros F E r HE. pose proof (ok_widths F E HE r) as H. unfold wdz, wd in H. lia.
Qed.

Lemma term_cmp_reg : forall F E t rs st r w sg x y,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs -> wdn F r = w ->
 cmp_range F st (OReg r) (Z.of_nat w) sg = Some (x,y) -> x <= extend w sg (rs r) <= y.
Proof.
 intros F E t rs st r w sg x y HE Hrs Hst Hw Hc. pose proof (term_wd_range F E r HE) as Hr.
 pose proof (cmp_range_sound F E HE t rs Hrs st (OReg r) (Z.of_nat w) sg x y Hst ltac:(lia)
   ltac:(intros r' Hr'; injection Hr' as <-; unfold wd; rewrite Hw; reflexivity) Hc) as Hb.
 unfold cval in Hb. rewrite Nat2Z.id in Hb. exact Hb.
Qed.

Lemma term_step_operand : forall F st k op w d a o x y,
 term_step_ok F st k (IBin None op w d a o) x y = true -> wf_instr F (IBin None op w d a o) = true ->
 forall r, o = OReg r -> wd F r = Z.of_nat w.
Proof.
 intros F st k op w d a o x y Hok Hwf r ->. unfold wf_instr in Hwf. cbn [dest] in Hwf.
 rewrite !andb_true_iff in Hwf. destruct Hwf as [[_ [_ Hb]] _].
 destruct k, op; cbn [term_step_ok] in Hok; try discriminate; cbn [opnd_width_ok] in Hb; apply Z.eqb_eq; exact Hb.
Qed.

Lemma term_cmp_opnd : forall F E t rs st o w sg x y,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs -> (1 <= w <= 64)%nat ->
 (forall r, o = OReg r -> wdn F r = w) ->
 cmp_range F st o (Z.of_nat w) sg = Some (x,y) -> x <= extend w sg (opv rs o) <= y.
Proof.
 intros F E t rs st o w sg x y HE Hrs Hst Hw Ho Hc.
 pose proof (cmp_range_sound F E HE t rs Hrs st o (Z.of_nat w) sg x y Hst ltac:(lia)
   ltac:(intros r' Hr'; unfold wd; rewrite (Ho r' Hr'); reflexivity) Hc) as Hb.
 unfold cval in Hb. rewrite Nat2Z.id in Hb. exact Hb.
Qed.

Lemma term_xfer_sound : forall F prog E tm s s' st ins,
 env_ok F E -> fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E tm s = Some s' ->
 state_ok F tm st (regs s) -> regs_ok F E (regs s) -> wf_instr F ins = true ->
 state_ok F (S tm) (term_xfer F st ins) (regs s') /\ regs_ok F E (regs s').
Proof.
 intros F prog E tm s s' st ins HE Hf Hs Hst Hrs Hwf.
 destruct (transfer_sound F prog E HE tm s s' st ins Hf Hs Hst Hrs Hwf) as [Hout Hrs'].
 split; [|exact Hrs']. unfold term_xfer.
 destruct (is_bottom (transfer F st ins)); [exact Hout|].
 destruct ins as [g d r0 |  | g op w d a o |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ]; try exact Hout.
 destruct g as [[neg p]|]; [exact Hout|].
 assert (Hopnd : forall r, match o with OReg r => Nat.eqb (wdn F r) w | OImm _ => true end = true ->
   o = OReg r -> wdn F r = w).
 { intros r Hm ->. apply Nat.eqb_eq. exact Hm. }
 destruct op; try exact Hout.
 - destruct (cmp_range F st (OReg a) (Z.of_nat w) true) as [[x y]|] eqn:Ha; [|exact Hout].
   destruct (cmp_range F st o (Z.of_nat w) true) as [[u v]|] eqn:Ho; [|exact Hout].
   match goal with |- context [if ?c then _ else _] => destruct c eqn:Hc end; [|exact Hout].
   rewrite !andb_true_iff in Hc. destruct Hc as [[[[[[[Hwd Hwa] Hw1] Hw64] Hwo] Hlo] Hhi] _].
   apply Nat.eqb_eq in Hwd, Hwa. apply Nat.leb_le in Hw1, Hw64. apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi.
   apply (term_assume_sound F E (S tm) _ (regs s') w true d _ _ HE Hrs' Hout Hwd ltac:(lia)).
   rewrite (term_step_bin F prog E tm s s' BAdd w d a o Hf Hs), Hwd. cbn [binop_eval extend].
   pose proof (term_cmp_reg F E tm (regs s) st a w true x y HE Hrs Hst Hwa Ha) as Bx.
   pose proof (term_cmp_opnd F E tm (regs s) st o w true u v HE Hrs Hst ltac:(lia) (fun r => Hopnd r Hwo) Ho) as Bu.
   cbn [extend] in Bx, Bu.
   rewrite (term_sext_sum w (regs s a) (opv (regs s) o) (sext w (regs s a) + sext w (opv (regs s) o)) ltac:(lia)).
   + lia.
   + lia.
   + apply eqm_add; [lia|apply eqm_sym, sext_eqm|apply eqm_sym, sext_eqm].
 - destruct (cmp_range F st (OReg a) (Z.of_nat w) true) as [[x y]|] eqn:Ha; [|exact Hout].
   destruct (cmp_range F st o (Z.of_nat w) true) as [[u v]|] eqn:Ho; [|exact Hout].
   match goal with |- context [if ?c then _ else _] => destruct c eqn:Hc end; [|exact Hout].
   rewrite !andb_true_iff in Hc. destruct Hc as [[[[[[[Hwd Hwa] Hw1] Hw64] Hwo] Hlo] Hhi] _].
   apply Nat.eqb_eq in Hwd, Hwa. apply Nat.leb_le in Hw1, Hw64. apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi.
   apply (term_assume_sound F E (S tm) _ (regs s') w true d _ _ HE Hrs' Hout Hwd ltac:(lia)).
   rewrite (term_step_bin F prog E tm s s' BSub w d a o Hf Hs), Hwd. cbn [binop_eval extend].
   pose proof (term_cmp_reg F E tm (regs s) st a w true x y HE Hrs Hst Hwa Ha) as Bx.
   pose proof (term_cmp_opnd F E tm (regs s) st o w true u v HE Hrs Hst ltac:(lia) (fun r => Hopnd r Hwo) Ho) as Bu.
   cbn [extend] in Bx, Bu.
   replace (regs s a - opv (regs s) o) with (regs s a + - opv (regs s) o) by ring.
   rewrite (term_sext_sum w (regs s a) (- opv (regs s) o) (sext w (regs s a) + - sext w (opv (regs s) o)) ltac:(lia)).
   + lia.
   + lia.
   + apply eqm_add; [lia|apply eqm_sym, sext_eqm|apply eqm_opp; [lia|apply eqm_sym, sext_eqm]].
Qed.

Lemma term_twalk_app : forall F st l1 l2, term_walk F st (l1 ++ l2) = term_walk F (term_walk F st l1) l2.
Proof. intros F st l1 l2. unfold term_walk. apply fold_left_app. Qed.

Lemma term_twalk_step : forall F prog b x z ins st, (x <= z)%nat -> fetch prog b z = Some ins ->
 term_walk F st (term_slice prog b x (S z)) = term_xfer F (term_walk F st (term_slice prog b x z)) ins.
Proof.
 intros F prog b x z ins st Hz Hf. rewrite (term_slice_snoc prog b x z ins Hz Hf), term_twalk_app. reflexivity.
Qed.

Lemma term_cmp_assume_sound : forall F E t st s rs holds k,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs -> state_ok F t s rs ->
 cmp_eval (cc k) (extend (cw k) (csg k) (rs (ca k))) (extend (cw k) (csg k) (opv rs (cb k))) = holds ->
 state_ok F t (term_cmp_assume F st s holds k) rs.
Proof.
 intros F E t st s rs holds k HE Hrs Hst Hs Hk. unfold term_cmp_assume.
 destruct (Nat.eqb (wdn F (ca k)) (cw k) && Nat.leb 1 (cw k) && Nat.leb (cw k) 64) eqn:Hc; [|exact Hs].
 rewrite !andb_true_iff, Nat.eqb_eq, !Nat.leb_le in Hc. destruct Hc as [[Hw Hw1] Hw64].
 destruct (cmp_range F st (OReg (ca k)) (Z.of_nat (cw k)) true) as [[x y]|] eqn:Hr; [|exact Hs].
 destruct (term_cmp_bounds k holds x y) as [[lo hi]|] eqn:Hb; [|exact Hs].
 apply (term_assume_sound F E t s rs (cw k) true (ca k) lo hi HE Hrs Hs Hw ltac:(lia)).
 pose proof (term_cmp_reg F E t rs st (ca k) (cw k) true x y HE Hrs Hst Hw Hr) as Bx.
 unfold term_cmp_bounds in Hb. destruct (cb k) as [r|z] eqn:Hcb; [discriminate|].
 destruct (csg k) eqn:Hsg; [|discriminate].
 cbn [opv extend] in Hk, Bx |- *.
 set (V := sext (cw k) (rs (ca k))) in *. set (c0 := sext (cw k) z) in *.
 assert (Hcv : cmp_eval (if holds then cc k else flip (cc k)) V c0 = true).
 { destruct holds; [exact Hk|rewrite flip_eval, Hk; reflexivity]. }
 destruct (if holds then cc k else flip (cc k)); cbn [cmp_eval] in Hcv; injection Hb as <- <-.
 - apply Z.ltb_lt in Hcv. lia.
 - apply Z.leb_le in Hcv. lia.
 - apply Z.ltb_lt in Hcv. lia.
 - apply Z.leb_le in Hcv. lia.
 - apply Z.eqb_eq in Hcv. lia.
 - apply negb_true_iff, Z.eqb_neq in Hcv.
   destruct (Z.eqb_spec x c0), (Z.eqb_spec y c0); lia.
Qed.

Lemma term_refine_sound : forall F E t rs st p b,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs -> negb (Z.eqb (rs p) 0) = b ->
 exists s, term_refine F st p b = Some s /\ state_ok F t s rs.
Proof.
 intros F E t rs st p b HE Hrs Hst Hb. unfold term_refine.
 destruct (refine_sound F E HE t rs Hrs st p b Hst Hb) as [s0 [Hr Hs0]]. rewrite Hr.
 destruct (is_bottom s0); [exists s0; split; [reflexivity|exact Hs0]|].
 eexists. split; [reflexivity|].
 assert (Hfold : forall cs s, Forall (cmp_holds rs) cs -> state_ok F t s rs ->
   state_ok F t (fold_left (fun acc k => term_cmp_assume F st acc true k) cs s) rs).
 { intros cs. induction cs as [|k cs IH]; intros s Hall Hs; [exact Hs|].
   inversion Hall as [|k' cs' Hk Hcs]; subst. cbn [fold_left]. apply IH; [exact Hcs|].
   apply (term_cmp_assume_sound F E t st s rs true k HE Hrs Hst Hs). exact Hk. }
 destruct (get st p) as [x|] eqn:Hg; [|exact Hs0].
 pose proof (proj1 Hst p x Hg) as Hx. unfold fact_ok, fact_okv in Hx.
 destruct x as [f al vr|kk|cs exact|cs exact]; try exact Hs0.
 - destruct b.
   + apply Hfold; [|exact Hs0]. apply (proj1 Hx). apply negb_true_iff, Z.eqb_neq in Hb. exact Hb.
   + destruct cs as [|k [|k2 cs]]; try exact Hs0. destruct exact; [|exact Hs0].
     apply negb_false_iff, Z.eqb_eq in Hb.
     apply (term_cmp_assume_sound F E t st s0 rs false k HE Hrs Hst Hs0).
     destruct (cmp_eval _ _ _) eqn:Hc; [|reflexivity]. exfalso.
     apply (proj2 Hx eq_refl); [|exact Hb]. constructor; [exact Hc|constructor].
 - destruct b.
   + destruct cs as [|k [|k2 cs]]; try exact Hs0. destruct exact; [|exact Hs0].
     apply negb_true_iff, Z.eqb_neq in Hb.
     apply (term_cmp_assume_sound F E t st s0 rs false k HE Hrs Hst Hs0).
     destruct (cmp_eval _ _ _) eqn:Hc; [|reflexivity]. exfalso.
     apply Hb, (proj2 Hx eq_refl). constructor; [exact Hc|constructor].
   + apply Hfold; [|exact Hs0]. apply (proj1 Hx). apply negb_false_iff, Z.eqb_eq in Hb. exact Hb.
Qed.

Lemma term_sext_trunc : forall n v, sext n (trunc n v) = sext n v.
Proof. intros n v. unfold sext, trunc. rewrite Z.mod_mod by (apply Z.pow_nonzero; lia). reflexivity. Qed.

Lemma term_step_selp : forall F prog E t s s' w' dd a b p,
 fetch prog (blk s) (idx s) = Some (ISelp None w' dd a b p) -> phase_step F prog E t s = Some s' ->
 regs s' dd = trunc (wdn F dd) (if Z.eqb (regs s p) 0 then opv (regs s) b else opv (regs s) a).
Proof.
 intros F prog E t s s' w' dd a b p Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs.
 injection Hs as <-. cbn [holds guard_of negb goto regs dest result]. unfold upd. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* A register that a selection wrote earlier in the block, with neither it
   nor the predicate written since, holds the selected value. *)
Lemma term_sel_value : forall F prog E tm s l m i r w' x1 x2 p,
 structured prog = true -> run' F prog E tm = Some s -> blk s = l -> (m < idx s <= i)%nat ->
 fetch prog l m = Some (ISelp None w' r (OImm x1) (OImm x2) p) -> r <> p ->
 is_cell F r = false -> is_cell F p = false ->
 (forall k ins, (m < k < i)%nat -> fetch prog l k = Some ins -> term_writes r ins = false /\ term_writes p ins = false) ->
 regs s r = trunc (wdn F r) (if Z.eqb (regs s p) 0 then x2 else x1).
Proof.
 intros F prog E tm. induction tm as [|tm IH]; intros s l m i r w' x1 x2 p Hstr Hr Hb Hmi Hfm Hrp Hcr Hcp Hk.
 - cbn in Hr. injection Hr as <-. cbn [idx] in Hmi. lia.
 - assert (Hx : exists s0, run' F prog E tm = Some s0 /\ phase_step F prog E tm s0 = Some s).
   { rewrite term_run_succ in Hr. destruct (run' F prog E tm) as [s0|]; [|discriminate]. exists s0. split; [reflexivity|exact Hr]. }
   clear Hr. destruct Hx as [s0 [Hr0 Hr]].
   destruct (fetch prog (blk s0) (idx s0)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hr. rewrite Hf in Hr. discriminate. }
   destruct (term_step_shape F prog E tm s0 s ins Hstr Hf Hr) as [[Hbb [Hii _]]|[Hi0 _]]; [|lia].
   assert (Hb0 : blk s0 = l) by congruence.
   destruct (Nat.eq_dec (idx s0) m) as [Hm|Hm].
   + assert (Hfs : fetch prog (blk s0) (idx s0) = Some (ISelp None w' r (OImm x1) (OImm x2) p))
       by (rewrite Hb0, Hm; exact Hfm).
     rewrite (term_step_selp F prog E tm s0 s w' r (OImm x1) (OImm x2) p Hfs Hr).
     rewrite (term_step_keeps F prog E tm s0 s _ p Hcp Hfs Hr).
     * reflexivity.
     * unfold term_writes. cbn [loop_writes dest existsb]. rewrite (proj2 (Pos.eqb_neq p r) (not_eq_sym Hrp)). reflexivity.
   + pose proof (IH s0 l m i r w' x1 x2 p Hstr Hr0 Hb0 ltac:(lia) Hfm Hrp Hcr Hcp Hk) as Hv.
     rewrite Hb0 in Hf. destruct (Hk (idx s0) ins ltac:(lia) Hf) as [Hwr Hwp].
     rewrite <- Hb0 in Hf.
     rewrite (term_step_keeps F prog E tm s0 s ins r Hcr Hf Hr Hwr), (term_step_keeps F prog E tm s0 s ins p Hcp Hf Hr Hwp).
     exact Hv.
Qed.

(* A step to the head over a back edge from block v takes the branch that
   ends v and keeps the registers. *)
Lemma term_latch_jump : forall F prog E v h tm s s' ins, term_shape prog = true -> In (v,h) (term_edges prog) ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E tm s = Some s' -> blk s = v -> blk s' = h -> idx s' = 0%nat ->
 S (idx s) = block_len prog v /\ last_instr (block_body prog v) = Some ins /\
 exists g, ins = IBra g h /\ holds g (regs s) = true /\ regs s' = regs s.
Proof.
 intros F prog E v h tm s s' ins Hshape Hv Hf Hs Hbl Hbh Hi0. pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_edges_spec prog v h Hv) as [Hl [Hhl [g Hg]]].
 destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hs) as [[_ [Hi _]]|[_ [Hlen [Hlast _]]]]; [lia|].
 rewrite Hbl in Hlen, Hlast. split; [exact Hlen|]. split; [exact Hlast|].
 rewrite Hg in Hlast. injection Hlast as <-. exists g. split; [reflexivity|].
 destruct (step_pos F prog E tm s s' (IBra g h) Hf Hs) as [[Hn [Hbra _]]|[[g' [tg [Hi' [Hg' Hp]]]]|[g' [Hi' _]]]].
 - exfalso. unfold next_pos in Hn. rewrite Hbl in Hn.
   destruct (Nat.ltb (S (idx s)) (block_len prog v)); injection Hn as Hb' Hi''; lia.
 - injection Hi' as <- _. split; [exact Hg'|]. exact (jump_regs F prog E tm s s' _ Hf eq_refl Hs).
 - discriminate.
Qed.

(* A property of registers R that the instruction at index m of block b
   establishes holds later in the block while no instruction writes R. *)
Lemma term_hist : forall F prog E (P : (reg -> Z) -> Prop) (R : list reg) b m,
 structured prog = true ->
 (forall t0 s0 s0', blk s0 = b -> idx s0 = m -> phase_step F prog E t0 s0 = Some s0' -> P (regs s0')) ->
 (forall rs rs', (forall r, In r R -> rs' r = rs r) -> P rs -> P rs') ->
 (forall r, In r R -> is_cell F r = false) ->
 forall tm s, run' F prog E tm = Some s -> blk s = b -> (m < idx s)%nat ->
 (forall k ins, (m < k < idx s)%nat -> fetch prog b k = Some ins -> forall r, In r R -> term_writes r ins = false) ->
 P (regs s).
Proof.
 intros F prog E P R b m Hstr Hset Hframe Hcell tm. induction tm as [|tm IH]; intros s Hr Hb Hm Hk.
 - cbn in Hr. injection Hr as <-. cbn [idx] in Hm. lia.
 - assert (Hx : exists s0, run' F prog E tm = Some s0 /\ phase_step F prog E tm s0 = Some s).
   { rewrite term_run_succ in Hr. destruct (run' F prog E tm) as [s0|]; [|discriminate].
     exists s0. split; [reflexivity|exact Hr]. }
   destruct Hx as [s0 [Hr0 Hs]]. clear Hr.
   destruct (fetch prog (blk s0) (idx s0)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   destruct (term_step_shape F prog E tm s0 s ins Hstr Hf Hs) as [[Hbb [Hii _]]|[Hi0 _]]; [|lia].
   assert (Hb0 : blk s0 = b) by congruence.
   destruct (Nat.eq_dec (idx s0) m) as [Hm0|Hm0].
   + exact (Hset tm s0 s Hb0 Hm0 Hs).
   + apply (Hframe (regs s0)).
     * intros r Hr'. apply (term_step_keeps F prog E tm s0 s ins r (Hcell r Hr') Hf Hs).
       rewrite Hb0 in Hf. exact (Hk (idx s0) ins ltac:(lia) Hf r Hr').
     * apply (IH s0 Hr0 Hb0 ltac:(lia)). intros k ins' Hk' Hf'. apply (Hk k ins'); [lia|exact Hf'].
Qed.

(* The certificate's states at the start of a block, restricted and walked,
   agree with every reachable state in the block. *)
Lemma term_cut_sound : forall keep F prog cert E, check_scalar_certificate F prog cert = true -> launch_ok F E ->
 structured prog = true -> forall k tm s, run' F prog E tm = Some s -> idx s = k ->
 (blk s < List.length prog)%nat -> term_block_wf F prog cert (blk s) = true ->
 exists F' st0, In (F',st0) (term_states F prog cert (blk s) 0) /\ is_bottom st0 = false /\
   term_same F F' /\ env_ok F' E /\ regs_ok F' E (regs s) /\
   state_ok F' tm (term_walk F' (term_cut F' keep st0) (term_slice prog (blk s) 0 k)) (regs s).
Proof.
 intros keep F prog cert E Hcert HE Hstr k. induction k as [|k IH]; intros tm s Hr Hk Hb Hwfb.
 - destruct (term_states_sound F prog cert E tm s Hcert HE Hr Hb) as [_ [F' [st [Hst [Hsame [HE' [Hrs Hok]]]]]]].
   rewrite Hk in Hst. exists F',st. split; [exact Hst|].
   split; [destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm st _ Hbot Hok)|reflexivity]|].
   split; [exact Hsame|]. split; [exact HE'|]. split; [exact Hrs|].
   rewrite term_slice_empty. unfold term_walk. cbn [fold_left]. apply trim_live_seed_sound. exact Hok.
 - destruct tm as [|tm].
   { cbn in Hr. injection Hr as <-. cbn in Hk. discriminate. }
   assert (Hx : exists s0, run' F prog E tm = Some s0 /\ phase_step F prog E tm s0 = Some s).
   { rewrite term_run_succ in Hr. destruct (run' F prog E tm) as [s0|]; [|discriminate].
     exists s0. split; [reflexivity|exact Hr]. }
   destruct Hx as [s0 [Hr0 Hs]].
   destruct (fetch prog (blk s0) (idx s0)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   destruct (term_step_shape F prog E tm s0 s ins Hstr Hf Hs) as [[Hbb [Hii _]]|[Hi0 _]]; [|lia].
   assert (Hk0 : idx s0 = k) by lia.
   assert (Hb0 : blk s0 = blk s) by congruence.
   destruct (IH tm s0 Hr0 Hk0 ltac:(rewrite Hb0; exact Hb) ltac:(rewrite Hb0; exact Hwfb))
     as [F' [st0 [Hst [Hnb [Hsame [HE' [Hrs Hok]]]]]]].
   rewrite Hb0 in Hst, Hok.
   exists F',st0. split; [exact Hst|]. split; [exact Hnb|]. split; [exact Hsame|]. split; [exact HE'|].
   assert (Hwfi : wf_instr F' ins = true).
   { unfold term_block_wf in Hwfb. pose proof (term_short_all_in _ _ _ _ Hwfb Hst) as Hw. cbv beta iota in Hw.
     rewrite Hnb in Hw. cbn [orb] in Hw. rewrite forallb_forall in Hw. apply Hw.
     rewrite Hb0 in Hf. unfold fetch in Hf. exact (nth_error_In _ _ Hf). }
   assert (Hs' : phase_step F' prog E tm s0 = Some s) by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
   destruct (term_xfer_sound F' prog E tm s0 s _ ins HE' Hf Hs' Hok Hrs Hwfi) as [Hok' Hrs'].
   split; [exact Hrs'|].
   rewrite (term_twalk_step F' prog (blk s) 0 k ins); [exact Hok'|lia|rewrite <- Hb0, <- Hk0; exact Hf].
Qed.

Section Counter.
Variable F : lfacts.
Variable prog : program.
Variable cert : scalar_certificate.
Variable E : env.
Variables l h : nat.
Variables (c : reg) (b i j : nat) (al : term_alias) (sh : option nat) (k : term_kind) (lo hi : Z)
          (op : binop) (w : nat) (t a : reg) (o : GateIsa.opnd).
Local Notation stp := (IBin None op w t a o).
Local Notation d := {| tc_c := c; tc_b := b; tc_i := i; tc_j := j; tc_step := IBin None op w t a o;
            tc_alias := al; tc_shift := sh; tc_kind := k; tc_lo := lo; tc_hi := hi |}.
Local Notation sg := (term_signed k).
Hypothesis Hcert : check_scalar_certificate F prog cert = true.
Hypothesis HE : launch_ok F E.
Hypothesis Hshape : term_shape prog = true.
Hypothesis He : In (l,h) (term_edges prog).
Hypothesis Hstatic : term_counter_static F prog h l d op w t a o = true.
Hypothesis Hentry : short_all (term_within lo hi) (term_entry_ranges F prog cert h l c w sg) = true.
Hypothesis Hstepok : term_counter_step F prog cert h l d w a = true.
Hypothesis Hback : (if forallb (Nat.eqb l) (term_latches prog h l) then term_counter_back F prog cert h l d w a
  else term_counter_multi F prog cert h l d w a) = true.
Hypothesis Hwfb : term_block_wf F prog cert b = true.
Hypothesis Hwfl : term_block_wf F prog cert l = true.
Local Notation uniq := (forallb (Nat.eqb l) (term_latches prog h l)).
Local Notation Kb := (term_counter_keep prog b d).
Local Notation Kl := (term_counter_keep prog l d).
Local Notation reach tm s := (run' F prog E tm = Some s).

Lemma tc_static :
 (b = h \/ b = l) /\ (j <= i)%nat /\ lo <= hi /\
 is_cell F c = false /\ is_cell F a = false /\ is_cell F t = false /\
 wdn F c = w /\ wdn F a = w /\ wdn F t = w.
Proof.
 pose proof Hstatic as Hst0.
 unfold term_counter_static in Hst0. cbn [tc_c tc_b tc_i tc_j tc_lo tc_hi] in Hst0.
 rewrite !andb_true_iff in Hst0.
 destruct Hst0 as [[[[[[[[[[[[_ Hb] Hji] Hlh] Hc] Ha] Ht] Hwc] Hwa] Hwt] _] _] _].
 rewrite orb_true_iff, !Nat.eqb_eq in Hb. apply Nat.leb_le in Hji. apply Z.leb_le in Hlh.
 apply negb_true_iff in Hc, Ha, Ht. apply Nat.eqb_eq in Hwc, Hwa, Hwt. tauto.
Qed.

Lemma tc_back_u : uniq = true -> term_counter_back F prog cert h l d w a = true.
Proof. intros Hu. rewrite Hu in Hback. exact Hback. Qed.

Lemma tc_multi : uniq = false -> b = h /\ h <> l /\ (forall v, In v (term_latches prog h l) -> (h < v)%nat) /\
 (term_counter_hull F prog cert h l d w a = Some None \/
  exists x y, term_counter_hull F prog cert h l d w a = Some (Some (x,y)) /\ lo <= x /\ y <= hi).
Proof.
 intros Hu. pose proof Hback as Hb0. rewrite Hu in Hb0. unfold term_counter_multi, term_counter_multi_with in Hb0.
 cbn [tc_b tc_lo tc_hi tc_j] in Hb0. rewrite !andb_true_iff in Hb0. destruct Hb0 as [[[Hbh Hhl] Hlat] Hh].
 apply Nat.eqb_eq in Hbh. apply negb_true_iff, Nat.eqb_neq in Hhl.
 split; [exact Hbh|]. split; [exact Hhl|]. split.
 - intros v Hv. rewrite forallb_forall in Hlat. apply Nat.ltb_lt. exact (Hlat v Hv).
 - unfold term_counter_hull. cbn [tc_j].
   destruct (term_counter_hull_with _ F prog h l d w a) as [[[x y]|]|]; [right|left; reflexivity|discriminate].
   rewrite andb_true_iff, !Z.leb_le in Hh. exists x,y. split; [reflexivity|exact Hh].
Qed.

Lemma tc_entered : term_entered prog h l = true \/ ((forall m, al <> AHead m) /\ sh = None).
Proof.
 pose proof Hstatic as Hst0.
 unfold term_counter_static in Hst0. cbn [tc_c tc_b tc_i tc_j tc_lo tc_hi tc_alias tc_shift] in Hst0.
 rewrite !andb_true_iff in Hst0.
 destruct Hst0 as [[[[[[[[[[[[He' _] _] _] _] _] _] _] _] _] _] _] _].
 rewrite orb_true_iff in He'. destruct He' as [He'|He']; [left; exact He'|right].
 split; [intros m ->; discriminate|]. destruct al, sh; first [reflexivity|discriminate].
Qed.

Lemma tc_writes : exists insw, term_region_writes prog h l c = [(b,i,insw)] /\ fetch prog b i = Some insw /\
 ((j = i /\ sh = None /\ t = c /\ fetch prog b i = Some stp /\ insw = stp) \/
  ((j < i)%nat /\ sh = None /\ insw = IMov None c t /\ t <> c /\ fetch prog b j = Some stp /\
   term_kept prog b (S j) i t = true /\
   (forall r, o = OReg r -> is_cell F r = false /\ term_kept prog b j i r = true)) \/
  (exists m, j = i /\ sh = Some m /\ insw = IMov None c t /\ t <> c /\ a = c /\ b = l /\ h <> l /\
   wf_instr F stp = true /\ term_region_writes prog h l t = [(h, m, stp)] /\
   (forall r, o = OReg r -> is_cell F r = false /\
     forall b2 m2 ins2, In (b2,m2,ins2) (term_region_writes prog h l r) -> b2 = h /\ (m2 < m)%nat))).
Proof.
 pose proof tc_static as [_ [Hji0 _]].
 pose proof Hstatic as Hst0.
 unfold term_counter_static in Hst0. cbn [tc_c tc_b tc_i tc_j tc_lo tc_hi tc_step tc_shift] in Hst0.
 rewrite !andb_true_iff in Hst0.
 destruct Hst0 as [[[_ Hw] _] _].
 destruct (term_region_writes prog h l c) as [|[[b' i'] insw] [|]] eqn:Hrw; try discriminate.
 exists insw. rewrite !andb_true_iff in Hw. destruct Hw as [[Hb' Hi'] Hrest].
 apply Nat.eqb_eq in Hb', Hi'. subst b' i'. split; [reflexivity|].
 assert (Hfw : fetch prog b i = Some insw).
 { assert (Hin : In (b,i,insw) (term_region_writes prog h l c)) by (rewrite Hrw; left; reflexivity).
   apply term_region_writes_spec in Hin. tauto. }
 split; [exact Hfw|].
 destruct (Nat.eqb_spec j i) as [->|Hne].
 - destruct sh as [m|].
   + right. right. exists m. rewrite !andb_true_iff in Hrest.
     destruct Hrest as [[[[[[[Hbl Hhl] Htc] Hac] Hwf] Hmov] Ht] Ho].
     apply Nat.eqb_eq in Hbl. apply negb_true_iff, Nat.eqb_neq in Hhl.
     apply negb_true_iff, Pos.eqb_neq in Htc. apply Pos.eqb_eq in Hac.
     destruct insw as [| | | | | | | | | | | | | | | | | | | | | | | |]; try discriminate.
     destruct g as [[neg p]|]; [discriminate|]. rewrite andb_true_iff, !Pos.eqb_eq in Hmov.
     destruct Hmov as [-> ->].
     destruct (term_region_writes prog h l t) as [|[[b2 m2] s2] [|]] eqn:Hrt; try discriminate.
     rewrite !andb_true_iff, !Nat.eqb_eq in Ht. destruct Ht as [[-> ->] Hs2].
     destruct (loop_instr_eq_dec s2 stp) as [->|]; [|discriminate].
     split; [reflexivity|]. split; [reflexivity|]. split; [reflexivity|]. split; [exact Htc|].
     split; [exact Hac|]. split; [exact Hbl|]. split; [exact Hhl|]. split; [exact Hwf|].
     split; [reflexivity|].
     intros r ->. rewrite andb_true_iff in Ho. destruct Ho as [Hc Hr].
     apply negb_true_iff in Hc. split; [exact Hc|].
     intros b2 m2 ins2 Hin2. rewrite forallb_forall in Hr. specialize (Hr _ Hin2). cbv beta iota in Hr.
     rewrite andb_true_iff, Nat.eqb_eq, Nat.ltb_lt in Hr. exact Hr.
   + left. rewrite andb_true_iff in Hrest. destruct Hrest as [Htc Hf]. apply Pos.eqb_eq in Htc.
     destruct (fetch prog b i) as [s0|] eqn:Hfi; [|discriminate].
     destruct (loop_instr_eq_dec s0 stp) as [->|]; [|discriminate].
     injection Hfw as <-. split; [reflexivity|]. split; [reflexivity|]. split; [exact Htc|]. split; reflexivity.
 - right. left. rewrite !andb_true_iff in Hrest.
   destruct Hrest as [[[[[Hsh Hmov] Htc] Hf] Hkt] Ho].
   assert (Hsh0 : sh = None) by (destruct sh; [discriminate|reflexivity]).
   destruct insw as [| | | | | | | | | | | | | | | | | | | | | | | |]; try discriminate.
   destruct g as [[neg p]|]; [discriminate|]. rewrite andb_true_iff, !Pos.eqb_eq in Hmov.
   destruct Hmov as [-> ->]. apply negb_true_iff, Pos.eqb_neq in Htc.
   destruct (fetch prog b j) as [s0|] eqn:Hfj; [|discriminate].
   destruct (loop_instr_eq_dec s0 stp) as [->|]; [|discriminate].
   assert (Hji : (j < i)%nat) by lia.
   split; [exact Hji|]. split; [exact Hsh0|]. split; [reflexivity|]. split; [exact Htc|]. split; [reflexivity|].
   split; [exact Hkt|]. intros r ->. rewrite andb_true_iff in Ho. destruct Ho as [Hc Hk].
   apply negb_true_iff in Hc. split; assumption.
Qed.

Lemma tc_alias_facts :
 match al with
 | ASame => a = c
 | ABlock m => a <> c /\ term_last_write prog b j a = Some (m, IMov None a c)
 | AHead m => a <> c /\ b = l /\ h <> l /\ term_region_writes prog h l a = [(h, m, IMov None a c)]
 end.
Proof.
 pose proof Hstatic as Hst0.
 unfold term_counter_static in Hst0. cbn [tc_c tc_b tc_i tc_j tc_lo tc_hi tc_step tc_alias] in Hst0.
 rewrite !andb_true_iff in Hst0. destruct Hst0 as [[_ Ha] _].
 destruct al as [|m|m].
 - apply Pos.eqb_eq. exact Ha.
 - rewrite andb_true_iff in Ha. destruct Ha as [Hne Hlw]. apply negb_true_iff, Pos.eqb_neq in Hne.
   split; [exact Hne|].
   destruct (term_last_write prog b j a) as [[m' [| | | | | | | | | | | | | | | | | | | | | | | |]]|]; try discriminate.
   destruct g as [[neg p]|]; [discriminate|]. rewrite !andb_true_iff, Nat.eqb_eq, !Pos.eqb_eq in Hlw.
   destruct Hlw as [[-> ->] ->]. reflexivity.
 - rewrite !andb_true_iff in Ha. destruct Ha as [[[Hne Hbl] Hhl] Hw].
   apply negb_true_iff, Pos.eqb_neq in Hne. apply Nat.eqb_eq in Hbl. apply negb_true_iff, Nat.eqb_neq in Hhl.
   split; [exact Hne|]. split; [exact Hbl|]. split; [exact Hhl|].
   destruct (term_region_writes prog h l a) as [|[[b' m'] [| | | | | | | | | | | | | | | | | | | | | | | |]] [|]];
     try discriminate; try (destruct g as [[? ?]|]; discriminate).
   destruct g as [[neg p]|]; [discriminate|]. rewrite !andb_true_iff, !Nat.eqb_eq, !Pos.eqb_eq in Hw.
   destruct Hw as [[[-> ->] ->] ->]. reflexivity.
Qed.

Lemma tc_guard_facts : exists g, last_instr (block_body prog l) = Some (IBra g h) /\
 forall neg p, g = Some (neg,p) -> is_cell F p = false.
Proof.
 destruct (term_edges_spec prog l h He) as [_ [_ [g Hg]]]. exists g. split; [exact Hg|].
 pose proof Hstatic as Hst0. unfold term_counter_static in Hst0. rewrite !andb_true_iff in Hst0.
 destruct Hst0 as [_ Hgd]. rewrite Hg in Hgd. intros neg p ->. apply negb_true_iff. exact Hgd.
Qed.

Lemma tc_unique : forall b' i' ins', (h <= b' <= l)%nat -> fetch prog b' i' = Some ins' ->
 term_writes c ins' = true -> b' = b /\ i' = i.
Proof.
 intros b' i' ins' Hb Hf Hw. destruct tc_writes as [insw [Hrw _]].
 assert (Hin : In (b',i',ins') (term_region_writes prog h l c)) by (apply term_region_writes_spec; tauto).
 rewrite Hrw in Hin. destruct Hin as [Hin|[]]. injection Hin as -> -> _. split; reflexivity.
Qed.

Lemma tc_write_pos : exists insw, fetch prog b i = Some insw /\ (h <= b <= l)%nat /\ term_writes c insw = true.
Proof.
 destruct tc_writes as [insw [Hrw _]]. exists insw.
 assert (Hin : In (b,i,insw) (term_region_writes prog h l c)) by (rewrite Hrw; left; reflexivity).
 apply term_region_writes_spec in Hin. tauto.
Qed.

(* The back edge's guard decides the update, when it is computed before it. *)
Definition tc_cond (rs : reg -> Z) : Prop :=
 match last_instr (block_body prog l) with
 | Some (IBra (Some (neg,p)) _) =>
   if Nat.eqb b l && Nat.leb (term_guard_from prog l p) i then negb (Z.eqb (rs p) 0) = negb neg else True
 | _ => True end.

Definition tc_alias_win (s : state) : Prop :=
 match al with
 | ASame => True
 | ABlock m => blk s = b -> (m < idx s <= j)%nat -> regs s a = regs s c
 | AHead m => term_pos_lt h m (blk s) (idx s) -> term_pos_le (blk s) (idx s) l j -> regs s a = regs s c
 end.

Definition tc_latch_win (F' : lfacts) (s0 : astate) (tm : nat) (s : state) : Prop :=
 let st := term_latch_start d l in
 match last_instr (block_body prog l) with
 | Some (IBra (Some (neg,p)) _) => let r := Nat.max (term_guard_from prog l p) st in
   ((idx s <= r)%nat -> state_ok F' tm (term_walk F' s0 (term_slice prog l st (idx s))) (regs s)) /\
   ((r <= idx s)%nat -> negb (Z.eqb (regs s p) 0) = negb neg ->
     exists sref, term_refine F' (term_walk F' s0 (term_slice prog l st r)) p (negb neg) = Some sref /\
       state_ok F' tm (term_walk F' sref (term_slice prog l r (idx s))) (regs s))
 | _ => state_ok F' tm (term_walk F' s0 (term_slice prog l st (idx s))) (regs s)
 end.

Record tc_inv (s1 : state) (tm : nat) (s : state) : Prop := {
 ti_head : lo <= extend w sg (regs s1 c) <= hi;
 ti_before : term_pos_le (blk s) (idx s) b i -> regs s c = regs s1 c;
 ti_after : term_pos_lt b i (blk s) (idx s) -> tc_cond (regs s) ->
   (term_rank k w (regs s c) < term_rank k w (regs s1 c))%nat;
 ti_range : b <> l -> (h < blk s)%nat -> exists x y,
   term_counter_hull F prog cert h l d w a = Some (Some (x,y)) /\ x <= extend w sg (regs s c) <= y;
 ti_alias : tc_alias_win s;
 ti_copy : (j < i)%nat -> blk s = b -> (j < idx s <= i)%nat ->
   regs s t = trunc w (binop_eval op w (regs s c) (opv (regs s) o));
 ti_shift : forall m, sh = Some m -> term_pos_lt h m (blk s) (idx s) -> term_pos_le (blk s) (idx s) b i ->
   regs s t = trunc w (binop_eval op w (regs s c) (opv (regs s) o));
 ti_state : blk s = b -> (j <= idx s)%nat -> exists F' st0,
   In (F',st0) (term_cut_states Kb F prog cert b j) /\ term_same F F' /\ env_ok F' E /\ regs_ok F' E (regs s) /\
   is_bottom st0 = false /\ term_wf_from F' prog b j = true /\
   state_ok F' tm (term_walk F' (term_assume_all F' d w a st0) (term_slice prog b j (idx s))) (regs s);
 ti_latch : uniq = true -> blk s = l -> (term_latch_start d l <= idx s)%nat -> exists F' st0,
   In (F',st0) (term_cut_states Kl F prog cert l (term_latch_start d l)) /\ term_same F F' /\ env_ok F' E /\
   regs_ok F' E (regs s) /\ is_bottom st0 = false /\ term_wf_from F' prog l (term_latch_start d l) = true /\
   tc_latch_win F' (term_latch_entry F prog cert h l d w a F' st0) tm s
}.

(* The only back edge to the head from inside the loop leaves block l. *)
Lemma tc_to_head : uniq = true -> forall tm s s' ins, fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E tm s = Some s' -> term_in h l (blk s) = true -> blk s' = h -> idx s' = 0%nat -> blk s = l.
Proof.
 intros Hlatches tm s s' ins Hf Hs Hin Hb Hi.
 pose proof (term_step_head F prog E Hshape l h tm s s' ins Hf Hs Hin Hb Hi) as Hedge.
 assert (Hv : (h <= blk s <= l)%nat) by (apply term_in_iff; exact Hin).
 pose proof (term_latches_in prog h l (blk s) Hedge Hv) as Hl.
 rewrite forallb_forall in Hlatches. apply Nat.eqb_eq. rewrite Nat.eqb_sym. exact (Hlatches _ Hl).
Qed.

Lemma tc_env : env_ok F E.
Proof. exact (env_of_launch F prog E (check_scalar_certificate_lf_ok F prog cert Hcert) HE). Qed.

Lemma tc_w : (1 <= w <= 64)%nat.
Proof. pose proof tc_static as [_ [_ [_ [_ [_ [_ [Hw _]]]]]]]. rewrite <- Hw. exact (term_wd_range F E c tc_env). Qed.

Lemma tc_at : forall tm s, reach tm s -> term_in h l (blk s) = true ->
 (blk s < List.length prog)%nat /\ (idx s < block_len prog (blk s))%nat /\ regs_ok F E (regs s) /\
 exists F' st, In (F',st) (term_states F prog cert (blk s) (idx s)) /\ term_same F F' /\
   env_ok F' E /\ regs_ok F' E (regs s) /\ state_ok F' tm st (regs s).
Proof.
 intros tm s Hr Hin. destruct (term_edges_spec prog l h He) as [Hl _].
 apply (term_in_iff l h) in Hin.
 destruct (term_states_sound F prog cert E tm s Hcert HE Hr ltac:(lia)) as [Hi [F' [st [Hst [Hsame [HE' [Hrs Hok]]]]]]].
 split; [lia|]. split; [exact Hi|]. split; [exact (proj1 (term_same_regs F F' E (regs s) Hsame) Hrs)|].
 exists F',st. tauto.
Qed.

(* The restricted states of block b or l agree with a reachable state there. *)
Lemma tc_cut_at : forall K v tm s, reach tm s -> blk s = v -> (v = b \/ v = l) ->
 exists F' st, In (F',st) (term_cut_states K F prog cert v (idx s)) /\ term_same F F' /\
   env_ok F' E /\ regs_ok F' E (regs s) /\ state_ok F' tm st (regs s).
Proof.
 intros K v tm s Hr Hb Hv. pose proof tc_write_pos as [insw [_ [Hbin _]]].
 destruct (term_edges_spec prog l h He) as [Hlen _].
 assert (Hwf : term_block_wf F prog cert (blk s) = true) by (rewrite Hb; destruct Hv as [->| ->]; assumption).
 destruct (term_cut_sound K F prog cert E Hcert HE (term_shape_structured prog Hshape) (idx s) tm s Hr eq_refl
   ltac:(rewrite Hb; destruct Hv as [->| ->]; lia) Hwf) as [F' [st0 [Hst [_ [Hsame [HE' [Hrs Hok]]]]]]].
 rewrite Hb in Hst, Hok.
 exists F', (term_walk F' (term_cut F' K st0) (term_slice prog v 0 (idx s))). split.
 - unfold term_cut_states. apply in_map_iff. exists (F',st0). split; [reflexivity|exact Hst].
 - tauto.
Qed.

Lemma tc_assume_all_sound : forall F' tm st rs, term_same F F' -> env_ok F' E -> regs_ok F' E rs ->
 state_ok F' tm st rs -> lo <= extend w sg (rs c) <= hi -> rs a = rs c ->
 state_ok F' tm (term_assume_all F' d w a st) rs.
Proof.
 intros F' tm st rs Hsame HE' Hrs Hst Hc Ha. pose proof tc_w as Hw.
 pose proof tc_static as [_ [_ [_ [_ [_ [_ [Hwc [Hwa _]]]]]]]].
 unfold term_assume_all. cbn [tc_kind tc_c tc_lo tc_hi].
 assert (H1 : state_ok F' tm (term_assume F' w sg c lo hi st) rs).
 { apply (term_assume_sound F' E tm st rs w sg c lo hi HE' Hrs Hst); [rewrite (term_same_wdn F F' c Hsame); assumption|exact Hw|exact Hc]. }
 destruct (Pos.eqb a c); [exact H1|].
 apply (term_assume_sound F' E tm _ rs w sg a lo hi HE' Hrs H1); [rewrite (term_same_wdn F F' a Hsame); assumption|exact Hw|].
 rewrite Ha. exact Hc.
Qed.

(* The step's window opens at index j of block b. *)
Lemma tc_state_entry : forall tm s, reach tm s -> blk s = b -> idx s = j ->
 lo <= extend w sg (regs s c) <= hi -> regs s a = regs s c ->
 exists F' st0, In (F',st0) (term_cut_states Kb F prog cert b j) /\ term_same F F' /\ env_ok F' E /\
   regs_ok F' E (regs s) /\ is_bottom st0 = false /\ term_wf_from F' prog b j = true /\
   state_ok F' tm (term_walk F' (term_assume_all F' d w a st0) (term_slice prog b j (idx s))) (regs s).
Proof.
 intros tm s Hr Hb Hj Hc Ha. pose proof tc_static as [Hbhl _].
 destruct (tc_cut_at Kb b tm s Hr Hb (or_introl eq_refl)) as [F' [st [Hst [Hsame [HE' [Hrs Hok]]]]]].
 rewrite Hj in Hst. exists F',st.
 assert (Hnb : is_bottom st = false).
 { destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm st _ Hbot Hok)|reflexivity]. }
 pose proof (term_short_all_in _ _ _ _ Hstepok Hst) as Helt. cbv beta iota in Helt.
 cbn [tc_b tc_j tc_i tc_c tc_kind tc_lo tc_hi tc_alias tc_step] in Helt. rewrite Hnb in Helt.
 destruct (term_wf_from F' prog b j) eqn:Hwf; [|discriminate].
 split; [exact Hst|]. split; [exact Hsame|]. split; [exact HE'|]. split; [exact Hrs|].
 split; [exact Hnb|]. split; [reflexivity|].
 rewrite Hj, term_slice_empty. cbn [walk]. exact (tc_assume_all_sound F' tm st (regs s) Hsame HE' Hrs Hok Hc Ha).
Qed.

Lemma tc_latch_check : uniq = true -> term_counter_hull F prog cert h l d w a <> Some None ->
 term_counter_latch F prog cert h l d w a = true.
Proof.
 intros Hu Hn. pose proof (tc_back_u Hu) as Hbk. unfold term_counter_back in Hbk. cbn [tc_b] in Hbk.
 destruct (Nat.eqb b l); [exact Hbk|].
 destruct (term_counter_hull F prog cert h l d w a) as [[[? ?]|]|]; [exact Hbk|contradiction|discriminate].
Qed.

(* The latch's window opens at its start index. *)
Lemma tc_latch_entry : uniq = true -> forall tm s, reach tm s -> blk s = l -> idx s = term_latch_start d l ->
 (b = l -> lo <= extend w sg (regs s c) <= hi /\ regs s a = regs s c) ->
 (b <> l -> exists x y, term_counter_hull F prog cert h l d w a = Some (Some (x,y)) /\
   x <= extend w sg (regs s c) <= y) ->
 exists F' st0, In (F',st0) (term_cut_states Kl F prog cert l (term_latch_start d l)) /\ term_same F F' /\ env_ok F' E /\
   regs_ok F' E (regs s) /\ is_bottom st0 = false /\ term_wf_from F' prog l (term_latch_start d l) = true /\
   tc_latch_win F' (term_latch_entry F prog cert h l d w a F' st0) tm s.
Proof.
 intros Hu tm s Hr Hb Hi Hl Hh. destruct (term_edges_spec prog l h He) as [Hlen [Hhl _]].
 destruct (tc_cut_at Kl l tm s Hr Hb (or_intror eq_refl)) as [F' [st [Hst [Hsame [HE' [Hrs Hok]]]]]].
 rewrite Hi in Hst. exists F',st.
 assert (Hnb : is_bottom st = false).
 { destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm st _ Hbot Hok)|reflexivity]. }
 assert (Hentry0 : state_ok F' tm (term_latch_entry F prog cert h l d w a F' st) (regs s) /\
   term_counter_latch F prog cert h l d w a = true).
 { unfold term_latch_entry, term_latch_entry_with. cbn [tc_b tc_kind tc_c]. destruct (Nat.eqb_spec b l) as [Hbl|Hbl].
   - destruct (Hl Hbl) as [Hc Ha]. split; [exact (tc_assume_all_sound F' tm st (regs s) Hsame HE' Hrs Hok Hc Ha)|].
     pose proof (tc_back_u Hu) as Hbk. unfold term_counter_back in Hbk. cbn [tc_b] in Hbk.
     rewrite (proj2 (Nat.eqb_eq b l) Hbl) in Hbk. exact Hbk.
   - destruct (Hh Hbl) as [x [y [Hhull Hxy]]]. rewrite Hhull. split.
     + apply (term_assume_sound F' E tm st (regs s) w sg c x y HE' Hrs Hok); [|exact tc_w|exact Hxy].
       rewrite (term_same_wdn F F' c Hsame). exact (proj1 (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 tc_static))))))).
     + apply (tc_latch_check Hu). rewrite Hhull. discriminate. }
 destruct Hentry0 as [Hs0 Hlatch].
 pose proof (term_short_all_in _ _ _ _ Hlatch Hst) as Helt. cbv beta iota in Helt.
 cbn [tc_b tc_j tc_i tc_c tc_kind tc_lo tc_hi tc_alias tc_step] in Helt. rewrite Hnb in Helt.
 destruct (term_wf_from F' prog l (term_latch_start d l)) eqn:Hwf; [|discriminate].
 split; [exact Hst|]. split; [exact Hsame|]. split; [exact HE'|]. split; [exact Hrs|].
 split; [exact Hnb|]. split; [reflexivity|].
 unfold tc_latch_win. rewrite Hi, term_slice_empty. cbn [walk].
 destruct (last_instr (block_body prog l)) as [[| | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|];
   try exact Hs0.
 split; [intros _; exact Hs0|].
 intros Hle Htk. assert (Hr0 : Nat.max (term_guard_from prog l p) (term_latch_start d l) = term_latch_start d l) by lia.
 rewrite Hr0, term_slice_empty. cbn [walk].
 destruct (term_refine_sound F' E tm (regs s) _ p (negb neg) HE' Hrs Hs0 Htk) as [sref [Href Hsref]].
 exists sref. split; [exact Href|exact Hsref].
Qed.

(* At the step's index, the copy holds the counter's value. *)
Lemma tc_alias_at_j : forall x, tc_alias_win x -> blk x = b -> idx x = j -> regs x a = regs x c.
Proof.
 intros x Hw Hb Hj. pose proof tc_alias_facts as Hal. unfold tc_alias_win in Hw.
 remember al as alv eqn:Halv in Hal, Hw. destruct alv as [|m|m].
 - rewrite Hal. reflexivity.
 - destruct Hal as [_ Hlw]. destruct (term_last_write_some prog b j a m _ Hlw) as [Hmj _].
   apply Hw; [exact Hb|lia].
 - destruct Hal as [_ [Hbl [Hhl _]]]. destruct (term_edges_spec prog l h He) as [_ [Hle _]].
   apply Hw; unfold term_pos_lt, term_pos_le; lia.
Qed.

Section Step.
Variables (s1 s s' : state) (tm : nat) (ins : instr).
Hypothesis Hinv : tc_inv s1 tm s.
Hypothesis Hr : reach tm s.
Hypothesis Hin : term_in h l (blk s) = true.
Hypothesis Hf : fetch prog (blk s) (idx s) = Some ins.
Hypothesis Hs : phase_step F prog E tm s = Some s'.
Hypothesis Hin' : term_in h l (blk s') = true.
Hypothesis Hnh : ~ (blk s' = h /\ idx s' = 0%nat).

Lemma tcs_reach : reach (S tm) s'.
Proof. rewrite term_run_succ, Hr. exact Hs. Qed.

Lemma tcs_not_write : ~ (blk s = b /\ idx s = i) -> term_writes c ins = false.
Proof.
 intros Hne. destruct (term_writes c ins) eqn:Hw; [|reflexivity]. exfalso. apply Hne.
 apply (tc_unique (blk s) (idx s) ins); [apply term_in_iff; exact Hin|exact Hf|exact Hw].
Qed.

Lemma tcs_keep : forall r, is_cell F r = false -> term_writes r ins = false -> regs s' r = regs s r.
Proof. intros r Hc Hw. exact (term_step_keeps F prog E tm s s' ins r Hc Hf Hs Hw). Qed.

Lemma tcs_keep_c : ~ (blk s = b /\ idx s = i) -> regs s' c = regs s c.
Proof.
 intros Hne. apply tcs_keep; [exact (proj1 (proj2 (proj2 (proj2 tc_static))))|exact (tcs_not_write Hne)].
Qed.

Lemma tcs_order : forall pb pi, (pb = h \/ pb = l) -> (pi < block_len prog pb)%nat ->
 (term_pos_le (blk s') (idx s') pb pi -> term_pos_lt (blk s) (idx s) pb pi) /\
 (term_pos_lt pb pi (blk s') (idx s') -> term_pos_le pb pi (blk s) (idx s)).
Proof. intros pb pi Hpb Hpi. exact (term_step_order F prog E l h Hshape He tm s s' ins pb pi Hf Hs Hin Hin' Hnh Hpb Hpi). Qed.

Lemma tcs_bi : (term_pos_le (blk s') (idx s') b i -> term_pos_lt (blk s) (idx s) b i) /\
 (term_pos_lt b i (blk s') (idx s') -> term_pos_le b i (blk s) (idx s)).
Proof.
 pose proof tc_write_pos as [insw [Hfw _]].
 apply tcs_order; [exact (proj1 tc_static)|].
 unfold fetch, block_len in *. apply nth_error_Some. rewrite Hfw. discriminate.
Qed.

Lemma tcs_before : term_pos_le (blk s') (idx s') b i -> regs s' c = regs s1 c.
Proof.
 intros Hle. pose proof (proj1 tcs_bi Hle) as Hlt.
 rewrite tcs_keep_c; [|unfold term_pos_lt in Hlt; lia].
 apply (ti_before _ _ _ Hinv). unfold term_pos_le, term_pos_lt in *. lia.
Qed.

Lemma tcs_prev : (0 < idx s')%nat -> blk s' = blk s /\ idx s' = S (idx s).
Proof. intros H. exact (term_step_prev F prog E Hshape tm s s' ins Hf Hs H). Qed.

Lemma tcs_regs : regs_ok F E (regs s).
Proof. exact (proj1 (proj2 (proj2 (tc_at tm s Hr Hin)))). Qed.

Lemma tcs_alias : tc_alias_win s'.
Proof.
 pose proof tc_alias_facts as Hal. pose proof (ti_alias _ _ _ Hinv) as Hwin.
 pose proof tc_static as [_ [Hji [_ [Hcc [Hca [_ [Hwc [Hwa _]]]]]]]].
 pose proof tc_write_pos as [insw [Hfw [Hbin _]]].
 assert (Hmov : forall a' c', ins = IMov None a' c' -> wdn F a' = w -> wdn F c' = w -> regs s' a' = regs s c').
 { intros a' c' -> Hw1 Hw2. rewrite (term_step_mov F prog E tm s s' a' c' Hf Hs).
   rewrite Hw1. apply trunc_small. pose proof (rs_range F E (regs s) tcs_regs c') as Hc'. rewrite Hw2 in Hc'. exact Hc'. }
 unfold tc_alias_win in *. remember al as alv eqn:Halv in Hal, Hwin |- *. destruct alv as [|m|m]; [exact I| |].
 - destruct Hal as [Hne Hlw]. intros Hb' Hm.
   destruct (term_last_write_some prog b j a m _ Hlw) as [Hmj [Hfm [_ Hnone]]].
   destruct (tcs_prev ltac:(lia)) as [Hbb Hii].
   destruct (Nat.eq_dec (idx s) m) as [Him|Him].
   + assert (Hins : ins = IMov None a c).
     { assert (Hbs : blk s = b) by congruence. rewrite Hbs, Him, Hfm in Hf. injection Hf as <-. reflexivity. }
     rewrite (Hmov a c Hins Hwa Hwc). symmetry. apply tcs_keep; [exact Hcc|].
     rewrite Hins. unfold term_writes. cbn. rewrite (proj2 (Pos.eqb_neq c a) (fun H => Hne (eq_sym H))). reflexivity.
   + assert (Hbs : blk s = b) by congruence.
     rewrite (tcs_keep a Hca), (tcs_keep c Hcc).
     * apply Hwin; [exact Hbs|lia].
     * apply tcs_not_write. intros [_ Hi']. lia.
     * apply (Hnone (idx s) ins); [lia|]. rewrite <- Hbs. exact Hf.
 - destruct Hal as [Hne [Hbl [Hhl Hrw]]]. intros Hlt Hle.
   assert (Hfm : fetch prog h m = Some (IMov None a c)).
   { assert (Hin0 : In (h,m,IMov None a c) (term_region_writes prog h l a)) by (rewrite Hrw; left; reflexivity).
     apply term_region_writes_spec in Hin0. tauto. }
   assert (Hmlen : (m < block_len prog h)%nat).
   { unfold fetch, block_len in *. apply nth_error_Some. rewrite Hfm. discriminate. }
   assert (Hjlen : (j < block_len prog l)%nat).
   { rewrite <- Hbl. assert (Hil : (i < block_len prog b)%nat).
     { unfold fetch, block_len in *. apply nth_error_Some. rewrite Hfw. discriminate. }
     pose proof (proj1 (proj2 tc_static)). lia. }
   pose proof (proj2 (tcs_order h m (or_introl eq_refl) Hmlen) Hlt) as Hle_s.
   pose proof (proj1 (tcs_order l j (or_intror eq_refl) Hjlen) Hle) as Hlt_s.
   assert (Hcw : regs s' c = regs s c).
   { apply tcs_keep_c. intros [Hb1 Hi1]. unfold term_pos_lt in Hlt_s. rewrite Hbl in Hb1. lia. }
   destruct (Nat.eq_dec (blk s) h) as [Hbh|Hbh]; [destruct (Nat.eq_dec (idx s) m) as [Him|Him]|].
   + assert (Hins : ins = IMov None a c) by (rewrite Hbh, Him, Hfm in Hf; injection Hf as <-; reflexivity).
     rewrite Hcw. exact (Hmov a c Hins Hwa Hwc).
   + rewrite Hcw, (tcs_keep a Hca).
     * apply Hwin; unfold term_pos_lt, term_pos_le in *; lia.
     * destruct (term_writes a ins) eqn:Hw; [|reflexivity]. exfalso.
       assert (Hin0 : In (blk s, idx s, ins) (term_region_writes prog h l a)).
       { apply term_region_writes_spec. split; [apply term_in_iff; exact Hin|]. split; [exact Hf|exact Hw]. }
       rewrite Hrw in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as Hb0 Hi0 _. exact (Him (eq_sym Hi0)).
   + rewrite Hcw, (tcs_keep a Hca).
     * apply Hwin; unfold term_pos_lt, term_pos_le in *; lia.
     * destruct (term_writes a ins) eqn:Hw; [|reflexivity]. exfalso.
       assert (Hin0 : In (blk s, idx s, ins) (term_region_writes prog h l a)).
       { apply term_region_writes_spec. split; [apply term_in_iff; exact Hin|]. split; [exact Hf|exact Hw]. }
       rewrite Hrw in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as Hb0 _ _. exact (Hbh (eq_sym Hb0)).
Qed.

Lemma term_opv_keep : forall (rs rs' : reg -> Z) (o' : GateIsa.opnd),
 (forall r, o' = OReg r -> rs' r = rs r) -> opv rs' o' = opv rs o'.
Proof. intros rs rs' [r|z] H; cbn [opv]; [exact (H r eq_refl)|reflexivity]. Qed.

Lemma tcs_copy : (j < i)%nat -> blk s' = b -> (j < idx s' <= i)%nat ->
 regs s' t = trunc w (binop_eval op w (regs s' c) (opv (regs s') o)).
Proof.
 intros Hji Hb' Hj'. pose proof tc_static as [_ [_ [_ [Hcc [_ [Hct [Hwc [_ Hwt]]]]]]]].
 destruct tc_writes as [insw [_ [_ [[Hji' _]|[[_ [_ [_ [Htc [Hfj [Hkt Hko]]]]]]|[m0 [Hji' _]]]]]]]; [lia| |lia].
 destruct (tcs_prev ltac:(lia)) as [Hbb Hii]. assert (Hbs : blk s = b) by congruence.
 assert (Ho' : opv (regs s') o = opv (regs s) o).
 { apply term_opv_keep. intros r Hr0. destruct (Hko r Hr0) as [Hrc Hkr].
   apply tcs_keep; [exact Hrc|]. apply (term_kept_spec prog b j i r (idx s) ins Hkr); [lia|rewrite <- Hbs; exact Hf]. }
 rewrite Ho'.
 destruct (Nat.eq_dec (idx s) j) as [Hij|Hij].
 - assert (Hins : ins = stp) by (rewrite Hbs, Hij, Hfj in Hf; injection Hf as <-; reflexivity).
   rewrite (term_step_bin F prog E tm s s' op w t a o (eq_trans Hf (f_equal Some Hins)) Hs), Hwt.
   rewrite (tc_alias_at_j s (ti_alias _ _ _ Hinv) Hbs Hij).
   rewrite tcs_keep_c; [reflexivity|]. intros [_ Hi']. lia.
 - rewrite (tcs_keep t Hct), tcs_keep_c.
   + rewrite <- Ho'. pose proof (ti_copy _ _ _ Hinv Hji Hbs ltac:(lia)) as H. rewrite H. rewrite Ho'. reflexivity.
   + intros [_ Hi']. lia.
   + apply (term_kept_spec prog b (S j) i t (idx s) ins Hkt); [lia|rewrite <- Hbs; exact Hf].
Qed.

Lemma tcs_shift : forall m, sh = Some m -> term_pos_lt h m (blk s') (idx s') -> term_pos_le (blk s') (idx s') b i ->
 regs s' t = trunc w (binop_eval op w (regs s' c) (opv (regs s') o)).
Proof.
 intros m Hsh Hlt Hle. pose proof tc_static as [_ [_ [_ [Hcc [_ [Hct [Hwc [_ Hwt]]]]]]]].
 destruct tc_writes as [insw [_ [Hfi [[_ [Hsh' _]]|[[_ [Hsh' _]]|[m0 [Hj [Hsh' [Hins [Htc [Hac [Hbl [Hhl [Hwf0 [Hrt Ho]]]]]]]]]]]]]]];
   [rewrite Hsh in Hsh'; discriminate|rewrite Hsh in Hsh'; discriminate|].
 rewrite Hsh in Hsh'. injection Hsh' as <-.
 destruct (term_edges_spec prog l h He) as [_ [Hhl' _]].
 assert (Hfm : fetch prog h m = Some stp).
 { assert (Hin0 : In (h,m,stp) (term_region_writes prog h l t)) by (rewrite Hrt; left; reflexivity).
   apply term_region_writes_spec in Hin0. tauto. }
 assert (Hmlen : (m < block_len prog h)%nat).
 { unfold fetch, block_len in *. apply nth_error_Some. rewrite Hfm. discriminate. }
 pose proof (proj2 (tcs_order h m (or_introl eq_refl) Hmlen) Hlt) as Hle_s.
 pose proof (proj1 tcs_bi Hle) as Hlt_s.
 assert (Hcw : regs s' c = regs s c) by (apply tcs_keep_c; intros [Hb1 Hi1]; unfold term_pos_lt in Hlt_s; lia).
 assert (Ho' : opv (regs s') o = opv (regs s) o).
 { apply term_opv_keep. intros r Hr0. destruct (Ho r Hr0) as [Hrc Hrw].
   apply tcs_keep; [exact Hrc|].
   destruct (term_writes r ins) eqn:Hw; [|reflexivity]. exfalso.
   assert (Hin0 : In (blk s, idx s, ins) (term_region_writes prog h l r)).
   { apply term_region_writes_spec. split; [apply term_in_iff; exact Hin|]. split; [exact Hf|exact Hw]. }
   destruct (Hrw _ _ _ Hin0) as [Hb2 Hm2]. unfold term_pos_le in Hle_s. lia. }
 rewrite Hcw, Ho'.
 destruct (Nat.eq_dec (blk s) h) as [Hbh|Hbh]; [destruct (Nat.eq_dec (idx s) m) as [Him|Him]|].
 - assert (Hins' : ins = stp) by (rewrite Hbh, Him, Hfm in Hf; injection Hf as <-; reflexivity).
   rewrite (term_step_bin F prog E tm s s' op w t a o (eq_trans Hf (f_equal Some Hins')) Hs), Hwt, Hac. reflexivity.
 - rewrite (tcs_keep t Hct).
   + apply (ti_shift _ _ _ Hinv m Hsh); unfold term_pos_lt, term_pos_le in *; lia.
   + destruct (term_writes t ins) eqn:Hw; [|reflexivity]. exfalso.
     assert (Hin0 : In (blk s, idx s, ins) (term_region_writes prog h l t)).
     { apply term_region_writes_spec. split; [apply term_in_iff; exact Hin|]. split; [exact Hf|exact Hw]. }
     rewrite Hrt in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as _ Hi0 _. exact (Him (eq_sym Hi0)).
 - rewrite (tcs_keep t Hct).
   + apply (ti_shift _ _ _ Hinv m Hsh); unfold term_pos_lt, term_pos_le in *; lia.
   + destruct (term_writes t ins) eqn:Hw; [|reflexivity]. exfalso.
     assert (Hin0 : In (blk s, idx s, ins) (term_region_writes prog h l t)).
     { apply term_region_writes_spec. split; [apply term_in_iff; exact Hin|]. split; [exact Hf|exact Hw]. }
     rewrite Hrt in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as Hb0 _ _. exact (Hbh (eq_sym Hb0)).
Qed.

Lemma tcs_state : blk s' = b -> (j <= idx s')%nat -> exists F' st0,
   In (F',st0) (term_cut_states Kb F prog cert b j) /\ term_same F F' /\ env_ok F' E /\ regs_ok F' E (regs s') /\
   is_bottom st0 = false /\ term_wf_from F' prog b j = true /\
   state_ok F' (S tm) (term_walk F' (term_assume_all F' d w a st0) (term_slice prog b j (idx s'))) (regs s').
Proof.
 intros Hb' Hj'. pose proof tc_static as [_ [Hji _]]. destruct (Nat.eq_dec (idx s') j) as [Hij|Hij].
 - apply (tc_state_entry (S tm) s' tcs_reach Hb' Hij).
   + rewrite tcs_before; [exact (ti_head _ _ _ Hinv)|unfold term_pos_le; lia].
   + exact (tc_alias_at_j s' tcs_alias Hb' Hij).
 - destruct (tcs_prev ltac:(lia)) as [Hbb Hii]. assert (Hbs : blk s = b) by congruence.
   destruct (ti_state _ _ _ Hinv Hbs ltac:(lia)) as [F' [st0 [Hst [Hsame [HE' [Hrs [Hnb [Hwf Hok]]]]]]]].
   exists F',st0. split; [exact Hst|]. split; [exact Hsame|]. split; [exact HE'|].
   assert (Hs' : phase_step F' prog E tm s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
   assert (Hwfi : wf_instr F' ins = true) by (apply (term_wf_at F' prog b j (idx s) ins Hwf); [lia|rewrite <- Hbs; exact Hf]).
   destruct (term_xfer_sound F' prog E tm s s' _ ins HE' Hf Hs' Hok Hrs Hwfi) as [Hok' Hrs'].
   split; [exact Hrs'|]. split; [exact Hnb|]. split; [exact Hwf|].
   rewrite Hii, (term_twalk_step F' prog b j (idx s) ins); [exact Hok'|lia|rewrite <- Hbs; exact Hf].
Qed.

Lemma tc_guard_keep : forall p0 neg0 tg0 z ins',
 last_instr (block_body prog l) = Some (IBra (Some (neg0,p0)) tg0) ->
 (term_guard_from prog l p0 <= z)%nat -> fetch prog l z = Some ins' -> term_writes p0 ins' = false.
Proof.
 intros p0 neg0 tg0 z ins' Hlast Hz Hfz. unfold term_guard_from in Hz.
 assert (Hzl : (z < block_len prog l)%nat).
 { unfold fetch, block_len in *. apply nth_error_Some. rewrite Hfz. discriminate. }
 destruct (term_last_write prog l (block_len prog l) p0) as [[k0 insk]|] eqn:Hlw.
 - destruct (term_last_write_some prog l _ p0 k0 insk Hlw) as [_ [_ [_ Hno]]].
   apply (Hno z ins'); [lia|exact Hfz].
 - exact (term_last_write_none prog l _ p0 z ins' Hlw Hzl Hfz).
Qed.

(* The update's step: the new counter value and its rank. *)
Lemma tcs_new : blk s = b -> idx s = i ->
 regs s' c = trunc w (binop_eval op w (regs s c) (opv (regs s) o)).
Proof.
 intros Hbs His. pose proof tc_static as [Hbhl [Hji [_ [Hcc [Hca [Hct [Hwc [Hwa Hwt]]]]]]]].
 destruct tc_writes as [insw [_ [Hfi [[Hj [_ [Htc [_ Hins]]]]|[[Hj [_ [Hins [Htc [Hfj [_ _]]]]]]|[m0 [Hj [Hsh [Hins [Htc [Hac [Hbl [Hhl _]]]]]]]]]]]]].
 - pose proof (term_step_bin F prog E tm s s' op w t a o ltac:(rewrite Hbs, His, Hfi, Hins; reflexivity) Hs) as Hv.
   rewrite Htc in Hv. rewrite Hv, Hwc. rewrite (tc_alias_at_j s (ti_alias _ _ _ Hinv) Hbs ltac:(lia)). reflexivity.
 - pose proof (term_step_mov F prog E tm s s' c t ltac:(rewrite Hbs, His, Hfi, Hins; reflexivity) Hs) as Hv.
   rewrite Hv, (ti_copy _ _ _ Hinv Hj Hbs ltac:(lia)), Hwc. apply trunc_small. apply trunc_range.
 - pose proof (term_step_mov F prog E tm s s' c t ltac:(rewrite Hbs, His, Hfi, Hins; reflexivity) Hs) as Hv.
   destruct (term_edges_spec prog l h He) as [_ [Hhl' _]].
   rewrite Hv, (ti_shift _ _ _ Hinv m0 Hsh); [|unfold term_pos_lt; lia|unfold term_pos_le; lia].
   rewrite Hwc. apply trunc_small. apply trunc_range.
Qed.

Lemma tcs_focus : blk s = b -> idx s = i -> tc_cond (regs s') ->
 forall F' Q, term_same F F' -> env_ok F' E -> regs_ok F' E (regs s) -> state_ok F' tm Q (regs s) ->
 state_ok F' tm (term_step_focus F' prog l d w a Q) (regs s).
Proof.
 intros Hbs His Hcond F' Q Hsame HE' Hrs HQ. pose proof tc_static as [Hbhl [Hji [_ [Hcc [Hca [Hct [Hwc [Hwa Hwt]]]]]]]].
 pose proof tc_guard_facts as Hgf. pose proof tc_guard_keep as Hgk. pose proof tcs_keep as Hkeep.
 pose proof tc_w as Hw. unfold term_step_focus. cbn [tc_j tc_i tc_c tc_kind tc_b tc_step].
 destruct (Nat.eqb_spec j i) as [Hj|Hj]; [|exact HQ]. cbn [negb].
 assert (Hwc' : wdn F' c = w) by (rewrite (term_same_wdn F F' c Hsame); exact Hwc).
 assert (Hwa' : wdn F' a = w) by (rewrite (term_same_wdn F F' a Hsame); exact Hwa).
 assert (H1 : state_ok F' tm (if Pos.eqb a c then Q else
   match cmp_range F' Q (OReg a) (Z.of_nat w) sg with
   | Some (x,y) => term_assume F' w sg c x y Q | None => Q end) (regs s)).
 { destruct (Pos.eqb a c); [exact HQ|].
   destruct (cmp_range F' Q (OReg a) (Z.of_nat w) sg) as [[x y]|] eqn:Hcr; [|exact HQ].
   pose proof (term_cmp_reg F' E tm (regs s) Q a w sg x y HE' Hrs HQ Hwa' Hcr) as Hxy.
   rewrite (tc_alias_at_j s (ti_alias _ _ _ Hinv) Hbs ltac:(lia)) in Hxy.
   exact (term_assume_sound F' E tm Q (regs s) w sg c x y HE' Hrs HQ Hwc' Hw Hxy). }
 destruct o as [r|z] eqn:Ho; [|exact H1].
 destruct (Nat.eqb b l && Nat.eqb (wdn F' r) w && negb (is_cell F' r)) eqn:Hbl; [|exact H1].
 rewrite !andb_true_iff, !Nat.eqb_eq, negb_true_iff in Hbl. destruct Hbl as [[Hbl Hwr] Hcr].
 destruct (term_guard_sel prog l i r) as [v|] eqn:Hsel; [|exact H1].
 unfold term_guard_sel in Hsel.
 destruct (last_instr (block_body prog l)) as [[| | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|] eqn:Hlast;
   try discriminate.
 destruct (term_last_write prog l i r) as [[m [| | | | | | | | | | |g0 w0 d0 a0 b0 p0| | | | | | | | | | | | |]]|] eqn:Hlw;
   try discriminate.
 destruct g0 as [[? ?]|]; [discriminate|]. destruct a0 as [?|x1]; [discriminate|]. destruct b0 as [?|x2]; [discriminate|].
 destruct (Pos.eqb p0 p && negb (Pos.eqb r p) && Nat.leb (term_guard_from prog l p) m) eqn:Hc; [|discriminate].
 injection Hsel as <-. rewrite !andb_true_iff, Pos.eqb_eq, negb_true_iff, Pos.eqb_neq, Nat.leb_le in Hc.
 destruct Hc as [[-> Hrp] Hgm].
 destruct (term_last_write_some prog l i r m _ Hlw) as [Hmi [Hfm [Hwm Hno]]].
 assert (Hrr : r = d0).
 { unfold term_writes in Hwm. cbn [loop_writes dest existsb] in Hwm. rewrite orb_false_r in Hwm. apply Pos.eqb_eq in Hwm. exact Hwm. }
 subst d0.
 destruct Hgf as [g [Hg Hgc]]. injection Hg as Hg _.
 pose proof (Hgc neg p (eq_sym Hg)) as Hcp.
 assert (Hcr0 : is_cell F r = false) by (rewrite <- (term_same_cell F F' r Hsame); exact Hcr).
 assert (Hval := term_sel_value F prog E tm s l m i r w0 x1 x2 p (term_shape_structured prog Hshape) Hr
   ltac:(congruence) ltac:(lia) Hfm Hrp Hcr0 Hcp
   (fun k0 ins0 Hk0 Hf0 => conj (Hno k0 ins0 Hk0 Hf0) (Hgk p neg tg k0 ins0 eq_refl ltac:(lia) Hf0))).
 assert (Hp : regs s' p = regs s p).
 { apply Hkeep; [exact Hcp|]. apply (Hgk p neg tg i ins eq_refl); [lia|].
   rewrite <- Hbl, <- Hbs, <- His. exact Hf. }
 unfold tc_cond in Hcond. rewrite Hlast, (proj2 (Nat.eqb_eq b l) Hbl), (proj2 (Nat.leb_le (term_guard_from prog l p) i) ltac:(lia)) in Hcond.
 cbn [andb] in Hcond. rewrite Hp in Hcond.
 assert (Hrv : regs s r = trunc w (if neg then x2 else x1)).
 { rewrite Hval, <- (term_same_wdn F F' r Hsame), Hwr.
   destruct neg, (Z.eqb (regs s p) 0); cbn in Hcond |- *; congruence. }
 apply (term_assume_sound F' E tm _ (regs s) w true r); [exact HE'|exact Hrs|exact H1|exact Hwr|exact Hw|].
 rewrite Hrv. cbn [extend]. rewrite term_sext_trunc. lia.
Qed.

Lemma tcs_write : blk s = b -> idx s = i ->
 tc_cond (regs s') -> (term_rank k w (regs s' c) < term_rank k w (regs s1 c))%nat.
Proof.
 intros Hbs His. pose proof tc_static as [Hbhl [Hji [_ [Hcc [Hca [Hct [Hwc [Hwa Hwt]]]]]]]].
 pose proof tc_w as Hw.
 assert (Hc0 : regs s c = regs s1 c) by (apply (ti_before _ _ _ Hinv); unfold term_pos_le; lia).
 pose proof (tcs_new Hbs His) as Hnew.
 destruct (ti_state _ _ _ Hinv Hbs ltac:(lia)) as [F' [st0 [Hst [Hsame [HE' [Hrs [Hnb [Hwf Hok]]]]]]]].
 rewrite His in Hok.
 assert (Hwfs : wf_instr F' stp = true).
 { destruct tc_writes as [insw [_ [Hfi [[Hj [_ [_ [_ Hins]]]]|[[_ [_ [_ [_ [Hfj _]]]]]|[m0 [_ [_ [_ [_ [_ [_ [_ [Hwf0 _]]]]]]]]]]]]]].
   - apply (term_wf_at F' prog b j i stp Hwf); [lia|rewrite Hfi, Hins; reflexivity].
   - apply (term_wf_at F' prog b j j stp Hwf); [lia|exact Hfj].
   - rewrite (term_same_wf F F' stp Hsame). exact Hwf0. }
 assert (Hwc' : wdn F' c = w) by (rewrite (term_same_wdn F F' c Hsame); exact Hwc).
 intros Hcond.
   pose proof (term_short_all_in _ _ _ _ Hstepok Hst) as Helt. cbv beta iota in Helt.
   cbn [tc_b tc_j tc_i tc_c tc_kind tc_lo tc_hi tc_alias tc_step] in Helt.
   rewrite Hnb, Hwf in Helt. cbn [negb] in Helt.
   set (si := term_walk F' (term_assume_all F' d w a st0) (term_slice prog b j i)) in *.
   assert (HQ : exists Q, (if Nat.eqb b l then term_guard_at F' prog l i si else Some si) = Some Q /\
     state_ok F' tm Q (regs s)).
   { destruct (Nat.eqb_spec b l) as [Hbl|Hbl]; [|exists si; split; [reflexivity|exact Hok]].
     unfold term_guard_at. unfold tc_cond in Hcond.
     destruct (last_instr (block_body prog l)) as [[| | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|] eqn:Hlast;
       try (exists si; split; [reflexivity|exact Hok]).
     rewrite (proj2 (Nat.eqb_eq b l) Hbl) in Hcond. cbn [andb] in Hcond.
     destruct (Nat.leb_spec (term_guard_from prog l p) i) as [Hle|Hgt]; [|exists si; split; [reflexivity|exact Hok]].
     assert (Hp : regs s' p = regs s p).
     { destruct tc_guard_facts as [g [Hg Hgc]]. rewrite Hlast in Hg. injection Hg as Hg _.
       apply tcs_keep; [exact (Hgc neg p (eq_sym Hg))|].
       apply (tc_guard_keep p neg tg i ins Hlast Hle). rewrite <- Hbl, <- Hbs, <- His. exact Hf. }
     rewrite Hp in Hcond.
     destruct (term_refine_sound F' E tm (regs s) si p (negb neg) HE' Hrs Hok Hcond) as [Q [HQ HQok]].
     exists Q. split; [exact HQ|exact HQok]. }
   destruct HQ as [Q [HQ HQok]]. rewrite HQ in Helt.
   assert (HQnb : is_bottom Q = false).
   { destruct (is_bottom Q) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm Q _ Hbot HQok)|reflexivity]. }
   rewrite HQnb in Helt.
   pose proof (tcs_focus Hbs His Hcond F' Q Hsame HE' Hrs HQok) as HFok.
   set (Q' := term_step_focus F' prog l d w a Q) in *.
   destruct (cmp_range F' Q' (OReg c) (Z.of_nat w) sg) as [[x y]|] eqn:Hcr; [|discriminate].
   pose proof (term_cmp_reg F' E tm (regs s) Q' c w sg x y HE' Hrs HFok Hwc' Hcr) as Hxy.
   pose proof (rs_range F' E (regs s) Hrs c) as Hcv. rewrite Hwc' in Hcv.
   pose proof (term_step_rank F' E tm Q' (regs s) k op w t a o x y (regs s c) HE' Hrs HFok Hw
     (term_step_operand F' Q' k op w t a o x y Helt Hwfs) Helt Hcv Hxy) as Hrank.
   rewrite Hnew, <- Hc0. exact Hrank.
Qed.

Lemma tcs_after : term_pos_lt b i (blk s') (idx s') -> tc_cond (regs s') ->
 (term_rank k w (regs s' c) < term_rank k w (regs s1 c))%nat.
Proof.
 intros Hlt Hcond. pose proof (proj2 tcs_bi Hlt) as Hle.
 destruct (Nat.eq_dec (blk s) b) as [Hb|Hb]; [destruct (Nat.eq_dec (idx s) i) as [Hi|Hi]|].
 - exact (tcs_write Hb Hi Hcond).
 - assert (Hlt_s : term_pos_lt b i (blk s) (idx s)) by (unfold term_pos_le, term_pos_lt in *; lia).
   rewrite tcs_keep_c by tauto. apply (ti_after _ _ _ Hinv Hlt_s).
   unfold tc_cond in Hcond |- *.
   destruct (last_instr (block_body prog l)) as [[| | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|] eqn:Hlast;
     try exact I.
   destruct (Nat.eqb b l && Nat.leb (term_guard_from prog l p) i) eqn:Happ; [|exact I].
   apply andb_true_iff in Happ. destruct Happ as [Hbl Hgi]. apply Nat.eqb_eq in Hbl. apply Nat.leb_le in Hgi.
   destruct tc_guard_facts as [g [Hg Hgc]]. rewrite Hlast in Hg. injection Hg as Hg _.
   rewrite <- (tcs_keep p (Hgc neg p (eq_sym Hg))); [exact Hcond|].
   apply (tc_guard_keep p neg tg (idx s) ins Hlast); [unfold term_pos_lt in Hlt_s; lia|].
   rewrite <- Hbl, <- Hb. exact Hf.
 - assert (Hlt_s : term_pos_lt b i (blk s) (idx s)) by (unfold term_pos_le, term_pos_lt in *; lia).
   rewrite tcs_keep_c by tauto. apply (ti_after _ _ _ Hinv Hlt_s).
   unfold tc_cond in Hcond |- *.
   destruct (last_instr (block_body prog l)) as [[| | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|] eqn:Hlast;
     try exact I.
   destruct (Nat.eqb b l && Nat.leb (term_guard_from prog l p) i) eqn:Happ; [|exact I].
   apply andb_true_iff in Happ. destruct Happ as [Hbl Hgi]. apply Nat.eqb_eq in Hbl.
   exfalso. apply term_in_iff in Hin. unfold term_pos_lt in Hlt_s. lia.
Qed.

Lemma tcs_range : b <> l -> (h < blk s')%nat -> exists x y,
   term_counter_hull F prog cert h l d w a = Some (Some (x,y)) /\ x <= extend w sg (regs s' c) <= y.
Proof.
 intros Hbl Hh'. pose proof tc_static as [Hbhl [Hji [_ [Hcc [_ [_ [Hwc _]]]]]]].
 assert (Hbh : b = h) by (destruct Hbhl; [assumption|contradiction]).
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 destruct (Nat.eq_dec (blk s) h) as [Hsh|Hsh].
 - (* the step leaves the head block *)
   pose proof (term_shape_structured prog Hshape) as Hstr.
   destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hs) as [[Hbb _]|[Hi0 [Hlen [Hlast [Hend|Htg]]]]];
     [lia|apply term_in_iff in Hin'; lia|].
   assert (Hbs : blk s = b) by congruence.
   assert (Hidx : (j <= idx s)%nat).
   { pose proof tc_write_pos as [insw [Hfw _]]. pose proof (term_fetch_len _ _ _ _ Hfw) as Hil.
     rewrite Hbs in Hlen. lia. }
   destruct (ti_state _ _ _ Hinv Hbs Hidx) as [F' [st0 [Hst [Hsame [HE' [Hrs [Hnb [Hwf Hok]]]]]]]].
   assert (Hs' : phase_step F' prog E tm s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
   assert (Hwfi : wf_instr F' ins = true) by (apply (term_wf_at F' prog b j (idx s) ins Hwf); [lia|rewrite <- Hbs; exact Hf]).
   assert (Hb' : (blk s' < List.length prog)%nat) by (apply term_in_iff in Hin'; lia).
   destruct (term_edge_sound F' prog E tm s s' _ ins HE' Hrs Hok Hwfi Hf Hlen Hs' Hi0 Hb') as [os [Hos [Hok' Hrs']]].
   set (sx := term_walk F' (term_assume_all F' d w a st0) (term_slice prog b j (idx s))) in *.
   assert (Hsxnb : is_bottom sx = false).
   { destruct (is_bottom sx) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm sx _ Hbot Hok)|reflexivity]. }
   assert (Hosnb : is_bottom os = false).
   { destruct (is_bottom os) eqn:Hbot; [exfalso; exact (bottom_not_ok F' (S tm) os _ Hbot Hok')|reflexivity]. }
   assert (Hhn : term_counter_hull F prog cert h l d w a <> None).
   { destruct (Bool.bool_dec uniq true) as [Hu|Hu].
     - pose proof (tc_back_u Hu) as Hbk. unfold term_counter_back in Hbk. cbn [tc_b] in Hbk.
       rewrite (proj2 (Nat.eqb_neq b l) Hbl) in Hbk. intros Hn. rewrite Hn in Hbk. discriminate.
     - destruct (tc_multi (Bool.not_true_is_false _ Hu)) as [_ [_ [_ [Hn|[x [y [Hn _]]]]]]]; rewrite Hn; discriminate. }
   unfold term_counter_hull, term_counter_hull_with in Hhn |- *. cbn [tc_j tc_i tc_c tc_kind] in Hhn |- *.
   set (L := flat_map _ (term_cut_states (term_counter_keep prog h d) F prog cert h j)) in Hhn |- *.
   assert (Hmem : In (cmp_range F' os (OReg c) (Z.of_nat w) sg) L).
   { subst L. apply in_flat_map. exists (F',st0). rewrite <- Hbh. split; [exact Hst|].
     cbv beta iota zeta. rewrite Hnb, Hwf. cbn [negb].
     replace (Nat.pred (block_len prog b)) with (idx s) by (rewrite <- Hbs; lia). fold sx. rewrite Hsxnb.
     rewrite <- Hbs, Hlast. apply in_flat_map. exists (blk s', Some os). split; [exact Hos|].
     cbv beta iota.
     rewrite (proj2 (Nat.ltb_lt (blk s) (blk s')) ltac:(lia)), (proj2 (Nat.leb_le (blk s') l) ltac:(apply term_in_iff in Hin'; lia)).
     rewrite Hosnb. left. reflexivity. }
   assert (Hwc' : wdn F' c = w) by (rewrite (term_same_wdn F F' c Hsame); exact Hwc).
   destruct (cmp_range F' os (OReg c) (Z.of_nat w) sg) as [[u v]|] eqn:Hcr.
   + destruct (term_range_hull L) as [[[x y]|]|] eqn:Hh;
       [|apply term_hull_empty in Hh; rewrite Hh in Hmem; contradiction|exfalso; apply Hhn; reflexivity].
     exists x,y. split; [reflexivity|].
     destruct (term_hull_some _ x y u v Hh Hmem) as [Hxu Hvy].
     pose proof (term_cmp_reg F' E (S tm) (regs s') os c w sg u v HE' Hrs' Hok' Hwc' Hcr). lia.
   + exfalso. exact (Hhn (term_hull_none _ Hmem)).
 - (* c keeps its value outside the head block *)
   assert (Hhs : (h < blk s)%nat) by (apply term_in_iff in Hin; lia).
   rewrite tcs_keep_c by (intros [Hb _]; congruence). exact (ti_range _ _ _ Hinv Hbl Hhs).
Qed.

Lemma tcs_latch : uniq = true -> blk s' = l -> (term_latch_start d l <= idx s')%nat -> exists F' st0,
   In (F',st0) (term_cut_states Kl F prog cert l (term_latch_start d l)) /\ term_same F F' /\ env_ok F' E /\
   regs_ok F' E (regs s') /\ is_bottom st0 = false /\ term_wf_from F' prog l (term_latch_start d l) = true /\
   tc_latch_win F' (term_latch_entry F prog cert h l d w a F' st0) (S tm) s'.
Proof.
 intros Hu Hb' Hst'. pose proof tc_static as [Hbhl [Hji _]].
 destruct (Nat.eq_dec (idx s') (term_latch_start d l)) as [Hi0|Hi0].
 - apply (tc_latch_entry Hu (S tm) s' tcs_reach Hb' Hi0).
   + intros Hbl. unfold term_latch_start in Hi0. cbn [tc_b tc_j] in Hi0.
     rewrite (proj2 (Nat.eqb_eq b l) Hbl) in Hi0.
     split.
     * rewrite tcs_before; [exact (ti_head _ _ _ Hinv)|unfold term_pos_le; lia].
     * apply (tc_alias_at_j s' tcs_alias); [congruence|exact Hi0].
   + intros Hbl. apply tcs_range; [exact Hbl|].
     assert (Hbh : b = h) by (destruct Hbhl; [assumption|contradiction]).
     destruct (term_edges_spec prog l h He) as [_ [Hhl _]]. assert (h <> l) by congruence. lia.
 - destruct (tcs_prev ltac:(lia)) as [Hbb Hii]. assert (Hbs : blk s = l) by congruence.
   destruct (ti_latch _ _ _ Hinv Hu Hbs ltac:(lia)) as [F' [st0 [Hst [Hsame [HE' [Hrs [Hnb [Hwf Hwin]]]]]]]].
   exists F',st0. split; [exact Hst|]. split; [exact Hsame|]. split; [exact HE'|].
   assert (Hs' : phase_step F' prog E tm s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
   assert (Hwfi : wf_instr F' ins = true)
     by (apply (term_wf_at F' prog l (term_latch_start d l) (idx s) ins Hwf); [lia|rewrite <- Hbs; exact Hf]).
   assert (Htr : forall st, state_ok F' tm st (regs s) -> state_ok F' (S tm) (term_xfer F' st ins) (regs s')).
   { intros st Hok. exact (proj1 (term_xfer_sound F' prog E tm s s' st ins HE' Hf Hs' Hok Hrs Hwfi)). }
   assert (Hrs' : regs_ok F' E (regs s')).
   { apply (term_same_regs F F' E (regs s') Hsame).
     destruct (tc_at (S tm) s' tcs_reach Hin') as [_ [_ [HrsF _]]]. exact HrsF. }
   split; [exact Hrs'|]. split; [exact Hnb|]. split; [exact Hwf|].
   assert (Hstep_slice : forall x st, (x <= idx s)%nat ->
     term_walk F' st (term_slice prog l x (idx s')) = term_xfer F' (term_walk F' st (term_slice prog l x (idx s))) ins).
   { intros x st Hx. rewrite Hii. apply term_twalk_step; [exact Hx|rewrite <- Hbs; exact Hf]. }
   unfold tc_latch_win in Hwin |- *.
   set (sl0 := term_latch_start d l) in *.
   destruct (last_instr (block_body prog l)) as [[| | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|] eqn:Hlast;
     try (rewrite (Hstep_slice sl0 _ ltac:(lia)); exact (Htr _ Hwin)).
   set (r := Nat.max (term_guard_from prog l p) sl0) in *.
   destruct Hwin as [P1 P2].
   assert (P1' : (idx s' <= r)%nat -> state_ok F' (S tm) (term_walk F' (term_latch_entry F prog cert h l d w a F' st0)
     (term_slice prog l sl0 (idx s'))) (regs s')).
   { intros Hle. rewrite (Hstep_slice sl0 _ ltac:(lia)). apply Htr. apply P1. lia. }
   split; [exact P1'|].
   intros Hle Htk. destruct (Nat.eq_dec (idx s') r) as [Heq|Hne].
   + pose proof (P1' ltac:(lia)) as Hok. rewrite Heq in Hok.
     destruct (term_refine_sound F' E (S tm) (regs s') _ p (negb neg) HE' Hrs' Hok Htk) as [sref [Href Hsref]].
     exists sref. split; [exact Href|]. rewrite Heq, term_slice_empty. exact Hsref.
   + assert (Hp : regs s' p = regs s p).
     { destruct tc_guard_facts as [g [Hg Hgc]]. rewrite Hlast in Hg. injection Hg as Hg _.
       apply tcs_keep; [exact (Hgc neg p (eq_sym Hg))|].
       apply (tc_guard_keep p neg tg (idx s) ins Hlast); [unfold r in *; lia|rewrite <- Hbs; exact Hf]. }
     rewrite Hp in Htk. destruct (P2 ltac:(lia) Htk) as [sref [Href Hsref]].
     exists sref. split; [exact Href|]. rewrite (Hstep_slice r _ ltac:(lia)). exact (Htr _ Hsref).
Qed.

Lemma tc_step_inv : tc_inv s1 (S tm) s'.
Proof.
 constructor.
 - exact (ti_head _ _ _ Hinv).
 - exact tcs_before.
 - exact tcs_after.
 - exact tcs_range.
 - exact tcs_alias.
 - exact tcs_copy.
 - exact tcs_shift.
 - exact tcs_state.
 - exact tcs_latch.
Qed.
End Step.

(* The interval holds on every edge into the head from outside the loop. *)
Lemma tc_entry_range : forall tm s s' ins, reach tm s -> term_in h l (blk s) = false ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E tm s = Some s' ->
 term_in h l (blk s') = true -> lo <= extend w sg (regs s' c) <= hi.
Proof.
 intros tm s s' ins Hr Hout Hf Hs Hin'.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 destruct (fetch_block F prog E _ _ _ Hf) as [Hbs _].
 destruct (term_states_sound F prog cert E tm s Hcert HE Hr Hbs) as [_ [F' [st [Hst [Hsame [HE' [Hrs Hok]]]]]]].
 assert (Hs' : phase_step F' prog E tm s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
 destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hs) as [[Hbb _]|[Hi' [Hlen [Hlast [Hend|Htg]]]]].
 - exfalso. rewrite Hbb in Hin'. congruence.
 - apply term_in_iff in Hin'. lia.
 - assert (Hnb : is_bottom st = false).
   { destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm st _ Hbot Hok)|reflexivity]. }
   assert (Hidx : idx s = Nat.pred (block_len prog (blk s))) by lia.
   rewrite Hidx in Hst.
   assert (Hmem : forall z, In z (flat_map (fun x => let '(F'',st'') := x in
       if is_bottom st'' then [] else
       if negb (wf_instr F'' ins) then [None] else
       flat_map (fun e => let '(u,os) := e in
         if term_in h l u then match os with
           | Some q => if is_bottom q then [] else [cmp_range F'' q (OReg c) (Z.of_nat w) sg]
           | None => [] end else [])
         (edge_states F'' prog (blk s) (transfer F'' st'' ins)))
       (term_states F prog cert (blk s) (Nat.pred (block_len prog (blk s))))) ->
     In z (term_entry_ranges F prog cert h l c w sg)).
   { intros z Hz. unfold term_entry_ranges. apply in_flat_map. exists (blk s).
     split; [apply in_seq; lia|]. rewrite Hout.
     assert (Hex : existsb (term_in h l) (edge_targets prog (blk s)) = true).
     { apply existsb_exists. exists (blk s'). split; [exact Htg|exact Hin']. }
     rewrite Hex. cbn [negb]. rewrite Hlast. exact Hz. }
   destruct (wf_instr F' ins) eqn:Hwf.
   + assert (Hb' : (blk s' < List.length prog)%nat) by (apply term_in_iff in Hin'; lia).
     destruct (term_edge_sound F' prog E tm s s' st ins HE' Hrs Hok Hwf
       Hf Hlen Hs' Hi' Hb') as [os [Hos [Hok' Hrs']]].
     assert (Hosnb : is_bottom os = false).
     { destruct (is_bottom os) eqn:Hbot; [exfalso; exact (bottom_not_ok F' (S tm) os _ Hbot Hok')|reflexivity]. }
     assert (Hin : In (cmp_range F' os (OReg c) (Z.of_nat w) sg) (term_entry_ranges F prog cert h l c w sg)).
     { apply Hmem. apply in_flat_map. exists (F',st). split; [exact Hst|].
       cbv beta iota. rewrite Hnb, Hwf. cbn [negb].
       apply in_flat_map. exists (blk s', Some os). split; [exact Hos|].
       cbv beta iota. rewrite Hin', Hosnb. left. reflexivity. }
     pose proof (term_short_all_in _ _ _ _ Hentry Hin) as Hw. unfold term_within in Hw.
     destruct (cmp_range F' os (OReg c) (Z.of_nat w) sg) as [[x y]|] eqn:Hcr; [|discriminate].
     rewrite andb_true_iff, !Z.leb_le in Hw.
     pose proof (term_cmp_reg F' E (S tm) (regs s') os c w sg x y HE' Hrs' Hok'
       ltac:(rewrite (term_same_wdn F F' c Hsame); exact (proj1 (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 tc_static)))))))) Hcr).
     lia.
   + exfalso. assert (Hin : In None (term_entry_ranges F prog cert h l c w sg)).
     { apply Hmem. apply in_flat_map. exists (F',st). split; [exact Hst|].
       cbv beta iota. rewrite Hnb, Hwf. left. reflexivity. }
     pose proof (term_short_all_in _ _ _ _ Hentry Hin) as Hw. discriminate.
Qed.

(* The latch is taken from its last instruction, a branch to the head. *)
Lemma tc_latch_step_at : forall v tm s s' ins, In (v,h) (term_edges prog) -> fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E tm s = Some s' -> blk s = v -> blk s' = h -> idx s' = 0%nat ->
 S (idx s) = block_len prog v /\ last_instr (block_body prog v) = Some ins /\
 exists g, ins = IBra g h /\ holds g (regs s) = true /\ regs s' = regs s.
Proof. intros v tm s s' ins Hv. exact (term_latch_jump F prog E v h tm s s' ins Hshape Hv). Qed.

Lemma tc_latch_step : forall tm s s' ins, fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E tm s = Some s' -> blk s = l -> blk s' = h -> idx s' = 0%nat ->
 S (idx s) = block_len prog l /\ last_instr (block_body prog l) = Some ins /\
 exists g, ins = IBra g h /\ holds g (regs s) = true /\ regs s' = regs s.
Proof. intros tm s s' ins. exact (tc_latch_step_at l tm s s' ins He). Qed.

(* The interval holds on the back edge. *)
Lemma tc_back_range : uniq = true -> forall s1 tm s s', tc_inv s1 tm s -> reach tm s -> blk s = l ->
 phase_step F prog E tm s = Some s' -> blk s' = h -> idx s' = 0%nat -> lo <= extend w sg (regs s' c) <= hi.
Proof.
 intros Hu s1 tm s s' Hinv Hr Hbl Hs Hbh Hi0.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 assert (Hin : term_in h l (blk s) = true) by (apply term_in_iff; lia).
 destruct (tc_at tm s Hr Hin) as [_ [Hidx _]].
 destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
 2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
 destruct (tc_latch_step tm s s' ins Hf Hs Hbl Hbh Hi0) as [Hlen [Hlast [g [Hins [Htaken Hregs]]]]].
 pose proof tc_static as [Hbhl [Hji _]].
 pose proof tc_write_pos as [insw [Hfw _]].
 assert (Hil : (i < block_len prog b)%nat).
 { unfold fetch, block_len in *. apply nth_error_Some. rewrite Hfw. discriminate. }
 assert (Hstart : (term_latch_start d l <= idx s)%nat).
 { unfold term_latch_start. cbn [tc_b tc_j]. destruct (Nat.eqb_spec b l) as [Hb|Hb]; [|lia].
   rewrite Hb in Hil. lia. }
 destruct (ti_latch _ _ _ Hinv Hu Hbl Hstart) as [F' [st0 [Hst [Hsame [HE' [Hrs [Hnb [Hwf Hwin]]]]]]]].
 assert (Hs' : phase_step F' prog E tm s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
 assert (Hwfi : wf_instr F' ins = true)
   by (apply (term_wf_at F' prog l (term_latch_start d l) (idx s) ins Hwf); [exact Hstart|rewrite <- Hbl; exact Hf]).
 assert (Hlatch : term_counter_latch F prog cert h l d w a = true).
 { destruct (Nat.eq_dec b l) as [Hb|Hb].
   - pose proof (tc_back_u Hu) as Hb2. unfold term_counter_back in Hb2. cbn [tc_b] in Hb2.
     rewrite (proj2 (Nat.eqb_eq b l) Hb) in Hb2. exact Hb2.
   - apply (tc_latch_check Hu).
     destruct (ti_range _ _ _ Hinv Hb ltac:(destruct Hbhl; [assert (h <> l) by congruence; lia|contradiction]))
       as [x [y [Hh _]]]. rewrite Hh. discriminate. }
 pose proof (term_short_all_in _ _ _ _ Hlatch Hst) as Helt. cbv beta iota in Helt.
 change (term_latch_entry_with (term_counter_hull F prog cert h l d w a) l d w a F' st0)
   with (term_latch_entry F prog cert h l d w a F' st0) in Helt.
 rewrite Hnb, Hwf in Helt. cbn [negb] in Helt.
 assert (Hq : exists q, term_latch F' prog l (term_latch_start d l) (term_latch_entry F prog cert h l d w a F' st0) = Some q /\
   state_ok F' (S tm) q (regs s')).
 { assert (Hfl : fetch prog l (idx s) = Some ins) by (rewrite <- Hbl; exact Hf).
   assert (Hskip : forall x st, (x <= idx s)%nat -> term_walk F' st (skipn x (block_body prog l)) =
     term_xfer F' (term_walk F' st (term_slice prog l x (idx s))) ins).
   { intros x st Hx. rewrite <- term_slice_all, <- Hlen. apply term_twalk_step; [exact Hx|exact Hfl]. }
   assert (Htr : forall st, state_ok F' tm st (regs s) -> state_ok F' (S tm) (term_xfer F' st ins) (regs s')).
   { intros st Hok. exact (proj1 (term_xfer_sound F' prog E tm s s' st ins HE' Hf Hs' Hok Hrs Hwfi)). }
   unfold tc_latch_win in Hwin. unfold term_latch. rewrite Hlast in Hwin |- *. rewrite Hins in Hwin |- *.
   destruct g as [[neg p]|].
   + set (r := Nat.max (term_guard_from prog l p) (term_latch_start d l)) in *.
     assert (Hgr : (term_guard_from prog l p <= idx s)%nat).
     { unfold term_guard_from. destruct (term_last_write prog l (block_len prog l) p) as [[k0 insk]|] eqn:Hlw; [|lia].
       destruct (term_last_write_some prog l _ p k0 insk Hlw) as [Hk0 [Hfk [Hwk _]]].
       destruct (Nat.eq_dec k0 (idx s)) as [Hke|Hke]; [|lia].
       exfalso. rewrite Hke, Hfl, Hins in Hfk. injection Hfk as <-. discriminate. }
     cbn [holds] in Htaken.
     assert (Htk : negb (Z.eqb (regs s p) 0) = negb neg) by (destruct neg, (Z.eqb (regs s p) 0); cbn in Htaken |- *; congruence).
     destruct (proj2 Hwin ltac:(unfold r; lia) Htk) as [sref [Href Hsref]].
     rewrite Href. exists (term_walk F' sref (skipn r (block_body prog l))). split; [reflexivity|].
     rewrite (Hskip r sref ltac:(unfold r; lia)). exact (Htr _ Hsref).
   + exists (term_walk F' (term_latch_entry F prog cert h l d w a F' st0) (skipn (term_latch_start d l) (block_body prog l))).
     split; [reflexivity|]. rewrite (Hskip _ _ Hstart). exact (Htr _ Hwin). }
 destruct Hq as [q [Hq Hqok]]. rewrite Hq in Helt.
 assert (Hqnb : is_bottom q = false).
 { destruct (is_bottom q) eqn:Hbot; [exfalso; exact (bottom_not_ok F' (S tm) q _ Hbot Hqok)|reflexivity]. }
 rewrite Hqnb in Helt. unfold term_within in Helt. cbn [tc_lo tc_hi tc_c tc_kind] in Helt.
 destruct (cmp_range F' q (OReg c) (Z.of_nat w) sg) as [[x y]|] eqn:Hcr; [|discriminate].
 rewrite andb_true_iff, !Z.leb_le in Helt.
 assert (Hrs' : regs_ok F' E (regs s')) by (rewrite Hregs; exact Hrs).
 pose proof (term_cmp_reg F' E (S tm) (regs s') q c w sg x y HE' Hrs' Hqok
   ltac:(rewrite (term_same_wdn F F' c Hsame); exact (proj1 (proj2 (proj2 (proj2 (proj2 (proj2 (proj2 tc_static)))))))) Hcr).
 lia.
Qed.

(* An iteration starts at the head with the counter in its interval. *)
Lemma tc_start : forall tm s, reach tm s -> blk s = h -> idx s = 0%nat ->
 lo <= extend w sg (regs s c) <= hi -> tc_inv s tm s.
Proof.
 intros tm s Hr Hb Hi Hc. pose proof tc_static as [Hbhl [Hji _]].
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 assert (Hhb : (h <= b)%nat) by (destruct Hbhl; lia).
 assert (Hwin : tc_alias_win s).
 { pose proof tc_alias_facts as Hal. unfold tc_alias_win.
   remember al as alv eqn:Halv in Hal |- *. destruct alv as [|m|m]; [exact I| |].
   - intros _ Hm. lia.
   - intros Hlt _. unfold term_pos_lt in Hlt. lia. }
 constructor.
 - exact Hc.
 - intros _. reflexivity.
 - intros Hlt. unfold term_pos_lt in Hlt. lia.
 - intros _ Hlt. unfold term_pos_lt in Hlt. lia.
 - exact Hwin.
 - intros _ _ Hj. lia.
 - intros m _ Hlt. unfold term_pos_lt in Hlt. lia.
 - intros Hbs Hj. assert (Hj0 : j = 0%nat) by lia. rewrite <- Hj0 in Hi.
   apply (tc_state_entry tm s Hr Hbs Hi Hc). exact (tc_alias_at_j s Hwin Hbs Hi).
 - intros Hu Hbl Hst. assert (Hst0 : idx s = term_latch_start d l) by lia.
   apply (tc_latch_entry Hu tm s Hr Hbl Hst0).
   + intros Hb'. split; [exact Hc|]. apply (tc_alias_at_j s Hwin); [congruence|].
     unfold term_latch_start in Hst0. cbn [tc_b tc_j] in Hst0. rewrite (proj2 (Nat.eqb_eq b l) Hb') in Hst0. exact Hst0.
   + intros Hb'. exfalso. destruct Hbhl; congruence.
Qed.

(* An iteration that enters the loop past its head, when the update and its
   copy lie in the latch block. *)
Lemma tc_start_mid : forall tm s, reach tm s -> b = l -> (forall m, al <> AHead m) -> sh = None ->
 term_in h l (blk s) = true -> idx s = 0%nat ->
 lo <= extend w sg (regs s c) <= hi -> tc_inv s tm s.
Proof.
 intros tm s Hr Hbl Hal Hsh0 Hin Hi Hc. pose proof tc_static as [Hbhl [Hji _]].
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 apply term_in_iff in Hin.
 assert (Hwin : tc_alias_win s).
 { unfold tc_alias_win. remember al as alv eqn:Halv. destruct alv as [|m|m]; [exact I| |].
   - intros _ Hm. lia.
   - exfalso. exact (Hal m eq_refl). }
 constructor.
 - exact Hc.
 - intros _. reflexivity.
 - intros Hlt. unfold term_pos_lt in Hlt. lia.
 - intros Hb'. contradiction.
 - exact Hwin.
 - intros _ _ Hj. lia.
 - intros m Hsh. rewrite Hsh0 in Hsh. discriminate.
 - intros Hbs Hj. assert (Hj0 : j = 0%nat) by lia. rewrite <- Hj0 in Hi.
   apply (tc_state_entry tm s Hr Hbs Hi Hc). exact (tc_alias_at_j s Hwin Hbs Hi).
 - intros Hu Hbl' Hst. assert (Hst0 : idx s = term_latch_start d l) by lia.
   apply (tc_latch_entry Hu tm s Hr Hbl' Hst0).
   + intros Hb'. split; [exact Hc|]. apply (tc_alias_at_j s Hwin); [congruence|].
     unfold term_latch_start in Hst0. cbn [tc_b tc_j] in Hst0. rewrite (proj2 (Nat.eqb_eq b l) Hb') in Hst0. exact Hst0.
   + intros Hb'. contradiction.
Qed.

(* A thread that entered the loop past its head, before it reaches the head,
   when the update lies in the head: the counter keeps its entry value. *)
Definition tc_partial (s : state) : Prop :=
 b = h /\ blk s <> h /\ lo <= extend w sg (regs s c) <= hi.

(* Every reachable state in the loop lies in an iteration whose invariant holds. *)
Lemma tc_global : forall tm s, reach tm s -> term_in h l (blk s) = true ->
 (exists s1, tc_inv s1 tm s /\ (blk s = h -> idx s = 0%nat -> s1 = s)) \/ tc_partial s.
Proof.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 pose proof (term_shape_edge prog Hshape l h He) as Hh0.
 pose proof tc_static as [Hbhl [_ [_ [Hcc _]]]].
 pose proof tc_write_pos as [insw [Hfw [_ Hww]]].
 intro tm. induction tm as [|tm IH]; intros s' Hr' Hin'.
 - exfalso. unfold run' in Hr'. cbn [run] in Hr'. injection Hr' as <-. apply term_in_iff in Hin'. cbn in Hin'. lia.
 - rewrite term_run_succ in Hr'. destruct (run' F prog E tm) as [s|] eqn:Hr; [|discriminate].
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hr'. rewrite Hf in Hr'. discriminate. }
   assert (Hr2 : reach (S tm) s') by (rewrite term_run_succ, Hr; exact Hr').
   (* c keeps its value at a step outside block b *)
   assert (Hkeep : term_in h l (blk s) = true -> blk s <> b -> regs s' c = regs s c).
   { intros Hin Hnb. apply (term_step_keeps F prog E tm s s' ins c Hcc Hf Hr').
     destruct (term_writes c ins) eqn:Hw; [|reflexivity]. exfalso. apply Hnb.
     apply term_in_iff in Hin.
     destruct (tc_unique (blk s) (idx s) ins Hin Hf Hw) as [Hb _]. exact Hb. }
   destruct (Nat.eq_dec (blk s') h) as [Hbh|Hbh]; [destruct (Nat.eq_dec (idx s') 0) as [Hi0|Hi0]|].
   + left. exists s'. split; [|intros _ _; reflexivity].
     apply (tc_start (S tm) s' Hr2 Hbh Hi0).
     destruct (term_in h l (blk s)) eqn:Hin.
     * pose proof (term_step_head F prog E Hshape l h tm s s' ins Hf Hr' Hin Hbh Hi0) as Hedge.
       destruct (tc_latch_step_at (blk s) tm s s' ins Hedge Hf Hr' eq_refl Hbh Hi0) as [_ [_ [g [_ [_ Hregs]]]]].
       destruct (Bool.bool_dec uniq true) as [Hu|Hu].
       -- pose proof (tc_to_head Hu tm s s' ins Hf Hr' Hin Hbh Hi0) as Hbl.
          destruct (IH s eq_refl Hin) as [[s1 [Hinv _]]|[Hbh' [Hnh Hcr]]].
          ++ exact (tc_back_range Hu s1 tm s s' Hinv Hr Hbl Hr' Hbh Hi0).
          ++ rewrite Hregs. exact Hcr.
       -- destruct (tc_multi (Bool.not_true_is_false _ Hu)) as [Hbh2 [Hhl2 [Hlat Hhull]]].
          rewrite Hregs.
          destruct (IH s eq_refl Hin) as [[s1 [Hinv _]]|[_ [_ Hcr]]]; [|exact Hcr].
          assert (Hv : (h < blk s)%nat).
          { apply Hlat. apply (term_latches_in prog h l (blk s) Hedge). apply term_in_iff. exact Hin. }
          destruct (ti_range _ _ _ Hinv ltac:(congruence) Hv) as [x [y [Hh Hxy]]].
          destruct Hhull as [Hn|[x' [y' [Hn [Hlo Hhi]]]]]; rewrite Hn in Hh; [discriminate|].
          injection Hh as <- <-. lia.
     * exact (tc_entry_range tm s s' ins Hr Hin Hf Hr' Hin').
   + destruct (term_in h l (blk s)) eqn:Hin.
     * destruct (IH s eq_refl Hin) as [[s1 [Hinv _]]|[_ [Hnh _]]].
       -- left. exists s1. split; [|intros _ Hi; contradiction].
          apply (tc_step_inv s1 s s' tm ins Hinv Hr Hin Hf Hr' Hin'). tauto.
       -- exfalso. destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hr') as [[Hbb _]|[Hi' _]].
          ++ apply Hnh. congruence.
          ++ apply Hi0. exact Hi'.
     * exfalso. destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hr') as [[Hbb _]|[Hi' [_ [_ [Hend|Htg]]]]].
       -- rewrite Hbb in Hin'. congruence.
       -- apply term_in_iff in Hin'. lia.
       -- apply Hi0. exact Hi'.
   + destruct (term_in h l (blk s)) eqn:Hin.
     * destruct (IH s eq_refl Hin) as [[s1 [Hinv _]]|[Hbh' [Hnh Hcr]]].
       -- left. exists s1. split; [|intros Hb; contradiction].
          apply (tc_step_inv s1 s s' tm ins Hinv Hr Hin Hf Hr' Hin'). tauto.
       -- right. split; [exact Hbh'|]. split; [exact Hbh|].
          rewrite (Hkeep eq_refl ltac:(congruence)). exact Hcr.
     * pose proof (tc_entry_range tm s s' ins Hr Hin Hf Hr' Hin') as Hcr.
       destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hr') as [[Hbb _]|[Hi' [_ [_ [Hend|Htg]]]]].
       -- exfalso. rewrite Hbb in Hin'. congruence.
       -- exfalso. apply term_in_iff in Hin'. lia.
       -- destruct (Nat.eq_dec b h) as [Hbh2|Hbh2].
          ++ right. split; [exact Hbh2|]. split; [exact Hbh|exact Hcr].
          ++ assert (Hbl : b = l) by (destruct Hbhl; [contradiction|assumption]).
             destruct tc_entered as [Hent|[Hal Hsh0]].
             ** exfalso. apply Hbh. apply (term_entered_head prog l h (blk s) (blk s') Hent); [|exact Hin|exact Htg|exact Hin'].
                exact (proj1 (fetch_block F prog E _ _ _ Hf)).
             ** left. exists s'. split; [|intros Hb; contradiction].
                exact (tc_start_mid (S tm) s' Hr2 Hbl Hal Hsh0 Hin' Hi' Hcr).
Qed.

Theorem tc_loop_proof : term_loop_proof F prog E h l.
Proof.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 exists (fun rs => term_rank k w (rs c)), tc_inv. split; [|split].
 - intros t0 s Hr Hb Hi. assert (Hin : term_in h l (blk s) = true) by (apply term_in_iff; lia).
   destruct (tc_global t0 s Hr Hin) as [[s1 [Hinv Heq]]|[_ [Hnh _]]]; [|contradiction].
   rewrite (Heq Hb Hi) in Hinv. exact Hinv.
 - intros s1 t0 s s' Hinv Hr Hin Hs Hin' Hnh.
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   exact (tc_step_inv s1 s s' t0 ins Hinv Hr Hin Hf Hs Hin' Hnh).
 - intros s1 t0 s s' Hinv Hr Hinl Hs Hbh Hi0.
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   pose proof (term_step_head F prog E Hshape l h t0 s s' ins Hf Hs Hinl Hbh Hi0) as Hedge.
   destruct (Bool.bool_dec uniq true) as [Hu|Hu].
   2:{ destruct (tc_multi (Bool.not_true_is_false _ Hu)) as [Hbh2 [Hhl2 [Hlat _]]].
       destruct (tc_latch_step_at (blk s) t0 s s' ins Hedge Hf Hs eq_refl Hbh Hi0) as [_ [_ [g [_ [_ Hregs]]]]].
       assert (Hv : (h < blk s)%nat).
       { apply Hlat. apply (term_latches_in prog h l (blk s) Hedge). apply term_in_iff. exact Hinl. }
       rewrite Hregs. apply (ti_after _ _ _ Hinv); [unfold term_pos_lt; left; lia|].
       unfold tc_cond. rewrite (proj2 (Nat.eqb_neq b l) ltac:(congruence)).
       destruct (last_instr (block_body prog l)) as [[| | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|]; exact I. }
   pose proof (tc_to_head Hu t0 s s' ins Hf Hs Hinl Hbh Hi0) as Hbl.
   destruct (tc_latch_step t0 s s' ins Hf Hs Hbl Hbh Hi0) as [Hlen [Hlast [g [Hins [Htaken Hregs]]]]].
   pose proof tc_static as [Hbhl _].
   pose proof tc_write_pos as [insw [Hfw [_ Hww]]].
   assert (Hlt : term_pos_lt b i (blk s) (idx s)).
   { unfold term_pos_lt. destruct (Nat.eq_dec b l) as [Hb|Hb].
     - right. split; [congruence|].
       assert (Hil : (i < block_len prog l)%nat).
       { rewrite <- Hb. unfold fetch, block_len in *. apply nth_error_Some. rewrite Hfw. discriminate. }
       destruct (Nat.eq_dec i (idx s)) as [Hie|Hie]; [|lia].
       exfalso. rewrite Hb, Hie, <- Hbl, Hf, Hins in Hfw. injection Hfw as <-. discriminate.
     - left. destruct Hbhl; [lia|contradiction]. }
   rewrite Hregs. apply (ti_after _ _ _ Hinv Hlt).
   unfold tc_cond. rewrite Hlast, Hins. destruct g as [[neg p]|]; [|exact I].
   destruct (Nat.eqb b l && Nat.leb (term_guard_from prog l p) i); [|exact I].
   cbn [holds] in Htaken. destruct neg, (Z.eqb (regs s p) 0); cbn in Htaken |- *; congruence.
Qed.
End Counter.

Lemma term_short_all_ext : forall A (f g : A -> bool) xs, (forall x, f x = g x) -> short_all f xs = short_all g xs.
Proof. intros A f g xs H. rewrite !short_all_same. induction xs as [|x xs IH]; [reflexivity|]. cbn. rewrite H, IH. reflexivity. Qed.

Lemma term_counter_back_shared : forall F prog cert h l d op w t a o,
 term_counter_static F prog h l d op w t a o = true ->
 term_counter_back_with (term_cut_states (term_counter_keep prog (tc_b d) d) F prog cert (tc_b d) (tc_j d))
   F prog cert h l d w a =
 term_counter_back F prog cert h l d w a.
Proof.
 intros F prog cert h l d op w t a o Hst.
 unfold term_counter_back_with, term_counter_back, term_counter_latch, term_counter_latch_with,
   term_latch_start, term_latch_entry_with.
 destruct (Nat.eqb_spec (tc_b d) l) as [Hbl|Hbl].
 - rewrite Hbl. apply term_short_all_ext. intros [F' st]. reflexivity.
 - assert (Hbh : tc_b d = h).
   { unfold term_counter_static in Hst. rewrite !andb_true_iff in Hst.
     destruct Hst as [[[[[[[[[[[[_ Hb] _] _] _] _] _] _] _] _] _] _] _].
     rewrite orb_true_iff, !Nat.eqb_eq in Hb. destruct Hb as [Hb|Hb]; [exact Hb|contradiction]. }
   unfold term_counter_hull. rewrite Hbh. reflexivity.
Qed.

Lemma term_counter_multi_shared : forall F prog cert h l d w a,
 term_counter_multi_with (term_cut_states (term_counter_keep prog (tc_b d) d) F prog cert (tc_b d) (tc_j d))
   F prog h l d w a =
 term_counter_multi F prog cert h l d w a.
Proof.
 intros F prog cert h l d w a. unfold term_counter_multi, term_counter_multi_with.
 destruct (Nat.eqb_spec (tc_b d) h) as [Hbh|Hbh]; [rewrite Hbh; reflexivity|reflexivity].
Qed.

Theorem term_counter_sound : forall F prog cert E l h dsc,
 check_scalar_certificate F prog cert = true -> launch_ok F E -> term_shape prog = true ->
 In (l,h) (term_edges prog) -> term_counter_ok F prog cert h l dsc = true -> term_loop_proof F prog E h l.
Proof.
 intros F prog cert E l h [c b i j step al sh k lo hi] Hcert HE Hshape He Hok.
 unfold term_counter_ok, term_step_regs in Hok. cbn [tc_step tc_lo tc_hi tc_c tc_kind] in Hok.
 destruct step as [ |  | g op w t a o |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ]; try discriminate.
 destruct g as [[neg p]|]; [discriminate|].
 destruct (term_counter_static F prog h l _ op w t a o) eqn:Hstatic; [|discriminate].
 destruct (short_all (term_within lo hi) (term_entry_ranges F prog cert h l c w (term_signed k))) eqn:Hentry; [|discriminate].
 cbn [tc_b] in Hok.
 destruct (term_block_wf F prog cert b) eqn:Hwfb; [|discriminate].
 destruct (term_block_wf F prog cert l) eqn:Hwfl; [|discriminate].
 cbn [andb negb] in Hok. cbv zeta in Hok.
 match type of Hok with context [term_counter_step_with ?S0 ?F0 ?p0 ?h0 ?l0 ?d0 ?w0 ?a0] =>
   destruct (term_counter_step_with S0 F0 p0 h0 l0 d0 w0 a0) eqn:Hstep; [|discriminate] end.
 change (term_counter_step F prog cert h l {| tc_c := c; tc_b := b; tc_i := i; tc_j := j;
   tc_step := IBin None op w t a o; tc_alias := al; tc_shift := sh; tc_kind := k; tc_lo := lo; tc_hi := hi |} w a = true) in Hstep.
 pose proof (term_counter_back_shared F prog cert h l _ op w t a o Hstatic) as Hshr.
 pose proof (term_counter_multi_shared F prog cert h l {| tc_c := c; tc_b := b; tc_i := i; tc_j := j;
   tc_step := IBin None op w t a o; tc_alias := al; tc_shift := sh; tc_kind := k; tc_lo := lo; tc_hi := hi |} w a) as Hmsh.
 cbn [negb] in Hok.
 assert (Hback : (if forallb (Nat.eqb l) (term_latches prog h l) then term_counter_back F prog cert h l
   {| tc_c := c; tc_b := b; tc_i := i; tc_j := j; tc_step := IBin None op w t a o; tc_alias := al;
      tc_shift := sh; tc_kind := k; tc_lo := lo; tc_hi := hi |} w a
   else term_counter_multi F prog cert h l
   {| tc_c := c; tc_b := b; tc_i := i; tc_j := j; tc_step := IBin None op w t a o; tc_alias := al;
      tc_shift := sh; tc_kind := k; tc_lo := lo; tc_hi := hi |} w a) = true).
 { destruct (forallb (Nat.eqb l) (term_latches prog h l)).
   - match type of Hok with context [term_counter_back_with ?S0 ?F0 ?p0 ?c0 ?h0 ?l0 ?d0 ?w0 ?a0] =>
       assert (Hbk : term_counter_back_with S0 F0 p0 c0 h0 l0 d0 w0 a0 = true) by exact Hok end.
     exact (eq_trans (eq_sym Hshr) Hbk).
   - match type of Hok with context [term_counter_multi_with ?S0 ?F0 ?p0 ?h0 ?l0 ?d0 ?w0 ?a0] =>
       assert (Hbk : term_counter_multi_with S0 F0 p0 h0 l0 d0 w0 a0 = true) by exact Hok end.
     exact (eq_trans (eq_sym Hmsh) Hbk). }
 exact (tc_loop_proof F prog cert E l h c b i j al sh k lo hi op w t a o Hcert HE Hshape He Hstatic Hentry Hstep Hback Hwfb Hwfl).
Qed.

(* ------------------------------------------------------------------ *)
(* The values of a binary search.                                      *)

Lemma bs_add_eval : forall u v x L D, bs_add u v = Some x -> bs_eval x L D = bs_eval u L D + bs_eval v L D.
Proof.
 intros u v x L D H. unfold bs_add in H.
 destruct (Z.eqb_spec (bv_r u) 0) as [Hu|Hu].
 - injection H as <-. unfold bs_eval. cbn [bv_k bv_p bv_q bv_r bv_j]. rewrite Hu. ring.
 - destruct (Z.eqb_spec (bv_r v) 0) as [Hv|Hv].
   + injection H as <-. unfold bs_eval. cbn [bv_k bv_p bv_q bv_r bv_j]. rewrite Hv. ring.
   + destruct (Z.eqb_spec (bv_j u) (bv_j v)) as [Hj|Hj]; [|discriminate].
     injection H as <-. unfold bs_eval. cbn [bv_k bv_p bv_q bv_r bv_j]. rewrite Hj. ring.
Qed.

Lemma bs_neg_eval : forall v L D, bs_eval (bs_neg v) L D = - bs_eval v L D.
Proof. intros v L D. unfold bs_eval, bs_neg. cbn [bv_k bv_p bv_q bv_r bv_j]. ring. Qed.

Lemma bs_lin_eval : forall k p q L D, bs_eval (bs_lin k p q) L D = k + p * L + q * D.
Proof. intros. unfold bs_eval, bs_lin. cbn [bv_k bv_p bv_q bv_r bv_j]. rewrite Z.mul_0_l. ring. Qed.

Lemma bs_half_eval : forall u x L D, bs_half u = Some x -> bs_eval x L D = bs_eval u L D / 2.
Proof.
 intros u x L D H. unfold bs_half in H.
 destruct (Z.eqb_spec (bv_r u) 0) as [Hr|Hr]; [|discriminate].
 destruct (Z.even (bv_p u)) eqn:He; [|discriminate]. cbn [andb] in H.
 apply Zeven_bool_iff, Zeven_ex in He. destruct He as [m Hm].
 assert (Hm2 : bv_p u / 2 = m) by (rewrite Hm, Z.mul_comm, Z.div_mul; lia).
 destruct (Z.eqb_spec (bv_q u) 1) as [Hq|Hq].
 - injection H as <-. unfold bs_eval. cbn [bv_k bv_p bv_q bv_r bv_j]. rewrite Hr, Hq, Hm2, Hm.
   replace (bv_k u + 2 * m * L + 1 * D + 0 * ((D + bv_j u) / 2)) with (m * L * 2 + (D + bv_k u)) by ring.
   rewrite Z.div_add_l by lia. ring.
 - destruct (Z.eqb_spec (bv_q u) 0) as [Hq0|Hq0]; [|discriminate].
   injection H as <-. rewrite bs_lin_eval. unfold bs_eval. rewrite Hr, Hq0, Hm2, Hm.
   replace (bv_k u + 2 * m * L + 0 * D + 0 * ((D + bv_j u) / 2)) with (m * L * 2 + bv_k u) by ring.
   rewrite Z.div_add_l by lia. ring.
Qed.

Lemma bs_half_step : forall x x' j, x <= x' -> 0 <= (x' + j) / 2 - (x + j) / 2 <= x' - x.
Proof.
 intros x x' j H.
 pose proof (Z.div_mod (x + j) 2 ltac:(lia)) as A. pose proof (Z.mod_pos_bound (x + j) 2 ltac:(lia)) as A'.
 pose proof (Z.div_mod (x' + j) 2 ltac:(lia)) as B. pose proof (Z.mod_pos_bound (x' + j) 2 ltac:(lia)) as B'.
 lia.
Qed.

Lemma bs_mono_le : forall q r j d d', d <= d' ->
 ((0 <= q /\ 0 <= r) \/ (r < 0 /\ 0 <= q + r)) -> bs_mono q r j d <= bs_mono q r j d'.
Proof.
 intros q r j d d' Hd Hc. unfold bs_mono. pose proof (bs_half_step d d' j Hd) as Hs.
 destruct Hc as [[Hq Hr]|[Hr Hqr]]; nia.
Qed.

Lemma bs_split_sound : forall q r j d1 d2 D, d1 <= D <= d2 ->
 fst (bs_split q r j d1 d2) <= bs_mono q r j D <= snd (bs_split q r j d1 d2).
Proof.
 intros q r j d1 d2 D HD. unfold bs_split, bs_mono. cbn [fst snd].
 pose proof (Z.div_mod (D + j) 2 ltac:(lia)) as Hdm. pose proof (Z.mod_pos_bound (D + j) 2 ltac:(lia)) as Hm.
 assert (Hf1 : (d1 + j) / 2 <= (D + j) / 2) by (apply Z.div_le_mono; lia).
 assert (Hf2 : (D + j) / 2 <= (d2 + j) / 2) by (apply Z.div_le_mono; lia).
 set (f := (D + j) / 2) in *. set (m := (D + j) mod 2) in *.
 set (f1 := (d1 + j) / 2) in *. set (f2 := (d2 + j) / 2) in *.
 assert (HD' : D = 2 * f + m - j) by lia.
 replace (q * D + r * f) with ((2 * q + r) * f + q * m - q * j) by (rewrite HD'; ring).
 assert (Hc : Z.min ((2 * q + r) * f1) ((2 * q + r) * f2) <= (2 * q + r) * f <= Z.max ((2 * q + r) * f1) ((2 * q + r) * f2)).
 { destruct (Z.le_gt_cases 0 (2 * q + r)) as [Hc|Hc].
   - assert ((2 * q + r) * f1 <= (2 * q + r) * f) by (apply Z.mul_le_mono_nonneg_l; lia).
     assert ((2 * q + r) * f <= (2 * q + r) * f2) by (apply Z.mul_le_mono_nonneg_l; lia). lia.
   - assert ((2 * q + r) * f2 <= (2 * q + r) * f) by (apply Z.mul_le_mono_nonpos_l; lia).
     assert ((2 * q + r) * f <= (2 * q + r) * f1) by (apply Z.mul_le_mono_nonpos_l; lia). lia. }
 assert (Hqm : Z.min 0 q <= q * m <= Z.max 0 q) by (assert (m = 0 \/ m = 1) as [-> | ->] by lia; lia).
 lia.
Qed.

Lemma bs_bounds_sound : forall v l1 l2 d1 d2 L D, l1 <= L <= l2 -> d1 <= D <= d2 ->
 fst (bs_bounds v l1 l2 d1 d2) <= bs_eval v L D <= snd (bs_bounds v l1 l2 d1 d2).
Proof.
 intros v l1 l2 d1 d2 L D HL HD. unfold bs_bounds, bs_eval.
 set (pl := if 0 <=? bv_p v then (bv_p v * l1, bv_p v * l2) else (bv_p v * l2, bv_p v * l1)).
 assert (HpL : fst pl <= bv_p v * L <= snd pl).
 { unfold pl. destruct (Z.leb_spec 0 (bv_p v)); cbn [fst snd]; nia. }
 pose proof (bs_split_sound (bv_q v) (bv_r v) (bv_j v) d1 d2 D HD) as Hs.
 set (sp := bs_split (bv_q v) (bv_r v) (bv_j v) d1 d2) in *. unfold bs_mono in Hs.
 destruct (((0 <=? bv_q v) && (0 <=? bv_r v)) || ((bv_r v <? 0) && (0 <=? bv_q v + bv_r v))) eqn:Hup.
 - assert (Hc : (0 <= bv_q v /\ 0 <= bv_r v) \/ (bv_r v < 0 /\ 0 <= bv_q v + bv_r v)).
   { rewrite orb_true_iff, !andb_true_iff, !Z.leb_le, Z.ltb_lt in Hup. tauto. }
   pose proof (bs_mono_le _ _ (bv_j v) d1 D (proj1 HD) Hc). pose proof (bs_mono_le _ _ (bv_j v) D d2 (proj2 HD) Hc).
   unfold bs_mono in *. cbn [fst snd]. lia.
 - destruct (((bv_q v <=? 0) && (bv_r v <=? 0)) || ((0 <? bv_r v) && (bv_q v + bv_r v <=? 0))) eqn:Hdn.
   + assert (Hc : (0 <= - bv_q v /\ 0 <= - bv_r v) \/ (- bv_r v < 0 /\ 0 <= - bv_q v + - bv_r v)).
     { rewrite orb_true_iff, !andb_true_iff, !Z.leb_le, Z.ltb_lt in Hdn. lia. }
     pose proof (bs_mono_le _ _ (bv_j v) d1 D (proj1 HD) Hc). pose proof (bs_mono_le _ _ (bv_j v) D d2 (proj2 HD) Hc).
     unfold bs_mono in *. cbn [fst snd]. lia.
   + cbn [fst snd]. lia.
Qed.

Lemma bs_double_eval : forall x L D, bs_eval (bs_double x) L D = 2 * bs_eval x L D.
Proof. intros x L D. unfold bs_eval, bs_double. cbn [bv_k bv_p bv_q bv_r bv_j]. ring. Qed.

(* Clearing the low bit of a w-bit value. *)
Lemma term_land_even : forall w a, (1 <= w)%nat -> 0 <= a < 2 ^ Z.of_nat w ->
 Z.land a (2 ^ Z.of_nat w - 2) = a - a mod 2.
Proof.
 intros w a Hw Ha. set (n := Z.of_nat w) in *.
 assert (E1 : 2 ^ n - 2 = Z.shiftl (Z.ones (n - 1)) 1).
 { rewrite Z.shiftl_mul_pow2, Z.ones_equiv by lia. change (2 ^ 1) with 2.
   replace n with (n - 1 + 1) at 1 by lia. rewrite Z.pow_add_r by lia. change (2 ^ 1) with 2. lia. }
 assert (E2 : a - a mod 2 = Z.shiftl (Z.shiftr a 1) 1).
 { rewrite Z.shiftl_mul_pow2, Z.shiftr_div_pow2 by lia. change (2 ^ 1) with 2.
   pose proof (Z.div_mod a 2 ltac:(lia)). lia. }
 rewrite E1, E2. apply Z.bits_inj'. intros i Hi. rewrite Z.land_spec.
 destruct (Z.eq_dec i 0) as [->|Hi0].
 - rewrite !Z.shiftl_spec_low by lia. apply andb_false_r.
 - rewrite !Z.shiftl_spec_high by lia. rewrite Z.shiftr_spec, Z.testbit_ones by lia.
   replace (i - 1 + 1) with i by lia.
   destruct (Z.ltb_spec (i - 1) (n - 1)) as [Hlt|Hge].
   + rewrite (proj2 (Z.leb_le 0 (i - 1)) ltac:(lia)). rewrite andb_true_r. reflexivity.
   + rewrite !andb_false_r. symmetry. rewrite <- (Z.mod_small a (2 ^ n)) by lia. apply Z.mod_pow2_bits_high. lia.
Qed.


Lemma bs_forget_find : forall env ins r,
 PM.find r (bs_forget env ins) = if existsb (Pos.eqb r) (loop_writes ins) then None else PM.find r env.
Proof.
 intros env ins r. unfold bs_forget. generalize (loop_writes ins) as ws. intros ws. revert env.
 induction ws as [|x ws IH]; intros env; [reflexivity|]. cbn [fold_left existsb].
 rewrite IH. destruct (Pos.eqb_spec r x) as [->|Hne].
 - cbn. destruct (existsb (Pos.eqb x) ws); [reflexivity|]. apply PM.grs.
 - cbn [orb]. destruct (existsb (Pos.eqb r) ws); [reflexivity|]. apply PM.gro. exact Hne.
Qed.

Lemma bs_keep_find : forall F bx rest d v r v',
 PM.find r (if Nat.eqb (wdn F d) (bx_w bx) && negb (is_cell F d) && bs_fits bx v then PM.add d v rest else rest) = Some v' ->
 (r = d /\ v' = v /\ wdn F d = bx_w bx /\ is_cell F d = false /\ bs_fits bx v = true) \/ PM.find r rest = Some v'.
Proof.
 intros F bx rest d v r v' H.
 destruct (Nat.eqb (wdn F d) (bx_w bx) && negb (is_cell F d) && bs_fits bx v) eqn:Hc; [|right; exact H].
 rewrite !andb_true_iff, Nat.eqb_eq, negb_true_iff in Hc. destruct Hc as [[Hw Hcell] Hfit].
 destruct (Pos.eq_dec r d) as [->|Hne].
 - rewrite PM.gss in H. injection H as <-. left. tauto.
 - rewrite PM.gso in H by exact Hne. right. exact H.
Qed.

Lemma bs_mul_range : forall c x x1 x2, x1 <= x <= x2 ->
 Z.min (c * x1) (c * x2) <= c * x <= Z.max (c * x1) (c * x2).
Proof. intros c x x1 x2 Hx. destruct (Z.le_gt_cases 0 c); nia. Qed.

Lemma bs_lin_one_sound : forall a b l1 l2 d1 d2 h1 h2 g L D, l1 <= L <= l2 -> d1 <= D <= d2 -> h1 <= L + D <= h2 ->
 fst (bs_lin_one a b l1 l2 d1 d2 h1 h2 g) <= a * L + b * D <= snd (bs_lin_one a b l1 l2 d1 d2 h1 h2 g).
Proof.
 intros a b l1 l2 d1 d2 h1 h2 g L D HL HD HH. unfold bs_lin_one. cbn [fst snd].
 pose proof (bs_mul_range (a - g) L l1 l2 HL). pose proof (bs_mul_range (b - g) D d1 d2 HD).
 pose proof (bs_mul_range g (L + D) h1 h2 HH).
 replace (a * L + b * D) with ((a - g) * L + (b - g) * D + g * (L + D)) by ring. lia.
Qed.

Lemma bs_lin_bounds_sound : forall a b l1 l2 d1 d2 h1 h2 L D, l1 <= L <= l2 -> d1 <= D <= d2 -> h1 <= L + D <= h2 ->
 fst (bs_lin_bounds a b l1 l2 d1 d2 h1 h2) <= a * L + b * D <= snd (bs_lin_bounds a b l1 l2 d1 d2 h1 h2).
Proof.
 intros a b l1 l2 d1 d2 h1 h2 L D HL HD HH. unfold bs_lin_bounds.
 pose proof (bs_lin_one_sound a b l1 l2 d1 d2 h1 h2 0 L D HL HD HH) as H0.
 pose proof (bs_lin_one_sound a b l1 l2 d1 d2 h1 h2 a L D HL HD HH) as H1.
 pose proof (bs_lin_one_sound a b l1 l2 d1 d2 h1 h2 b L D HL HD HH) as H2.
 destruct (bs_lin_one a b l1 l2 d1 d2 h1 h2 0) as [x0 y0], (bs_lin_one a b l1 l2 d1 d2 h1 h2 a) as [x1 y1],
   (bs_lin_one a b l1 l2 d1 d2 h1 h2 b) as [x2 y2]. cbn [fst snd] in *. lia.
Qed.

Lemma bs_hbounds_sound : forall v l1 l2 d1 d2 h1 h2 L D, l1 <= L <= l2 -> d1 <= D <= d2 -> h1 <= L + D <= h2 ->
 fst (bs_hbounds v l1 l2 d1 d2 h1 h2) <= bs_eval v L D <= snd (bs_hbounds v l1 l2 d1 d2 h1 h2).
Proof.
 intros v l1 l2 d1 d2 h1 h2 L D HL HD HH. unfold bs_hbounds.
 pose proof (bs_bounds_sound v l1 l2 d1 d2 L D HL HD) as Hb.
 pose proof (bs_lin_bounds_sound (2 * bv_p v) (2 * bv_q v + bv_r v) l1 l2 d1 d2 h1 h2 L D HL HD HH) as Hl.
 destruct (bs_bounds v l1 l2 d1 d2) as [x y].
 destruct (bs_lin_bounds (2 * bv_p v) (2 * bv_q v + bv_r v) l1 l2 d1 d2 h1 h2) as [u t]. cbn [fst snd] in *.
 set (r := bv_r v) in *. set (c := 2 * bv_k v + r * bv_j v).
 pose proof (Z.div_mod (D + bv_j v) 2 ltac:(lia)) as Hdm. pose proof (Z.mod_pos_bound (D + bv_j v) 2 ltac:(lia)) as Hm.
 set (f := (D + bv_j v) / 2) in *. set (mb := (D + bv_j v) mod 2) in *.
 assert (H2 : 2 * bs_eval v L D = c + (2 * bv_p v * L + (2 * bv_q v + r) * D) - r * mb).
 { unfold bs_eval, c. fold f r. lia. }
 assert (Hrm : Z.min r 0 <= r * mb <= Z.max r 0) by (assert (mb = 0 \/ mb = 1) as [-> | ->] by lia; lia).
 split.
 - apply Z.max_lub; [lia|].
   assert (Hneg : - bs_eval v L D <= (- (c + u - Z.max r 0)) / 2) by (apply Z.div_le_lower_bound; lia). lia.
 - apply Z.min_glb; [lia|]. apply Z.div_le_lower_bound; lia.
Qed.

Lemma bs_box_sound : forall bx v L D, bx_l1 bx <= L <= bx_l2 bx -> bx_d1 bx <= D <= bx_d2 bx ->
 bx_h1 bx <= L + D <= bx_h2 bx -> fst (bs_box_bounds bx v) <= bs_eval v L D <= snd (bs_box_bounds bx v).
Proof. intros bx v L D HL HD HH. exact (bs_hbounds_sound v _ _ _ _ _ _ L D HL HD HH). Qed.

Lemma bs_fits_range : forall bx v L D, bs_fits bx v = true ->
 bx_l1 bx <= L <= bx_l2 bx -> bx_d1 bx <= D <= bx_d2 bx -> bx_h1 bx <= L + D <= bx_h2 bx ->
 - 2 ^ (Z.of_nat (bx_w bx) - 1) <= bs_eval v L D < 2 ^ (Z.of_nat (bx_w bx) - 1).
Proof.
 intros bx v L D H HL HD HH. unfold bs_fits in H.
 pose proof (bs_box_sound bx v L D HL HD HH) as Hb.
 destruct (bs_box_bounds bx v) as [x y]. cbn [fst snd] in Hb.
 rewrite andb_true_iff, Z.leb_le, Z.ltb_lt in H. lia.
Qed.

Lemma bs_nonneg_sound : forall bx v L D, bs_nonneg bx v = true ->
 bx_l1 bx <= L <= bx_l2 bx -> bx_d1 bx <= D <= bx_d2 bx -> bx_h1 bx <= L + D <= bx_h2 bx -> 0 <= bs_eval v L D.
Proof.
 intros bx v L D H HL HD HH. unfold bs_nonneg in H. apply Z.leb_le in H.
 pose proof (bs_box_sound bx v L D HL HD HH) as Hb. lia.
Qed.

(* One step keeps every tracked register's exact value. *)
Lemma bs_step_sound : forall F prog E bx env ins t s s' L D,
 env_ok F E -> regs_ok F E (regs s) ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 bx_l1 bx <= L <= bx_l2 bx -> bx_d1 bx <= D <= bx_d2 bx -> bx_h1 bx <= L + D <= bx_h2 bx -> (1 <= bx_w bx <= 64)%nat ->
 (forall r v, PM.find r env = Some v ->
   wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (regs s r) = bs_eval v L D) ->
 forall r v, PM.find r (bs_step F bx env ins) = Some v ->
   wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (regs s' r) = bs_eval v L D.
Proof.
 intros F prog E bx env ins t s s' L D HE Hrs Hf Hs HL HD HH Hw Henv.
 set (w := bx_w bx) in *.
 assert (Hrest : forall r v, PM.find r (bs_forget env ins) = Some v ->
   wdn F r = w /\ is_cell F r = false /\ sext w (regs s' r) = bs_eval v L D).
 { intros r v H. rewrite bs_forget_find in H. destruct (existsb (Pos.eqb r) (loop_writes ins)) eqn:Hw0; [discriminate|].
   destruct (Henv r v H) as [Hwr [Hc Hv]]. split; [exact Hwr|]. split; [exact Hc|].
   rewrite (term_step_keeps F prog E t s s' ins r Hc Hf Hs); [exact Hv|]. exact Hw0. }
 assert (Hval : forall d v, (wdn F d = w -> bs_fits bx v = true -> sext w (regs s' d) = bs_eval v L D) ->
   forall r v', PM.find r (if Nat.eqb (wdn F d) w && negb (is_cell F d) && bs_fits bx v
     then PM.add d v (bs_forget env ins) else bs_forget env ins) = Some v' ->
   wdn F r = w /\ is_cell F r = false /\ sext w (regs s' r) = bs_eval v' L D).
 { intros d v Hd r v' H. destruct (bs_keep_find F bx _ d v r v' H) as [[-> [-> [Hwd [Hcd Hf0]]]]|H'].
   - split; [exact Hwd|]. split; [exact Hcd|exact (Hd Hwd Hf0)].
   - exact (Hrest r v' H'). }
 assert (Hrange : forall z, (- 2 ^ (Z.of_nat w - 1) <= z < 2 ^ (Z.of_nat w - 1)) -> sext w (trunc w z) = z).
 { intros z Hz. replace (trunc w z) with (trunc w (z + 0)) by (rewrite Z.add_0_r; reflexivity).
   apply (term_sext_sum w z 0 z ltac:(lia) Hz). rewrite Z.add_0_r. apply eqm_refl. }
 assert (Hopnd : forall o v, bs_opnd bx env o = Some v -> sext w (opv (regs s) o) = bs_eval v L D).
 { intros [r|z] v H; cbn [bs_opnd opv] in *.
   - exact (proj2 (proj2 (Henv r v H))).
   - injection H as <-. rewrite bs_lin_eval. unfold w. ring. }
 intros r v Hfind. unfold bs_step in Hfind. fold w in Hfind.
 destruct ins as [g d r0 | g d z | g op w' d a o |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ];
   try (exact (Hrest r v Hfind)); destruct g as [[neg p]|]; try (exact (Hrest r v Hfind)).
 - (* a copy *)
   destruct (PM.find r0 env) as [u|] eqn:Hu; [|exact (Hrest r v Hfind)].
   refine (Hval d u _ r v Hfind). intros Hwd _.
   destruct (Henv r0 u Hu) as [Hw0 [_ Hv0]].
   rewrite (term_step_mov F prog E t s s' d r0 Hf Hs), Hwd.
   rewrite trunc_small; [exact Hv0|]. pose proof (rs_range F E (regs s) Hrs r0) as Hb. rewrite Hw0 in Hb. exact Hb.
 - (* a constant *)
   refine (Hval d (bs_lin (sext w z) 0 0) _ r v Hfind). intros Hwd _.
   rewrite (term_step_imm F prog E t s s' d z Hf Hs), Hwd, term_sext_trunc, bs_lin_eval. ring.
 - (* arithmetic *)
   assert (Hbin : wdn F d = w -> regs s' d = trunc w (binop_eval op w' (regs s a) (opv (regs s) o))).
   { intros Hwd. rewrite (term_step_bin F prog E t s s' op w' d a o Hf Hs), Hwd. reflexivity. }
   destruct op; try (exact (Hrest r v Hfind)).
   + (* addition *)
     destruct (Nat.eqb_spec w' w) as [->|_]; [|exact (Hrest r v Hfind)].
     destruct (PM.find a env) as [u|] eqn:Hu; [|exact (Hrest r v Hfind)].
     destruct (bs_opnd bx env o) as [vo|] eqn:Ho; [|exact (Hrest r v Hfind)].
     destruct (bs_add u vo) as [x|] eqn:Hx; [|exact (Hrest r v Hfind)].
     refine (Hval d x _ r v Hfind). intros Hwd Hf0.
     rewrite (Hbin Hwd). cbn [binop_eval].
     pose proof (bs_fits_range bx x L D Hf0 HL HD HH) as Hxr. rewrite (bs_add_eval u vo x L D Hx) in Hxr |- *.
     apply term_sext_sum; [lia|exact Hxr|].
     apply eqm_add; [lia| |].
     * rewrite <- (proj2 (proj2 (Henv a u Hu))). apply eqm_sym, sext_eqm.
     * rewrite <- (Hopnd o vo Ho). apply eqm_sym, sext_eqm.
   + (* subtraction *)
     destruct (Nat.eqb_spec w' w) as [->|_]; [|exact (Hrest r v Hfind)].
     destruct (PM.find a env) as [u|] eqn:Hu; [|exact (Hrest r v Hfind)].
     destruct (bs_opnd bx env o) as [vo|] eqn:Ho; [|exact (Hrest r v Hfind)].
     destruct (bs_add u (bs_neg vo)) as [x|] eqn:Hx; [|exact (Hrest r v Hfind)].
     refine (Hval d x _ r v Hfind). intros Hwd Hf0.
     rewrite (Hbin Hwd). cbn [binop_eval].
     pose proof (bs_fits_range bx x L D Hf0 HL HD HH) as Hxr.
     rewrite (bs_add_eval u (bs_neg vo) x L D Hx), bs_neg_eval in Hxr |- *.
     replace (regs s a - opv (regs s) o) with (regs s a + - opv (regs s) o) by ring.
     apply term_sext_sum; [lia|exact Hxr|].
     apply eqm_add; [lia| |].
     * rewrite <- (proj2 (proj2 (Henv a u Hu))). apply eqm_sym, sext_eqm.
     * rewrite <- (Hopnd o vo Ho). apply eqm_opp; [lia|apply eqm_sym, sext_eqm].
   + (* logical shift right of a value at least 0: half, or the sign bit *)
     destruct o as [ro|k0]; [exact (Hrest r v Hfind)|].
     destruct (Nat.eqb w' w && Nat.leb 2 w) eqn:Hk; [|exact (Hrest r v Hfind)].
     rewrite andb_true_iff, Nat.eqb_eq, Nat.leb_le in Hk. destruct Hk as [-> Hw2].
     destruct (PM.find a env) as [u|] eqn:Hu; [|exact (Hrest r v Hfind)].
     destruct (bs_nonneg bx u) eqn:Hnn; [|exact (Hrest r v Hfind)].
     destruct (Henv a u Hu) as [Hwa [_ Hva]].
     pose proof (bs_nonneg_sound bx u L D Hnn HL HD HH) as Hpos.
     pose proof (rs_range F E (regs s) Hrs a) as Hra. rewrite Hwa in Hra.
     assert (Hsx : sext w (regs s a) = regs s a /\ regs s a < 2 ^ (Z.of_nat w - 1)).
     { assert (Hp2 : 2 ^ Z.of_nat w = 2 * 2 ^ (Z.of_nat w - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
       unfold sext in Hva |- *. rewrite trunc_small in Hva |- * by exact Hra.
       destruct (Z.ltb_spec (regs s a) (2 ^ (Z.of_nat w - 1))) as [Hl|Hg]; [split; [reflexivity|exact Hl]|lia]. }
     destruct Hsx as [Hsx Hsm].
     destruct (Z.eqb_spec k0 1) as [->|Hk1].
     * destruct (bs_half u) as [x|] eqn:Hx; [|exact (Hrest r v Hfind)].
       refine (Hval d x _ r v Hfind). intros Hwd Hf0.
       rewrite (Hbin Hwd). cbn [binop_eval opv].
       assert (Hs1 : shamt w 1 = 1) by (unfold shamt; rewrite trunc_small by (cbn; lia); lia).
       rewrite Hs1, (trunc_small w (regs s a)) by exact Hra. change (2 ^ 1) with 2.
       rewrite <- Hsx, Hva, <- (bs_half_eval u x L D Hx).
       apply Hrange. exact (bs_fits_range bx x L D Hf0 HL HD HH).
     * destruct (Z.eqb_spec k0 (Z.of_nat w - 1)) as [Hk0|_]; [|exact (Hrest r v Hfind)].
       refine (Hval d (bs_lin 0 0 0) _ r v Hfind). intros Hwd _.
       rewrite (Hbin Hwd). cbn [binop_eval opv]. rewrite bs_lin_eval.
       rewrite (trunc_small w (regs s a)) by exact Hra.
       assert (Hs1 : shamt w k0 = Z.of_nat w - 1) by (unfold shamt; rewrite trunc_small by (cbn; lia); lia).
       rewrite Hs1, Z.div_small by lia. rewrite (trunc_small w 0) by (pose proof (Z.pow_pos_nonneg 2 (Z.of_nat w)); lia).
       unfold sext. rewrite trunc_small by (pose proof (Z.pow_pos_nonneg 2 (Z.of_nat w)); lia).
       destruct (Z.ltb_spec 0 (2 ^ (Z.of_nat w - 1))); [ring|pose proof (Z.pow_pos_nonneg 2 (Z.of_nat w - 1)); lia].
   + (* arithmetic shift right: half, or the sign of a value at least 0 *)
     destruct o as [ro|k0]; [exact (Hrest r v Hfind)|].
     destruct (Nat.eqb w' w && Nat.leb 2 w) eqn:Hk; [|exact (Hrest r v Hfind)].
     rewrite andb_true_iff, Nat.eqb_eq, Nat.leb_le in Hk. destruct Hk as [-> Hw2].
     destruct (PM.find a env) as [u|] eqn:Hu; [|exact (Hrest r v Hfind)].
     destruct (Z.eqb_spec k0 1) as [->|Hk1].
     * destruct (bs_half u) as [x|] eqn:Hx; [|exact (Hrest r v Hfind)].
       refine (Hval d x _ r v Hfind). intros Hwd Hf0.
       rewrite (Hbin Hwd). cbn [binop_eval opv].
       assert (Hs1 : shamt w 1 = 1) by (unfold shamt; rewrite trunc_small by (cbn; lia); lia).
       rewrite Hs1. change (2 ^ 1) with 2.
       rewrite (proj2 (proj2 (Henv a u Hu))), <- (bs_half_eval u x L D Hx).
       apply Hrange. exact (bs_fits_range bx x L D Hf0 HL HD HH).
     * destruct (Z.eqb_spec k0 (Z.of_nat w - 1)) as [Hk0|_]; cbn [andb]; [|exact (Hrest r v Hfind)].
       destruct (bs_nonneg bx u) eqn:Hnn; [|exact (Hrest r v Hfind)].
       refine (Hval d (bs_lin 0 0 0) _ r v Hfind). intros Hwd _.
       rewrite (Hbin Hwd). cbn [binop_eval opv]. rewrite bs_lin_eval.
       destruct (Henv a u Hu) as [_ [_ Hva]].
       pose proof (bs_nonneg_sound bx u L D Hnn HL HD HH) as Hpos.
       pose proof (term_sext_bounds w (regs s a) ltac:(lia)) as Hb.
       assert (Hs1 : shamt w k0 = Z.of_nat w - 1) by (unfold shamt; rewrite trunc_small by (cbn; lia); lia).
       rewrite Hs1, Hva, Z.div_small by lia.
       rewrite (trunc_small w 0) by (pose proof (Z.pow_pos_nonneg 2 (Z.of_nat w)); lia).
       unfold sext. rewrite trunc_small by (pose proof (Z.pow_pos_nonneg 2 (Z.of_nat w)); lia).
       destruct (Z.ltb_spec 0 (2 ^ (Z.of_nat w - 1))); [ring|pose proof (Z.pow_pos_nonneg 2 (Z.of_nat w - 1)); lia].
   + (* clearing the low bit: twice the half *)
     destruct o as [ro|z]; [exact (Hrest r v Hfind)|].
     destruct (Nat.eqb w' w && Nat.leb 2 w && Z.eqb (trunc w z) (2 ^ Z.of_nat w - 2)) eqn:Hk; [|exact (Hrest r v Hfind)].
     rewrite !andb_true_iff, Nat.eqb_eq, Nat.leb_le, Z.eqb_eq in Hk. destruct Hk as [[-> Hw2] Hz].
     destruct (PM.find a env) as [u|] eqn:Hu; [|exact (Hrest r v Hfind)].
     destruct (bs_half u) as [x|] eqn:Hx; [|exact (Hrest r v Hfind)].
     refine (Hval d (bs_double x) _ r v Hfind). intros Hwd Hf0.
     rewrite (Hbin Hwd). cbn [binop_eval opv]. rewrite Hz.
     destruct (Henv a u Hu) as [Hwa [_ Hva]].
     pose proof (rs_range F E (regs s) Hrs a) as Hra. rewrite Hwa in Hra.
     rewrite (trunc_small w (regs s a)) by exact Hra.
     rewrite (term_land_even w (regs s a) ltac:(lia) Hra).
     pose proof (bs_fits_range bx (bs_double x) L D Hf0 HL HD HH) as Hfit.
     rewrite bs_double_eval, (bs_half_eval u x L D Hx) in Hfit |- *. rewrite <- Hva in Hfit |- *.
     set (A := regs s a) in *. set (V := sext w A) in *.
     assert (HV : V mod 2 = A mod 2).
     { unfold V, sext. rewrite trunc_small by exact Hra.
       destruct (Z.ltb (A) (2 ^ (Z.of_nat w - 1))); [reflexivity|].
       replace (2 ^ Z.of_nat w) with (2 * 2 ^ (Z.of_nat w - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
       replace (A - 2 * 2 ^ (Z.of_nat w - 1)) with (A + (- 2 ^ (Z.of_nat w - 1)) * 2) by ring.
       apply Z.mod_add. lia. }
     pose proof (Z.div_mod V 2 ltac:(lia)) as Hdm.
     replace (A - A mod 2) with (A + - (A mod 2)) by ring.
     apply term_sext_sum; [lia|exact Hfit|].
     replace (2 * (V / 2)) with (V + - (V mod 2)) by lia. rewrite HV.
     apply eqm_add; [lia|apply eqm_sym, sext_eqm|apply eqm_refl].
Qed.
Lemma bs_prefix_next : forall prog h b i ins, (h <= b)%nat -> fetch prog b i = Some ins ->
 bs_prefix prog h b (S i) = bs_prefix prog h b i ++ [ins].
Proof.
 intros prog h b i ins Hb Hf. unfold bs_prefix. rewrite <- app_assoc. f_equal.
 pose proof (term_slice_snoc prog b 0 i ins ltac:(lia) Hf) as H. unfold term_slice in H.
 rewrite !Nat.sub_0_r in H. cbn [skipn] in H. exact H.
Qed.

Lemma bs_prefix_block : forall prog h b, (h <= b)%nat ->
 bs_prefix prog h (S b) 0 = bs_prefix prog h b (block_len prog b).
Proof.
 intros prog h b Hb. unfold bs_prefix. rewrite firstn_O, app_nil_r.
 unfold block_len. rewrite firstn_all.
 replace (S b - h)%nat with (b - h + 1)%nat by lia. rewrite seq_app, map_app, concat_app.
 cbn [seq map concat]. replace (h + (b - h))%nat with b by lia. rewrite app_nil_r. reflexivity.
Qed.

Lemma term_step_setp : forall F prog E t s s' c w' sg dd ra rb,
 fetch prog (blk s) (idx s) = Some (ISetp None c w' sg dd ra rb) -> phase_step F prog E t s = Some s' ->
 regs s' dd = trunc (wdn F dd) (bool_z (cmp_eval c (extend w' sg (regs s ra)) (extend w' sg (opv (regs s) rb)))).
Proof.
 intros F prog E t s s' c w' sg dd ra rb Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs.
 injection Hs as <-. cbn [holds guard_of negb goto regs dest result]. unfold upd. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma ts_guard_min_regs : forall c ra rb lo hi m, ts_guard_min c ra rb lo hi = Some m ->
 (ra = lo /\ rb = hi) \/ (ra = hi /\ rb = lo).
Proof.
 intros c ra rb lo hi m H. unfold ts_guard_min in H.
 destruct (Pos.eqb_spec ra lo), (Pos.eqb_spec rb hi); cbn [andb] in H; [left; tauto| | |];
 destruct (Pos.eqb_spec ra hi), (Pos.eqb_spec rb lo); cbn [andb] in H; try discriminate; right; tauto.
Qed.

Lemma ts_guard_min_sound : forall c ra rb lo hi m w (rs : reg -> Z), ts_guard_min c ra rb lo hi = Some m ->
 cmp_eval c (sext w (rs ra)) (sext w (rs rb)) = true -> m <= sext w (rs hi) - sext w (rs lo).
Proof.
 intros c ra rb lo hi m w rs H Hc. unfold ts_guard_min in H.
 destruct (Pos.eqb ra lo && Pos.eqb rb hi) eqn:H1.
 - apply andb_true_iff in H1. destruct H1 as [H1 H2]. apply Pos.eqb_eq in H1, H2. subst.
   destruct c; try discriminate; injection H as <-; cbn [cmp_eval] in Hc;
     [apply Z.ltb_lt in Hc|apply Z.leb_le in Hc]; lia.
 - destruct (Pos.eqb ra hi && Pos.eqb rb lo) eqn:H2; [|discriminate].
   apply andb_true_iff in H2. destruct H2 as [H2 H3]. apply Pos.eqb_eq in H2, H3. subst.
   destruct c; try discriminate; injection H as <-; cbn [cmp_eval] in Hc;
     [apply Z.ltb_lt in Hc|apply Z.leb_le in Hc]; lia.
Qed.

(* The order of positions across one step that moves to the next position. *)
Lemma ts_order : forall prog bs is bs' is' X Y,
 ((bs' = bs /\ is' = S is) \/ (bs' = S bs /\ is' = 0%nat /\ S is = block_len prog bs)) ->
 (Y < block_len prog X)%nat ->
 (term_pos_lt X Y bs' is' <-> term_pos_le X Y bs is).
Proof.
 intros prog bs is bs' is' X Y Hs HY. unfold term_pos_lt, term_pos_le.
 destruct Hs as [[-> ->]|[-> [-> Hl]]]; [lia|].
 destruct (Nat.eq_dec X bs) as [->|]; lia.
Qed.

Lemma term_pos_lt_irrefl : forall b i, ~ term_pos_lt b i b i.
Proof. intros b i. unfold term_pos_lt. lia. Qed.

(* A step into the blocks h to l from a block outside them is covered by one
   of the states on the edges into them, unless that list has None. *)
Lemma term_entry_states_sound : forall F prog cert E h l tm s s' ins,
 check_scalar_certificate F prog cert = true -> launch_ok F E -> structured prog = true ->
 run' F prog E tm = Some s -> term_in h l (blk s) = false ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E tm s = Some s' -> term_in h l (blk s') = true ->
 (blk s' < List.length prog)%nat ->
 In None (term_entry_states F prog cert h l) \/
 exists F' os, In (Some (F',os)) (term_entry_states F prog cert h l) /\ term_same F F' /\ env_ok F' E /\
   regs_ok F' E (regs s') /\ state_ok F' (S tm) os (regs s').
Proof.
 intros F prog cert E h l tm s s' ins Hcert HE Hstr Hr Hout Hf Hs Hin' Hb'.
 destruct (fetch_block F prog E _ _ _ Hf) as [Hbs _].
 destruct (term_states_sound F prog cert E tm s Hcert HE Hr Hbs) as [_ [F' [st [Hst [Hsame [HE' [Hrs Hok]]]]]]].
 assert (Hs' : phase_step F' prog E tm s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
 destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hs) as [[Hbb _]|[Hi' [Hlen [Hlast [Hend|Htg]]]]].
 - exfalso. rewrite Hbb in Hin'. congruence.
 - lia.
 - assert (Hnb : is_bottom st = false).
   { destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm st _ Hbot Hok)|reflexivity]. }
   assert (Hidx : idx s = Nat.pred (block_len prog (blk s))) by lia.
   rewrite Hidx in Hst.
   assert (Hmem : forall z, In z (flat_map (fun x => let '(F'',st'') := x in
       if is_bottom st'' then [] else
       if negb (wf_instr F'' ins) then [None] else
       flat_map (fun e => let '(u,os) := e in
         if term_in h l u then match os with
           | Some q => if is_bottom q then [] else [Some (F'',q)]
           | None => [] end else [])
         (edge_states F'' prog (blk s) (transfer F'' st'' ins)))
       (term_states F prog cert (blk s) (Nat.pred (block_len prog (blk s))))) ->
     In z (term_entry_states F prog cert h l)).
   { intros z Hz. unfold term_entry_states. apply in_flat_map. exists (blk s).
     split; [apply in_seq; lia|]. rewrite Hout.
     assert (Hex : existsb (term_in h l) (edge_targets prog (blk s)) = true).
     { apply existsb_exists. exists (blk s'). split; [exact Htg|exact Hin']. }
     rewrite Hex. cbn [negb]. rewrite Hlast. exact Hz. }
   destruct (wf_instr F' ins) eqn:Hwf.
   + destruct (term_edge_sound F' prog E tm s s' st ins HE' Hrs Hok Hwf Hf Hlen Hs' Hi' Hb') as [os [Hos [Hok' Hrs']]].
     assert (Hosnb : is_bottom os = false).
     { destruct (is_bottom os) eqn:Hbot; [exfalso; exact (bottom_not_ok F' (S tm) os _ Hbot Hok')|reflexivity]. }
     right. exists F', os. split; [|tauto].
     apply Hmem. apply in_flat_map. exists (F',st). split; [exact Hst|].
     cbv beta iota. rewrite Hnb, Hwf. cbn [negb].
     apply in_flat_map. exists (blk s', Some os). split; [exact Hos|].
     cbv beta iota. rewrite Hin', Hosnb. left. reflexivity.
   + left. apply Hmem. apply in_flat_map. exists (F',st). split; [exact Hst|].
     cbv beta iota. rewrite Hnb, Hwf. left. reflexivity.
Qed.

Section Search.
Variable F : lfacts.
Variable prog : program.
Variable cert : scalar_certificate.
Variable E : env.
Variables l h : nat.
Variable d : term_search.
Hypothesis Hcert : check_scalar_certificate F prog cert = true.
Hypothesis HE : launch_ok F E.
Hypothesis Hshape : term_shape prog = true.
Hypothesis He : In (l,h) (term_edges prog).
Hypothesis Hok : term_search_ok F prog cert h l d = true.
Local Notation reach tm s := (run' F prog E tm = Some s).
Local Notation lo := (ts_lo d).
Local Notation hi := (ts_hi d).
Local Notation w := (ts_w d).
Local Notation q := (ts_q d).


Lemma ts_static :
 (forall v t0, (h <= v < l)%nat -> In t0 (edge_targets prog v) -> t0 = S v \/ (l < t0)%nat) /\
 lo <> hi /\ q <> lo /\ q <> hi /\ wdn F lo = w /\ wdn F hi = w /\ (2 <= w <= 64)%nat /\
 is_cell F lo = false /\ is_cell F hi = false /\ is_cell F q = false /\
 term_region_writes prog h l lo = [(ts_b1 d, ts_i1 d, ts_sel_lo d)] /\
 term_region_writes prog h l hi = [(ts_b2 d, ts_i2 d, ts_sel_hi d)] /\
 (forall b' i' ins, (h <= b' <= l)%nat -> fetch prog b' i' = Some ins -> term_writes q ins = true ->
   ~ (term_pos_lt (ts_b1 d) (ts_i1 d) b' i' /\ term_pos_lt b' i' (ts_b2 d) (ts_i2 d)) /\
   ~ (term_pos_lt (ts_b2 d) (ts_i2 d) b' i' /\ term_pos_lt b' i' (ts_b1 d) (ts_i1 d))).
Proof.
 pose proof Hok as H0. unfold term_search_ok in H0. rewrite !andb_true_iff in H0.
 destruct H0 as [[[[[[[[[[[[[Hst Hlh] Hql] Hqh] Hwl] Hwh] Hw2] Hw64] Hcl] Hch] Hcq] Hwr] _] _].
 split.
 { intros v t0 Hv Ht. rewrite short_all_same, forallb_forall in Hst.
   specialize (Hst v ltac:(apply in_seq; lia)). rewrite short_all_same, forallb_forall in Hst.
   specialize (Hst t0 Ht). rewrite orb_true_iff, Nat.eqb_eq, Nat.ltb_lt in Hst. exact Hst. }
 apply negb_true_iff, Pos.eqb_neq in Hlh, Hql, Hqh. apply Nat.eqb_eq in Hwl, Hwh. apply Nat.leb_le in Hw2, Hw64.
 apply negb_true_iff in Hcl, Hch, Hcq.
 do 5 (split; [assumption|]). split; [lia|]. do 3 (split; [assumption|]).
 destruct (term_region_writes prog h l lo) as [|[[b1 i1] s1] [|]] eqn:Hrl; try discriminate.
 destruct (term_region_writes prog h l hi) as [|[[b2 i2] s2] [|]] eqn:Hrh; try discriminate.
 rewrite !andb_true_iff in Hwr. destruct Hwr as [[[[[[Hb1 Hi1] Hb2] Hi2] Hs1] Hs2] Hq].
 apply Nat.eqb_eq in Hb1, Hi1, Hb2, Hi2. subst b1 i1 b2 i2.
 destruct (loop_instr_eq_dec s1 (ts_sel_lo d)) as [->|]; [|discriminate].
 destruct (loop_instr_eq_dec s2 (ts_sel_hi d)) as [->|]; [|discriminate].
 split; [reflexivity|]. split; [reflexivity|].
 intros b' i' ins Hb Hf Hw. rewrite forallb_forall in Hq.
 assert (Hin : In (b',i',ins) (flat_map (fun v => map (fun x => (v, fst x, snd x)) (term_indexed prog v)) (seq h (S l - h)))).
 { apply in_flat_map. exists b'. split; [apply in_seq; lia|]. apply in_map_iff. exists (i',ins).
   split; [reflexivity|]. apply term_indexed_spec. exact Hf. }
 specialize (Hq _ Hin). cbv beta iota in Hq. rewrite Hw in Hq. cbn [andb negb] in Hq.
 apply negb_true_iff in Hq. rewrite orb_false_iff, !andb_false_iff in Hq.
 unfold pos_ltb in Hq. unfold term_pos_lt.
 destruct Hq as [Hq1 Hq2]. split.
 - intros [A B]. destruct Hq1 as [Hq1|Hq1]; rewrite orb_false_iff, andb_false_iff, Nat.ltb_ge, Nat.eqb_neq, Nat.ltb_ge in Hq1; lia.
 - intros [A B]. destruct Hq2 as [Hq2|Hq2]; rewrite orb_false_iff, andb_false_iff, Nat.ltb_ge, Nat.eqb_neq, Nat.ltb_ge in Hq2; lia.
Qed.

Lemma ts_guard : exists neg p c ra rb k0,
 last_instr (block_body prog l) = Some (IBra (Some (neg,p)) h) /\ is_cell F p = false /\ p <> lo /\ p <> hi /\
 term_last_write prog l (block_len prog l) p = Some (k0, ISetp None c w true p ra (OReg rb)) /\
 term_pos_lt (ts_b1 d) (ts_i1 d) l k0 /\ term_pos_lt (ts_b2 d) (ts_i2 d) l k0 /\
 ts_guard_min (if neg then flip c else c) ra rb lo hi = Some (ts_dmin d).
Proof.
 pose proof Hok as H0. unfold term_search_ok in H0. rewrite !andb_true_iff in H0.
 destruct H0 as [[_ Hg] _].
 destruct (term_edges_spec prog l h He) as [_ [_ [g Hlast]]]. rewrite Hlast in Hg.
 destruct g as [[neg p]|]; [|discriminate].
 rewrite !andb_true_iff in Hg. destruct Hg as [[[Hc Hpl] Hph] Hw].
 apply negb_true_iff in Hc. apply negb_true_iff, Pos.eqb_neq in Hpl, Hph.
 destruct (term_last_write prog l (block_len prog l) p) as [[k0 ins]|] eqn:Hlw; [|discriminate].
 destruct ins as [| | | | | | | g' c w' sg p' ra [rb|z]| | | | | | | | | | | | | | | | |]; try discriminate;
   destruct g' as [[? ?]|]; try discriminate; destruct sg; try discriminate.
 rewrite !andb_true_iff in Hw. destruct Hw as [[[[Hw' Hp'] H1] H2] Hm].
 apply Nat.eqb_eq in Hw'. apply Pos.eqb_eq in Hp'. subst w' p'.
 destruct (ts_guard_min (if neg then flip c else c) ra rb lo hi) as [m|] eqn:Hgm; [|discriminate].
 apply Z.eqb_eq in Hm. subst m.
 exists neg,p,c,ra,rb,k0. unfold pos_ltb in H1, H2. unfold term_pos_lt.
 rewrite orb_true_iff, andb_true_iff, Nat.ltb_lt, Nat.eqb_eq, Nat.ltb_lt in H1, H2.
 repeat split; try assumption; lia.
Qed.

Definition ts_bx (l1 l2 h1 h2 : Z) : bs_box := ts_box_of w (ts_dmin d) l1 l2 h1 h2.

Lemma ts_eval_sound : forall bx, ts_eval_ok F prog bx h d = true ->
 exists fx fy, PM.find (ts_x d) (bs_env_at F prog bx h (ts_b1 d) (ts_i1 d) lo hi) = Some fx /\
   PM.find (ts_y d) (bs_env_at F prog bx h (ts_b2 d) (ts_i2 d) lo hi) = Some fy /\
   forall L D, bx_l1 bx <= L <= bx_l2 bx -> bx_d1 bx <= D <= bx_d2 bx -> bx_h1 bx <= L + D <= bx_h2 bx ->
     1 <= bs_eval fx L D - L /\ 1 <= L + D - bs_eval fy L D.
Proof.
 intros bx Hr. unfold ts_eval_ok in Hr.
 destruct (PM.find (ts_x d) _) as [fx|] eqn:Hfx; [|discriminate].
 destruct (PM.find (ts_y d) _) as [fy|] eqn:Hfy; [|discriminate].
 exists fx,fy. split; [reflexivity|]. split; [reflexivity|].
 destruct (bs_add fx (bs_lin 0 (-1) 0)) as [ex|] eqn:Hex; [|discriminate].
 destruct (bs_add (bs_lin 0 1 1) (bs_neg fy)) as [ey|] eqn:Hey; [|discriminate].
 rewrite andb_true_iff, !Z.leb_le in Hr. destruct Hr as [Hxl Hyl].
 intros L D HL HD HH.
 pose proof (bs_box_sound bx ex L D HL HD HH) as Bx.
 pose proof (bs_box_sound bx ey L D HL HD HH) as By.
 rewrite (bs_add_eval _ _ _ L D Hex), bs_lin_eval in Bx.
 rewrite (bs_add_eval _ _ _ L D Hey), bs_lin_eval, bs_neg_eval in By. lia.
Qed.

Lemma ts_ranges : ts_cert_ok F prog cert h d = true ->
 ts_head_range F prog cert h lo w <> None /\ ts_head_range F prog cert h hi w <> None /\
 forall l1 l2 h1 h2, ts_head_range F prog cert h lo w = Some (Some (l1,l2)) ->
   ts_head_range F prog cert h hi w = Some (Some (h1,h2)) -> Z.max (ts_dmin d) (h1 - l2) <= h2 - l1 ->
   ts_eval_ok F prog (ts_bx l1 l2 h1 h2) h d = true.
Proof.
 intros Hr. unfold ts_cert_ok in Hr.
 destruct (ts_head_range F prog cert h lo w) as [[[l1 l2]|]|] eqn:Hl;
 destruct (ts_head_range F prog cert h hi w) as [[[h1 h2]|]|] eqn:Hh; try discriminate;
 (split; [discriminate|]); (split; [discriminate|]); intros l1' l2' h1' h2' Hl' Hh'; try discriminate.
 injection Hl' as <- <-. injection Hh' as <- <-. intros Hd.
 rewrite (proj2 (Z.leb_le _ _) Hd) in Hr. exact Hr.
Qed.

(* The bounds at the head, from the certificate's states there. *)
Lemma ts_head_in : forall t s r, reach t s -> blk s = h -> idx s = 0%nat -> wdn F r = w ->
 ts_head_range F prog cert h r w <> None ->
 exists u v, ts_head_range F prog cert h r w = Some (Some (u,v)) /\ u <= sext w (regs s r) <= v.
Proof.
 intros t s r Hr Hb Hi Hw Hn. destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 destruct (term_states_sound F prog cert E t s Hcert HE Hr ltac:(lia)) as [_ [F' [st [Hst [Hsame [HE' [Hrs Hok']]]]]]].
 rewrite Hb, Hi in Hst.
 assert (Hnb : is_bottom st = false).
 { destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' t st _ Hbot Hok')|reflexivity]. }
 set (L0 := map (fun x => let '(F'',st'') := x in
   if is_bottom st'' then Some (0,0) else cmp_range F'' st'' (OReg r) (Z.of_nat w) true) (term_states F prog cert h 0)).
 assert (Hin : In (cmp_range F' st (OReg r) (Z.of_nat w) true) L0).
 { subst L0. apply in_map_iff. exists (F',st). split; [cbv beta iota; rewrite Hnb; reflexivity|exact Hst]. }
 unfold ts_head_range in Hn |- *. fold L0 in Hn |- *.
 destruct (cmp_range F' st (OReg r) (Z.of_nat w) true) as [[u0 v0]|] eqn:Hc.
 - destruct (term_range_hull L0) as [[[u v]|]|] eqn:Hh.
   + exists u,v. split; [reflexivity|].
     destruct (term_hull_some L0 u v u0 v0 Hh Hin).
     pose proof (term_cmp_reg F' E t (regs s) st r w true u0 v0 HE' Hrs Hok' ltac:(rewrite (term_same_wdn F F' r Hsame); exact Hw) Hc) as Hb0.
     cbn [extend] in Hb0. lia.
   + apply term_hull_empty in Hh. rewrite Hh in Hin. contradiction.
   + contradiction.
 - exfalso. apply Hn. exact (term_hull_none L0 Hin).
Qed.

(* Inside the loop the run goes straight: each step moves to the next position. *)
Lemma ts_next : forall t s s' ins, reach t s -> term_in h l (blk s) = true ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 term_in h l (blk s') = true -> ~ (blk s' = h /\ idx s' = 0%nat) ->
 bs_prefix prog h (blk s') (idx s') = bs_prefix prog h (blk s) (idx s) ++ [ins] /\
 ((blk s' = blk s /\ idx s' = S (idx s)) \/ (blk s' = S (blk s) /\ idx s' = 0%nat /\ S (idx s) = block_len prog (blk s))).
Proof.
 intros t s s' ins Hr Hin Hf Hs Hin' Hnh. pose proof (term_shape_structured prog Hshape) as Hstr.
 pose proof ts_static as [Hstraight _].
 destruct (term_edges_spec prog l h He) as [Hl [Hhl [g Hlast]]].
 apply term_in_iff in Hin, Hin'.
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hb [Hi _]]|[Hi0 [Hlen [Hli [Hend|Htg]]]]].
 - rewrite Hb, Hi. split; [apply bs_prefix_next; [lia|exact Hf]|left; split; reflexivity].
 - lia.
 - destruct (Nat.eq_dec (blk s) l) as [Hbl|Hbl].
   + exfalso. unfold edge_targets in Htg. rewrite Hbl, Hlast in Htg.
     destruct g as [[? ?]|]; cbn in Htg; [destruct Htg as [Ht|[Ht|[]]]|destruct Htg as [Ht|[]]]; try lia.
     all: apply Hnh; split; [lia|exact Hi0].
   + destruct (Hstraight (blk s) (blk s') ltac:(lia) Htg) as [Hn|Hn]; [|lia].
     split.
     * rewrite Hn, Hi0, bs_prefix_block by lia. rewrite <- Hlen. apply bs_prefix_next; [lia|exact Hf].
     * right. split; [exact Hn|]. split; [exact Hi0|exact Hlen].
Qed.


Lemma ts_env : env_ok F E.
Proof. exact (env_of_launch F prog E (check_scalar_certificate_lf_ok F prog cert Hcert) HE). Qed.

Lemma ts_regs : forall t s, reach t s -> term_in h l (blk s) = true -> regs_ok F E (regs s).
Proof.
 intros t s Hr Hin. destruct (term_edges_spec prog l h He) as [Hl _].
 assert (Hb : (h <= blk s <= l)%nat) by (apply term_in_iff; exact Hin).
 destruct (term_states_sound F prog cert E t s Hcert HE Hr ltac:(lia)) as [_ [F' [st [_ [Hsame [_ [Hrs _]]]]]]].
 exact (proj1 (term_same_regs F F' E (regs s) Hsame) Hrs).
Qed.

Lemma ts_latch_step : forall tm s s' ins, fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E tm s = Some s' -> blk s = l -> blk s' = h -> idx s' = 0%nat ->
 S (idx s) = block_len prog l /\ last_instr (block_body prog l) = Some ins /\
 exists g, ins = IBra g h /\ holds g (regs s) = true /\ regs s' = regs s.
Proof.
 intros tm s s' ins Hf Hs Hbl Hbh Hi0. pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl [g Hg]]].
 destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hs) as [[_ [Hi _]]|[_ [Hlen [Hlast _]]]]; [lia|].
 rewrite Hbl in Hlen, Hlast. split; [exact Hlen|]. split; [exact Hlast|].
 rewrite Hg in Hlast. injection Hlast as <-. exists g. split; [reflexivity|].
 destruct (step_pos F prog E tm s s' (IBra g h) Hf Hs) as [[Hn [Hbra _]]|[[g' [tg [Hi' [Hg' Hp]]]]|[g' [Hi' _]]]].
 - exfalso. unfold next_pos in Hn. rewrite Hbl in Hn.
   destruct (Nat.ltb (S (idx s)) (block_len prog l)); injection Hn as Hb' Hi''; lia.
 - injection Hi' as <- _. split; [exact Hg'|]. exact (jump_regs F prog E tm s s' _ Hf eq_refl Hs).
 - discriminate.
Qed.

Lemma ts_pos_valid : fetch prog (ts_b1 d) (ts_i1 d) = Some (ts_sel_lo d) /\ fetch prog (ts_b2 d) (ts_i2 d) = Some (ts_sel_hi d) /\
 (h <= ts_b1 d <= l)%nat /\ (h <= ts_b2 d <= l)%nat.
Proof.
 pose proof ts_static as [_ [_ [_ [_ [_ [_ [_ [_ [_ [_ [Hrl [Hrh _]]]]]]]]]]]].
 assert (H1 : In (ts_b1 d, ts_i1 d, ts_sel_lo d) (term_region_writes prog h l lo)) by (rewrite Hrl; left; reflexivity).
 assert (H2 : In (ts_b2 d, ts_i2 d, ts_sel_hi d) (term_region_writes prog h l hi)) by (rewrite Hrh; left; reflexivity).
 apply term_region_writes_spec in H1, H2. tauto.
Qed.

Lemma ts_sel_lo_writes : forall r, term_writes r (ts_sel_lo d) = Pos.eqb r lo.
Proof. intros r. unfold ts_sel_lo, term_writes. destruct (ts_swap d); cbn; apply orb_false_r. Qed.

Lemma ts_sel_hi_writes : forall r, term_writes r (ts_sel_hi d) = Pos.eqb r hi.
Proof. intros r. unfold ts_sel_hi, term_writes. destruct (ts_swap d); cbn; apply orb_false_r. Qed.

Definition ts_takes (qv : Z) : bool := if ts_swap d then Z.eqb qv 0 else negb (Z.eqb qv 0).

Lemma ts_sel_lo_step : forall t s s', fetch prog (blk s) (idx s) = Some (ts_sel_lo d) ->
 phase_step F prog E t s = Some s' -> regs_ok F E (regs s) ->
 regs s' lo = if ts_takes (regs s q) then trunc w (regs s (ts_x d)) else regs s lo.
Proof.
 intros t s s' Hf Hs Hrs. pose proof ts_static as [_ [_ [_ [_ [Hwl _]]]]].
 assert (Hlo : trunc w (regs s lo) = regs s lo)
   by (apply trunc_small; rewrite <- Hwl; exact (rs_range F E (regs s) Hrs lo)).
 unfold ts_sel_lo in Hf. unfold ts_takes.
 destruct (ts_swap d); rewrite (term_step_selp F prog E t s s' _ _ _ _ _ Hf Hs), Hwl; cbn [opv];
   destruct (Z.eqb (regs s q) 0); cbn [negb]; [reflexivity|exact Hlo|exact Hlo|reflexivity].
Qed.

Lemma ts_sel_hi_step : forall t s s', fetch prog (blk s) (idx s) = Some (ts_sel_hi d) ->
 phase_step F prog E t s = Some s' -> regs_ok F E (regs s) ->
 regs s' hi = if ts_takes (regs s q) then regs s hi else trunc w (regs s (ts_y d)).
Proof.
 intros t s s' Hf Hs Hrs. pose proof ts_static as [_ [_ [_ [_ [_ [Hwh _]]]]]].
 assert (Hhi : trunc w (regs s hi) = regs s hi)
   by (apply trunc_small; rewrite <- Hwh; exact (rs_range F E (regs s) Hrs hi)).
 unfold ts_sel_hi in Hf. unfold ts_takes.
 destruct (ts_swap d); rewrite (term_step_selp F prog E t s s' _ _ _ _ _ Hf Hs), Hwh; cbn [opv];
   destruct (Z.eqb (regs s q) 0); cbn [negb]; [exact Hhi|reflexivity|reflexivity|exact Hhi].
Qed.

Definition ts_L (s1 : state) : Z := sext w (regs s1 lo).
Definition ts_D (s1 : state) : Z := sext w (regs s1 hi) - sext w (regs s1 lo).

Definition ts_box (s1 : state) (l1 l2 h1 h2 : Z) : Prop :=
 l1 <= ts_L s1 <= l2 /\ Z.max (ts_dmin d) (h1 - l2) <= ts_D s1 <= h2 - l1 /\ h1 <= ts_L s1 + ts_D s1 <= h2.

(* At the start s1 of an iteration whose distance lets the back edge's
   comparison hold, a box holds whose walk proves the bounds' moves. *)
Definition ts_head_ok (s1 : state) : Prop :=
 ts_dmin d <= ts_D s1 -> exists l1 l2 h1 h2, ts_box s1 l1 l2 h1 h2 /\ ts_eval_ok F prog (ts_bx l1 l2 h1 h2) h d = true.

Definition ts_env_ok (bx : bs_box) (s1 s : state) : Prop :=
 forall r v, PM.find r (bs_env_at F prog bx h (blk s) (idx s) lo hi) = Some v ->
   wdn F r = w /\ is_cell F r = false /\ sext w (regs s r) = bs_eval v (ts_L s1) (ts_D s1).

Definition ts_val_ok (s1 : state) (r : reg) (b i : nat) (xv : Z) : Prop :=
 forall l1 l2 h1 h2 f, ts_box s1 l1 l2 h1 h2 ->
   PM.find r (bs_env_at F prog (ts_bx l1 l2 h1 h2) h b i lo hi) = Some f -> sext w xv = bs_eval f (ts_L s1) (ts_D s1).

Definition ts_after (b i : nat) (s : state) : Prop := term_pos_lt b i (blk s) (idx s).

(* The rank: the distance between the bounds while the back edge's
   comparison can hold, and a larger constant before. *)
Definition ts_rank (rs : reg -> Z) : nat :=
 let D := sext w (rs hi) - sext w (rs lo) in
 if ts_dmin d <=? D then Z.to_nat (D + 2 ^ 64) else Z.to_nat (2 ^ 66).

Section Iter.
Variables (neg : bool) (p : reg) (c : cmp) (ra rb : reg) (k0 : nat).
Hypothesis Hlast : last_instr (block_body prog l) = Some (IBra (Some (neg,p)) h).
Hypothesis Hpc : is_cell F p = false.
Hypothesis Hplo : p <> lo.
Hypothesis Hphi : p <> hi.
Hypothesis Hlw : term_last_write prog l (block_len prog l) p = Some (k0, ISetp None c w true p ra (OReg rb)).
Hypothesis Hk1 : term_pos_lt (ts_b1 d) (ts_i1 d) l k0.
Hypothesis Hk2 : term_pos_lt (ts_b2 d) (ts_i2 d) l k0.
Hypothesis Hgm : ts_guard_min (if neg then flip c else c) ra rb lo hi = Some (ts_dmin d).

(* An iteration from its start s1 at the head: the exact values that the
   evaluator keeps, the bounds before and after their selections, one value
   of q for both selections, and the back edge's comparison. *)
Record ts_inv (s1 : state) (tm : nat) (s : state) : Prop := {
 si_head : blk s1 = h /\ idx s1 = 0%nat /\ exists t1, reach t1 s1;
 si_env : forall l1 l2 h1 h2, ts_box s1 l1 l2 h1 h2 -> ts_env_ok (ts_bx l1 l2 h1 h2) s1 s;
 si_lo : ~ ts_after (ts_b1 d) (ts_i1 d) s -> regs s lo = regs s1 lo;
 si_hi : ~ ts_after (ts_b2 d) (ts_i2 d) s -> regs s hi = regs s1 hi;
 si_sel : exists qv xv yv,
   (ts_after (ts_b1 d) (ts_i1 d) s ->
     regs s lo = (if ts_takes qv then xv else regs s1 lo) /\ ts_val_ok s1 (ts_x d) (ts_b1 d) (ts_i1 d) xv) /\
   (ts_after (ts_b2 d) (ts_i2 d) s ->
     regs s hi = (if ts_takes qv then regs s1 hi else yv) /\ ts_val_ok s1 (ts_y d) (ts_b2 d) (ts_i2 d) yv) /\
   ((ts_after (ts_b1 d) (ts_i1 d) s /\ ~ ts_after (ts_b2 d) (ts_i2 d) s) \/
    (ts_after (ts_b2 d) (ts_i2 d) s /\ ~ ts_after (ts_b1 d) (ts_i1 d) s) -> regs s q = qv);
 si_guard : blk s = l -> (k0 < idx s)%nat ->
   regs s p = trunc (wdn F p) (bool_z (cmp_eval c (sext w (regs s ra)) (sext w (regs s rb))))
}.

Lemma ts_start : forall t s, reach t s -> blk s = h -> idx s = 0%nat -> ts_inv s t s.
Proof.
 intros t s Hr Hb Hi.
 pose proof ts_static as [_ [_ [_ [_ [Hwl [Hwh [_ [Hcl [Hch _]]]]]]]]].
 pose proof ts_pos_valid as [_ [_ [Hb1 Hb2]]].
 constructor.
 - split; [exact Hb|]. split; [exact Hi|]. exists t. exact Hr.
 - intros l1 l2 h1 h2 _ r v Hfind. unfold bs_env_at, bs_prefix in Hfind. rewrite Hb, Hi, Nat.sub_diag in Hfind.
   cbn [seq map List.concat firstn app fold_left] in Hfind. unfold bs_start in Hfind. unfold ts_L, ts_D.
   destruct (Pos.eq_dec r hi) as [->|Hnh].
   + rewrite PM.gss in Hfind. injection Hfind as <-. split; [exact Hwh|]. split; [exact Hch|]. rewrite bs_lin_eval. ring.
   + rewrite PM.gso in Hfind by exact Hnh. destruct (Pos.eq_dec r lo) as [->|Hnl].
     * rewrite PM.gss in Hfind. injection Hfind as <-. split; [exact Hwl|]. split; [exact Hcl|]. rewrite bs_lin_eval. ring.
     * rewrite PM.gso in Hfind by exact Hnl. rewrite PM.gempty in Hfind. discriminate.
 - intros _. reflexivity.
 - intros _. reflexivity.
 - exists 0, 0, 0. unfold ts_after. rewrite Hb, Hi. unfold term_pos_lt.
   split; [intros H; lia|]. split; [intros H; lia|]. intros [[H _]|[H _]]; lia.
 - intros Hbl Hk. lia.
Qed.

Section Step.
Variables (s1 : state) (tm : nat) (s s' : state) (ins : instr).
Hypothesis Hinv : ts_inv s1 tm s.
Hypothesis Hr : reach tm s.
Hypothesis Hin : term_in h l (blk s) = true.
Hypothesis Hf : fetch prog (blk s) (idx s) = Some ins.
Hypothesis Hs : phase_step F prog E tm s = Some s'.
Hypothesis Hin' : term_in h l (blk s') = true.
Hypothesis Hnh : ~ (blk s' = h /\ idx s' = 0%nat).

Lemma tss_next : bs_prefix prog h (blk s') (idx s') = bs_prefix prog h (blk s) (idx s) ++ [ins] /\
 ((blk s' = blk s /\ idx s' = S (idx s)) \/ (blk s' = S (blk s) /\ idx s' = 0%nat /\ S (idx s) = block_len prog (blk s))).
Proof. exact (ts_next tm s s' ins Hr Hin Hf Hs Hin' Hnh). Qed.

Lemma tss_keep : forall r, is_cell F r = false -> term_writes r ins = false -> regs s' r = regs s r.
Proof. intros r Hc Hw. exact (term_step_keeps F prog E tm s s' ins r Hc Hf Hs Hw). Qed.

Lemma tss_lo : ~ (blk s = ts_b1 d /\ idx s = ts_i1 d) -> regs s' lo = regs s lo.
Proof.
 intros Hn. pose proof ts_static as [_ [_ [_ [_ [_ [_ [_ [Hcl [_ [_ [Hrl _]]]]]]]]]]].
 apply tss_keep; [exact Hcl|]. destruct (term_writes lo ins) eqn:Hw; [|reflexivity]. exfalso. apply Hn.
 assert (Hx : In (blk s, idx s, ins) (term_region_writes prog h l lo)).
 { apply term_region_writes_spec. split; [apply term_in_iff; exact Hin|]. split; [exact Hf|exact Hw]. }
 rewrite Hrl in Hx. destruct Hx as [Hx|[]]. injection Hx as Hx1 Hx2 _. split; congruence.
Qed.

Lemma tss_hi : ~ (blk s = ts_b2 d /\ idx s = ts_i2 d) -> regs s' hi = regs s hi.
Proof.
 intros Hn. pose proof ts_static as [_ [_ [_ [_ [_ [_ [_ [_ [Hch [_ [_ [Hrh _]]]]]]]]]]]].
 apply tss_keep; [exact Hch|]. destruct (term_writes hi ins) eqn:Hw; [|reflexivity]. exfalso. apply Hn.
 assert (Hx : In (blk s, idx s, ins) (term_region_writes prog h l hi)).
 { apply term_region_writes_spec. split; [apply term_in_iff; exact Hin|]. split; [exact Hf|exact Hw]. }
 rewrite Hrh in Hx. destruct Hx as [Hx|[]]. injection Hx as Hx1 Hx2 _. split; congruence.
Qed.

Lemma tss_q : (term_pos_lt (ts_b1 d) (ts_i1 d) (blk s) (idx s) /\ term_pos_lt (blk s) (idx s) (ts_b2 d) (ts_i2 d)) \/
 (term_pos_lt (ts_b2 d) (ts_i2 d) (blk s) (idx s) /\ term_pos_lt (blk s) (idx s) (ts_b1 d) (ts_i1 d)) ->
 regs s' q = regs s q.
Proof.
 intros Hbt. pose proof ts_static as [_ [_ [_ [_ [_ [_ [_ [_ [_ [Hcq [_ [_ Hqk]]]]]]]]]]]].
 apply tss_keep; [exact Hcq|]. destruct (term_writes q ins) eqn:Hw; [|reflexivity]. exfalso.
 assert (Hb : (h <= blk s <= l)%nat) by (apply term_in_iff; exact Hin).
 destruct (Hqk (blk s) (idx s) ins Hb Hf Hw) as [N1 N2]. destruct Hbt; tauto.
Qed.

Lemma tss_after1 : ts_after (ts_b1 d) (ts_i1 d) s' <-> term_pos_le (ts_b1 d) (ts_i1 d) (blk s) (idx s).
Proof.
 unfold ts_after. apply (ts_order prog (blk s) (idx s) (blk s') (idx s')); [exact (proj2 tss_next)|].
 exact (term_fetch_len _ _ _ _ (proj1 ts_pos_valid)).
Qed.

Lemma tss_after2 : ts_after (ts_b2 d) (ts_i2 d) s' <-> term_pos_le (ts_b2 d) (ts_i2 d) (blk s) (idx s).
Proof.
 unfold ts_after. apply (ts_order prog (blk s) (idx s) (blk s') (idx s')); [exact (proj2 tss_next)|].
 exact (term_fetch_len _ _ _ _ (proj1 (proj2 ts_pos_valid))).
Qed.

Lemma tss_ne : ~ (ts_b1 d = ts_b2 d /\ ts_i1 d = ts_i2 d).
Proof.
 pose proof ts_static as [_ [Hlh _]]. pose proof ts_pos_valid as [Hf1 [Hf2 _]].
 intros [E1 E2]. rewrite E1, E2, Hf2 in Hf1. injection Hf1 as Hsel.
 unfold ts_sel_lo, ts_sel_hi in Hsel. destruct (ts_swap d); injection Hsel; intros; congruence.
Qed.

Lemma tss_inv : ts_inv s1 (S tm) s'.
Proof.
 pose proof ts_static as [Hst [Hlh [Hql [Hqh [Hwl [Hwh [Hw [Hcl [Hch [Hcq [Hrl [Hrh Hqk]]]]]]]]]]]].
 pose proof ts_pos_valid as [Hf1 [Hf2 [Hb1 Hb2]]].
 pose proof tss_after1 as A1. pose proof tss_after2 as A2. pose proof tss_ne as Hne.
 pose proof (ts_regs tm s Hr Hin) as Hrs.
 destruct tss_next as [Hpre Hsucc].
 pose proof Hinv as [Hhead Henv Hlo Hhi Hsel Hg].
 constructor.
 - exact Hhead.
 - intros l1 l2 h1 h2 Hbox r v Hfind. unfold bs_env_at in Hfind. rewrite Hpre, fold_left_app in Hfind.
   cbn [fold_left] in Hfind. pose proof Hbox as [HL [HD HH]].
   exact (bs_step_sound F prog E (ts_bx l1 l2 h1 h2) (bs_env_at F prog (ts_bx l1 l2 h1 h2) h (blk s) (idx s) lo hi)
     ins tm s s' (ts_L s1) (ts_D s1) ts_env Hrs Hf Hs HL HD HH ltac:(unfold ts_bx, ts_box_of; cbn [bx_w]; lia)
     (Henv l1 l2 h1 h2 Hbox) r v Hfind).
 - intros Hna. assert (Hle : ~ term_pos_le (ts_b1 d) (ts_i1 d) (blk s) (idx s)) by (intros X; apply Hna, A1, X).
   rewrite tss_lo by (clear - Hle; unfold term_pos_le in Hle; lia). apply Hlo. clear - Hle. unfold ts_after, term_pos_le, term_pos_lt in Hle |- *. lia.
 - intros Hna. assert (Hle : ~ term_pos_le (ts_b2 d) (ts_i2 d) (blk s) (idx s)) by (intros X; apply Hna, A2, X).
   rewrite tss_hi by (clear - Hle; unfold term_pos_le in Hle; lia). apply Hhi. clear - Hle. unfold ts_after, term_pos_le, term_pos_lt in Hle |- *. lia.
 - destruct Hsel as [qv [xv [yv [S1 [S2 Q]]]]].
   assert (Hord : term_pos_lt (ts_b1 d) (ts_i1 d) (ts_b2 d) (ts_i2 d) \/ term_pos_lt (ts_b2 d) (ts_i2 d) (ts_b1 d) (ts_i1 d))
     by (clear - Hne; unfold term_pos_lt; lia).
   assert (Hat1 : {blk s = ts_b1 d /\ idx s = ts_i1 d} + {~ (blk s = ts_b1 d /\ idx s = ts_i1 d)}).
   { destruct (Nat.eq_dec (blk s) (ts_b1 d)), (Nat.eq_dec (idx s) (ts_i1 d)); [left|right|right|right]; tauto. }
   assert (Hat2 : {blk s = ts_b2 d /\ idx s = ts_i2 d} + {~ (blk s = ts_b2 d /\ idx s = ts_i2 d)}).
   { destruct (Nat.eq_dec (blk s) (ts_b2 d)), (Nat.eq_dec (idx s) (ts_i2 d)); [left|right|right|right]; tauto. }
   destruct Hat1 as [[E1 E2]|N1]; [|destruct Hat2 as [[E1 E2]|N2]].
   + (* the selection of lo *)
     assert (Hins : ins = ts_sel_lo d) by (rewrite E1, E2, Hf1 in Hf; congruence).
     pose proof (ts_sel_lo_step tm s s' ltac:(rewrite <- Hins; exact Hf) Hs Hrs) as Hnew.
     assert (Hlo1 : regs s lo = regs s1 lo) by (apply Hlo; clear - E1 E2; unfold ts_after, term_pos_lt; lia).
     assert (Hxv : ts_val_ok s1 (ts_x d) (ts_b1 d) (ts_i1 d) (trunc w (regs s (ts_x d)))).
     { intros l1 l2 h1 h2 fx Hbox Hfx. rewrite term_sext_trunc. rewrite <- E1, <- E2 in Hfx.
       exact (proj2 (proj2 (Henv l1 l2 h1 h2 Hbox (ts_x d) fx Hfx))). }
     assert (Hhi' : regs s' hi = regs s hi) by (apply tss_hi; intros [F1 F2]; apply Hne; split; congruence).
     assert (Hq' : regs s' q = regs s q).
     { apply tss_keep; [exact Hcq|]. rewrite Hins, ts_sel_lo_writes. apply Pos.eqb_neq. exact Hql. }
     destruct Hord as [O12|O21].
     * exists (regs s q), (trunc w (regs s (ts_x d))), yv. split; [|split].
       -- intros _. rewrite Hnew, Hlo1. split; [reflexivity|exact Hxv].
       -- intros Ha2. exfalso. apply A2 in Ha2. clear - Ha2 O12 E1 E2. unfold term_pos_le, term_pos_lt in Ha2, O12. lia.
       -- intros [[_ _]|[Ha2 _]]; [exact Hq'|]. exfalso. apply A2 in Ha2. clear - Ha2 O12 E1 E2. unfold term_pos_le, term_pos_lt in Ha2, O12. lia.
     * assert (Hqv : regs s q = qv).
       { apply Q. right. unfold ts_after. rewrite E1, E2. split; [exact O21|apply term_pos_lt_irrefl]. }
       exists qv, (trunc w (regs s (ts_x d))), yv. split; [|split].
       -- intros _. rewrite Hnew, Hlo1, Hqv. split; [reflexivity|exact Hxv].
       -- intros _. rewrite Hhi'. apply S2. unfold ts_after. rewrite E1, E2. exact O21.
       -- intros [[_ Hn2]|[_ Hn1]]; exfalso; [apply Hn2, A2|apply Hn1, A1]; clear - O21 E1 E2; unfold term_pos_le, term_pos_lt in O21 |- *; lia.
   + (* the selection of hi *)
     assert (Hins : ins = ts_sel_hi d) by (rewrite E1, E2, Hf2 in Hf; congruence).
     pose proof (ts_sel_hi_step tm s s' ltac:(rewrite <- Hins; exact Hf) Hs Hrs) as Hnew.
     assert (Hhi1 : regs s hi = regs s1 hi) by (apply Hhi; clear - E1 E2; unfold ts_after, term_pos_lt; lia).
     assert (Hyv : ts_val_ok s1 (ts_y d) (ts_b2 d) (ts_i2 d) (trunc w (regs s (ts_y d)))).
     { intros l1 l2 h1 h2 fy Hbox Hfy. rewrite term_sext_trunc. rewrite <- E1, <- E2 in Hfy.
       exact (proj2 (proj2 (Henv l1 l2 h1 h2 Hbox (ts_y d) fy Hfy))). }
     assert (Hlo' : regs s' lo = regs s lo) by (apply tss_lo; exact N1).
     assert (Hq' : regs s' q = regs s q).
     { apply tss_keep; [exact Hcq|]. rewrite Hins, ts_sel_hi_writes. apply Pos.eqb_neq. exact Hqh. }
     destruct Hord as [O12|O21].
     * assert (Hqv : regs s q = qv).
       { apply Q. left. unfold ts_after. rewrite E1, E2. split; [exact O12|apply term_pos_lt_irrefl]. }
       exists qv, xv, (trunc w (regs s (ts_y d))). split; [|split].
       -- intros _. rewrite Hlo'. apply S1. unfold ts_after. rewrite E1, E2. exact O12.
       -- intros _. rewrite Hnew, Hhi1, Hqv. split; [reflexivity|exact Hyv].
       -- intros [[_ Hn2]|[_ Hn1]]; exfalso; [apply Hn2, A2|apply Hn1, A1]; clear - O12 E1 E2; unfold term_pos_le, term_pos_lt in O12 |- *; lia.
     * exists (regs s q), xv, (trunc w (regs s (ts_y d))). split; [|split].
       -- intros Ha1. exfalso. apply A1 in Ha1. clear - Ha1 O21 E1 E2. unfold term_pos_le, term_pos_lt in Ha1, O21. lia.
       -- intros _. rewrite Hnew, Hhi1. split; [reflexivity|exact Hyv].
       -- intros [[Ha1 _]|[_ _]]; [|exact Hq']. exfalso. apply A1 in Ha1. clear - Ha1 O21 E1 E2. unfold term_pos_le, term_pos_lt in Ha1, O21. lia.
   + (* another instruction *)
     exists qv, xv, yv. split; [|split].
     * intros Ha1. apply A1 in Ha1. rewrite (tss_lo N1). apply S1. clear - Ha1 N1. unfold ts_after, term_pos_le, term_pos_lt in Ha1 |- *. lia.
     * intros Ha2. apply A2 in Ha2. rewrite (tss_hi N2). apply S2. clear - Ha2 N2. unfold ts_after, term_pos_le, term_pos_lt in Ha2 |- *. lia.
     * intros Hbt.
       assert (Hbt' : (term_pos_le (ts_b1 d) (ts_i1 d) (blk s) (idx s) /\ ~ term_pos_le (ts_b2 d) (ts_i2 d) (blk s) (idx s)) \/
         (term_pos_le (ts_b2 d) (ts_i2 d) (blk s) (idx s) /\ ~ term_pos_le (ts_b1 d) (ts_i1 d) (blk s) (idx s))).
       { destruct Hbt as [[Ha Hn]|[Ha Hn]]; [left|right]; split;
           [apply A1, Ha|intros X; apply Hn, A2, X|apply A2, Ha|intros X; apply Hn, A1, X]. }
       rewrite tss_q.
       -- apply Q. clear - Hbt' N1 N2. unfold ts_after, term_pos_le, term_pos_lt in Hbt' |- *. lia.
       -- clear - Hbt' N1 N2. unfold term_pos_le, term_pos_lt in Hbt' |- *. lia.
 - intros Hbl' Hk'.
   destruct Hsucc as [[Eb Ei]|[Eb [Ei _]]]; [|lia].
   pose proof (term_last_write_some prog l (block_len prog l) p k0 _ Hlw) as [Hkl [Hfk [_ Hlater]]].
   assert (Hbl : blk s = l) by congruence.
   assert (Hk0 : (k0 <= idx s)%nat) by (clear - Hk' Ei; lia).
   assert (Kl : ~ (blk s = ts_b1 d /\ idx s = ts_i1 d)) by (clear - Hk1 Hbl Hk0; unfold term_pos_lt in Hk1; lia).
   assert (Kh : ~ (blk s = ts_b2 d /\ idx s = ts_i2 d)) by (clear - Hk2 Hbl Hk0; unfold term_pos_lt in Hk2; lia).
   assert (Hra : regs s' ra = regs s ra /\ regs s' rb = regs s rb).
   { destruct (ts_guard_min_regs _ _ _ _ _ _ Hgm) as [[Ea Eb']|[Ea Eb']]; rewrite Ea, Eb';
       split; first [exact (tss_lo Kl)|exact (tss_hi Kh)]. }
   destruct Hra as [Hra Hrb]. rewrite Hra, Hrb.
   destruct (Nat.eq_dec (idx s) k0) as [Ek|Nk].
   + assert (Hins : ins = ISetp None c w true p ra (OReg rb)) by (rewrite Hbl, Ek, Hfk in Hf; congruence).
     rewrite (term_step_setp F prog E tm s s' c w true p _ _ ltac:(rewrite <- Hins; exact Hf) Hs). reflexivity.
   + assert (Hlt : (k0 < idx s)%nat) by (clear - Hk0 Nk; lia).
     rewrite (tss_keep p Hpc); [apply Hg; [exact Hbl|exact Hlt]|].
     apply (Hlater (idx s) ins); [|rewrite <- Hbl; exact Hf].
     split; [exact Hlt|rewrite <- Hbl; exact (term_fetch_len _ _ _ _ Hf)].
Qed.
End Step.


(* The back edge lowers the rank: the comparison bounds the new distance
   from below, and each selection moves one bound strictly inside. *)
(* Over a back edge from an iteration that starts at s1, the distance lets
   the comparison hold, and from a start whose distance does, lo does not
   fall, hi does not rise, and the distance falls. *)
Lemma ts_back_vals : forall s1 tm s s', ts_head_ok s1 -> ts_inv s1 tm s -> reach tm s -> blk s = l ->
 phase_step F prog E tm s = Some s' -> blk s' = h -> idx s' = 0%nat ->
 ts_dmin d <= ts_D s' /\
 (ts_dmin d <= ts_D s1 -> ts_L s1 <= ts_L s' /\ ts_L s' + ts_D s' <= ts_L s1 + ts_D s1 /\ ts_D s' < ts_D s1).
Proof.
 intros s1 tm s s' Hhead Hinv Hr Hbl Hs Hbh Hi0.
 pose proof ts_static as [_ [_ [_ [_ [Hwl [Hwh [Hw _]]]]]]].
 destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
 2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
 destruct (ts_latch_step tm s s' ins Hf Hs Hbl Hbh Hi0) as [Hlen [Hlast' [g [Hins [Htaken Hregs]]]]].
 assert (Hgeq : g = Some (neg,p)) by congruence. subst g.
 pose proof (term_last_write_some prog l (block_len prog l) p k0 _ Hlw) as [Hkl [Hfk _]].
 assert (Hk0 : (k0 < idx s)%nat).
 { destruct (Nat.eq_dec k0 (idx s)) as [Ek|Nk]; [|clear - Hkl Hlen Nk; lia].
   exfalso. rewrite <- Ek, Hbl, Hfk, Hins in Hf. discriminate. }
 pose proof Hinv as [[Hb1 [Hi1 [t1 Hr1]]] _ _ _ Hsel Hg].
 pose proof (Hg Hbl Hk0) as Hgv.
 assert (Htr : forall b, trunc (wdn F p) (bool_z b) = bool_z b).
 { intros b. pose proof (term_wd_range F E p ts_env) as Hwp. apply trunc_small.
   assert (2 ^ 1 <= 2 ^ Z.of_nat (wdn F p)) by (apply Z.pow_le_mono_r; lia).
   destruct b; cbn [bool_z]; lia. }
 assert (Hcmp : cmp_eval (if neg then flip c else c) (sext w (regs s ra)) (sext w (regs s rb)) = true).
 { cbn [holds] in Htaken. rewrite Hgv, Htr in Htaken.
   destruct neg, (cmp_eval c (sext w (regs s ra)) (sext w (regs s rb))) eqn:Hc; cbn in Htaken; try discriminate.
   - rewrite flip_eval, Hc. reflexivity.
   - reflexivity. }
 pose proof (ts_guard_min_sound _ _ _ _ _ _ w (regs s) Hgm Hcmp) as HDn.
 assert (Ha1 : ts_after (ts_b1 d) (ts_i1 d) s) by (clear - Hk1 Hbl Hk0; unfold ts_after, term_pos_lt in *; rewrite Hbl; lia).
 assert (Ha2 : ts_after (ts_b2 d) (ts_i2 d) s) by (clear - Hk2 Hbl Hk0; unfold ts_after, term_pos_lt in *; rewrite Hbl; lia).
 destruct Hsel as [qv [xv [yv [S1 [S2 _]]]]].
 destruct (S1 Ha1) as [Hlov Hxv]. destruct (S2 Ha2) as [Hhiv Hyv].
 unfold ts_D, ts_L. rewrite Hregs.
 split; [exact HDn|]. intros Hdo.
 destruct (Hhead Hdo) as [l1 [l2 [h1 [h2 [Hbx Hev]]]]].
 destruct (ts_eval_sound _ Hev) as [fx [fy [Hfx [Hfy Hineq]]]].
 pose proof (Hxv l1 l2 h1 h2 fx Hbx Hfx) as Ex. pose proof (Hyv l1 l2 h1 h2 fy Hbx Hfy) as Ey.
 pose proof Hbx as [HL [HD HH]].
 destruct (Hineq (ts_L s1) (ts_D s1)) as [I1 I2]; [unfold ts_bx, ts_box_of; cbn [bx_l1 bx_l2 bx_d1 bx_d2 bx_h1 bx_h2]; lia ..|].
 unfold ts_L, ts_D in Ex, Ey, I1, I2, Hdo.
 destruct (ts_takes qv); rewrite Hlov, Hhiv; clear - Ex Ey I1 I2 Hdo; lia.
Qed.

Lemma ts_back : forall s1 tm s s', ts_head_ok s1 -> ts_inv s1 tm s -> reach tm s -> blk s = l ->
 phase_step F prog E tm s = Some s' -> blk s' = h -> idx s' = 0%nat ->
 (ts_rank (regs s') < ts_rank (regs s1))%nat.
Proof.
 intros s1 tm s s' Hhead Hinv Hr Hbl Hs Hbh Hi0.
 pose proof ts_static as [_ [_ [_ [_ [_ [_ [Hw _]]]]]]].
 destruct (ts_back_vals s1 tm s s' Hhead Hinv Hr Hbl Hs Hbh Hi0) as [HDn Hmono].
 assert (M : 2 ^ (Z.of_nat w - 1) <= 2 ^ 63) by (apply Z.pow_le_mono_r; lia).
 assert (P64 : 2 ^ 64 = 2 * 2 ^ 63) by reflexivity.
 assert (P66 : 2 ^ 66 = 8 * 2 ^ 63) by reflexivity.
 pose proof (term_sext_bounds w (regs s' lo) ltac:(lia)) as B1.
 pose proof (term_sext_bounds w (regs s' hi) ltac:(lia)) as B2.
 pose proof (term_sext_bounds w (regs s1 lo) ltac:(lia)) as B3.
 pose proof (term_sext_bounds w (regs s1 hi) ltac:(lia)) as B4.
 unfold ts_rank. cbv zeta. unfold ts_D, ts_L in HDn, Hmono.
 destruct (Z.leb_spec (ts_dmin d) (sext w (regs s' hi) - sext w (regs s' lo))) as [_|Hc]; [|clear - HDn Hc; lia].
 destruct (Z.leb_spec (ts_dmin d) (sext w (regs s1 hi) - sext w (regs s1 lo))) as [Hdo|Hdo].
 - destruct (Hmono Hdo) as [_ [_ Hlt]]. apply Z2Nat.inj_lt; clear - Hlt B1 B2 B3 B4 M P64; lia.
 - apply Z2Nat.inj_lt; clear - Hdo B1 B2 B3 B4 M P64 P66; lia.
Qed.

(* The certificate's ranges at the head give the box of every start. *)
Lemma ts_cert_head : ts_cert_ok F prog cert h d = true ->
 forall t s, reach t s -> blk s = h -> idx s = 0%nat -> ts_head_ok s.
Proof.
 intros Hc t s Hr Hb Hi Hdo.
 pose proof ts_static as [_ [_ [_ [_ [Hwl [Hwh _]]]]]].
 destruct (ts_ranges Hc) as [Hn1 [Hn2 Hbox]].
 destruct (ts_head_in t s lo Hr Hb Hi Hwl Hn1) as [l1 [l2 [HR1 HL]]].
 destruct (ts_head_in t s hi Hr Hb Hi Hwh Hn2) as [h1 [h2 [HR2 HH]]].
 exists l1, l2, h1, h2. unfold ts_D, ts_L in *. split.
 - unfold ts_box, ts_L, ts_D. clear - HL HH Hdo. lia.
 - apply (Hbox l1 l2 h1 h2 HR1 HR2). clear - HL HH Hdo. lia.
Qed.

(* A step from the loop to the start of its head is the back edge from l. *)
Lemma ts_to_head : forall t s s' ins, term_in h l (blk s) = true -> fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E t s = Some s' -> blk s' = h -> idx s' = 0%nat -> blk s = l.
Proof.
 intros t s s' ins Hinl Hf Hs Hbh Hi0.
 pose proof ts_static as [Hst _]. apply term_in_iff in Hinl.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hbb [Hi _]]|[_ [_ [_ [Hend|Htg]]]]].
 - lia.
 - destruct (term_edges_spec prog l h He) as [Hl _]. lia.
 - destruct (Nat.eq_dec (blk s) l) as [Hb|Hb]; [exact Hb|].
   destruct (Hst (blk s) (blk s') ltac:(lia) Htg) as [Hn|Hn]; lia.
Qed.

Section Entry.
Variables A B : Z.
Hypothesis Hent : term_entered prog h l = true.
Hypothesis Hebox : ts_entry_box F prog cert h l lo hi w (ts_dmin d) = Some (A,B).
Hypothesis HAB : A + ts_dmin d <= B.
Hypothesis Heval : ts_eval_ok F prog (ts_bx A (B - ts_dmin d) (A + ts_dmin d) B) h d = true.

(* What every start of an iteration keeps: lo at least A, hi at most B, and
   a distance that lets the back edge's comparison hold. *)
Definition ts_kept (s1 : state) : Prop :=
 A <= ts_L s1 /\ ts_L s1 + ts_D s1 <= B /\ ts_dmin d <= ts_D s1.

Lemma ts_kept_head : forall s1, ts_kept s1 -> ts_head_ok s1.
Proof.
 intros s1 [H1 [H2 H3]] _. exists A, (B - ts_dmin d), (A + ts_dmin d), B.
 split; [|exact Heval]. unfold ts_box. lia.
Qed.

Lemma ts_entry_fold : forall xs A0 B0,
 fold_right (fun e acc => match e, acc with
   | Some (F',st), Some (A1,B1) =>
     match cmp_range F' st (OReg lo) (Z.of_nat w) true, cmp_range F' st (OReg hi) (Z.of_nat w) true with
     | Some (a1,a2), Some (b1,b2) => if ts_dmin d <=? b1 - a2 then Some (Z.min A1 a1, Z.max B1 b2) else None
     | _, _ => None end
   | _, _ => None end) (Some (2 ^ 64, - 2 ^ 64)) xs = Some (A0,B0) ->
 ~ In None xs /\ forall F' st, In (Some (F',st)) xs -> exists a1 a2 b1 b2,
   cmp_range F' st (OReg lo) (Z.of_nat w) true = Some (a1,a2) /\
   cmp_range F' st (OReg hi) (Z.of_nat w) true = Some (b1,b2) /\ ts_dmin d <= b1 - a2 /\ A0 <= a1 /\ b2 <= B0.
Proof.
 induction xs as [|e xs IH]; intros A0 B0 H.
 - split; [intros []|]. intros F' st [].
 - cbn [fold_right] in H.
   destruct e as [[F1 st1]|]; [|discriminate].
   destruct (fold_right _ _ xs) as [[A1 B1]|] eqn:Hf; [|discriminate].
   destruct (cmp_range F1 st1 (OReg lo) (Z.of_nat w) true) as [[a1 a2]|] eqn:Ha; [|discriminate].
   destruct (cmp_range F1 st1 (OReg hi) (Z.of_nat w) true) as [[b1 b2]|] eqn:Hb; [|discriminate].
   destruct (Z.leb_spec (ts_dmin d) (b1 - a2)) as [Hm|]; [|discriminate].
   injection H as <- <-. destruct (IH A1 B1 eq_refl) as [Hn Hall]. split.
   + intros [Hc|Hc]; [discriminate|contradiction].
   + intros F' st [Hc|Hc].
     * injection Hc as <- <-. exists a1, a2, b1, b2. repeat split; try assumption; lia.
     * destruct (Hall F' st Hc) as [u1 [u2 [v1 [v2 [Hu [Hv [Hd [HA HB]]]]]]]].
       exists u1, u2, v1, v2. repeat split; try assumption; lia.
Qed.

Lemma ts_kept_entry : forall tm s s' ins, reach tm s -> term_in h l (blk s) = false ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E tm s = Some s' -> term_in h l (blk s') = true ->
 ts_kept s'.
Proof.
 intros tm s s' ins Hr Hout Hf Hs Hin'.
 pose proof ts_static as [_ [_ [_ [_ [Hwl [Hwh [Hw _]]]]]]].
 destruct (term_edges_spec prog l h He) as [Hl _].
 unfold ts_entry_box in Hebox. destruct (ts_entry_fold _ A B Hebox) as [Hnone Hall].
 destruct (term_entry_states_sound F prog cert E h l tm s s' ins Hcert HE (term_shape_structured prog Hshape)
   Hr Hout Hf Hs Hin' ltac:(apply term_in_iff in Hin'; lia)) as [Hn|[F' [os [Hin [Hsame [HE' [Hrs' Hok']]]]]]].
 - contradiction.
 - destruct (Hall F' os Hin) as [a1 [a2 [b1 [b2 [Ha [Hb [Hd [HA HB]]]]]]]].
   pose proof (term_cmp_reg F' E (S tm) (regs s') os lo w true a1 a2 HE' Hrs' Hok'
     ltac:(rewrite (term_same_wdn F F' lo Hsame); exact Hwl) Ha) as Hlo.
   pose proof (term_cmp_reg F' E (S tm) (regs s') os hi w true b1 b2 HE' Hrs' Hok'
     ltac:(rewrite (term_same_wdn F F' hi Hsame); exact Hwh) Hb) as Hhi.
   cbn [extend] in Hlo, Hhi. unfold ts_kept, ts_L, ts_D. lia.
Qed.

Lemma ts_kept_back : forall s1 tm s s', ts_kept s1 -> ts_inv s1 tm s -> reach tm s -> blk s = l ->
 phase_step F prog E tm s = Some s' -> blk s' = h -> idx s' = 0%nat -> ts_kept s'.
Proof.
 intros s1 tm s s' Hk Hinv Hr Hbl Hs Hbh Hi0.
 destruct (ts_back_vals s1 tm s s' (ts_kept_head s1 Hk) Hinv Hr Hbl Hs Hbh Hi0) as [HDn Hmono].
 destruct Hk as [H1 [H2 H3]]. destruct (Hmono H3) as [M1 [M2 _]].
 unfold ts_kept. lia.
Qed.

(* Every reachable state in the loop lies in an iteration whose start keeps
   the box. *)
Lemma ts_global : forall tm s, reach tm s -> term_in h l (blk s) = true ->
 exists s1, ts_inv s1 tm s /\ ts_kept s1 /\ (blk s = h -> idx s = 0%nat -> s1 = s).
Proof.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 pose proof (term_shape_edge prog Hshape l h He) as Hh0.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 intro tm. induction tm as [|tm IH]; intros s' Hr' Hin'.
 - exfalso. unfold run' in Hr'. cbn [run] in Hr'. injection Hr' as <-. apply term_in_iff in Hin'. cbn in Hin'. lia.
 - rewrite term_run_succ in Hr'. destruct (run' F prog E tm) as [s|] eqn:Hr; [|discriminate].
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hr'. rewrite Hf in Hr'. discriminate. }
   assert (Hr2 : reach (S tm) s') by (rewrite term_run_succ, Hr; exact Hr').
   destruct (Nat.eq_dec (blk s') h) as [Hbh|Hbh]; [destruct (Nat.eq_dec (idx s') 0) as [Hi0|Hi0]|].
   + exists s'. split; [exact (ts_start (S tm) s' Hr2 Hbh Hi0)|]. split; [|intros _ _; reflexivity].
     destruct (term_in h l (blk s)) eqn:Hin.
     * destruct (IH s eq_refl Hin) as [s1 [Hinv [Hk _]]].
       exact (ts_kept_back s1 tm s s' Hk Hinv Hr (ts_to_head tm s s' ins Hin Hf Hr' Hbh Hi0) Hr' Hbh Hi0).
     * exact (ts_kept_entry tm s s' ins Hr Hin Hf Hr' Hin').
   + destruct (term_in h l (blk s)) eqn:Hin.
     * destruct (IH s eq_refl Hin) as [s1 [Hinv [Hk _]]].
       exists s1. split; [exact (tss_inv s1 tm s s' ins Hinv Hr Hin Hf Hr' Hin' ltac:(tauto))|].
       split; [exact Hk|intros _ Hi; contradiction].
     * exfalso. destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hr') as [[Hbb _]|[Hi' _]].
       -- rewrite Hbb in Hin'. congruence.
       -- apply Hi0. exact Hi'.
   + destruct (term_in h l (blk s)) eqn:Hin.
     * destruct (IH s eq_refl Hin) as [s1 [Hinv [Hk _]]].
       exists s1. split; [exact (tss_inv s1 tm s s' ins Hinv Hr Hin Hf Hr' Hin' ltac:(tauto))|].
       split; [exact Hk|intros Hb; contradiction].
     * exfalso. destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hr') as [[Hbb _]|[_ [_ [_ [Hend|Htg]]]]].
       -- rewrite Hbb in Hin'. congruence.
       -- apply term_in_iff in Hin'. lia.
       -- apply Hbh. apply (term_entered_head prog l h (blk s) (blk s') Hent); [|exact Hin|exact Htg|exact Hin'].
          exact (proj1 (fetch_block F prog E _ _ _ Hf)).
Qed.
End Entry.

Theorem ts_loop_proof_at : term_loop_proof F prog E h l.
Proof.
 assert (Hmode : forall t s, reach t s -> blk s = h -> idx s = 0%nat -> ts_head_ok s).
 { pose proof Hok as H0. unfold term_search_ok in H0. rewrite !andb_true_iff in H0. destruct H0 as [_ Hfin].
   destruct (ts_cert_ok F prog cert h d) eqn:Hc; [exact (ts_cert_head Hc)|].
   unfold ts_entry_ok in Hfin. cbv zeta in Hfin.
   destruct (term_entered prog h l) eqn:Hent; cbn [negb] in Hfin; [|discriminate].
   destruct (ts_entry_box F prog cert h l lo hi w (ts_dmin d)) as [[A B]|] eqn:Hebox; [|discriminate].
   destruct (Z.leb_spec (A + ts_dmin d) B) as [HAB|]; [|discriminate].
   intros t s Hr Hb Hi.
   destruct (term_edges_spec prog l h He) as [_ [Hhl _]].
   destruct (ts_global A B Hent Hebox HAB Hfin t s Hr ltac:(apply term_in_iff; lia)) as [s1 [_ [Hk Heq]]].
   rewrite <- (Heq Hb Hi). exact (ts_kept_head A B HAB Hfin s1 Hk). }
 exists ts_rank, (fun s1 t s => ts_inv s1 t s /\ ts_head_ok s1). split; [|split].
 - intros t s Hr Hb Hi. split; [exact (ts_start t s Hr Hb Hi)|exact (Hmode t s Hr Hb Hi)].
 - intros s1 t s s' [Hinv Hh] Hr Hin Hs Hin' Hnh.
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   split; [exact (tss_inv s1 t s s' ins Hinv Hr Hin Hf Hs Hin' Hnh)|exact Hh].
 - intros s1 t s s' [Hinv Hh] Hr Hinl Hs Hbh Hi0.
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   exact (ts_back s1 t s s' Hh Hinv Hr (ts_to_head t s s' ins Hinl Hf Hs Hbh Hi0) Hs Hbh Hi0).
Qed.
End Iter.
Theorem ts_loop_proof : term_loop_proof F prog E h l.
Proof.
 destruct ts_guard as [neg [p [c [ra [rb [k0 [Hlast [Hpc [Hplo [Hphi [Hlw [Hk1 [Hk2 Hgm]]]]]]]]]]]]].
 eapply ts_loop_proof_at; eassumption.
Qed.
End Search.

Theorem term_search_sound : forall F prog cert E l h dsc,
 check_scalar_certificate F prog cert = true -> launch_ok F E -> term_shape prog = true ->
 In (l,h) (term_edges prog) -> term_search_ok F prog cert h l dsc = true -> term_loop_proof F prog E h l.
Proof. intros F prog cert E l h dsc Hcert HE Hshape He Hok. eapply ts_loop_proof; eassumption. Qed.

(* A loop whose back edge is never taken ends without a rank. *)
Theorem term_dead_sound : forall F prog cert E l h,
 check_scalar_certificate F prog cert = true -> launch_ok F E -> term_shape prog = true ->
 In (l,h) (term_edges prog) -> term_dead_ok F prog cert h l = true -> term_loop_proof F prog E h l.
Proof.
 intros F prog cert E l h Hcert HE Hshape He Hdead.
 exists (fun _ => O), (fun _ _ _ => True). split; [intros; exact I|]. split; [intros; exact I|].
 intros s1 t s s' _ Hr Hinl Hs Hbh Hi0. exfalso.
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
 2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
 pose proof (term_step_head F prog E Hshape l h t s s' ins Hf Hs Hinl Hbh Hi0) as Hedge.
 assert (Hv : (h <= blk s <= l)%nat) by (apply term_in_iff; exact Hinl).
 pose proof (term_latches_in prog h l (blk s) Hedge Hv) as Hlat.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hbb [Hi _]]|[_ [Hlen [Hlast _]]]].
 { exfalso. rewrite Hbb in Hbh. lia. }
 destruct (term_states_sound F prog cert E t s Hcert HE Hr ltac:(lia)) as [_ [F' [st [Hst [Hsame [HE' [Hrs Hok]]]]]]].
 unfold term_dead_ok in Hdead. rewrite forallb_forall in Hdead. specialize (Hdead _ Hlat).
 rewrite Hlast in Hdead.
 assert (Hidx : idx s = Nat.pred (block_len prog (blk s))) by lia.
 rewrite Hidx in Hst.
 pose proof (term_short_all_in _ _ _ _ Hdead Hst) as Hx. cbv beta iota in Hx.
 assert (Hnb : is_bottom st = false).
 { destruct (is_bottom st) eqn:Hb; [exfalso; exact (bottom_not_ok F' t st _ Hb Hok)|reflexivity]. }
 rewrite Hnb in Hx. apply andb_true_iff in Hx. destruct Hx as [Hwf Hedges].
 assert (Hs' : phase_step F' prog E t s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hs).
 destruct (term_edge_sound F' prog E t s s' st ins HE' Hrs Hok Hwf Hf Hlen Hs' Hi0 ltac:(lia)) as [os [Hin [Hos _]]].
 rewrite forallb_forall in Hedges. specialize (Hedges _ Hin). cbv beta iota in Hedges.
 rewrite Hbh, Nat.eqb_refl in Hedges. exact (bottom_not_ok F' (S t) os _ Hedges Hos).
Qed.

(* An accepted kernel ends: every thread of every launch reaches a position
   without an instruction. *)
(* ------------------------------------------------------------------ *)
(* Searches over a loop body that branches: soundness.                 *)

Definition ds_ok (F : lfacts) (bx : bs_box) (L D : Z) (rs : reg -> Z) (cs : ds_case) : Prop :=
 (forall r v, PM.find r (dc_env cs) = Some v ->
   wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (rs r) = bs_eval v L D) /\
 (forall p bv, In (p,bv) (dc_preds cs) -> is_cell F p = false /\ negb (Z.eqb (rs p) 0) = bv).

Lemma ds_pred_in : forall cs p bv, ds_pred cs p = Some bv -> In (p,bv) (dc_preds cs).
Proof.
 intros cs p bv H. unfold ds_pred in H.
 destruct (find (fun x => Pos.eqb (fst x) p) (dc_preds cs)) as [[p' v]|] eqn:Hf; [|discriminate].
 injection H as <-. apply find_some in Hf. destruct Hf as [Hin Hp]. cbn [fst] in Hp.
 apply Pos.eqb_eq in Hp. subst p'. exact Hin.
Qed.

Lemma ds_preds_step : forall F prog E t s s' ins ps,
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 (forall p bv, In (p,bv) ps -> is_cell F p = false /\ negb (Z.eqb (regs s p) 0) = bv) ->
 forall p bv, In (p,bv) (filter (fun x => negb (term_writes (fst x) ins)) ps) ->
   is_cell F p = false /\ negb (Z.eqb (regs s' p) 0) = bv.
Proof.
 intros F prog E t s s' ins ps Hf Hs Hps p bv Hin. apply filter_In in Hin. destruct Hin as [Hin Hw].
 cbn [fst] in Hw. apply negb_true_iff in Hw. destruct (Hps p bv Hin) as [Hc Hv]. split; [exact Hc|].
 rewrite (term_step_keeps F prog E t s s' ins p Hc Hf Hs Hw). exact Hv.
Qed.

Lemma ds_forget_step : forall F prog E t s s' ins bx env L D,
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 (forall r v, PM.find r env = Some v ->
   wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (regs s r) = bs_eval v L D) ->
 forall r v, PM.find r (bs_forget env ins) = Some v ->
   wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (regs s' r) = bs_eval v L D.
Proof.
 intros F prog E t s s' ins bx env L D Hf Hs Henv r v H. rewrite bs_forget_find in H.
 destruct (existsb (Pos.eqb r) (loop_writes ins)) eqn:Hwr; [discriminate|].
 destruct (Henv r v H) as [Hw1 [Hc1 Hv1]]. split; [exact Hw1|]. split; [exact Hc1|].
 rewrite (term_step_keeps F prog E t s s' ins r Hc1 Hf Hs); [exact Hv1|]. unfold term_writes. exact Hwr.
Qed.

Lemma ds_pick_sound : forall F bx env rest d0 o (rs rs' : reg -> Z) L D,
 (forall r v, PM.find r env = Some v -> wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (rs r) = bs_eval v L D) ->
 (forall r v, PM.find r rest = Some v -> wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (rs' r) = bs_eval v L D) ->
 rs' d0 = trunc (wdn F d0) (opv rs o) ->
 forall r v, PM.find r (ds_pick F bx env rest d0 o) = Some v ->
   wdn F r = bx_w bx /\ is_cell F r = false /\ sext (bx_w bx) (rs' r) = bs_eval v L D.
Proof.
 intros F bx env rest d0 o rs rs' L D Henv Hrest Hd r v H. unfold ds_pick in H.
 destruct (bs_opnd bx env o) as [v0|] eqn:Ho; [|exact (Hrest r v H)].
 destruct (Nat.eqb (wdn F d0) (bx_w bx) && negb (is_cell F d0)) eqn:Hc; [|exact (Hrest r v H)].
 rewrite andb_true_iff, Nat.eqb_eq, negb_true_iff in Hc. destruct Hc as [Hw Hcell].
 destruct (Pos.eq_dec r d0) as [->|Hne].
 - rewrite PM.gss in H. injection H as <-. split; [exact Hw|]. split; [exact Hcell|].
   rewrite Hd, Hw, term_sext_trunc. destruct o as [x|z]; cbn [bs_opnd opv] in Ho |- *.
   + exact (proj2 (proj2 (Henv x v0 Ho))).
   + injection Ho as <-. rewrite bs_lin_eval. ring.
 - rewrite PM.gso in H by exact Hne. exact (Hrest r v H).
Qed.

(* A step of the walk keeps a case that agrees with the registers. *)
Lemma ds_step_sound : forall F prog E bx cs ins t s s' L D,
 env_ok F E -> regs_ok F E (regs s) -> fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 bx_l1 bx <= L <= bx_l2 bx -> bx_d1 bx <= D <= bx_d2 bx -> bx_h1 bx <= L + D <= bx_h2 bx -> (1 <= bx_w bx <= 64)%nat ->
 ds_ok F bx L D (regs s) cs -> exists cs', In cs' (ds_step F bx cs ins) /\ ds_ok F bx L D (regs s') cs'.
Proof.
 intros F prog E bx cs ins t s s' L D HE Hrs Hf Hs HL HD HH Hw [Henv Hps].
 pose proof (ds_preds_step F prog E t s s' ins (dc_preds cs) Hf Hs Hps) as Hps'.
 pose proof (ds_forget_step F prog E t s s' ins bx (dc_env cs) L D Hf Hs Henv) as Hrest.
 assert (Hdef : ds_ok F bx L D (regs s')
   {| dc_env := bs_step F bx (dc_env cs) ins; dc_preds := filter (fun x => negb (term_writes (fst x) ins)) (dc_preds cs) |}).
 { split; [exact (bs_step_sound F prog E bx (dc_env cs) ins t s s' L D HE Hrs Hf Hs HL HD HH Hw Henv)|exact Hps']. }
 destruct ins as [| | | | | | | | | | |g w0 d0 a b p| | | | | | | | | | | | |];
   try (eexists; split; [left; reflexivity|exact Hdef]).
 destruct g as [[gn gp]|]; [eexists; split; [left; reflexivity|exact Hdef]|].
 pose proof (term_step_selp F prog E t s s' w0 d0 a b p Hf Hs) as Hsel.
 cbn [ds_step].
 destruct (ds_pred cs p) as [[|]|] eqn:Hp.
 - apply ds_pred_in in Hp. destruct (Hps p true Hp) as [_ Hpv].
   eexists. split; [left; reflexivity|]. split; [|exact Hps'].
   apply (ds_pick_sound F bx (dc_env cs) _ d0 a (regs s) (regs s') L D Henv Hrest).
   rewrite Hsel. destruct (Z.eqb (regs s p) 0); [discriminate|reflexivity].
 - apply ds_pred_in in Hp. destruct (Hps p false Hp) as [_ Hpv].
   eexists. split; [left; reflexivity|]. split; [|exact Hps'].
   apply (ds_pick_sound F bx (dc_env cs) _ d0 b (regs s) (regs s') L D Henv Hrest).
   rewrite Hsel. destruct (Z.eqb (regs s p) 0); [reflexivity|discriminate].
 - destruct (Pos.eqb p d0 || is_cell F p) eqn:Hpd.
   + eexists. split; [left; reflexivity|]. split; [exact Hrest|exact Hps'].
   + rewrite orb_false_iff, Pos.eqb_neq in Hpd. destruct Hpd as [Hpd Hcp].
     assert (Hpk : regs s' p = regs s p).
     { apply (term_step_keeps F prog E t s s' _ p Hcp Hf Hs). unfold term_writes. cbn [loop_writes dest existsb].
       rewrite (proj2 (Pos.eqb_neq p d0) Hpd). reflexivity. }
     destruct (Z.eqb_spec (regs s p) 0) as [Hz|Hz].
     * eexists. split; [right; left; reflexivity|]. split.
       -- apply (ds_pick_sound F bx (dc_env cs) _ d0 b (regs s) (regs s') L D Henv Hrest).
          rewrite Hsel. try rewrite (proj2 (Z.eqb_eq _ _) Hz). reflexivity.
       -- intros q bv [Hq|Hq]; [|exact (Hps' q bv Hq)]. injection Hq as <- <-.
          split; [exact Hcp|]. rewrite Hpk, Hz. reflexivity.
     * eexists. split; [left; reflexivity|]. split.
       -- apply (ds_pick_sound F bx (dc_env cs) _ d0 a (regs s) (regs s') L D Henv Hrest).
          rewrite Hsel. try rewrite (proj2 (Z.eqb_neq _ _) Hz). reflexivity.
       -- intros q bv [Hq|Hq]; [|exact (Hps' q bv Hq)]. injection Hq as <- <-.
          split; [exact Hcp|]. rewrite Hpk. destruct (Z.eqb_spec (regs s p) 0); [contradiction|reflexivity].
Qed.

Lemma ds_walk_snoc : forall F bx cases body ins,
 ds_walk F bx cases (body ++ [ins]) = flat_map (fun c => ds_step F bx c ins) (ds_walk F bx cases body).
Proof. intros. unfold ds_walk. rewrite fold_left_app. reflexivity. Qed.

Lemma ds_build_length : forall F bx prog h start n, List.length (ds_build F bx prog h start n) = n.
Proof.
 intros F bx prog h start n. induction n as [|n IH]; cbn [ds_build]; [reflexivity|].
 rewrite app_length, IH. cbn. lia.
Qed.

Lemma ds_build_nth : forall F bx prog h start n k, (k < n)%nat ->
 nth k (ds_build F bx prog h start n) [] = ds_entry_of F bx prog h start (ds_build F bx prog h start k) (h + k).
Proof.
 intros F bx prog h start n. induction n as [|n IH]; intros k Hk; [lia|].
 cbn [ds_build]. destruct (Nat.eq_dec k n) as [->|Hne].
 - rewrite app_nth2 by (rewrite ds_build_length; lia). rewrite ds_build_length, Nat.sub_diag. reflexivity.
 - rewrite app_nth1 by (rewrite ds_build_length; lia). apply IH. lia.
Qed.

Section DSearch.
Variable F : lfacts.
Variable prog : program.
Variable cert : scalar_certificate.
Variable E : env.
Variables l h : nat.
Variable d : term_dsearch.
Hypothesis Hcert : check_scalar_certificate F prog cert = true.
Hypothesis HE : launch_ok F E.
Hypothesis Hshape : term_shape prog = true.
Hypothesis He : In (l,h) (term_edges prog).
Hypothesis Hok : term_dsearch_ok F prog cert h l d = true.
Local Notation reach tm s := (run' F prog E tm = Some s).
Local Notation lo := (dsr_lo d).
Local Notation hi := (dsr_hi d).
Local Notation w := (wdn F lo).

Lemma ds_static :
 forallb (Nat.eqb l) (term_latches prog h l) = true /\
 (forall v t0, (h <= v <= l)%nat -> In t0 (edge_targets prog v) -> (v < t0)%nat \/ (v = l /\ t0 = h)) /\
 lo <> hi /\ wdn F hi = w /\ (2 <= w <= 64)%nat /\ is_cell F lo = false /\ is_cell F hi = false.
Proof.
 pose proof Hok as H0. unfold term_dsearch_ok in H0. rewrite !andb_true_iff in H0.
 destruct H0 as [[[[[[[[Hlat Hfw] Hlh] Hwh] Hw2] Hw64] Hcl] Hch] _].
 split; [exact Hlat|]. split.
 { intros v t0 Hv Ht. rewrite forallb_forall in Hfw. specialize (Hfw v ltac:(apply in_seq; lia)).
   rewrite forallb_forall in Hfw. specialize (Hfw t0 Ht).
   rewrite orb_true_iff, andb_true_iff, Nat.ltb_lt, !Nat.eqb_eq in Hfw. exact Hfw. }
 apply negb_true_iff, Pos.eqb_neq in Hlh. apply Nat.eqb_eq in Hwh. apply Nat.leb_le in Hw2, Hw64.
 apply negb_true_iff in Hcl, Hch. tauto.
Qed.

Lemma ds_rest : exists m neg p, ds_guard_min F prog l lo hi w = Some m /\
 last_instr (block_body prog l) = Some (IBra (Some (neg,p)) h) /\
 match ts_head_range F prog cert h lo w, ts_head_range F prog cert h hi w with
 | Some (Some (l1,l2)), Some (Some (h1,h2)) =>
   let bx := {| bx_w := w; bx_l1 := l1; bx_l2 := l2; bx_d1 := Z.max m (h1 - l2); bx_d2 := h2 - l1;
                bx_h1 := h1; bx_h2 := h2 |} in
   if Z.max m (h1 - l2) <=? h2 - l1 then
     forallb (ds_case_ok bx neg p lo hi)
       (ds_walk F bx (nth (l - h) (ds_build F bx prog h {| dc_env := bs_start lo hi; dc_preds := [] |} (S l - h)) [])
         (term_slice prog l 0 (Nat.pred (block_len prog l)))) = true
   else True
 | Some None, Some _ | Some (Some _), Some None => True
 | _, _ => False end.
Proof.
 pose proof Hok as H0. unfold term_dsearch_ok in H0. rewrite !andb_true_iff in H0. destruct H0 as [_ Hr].
 destruct (term_edges_spec prog l h He) as [_ [_ [g Hg]]].
 destruct (ds_guard_min F prog l lo hi w) as [m|]; [|discriminate].
 rewrite Hg in Hr |- *. destruct g as [[neg p]|]; [|discriminate].
 exists m, neg, p. split; [reflexivity|]. split; [reflexivity|].
 destruct (ts_head_range F prog cert h lo w) as [[[l1 l2]|]|], (ts_head_range F prog cert h hi w) as [[[h1 h2]|]|];
   try exact I; try discriminate.
 cbv zeta in Hr |- *. destruct (Z.max m (h1 - l2) <=? h2 - l1); [exact Hr|exact I].
Qed.

Lemma ds_env : env_ok F E.
Proof. exact (env_of_launch F prog E (check_scalar_certificate_lf_ok F prog cert Hcert) HE). Qed.

(* The back edge's guard bounds the distance at the end of the latch. *)
Lemma ds_back_min : forall m, ds_guard_min F prog l lo hi w = Some m ->
 forall tm s s', reach tm s -> blk s = l -> phase_step F prog E tm s = Some s' -> blk s' = h -> idx s' = 0%nat ->
 m <= sext w (regs s hi) - sext w (regs s lo).
Proof.
 intros m Hgm tm s s' Hr Hbl Hs Hbh Hi0.
 pose proof ds_static as [_ [_ [Hlh [Hwh [Hw [Hcl Hch]]]]]].
 pose proof (term_shape_structured prog Hshape) as Hstr.
 destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
 2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
 destruct (term_latch_jump F prog E l h tm s s' ins Hshape He Hf Hs Hbl Hbh Hi0) as [Hlen [Hlast [g [Hins [Htaken _]]]]].
 unfold ds_guard_min in Hgm. rewrite Hlast, Hins in Hgm.
 destruct g as [[neg p]|]; [|discriminate].
 destruct (term_last_write prog l (block_len prog l) p) as [[k0 [| | | | | | |gs cc ws sgs dds ras rbs| | | | | | | | | | | | | | | | |]]|] eqn:Hlw;
   try discriminate.
 destruct gs as [[? ?]|]; [discriminate|]. destruct sgs; [|discriminate]. destruct rbs as [rbs|?]; [|discriminate].
 destruct (Nat.eqb ws w && Pos.eqb dds p && negb (Pos.eqb ras p) && negb (Pos.eqb rbs p) &&
   negb (is_cell F p) && negb (is_cell F ras) && negb (is_cell F rbs)) eqn:Hc; [|discriminate].
 rewrite !andb_true_iff, Nat.eqb_eq, Pos.eqb_eq, !negb_true_iff, !Pos.eqb_neq in Hc.
 destruct Hc as [[[[[[-> ->] Hra] Hrb] Hcp] Hcra] Hcrb].
 destruct (term_last_write_some prog l _ p k0 _ Hlw) as [Hkl [Hfk [_ Hnop]]].
 assert (Hk0 : (k0 < idx s)%nat).
 { destruct (Nat.eq_dec k0 (idx s)) as [Ek|Nk]; [|lia].
   exfalso. rewrite Hbl, <- Ek, Hfk in Hf. injection Hf as Hf. rewrite Hins in Hf. discriminate. }
 (* the guard's value at the branch *)
 assert (Hafter : forall y, filter (fun z => Nat.ltb k0 (fst z) && term_writes y (snd z)) (term_indexed prog l) = [] ->
   forall k ins', (k0 < k)%nat -> fetch prog l k = Some ins' -> term_writes y ins' = false).
 { intros y Hy k ins' Hk Hfk'. destruct (term_writes y ins') eqn:Hw'; [|reflexivity]. exfalso.
   assert (Hin : In (k,ins') (filter (fun z => Nat.ltb k0 (fst z) && term_writes y (snd z)) (term_indexed prog l))).
   { apply filter_In. split; [apply term_indexed_spec; exact Hfk'|]. cbn [fst snd]. rewrite Hw'.
     rewrite (proj2 (Nat.ltb_lt k0 k) Hk). reflexivity. }
   rewrite Hy in Hin. exact Hin. }
 assert (Hends : forall y x, ds_ends prog l k0 y x = true -> wdn F y = w -> is_cell F y = false -> is_cell F x = false ->
   sext w (regs s y) = sext w (regs s x)).
 { intros y x Hed Hwy Hcy Hcx. unfold ds_ends in Hed.
   destruct (filter (fun z => Nat.ltb k0 (fst z) && term_writes x (snd z)) (term_indexed prog l)) eqn:Hx; [|discriminate].
   destruct (Pos.eqb_spec y x) as [->|Hyx]; [reflexivity|].
   destruct (filter (fun z => Nat.ltb k0 (fst z) && term_writes y (snd z)) (term_indexed prog l))
     as [|[m2 [| | | | | | | | | | | | | | | | | | | | | | | |]] [|]] eqn:Hy; try discriminate;
     destruct g as [[? ?]|]; try discriminate.
   rewrite andb_true_iff, !Pos.eqb_eq in Hed. destruct Hed as [-> ->].
   assert (Hm2 : In (m2, IMov None y x) (filter (fun z => Nat.ltb k0 (fst z) && term_writes y (snd z)) (term_indexed prog l)))
     by (rewrite Hy; left; reflexivity).
   apply filter_In in Hm2. destruct Hm2 as [Hm2 Hm2']. apply term_indexed_spec in Hm2. cbn [fst snd] in Hm2'.
   apply andb_true_iff in Hm2'. destruct Hm2' as [Hkm _]. apply Nat.ltb_lt in Hkm.
   assert (Hm2s : (m2 < idx s)%nat).
   { assert (Hml : (m2 < block_len prog l)%nat).
     { unfold fetch, block_len in *. apply nth_error_Some. rewrite Hm2. discriminate. }
     destruct (Nat.eq_dec m2 (idx s)) as [Em|Nm]; [|lia].
     exfalso. rewrite Hbl, <- Em, Hm2 in Hf. injection Hf as Hf. rewrite Hins in Hf. discriminate. }
   assert (Hval : regs s y = trunc (wdn F y) (regs s x)).
   { refine (term_hist F prog E (fun rs => rs y = trunc (wdn F y) (rs x)) [y; x] l m2 Hstr _ _ _ tm s Hr Hbl Hm2s _).
     - intros t0 s0 s0' Hb0 Hi0' Hs0. assert (Hf0 : fetch prog (blk s0) (idx s0) = Some (IMov None y x))
         by (rewrite Hb0, Hi0'; exact Hm2).
       rewrite (term_step_mov F prog E t0 s0 s0' y x Hf0 Hs0).
       rewrite (term_step_keeps F prog E t0 s0 s0' _ x Hcx Hf0 Hs0); [reflexivity|].
       unfold term_writes. cbn [loop_writes dest existsb]. rewrite (proj2 (Pos.eqb_neq x y) (fun H => Hyx (eq_sym H))). reflexivity.
     - intros rs rs' Hsame Hp0. rewrite (Hsame y (or_introl eq_refl)), (Hsame x (or_intror (or_introl eq_refl))). exact Hp0.
     - intros r [<-|[<-|[]]]; assumption.
     - intros k ins' Hk Hfk' r Hr'. destruct Hr' as [Hr'|[Hr'|[]]]; subst r.
       + destruct (term_writes y ins') eqn:Hw'; [|reflexivity]. exfalso.
         assert (Hin : In (k,ins') (filter (fun z => Nat.ltb k0 (fst z) && term_writes y (snd z)) (term_indexed prog l))).
         { apply filter_In. split; [apply term_indexed_spec; exact Hfk'|]. cbn [fst snd]. rewrite Hw'.
           rewrite (proj2 (Nat.ltb_lt k0 k) ltac:(lia)). reflexivity. }
         rewrite Hy in Hin. destruct Hin as [Hin|[]]. injection Hin as Hk' _. lia.
       + exact (Hafter x Hx k ins' ltac:(lia) Hfk'). }
   rewrite Hval, Hwy, term_sext_trunc. reflexivity. }
 assert (Hp : regs s p = trunc (wdn F p) (bool_z (cmp_eval cc (extend w true (regs s ras)) (extend w true (regs s rbs))))).
 { refine (term_hist F prog E (fun rs => rs p = trunc (wdn F p) (bool_z (cmp_eval cc (extend w true (rs ras)) (extend w true (rs rbs)))))
     [p; ras; rbs] l k0 Hstr _ _ _ tm s Hr Hbl Hk0 _).
   - intros t0 s0 s0' Hb0 Hi0' Hs0. assert (Hf0 : fetch prog (blk s0) (idx s0) = Some (ISetp None cc w true p ras (OReg rbs)))
       by (rewrite Hb0, Hi0'; exact Hfk).
     rewrite (term_step_setp F prog E t0 s0 s0' cc w true p ras (OReg rbs) Hf0 Hs0). cbn [opv].
     rewrite (term_step_keeps F prog E t0 s0 s0' _ ras Hcra Hf0 Hs0), (term_step_keeps F prog E t0 s0 s0' _ rbs Hcrb Hf0 Hs0);
       [reflexivity| |];
       unfold term_writes; cbn [loop_writes dest existsb]; [rewrite (proj2 (Pos.eqb_neq rbs p) Hrb)|rewrite (proj2 (Pos.eqb_neq ras p) Hra)];
       reflexivity.
   - intros rs rs' Hsame Hp0. rewrite (Hsame p (or_introl eq_refl)), (Hsame ras (or_intror (or_introl eq_refl))),
       (Hsame rbs (or_intror (or_intror (or_introl eq_refl)))). exact Hp0.
   - intros r [<-|[<-|[<-|[]]]]; assumption.
   - intros k ins' Hk Hfk' r Hr'. destruct Hr' as [Hr'|[Hr'|[Hr'|[]]]]; subst r.
     + apply (Hnop k ins'); [lia|exact Hfk'].
     + (* the operands are not written after k0 *)
       destruct (ds_ends prog l k0 lo ras && ds_ends prog l k0 hi rbs) eqn:E1;
       [rewrite andb_true_iff in E1; destruct E1 as [E1 _]; unfold ds_ends in E1;
        destruct (filter (fun z => Nat.ltb k0 (fst z) && term_writes ras (snd z)) (term_indexed prog l)) eqn:Hx; [|discriminate];
        exact (Hafter ras Hx k ins' ltac:(lia) Hfk')|].
       destruct (ds_ends prog l k0 hi ras && ds_ends prog l k0 lo rbs) eqn:E2; [|discriminate].
       rewrite andb_true_iff in E2. destruct E2 as [E2 _]. unfold ds_ends in E2.
       destruct (filter (fun z => Nat.ltb k0 (fst z) && term_writes ras (snd z)) (term_indexed prog l)) eqn:Hx; [|discriminate].
       exact (Hafter ras Hx k ins' ltac:(lia) Hfk').
     + destruct (ds_ends prog l k0 lo ras && ds_ends prog l k0 hi rbs) eqn:E1;
       [rewrite andb_true_iff in E1; destruct E1 as [_ E1]; unfold ds_ends in E1;
        destruct (filter (fun z => Nat.ltb k0 (fst z) && term_writes rbs (snd z)) (term_indexed prog l)) eqn:Hx; [|discriminate];
        exact (Hafter rbs Hx k ins' ltac:(lia) Hfk')|].
       destruct (ds_ends prog l k0 hi ras && ds_ends prog l k0 lo rbs) eqn:E2; [|discriminate].
       rewrite andb_true_iff in E2. destruct E2 as [_ E2]. unfold ds_ends in E2.
       destruct (filter (fun z => Nat.ltb k0 (fst z) && term_writes rbs (snd z)) (term_indexed prog l)) eqn:Hx; [|discriminate].
       exact (Hafter rbs Hx k ins' ltac:(lia) Hfk'). }
 assert (Htr : forall b, trunc (wdn F p) (bool_z b) = bool_z b).
 { intros b. pose proof (term_wd_range F E p ds_env) as Hwp. apply trunc_small.
   assert (2 ^ 1 <= 2 ^ Z.of_nat (wdn F p)) by (apply Z.pow_le_mono_r; lia).
   destruct b; cbn [bool_z]; lia. }
 assert (Hcmp : cmp_eval (if neg then flip cc else cc) (sext w (regs s ras)) (sext w (regs s rbs)) = true).
 { cbn [holds] in Htaken. rewrite Hp, Htr in Htaken. cbn [extend] in Htaken.
   destruct neg, (cmp_eval cc (sext w (regs s ras)) (sext w (regs s rbs))) eqn:Hcv; cbn in Htaken; try discriminate.
   - rewrite flip_eval, Hcv. reflexivity.
   - reflexivity. }
 destruct (ds_ends prog l k0 lo ras && ds_ends prog l k0 hi rbs) eqn:E1.
 - rewrite andb_true_iff in E1. destruct E1 as [E1 E2].
   assert (Hc' : cmp_eval (if neg then flip cc else cc) (sext w (regs s lo)) (sext w (regs s hi)) = true)
     by (rewrite (Hends lo ras E1 eq_refl Hcl Hcra), (Hends hi rbs E2 Hwh Hch Hcrb); exact Hcmp).
   exact (ts_guard_min_sound _ _ _ _ _ _ w (regs s) Hgm Hc').
 - destruct (ds_ends prog l k0 hi ras && ds_ends prog l k0 lo rbs) eqn:E2; [|discriminate].
   rewrite andb_true_iff in E2. destruct E2 as [E2 E3].
   assert (Hc' : cmp_eval (if neg then flip cc else cc) (sext w (regs s hi)) (sext w (regs s lo)) = true)
     by (rewrite (Hends hi ras E2 Hwh Hch Hcra), (Hends lo rbs E3 eq_refl Hcl Hcrb); exact Hcmp).
   exact (ts_guard_min_sound _ _ _ _ _ _ w (regs s) Hgm Hc').
Qed.

Lemma ds_regs : forall t s, reach t s -> term_in h l (blk s) = true -> regs_ok F E (regs s).
Proof.
 intros t s Hr Hin. destruct (term_edges_spec prog l h He) as [Hl _]. apply (term_in_iff l h) in Hin.
 destruct (term_states_sound F prog cert E t s Hcert HE Hr ltac:(lia)) as [_ [F' [_ [_ [Hsame [_ [Hrs _]]]]]]].
 exact (proj1 (term_same_regs F F' E (regs s) Hsame) Hrs).
Qed.

(* A reachable head has a state of the certificate, so its range of r is not empty. *)
Lemma ds_head_some : forall t s r, reach t s -> blk s = h -> idx s = 0%nat ->
 ts_head_range F prog cert h r w <> Some None.
Proof.
 intros t s r Hr Hb Hi Hn. destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 destruct (term_states_sound F prog cert E t s Hcert HE Hr ltac:(lia)) as [_ [F' [st [Hst _]]]].
 rewrite Hb, Hi in Hst. unfold ts_head_range in Hn. apply term_hull_empty in Hn.
 apply map_eq_nil in Hn. rewrite Hn in Hst. exact Hst.
Qed.

Lemma ds_head_in : forall t s r u v, reach t s -> blk s = h -> idx s = 0%nat -> wdn F r = w ->
 ts_head_range F prog cert h r w = Some (Some (u,v)) -> u <= sext w (regs s r) <= v.
Proof.
 intros t s r u v Hr Hb Hi Hw Hh. destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 destruct (term_states_sound F prog cert E t s Hcert HE Hr ltac:(lia)) as [_ [F' [st [Hst [Hsame [HE' [Hrs Hok']]]]]]].
 rewrite Hb, Hi in Hst.
 assert (Hnb : is_bottom st = false).
 { destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' t st _ Hbot Hok')|reflexivity]. }
 set (L0 := map (fun x => let '(F'',st'') := x in
   if is_bottom st'' then Some (0,0) else cmp_range F'' st'' (OReg r) (Z.of_nat w) true) (term_states F prog cert h 0)).
 assert (Hin : In (cmp_range F' st (OReg r) (Z.of_nat w) true) L0).
 { subst L0. apply in_map_iff. exists (F',st). split; [cbv beta iota; rewrite Hnb; reflexivity|exact Hst]. }
 unfold ts_head_range in Hh. fold L0 in Hh.
 destruct (cmp_range F' st (OReg r) (Z.of_nat w) true) as [[u0 v0]|] eqn:Hc.
 - destruct (term_hull_some L0 u v u0 v0 Hh Hin).
   pose proof (term_cmp_reg F' E t (regs s) st r w true u0 v0 HE' Hrs Hok' ltac:(rewrite (term_same_wdn F F' r Hsame); exact Hw) Hc) as Hb0.
   cbn [extend] in Hb0. lia.
 - rewrite (term_hull_none L0 Hin) in Hh. discriminate.
Qed.

(* The iteration's invariant, for the head's ranges [l1, l2] of lo and
   [h1, h2] of hi. *)
Section DMain.
Variables (m : Z) (neg : bool) (p : reg) (l1 l2 h1 h2 : Z).
Hypothesis Hgm : ds_guard_min F prog l lo hi w = Some m.
Hypothesis Hlast : last_instr (block_body prog l) = Some (IBra (Some (neg,p)) h).
Hypothesis HR1 : ts_head_range F prog cert h lo w = Some (Some (l1,l2)).
Hypothesis HR2 : ts_head_range F prog cert h hi w = Some (Some (h1,h2)).
Local Notation bx := {| bx_w := w; bx_l1 := l1; bx_l2 := l2; bx_d1 := Z.max m (h1 - l2); bx_d2 := h2 - l1;
 bx_h1 := h1; bx_h2 := h2 |}.
Local Notation start := {| dc_env := bs_start lo hi; dc_preds := [] |}.
Local Notation tbl := (ds_build F bx prog h start (S l - h)).
Hypothesis Hfin : Z.max m (h1 - l2) <= h2 - l1 ->
 forallb (ds_case_ok bx neg p lo hi)
   (ds_walk F bx (nth (l - h) tbl []) (term_slice prog l 0 (Nat.pred (block_len prog l)))) = true.

Definition ds_cases (b i : nat) : list ds_case := ds_walk F bx (nth (b - h) tbl []) (term_slice prog b 0 i).

Lemma ds_entry_eq : forall b, (h <= b <= l)%nat ->
 nth (b - h) tbl [] = ds_entry_of F bx prog h start (ds_build F bx prog h start (b - h)) b.
Proof. intros b Hb. rewrite ds_build_nth by lia. replace (h + (b - h))%nat with b by lia. reflexivity. Qed.

Lemma ds_cases_head : In start (ds_cases h 0).
Proof.
 destruct (term_edges_spec prog l h He) as [_ [Hhl _]].
 unfold ds_cases. rewrite term_slice_empty. unfold ds_walk. cbn [fold_left].
 rewrite ds_entry_eq by lia. unfold ds_entry_of. rewrite Nat.eqb_refl. left. reflexivity.
Qed.

Lemma ds_cases_next : forall b i ins cs cs', fetch prog b i = Some ins -> In cs (ds_cases b i) ->
 In cs' (ds_step F bx cs ins) -> In cs' (ds_cases b (S i)).
Proof.
 intros b i ins cs cs' Hf Hcs Hcs'. unfold ds_cases in *.
 rewrite (term_slice_snoc prog b 0 i ins (Nat.le_0_l i) Hf), ds_walk_snoc. apply in_flat_map. exists cs. tauto.
Qed.

Lemma ds_cases_edge : forall v b', (h <= v)%nat -> (v < b')%nat -> (b' <= l)%nat -> In b' (edge_targets prog v) ->
 forall cs, In cs (ds_cases v (block_len prog v)) -> In cs (ds_cases b' 0).
Proof.
 intros v b' Hv Hvb Hbl Hed cs Hcs. unfold ds_cases in *. rewrite term_slice_empty.
 replace (ds_walk F bx (nth (b' - h) tbl []) []) with (nth (b' - h) tbl []) by reflexivity.
 rewrite ds_entry_eq by lia. unfold ds_entry_of. rewrite (proj2 (Nat.eqb_neq b' h) ltac:(lia)).
 apply in_flat_map. exists v. split; [apply in_seq; lia|].
 assert (Hex : existsb (Nat.eqb b') (edge_targets prog v) = true)
   by (apply existsb_exists; exists b'; split; [exact Hed|apply Nat.eqb_refl]).
 rewrite Hex. rewrite ds_build_nth by lia. replace (h + (v - h))%nat with v by lia.
 rewrite <- ds_entry_eq by lia. rewrite term_slice_all in Hcs. cbn [skipn] in Hcs. exact Hcs.
Qed.

Definition ds_L (s1 : state) : Z := sext w (regs s1 lo).
Definition ds_D (s1 : state) : Z := sext w (regs s1 hi) - sext w (regs s1 lo).

Definition ds_inv (s1 : state) (tm : nat) (s : state) : Prop :=
 (blk s1 = h /\ idx s1 = 0%nat /\ exists t1, reach t1 s1) /\
 (m <= ds_D s1 -> exists cs, In cs (ds_cases (blk s) (idx s)) /\ ds_ok F bx (ds_L s1) (ds_D s1) (regs s) cs).

Definition ds_rank (rs : reg -> Z) : nat :=
 let D := sext w (rs hi) - sext w (rs lo) in
 if m <=? D then Z.to_nat (D + 2 ^ 64) else Z.to_nat (2 ^ 66).

Lemma ds_box : forall t1 s1, reach t1 s1 -> blk s1 = h -> idx s1 = 0%nat -> m <= ds_D s1 ->
 l1 <= ds_L s1 <= l2 /\ Z.max m (h1 - l2) <= ds_D s1 <= h2 - l1 /\ h1 <= ds_L s1 + ds_D s1 <= h2.
Proof.
 intros t1 s1 Hr Hb Hi Hm. pose proof ds_static as [_ [_ [_ [Hwh _]]]].
 pose proof (ds_head_in t1 s1 lo l1 l2 Hr Hb Hi eq_refl HR1) as HL.
 pose proof (ds_head_in t1 s1 hi h1 h2 Hr Hb Hi Hwh HR2) as HH.
 unfold ds_L, ds_D in *. lia.
Qed.

Lemma ds_start : forall t s, reach t s -> blk s = h -> idx s = 0%nat -> ds_inv s t s.
Proof.
 intros t s Hr Hb Hi. split; [split; [exact Hb|]; split; [exact Hi|exists t; exact Hr]|].
 intros Hm. exists start. split; [rewrite Hb, Hi; exact ds_cases_head|].
 pose proof ds_static as [_ [_ [_ [Hwh [_ [Hcl Hch]]]]]].
 split.
 - intros r v Hfind. unfold bs_start in Hfind. cbn [dc_env] in Hfind.
   destruct (Pos.eq_dec r hi) as [->|Hne].
   + rewrite PM.gss in Hfind. injection Hfind as <-. split; [exact Hwh|]. split; [exact Hch|].
     rewrite bs_lin_eval. unfold ds_L, ds_D. cbn [bx_w]. ring.
   + rewrite PM.gso in Hfind by exact Hne. destruct (Pos.eq_dec r lo) as [->|Hne2].
     * rewrite PM.gss in Hfind. injection Hfind as <-. split; [reflexivity|]. split; [exact Hcl|].
       rewrite bs_lin_eval. unfold ds_L. cbn [bx_w]. ring.
     * rewrite PM.gso in Hfind by exact Hne2. rewrite PM.gempty in Hfind. discriminate.
 - intros q bv [].
Qed.

Lemma ds_step_inv : forall s1 t s s' ins, ds_inv s1 t s -> reach t s -> term_in h l (blk s) = true ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 term_in h l (blk s') = true -> ~ (blk s' = h /\ idx s' = 0%nat) -> ds_inv s1 (S t) s'.
Proof.
 intros s1 t s s' ins [Hhead Hcase] Hr Hin Hf Hs Hin' Hnh. split; [exact Hhead|].
 intros Hm. destruct Hhead as [Hb1 [Hi1 [t1 Hr1]]].
 destruct (Hcase Hm) as [cs [Hcs Hok']].
 destruct (ds_box t1 s1 Hr1 Hb1 Hi1 Hm) as [HL [HD HH]].
 pose proof ds_static as [_ [Hfw [_ [_ [Hw _]]]]].
 destruct (term_edges_spec prog l h He) as [Hl [Hhl _]].
 destruct (ds_step_sound F prog E bx cs ins t s s' (ds_L s1) (ds_D s1) ds_env (ds_regs t s Hr Hin) Hf Hs HL HD HH
   ltac:(cbn [bx_w]; lia) Hok') as [cs' [Hin_cs' Hok'']].
 exists cs'. split; [|exact Hok''].
 pose proof (term_shape_structured prog Hshape) as Hstr.
 apply (term_in_iff l h) in Hin, Hin'.
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hbb [Hii _]]|[Hi0 [Hlen [_ [Hend|Htg]]]]].
 - rewrite Hbb, Hii. exact (ds_cases_next (blk s) (idx s) ins cs cs' Hf Hcs Hin_cs').
 - exfalso. lia.
 - destruct (Hfw (blk s) (blk s') Hin Htg) as [Hlt|[Hbl Hbh]]; [|exfalso; apply Hnh; split; assumption].
   rewrite Hi0. apply (ds_cases_edge (blk s) (blk s')); [lia|exact Hlt|lia|exact Htg|].
   rewrite <- Hlen. exact (ds_cases_next (blk s) (idx s) ins cs cs' Hf Hcs Hin_cs').
Qed.

Lemma ds_back : forall s1 t s s', ds_inv s1 t s -> reach t s -> term_in h l (blk s) = true ->
 phase_step F prog E t s = Some s' -> blk s' = h -> idx s' = 0%nat -> (ds_rank (regs s') < ds_rank (regs s1))%nat.
Proof.
 intros s1 t s s' [[Hb1 [Hi1 [t1 Hr1]]] Hcase] Hr Hinl Hs Hbh Hi0.
 pose proof ds_static as [Hlat [_ [_ [Hwh [Hw [Hcl Hch]]]]]].
 destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
 2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
 pose proof (term_step_head F prog E Hshape l h t s s' ins Hf Hs Hinl Hbh Hi0) as Hedge.
 assert (Hbl : blk s = l).
 { assert (Hv : (h <= blk s <= l)%nat) by (apply (term_in_iff l h); exact Hinl).
   pose proof (term_latches_in prog h l (blk s) Hedge Hv) as Hl0.
   rewrite forallb_forall in Hlat. apply Nat.eqb_eq. rewrite Nat.eqb_sym. exact (Hlat _ Hl0). }
 destruct (term_latch_jump F prog E l h t s s' ins Hshape He Hf Hs Hbl Hbh Hi0) as [Hlen [Hlast' [g [Hins [Htaken Hregs]]]]].
 pose proof (ds_back_min m Hgm t s s' Hr Hbl Hs Hbh Hi0) as HDn.
 assert (M : 2 ^ (Z.of_nat w - 1) <= 2 ^ 63) by (apply Z.pow_le_mono_r; lia).
 assert (P64 : 2 ^ 64 = 2 * 2 ^ 63) by reflexivity.
 assert (P66 : 2 ^ 66 = 8 * 2 ^ 63) by reflexivity.
 pose proof (term_sext_bounds w (regs s lo) ltac:(lia)) as B1.
 pose proof (term_sext_bounds w (regs s hi) ltac:(lia)) as B2.
 pose proof (term_sext_bounds w (regs s1 lo) ltac:(lia)) as B3.
 pose proof (term_sext_bounds w (regs s1 hi) ltac:(lia)) as B4.
 unfold ds_rank. rewrite Hregs. cbv zeta.
 destruct (Z.leb_spec m (sext w (regs s hi) - sext w (regs s lo))) as [_|Hc]; [|clear - HDn Hc; lia].
 destruct (Z.leb_spec m (sext w (regs s1 hi) - sext w (regs s1 lo))) as [Hdo|Hdo].
 - destruct (Hcase Hdo) as [cs [Hcs [Henv Hps]]].
   destruct (ds_box t1 s1 Hr1 Hb1 Hi1 Hdo) as [HL [HD HH]].
   assert (Hidx : idx s = Nat.pred (block_len prog l)) by lia.
   pose proof (Hfin ltac:(clear - HD; lia)) as Hall. rewrite forallb_forall in Hall.
   assert (Hcs' : In cs (ds_walk F bx (nth (l - h) tbl []) (term_slice prog l 0 (Nat.pred (block_len prog l)))))
     by (rewrite <- Hidx; unfold ds_cases in Hcs; rewrite Hbl in Hcs; exact Hcs).
   specialize (Hall cs Hcs'). unfold ds_case_ok in Hall.
   rewrite Hlast, Hins in Hlast'. injection Hlast' as Hg. rewrite <- Hg in Htaken.
   destruct (ds_pred cs p) as [bv|] eqn:Hp;
     [destruct (Bool.eqb_spec bv neg) as [Hbv|Hbv];
       [exfalso; apply ds_pred_in in Hp; destruct (Hps p bv Hp) as [_ Hpv];
        cbn [holds] in Htaken; rewrite Hpv, Hbv in Htaken; destruct neg; discriminate|]|].
   all: cbn [orb] in Hall.
   all: destruct (PM.find lo (dc_env cs)) as [vl|] eqn:Hvl; [|discriminate].
   all: destruct (PM.find hi (dc_env cs)) as [vh|] eqn:Hvh; [|discriminate].
   all: destruct (bs_add vh (bs_neg vl)) as [e|] eqn:He1; [|discriminate].
   all: destruct (bs_add e (bs_lin 0 0 (-1))) as [e'|] eqn:He2; [|discriminate].
   all: apply Z.leb_le in Hall.
   all: pose proof (bs_box_sound bx e' (ds_L s1) (ds_D s1) HL HD HH) as [_ Hup].
   all: rewrite (bs_add_eval e (bs_lin 0 0 (-1)) e' _ _ He2), (bs_add_eval vh (bs_neg vl) e _ _ He1), bs_neg_eval, bs_lin_eval in Hup.
   all: destruct (Henv lo vl Hvl) as [_ [_ Elo]]; destruct (Henv hi vh Hvh) as [_ [_ Ehi]]; cbn [bx_w bx_l1 bx_l2 bx_d1 bx_d2] in *.
   all: unfold ds_L, ds_D in *.
   all: apply Z2Nat.inj_lt; clear - Hup Hall Elo Ehi B1 B2 B3 B4 M P64 HDn Hdo; lia.
 - apply Z2Nat.inj_lt; clear - Hdo HDn B1 B2 B3 B4 M P64 P66; lia.
Qed.

Theorem ds_loop_main : term_loop_proof F prog E h l.
Proof.
 exists ds_rank, ds_inv. split; [|split].
 - intros t s Hr Hb Hi. exact (ds_start t s Hr Hb Hi).
 - intros s1 t s s' Hinv Hr Hin Hs Hin' Hnh.
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   exact (ds_step_inv s1 t s s' ins Hinv Hr Hin Hf Hs Hin' Hnh).
 - intros s1 t s s' Hinv Hr Hin Hs Hbh Hi0. exact (ds_back s1 t s s' Hinv Hr Hin Hs Hbh Hi0).
Qed.
End DMain.

Theorem ds_loop_proof : term_loop_proof F prog E h l.
Proof.
 destruct ds_rest as [m [neg [p [Hgm [Hlast Hrest]]]]].
 destruct (ts_head_range F prog cert h lo w) as [[[l1 l2]|]|] eqn:HR1, (ts_head_range F prog cert h hi w) as [[[h1 h2]|]|] eqn:HR2;
   try contradiction.
 - apply (ds_loop_main m neg p l1 l2 h1 h2 Hgm Hlast HR1 HR2).
   intros Hle. cbv zeta in Hrest. rewrite (proj2 (Z.leb_le _ _) Hle) in Hrest. exact Hrest.
 - exists (fun _ => O), (fun _ _ _ => False). split; [|split; intros; contradiction].
   intros t s Hr Hb Hi. exact (ds_head_some t s hi Hr Hb Hi HR2).
 - exists (fun _ => O), (fun _ _ _ => False). split; [|split; intros; contradiction].
   intros t s Hr Hb Hi. exact (ds_head_some t s lo Hr Hb Hi HR1).
 - exists (fun _ => O), (fun _ _ _ => False). split; [|split; intros; contradiction].
   intros t s Hr Hb Hi. exact (ds_head_some t s lo Hr Hb Hi HR1).
Qed.

End DSearch.

Theorem term_dsearch_sound : forall F prog cert E l h dsc,
 check_scalar_certificate F prog cert = true -> launch_ok F E -> term_shape prog = true ->
 In (l,h) (term_edges prog) -> term_dsearch_ok F prog cert h l dsc = true -> term_loop_proof F prog E h l.
Proof. intros F prog cert E l h dsc Hcert HE Hshape He Hok. exact (ds_loop_proof F prog cert E l h dsc Hcert HE Hshape He Hok). Qed.

(* ------------------------------------------------------------------ *)
(* Counters that a test keeps from one value: soundness.               *)

Lemma te_extend_eq : forall w sg x y, (1 <= w)%nat ->
 (extend w sg x = extend w sg y <-> trunc w x = trunc w y).
Proof.
 intros w sg x y Hw. destruct sg; cbn [extend]; [|reflexivity].
 unfold sext. pose proof (trunc_range w x) as Hx. pose proof (trunc_range w y) as Hy.
 assert (Hp : 2 ^ Z.of_nat w = 2 * 2 ^ (Z.of_nat w - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
 destruct (Z.ltb_spec (trunc w x) (2 ^ (Z.of_nat w - 1))) as [Lx|Lx], (Z.ltb_spec (trunc w y) (2 ^ (Z.of_nat w - 1))) as [Ly|Ly];
   split; intros Heq; lia.
Qed.

(* The rank of value v, for the step dz and the excluded value k. *)
Definition te_rk (w : nat) (dz k v : Z) : Z := trunc w (k - dz - v) / dz.

Lemma te_rk_step : forall w j k v, 0 <= j <= Z.of_nat w -> (2 ^ j | k) -> (2 ^ j | v) ->
 trunc w (v + 2 ^ j) <> trunc w k ->
 1 <= te_rk w (2 ^ j) k v /\ te_rk w (2 ^ j) k (trunc w (v + 2 ^ j)) = te_rk w (2 ^ j) k v - 1.
Proof.
 intros w j k v Hj Hk Hv Hne. unfold te_rk, trunc in *.
 assert (HM : 0 < 2 ^ Z.of_nat w) by (apply Z.pow_pos_nonneg; lia).
 assert (Hs : 0 < 2 ^ j) by (apply Z.pow_pos_nonneg; lia).
 assert (HsM : (2 ^ j | 2 ^ Z.of_nat w)) by (apply term_pow2_divide; lia).
 set (M := 2 ^ Z.of_nat w) in *. set (sz := 2 ^ j) in *.
 set (a := (k - sz - v) mod M).
 assert (Ha : 0 <= a < M) by (apply Z.mod_pos_bound; exact HM).
 assert (Hsa : (sz | a)).
 { unfold a. rewrite Z.mod_eq by lia.
   apply Z.divide_sub_r; [|apply Z.divide_mul_l; exact HsM].
   apply Z.divide_sub_r; [apply Z.divide_sub_r; [exact Hk|apply Z.divide_refl]|exact Hv]. }
 assert (Ha0 : a <> 0).
 { intros Ha0. apply Hne. replace k with ((k - sz - v) + (v + sz)) by ring.
   rewrite (Z.add_mod (k - sz - v) (v + sz) M) by lia. fold a. rewrite Ha0, Z.add_0_l, Z.mod_mod by lia. reflexivity. }
 destruct Hsa as [q Hq].
 assert (Hq1 : 1 <= q) by nia.
 rewrite Zminus_mod_idemp_r. replace (k - sz - (v + sz)) with ((k - sz - v) - sz) by ring.
 rewrite <- Zminus_mod_idemp_l. fold a. rewrite (Z.mod_small (a - sz) M) by nia.
 rewrite Hq. replace (q * sz - sz) with ((q - 1) * sz) by ring. rewrite !Z.div_mul by lia. lia.
Qed.

Lemma te_swap_eval : forall c x y, cmp_eval (te_swap c) x y = cmp_eval c y x.
Proof. intros [] x y; cbn; try reflexivity; rewrite Z.eqb_sym; reflexivity. Qed.

Lemma te_excluded_sound : forall cc lo hi k x y, te_excluded cc lo hi = Some k -> lo <= y <= hi ->
 cmp_eval cc x y = true -> x <> k.
Proof.
 intros cc lo hi k x y Hk Hy Hc. destruct cc; cbn [te_excluded cmp_eval] in Hk, Hc.
 - injection Hk as <-. apply Z.ltb_lt in Hc. lia.
 - injection Hk as <-. apply Z.leb_le in Hc. lia.
 - injection Hk as <-. apply Z.ltb_lt in Hc. lia.
 - injection Hk as <-. apply Z.leb_le in Hc. lia.
 - discriminate.
 - destruct (Z.eqb_spec lo hi); [injection Hk as <-|discriminate].
   apply negb_true_iff, Z.eqb_neq in Hc. lia.
Qed.

Lemma te_mod_divide : forall a n m, 0 < n -> 0 < m -> (n | m) -> (a mod m) mod n = a mod n.
Proof.
 intros a n m Hn Hm [k ->].
 rewrite (Z.div_mod a (k * n)) at 2 by lia.
 rewrite Z.add_comm. replace (k * n * (a / (k * n))) with ((k * (a / (k * n))) * n) by ring.
 rewrite Z.mod_add by lia. reflexivity.
Qed.

(* A value extended from a width at least w keeps its low w bits. *)
Lemma te_trunc_extend : forall w W sg u, (w <= W)%nat -> trunc w (extend W sg u) = trunc w u.
Proof.
 intros w W sg u HwW. pose proof (extend_eqm W sg u) as Heq. unfold eqm in Heq. unfold trunc.
 assert (H1 : 0 < 2 ^ Z.of_nat w) by (apply Z.pow_pos_nonneg; lia).
 assert (H2 : 0 < 2 ^ Z.of_nat W) by (apply Z.pow_pos_nonneg; lia).
 assert (H3 : (2 ^ Z.of_nat w | 2 ^ Z.of_nat W)) by (apply term_pow2_divide; lia).
 rewrite <- (te_mod_divide (extend W sg u) _ _ H1 H2 H3), Heq, (te_mod_divide u _ _ H1 H2 H3). reflexivity.
Qed.

Lemma te_sext_id : forall n x, (1 <= n)%nat -> - 2 ^ (Z.of_nat n - 1) <= x < 2 ^ (Z.of_nat n - 1) -> sext n x = x.
Proof.
 intros n x Hn Hx. unfold sext, trunc.
 assert (Hp : 2 ^ Z.of_nat n = 2 * 2 ^ (Z.of_nat n - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
 destruct (Z.leb_spec 0 x).
 - rewrite Z.mod_small by lia. destruct (Z.ltb_spec x (2 ^ (Z.of_nat n - 1))); lia.
 - replace (x mod 2 ^ Z.of_nat n) with (x + 2 ^ Z.of_nat n).
   + destruct (Z.ltb_spec (x + 2 ^ Z.of_nat n) (2 ^ (Z.of_nat n - 1))); lia.
   + rewrite <- (Z.mod_small (x + 2 ^ Z.of_nat n) (2 ^ Z.of_nat n)) by lia.
     rewrite <- Z.add_mod_idemp_r by lia. rewrite Z.mod_same by lia. rewrite Z.add_0_r. reflexivity.
Qed.

Lemma te_extend_trunc : forall n sg v, extend n sg (trunc n v) = extend n sg v.
Proof.
 intros n [] v; cbn [extend]; [apply term_sext_trunc|].
 unfold trunc. apply Z.mod_mod. apply Z.pow_nonzero; lia.
Qed.

Lemma te_step_cvt : forall F prog E t s s' t0 fr sg d r,
 fetch prog (blk s) (idx s) = Some (ICvt None t0 fr sg d r) -> phase_step F prog E t s = Some s' ->
 regs s' d = trunc (wdn F d) (extend fr sg (regs s r)).
Proof.
 intros F prog E t s s' t0 fr sg d r Hf Hs. unfold phase_step, step in Hs. rewrite Hf in Hs.
 injection Hs as <-. cbn [holds guard_of negb goto regs dest result]. unfold upd. rewrite Pos.eqb_refl. reflexivity.
Qed.

Section TEq.
Variable F : lfacts.
Variable prog : program.
Variable cert : scalar_certificate.
Variable E : env.
Variables l h : nat.
Variable d : term_eqexit.
Hypothesis Hcert : check_scalar_certificate F prog cert = true.
Hypothesis HE : launch_ok F E.
Hypothesis Hshape : term_shape prog = true.
Hypothesis He : In (l,h) (term_edges prog).
Hypothesis Hok : term_eqexit_ok F prog cert h l d = true.
Local Notation reach tm s := (run' F prog E tm = Some s).
Local Notation c := (te_c d).
Local Notation b := (te_b d).
Local Notation i := (te_i d).
Local Notation w := (wdn F c).

Lemma te_static :
 (h <= b <= l)%nat /\ is_cell F c = false /\
 (forall v t0, (h <= v < b)%nat -> In t0 (edge_targets prog v) -> (t0 <= b)%nat \/ term_in h l t0 = false) /\
 (forall v, In v (term_latches prog h l) -> (b <= v)%nat) /\
 exists a al dz k, term_region_writes prog h l c = [(b, i, IBin None BAdd w c a (OImm dz))] /\
   te_alias F prog h l b i c a = Some al /\ te_test F prog cert h l b i c = Some k /\
   te_step_ok F prog cert h l b i c a dz k = true.
Proof.
 pose proof Hok as H0. unfold term_eqexit_ok in H0. cbv zeta in H0.
 match type of H0 with (if negb ?X then _ else _) = true => destruct X eqn:H1 end; cbn [negb] in H0; [|discriminate].
 rewrite !andb_true_iff in H1. destruct H1 as [[[[Hhb Hbl] Hc] Hst] Hlat].
 apply Nat.leb_le in Hhb, Hbl. apply negb_true_iff in Hc.
 split; [lia|]. split; [exact Hc|]. split.
 { intros v t0 Hv Ht0. rewrite forallb_forall in Hst. specialize (Hst v ltac:(apply in_seq; lia)).
   rewrite forallb_forall in Hst. specialize (Hst t0 Ht0). apply orb_true_iff in Hst.
   destruct Hst as [Hst|Hst]; [left; apply Nat.leb_le; exact Hst|right; apply negb_true_iff; exact Hst]. }
 split.
 { intros v Hv. rewrite forallb_forall in Hlat. apply Nat.leb_le. exact (Hlat v Hv). }
 destruct (term_region_writes prog h l c) as [|[[b' i'] [ |  | g op w' c1 a [?|dz] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ]] [|]];
   try discriminate; destruct g as [[? ?]|]; try discriminate; destruct op; try discriminate.
 match type of H0 with (if negb ?X then _ else _) = true => destruct X eqn:H2 end; cbn [negb] in H0; [|discriminate].
 destruct (te_alias F prog h l b i c a) as [al|] eqn:Hal; [|rewrite andb_false_r in H2; discriminate].
 rewrite andb_true_r, !andb_true_iff, !Nat.eqb_eq, !Pos.eqb_eq in H2.
 destruct H2 as [[[-> ->] ->] ->].
 destruct (te_test F prog cert h l b i c) as [k|] eqn:Hz; [|discriminate].
 exists a, al, dz, k. split; [reflexivity|]. split; [exact Hal|]. split; [reflexivity|exact H0].
Qed.

Lemma te_regs : forall t s, reach t s -> term_in h l (blk s) = true -> regs_ok F E (regs s).
Proof.
 intros t s Hr Hin. destruct (term_edges_spec prog l h He) as [Hl _]. apply (term_in_iff l h) in Hin.
 destruct (term_states_sound F prog cert E t s Hcert HE Hr ltac:(lia)) as [_ [F' [_ [_ [Hsame [_ [Hrs _]]]]]]].
 exact (proj1 (term_same_regs F F' E (regs s) Hsame) Hrs).
Qed.

Lemma te_env : env_ok F E.
Proof. exact (env_of_launch F prog E (check_scalar_certificate_lf_ok F prog cert Hcert) HE). Qed.

Lemma te_wc : (1 <= w <= 64)%nat.
Proof. exact (term_wd_range F E c te_env). Qed.

(* c is written only at the update. *)
Lemma te_only : forall a dz bb k ins, term_region_writes prog h l c = [(b, i, IBin None BAdd w c a (OImm dz))] ->
 (h <= bb <= l)%nat -> fetch prog bb k = Some ins -> term_writes c ins = true -> bb = b /\ k = i.
Proof.
 intros a dz bb k ins Hrw Hbb Hf Hw0.
 assert (Hin0 : In (bb, k, ins) (term_region_writes prog h l c))
   by (apply term_region_writes_spec; split; [exact Hbb|]; split; [exact Hf|exact Hw0]).
 rewrite Hrw in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as -> -> _. split; reflexivity.
Qed.

Lemma te_outer_spec : (l <= te_outer prog h l < List.length prog)%nat.
Proof.
 destruct (term_edges_spec prog l h He) as [Hl _]. unfold te_outer.
 assert (Hall : forall v, In v (term_latches prog h (List.length prog)) -> (v < List.length prog)%nat).
 { intros v Hv. unfold term_latches in Hv. apply in_map_iff in Hv. destruct Hv as [[v' h'] [<- Hv]].
   apply filter_In in Hv. destruct Hv as [Hv _]. exact (proj1 (term_edges_spec prog v' h' Hv)). }
 induction (term_latches prog h (List.length prog)) as [|v vs IH]; cbn [fold_right]; [lia|].
 assert (Hv : (v < List.length prog)%nat) by (apply Hall; left; reflexivity).
 assert (IH' : (l <= fold_right Nat.max l vs < List.length prog)%nat) by (apply IH; intros u Hu; apply Hall; right; exact Hu).
 lia.
Qed.

(* The step is 2^j; a step above 1 adds to c itself, which is a multiple of
   2^j on every edge into the largest loop with head h and has no other
   write there. *)
Lemma te_step_facts : forall a dz k, te_step_ok F prog cert h l b i c a dz k = true ->
 exists j, dz = 2 ^ j /\ 0 <= j < Z.of_nat w /\ (2 ^ j | k) /\
 (j = 0 \/ (0 < j /\ a = c /\
   (forall bb kk ins, (h <= bb <= te_outer prog h l)%nat -> fetch prog bb kk = Some ins -> term_writes c ins = true ->
     bb = b /\ kk = i) /\
   short_all (fun o => match o with Some x => j <=? x | None => false end)
     (te_entry_aligns F prog cert h (te_outer prog h l) c) = true)).
Proof.
 intros a dz k H. pose proof te_wc as Hw. unfold te_step_ok in H.
 destruct (Z.eqb_spec dz 1) as [->|Hz1].
 { exists 0. split; [reflexivity|]. split; [lia|]. split; [apply Z.divide_1_l|]. left. reflexivity. }
 cbv zeta in H.
 destruct (Pos.eqb a c && Z.ltb 1 dz && Z.eqb (2 ^ Z.log2 dz) dz && Z.ltb (Z.log2 dz) (Z.of_nat (wdn F c)) &&
   Z.eqb (k mod dz) 0 &&
   match term_region_writes prog h (te_outer prog h l) c with [(b', i', _)] => Nat.eqb b' b && Nat.eqb i' i | _ => false end)
   eqn:H1; [|discriminate].
 rewrite !andb_true_iff, Pos.eqb_eq, Z.ltb_lt, !Z.eqb_eq, Z.ltb_lt in H1.
 destruct H1 as [[[[[-> H1z] Hp] Hjw] Hk] Hrg].
 exists (Z.log2 dz). pose proof (Z.log2_nonneg dz) as Hj0.
 assert (Hj : 0 < Z.log2 dz) by (apply Z.log2_pos; lia).
 split; [symmetry; exact Hp|]. split; [lia|]. split.
 { rewrite Hp. apply Z.mod_divide; [lia|exact Hk]. }
 right. split; [exact Hj|]. split; [reflexivity|]. split; [|exact H].
 intros bb kk ins Hbb Hf Hw0.
 assert (Hin0 : In (bb, kk, ins) (term_region_writes prog h (te_outer prog h l) c))
   by (apply term_region_writes_spec; split; [exact Hbb|]; split; [exact Hf|exact Hw0]).
 destruct (term_region_writes prog h (te_outer prog h l) c) as [|[[b' i'] ins'] [|]]; try discriminate.
 apply andb_true_iff in Hrg. rewrite !Nat.eqb_eq in Hrg. destruct Hrg as [-> ->].
 destruct Hin0 as [Hin0|[]]. injection Hin0 as -> -> _. split; reflexivity.
Qed.

Lemma te_div_trunc : forall j x, 0 <= j <= Z.of_nat w -> (2 ^ j | x) -> (2 ^ j | trunc w x).
Proof.
 intros j x Hj Hx. unfold trunc. rewrite Z.mod_eq by (apply Z.pow_nonzero; lia).
 apply Z.divide_sub_r; [exact Hx|]. apply Z.divide_mul_l. apply term_pow2_divide. lia.
Qed.

(* A counter c := c + dz, its only write in the blocks h to L, is a
   multiple of 2^j in them when 2^j divides dz and c on every edge into
   them. *)
Lemma te_region_aligned : forall j L dz, 0 < j <= Z.of_nat w -> (L < List.length prog)%nat ->
 (2 ^ j | dz) -> fetch prog b i = Some (IBin None BAdd w c c (OImm dz)) ->
 (forall bb kk ins, (h <= bb <= L)%nat -> fetch prog bb kk = Some ins -> term_writes c ins = true -> bb = b /\ kk = i) ->
 short_all (fun o => match o with Some x => j <=? x | None => false end) (te_entry_aligns F prog cert h L c) = true ->
 forall tm s, reach tm s -> term_in h L (blk s) = true -> (2 ^ j | regs s c).
Proof.
 intros j L dz Hj HL Hdz Hfi Honly Hentry.
 pose proof (term_shape_structured prog Hshape) as Hstr.
 pose proof (term_shape_edge prog Hshape l h He) as Hh0.
 pose proof te_static as [_ [Hcc _]].
 intro tm. induction tm as [|tm IH]; intros s' Hr' Hin'.
 - exfalso. unfold run' in Hr'. cbn [run] in Hr'. injection Hr' as <-. apply term_in_iff in Hin'. cbn in Hin'. lia.
 - rewrite term_run_succ in Hr'. destruct (run' F prog E tm) as [s|] eqn:Hr; [|discriminate].
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hr'. rewrite Hf in Hr'. discriminate. }
   destruct (term_in h L (blk s)) eqn:Hin.
   + pose proof (IH s eq_refl Hin) as Hc.
     destruct (term_writes c ins) eqn:Hw0.
     * destruct (Honly (blk s) (idx s) ins ltac:(apply term_in_iff in Hin; exact Hin) Hf Hw0) as [Hbs His].
       assert (Hins : ins = IBin None BAdd w c c (OImm dz)) by (rewrite Hbs, His, Hfi in Hf; congruence).
       rewrite Hins in Hf.
       rewrite (term_step_bin F prog E tm s s' BAdd w c c (OImm dz) Hf Hr'). cbn [binop_eval opv].
       apply te_div_trunc; [lia|]. apply Z.divide_add_r; assumption.
     * rewrite (term_step_keeps F prog E tm s s' ins c Hcc Hf Hr' Hw0). exact Hc.
   + destruct (fetch_block F prog E _ _ _ Hf) as [Hbs _].
     destruct (term_states_sound F prog cert E tm s Hcert HE Hr Hbs) as [_ [F' [st [Hst [Hsame [HE' [Hrs Hok']]]]]]].
     assert (Hs' : phase_step F' prog E tm s = Some s') by (rewrite (term_same_step F F' prog E Hsame); exact Hr').
     destruct (term_step_shape F prog E tm s s' ins Hstr Hf Hr') as [[Hbb _]|[Hi' [Hlen [Hlast [Hend|Htg]]]]].
     * exfalso. rewrite Hbb in Hin'. congruence.
     * exfalso. apply term_in_iff in Hin'. lia.
     * assert (Hnb : is_bottom st = false).
       { destruct (is_bottom st) eqn:Hbot; [exfalso; exact (bottom_not_ok F' tm st _ Hbot Hok')|reflexivity]. }
       assert (Hidx : idx s = Nat.pred (block_len prog (blk s))) by lia.
       rewrite Hidx in Hst.
       assert (Hmem : forall z, In z (flat_map (fun x => let '(F'',st'') := x in
           if is_bottom st'' then [] else
           if negb (wf_instr F'' ins) then [None] else
           flat_map (fun e => let '(u,os) := e in
             if term_in h L u then match os with
               | Some q => if is_bottom q then [] else [Some (reg_align F'' q c)]
               | None => [] end else [])
             (edge_states F'' prog (blk s) (transfer F'' st'' ins)))
           (term_states F prog cert (blk s) (Nat.pred (block_len prog (blk s))))) ->
         In z (te_entry_aligns F prog cert h L c)).
       { intros z Hz. unfold te_entry_aligns. apply in_flat_map. exists (blk s).
         split; [apply in_seq; lia|]. rewrite Hin.
         assert (Hex : existsb (term_in h L) (edge_targets prog (blk s)) = true).
         { apply existsb_exists. exists (blk s'). split; [exact Htg|exact Hin']. }
         rewrite Hex. cbn [negb]. rewrite Hlast. exact Hz. }
       destruct (wf_instr F' ins) eqn:Hwf.
       -- assert (Hb' : (blk s' < List.length prog)%nat) by (apply term_in_iff in Hin'; lia).
          destruct (term_edge_sound F' prog E tm s s' st ins HE' Hrs Hok' Hwf
            Hf Hlen Hs' Hi' Hb') as [os [Hos [Hok2 Hrs2]]].
          assert (Hosnb : is_bottom os = false).
          { destruct (is_bottom os) eqn:Hbot; [exfalso; exact (bottom_not_ok F' (S tm) os _ Hbot Hok2)|reflexivity]. }
          assert (Hin2 : In (Some (reg_align F' os c)) (te_entry_aligns F prog cert h L c)).
          { apply Hmem. apply in_flat_map. exists (F',st). split; [exact Hst|].
            cbv beta iota. rewrite Hnb, Hwf. cbn [negb].
            apply in_flat_map. exists (blk s', Some os). split; [exact Hos|].
            cbv beta iota. rewrite Hin', Hosnb. left. reflexivity. }
          pose proof (term_short_all_in _ _ _ _ Hentry Hin2) as Hal. cbv beta iota in Hal. apply Z.leb_le in Hal.
          pose proof (reg_align_sound F' E HE' (S tm) os (regs s') Hok2 Hrs2 c) as Ha.
          pose proof (aligned_mono _ _ _ Hal Ha) as Ha'. unfold aligned in Ha'. rewrite Z.max_r in Ha' by lia. exact Ha'.
       -- exfalso. assert (Hin2 : In None (te_entry_aligns F prog cert h L c)).
          { apply Hmem. apply in_flat_map. exists (F',st). split; [exact Hst|].
            cbv beta iota. rewrite Hnb, Hwf. left. reflexivity. }
          pose proof (term_short_all_in _ _ _ _ Hentry Hin2) as Hal. discriminate.
Qed.

(* The reading of c by the comparison at index kk. *)
Lemma te_reading_sound : forall a dz kk x w' sg st tm s,
 term_region_writes prog h l c = [(b, i, IBin None BAdd w c a (OImm dz))] ->
 te_reading F prog b i kk c x w' sg = Some st -> reach tm s -> blk s = b -> (kk < idx s)%nat ->
 (idx s < block_len prog b)%nat ->
 extend w' sg (regs s x) = extend w st (regs s c).
Proof.
 intros a dz kk x w' sg st tm s Hrw Hr Hreach Hbs Hks Hlen. unfold te_reading in Hr.
 pose proof te_static as [Hhb [Hcc _]].
 destruct (Pos.eqb_spec x c) as [->|Hxc].
 { destruct (Nat.eqb_spec w' w) as [->|]; [injection Hr as <-; reflexivity|discriminate]. }
 destruct (is_cell F x) eqn:Hxcell; [discriminate|].
 destruct (term_last_write prog b kk x) as [[m [  |  |  |  |  | g t0 w2 sgx x' c' |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  ]]|] eqn:Hlw;
   try discriminate; destruct g as [[? ?]|]; try discriminate.
 destruct (Pos.eqb x' x && Pos.eqb c' c && Nat.eqb w2 w && Nat.eqb (wdn F x) w' && Nat.ltb i m &&
   term_kept prog b (S m) (block_len prog b) x) eqn:Hc; [|discriminate].
 rewrite !andb_true_iff, !Pos.eqb_eq, !Nat.eqb_eq, Nat.ltb_lt in Hc.
 destruct Hc as [[[[[-> ->] ->] Hwx] Him] Hkept].
 destruct (term_last_write_some prog b kk x m _ Hlw) as [Hmk [Hfm _]].
 assert (Hx : regs s x = trunc (wdn F x) (extend w sgx (regs s c))).
 { refine (term_hist F prog E (fun rs => rs x = trunc (wdn F x) (extend w sgx (rs c))) [x; c] b m
     (term_shape_structured prog Hshape) _ _ _ tm s Hreach Hbs ltac:(lia) _).
   - intros t1 s0 s0' Hb0 Hi0 Hs0. assert (Hf0 : fetch prog (blk s0) (idx s0) = Some (ICvt None t0 w sgx x c))
       by (rewrite Hb0, Hi0; exact Hfm).
     rewrite (te_step_cvt F prog E t1 s0 s0' t0 w sgx x c Hf0 Hs0).
     rewrite (term_step_keeps F prog E t1 s0 s0' _ c Hcc Hf0 Hs0); [reflexivity|].
     unfold term_writes. cbn [loop_writes dest existsb]. rewrite (proj2 (Pos.eqb_neq c x) (not_eq_sym Hxc)). reflexivity.
   - intros rs rs' Hsame Hp0. rewrite (Hsame x (or_introl eq_refl)), (Hsame c (or_intror (or_introl eq_refl))). exact Hp0.
   - intros r [<-|[<-|[]]]; assumption.
   - intros k' ins' Hk' Hfk' r Hr'. destruct Hr' as [Hr'|[Hr'|[]]]; subst r.
     + apply (term_kept_spec prog b (S m) (block_len prog b) x k' ins' Hkept); [lia|exact Hfk'].
     + destruct (term_writes c ins') eqn:Hw'; [|reflexivity]. exfalso.
       destruct (te_only a dz b k' ins' Hrw ltac:(lia) Hfk' Hw') as [_ ->]. lia. }
 pose proof te_wc as Hw. pose proof (term_wd_range F E x te_env) as Hwx'.
 assert (Hcr : 0 <= regs s c < 2 ^ Z.of_nat w).
 { assert (Hin : term_in h l (blk s) = true) by (apply (term_in_iff l h); rewrite Hbs; lia).
   exact (rs_range F E (regs s) (te_regs tm s Hreach Hin) c). }
 rewrite Hx, Hwx.
 destruct sgx.
 - destruct (sg && Nat.leb w w') eqn:Hs; [|discriminate]. injection Hr as <-.
   apply andb_true_iff in Hs. destruct Hs as [-> Hle]. apply Nat.leb_le in Hle. cbn [extend].
   rewrite term_sext_trunc. pose proof (term_sext_bounds w (regs s c) ltac:(lia)) as Hb.
   apply te_sext_id; [lia|].
   assert (2 ^ (Z.of_nat w - 1) <= 2 ^ (Z.of_nat w' - 1)) by (apply Z.pow_le_mono_r; lia). lia.
 - destruct (Nat.ltb w w' || (negb sg && Nat.leb w w')) eqn:Hs; [|discriminate]. injection Hr as <-.
   cbn [extend]. rewrite (trunc_small w (regs s c) Hcr).
   assert (Hle : (w <= w')%nat) by (apply orb_true_iff in Hs; destruct Hs as [Hs|Hs];
     [apply Nat.ltb_lt in Hs; lia|apply andb_true_iff in Hs; apply Nat.leb_le; tauto]).
   assert (Hpw : 2 ^ Z.of_nat w <= 2 ^ Z.of_nat w') by (apply Z.pow_le_mono_r; lia).
   rewrite (trunc_small w' (regs s c)) by lia.
   destruct sg; cbn [extend]; [|apply trunc_small; lia].
   apply sext_small; [lia|].
   apply orb_true_iff in Hs. destruct Hs as [Hs|Hs]; [|cbn in Hs; discriminate].
   apply Nat.ltb_lt in Hs.
   assert (2 ^ Z.of_nat w <= 2 ^ (Z.of_nat w' - 1)) by (apply Z.pow_le_mono_r; lia). lia.
Qed.

(* The value that te_value gives is not x's when x cc y holds at the end of
   block b. *)
Lemma te_value_sound : forall bb ws st cc w' sg y k tm s X,
 te_value F prog cert bb ws st cc w' sg y = Some k -> reach tm s -> blk s = bb ->
 S (idx s) = block_len prog bb -> (bb < List.length prog)%nat ->
 cmp_eval cc X (extend w' sg (opv (regs s) y)) = true -> X <> k /\ extend ws st k = k.
Proof.
 intros bb ws st cc w' sg y k tm s X Hk Hreach Hbs Hlen Hbl Hc. unfold te_value in Hk.
 destruct (Nat.leb 1 w' && Nat.leb w' 64 && match y with OReg r => Nat.eqb (wdn F r) w' | OImm _ => true end) eqn:Hy;
   cbn [negb] in Hk; [|discriminate].
 rewrite !andb_true_iff, !Nat.leb_le in Hy. destruct Hy as [[Hw1 Hw2] Hyr].
 assert (Hfits : forall k0, (if Z.eqb (extend ws st k0) k0 then Some k0 else None) = Some k -> k0 = k /\ extend ws st k = k).
 { intros k0 H. destruct (Z.eqb_spec (extend ws st k0) k0) as [E0|]; [injection H as <-; split; [reflexivity|exact E0]|discriminate]. }
 cbv zeta in Hk.
 destruct (match te_excluded cc (fst (match y with OImm z => (extend w' sg z, extend w' sg z)
                                       | OReg _ => (lo_ok (Z.of_nat w') sg, hi_ok (Z.of_nat w') sg) end))
                                 (snd (match y with OImm z => (extend w' sg z, extend w' sg z)
                                       | OReg _ => (lo_ok (Z.of_nat w') sg, hi_ok (Z.of_nat w') sg) end)) with
           | Some k0 => if Z.eqb (extend ws st k0) k0 then Some k0 else None | None => None end) as [k1|] eqn:Hs.
 - injection Hk as <-.
   destruct (te_excluded cc _ _) as [k0|] eqn:Hex; [|discriminate].
   destruct (Hfits k0 Hs) as [<- Hext]. split; [|exact Hext].
   refine (te_excluded_sound cc _ _ k0 X _ Hex _ Hc).
   destruct y as [r|z0]; cbn [fst snd opv].
   + apply extend_range. exact Hw1.
   + lia.
 - clear Hs.
   destruct (term_range_hull _) as [[[lo hi]|]|] eqn:Hhull; try discriminate.
   destruct (te_excluded cc lo hi) as [k0|] eqn:Hex; [|discriminate].
   destruct (Hfits k0 Hk) as [<- Hext]. split; [|exact Hext].
   refine (te_excluded_sound cc lo hi k0 X _ Hex _ Hc).
   destruct (term_states_sound F prog cert E tm s Hcert HE Hreach ltac:(rewrite Hbs; exact Hbl))
     as [_ [F' [st' [Hin [Hsame [HE' [Hrs' Hst']]]]]]].
   rewrite Hbs in Hin. replace (idx s) with (Nat.pred (block_len prog bb)) in Hin by lia.
   assert (Hnb : is_bottom st' = false).
   { destruct (is_bottom st') eqn:Hb; [|reflexivity]. exfalso. exact (bottom_not_ok F' tm st' (regs s) Hb Hst'). }
   destruct (cmp_range F' st' y (Z.of_nat w') sg) as [[u v]|] eqn:Hcr.
   + assert (Hinr : In (Some (u,v)) (flat_map (fun x => let '(F0,st0) := x in
         if is_bottom st0 then [] else [cmp_range F0 st0 y (Z.of_nat w') sg])
         (term_states F prog cert bb (Nat.pred (block_len prog bb))))).
     { apply in_flat_map. exists (F',st'). split; [exact Hin|]. rewrite Hnb, Hcr. left. reflexivity. }
     destruct (term_hull_some _ lo hi u v Hhull Hinr) as [Hlo Hhi].
     assert (Hyw : forall r, y = OReg r -> wdn F' r = w').
     { intros r ->. rewrite (term_same_wdn F F' r Hsame). apply Nat.eqb_eq. exact Hyr. }
     pose proof (term_cmp_opnd F' E tm (regs s) st' y w' sg u v HE' Hrs' Hst' ltac:(lia) Hyw Hcr). lia.
   + exfalso. assert (Hinr : In None (flat_map (fun x => let '(F0,st0) := x in
         if is_bottom st0 then [] else [cmp_range F0 st0 y (Z.of_nat w') sg])
         (term_states F prog cert bb (Nat.pred (block_len prog bb))))).
     { apply in_flat_map. exists (F',st'). split; [exact Hin|]. rewrite Hnb, Hcr. left. reflexivity. }
     rewrite (term_hull_none _ Hinr) in Hhull. discriminate.
Qed.

(* On the edge from b's branch that stays in the loop, c is not the excluded
   value z. *)
Lemma te_test_sound : forall a dz z tm s s' ins,
 term_region_writes prog h l c = [(b, i, IBin None BAdd w c a (OImm dz))] ->
 te_test F prog cert h l b i c = Some z ->
 reach tm s -> blk s = b -> fetch prog b (idx s) = Some ins -> S (idx s) = block_len prog b ->
 phase_step F prog E tm s = Some s' -> term_in h l (blk s') = true -> trunc w (regs s c) <> trunc w z.
Proof.
 intros a dz z tm s s' ins Hrw Hz Hreach Hbs Hf Hlen Hs Hin'.
 pose proof te_static as [Hhb [Hcc _]].
 destruct (term_edges_spec prog l h He) as [Hlp _].
 unfold te_test in Hz.
 destruct (last_instr (block_body prog b)) as [[ | | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|] eqn:Hlast;
   try discriminate.
 assert (Hins : ins = IBra (Some (neg,p)) tg).
 { pose proof (last_of_nth F prog E _ _ _ Hf Hlen) as Hl. congruence. }
 cbv zeta in Hz.
 destruct (if term_in h l tg then (if Nat.eqb b l then Some (negb neg) else None)
           else if Nat.leb (S b) l then Some neg else None) as [v|] eqn:Hstay; [|discriminate].
 assert (Hv : (term_in h l tg = true /\ b = l /\ v = negb neg) \/ (term_in h l tg = false /\ (S b <= l)%nat /\ v = neg)).
 { destruct (term_in h l tg); [destruct (Nat.eqb_spec b l)|destruct (Nat.leb_spec (S b) l)];
     try discriminate; injection Hstay as <-; [left|right]; tauto. }
 clear Hstay.
 destruct (term_last_write prog b (block_len prog b) p) as [[kk [  |  |  |  |  |  |  | g cc w' sg p' ra ob |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  ]]|] eqn:Hlw;
   try discriminate; destruct g as [[? ?]|]; try discriminate.
 destruct (Pos.eqb p' p && negb (is_cell F p) && Nat.ltb i kk &&
   forallb (fun r => negb (Pos.eqb r p) && negb (is_cell F r) && term_kept prog b (S kk) (block_len prog b) r)
     (ra :: opnd_inputs ob)) eqn:Hchk; cbn [negb] in Hz; [|discriminate].
 rewrite !andb_true_iff, Pos.eqb_eq, negb_true_iff, Nat.ltb_lt, forallb_forall in Hchk.
 destruct Hchk as [[[-> Hpc] Hik] Hops].
 assert (Hop : forall r, In r (ra :: opnd_inputs ob) -> r <> p /\ is_cell F r = false /\
   term_kept prog b (S kk) (block_len prog b) r = true).
 { intros r Hr. pose proof (Hops r Hr) as H. rewrite !andb_true_iff, !negb_true_iff, Pos.eqb_neq in H. tauto. }
 clear Hops.
 destruct (term_last_write_some prog b _ p kk _ Hlw) as [Hkl [Hfk [_ Hno]]].
 assert (Hkidx : (kk < idx s)%nat).
 { destruct (Nat.eq_dec kk (idx s)) as [Ek|Nk]; [|lia].
   exfalso. rewrite <- Ek, Hfk, Hins in Hf. discriminate. }
 assert (Hf' : fetch prog (blk s) (idx s) = Some ins) by (rewrite Hbs; exact Hf).
 (* the comparison's value at the branch *)
 assert (Hp : regs s p = trunc (wdn F p) (bool_z (cmp_eval cc (extend w' sg (regs s ra)) (extend w' sg (opv (regs s) ob))))).
 { refine (term_hist F prog E (fun rs => rs p = trunc (wdn F p) (bool_z (cmp_eval cc (extend w' sg (rs ra)) (extend w' sg (opv rs ob)))))
     (p :: ra :: opnd_inputs ob) b kk (term_shape_structured prog Hshape) _ _ _ tm s Hreach Hbs Hkidx _).
   - intros t0 s0 s0' Hb0 Hi0' Hs0. assert (Hf0 : fetch prog (blk s0) (idx s0) = Some (ISetp None cc w' sg p ra ob))
       by (rewrite Hb0, Hi0'; exact Hfk).
     rewrite (term_step_setp F prog E t0 s0 s0' cc w' sg p ra ob Hf0 Hs0).
     assert (Hk : forall r, In r (ra :: opnd_inputs ob) -> regs s0' r = regs s0 r).
     { intros r Hr. destruct (Hop r Hr) as [Hrp [Hrc _]].
       apply (term_step_keeps F prog E t0 s0 s0' _ r Hrc Hf0 Hs0).
       unfold term_writes. cbn [loop_writes dest existsb]. rewrite (proj2 (Pos.eqb_neq r p) Hrp). reflexivity. }
     rewrite (Hk ra (or_introl eq_refl)).
     destruct ob as [r|z0]; cbn [opv]; [|reflexivity].
     rewrite (Hk r (or_intror (or_introl eq_refl))). reflexivity.
   - intros rs rs' Hsame Hp0. rewrite (Hsame p (or_introl eq_refl)), (Hsame ra (or_intror (or_introl eq_refl))).
     destruct ob as [r|z0]; cbn [opv] in *; [|exact Hp0].
     rewrite (Hsame r (or_intror (or_intror (or_introl eq_refl)))). exact Hp0.
   - intros r [<-|Hr]; [exact Hpc|]. apply (Hop r Hr).
   - intros k' ins' Hk' Hfk' r [<-|Hr].
     + apply (Hno k' ins'); [lia|exact Hfk'].
     + destruct (Hop r Hr) as [_ [_ Hkept]].
       apply (term_kept_spec prog b (S kk) (block_len prog b) r k' ins' Hkept); [lia|exact Hfk']. }
 assert (Htr : forall bb, trunc (wdn F p) (bool_z bb) = bool_z bb).
 { intros bb. pose proof (term_wd_range F E p te_env) as Hwp. apply trunc_small.
   assert (2 ^ 1 <= 2 ^ Z.of_nat (wdn F p)) by (apply Z.pow_le_mono_r; lia). destruct bb; cbn [bool_z]; lia. }
 rewrite Htr in Hp.
 (* the edge stays in the loop: the comparison has value v *)
 assert (Hcmp : cmp_eval cc (extend w' sg (regs s ra)) (extend w' sg (opv (regs s) ob)) = v).
 { destruct (step_pos F prog E tm s s' ins Hf' Hs) as [[Hn [Hbra _]]|[[g' [tg' [Hi' [Hg Hp']]]]|[g' [Hi' _]]]].
   - specialize (Hbra _ _ Hins). cbn [holds] in Hbra. rewrite Hp in Hbra.
     assert (Hb' : blk s' = S b).
     { unfold next_pos in Hn. rewrite Hbs in Hn. rewrite <- Hlen, Nat.ltb_irrefl in Hn. injection Hn as Hn _. exact Hn. }
     destruct Hv as [[Htg [Ebl ->]]|[Htg [Hsb ->]]].
     + exfalso. apply (term_in_iff l h) in Hin'. lia.
     + destruct (cmp_eval cc (extend w' sg (regs s ra)) (extend w' sg (opv (regs s) ob))), neg; cbn in Hbra |- *; congruence.
   - rewrite Hins in Hi'. injection Hi' as <- <-. injection Hp' as Hb'' _. cbn [holds] in Hg. rewrite Hp in Hg.
     destruct Hv as [[Htg [Ebl ->]]|[Htg [Hsb ->]]].
     + destruct (cmp_eval cc (extend w' sg (regs s ra)) (extend w' sg (opv (regs s) ob))), neg; cbn in Hg |- *; congruence.
     + exfalso. rewrite <- Hb'', Hin' in Htg. discriminate.
   - rewrite Hins in Hi'. discriminate. }
 assert (Hcv : cmp_eval (if v then cc else flip cc) (extend w' sg (regs s ra)) (extend w' sg (opv (regs s) ob)) = true).
 { destruct v; [exact Hcmp|rewrite flip_eval, Hcmp; reflexivity]. }
 assert (Hlen' : (idx s < block_len prog b)%nat) by lia.
 assert (Hbl : (b < List.length prog)%nat) by lia.
 destruct (te_reading F prog b i kk c ra w' sg) as [st|] eqn:Hrd.
 - pose proof (te_reading_sound a dz kk ra w' sg st tm s Hrw Hrd Hreach Hbs Hkidx Hlen') as Hx.
   rewrite Hx in Hcv.
   destruct (te_value_sound b w st _ w' sg ob z tm s _ Hz Hreach Hbs Hlen Hbl Hcv) as [Hne Hext].
   intros Heq. apply Hne. rewrite <- te_extend_trunc, Heq, te_extend_trunc. exact Hext.
 - destruct ob as [x|z0]; [|discriminate].
   destruct (te_reading F prog b i kk c x w' sg) as [st|] eqn:Hrd2; [|discriminate].
   pose proof (te_reading_sound a dz kk x w' sg st tm s Hrw Hrd2 Hreach Hbs Hkidx Hlen') as Hx.
   cbn [opv] in Hcv. rewrite Hx, <- te_swap_eval in Hcv.
   destruct (te_value_sound b w st _ w' sg (OReg ra) z tm s _ Hz Hreach Hbs Hlen Hbl Hcv) as [Hne Hext].
   intros Heq. apply Hne. rewrite <- te_extend_trunc, Heq, te_extend_trunc. exact Hext.
Qed.

Section TEqMain.
Variable a : reg.
Variable al : option (reg * nat * bool * nat).
Variables dz kx j : Z.
Hypothesis Hrw : term_region_writes prog h l c = [(b, i, IBin None BAdd w c a (OImm dz))].
Hypothesis Hal : te_alias F prog h l b i c a = Some al.
Hypothesis Hz : te_test F prog cert h l b i c = Some kx.
Hypothesis Hdz : dz = 2 ^ j.
Hypothesis Hj : 0 <= j <= Z.of_nat w.
Hypothesis Hkj : (2 ^ j | kx).
Hypothesis Hdiv : forall tm s, reach tm s -> term_in h l (blk s) = true -> (2 ^ j | regs s c).

Lemma te_alias_none : al = None -> a = c.
Proof.
 intros ->. unfold te_alias in Hal. destruct (Pos.eqb_spec a c) as [->|]; [reflexivity|].
 destruct (negb _); [discriminate|].
 destruct (term_last_write prog b i a) as [[m [ | | | | | g t0 W sg1 a' y | | | | | | | | | | | | | | | | | | | ]]|];
   try discriminate; destruct g as [[? ?]|]; try discriminate.
 destruct (_ && _); [|discriminate].
 destruct (term_region_writes prog h l y) as [|[[b2 my] [ | | | | | g t2 w2 sg2 y' c' | | | | | | | | | | | | | | | | | | | ]] [|]];
   try discriminate; destruct g as [[? ?]|]; try discriminate.
 destruct (_ && _); discriminate.
Qed.

Lemma te_alias_some : forall y W sg2 my, al = Some (y,W,sg2,my) ->
 a <> c /\ b = l /\ (h < l)%nat /\ is_cell F a = false /\ wdn F a = w /\
 (exists m t0 sg1, term_last_write prog b i a = Some (m, ICvt None t0 W sg1 a y)) /\
 (w <= W)%nat /\ wdn F y = W /\ y <> c /\ is_cell F y = false /\
 exists t2, term_region_writes prog h l y = [(h, my, ICvt None t2 w sg2 y c)].
Proof.
 intros y W sg2 my ->. unfold te_alias in Hal. destruct (Pos.eqb_spec a c) as [->|Hac]; [discriminate|].
 destruct (Nat.eqb b l && Nat.ltb h l && negb (is_cell F a) && Nat.eqb (wdn F a) (wdn F c)) eqn:H1;
   cbn [negb] in Hal; [|discriminate].
 rewrite !andb_true_iff, !Nat.eqb_eq, Nat.ltb_lt, negb_true_iff in H1. destruct H1 as [[[Hbl Hhl] Hacell] Hwa].
 destruct (term_last_write prog b i a) as [[m [ | | | | | g t0 W' sg1 a' y' | | | | | | | | | | | | | | | | | | | ]]|] eqn:Hlw;
   try discriminate; destruct g as [[? ?]|]; try discriminate.
 destruct (Pos.eqb a' a && Nat.leb (wdn F c) W' && Nat.eqb (wdn F y') W' && negb (Pos.eqb y' c) && negb (is_cell F y')) eqn:H2;
   [|discriminate].
 rewrite !andb_true_iff, Pos.eqb_eq, Nat.leb_le, Nat.eqb_eq, !negb_true_iff, Pos.eqb_neq in H2.
 destruct H2 as [[[[-> HwW] Hwy] Hyc] Hycell].
 destruct (term_region_writes prog h l y') as [|[[b2 my'] [ | | | | | g t2 w2 sg2' y'' c' | | | | | | | | | | | | | | | | | | | ]] [|]] eqn:Hrwy;
   try discriminate; destruct g as [[? ?]|]; try discriminate.
 destruct (Nat.eqb b2 h && Nat.eqb w2 (wdn F c) && Pos.eqb y'' y' && Pos.eqb c' c) eqn:H3; [|discriminate].
 rewrite !andb_true_iff, !Nat.eqb_eq, !Pos.eqb_eq in H3. destruct H3 as [[[-> ->] ->] ->].
 injection Hal as <- <- <- <-.
 split; [exact Hac|]. split; [exact Hbl|]. split; [exact Hhl|]. split; [exact Hacell|]. split; [exact Hwa|].
 split; [exists m, t0, sg1; reflexivity|]. split; [exact HwW|]. split; [exact Hwy|]. split; [exact Hyc|].
 split; [exact Hycell|]. exists t2. exact Hrwy.
Qed.

(* The branch that ends b, and the update before its last instruction. *)
Lemma te_branch : exists neg p tg, last_instr (block_body prog b) = Some (IBra (Some (neg,p)) tg) /\
 (term_in h l tg = true -> b = l) /\ (S i < block_len prog b)%nat.
Proof.
 pose proof Hz as Hz'. unfold te_test in Hz'.
 destruct (last_instr (block_body prog b)) as [[ | | | | | | | | | | | | | | |[[neg p]|] tg| | | | | | | | |]|] eqn:Hlast;
   try discriminate.
 cbv zeta in Hz'.
 destruct (if term_in h l tg then (if Nat.eqb b l then Some (negb neg) else None)
           else if Nat.leb (S b) l then Some neg else None) as [v|] eqn:Hstay; [|discriminate].
 exists neg, p, tg. split; [reflexivity|]. split.
 { intros Htg. rewrite Htg in Hstay. destruct (Nat.eqb_spec b l); [assumption|discriminate]. }
 destruct (term_last_write prog b (block_len prog b) p) as [[kk [  |  |  |  |  |  |  | g cc w' sg p' ra ob |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  ]]|] eqn:Hlw;
   try discriminate; destruct g as [[? ?]|]; try discriminate.
 destruct (Pos.eqb p' p && negb (is_cell F p) && Nat.ltb i kk && _) eqn:Hchk; cbn [negb] in Hz'; [|discriminate].
 rewrite !andb_true_iff, Nat.ltb_lt in Hchk. destruct Hchk as [[[_ _] Hik] _].
 destruct (term_last_write_some prog b _ p kk _ Hlw) as [Hkl _]. lia.
Qed.

Definition te_al (s : state) : Prop :=
 match al with
 | None => True
 | Some (y, W, sg2, my) => ((h < blk s)%nat \/ (blk s = h /\ (my < idx s)%nat)) ->
   regs s y = trunc W (extend w sg2 (regs s c))
 end.

Definition te_rank (rs : reg -> Z) : nat := Z.to_nat (te_rk w dz kx (rs c)).

Definition te_inv (s1 : state) (tm : nat) (s : state) : Prop :=
 blk s1 = h /\ idx s1 = 0%nat /\
 (((blk s < b)%nat \/ (blk s = b /\ (idx s <= i)%nat)) ->
   (regs s c = regs s1 c /\ te_al s) \/ (al = None /\ te_rk w dz kx (regs s c) < te_rk w dz kx (regs s1 c))) /\
 ((blk s = b /\ (i < idx s)%nat) -> exists v, 0 <= v < 2 ^ Z.of_nat w /\ (2 ^ j | v) /\ regs s c = trunc w (v + dz) /\
   te_rk w dz kx v <= te_rk w dz kx (regs s1 c)) /\
 ((b < blk s)%nat -> te_rk w dz kx (regs s c) < te_rk w dz kx (regs s1 c)).

Lemma te_start : forall t s, reach t s -> blk s = h -> idx s = 0%nat -> te_inv s t s.
Proof.
 intros t s Hr Hb Hi. pose proof te_static as [Hhb _].
 split; [exact Hb|]. split; [exact Hi|]. split.
 - intros _. left. split; [reflexivity|]. unfold te_al. destruct al as [[[[y W] sg2] my]|]; [|exact I].
   intros [Hlt|[_ Hlt]]; lia.
 - split.
   + intros [Hb' Hi']. lia.
   + intros Hb'. lia.
Qed.

(* c changes only at the update. *)
Lemma te_keep : forall t s s' ins, term_in h l (blk s) = true -> fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E t s = Some s' -> ~ (blk s = b /\ idx s = i) -> regs s' c = regs s c.
Proof.
 intros t s s' ins Hin Hf Hs Hne. pose proof te_static as [_ [Hcc _]].
 apply (term_step_keeps F prog E t s s' ins c Hcc Hf Hs).
 destruct (term_writes c ins) eqn:Hw0; [|reflexivity]. exfalso. apply Hne.
 exact (te_only a dz (blk s) (idx s) ins Hrw ltac:(apply (term_in_iff l h); exact Hin) Hf Hw0).
Qed.

(* The copy y of c stays c's conversion. *)
Lemma te_al_step : forall t s s' ins, term_in h l (blk s) = true -> fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E t s = Some s' -> term_in h l (blk s') = true -> ~ (blk s' = h /\ idx s' = 0%nat) ->
 regs s' c = regs s c -> te_al s -> te_al s'.
Proof.
 intros t s s' ins Hin Hf Hs Hin' Hnh Hc' Hal0. unfold te_al in *.
 pose proof te_alias_some as Hsome.
 destruct al as [[[[y W] sg2] my]|] eqn:Hal'; [|exact I].
 destruct (Hsome y W sg2 my eq_refl) as [_ [_ [_ [_ [_ [_ [_ [Hwy [_ [Hycell [t2 Hrwy]]]]]]]]]]].
 assert (Hfy : fetch prog h my = Some (ICvt None t2 w sg2 y c)).
 { assert (Hin0 : In (h, my, ICvt None t2 w sg2 y c) (term_region_writes prog h l y)) by (rewrite Hrwy; left; reflexivity).
   apply term_region_writes_spec in Hin0. tauto. }
 intros Hafter'.
 destruct (Nat.eq_dec (blk s) h) as [Hbh|Hbh]; [destruct (Nat.eq_dec (idx s) my) as [Him|Him]|].
 - (* the write of y *)
   assert (Hf0 : fetch prog (blk s) (idx s) = Some (ICvt None t2 w sg2 y c)) by (rewrite Hbh, Him; exact Hfy).
   rewrite (te_step_cvt F prog E t s s' t2 w sg2 y c Hf0 Hs), Hwy, Hc'. reflexivity.
 - assert (Hyk : regs s' y = regs s y).
   { apply (term_step_keeps F prog E t s s' ins y Hycell Hf Hs).
     destruct (term_writes y ins) eqn:Hw0; [|reflexivity]. exfalso. apply Him.
     assert (Hin0 : In (blk s, idx s, ins) (term_region_writes prog h l y)).
     { apply term_region_writes_spec. split; [apply (term_in_iff l h); exact Hin|]. split; [exact Hf|exact Hw0]. }
     rewrite Hrwy in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as _ -> _. reflexivity. }
   rewrite Hyk, Hc'. apply Hal0. right. split; [exact Hbh|].
   pose proof (term_fetch_len prog h my _ Hfy) as Hmy.
   destruct (term_step_shape F prog E t s s' ins (term_shape_structured prog Hshape) Hf Hs)
     as [[Hbb [Hii _]]|[Hi0 [Hlen _]]].
   + destruct Hafter' as [Hlt|[Hb' Hlt]]; [lia|]. lia.
   + rewrite Hbh in Hlen. lia.
 - assert (Hyk : regs s' y = regs s y).
   { apply (term_step_keeps F prog E t s s' ins y Hycell Hf Hs).
     destruct (term_writes y ins) eqn:Hw0; [|reflexivity]. exfalso. apply Hbh.
     assert (Hin0 : In (blk s, idx s, ins) (term_region_writes prog h l y)).
     { apply term_region_writes_spec. split; [apply (term_in_iff l h); exact Hin|]. split; [exact Hf|exact Hw0]. }
     rewrite Hrwy in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as -> _ _. reflexivity. }
   rewrite Hyk, Hc'. apply Hal0. left. apply (term_in_iff l h) in Hin. lia.
Qed.

(* At the update, a holds c's value. *)
Lemma te_source : forall t s, reach t s -> blk s = b -> idx s = i -> te_al s -> regs s a = regs s c.
Proof.
 intros t s Hr Hbs His Hal0.
 pose proof te_alias_some as Hsome. pose proof te_alias_none as Hnone.
 destruct al as [[[[y W] sg2] my]|] eqn:Hal'; [|rewrite (Hnone eq_refl); reflexivity].
 destruct (Hsome y W sg2 my eq_refl) as [Hac [Hbl [Hhl [Hacell [Hwa [[m [t0 [sg1 Hlw]]] [HwW [Hwy [Hyc [Hycell [t2 Hrwy]]]]]]]]]]].
 pose proof te_static as [Hhb [Hcc _]].
 unfold te_al in Hal0. rewrite Hal' in Hal0.
 assert (Hy : regs s y = trunc W (extend w sg2 (regs s c))) by (apply Hal0; left; lia).
 destruct (term_last_write_some prog b i a m _ Hlw) as [Hmi [Hfm [_ Hno]]].
 assert (Ha : regs s a = trunc w (extend W sg1 (regs s y))).
 { rewrite <- Hwa.
   refine (term_hist F prog E (fun rs => rs a = trunc (wdn F a) (extend W sg1 (rs y))) [a; y] b m
     (term_shape_structured prog Hshape) _ _ _ t s Hr Hbs ltac:(lia) _).
   - intros t1 s0 s0' Hb0 Hi0 Hs0. assert (Hf0 : fetch prog (blk s0) (idx s0) = Some (ICvt None t0 W sg1 a y))
       by (rewrite Hb0, Hi0; exact Hfm).
     rewrite (te_step_cvt F prog E t1 s0 s0' t0 W sg1 a y Hf0 Hs0).
     rewrite (term_step_keeps F prog E t1 s0 s0' _ y Hycell Hf0 Hs0); [reflexivity|].
     unfold term_writes. cbn [loop_writes dest existsb].
     destruct (Pos.eqb_spec y a) as [->|]; [|reflexivity].
     exfalso. assert (Hin0 : In (b, m, ICvt None t0 W sg1 a a) (term_region_writes prog h l a)).
     { apply term_region_writes_spec. split; [lia|]. split; [exact Hfm|]. unfold term_writes. cbn. rewrite Pos.eqb_refl. reflexivity. }
     rewrite Hrwy in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as Hb2 _ _. lia.
   - intros rs rs' Hsame Hp0. rewrite (Hsame a (or_introl eq_refl)), (Hsame y (or_intror (or_introl eq_refl))). exact Hp0.
   - intros r [<-|[<-|[]]]; assumption.
   - intros k' ins' Hk' Hfk' r Hr'. destruct Hr' as [Hr'|[Hr'|[]]]; subst r.
     + apply (Hno k' ins'); [lia|exact Hfk'].
     + destruct (term_writes y ins') eqn:Hw'; [|reflexivity]. exfalso.
       assert (Hin0 : In (b, k', ins') (term_region_writes prog h l y))
         by (apply term_region_writes_spec; split; [lia|]; split; [exact Hfk'|exact Hw']).
       rewrite Hrwy in Hin0. destruct Hin0 as [Hin0|[]]. injection Hin0 as Hb2 _ _. lia. }
 assert (Hin : term_in h l (blk s) = true) by (apply (term_in_iff l h); lia).
 pose proof (rs_range F E (regs s) (te_regs t s Hr Hin) c) as Hcr.
 rewrite Ha, Hy, te_trunc_extend by exact HwW.
 replace (trunc W (extend w sg2 (regs s c))) with (extend W false (extend w sg2 (regs s c))) by reflexivity.
 rewrite te_trunc_extend by exact HwW. rewrite te_trunc_extend by lia. apply trunc_small. exact Hcr.
Qed.

Lemma te_step_inv : forall s1 t s s' ins, te_inv s1 t s -> reach t s -> term_in h l (blk s) = true ->
 fetch prog (blk s) (idx s) = Some ins -> phase_step F prog E t s = Some s' ->
 term_in h l (blk s') = true -> ~ (blk s' = h /\ idx s' = 0%nat) -> te_inv s1 (S t) s'.
Proof.
 intros s1 t s s' ins [Hb1 [Hi1 [Ipre [Imid Ipost]]]] Hr Hin Hf Hs Hin' Hnh.
 pose proof te_static as [Hhb [Hcc [Hpath [Hlat _]]]].
 destruct te_branch as [neg [p [tg [Hlastb [Htgl Hilen]]]]].
 destruct (term_edges_spec prog l h He) as [Hlp _].
 pose proof (term_shape_structured prog Hshape) as Hstr.
 pose proof (te_regs t s Hr Hin) as Hrs.
 pose proof (rs_range F E (regs s) Hrs c) as Hcr.
 split; [exact Hb1|]. split; [exact Hi1|].
 assert (Hfi : fetch prog b i = Some (IBin None BAdd w c a (OImm dz))).
 { assert (Hin0 : In (b, i, IBin None BAdd w c a (OImm dz)) (term_region_writes prog h l c)) by (rewrite Hrw; left; reflexivity).
   apply term_region_writes_spec in Hin0. tauto. }
 (* a pre state keeps its facts when c does not change *)
 assert (Hpre_keep : regs s' c = regs s c -> ((blk s < b)%nat \/ (blk s = b /\ (idx s <= i)%nat)) ->
   (regs s' c = regs s1 c /\ te_al s') \/ (al = None /\ te_rk w dz kx (regs s' c) < te_rk w dz kx (regs s1 c))).
 { intros Hc' Hpre. rewrite Hc'. destruct (Ipre Hpre) as [[Heq Hal0]|Hr0]; [left|right; exact Hr0].
   split; [exact Heq|]. exact (te_al_step t s s' ins Hin Hf Hs Hin' Hnh Hc' Hal0). }
 destruct (term_step_shape F prog E t s s' ins Hstr Hf Hs) as [[Hbb [Hii _]]|[Hi0 [Hlen [Hlastb' [Hend|Htgt]]]]].
 - (* inside a block *)
   destruct (Nat.eq_dec (blk s) b) as [Hbs|Hbs]; [destruct (Nat.eq_dec (idx s) i) as [His|His]|].
   + (* the update *)
     assert (Hins : ins = IBin None BAdd w c a (OImm dz)) by (rewrite Hbs, His, Hfi in Hf; injection Hf as <-; reflexivity).
     pose proof (term_step_bin F prog E t s s' BAdd w c a (OImm dz) ltac:(rewrite Hf, Hins; reflexivity) Hs) as Hv.
     cbn [binop_eval opv] in Hv.
     assert (Hpre : (blk s < b)%nat \/ (blk s = b /\ (idx s <= i)%nat)) by (right; split; [exact Hbs|lia]).
     assert (Ha : regs s a = regs s c).
     { destruct (Ipre Hpre) as [[_ Hal0]|[Hn _]]; [exact (te_source t s Hr Hbs His Hal0)|].
       rewrite (te_alias_none Hn). reflexivity. }
     split; [intros [Hlt|[_ Hle]]; lia|]. split.
     * intros _. exists (regs s c). split; [exact Hcr|]. split; [exact (Hdiv t s Hr Hin)|]. split; [rewrite Hv, Ha; reflexivity|].
       destruct (Ipre Hpre) as [[Heq _]|[_ Hlt]]; [rewrite Heq; lia|lia].
     * intros Hlt. lia.
   + pose proof (te_keep t s s' ins Hin Hf Hs ltac:(intros [_ H]; contradiction)) as Hc'.
     split; [intros [Hlt|[Hb' Hle]]; [lia|]; apply (Hpre_keep Hc'); right; split; lia|]. split.
     * intros [Hb' Hlt]. rewrite Hc'. apply Imid. split; lia.
     * intros Hlt. lia.
   + pose proof (te_keep t s s' ins Hin Hf Hs ltac:(intros [H _]; contradiction)) as Hc'.
     split; [intros [Hlt|[Hb' Hle]]; [apply (Hpre_keep Hc'); left; lia|lia]|]. split.
     * intros [Hb' Hlt]. lia.
     * intros Hlt. rewrite Hc'. apply Ipost. lia.
 - exfalso. apply (term_in_iff l h) in Hin'. lia.
 - (* to the start of a block *)
   assert (Hnotupd : ~ (blk s = b /\ idx s = i)).
   { intros [Hbs His]. rewrite Hbs in Hlen. lia. }
   pose proof (te_keep t s s' ins Hin Hf Hs Hnotupd) as Hc'.
   apply (term_in_iff l h) in Hin, Hin'.
   destruct (Nat.lt_ge_cases (blk s) b) as [Hlt|Hge].
   + destruct (Hpath (blk s) (blk s') ltac:(lia) Htgt) as [Hle|Hout].
     2:{ exfalso. apply (term_in_iff l h) in Hin'. rewrite Hin' in Hout. discriminate. }
     split; [intros _; apply (Hpre_keep Hc'); left; exact Hlt|]. split.
     * intros [_ Hlt']. lia.
     * intros Hlt'. lia.
   + destruct (Nat.eq_dec (blk s) b) as [Hbs|Hbs].
     * (* the test *)
       assert (Hfb : fetch prog b (idx s) = Some ins) by (rewrite <- Hbs; exact Hf).
       assert (Hlenb : S (idx s) = block_len prog b) by (rewrite <- Hbs; exact Hlen).
       pose proof (te_test_sound a dz kx t s s' ins Hrw Hz Hr Hbs Hfb Hlenb Hs ltac:(apply (term_in_iff l h); exact Hin')) as Hne.
       destruct (Imid ltac:(split; [exact Hbs|lia])) as [v [Hvr [Hvd [Hcv Hrk]]]].
       assert (Htt : trunc w (trunc w (v + dz)) = trunc w (v + dz)) by exact (te_extend_trunc w false (v + dz)).
       rewrite Hcv, Htt, Hdz in Hne.
       destruct (te_rk_step w j kx v Hj Hkj Hvd Hne) as [H1 H2]. rewrite <- Hdz in H1, H2.
       assert (Hins : ins = IBra (Some (neg,p)) tg) by (rewrite Hbs in Hlastb'; congruence).
       destruct (step_pos F prog E t s s' ins Hf Hs) as [[Hn _]|[[g' [tg' [Hi' [_ Hp']]]]|[g' [Hi' _]]]].
       -- assert (Hb' : blk s' = S b).
          { unfold next_pos in Hn. rewrite Hbs in Hn, Hlen. rewrite <- Hlen, Nat.ltb_irrefl in Hn. injection Hn as Hn _. exact Hn. }
          split; [intros [Hlt'|[Hb'' _]]; lia|]. split; [intros [Hb'' _]; lia|].
          intros _. rewrite Hc', Hcv, H2. lia.
       -- exfalso. rewrite Hins in Hi'. injection Hi' as <- <-. injection Hp' as Hb'' Hi''.
          assert (Htg : term_in h l tg = true) by (apply (term_in_iff l h); lia).
          pose proof (Htgl Htg) as Ebl. subst l.
          destruct (term_edges_spec prog b h He) as [_ [_ [g'' Hg'']]].
          rewrite Hlastb in Hg''. injection Hg'' as _ Etg. apply Hnh. split; lia.
       -- rewrite Hins in Hi'. discriminate.
     * assert (Hpost : (b < blk s)%nat) by lia.
       split.
       -- intros _. right. split.
          ++ pose proof te_alias_some as Hsome.
             destruct al as [[[[y W] sg2] my]|] eqn:Hal'; [|reflexivity].
             exfalso. destruct (Hsome y W sg2 my eq_refl) as [_ [Ebl _]]. lia.
          ++ rewrite Hc'. apply Ipost. exact Hpost.
       -- split.
          ++ intros [_ Hlt']. lia.
          ++ intros _. rewrite Hc'. apply Ipost. exact Hpost.
Qed.

Lemma te_back : forall s1 t s s', te_inv s1 t s -> reach t s -> term_in h l (blk s) = true ->
 phase_step F prog E t s = Some s' -> blk s' = h -> idx s' = 0%nat -> (te_rank (regs s') < te_rank (regs s1))%nat.
Proof.
 intros s1 t s s' [_ [_ [_ [Imid Ipost]]]] Hr Hin Hs Hbh Hi0.
 pose proof te_static as [_ [_ [_ [Hlat _]]]].
 destruct te_branch as [neg [p [tg [_ [_ Hilen]]]]].
 destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
 2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
 pose proof (term_step_head F prog E Hshape l h t s s' ins Hf Hs Hin Hbh Hi0) as Hedge.
 assert (Hv : (h <= blk s <= l)%nat) by (apply (term_in_iff l h); exact Hin).
 pose proof (Hlat (blk s) (term_latches_in prog h l (blk s) Hedge Hv)) as Hbv.
 destruct (term_latch_jump F prog E (blk s) h t s s' ins Hshape Hedge Hf Hs eq_refl Hbh Hi0) as [Hlen [_ [g [_ [_ Hregs]]]]].
 assert (Hlt : te_rk w dz kx (regs s c) < te_rk w dz kx (regs s1 c)).
 { destruct (Nat.eq_dec (blk s) b) as [Hbs|Hbs]; [|apply Ipost; lia].
   assert (Hfb : fetch prog b (idx s) = Some ins) by (rewrite <- Hbs; exact Hf).
   assert (Hlenb : S (idx s) = block_len prog b) by (rewrite <- Hbs; exact Hlen).
   assert (Hin' : term_in h l (blk s') = true) by (rewrite Hbh; apply (term_in_iff l h); lia).
   pose proof (te_test_sound a dz kx t s s' ins Hrw Hz Hr Hbs Hfb Hlenb Hs Hin') as Hne.
   destruct (Imid ltac:(split; [exact Hbs|lia])) as [v [Hvr [Hvd [Hcv Hrk]]]].
   assert (Htt : trunc w (trunc w (v + dz)) = trunc w (v + dz)) by exact (te_extend_trunc w false (v + dz)).
   rewrite Hcv, Htt, Hdz in Hne.
   destruct (te_rk_step w j kx v Hj Hkj Hvd Hne) as [H1 H2]. rewrite <- Hdz in H1, H2. rewrite Hcv, H2. lia. }
 assert (Hdp : 0 < dz) by (rewrite Hdz; apply Z.pow_pos_nonneg; lia).
 assert (R1 : 0 <= te_rk w dz kx (regs s c)) by (unfold te_rk; apply Z.div_pos; [apply trunc_range|exact Hdp]).
 assert (R2 : 0 <= te_rk w dz kx (regs s1 c)) by (unfold te_rk; apply Z.div_pos; [apply trunc_range|exact Hdp]).
 unfold te_rank. rewrite Hregs. apply Z2Nat.inj_lt; lia.
Qed.

Theorem te_loop_main : term_loop_proof F prog E h l.
Proof.
 exists te_rank, te_inv. split; [|split].
 - intros t s Hr Hb Hi. exact (te_start t s Hr Hb Hi).
 - intros s1 t s s' Hinv Hr Hin Hs Hin' Hnh.
   destruct (fetch prog (blk s) (idx s)) as [ins|] eqn:Hf.
   2:{ unfold phase_step, step in Hs. rewrite Hf in Hs. discriminate. }
   exact (te_step_inv s1 t s s' ins Hinv Hr Hin Hf Hs Hin' Hnh).
 - intros s1 t s s' Hinv Hr Hin Hs Hbh Hi0. exact (te_back s1 t s s' Hinv Hr Hin Hs Hbh Hi0).
Qed.
End TEqMain.

Theorem te_loop_proof : term_loop_proof F prog E h l.
Proof.
 destruct te_static as [Hhb [_ [_ [_ [a [al [dz [k [Hrw [Hal [Hz Hst]]]]]]]]]]].
 destruct (te_step_facts a dz k Hst) as [j [Hdz [Hj [Hkj Hrest]]]].
 assert (Hdiv : forall tm s, reach tm s -> term_in h l (blk s) = true -> (2 ^ j | regs s c)).
 { destruct Hrest as [-> | [Hj0 [-> [Honly Hentry]]]].
   - intros. apply Z.divide_1_l.
   - pose proof te_outer_spec as HL.
     assert (Hfi : fetch prog b i = Some (IBin None BAdd w c c (OImm dz))).
     { assert (Hin0 : In (b, i, IBin None BAdd w c c (OImm dz)) (term_region_writes prog h l c)) by (rewrite Hrw; left; reflexivity).
       apply term_region_writes_spec in Hin0. tauto. }
     intros tm s Hr Hin. apply (te_region_aligned j (te_outer prog h l) dz ltac:(lia) ltac:(lia)
       ltac:(rewrite Hdz; apply Z.divide_refl) Hfi Honly Hentry tm s Hr).
     apply term_in_iff. apply term_in_iff in Hin. lia. }
 exact (te_loop_main a al dz k j Hrw Hal Hz Hdz ltac:(lia) Hkj Hdiv).
Qed.
End TEq.

Theorem term_eqexit_sound : forall F prog cert E l h dsc,
 check_scalar_certificate F prog cert = true -> launch_ok F E -> term_shape prog = true ->
 In (l,h) (term_edges prog) -> term_eqexit_ok F prog cert h l dsc = true -> term_loop_proof F prog E h l.
Proof. intros F prog cert E l h dsc Hcert HE Hshape He Hok. exact (te_loop_proof F prog cert E l h dsc Hcert HE Hshape He Hok). Qed.

Theorem term_check_sound : forall F prog cert,
 check_scalar_certificate F prog cert = true -> term_check F prog cert = true -> kernel_ends F prog.
Proof.
 intros F prog cert Hcert Hcheck E HE. unfold term_check in Hcheck.
 destruct (term_shape prog) eqn:Hshape; [|discriminate].
 apply (term_runs_end F prog E Hshape).
 - intros t s Hr Hb. exact (proj1 (term_states_sound F prog cert E t s Hcert HE Hr Hb)).
 - intros l h He. pose proof (term_short_all_in _ _ _ _ Hcheck He) as Hloop.
   unfold term_loop_ok in Hloop.
   destruct (short_any (term_eqexit_ok F prog cert h l) (term_propose_eqexits prog h l)) eqn:Hq.
   { rewrite short_any_same in Hq. apply existsb_exists in Hq.
     destruct Hq as [dsc [_ Hok]]. exact (term_eqexit_sound F prog cert E l h dsc Hcert HE Hshape He Hok). }
   destruct (short_any (term_counter_ok F prog cert h l) (term_propose_counters F prog cert h l)) eqn:Hc.
   + rewrite short_any_same in Hc. apply existsb_exists in Hc.
     destruct Hc as [dsc [_ Hok]]. exact (term_counter_sound F prog cert E l h dsc Hcert HE Hshape He Hok).
   + destruct (short_any (term_search_ok F prog cert h l) (term_propose_searches prog h l)) eqn:Hs.
     * rewrite short_any_same in Hs. apply existsb_exists in Hs.
       destruct Hs as [dsc [_ Hok]]. exact (term_search_sound F prog cert E l h dsc Hcert HE Hshape He Hok).
     * destruct (short_any (term_dsearch_ok F prog cert h l) (term_propose_dsearches prog h l)) eqn:Hd.
       -- rewrite short_any_same in Hd. apply existsb_exists in Hd.
          destruct Hd as [dsc [_ Hok]]. exact (term_dsearch_sound F prog cert E l h dsc Hcert HE Hshape He Hok).
       -- exact (term_dead_sound F prog cert E l h Hcert HE Hshape He Hloop).
Qed.
