"""RQ2: the rules of the checker that the gate's acceptances need.

Each rule of `RULES` is one change to the checker's definitions that turns
the rule off: the rule's result becomes the conservative one that a checker
without it gives. Each group of `GROUPS` turns off a set of related rules
together. A copy of `formal/` with a group's changes is built without its
proofs (`coqc -vos`), since the changes break them and only the checker's
decisions are measured. The corpus is a complete registration record
(`eval.rq4_cost.registration`): every safe kernel that the gate accepts as
written, and, as a soundness control, a sample of the defects fixed by a seed
and stratified by task set and toolchain. Each is decided again from its
archived Rocq terms with the unchanged copy (the baseline) and with each
group's copy. The accepted kernels that a group's copy refuses, or does not
decide in the time limit, are those that need the group (design section 9).

Run from the repository root, on the proof cores and without a GPU:

    CUDA_VISIBLE_DEVICES= taskset -c 32-63 uv run python -m eval.rq2_coverage.rule_ablation \\
        --registration eval/records/registration_<date>.json \\
        --output eval/records/rule_ablation_<date>.json --workers 32 --timeout 360
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

from eval.checkpoint import Checkpoint, read_artifact, snapshot_inputs
from opgate.gate import CHECK_QUERY, TIMEOUT, formal_dir

# This experiment intentionally skips proofs (-vos); it measures the same
# explicit-certificate proposal/checker used by registration, never authority.
QUERY = """
Definition certificate : text_certificate :=
  Eval vm_compute in propose_text_certificate text size mfs.
Eval vm_compute in check_text_certificate_ok text size mfs certificate.
"""
REGISTRATION_KIND = (
    "registration with raw verification inputs and the service's budget"  # registration.KIND
)
KIND = "rule group ablation of registration decisions"
DEFECTS = 200  # the defects of the soundness control
SEED = 20261004  # the seed of the defect sample

Outcome = Literal["accepted", "refused", "timeout"]


@dataclass(frozen=True)
class Rule:
    """A rule of the checker and the change to `formal/` that turns it off."""

    name: str
    description: str
    file: str
    old: str
    new: str


RULES = (
    Rule(
        "checked interval spatial certificates",
        "validate a prepared interval-loop cache with every instruction, access, initialization and CFG edge obligation",
        "GateCertificate.v",
        "| Some prepared => match propose_prepared_interval F prog prepared with\n   | Some cert => ScalarCached cert\n   | None => ScalarMissing\n   end",
        "| Some prepared => ScalarMissing",
    ),
    Rule(
        "scalar-to-vector bit unpack lowering",
        "lower every declared low-to-high unpack field through exact scalar conversion and bit extraction",
        "GateFront.v",
        "then unpack_move g sf ops else",
        'then refuse "scalar-to-vector unpack lowering is disabled" else',
    ),
    Rule(
        "mixed context event schedules",
        "prove a complete common CTA-context word over every placed CTA and warp event",
        "GateFront.v",
        "Definition front_mixed_context_shape K org ids pp events :=\n match placed_events K org ids pp(source_barriers K org)with\n |Some cta=>if negb(Nat.eqb(List.length cta)0)&&negb(event_word_eqb cta events)\n   then mixed_context_shape_select(map(map fst)pp)events else None\n |None=>None end.",
        "Definition front_mixed_context_shape (K:pkernel) (org:list place) (ids:PM.t reg)\n (pp:list(list placed)) (events:list pc):option mixed_context_shape:=None.",
    ),
    Rule(
        "affine interval bound witnesses",
        "retain whole dead interval facts that connect a bounded weighted launch expression to a live access bound",
        "GateDomain.v",
        "retain_affine_bridges st (trim_live_seed (live_join_seed table incoming b st) st).",
        "trim_live_seed (live_join_seed table incoming b st) st.",
    ),
    Rule(
        "fixed-goal projection components",
        "retain dynamic constraint components connecting two distinct fixed query atoms",
        "GateDomain.v",
        "let initial := if null moving then projection_fixed_seeds ts cs else moving in",
        "let initial := moving in",
    ),
    Rule(
        "deterministic uniform-prefix context values",
        "derive controller values from checked pure integer prefixes over common launch inputs",
        "GateContext.v",
        "|Some uentry=>uniform_entry_check F uniform prog events d uentry|None=>false end.",
        "|Some uentry=>false|None=>false end.",
    ),
    Rule(
        "CTA context event schedules",
        "prove one complete finite word for each shared CTA launch context",
        "GateContext.v",
        "Definition context_shape_select prog events :=\n match context_shape_candidates false prog events with\n |[d]=>Some d\n |[]=>match context_shape_candidates true prog events with [d]=>Some d|_=>None end\n |_=>None end.",
        "Definition context_shape_select(prog:program)(events:list pc):option context_shape:=None.",
    ),
    Rule(
        "checked compound-predicate occurrence loops",
        "discover event-free prior-value loop guards through checked conjunction dependencies and validate their offset invariant before collecting a schedule",
        "GateLoop.v",
        "if occurrence_compound_segments_supported prog events cut then Some(compound_occurrence_summary F prog events)",
        "if false then Some(compound_occurrence_summary F prog events)",
    ),
    Rule(
        "natural machine bounds for modular form equalities",
        "intersect inferred unsigned register ranges with machine bounds before proving a 64-bit modular relation is an integer equality",
        "GateDomain.v",
        "match inter (f_rr ranges r) (natural_range (r,EX)), form_range_with st (atom_range_with ranges) f with",
        "match f_rr ranges r, form_range_with st (atom_range_with ranges) f with",
    ),
    Rule(
        "CFG liveness boundary trimming",
        "forget whole non-live facts and constraints in candidate edge states before joining",
        "GateDomain.v",
        "let incoming := live_incoming_controls prog in\n  let lives := live_program prog in\n  iterate_with_boundary (trim_join_live_state lives incoming)",
        "iterate_with_boundary (fun _ st => st)",
    ),
    Rule(
        "checked signed-counter occurrence loops",
        "prove signed increasing or decreasing loop progress with exact nonwrapping post-states after legacy capability rejection",
        "GateLoop.v",
        "if occurrence_signed_segments_supported prog events cut then Some(occurrence_signed_loop_summary F prog events)else\n if occurrence_compound_segments_supported",
        "if false then Some(occurrence_signed_loop_summary F prog events)else\n if occurrence_compound_segments_supported",
    ),
    Rule(
        "missing certificate cases preflight",
        "reject absent single or split cases before the link, prologue, and termination checks",
        "GateParse.v",
        "if certificate_has_cases cert then\n match r with",
        "if true then\n match r with",
    ),
    Rule(
        "compared induction counter selection",
        "select the available checked comparison-counter descriptor before proposal, preserving matching straight-counter priority",
        "GateLoop.v",
        "if loop_counter_compared prog head d then Some(Loop_straight_descriptor d)",
        "if true then Some(Loop_straight_descriptor d)",
    ),
    Rule(
        "local-cell straight-loop progress",
        "prove straight-loop progress while local cells remain disjoint from both ranking registers",
        "GateLoop.v",
        "if is_cell F (li_counter d) || is_cell F (li_stride d) then None else",
        "if negb(null(lf_cells F)) then None else",
    ),
    Rule(
        "local-cell branched-loop progress",
        "prove branched-loop progress while local cells remain disjoint from every protected register",
        "GateLoop.v",
        "if short_all (fun r=>negb(is_cell F r)) protected then",
        "if null(lf_cells F) then",
    ),
    Rule(
        "checked physical-warp event words",
        "prove every complete warp word over all events and equal ordered CTA projections from concrete launch coordinates",
        "GateFront.v",
        "match warp_scoped_check F (map(map fst)pp) events cta r count with\n         | Some _=>true | None=>false end",
        "false",
    ),
    Rule(
        "warp-uniform entry exits",
        "prove an entry branch that leaves before every event uniform in each warp from the CTA-uniform launch values",
        "GateFront.v",
        "if warp_exit_selected (map(map fst)pp) events cta",
        "if false",
    ),
    Rule(
        "unrolled event-free range loops",
        "walk a Numba range loop between events round by round until its latch branch is decided",
        "GateLoop.v",
        "if occurrence_unrolled_segments_supported prog events cut then Some(unrolled_occurrence_summary F prog events)",
        "if false then Some(unrolled_occurrence_summary F prog events)",
    ),
    Rule(
        "loop end at a counter value",
        "end a loop whose counter one block increments by 1 and whose test keeps it from one value, ranked modulo the width",
        "GateTerm.v",
        "if short_any (term_eqexit_ok F prog cert h l) (term_propose_eqexits prog h l) then true",
        "if false then true",
    ),
    Rule(
        "loop end by a counter",
        "end a loop whose counter moves by a step bounded away from 0, or halves or doubles, inside its proved interval",
        "GateTerm.v",
        "else if short_any (term_counter_ok F prog cert h l) (term_propose_counters F prog cert h l) then true",
        "else if false then true",
    ),
    Rule(
        "loop end by a binary search",
        "end a loop whose two selected bounds move strictly inside the distance between them",
        "GateTerm.v",
        "else if short_any (term_search_ok F prog cert h l) (term_propose_searches prog h l) then true",
        "else if false then true",
    ),
    Rule(
        "loop end by a merge-path search",
        "end a loop whose bounds lower their distance on every path through the loop",
        "GateTerm.v",
        "else if short_any (term_dsearch_ok F prog cert h l) (term_propose_dsearches prog h l) then true",
        "else if false then true",
    ),
    Rule(
        "loop end by a dead back edge",
        "end a loop whose back edges have no state at any latch",
        "GateTerm.v",
        "else term_dead_ok F prog cert h l.",
        "else false.",
    ),
    Rule(
        "loop end by a power-of-two counter step",
        "end a loop whose counter steps by 2^j from a multiple of 2^j toward a multiple of 2^j",
        "GateTerm.v",
        "if Pos.eqb a c && Z.ltb 1 z",
        "if false && Pos.eqb a c && Z.ltb 1 z",
    ),
    Rule(
        "counter kept from one value by an order test",
        "end a loop whose counter test after the increment, as c < n or c <= n, keeps the counter from one value",
        "GateTerm.v",
        " | CLt => Some hi | CLe => Some (hi + 1) | CGt => Some lo | CGe => Some (lo - 1)",
        " | CLt => None | CLe => None | CGt => None | CGe => None",
    ),
    Rule(
        "counter update through conversions",
        "end a loop whose latch increments a counter read through a conversion to 64 bits and back",
        "GateTerm.v",
        "if negb (Nat.eqb b l && Nat.ltb h l && negb (is_cell F a)",
        "if negb (false && Nat.eqb b l && Nat.ltb h l && negb (is_cell F a)",
    ),
    Rule(
        "binary search bounds from the loop entry",
        "bound a binary search's lo, hi and distance by the edges into a loop entered only at its head",
        "GateTerm.v",
        "if ts_cert_ok F prog cert h d then true else ts_entry_ok F prog cert h l d.",
        "if ts_cert_ok F prog cert h d then true else false.",
    ),
    Rule(
        "step-1 inequality counted loops",
        "accept a counted loop around events whose test is a counter different from its limit when the step is 1",
        "GateContext.v",
        "if Nat.eqb width tw then\n     if Pos.eqb counter a then Some(Counted_ne false signed,predicate,b)",
        "if false then\n     if Pos.eqb counter a then Some(Counted_ne false signed,predicate,b)",
    ),
    Rule(
        "rank-ended loop summaries",
        "summarize an event-free loop with one way out by the control registers it keeps, its end proved by the termination check",
        "GateEnds.v",
        "if ends_route_selected prog events then ends_progress_check F prog events",
        "if false then ends_progress_check F prog events",
    ),
    Rule(
        "acyclic event segments",
        "check one occurrence word for a program whose segments between events are acyclic",
        "GateEnds.v",
        "else if closed_route_selected prog events then closed_progress_check F prog events",
        "else if false then closed_progress_check F prog events",
    ),
    Rule(
        "shared value participation",
        "prove the waiting-site sequence from values that are the same in every thread of a block, shared memory included",
        "GateFront.v",
        "else front_shared_check Mf K org ids pp F events",
        "else false",
    ),
    Rule(
        "shared value regions with loops",
        "let the blocks that a divergent branch's threads pass before they meet again hold loops, for a kernel that ends",
        "GateShared.v",
        "forallb (fun d => Nat.eqb d J || mem_nat d R) (block_succs prog c)) R.",
        "forallb (fun d => Nat.ltb c d && (Nat.eqb d J || mem_nat d R)) (block_succs prog c)) R.",
    ),
    Rule(
        "shared address facts across blocks",
        "keep the thread index fields and shared addresses that the first block computes at the start of later blocks",
        "GateShared.v",
        "  let G := entry_facts_at F tidr uniform T C prog P dirty in",
        "  let G := [] in",
    ),
    Rule(
        "per-edge revisit limits",
        "allow a block's entry state LIMIT changes for each edge into it, rather than LIMIT in all",
        "GateDomain.v",
        "  map (fun d => (LIMIT * Nat.max 1 d)%nat)",
        "  map (fun d => LIMIT)",
    ),
    Rule(
        "stable range layers",
        "stop the cached range layers at the first layer whose cached ranges equal those of the layer below",
        "GateDomain.v",
        "   if range_pairs_eqb next pairs then prev",
        "   if false then prev",
    ),
    Rule(
        "checked loop summaries in occurrence segments",
        "consume a finite checked event-free loop trace and exact endpoint state inside the complete event word",
        "GateLoop.v",
        "else occurrence_checked_word(occurrence_loop_summary F prog events)F prog events.",
        "else None.",
    ),
    Rule(
        "checked forward-branch loop summaries",
        "check every event-free forward path, counter preservation, positive nonwrapping increment, and inductive backedge",
        "GateLoop.v",
        "Definition loop_summary_at (prog:program) (head:pc) : option loop_summary_descriptor :=\n match loop_at prog head with\n | Some d=>if loop_counter_compared prog head d then Some(Loop_straight_descriptor d)\n   else match loop_dag_at prog head with\n     | Some shape=>Some(Loop_dag_descriptor shape)\n     | None=>Some(Loop_straight_descriptor d) end\n | None=>match loop_dag_at prog head with Some d=>Some(Loop_dag_descriptor d)|None=>None end\n end.\n",
        "Definition loop_summary_at (prog:program) (head:pc) : option loop_summary_descriptor :=\n option_map Loop_straight_descriptor(loop_at prog head).\n",
    ),
    Rule(
        "shared ranked prefix for straight event tails",
        "prove one ranked prefix and concrete fallthrough to every ordered event in the same block",
        "GateLoop.v",
        "else event_tail_shape prog events.",
        "else false.",
    ),
    Rule(
        "finite repeated event words",
        "prove a common complete occurrence word on actual IR, with an event-free suffix",
        "GateLoop.v",
        "match occurrence_program_word_general F prog events with Some _=>true|None=>false end",
        "false",
    ),
    Rule(
        "structural straight-line event progress",
        "prove actual straight-line IR reaches every checked event once in source order",
        "GateLoop.v",
        "if lf_ok F then short_all (phase_event_at prog) events else false",
        "false",
    ),
    Rule(
        "mixed collective event markers",
        "prove common one-visit arrival for barriers and checked full-mask warp collectives",
        "GateBarrier.v",
        "match i with INop | IDef None _ => true | _ => false end",
        "match i with INop => true | _ => false end",
    ),
    Rule(
        "loop increment source aliases",
        "prove a copied increment source equals the loop-carried counter",
        "GateLoop.v",
        "else register_fact_le F st (li_source d) (loop_alias d)",
        "else false",
    ),
    Rule(
        "ranked CTA loop summaries",
        "prove finite loop exit using a checked upper bound, positive stride, no wrap, and an inductive invariant",
        "GateLoop.v",
        "loop_seek (List.length (List.concat prog)) 0 prog pos pos",
        "(@None loop_info)",
    ),
    Rule(
        "ordered CTA barrier events",
        "check universal progress and common source event occurrence order",
        "GateFront.v",
        "| Some events=>if front_event_check Mf K org ids pp F events then true",
        "| Some events=>if false then true",
    ),
    Rule(
        "finite CTA barrier phases",
        "prove all threads reach one exact CTA barrier through finite control flow",
        "GateFront.v",
        "| Some event=>single_phase_check F (map (map fst) pp) event",
        "| Some _=>false",
    ),
    Rule(
        "symbolic counted-loop starts",
        "propose checked exit bounds when a counter starts at a symbolic affine expression",
        "GateDomain.v",
        "if Z.eqb (fm a) 64 && Z.eqb (fm a') 64 && terms_eqb (ft a) (ft a') &&",
        "if Z.eqb (fm a) 64 && Z.eqb (fm a') 64 && null (ft a) && null (ft a') &&",
    ),
    Rule(
        "equality counted-loop exits",
        "recognize both equality and inequality orientations of a checked counted-loop exit",
        "GateDomain.v",
        "| (CEq | CNe), true, OReg rb => counter rb (ca k) 1 ++ counter (ca k) rb (-1)",
        "| CNe, true, OReg rb => counter rb (ca k) 1 ++ counter (ca k) rb (-1)",
    ),
    Rule(
        "layout-independent counted bounds",
        "offer checked counted-loop exit bounds independent of numeric block order",
        "GateDomain.v",
        "exit_bound_candidates old out++start_bound_candidates old out",
        "(if loop_edge then exit_bound_candidates old out else [])++start_bound_candidates old out",
    ),
    Rule(
        "monotone affine start bounds",
        "retain a checked lower or upper bound by a changing counter’s fixed affine start",
        "GateDomain.v",
        "++start_bound_candidates old out",
        "",
    ),
    Rule(
        "numeric loop pivot priority",
        "prefer a bounded counter over an unbounded pointer when selecting an affine pivot",
        "GateDomain.v",
        "fold_right ins [] bounded_steps ++ fold_right ins [] unbounded_steps",
        "fold_right ins [] steps",
    ),
    Rule(
        "universal bounds before widening",
        "clip vacuous constraint endpoints to universal machine ranges before widening",
        "GateDomain.v",
        "Definition clip_join_state (st : astate) : astate :=\n {|amap:=amap st;acons:=map clip_constraint (acons st)|}.",
        "Definition clip_join_state (st : astate) : astate :=\n let _ := clip_constraint in st.",
    ),
    Rule(
        "constant selection atom interpretations",
        "check signed, shifted and bitfield readings of a constant alternative when moving a selected bound",
        "GateDomain.v",
        "if cons_mentions r k && (negb(cons_mentions d k)||Pos.eqb r d)\n then",
        "if cons_mentions r k && (negb(cons_mentions d k)||Pos.eqb r d) &&\n (match b with OImm _ => forallb (fun u => negb (Pos.eqb (fst (fst u)) r) || value_atom r (fst u)) (fst (fst k)) | _ => true end)\n then",
    ),
    Rule(
        "conditional load defaults",
        "retain exact inactive defaults and the full destination range for guarded loads",
        "GateDomain.v",
        "default_load st (sentinel_load st (guarded_index_load st g (Z.of_nat bits) d a off) g (Z.of_nat bits) d) g d",
        "sentinel_load st (guarded_index_load st g (Z.of_nat bits) d a off) g (Z.of_nat bits) d",
    ),
    Rule(
        "conditional load guard recovery",
        "refine original load predicates only after a checked projection under the current access guard",
        "GateDomain.v",
        "| Some (_,p) => refine_origins refined (guard_origins st p)",
        "| Some (_,p) => refine_guard refined None",
    ),
    Rule(
        "low bitfield source bound",
        "a machine register is at least its low unsigned bitfield",
        "GateDomain.v",
        "if Z.eqb a 0 && (0 <=? n) &&\n         (existsb (atom_eqb (r,EX)) xs || existsb (atom_eqb (r,EZ)) xs) then",
        "if false then",
    ),
    Rule(
        "modular low-mask residue",
        "retain predicate coefficients modulo the number of low bits selected by a mask",
        "GateDomain.v",
        "then mask_low_residue st fa k bw n else linear",
        "then linear else linear",
    ),
    Rule(
        "quotient right shift",
        "retain exact quotient terms when the remaining low terms are proved below the divisor",
        "GateDomain.v",
        "| None=>quotient_shift st fa k bw signed end.",
        "| None=>None end.",
    ),
    Rule(
        "copied predicate recovery",
        "recover a predicate fact from an exact widened bit copy compared with zero",
        "GateDomain.v",
        "match copied_zero_predicate st c w a b with",
        "match (None : option fact) with",
    ),
    Rule(
        "checked conversion form lift",
        "preserve affine provenance when a full-width conversion is proved within its signed or unsigned range",
        "GateDomain.v",
        "match converted_form st s from signed with",
        "match (None : option form) with",
    ),
    Rule(
        "comparison alias transport",
        "transport saved operands across a proved equal constant step while retaining wrap cases",
        "GateDomain.v",
        "      if comparison_alias st (ca k) d then\n        match cmp_step_direct",
        "      if false then\n        match cmp_step_direct",
    ),
    Rule(
        "selection comparison normalization",
        "rebase proved-equal saved operands and orient comparisons before checking both selection branches",
        "GateDomain.v",
        "  let st := selection_predicate st (selection_register d a b) p in\n",
        "",
    ),
    Rule(
        "OR already-set bits",
        "preserve the operand when its complement is disjoint from a constant OR mask",
        "GateDomain.v",
        "if low_bits_clear st (normalize st (fshift (fscale fa (-1) 64) (-1) (-1))) k\n  then FForm fa 0 None else",
        "if false\n  then FForm fa 0 None else",
    ),
    Rule(
        "complemented contiguous OR",
        "retain a complemented register plus its exact contiguous masked bit field",
        "GateDomain.v",
        "match or_complement st fa k with",
        "match (if false then or_complement st fa k else None) with",
    ),
    Rule(
        "rounded mask relations",
        "a contiguous high mask remains between its source minus discarded low bits and its source",
        "GateDomain.v",
        "mask_const_cons st (transfer_dest st d None (value st i) false) (Z.of_nat w) d a k",
        "transfer_dest st d None (value st i) false",
    ),
    Rule(
        "symbolic minmax common bounds",
        "transport a relative lower or upper bound shared by both symbolic minmax operands",
        "GateDomain.v",
        "then minmax_common_cons st out mn sg w d a b else",
        "then out else",
    ),
    Rule(
        "bit-aware projection residuals",
        "range a canceled constraint residual using correlations between whole-register and bit-field atoms",
        "GateDomain.v",
        "match plain_range ar {|fm:=64;ft:=rest;flo:=n*flo f;fhi:=n*fhi f|} with",
        "match (if false then plain_range ar {|fm:=64;ft:=rest;flo:=n*flo f;fhi:=n*fhi f|} else plain_sum ar rest (n*flo f) (n*fhi f)) with",
    ),
    Rule(
        "exact masked result shortcut",
        "skip redundant rounded-mask range work when ordinary transfer already determines the destination exactly",
        "GateDomain.v",
        " if match get out d with\n      | Some (FForm f _ _) => Z.eqb (fm f) 64 && null (ft f) && Z.eqb (flo f) (fhi f)\n      | _ => false end then out else\n",
        " if false then out else\n",
    ),
    Rule(
        "derived size entry constraints",
        "seed analysis with exact size quotients independently checked at every launch",
        "GateDomain.v",
        "Definition empty_state : astate := {| amap := PM.empty fact; acons := lf_derived F |}.",
        "Definition empty_state : astate := {| amap := PM.empty fact; acons := let _ := F in [] |}.",
    ),
    Rule(
        "signed selection ordering",
        "retain a full-width signed order proved for both branches of a register/constant selection",
        "GateDomain.v",
        "if negb ((0 <? w) && Z.eqb w (wd d) && Z.eqb w (wd r)) || Pos.eqb r d then out else",
        "if true then out else",
    ),
    Rule(
        "comparison operand snapshots",
        "retain compared values in fresh same-width registers before later operand writes",
        "GateFront.v",
        "   mapT snapshot_block (split_blocks p)).",
        "   ret (split_blocks p)).",
    ),
    Rule(
        "general integer conversion widths",
        "retain signed or unsigned full-source conversions beyond the 32-bit special cases",
        "GateDomain.v",
        "else if Nat.eqb from (wdn s) then\n        match converted_form st s from signed with",
        "else if false then\n        match converted_form st s from signed with",
    ),
    Rule(
        "left shift explicit alignment",
        "retain the low zero bits from a constant shift when a form has a narrower modulus",
        "GateDomain.v",
        "Some (FForm (fscale fa (2 ^ Z.min (k mod 2 ^ 32) bw) bw) (Z.min (k mod 2 ^ 32) bw) None)",
        "Some (FForm (fscale fa (2 ^ Z.min (k mod 2 ^ 32) bw) bw) 0 None)",
    ),
    Rule(
        "symbolic step alignment",
        "retain alignment of exact affine loop steps with aligned symbolic terms",
        "GateDomain.v",
        "if Z.eqb (flo s) (fhi s) then form_align st s else 0",
        "if false then form_align st s else if null (ft s) && Z.eqb (flo s) (fhi s) then v2 (flo s) else 0",
    ),
    Rule(
        "index load width eligibility",
        "skip index-value inference when no declared index element matches the load width",
        "GateDomain.v",
        "existsb (fun pair => match sp_index (snd pair) with\n"
        "   | Some ix => Z.eqb bits (8 * ix_elem ix)\n"
        "   | None => false end) (lf_roles F)",
        "if null (lf_roles F) then true else true",
    ),
    Rule(
        "launch widening thresholds",
        "retain launch scalar endpoints and strict guard limits when widening loop ranges",
        "GateDomain.v",
        "flat_map (fun pair => let '(_, (lo, hi)) := pair in [lo; lo - 1; hi; hi - 1]) (lf_bounds F)",
        "[]",
    ),
    Rule(
        "loop pivot ordering",
        "prefer a loop cursor with fewer symbolic start terms before choosing by register number",
        "GateDomain.v",
        "Z.of_nat (length (ft a)), (if Z.eqb (fm a) 64 then 0 else 1), Z.abs s, r)",
        "(if null (ft a) then 0 else 1), (if Z.eqb (fm a) 64 then 0 else 1), Z.abs s, r)",
    ),
    Rule(
        "in-place select constraints",
        "a select may update its source register while substituting every occurrence in its constraint",
        "GateDomain.v",
        "(negb(cons_mentions d k)||Pos.eqb r d)",
        "negb(cons_mentions d k)",
    ),
    Rule(
        "select through copied operands",
        "a select also retains constraints on the original value behind its copied operand",
        "GateDomain.v",
        "if same a a' && same b b' then direct else direct ++ branch_work st w d a' b'",
        "if true then direct else direct ++ branch_work st w d a' b'",
    ),
    Rule(
        "signed loop candidates",
        "a join proposes signed constraints when a cursor can cross unsigned zero",
        "GateDomain.v",
        "signed_candidates old out++signed_candidates out old",
        "[]",
    ),
    Rule(
        "signed interval normalization",
        "a negative 64-bit interval is read through its signed atom modulo the form width",
        "GateDomain.v",
        "if lo <? 0 then (fatom m (r, ES) c, false) else (fatom m x c, false)",
        "if false then (fatom m (r, ES) c, false) else (fatom m x c, false)",
    ),
    Rule(
        "signed interval comparisons",
        "a signed 64-bit comparison keeps the signed atom of an interval cursor",
        "GateDomain.v",
        "| x => interval_only x end) &&",
        "| x => false end) &&",
    ),
    Rule(
        "signed unknown comparisons",
        "read an unknown 64-bit value through its signed atom when a signed guard bounds it",
        "GateDomain.v",
        "sg && (match get st r with None | Some (FAlign _) => true | x => interval_only x end) &&",
        "sg && (match get st r with None | Some (FAlign _) => false | x => interval_only x end) &&",
    ),
    Rule(
        "signed modular steps",
        "step detection reads a 64-bit signed atom modulo the register width",
        "GateDomain.v",
        "| EX | ES => if Z.eqb (fm f) 64 && Z.eqb w 64 then Some",
        "| EX => if Z.eqb (fm f) 64 && Z.eqb w 64 then Some",
    ),
    Rule(
        "constant interval facts",
        "a register's explicit constant interval is available without recursive expansion",
        "GateDomain.v",
        "if Nat.eqb (List.length (ft f)) 0 && (wd r <=? fm f) && (0 <=? flo f) && (fhi f <? 2 ^ wd r)",
        "if false",
    ),
    Rule(
        "conditional select constraints",
        "each select alternative is checked under its own predicate refinement",
        "GateDomain.v",
        "let out := branch sb sa (branch sa sb base w d a b) w d b a in",
        "let out := base in",
    ),
    Rule(
        "conditional select values",
        "a select joins values from the predicate-refined states",
        "GateDomain.v",
        "Some sa, Some sb => join_fact sa sb (sel_fact sa a w) (sel_fact sb b w)",
        "Some sa, Some sb => join_fact st st (sel_fact st a w) (sel_fact st b w)",
    ),
    Rule(
        "minmax launch relations",
        "nonnegative min and max keep bounds relative to fixed launch atoms",
        "GateDomain.v",
        "if negb (null (ft fa) || null (ft fb)) then minmax_common_cons st out mn sg w d a b else\n   let terms := sorted_terms (ft fa ++ ft fb) in",
        "if negb (null (ft fa) || null (ft fb)) then minmax_common_cons st out mn sg w d a b else\n   if true then out else\n   let terms := sorted_terms (ft fa ++ ft fb) in",
    ),
    Rule(
        "minmax relations",
        "min and max retain their order relative to both operands",
        "GateDomain.v",
        "minmax_cons (transfer_dest st d None (value st i) false) mn sg w d a b",
        "transfer_dest st d None (value st i) false",
    ),
    Rule(
        "scaled inequalities",
        "partial coefficient cancellation and integer rounding of scaled bounds",
        "GateDomain.v",
        "inter acc (projected_cons_range ar g k)",
        "inter acc (cons_range ar g k)",
    ),
    Rule(
        "quotient relations",
        "a nonnegative 64-bit value minus twice its half is zero or one",
        "GateDomain.v",
        "  match const_of st b 32 false with\n  | None => st1\n  | Some k =>",
        "  match (None : option Z) with\n  | None => st1\n  | Some k =>",
    ),
    Rule(
        "select symmetry",
        "both operands of a select can supply a preserved constraint",
        "GateDomain.v",
        "if Pos.eqb r s then first else first ++ jobs b a",
        "if true then first else first ++ jobs b a",
    ),
    Rule(
        "projected cursor bounds",
        "preserve bounds against fixed scalar parameters before overwriting a cursor",
        "GateDomain.v",
        "if Z.eqb (wd r) 64 && is_fixed r then project_upper s d r else s",
        "if false then project_upper s d r else s",
    ),
    Rule(
        "launch geometry",
        "the binding's block size makes %ntid a constant and bounds %tid",
        "GateFront.v",
        "(nx, (bx, bx)); (ny, (by_, by_)); (nz, (bz, bz));",
        "(nx, (1, 1024)); (ny, (1, 1024)); (nz, (1, 64));",
    ),
    Rule(
        "R1 on the atom",
        "a guard also constrains the compared register's own atom",
        "GateDomain.v",
        "  match cmp_atom st a w sg with\n  | Some ga => if terms_eqb (ft fa) (ft ga)",
        "  match (if is_fixed a && false then cmp_atom st a w sg else None) with\n"
        "  | Some ga => if terms_eqb (ft fa) (ft ga)",
    ),
    Rule(
        "R3",
        "a load from an index input has the input's declared range",
        "GateDomain.v",
        "              | None => index_load st (transfer_dest st d None None false) (Z.of_nat bits) d a off",
        "              | None => transfer_dest st d None None false",
    ),
    Rule(
        "masked load",
        "R3 under a guard, joined with the old value and kept under a one-bit guard",
        "GateDomain.v",
        "              | Some _ => default_load st (sentinel_load st (guarded_index_load st g (Z.of_nat bits) d a off) g (Z.of_nat bits) d) g d",
        "              | Some _ => transfer_dest st d g None false",
    ),
    Rule(
        "R4",
        "x & m relates the result to the mask: r - m <= 0",
        "GateDomain.v",
        "and_cons st (transfer_dest st d None (value st i) false) (Z.of_nat w) d a rb",
        "transfer_dest st d None (value st i) false",
    ),
    Rule(
        "R5 moves",
        "comparisons of predicates move with a register's constant step",
        "GateDomain.v",
        "  if negb (cmp_mentions k d) then Some k else\n  match const_step step, cb k with",
        "  if negb (cmp_mentions k d) then Some k else\n  match (None : option Z), cb k with",
    ),
    Rule(
        "R6 implied",
        "a join keeps a constraint of one edge that follows from the other edge",
        "GateDomain.v",
        "if prepared_run s check prep then [k] else []",
        "if prepared_run s check prep then [] else []",
    ),
    Rule(
        "R7",
        "a check projects the constraints onto the access's atoms",
        "GateDomain.v",
        "  let cs' := fold_left (fun acc x => compact_projection (fm_step x acc)) elim\n"
        "                       (compact_projection kept) in",
        "  let cs' := kept in",
    ),
    Rule(
        "register-indexed local stores",
        "bound a local store's offset range and forget every cell its span may overlap",
        "GateDomain.v",
        "  match local_at A with\n  | Some (u, c) => Some (u, c, c)\n  | None =>\n"
        "      if negb (Z.eqb (fm A) 64) then None else",
        "  match local_at A with\n  | Some (u, c) => Some (u, c, c)\n  | None =>\n"
        "      if true then None else",
    ),
    Rule(
        "R8 cells",
        "a local load reads back the value that a store left in its cell",
        "GateDomain.v",
        "  | ILdL g bits d a off fr => load_cell st g bits d a off fr",
        "  | ILdL g bits d a off fr => transfer_dest st d g None false",
    ),
    Rule(
        "windows",
        "a shared or local access inside a declared variable is accepted",
        "GateDomain.v",
        "  | Some sg => var_of vars None sg a off width\n  end.",
        "  | Some sg => false\n  end.",
    ),
    Rule(
        "selp",
        "selp is a copy of its register, or b + (a - b) * p of two constants",
        "GateDomain.v",
        "let st1 := transfer_dest st d g (sel_value st a b p (Z.of_nat w)) false in",
        "let st1 := transfer_dest st d g None false in",
    ),
    Rule(
        "select of a register",
        "a selp of a register and another register or a constant moves a constraint that holds of both",
        "GateDomain.v",
        "sel_refined_cons_prepared st st1 (Z.of_nat w) d a b p",
        "st1",
    ),
    Rule(
        "min and max",
        "a min or max whose operands' ranges decide it is that operand",
        "GateDomain.v",
        "          if (if is_min then a1 <=? b0 else b1 <=? a0) then",
        "          if false then",
    ),
    Rule(
        "or of disjoint bits",
        "x | k is x + k when they share no bit",
        "GateDomain.v",
        "  if low_bits_clear st fa k then FForm (fshift fa k k) 0 None else",
        "  if low_bits_clear st fa k && false then FForm (fshift fa k k) 0 None else",
    ),
    Rule(
        "or.pred",
        "a failed disjunction keeps the negation of each comparison",
        "GateDomain.v",
        "| _,_ => pred_disj (get st a) (get st b) end",
        "| _,_ => None end",
    ),
    Rule(
        "xor.pred",
        "an exclusive or keeps the comparisons of both operands",
        "GateDomain.v",
        "  | IXorp _ _ a b => pred_common (get st a) (get st b)",
        "  | IXorp _ _ a b => None",
    ),
    Rule(
        "xor",
        "xor retains bounded register ranges and exact constant results",
        "GateDomain.v",
        "  | BXor => xor_fact st bw a b",
        "  | BXor => None",
    ),
    Rule(
        "constant immediate xor",
        "evaluate an immediate xor when the source register is proved constant",
        "GateDomain.v",
        "    | Some c => Some (FForm (fconst bw (Z.lxor c (z mod 2 ^ bw))) 0 None)",
        "    | Some _ => None",
    ),
    Rule(
        "bounded immediate xor",
        "retain a bounded register xor an immediate and their common power-of-two alignment",
        "GateDomain.v",
        "    | None =>\n      match reg_range st a with\n      | Some (a0, a1) =>\n          let k := z mod 2 ^ bw in\n          let n := Z.max (bitlen a1) (bitlen k) in\n          if (0 <=? a0) && (2 ^ n <=? 2 ^ bw)\n          then Some (FForm (range_form bw 0 (2 ^ n - 1))\n                           (Z.min (reg_align st a) (v2 k)) None)\n          else None\n      | None => None end\n",
        "    | None => None\n",
    ),
    Rule(
        "immediate-first subtraction",
        "lower constant minus register to exactly equivalent modular multiply-add",
        "GateFront.v",
        "| BSub => Some (IMad g w d b (OImm (-1)) (OImm z))",
        "| BSub => None",
    ),
    Rule(
        "or alignment",
        "x | y of two ranges keeps the alignment the two share",
        "GateDomain.v",
        "then Some (FForm (range_form bw 0 (a1 + b1)) (Z.min (falign st fa) (falign st fb)) None)",
        "then Some (FForm (range_form bw 0 (a1 + b1)) 0 None)",
    ),
    Rule(
        "high bits",
        "x less its low n bits is 2^n (x >> n), and x less its bits from n is x mod 2^n",
        "GateDomain.v",
        "      if whole && high && (0 <=? n) then",
        "      if false then",
    ),
    Rule(
        "bit-field bound",
        "a guard on a bit field or a shift of a register bounds that atom",
        "GateDomain.v",
        "          if match e with EBf _ _ | EShr _ => true | _ => false end",
        "          if false",
    ),
    Rule(
        "mask to the width",
        "x & -2^s keeps bits [s, wd x) of x, which has no bit from its width up",
        "GateDomain.v",
        "      if plain st a && (s <? wd a) then",
        "      if plain st a && (j <=? wd a) then",
    ),
    Rule(
        "vector of one",
        "the stored value { %r } of a store is the register %r",
        "GateFront.v",
        "operand (match single o1 with Some r => PReg r | None => o1 end)",
        "operand o1",
    ),
    Rule(
        "signed reading",
        "a comparison reads a copied register with its own extension",
        "GateDomain.v",
        "      then atom64 x (if sg then ES else EZ) else f",
        "      then f else f",
    ),
    Rule(
        "index value by bits",
        "a stored index below the sign bit is one value, read signed or unsigned",
        "GateDomain.v",
        "  by_form || by_bits.",
        "  by_form.",
    ),
    Rule(
        "absolute value",
        "abs.s<w> is the larger of a and -a",
        "GateFront.v",
        '      if String.eqb mn "abs" && (String.eqb t "s16" || String.eqb t "s32" || String.eqb t "s64") then',
        "      if false then",
    ),
    Rule(
        "R1 on the atom of a constant",
        "a comparison with a constant also constrains the register's own atom when a constraint mentions it",
        "GateDomain.v",
        "  | Some ga => if existsb (cons_mentions a) (acons st) then settle st (relate st ga fb dlo dhi) else Some st",
        "  | Some ga => Some st",
    ),
    Rule(
        "copy constraints",
        "a copy has its source's constraints, and its source's range as a constraint",
        "GateDomain.v",
        "              let st1 := sel_cons st (transfer_dest st d None (value st i) false) (wd d) d (OReg s) (OReg s) in\n"
        "              if existsb (cons_mentions s) (acons st) && Z.eqb (wd d) (wd s) then",
        "              let st1 := transfer_dest st d None (value st i) false in\n              if false then",
    ),
    Rule(
        "exact forms in a check",
        "a check reads a register's exact form as an equality on the register's atom",
        "GateDomain.v",
        "  let eqs := map (canon_cons_with ranges) (flat_map (form_eq_with ranges st) xs) in",
        "  let eqs : list constraint := [] in",
    ),
    Rule(
        "join through copies",
        "a join proposes each constraint read through its side's copies",
        "GateDomain.v",
        "if terms_eqb ts ts' then [] else [(ts',lo,hi)]",
        "if terms_eqb ts ts' then [] else []",
    ),
    Rule(
        "bounded pivot",
        "a join reads registers that step alike through one that a constraint mentions",
        "GateDomain.v",
        "  let bound := fun r => existsb (cons_mentions r) (acons old ++ acons out) in",
        "  let bound := fun r : reg => false in",
    ),
    Rule(
        "range of a sign extension",
        "a sign extension of a register of unknown range lies in the signed range of its width",
        "GateDomain.v",
        "  | ES => Some (match f_sr fs r with\n"
        "                | Some rg => rg\n"
        "                | None => (- 2 ^ (wd r - 1), 2 ^ (wd r - 1) - 1)\n"
        "                end)",
        "  | ES => f_sr fs r",
    ),
    Rule(
        "constraints through copies",
        "a state implies a constraint that one of its own has, both read through its copies",
        "GateDomain.v",
        """(if (lo <=? lo') && (hi' <=? hi) then
                           if terms_eqb ts0 ts' then true else
                           if terms_eqb (value_terms ts) (value_terms (copy_terms a ts'))
                           then terms_eqb (read tt) (canon_terms a (copy_terms a ts')) else false
                         else false)""",
        "terms_eqb ts0 ts'",
    ),
    Rule(
        "elimination by an equality",
        "a check eliminates an atom through an equality that has it",
        "GateDomain.v",
        "  match find (fun k : constraint => let '(_, lo, hi) := k in Z.eqb lo hi) with_x with",
        "  match (None : option constraint) with",
    ),
    Rule(
        "signed shared load",
        "a signed shared load narrower than its register is its bits and their sign extension",
        "GateFront.v",
        "                   else (v <- fresh bits ;; ret (Some [ILdS g bits v a off; ICvt g wd bits true dest v]))",
        "                   else ret None",
    ),
    Rule(
        "shared load at a variable",
        "a shared load's address may be a shared variable",
        "GateFront.v",
        '      | Some v => a <- reg_of ("shared:" ++ v) (Some 32%nat) ;; ret (Some (a, off))',
        "      | Some v => ret None",
    ),
    Rule(
        "scalar shared variable",
        "a scalar .shared declaration is a variable of one element",
        "GateParse.v",
        "  if ch hi p =? 59 then Some (name, align, bytes, p + 1) else\n",
        "",
    ),
    Rule(
        "low bits of a size parameter",
        "a load of a size parameter's low bits gets those bits of the sizes' value",
        "GateFront.v",
        "          if negb (Z.eqb off 0) || Nat.ltb wp w then ret None else",
        "          if negb (Z.eqb off 0) || negb (Nat.eqb w wp) then ret None else",
    ),
    Rule(
        "low half of a register",
        "a conversion of a 64-bit register's low 32 bits is the register modulo 2^32",
        "GateDomain.v",
        "      else if Nat.eqb to 32 && (Nat.eqb from 64 || Nat.eqb from 32 && Z.eqb (wd s) 64) then",
        "      else if Nat.eqb to 32 && Nat.eqb from 64 then",
    ),
    Rule(
        "relational extension",
        "project constraints before extending a nonwrapping 32-bit form",
        "GateDomain.v",
        "transfer_dest st d None (relational_cvt st to from sg d s) false",
        "transfer_dest st d None (value st i) false",
    ),
    Rule(
        "nested loop relations",
        "recover loop-stable relations and remove unsupported stale facts",
        "GateDomain.v",
        "Some (if ordinary_loop_recovery prog then prune_n 4 prog (recover_n 4 prog inv) else inv)",
        "Some inv",
    ),
    Rule(
        "canonical projection",
        "merge equivalent projected inequalities before elimination reaches its capacity",
        "GateDomain.v",
        "Definition compact_projection (cs : list constraint) : list constraint :=\n fold_right (fun k cs => let '(ts,lo,hi) := reduce_cons (orient_cons k) in\n   let same := fun k' => let '(ts',_,_) := k' in terms_eqb ts ts' in\n   let '(lo,hi) := fold_left (fun acc k' => let '(l,h) := acc in let '(_,l',h') := k' in\n      (Z.max l l',Z.min h h')) (filter same cs) (lo,hi) in\n   (ts,lo,hi) :: filter (fun k' => negb (same k')) cs) [] cs.",
        "Definition compact_projection (cs : list constraint) : list constraint := map (fun k => reduce_cons (orient_cons k)) cs.",
    ),
    Rule(
        "typed projection ranges",
        "intersect projected atoms with their machine-width ranges",
        "GateDomain.v",
        "let ar := fun x => inter (ar0 x) (natural_range x) in",
        "let ar := ar0 in",
    ),
    Rule(
        "cursor sum implication",
        "prove three-variable cursor bounds by constraint projection",
        "GateDomain.v",
        "if cursor_sum ts || (Z.eqb lo hi && mixed_cursor_sum ts) ||",
        "if false ||",
    ),
    Rule(
        "ordered scalar selection",
        "preserve register bounds across selection against a constant",
        "GateDomain.v",
        "select_order st (sel_refined_cons_prepared st st1 (Z.of_nat w) d a b p) (Z.of_nat w) d (known_operand st a) (known_operand st b)",
        "sel_refined_cons_prepared st st1 (Z.of_nat w) d a b p",
    ),
    Rule(
        "variable affine steps",
        "transport coupled constraints through nonwrapping variable additions and subtractions",
        "GateDomain.v",
        "arithmetic_relations st (transfer_live st i) i",
        "transfer_live st i",
    ),
    Rule(
        "paired cursor inference",
        "infer a bound on two changing cursors from their loop-entry values",
        "GateDomain.v",
        "pair_candidates old out++pair_candidates out old++",
        "",
    ),
    Rule(
        "stable widening constraints",
        "stop introducing new constraint shapes once loop widening starts",
        "GateDomain.v",
        "prepared_keep_two widen_now old (if widen_now then [] else novel_jobs old(acons out))",
        "prepared_keep_two widen_now old (novel_jobs old(acons out))",
    ),
    Rule(
        "projection pivots",
        "intersect bounds from every distinct shared coefficient ratio",
        "GateDomain.v",
        "Definition projected_cons_range (ar : atom -> option range) (f : form) (k : constraint) : option range :=\n let '(ts,_,_) := k in\n let pivots := fold_left (fun acc t => let '(x,c1) := t in let c0 := tget ts x in\n   if Z.eqb c0 0 || existsb (fun p => Z.eqb (fst p * c0) (c1 * snd p)) acc then acc else (c1,c0)::acc) (ft f) [] in\n fold_left (fun acc p => inter acc (projection_at ar f k (fst p) (snd p))) pivots None.\n",
        "Definition projected_cons_range (ar : atom -> option range) (f : form) (k : constraint) : option range :=\n let '(ts,_,_) := k in\n let pick := filter (fun t => tmem ts (fst t)) (ft f) in\n let core := filter (fun t => tmem (ft f) (fst t)) ts in\n match pick,core with (_,c1)::_,(_,c0)::_ => projection_at ar f k c1 c0 | _,_ => None end.\n",
    ),
    Rule(
        "quotient complements",
        "bound a register minus its shifted value without requiring exact coefficient cancellation",
        "GateDomain.v",
        "    | None, ((r, EShr n), _) =>\n        match find (fun u => match u with ((r',EX),_) | ((r',EZ),_) => Pos.eqb r r' | _ => false end) ts with\n        | Some ((_,e),c) => Some (r,e,EShr n,n,c) | _ => None end",
        "    | None, ((r, EShr n), c') =>\n        if Z.eqb (Z.rem c' (2 ^ n)) 0 then\n          match whole r (- (c' / 2 ^ n)) with Some ((_, e), c) => Some (r, e, EShr n, n, c) | _ => None end\n        else None",
    ),
    Rule(
        "forward cursor joins",
        "infer coupled cursor bounds when an early exit joins a loop exit",
        "GateDomain.v",
        "pair_candidates old out++pair_candidates out old++",
        "(if loop_edge then pair_candidates old out++pair_candidates out old else [])++",
    ),
    Rule(
        "conditional cursor steps",
        "check cursor-step constraints for both alternatives of an in-place selection",
        "GateDomain.v",
        "Definition selection_steps (st out : astate) (w : Z) (d : reg) (a b : opnd) : astate :=\n match a,b with OReg r,OReg s =>\n   if Pos.eqb d r then selected_step_cached st out w d s else\n   if Pos.eqb d s then selected_step_cached st out w d r else out\n | _,_ => out end.\n",
        "Definition selection_steps (st out : astate) (w : Z) (d : reg) (a b : opnd) : astate := if Z.eqb (wd d) 64 then out else out.\n",
    ),
    Rule(
        "selected register constants",
        "use exact register constants when preserving selected operand order",
        "GateDomain.v",
        "(known_operand st a) (known_operand st b)",
        "a b",
    ),
    Rule(
        "target form equations",
        "include checked target registers when collecting exact projection equations",
        "GateDomain.v",
        "let xs := atoms_nodup (map fst (ft g) ++ flat_map (fun k => let '(ts, _, _) := k in map fst ts) cs) [] in",
        "let xs := atoms_nodup (flat_map (fun k => let '(ts, _, _) := k in map fst ts) cs) [] in",
    ),
    Rule(
        "equation dependency elimination",
        "eliminate intermediate atoms introduced by exact projection equations",
        "GateDomain.v",
        "let elim := filter (fun x => if tmem (ft g) x then false\n"
        "                              else short_any (atom_eqb x) members) expanded in",
        "let elim := filter (fun x => if tmem (ft g) x then false\n"
        "                              else short_any (atom_eqb x) members) xs in",
    ),
    Rule(
        "masked sentinel implications",
        "retain the load guard implied by a value below its inactive signed-maximum sentinel",
        "GateDomain.v",
        "sentinel_load st (guarded_index_load st g (Z.of_nat bits) d a off) g (Z.of_nat bits) d",
        "guarded_index_load st g (Z.of_nat bits) d a off",
    ),
    Rule(
        "implied selection predicates",
        "refine relevant load predicates whose projected value becomes known inside a selection",
        "GateDomain.v",
        "| Some s => refine_origins s origins | None => None end.",
        "| Some s => Some s | None => None end.",
    ),
    Rule(
        "selection machine ranges",
        "supply complete machine-type ranges for unknown operands of relevant selection comparisons",
        "GateDomain.v",
        "refine (type_comparisons st p) p truth",
        "refine st p truth",
    ),
    Rule(
        "unsigned equality refinement",
        "use the unsigned reading of equal bit patterns to retain bounds across the sign bit",
        "GateDomain.v",
        "refine_cmp_raw st (unsigned_equality k)",
        "refine_cmp_raw st k",
    ),
    Rule(
        "complementary predicate bits",
        "retain that opposite exact comparisons produce predicate bits summing to one",
        "GateDomain.v",
        "complement_cons st out d {|cc:=c;cw:=w;csg:=sg;ca:=a;cb:=b|}",
        "out",
    ),
    Rule(
        "register implication checks",
        "validate register ranges and predicates from the whole source state",
        "GateDomain.v",
        "Definition register_fact_le (st : astate) (r : reg) (g : fact) : bool :=\n if fact_le_o st (eff_fact st r) (Some g) then true else\n match g with\n | FPred _ _ => predicate_fact_le st r g\n | FForm f al None =>\n   if Z.eqb (wd r) 1 && Z.eqb (fm f) 1 && null (ft f) && (al <=? 0) then\n     if (flo f <=? 0) && (1 <=? fhi f) then true else\n     match predicate_constant st r with Some c => (flo f <=? c) && (c <=? fhi f) | None => false end else\n   if Z.eqb (fm f) 64 && Z.eqb (wd r) 64 && null (ft f) && (al <=? 0) then\n   match check_range st (fatom 64 (r,EX) 1) with\n   | Some (lo,hi) => (flo f <=? lo) && (hi <=? fhi f)\n   | None => false end else false\n | _ => false end.\n\n",
        "Definition register_fact_le (st : astate) (r : reg) (g : fact) : bool :=\n if fact_le_o st (eff_fact st r) (Some g) then true else\n match g with\n | FPred _ _ => false\n | FForm f al None =>\n   if Z.eqb (wd r) 1 && Z.eqb (fm f) 1 && null (ft f) && (al <=? 0) then\n     if (flo f <=? 0) && (1 <=? fhi f) then true else\n     match predicate_constant st r with Some c => (flo f <=? c) && (c <=? fhi f) | None => false end else\n   false\n | _ => false end.\n\n",
    ),
    Rule(
        "semantic predicate joins",
        "retain predicates equivalent under the states at a control-flow join",
        "GateDomain.v",
        "if predicate_fact_le out r f then Some f else\n                    if predicate_fact_le old r g then Some g else None",
        "None",
    ),
    Rule(
        "exact pair cursor sums",
        "propose constant sums of two cursors only when both edges prove the exact sum",
        "GateDomain.v",
        "if null (ft h) then if short_all checks [upper;lower] then [(ts,flo h,fhi h)] else []",
        "if null (ft h) then []",
    ),
    Rule(
        "triple cursor invariants",
        "retain a checked sum of two cursors and an iteration counter",
        "GateDomain.v",
        "++triple_candidates old out++triple_candidates out old",
        "",
    ),
    Rule(
        "loop exit cursor sums",
        "retain the consumed-count sum when a loop exit meets the zero-iteration path",
        "GateDomain.v",
        "if loop_edge then [] else exit_sum_candidates old out++exit_sum_candidates out old",
        "[]",
    ),
    Rule(
        "counter range recovery",
        "recover finite cursor bounds from exact conserved sums after widening",
        "GateDomain.v",
        "recover_cursor_ranges ({| amap := m;",
        "({| amap := m;",
    ),
    Rule(
        "mixed width cursor projection",
        "project exact coupled sums containing zero-extended 32-bit positions",
        "GateDomain.v",
        "cursor_sum ts || (Z.eqb lo hi && mixed_cursor_sum ts)",
        "cursor_sum ts",
    ),
    Rule(
        "predicate negation translation",
        "translate one-bit not.pred with its exact truth table",
        "GateFront.v",
        'String.eqb mn "not" && list_eqs sf ["pred"] && Nat.eqb wd 1',
        "false",
    ),
    Rule(
        "negated predicate facts",
        "retain the comparisons and exactness of a negated predicate",
        "GateDomain.v",
        "if Nat.eqb w 1 && Z.eqb (wd a) 1 then",
        "if false then",
    ),
    Rule(
        "constant Boolean identities",
        "preserve operand facts through conjunction and disjunction with known bits",
        "GateDomain.v",
        "  | IAndp _ _ a b => match known_bit st a,known_bit st b with\n    | Some false,_ | _,Some false => bit_fact false\n    | Some true,_ => get st b | _,Some true => get st a\n    | _,_ => pred_union (get st a) (get st b) end\n  | IOrp _ _ a b => match known_bit st a,known_bit st b with\n    | Some true,_ | _,Some true => bit_fact true\n    | Some false,_ => get st b | _,Some false => get st a\n    | _,_ => pred_disj (get st a) (get st b) end\n",
        "  | IAndp _ _ a b => pred_union (get st a) (get st b)\n  | IOrp _ _ a b => pred_disj (get st a) (get st b)\n",
    ),
    Rule(
        "predicate result folding",
        "retain a proved constant result when its contradictory branch is impossible",
        "GateDomain.v",
        "fold_predicate (selection_bounds st (sum_bounds (half_bounds st (arithmetic_relations st (transfer_live st i) i) i) i) i) i",
        "selection_bounds st (sum_bounds (half_bounds st (arithmetic_relations st (transfer_live st i) i) i) i) i",
    ),
    Rule(
        "compound false branches",
        "join the possible failing comparisons of an exact conjunction",
        "GateDomain.v",
        "Definition refine_cmps (st : astate) (cs : list cmpf) (exact all : bool) : option astate :=\n  if all then refine_all st cs\n  else if negb exact then Some st\n  else join_alternatives st (flat_map (fun k => match refine_cmp st (negate k) with Some s => [s] | None => [] end) cs).\n\n",
        "Definition refine_cmps (st : astate) (cs : list cmpf) (exact all : bool) : option astate :=\n  if all then refine_all st cs\n  else if negb exact then Some st\n  else match cs with\n       | [k] => refine_cmp st (negate k)\n       | _ => if forallb (fun k => negb (is_some (refine_cmp st (negate k)))) cs then None else Some st\n       end.\n\n",
    ),
    Rule(
        "predicate bit joins",
        "retain a common known bit when ordinary predicate joins lose exactness",
        "GateDomain.v",
        "let j := join_predicate_bits_ready old out r j in",
        "",
    ),
    Rule(
        "predicate bit implication",
        "validate inferred bit constants and the natural one-bit range",
        "GateDomain.v",
        "   if Z.eqb (wd r) 1 && Z.eqb (fm f) 1 && null (ft f) && (al <=? 0) then\n     if (flo f <=? 0) && (1 <=? fhi f) then true else\n     match predicate_constant st r with Some c => (flo f <=? c) && (c <=? fhi f) | None => false end else\n",
        "",
    ),
    Rule(
        "moving loop bounds",
        "infer counter bounds when the counter and an existing fixed expression bound advance together",
        "GateDomain.v",
        "moving_bound_candidates old out++moving_bound_candidates out old",
        "[]",
    ),
    Rule(
        "affine range widening",
        "widen both affine input ranges when their shared terms outlive the first iterations",
        "GateDomain.v",
        "  let widen_old := fun (f jf : fact) =>\n    match f with\n    | FForm a al _ =>\n        if null (ft a) then widen extra f jf else\n        match form_range old a with\n        | Some (p, q) =>\n            match jf with\n            | FForm b bl _ =>\n                match form_range old b, form_range out b with\n                | Some (l,h), Some (l',h') =>\n                    widen extra (FForm (range_form (fm a) p q) al None)\n                          (FForm (range_form (fm b) (Z.min l l') (Z.max h h')) bl None)\n                | _,_ => None end\n            | _ => None end\n        | None => None\n        end\n    | _ => widen extra f jf\n    end in\n",
        "  let widen_old := fun (f jf : fact) =>\n    match f with\n    | FForm a al _ =>\n        if null (ft a) then widen extra f jf else\n        match form_range old a with\n        | Some (p, q) => widen extra (FForm (range_form (fm a) p q) al None) jf\n        | None => None\n        end\n    | _ => widen extra f jf\n    end in\n",
    ),
    Rule(
        "bounded narrow lifting",
        "check the required extension bounds before recursively expanding copied ranges",
        "GateDomain.v",
        "match bounded_form_range st f (match e with ES => -2^31 | _ => 0 end) (match e with ES => 2^31-1 | _ => 2^32-1 end) with",
        "match form_range st f with",
    ),
    Rule(
        "private projected bounds",
        "project shared and local-memory bounds through related constraints",
        "GateDomain.v",
        "else check_range st f\n  | None => check_range st f",
        "else None\n  | None => None",
    ),
    Rule(
        "bounded private ranges",
        "query the required allocation bounds before expanding recursive ranges",
        "GateDomain.v",
        "match bounded_form_range st f lo hi with",
        "match form_range st f with",
    ),
    Rule(
        "shallow span gap checks",
        "use the shallow range pass before the full projected access check",
        "GateDomain.v",
        "Definition inside (sg : astate) (r : option range) (rest ext : form) (width : Z) : bool :=\n  match r with\n  | None => false\n  | Some (x, y) =>\n      let direct := null (ft ext) && (y + width <=? flo ext) in\n      let D := fadd ext (fscale rest (-1) 64) in\n      let through := match form_range_near sg D, form_range_near sg ext with\n                     | Some (xd, yd), Some (xe, ye) =>\n                         (width <=? xd) && (y + yd <? 2 ^ 64) && (0 <=? xe) && (ye <? 2 ^ 64)\n                     | _, _ => false\n                     end in\n      (0 <=? x) && (direct || through)\n  end.\n\n",
        "Definition inside (sg : astate) (r : option range) (rest ext : form) (width : Z) : bool :=\n  match r with\n  | None => false\n  | Some (x, y) =>\n      let direct := null (ft ext) && (y + width <=? flo ext) in\n      let D := fadd ext (fscale rest (-1) 64) in\n      let through := match form_range sg D, form_range sg ext with\n                     | Some (xd, yd), Some (xe, ye) =>\n                         (width <=? xd) && (y + yd <? 2 ^ 64) && (0 <=? xe) && (ye <? 2 ^ 64)\n                     | _, _ => false\n                     end in\n      (0 <=? x) && (direct || through)\n  end.\n\n",
    ),
    Rule(
        "shallow span offset checks",
        "bound the offset with the shallow pass before full access projection",
        "GateDomain.v",
        "inside sg (form_range_near sg rest) rest (sp_ext sp) width",
        "inside sg (form_range sg rest) rest (sp_ext sp) width",
    ),
    Rule(
        "memoized access projection",
        "share cached depth ranges across canonicalization, equalities and access projection",
        "GateDomain.v",
        " let ranges := rft_on (cons_index (acons st)) st (term_regs (projection_terms st extra f)) in\n",
        " let ranges := tighten (cons_index (acons st)) st (rf_plain (cons_index (acons st)) DEPTH st) in\n",
    ),
    Rule(
        "memoized range contexts",
        "reuse deep register-range expansions in every range query",
        "GateDomain.v",
        "Definition rft_on (ix : PM.t (list constraint)) (st : astate) (regs : list reg) : rfuns :=\n  tighten ix st (rf_on (reach_regs ix st regs) ix DEPTH st).\n",
        "Definition rft_on (ix : PM.t (list constraint)) (st : astate) (regs : list reg) : rfuns :=\n  tighten ix st (rf_plain ix DEPTH st).\n",
    ),
    Rule(
        "shared lifting ranges",
        "compute one range context for the residue and integer-value bounds",
        "GateDomain.v",
        "Definition lift_by_range (st : astate) (r : reg) (f : form) (e : ext) : option form :=\n  match e with\n  | EZ | ES =>\n      if negb (Z.eqb (wd r) 32) || negb (Z.eqb (fm f) 32) then None else\n      let f' := mk 32 (swap_to e (ft f)) (flo f) (fhi f) in\n      let ix := cons_index (acons st) in\n      let ar := atom_range_with (rft_on ix st (range_reads ix (ft f' ++ [((r,e),1)]))) in\n      let n := match e with ES => (- 2 ^ 31, 2 ^ 31 - 1) | _ => (0, 2 ^ 32 - 1) end in\n      let '(a, b) := match form_range_with st ar {| fm := 64; ft := [((r, e), 1)]; flo := 0; fhi := 0 |} with\n                     | Some (a, b) => (Z.max a (fst n), Z.min b (snd n))\n                     | None => n\n                     end in\n      let '(a,b) := round (Z.min 32 (reg_align st r)) (a,b) in\n      match form_range_with st (fun x => match ar x with Some g => Some g | None => natural_range x end) f' with\n      | Some (p, q) =>\n          if (q - a <? 2 ^ 32) && (b - p <? 2 ^ 32)\n          then Some {| fm := 64; ft := ft f'; flo := flo f'; fhi := fhi f' |} else None\n      | None => None\n      end\n  | _ => None\n  end.\n\n",
        "Definition lift_by_range (st : astate) (r : reg) (f : form) (e : ext) : option form :=\n  match e with\n  | EZ | ES =>\n      if negb (Z.eqb (wd r) 32) || negb (Z.eqb (fm f) 32) then None else\n      let f' := mk 32 (swap_to e (ft f)) (flo f) (fhi f) in\n      let n := match e with ES => (- 2 ^ 31, 2 ^ 31 - 1) | _ => (0, 2 ^ 32 - 1) end in\n      let '(a, b) := match form_range_with st (atom_range_with (rft_on (cons_index (acons st)) st (range_reads (cons_index (acons st)) (ft f' ++ [((r,e),1)])))) {| fm := 64; ft := [((r, e), 1)]; flo := 0; fhi := 0 |} with\n                     | Some (a, b) => (Z.max a (fst n), Z.min b (snd n))\n                     | None => n\n                     end in\n      let '(a,b) := round (Z.min 32 (reg_align st r)) (a,b) in\n      match form_range_with st (fun x => match (atom_range_with (rft_on (cons_index (acons st)) st (range_reads (cons_index (acons st)) (ft f' ++ [((r,e),1)])))) x with Some g => Some g | None => natural_range x end) f' with\n      | Some (p, q) =>\n          if (q - a <? 2 ^ 32) && (b - p <? 2 ^ 32)\n          then Some {| fm := 64; ft := ft f'; flo := flo f'; fhi := fhi f' |} else None\n      | None => None\n      end\n  | _ => None\n  end.\n\n",
    ),
    Rule(
        "query range caches",
        "cache the range layers of a query on the registers that its ranges reach",
        "GateDomain.v",
        "  tighten ix st (rf_on (reach_regs ix st regs) ix DEPTH st).",
        "  tighten ix st (rf_with_cache ix true DEPTH st).",
    ),
    Rule(
        "affine quotient bounds",
        "project midpoint bounds against materialized affine launch expressions",
        "GateDomain.v",
        "  fold_left (fun out f => project_affine_upper out d f) (fixed_affine_targets st) out.",
        "  out.",
    ),
    Rule(
        "combined clamp bounds",
        "retain a constant clamp bound relative to the sum of positive fixed terms",
        "GateDomain.v",
        "   let terms := sorted_terms (ft fa ++ ft fb) in\n   let combined := filter (fun t => is_fixed (fst (fst t)) && (0 <? snd t)) terms in\n   let targets := map (fun t => [t]) terms ++ (if Nat.leb 2 (length combined) then [combined] else []) in",
        "   let terms := sorted_terms (ft fa ++ ft fb) in\n   let combined := filter (fun t => is_fixed (fst (fst t)) && (0 <? snd t)) terms in\n   let targets := map (fun t => [t]) terms ++ [] in",
    ),
    Rule(
        "weighted cursor implication",
        "project one cursor against bounded weighted launch terms at joins",
        "GateDomain.v",
        " || affine_cursor_bound ts",
        "",
    ),
    Rule(
        "symbolic private extent",
        "check a private access against its complete launch-sized allocation",
        "GateDomain.v",
        "| Some (_,hi) => (upper + hi <=? window_size) && inside_deep st rest bytes width",
        "| Some (_,hi) => false",
    ),
    Rule(
        "shared search recovery",
        "recover and validate register relations in loops over launch-sized shared arrays",
        "GateDomain.v",
        "| ILdS _ _ _ _ _ | IShA _ _ _ _ _ _ => existsb (fun p => negb (null (ft (snd p)))) (lf_shared F)",
        "| ILdS _ _ _ _ _ | IShA _ _ _ _ _ _ => false",
    ),
    Rule(
        "centered exact constants",
        "preserve descending counters with equivalent centered machine-width constants",
        "GateDomain.v",
        "  transfer_dest_raw st d g (canonical_fact d new) total.",
        "  transfer_dest_raw st d g new total.",
    ),
    Rule(
        "bounded narrow loop pivots",
        "retain nonnegative 32-bit cursor relations as checked loop invariant proposals",
        "GateDomain.v",
        "if negb (can_pivot a b) then",
        "if negb (Z.eqb (fm a) 64) then",
    ),
    Rule(
        "aligned integer lifting",
        "exclude impossible wrapped representatives using the register alignment",
        "GateDomain.v",
        "      let '(a,b) := round (Z.min 32 (reg_align st r)) (a,b) in\n",
        "",
    ),
    Rule(
        "aligned private offsets",
        "round allocation-relative access ranges by the known address and base alignment",
        "GateDomain.v",
        "| Some rg => let '(x,y) := round al rg in (upper + flo bytes",
        "| Some rg => let '(x,y) := rg in (upper + flo bytes",
    ),
    Rule(
        "bounded loop cases",
        "check separate loop invariants after partitioning launch-fixed allocation selections",
        "GateDomain.v",
        "| Some (r,cut) => let plan := loop_plan prog in\n     if check_partition",
        "| Some (r,cut) => let plan := [] in\n     if check_partition",
    ),
    Rule(
        "conditional range loop cases",
        "discover bounded cases for a counter advanced by a selected continuation bit",
        "GateDomain.v",
        "| Some initial => let count := limit-initial+1 in\n"
        "         if (1 <=? count) && (count <=? 8)",
        "| Some initial => let count := limit-initial+1 in\n         if false",
    ),
    Rule(
        "shared allocation launch cases",
        "check exhaustive launch-fixed cases for shared allocation selections",
        "GateDomain.v",
        " selection_split F (concat prog) (PM.empty reg) (PM.empty (reg * Z)).",
        " None.",
    ),
    Rule(
        "counted-loop exit bounds",
        "propose a checked strict counter bound from a counted-loop exit",
        "GateDomain.v",
        "exit_bound_candidates old out++start_bound",
        "[]++start_bound",
    ),
    Rule(
        "constant-entry pair projection",
        "project two-register cursor bounds when one starts at a constant",
        "GateDomain.v",
        " || pair_cursor a ts",
        "",
    ),
    Rule(
        "conversion residue facts",
        "retain the low 32-bit residue when an exact 64-bit lift is not yet proved",
        "GateDomain.v",
        "| None => Some (FForm (fat f 32) 0 (if signed then None else Some (0, 2 ^ 32 - 1)))",
        "| None => None",
    ),
    Rule(
        "signed conversion windows",
        "constrain the signed reading of an extended 32-bit residue",
        "GateDomain.v",
        "cvt_signed_bound st (transfer_dest st d None (relational_cvt st to from sg d s) false) to from sg d s",
        "transfer_dest st d None (relational_cvt st to from sg d s) false",
    ),
    Rule(
        "residue range equalities",
        "recover an exact integer equality from a residue and proved windows",
        "GateDomain.v",
        "project_range st (flat_map (residue_eq (project_range st []) st) (atoms_nodup (map fst (ft f)) [])) f",
        "project_range st [] f",
    ),
    Rule(
        "shifted masked forms",
        "preserve the exact shifted source when masking a shifted register",
        "GateDomain.v",
        "      match shr_of e with\n      | Some k =>",
        "      match (match e with EX => Some 0 | _ => None end) with\n      | Some k =>",
    ),
    Rule(
        "shift source relations",
        "carry a source bound through its quotient by a power of two",
        "GateDomain.v",
        "  let links := shift_links xs in",
        "  let links : list constraint := [] in",
    ),
    Rule(
        "integer constraint reduction",
        "divide shared coefficient factors and round integer bounds inward",
        "GateDomain.v",
        "let '(ts,lo,hi) := reduce_cons (orient_cons k) in",
        "let '(ts,lo,hi) := orient_cons k in",
    ),
    Rule(
        "variable remainder intervals",
        "bound a nonnegative remainder by the dividend and a positive variable divisor",
        "GateDomain.v",
        "      if rm && (0 <=? x) then",
        "      if false then",
    ),
    Rule(
        "symbolic remainder relations",
        "retain the remainder upper bound relative to its exact divisor form",
        "GateDomain.v",
        "              rem_cons st (transfer_dest st d None (value st i) false) sg (Z.of_nat w) d a b",
        "              transfer_dest st d None (value st i) false",
    ),
    Rule(
        "constraint bounded divisors",
        "prove divisor positivity and representability through existing constraints",
        "GateDomain.v",
        "match cmp_range st (OReg a) w sg, cmp_deep_range st b w sg with",
        "match cmp_range st (OReg a) w sg, cmp_range st b w sg with",
    ),
    Rule(
        "shallow constraint interval classification",
        "decide contained or disjoint shallow integer ranges before relational projection",
        "GateDomain.v",
        "match range_constraint lo hi rg with",
        "match (None : option bool) with",
    ),
    Rule(
        "midpoint halving bounds",
        "bound the half of a two-register difference below the difference, so that lo + half < hi",
        "GateDomain.v",
        "(half_bounds st (arithmetic_relations st (transfer_live st i) i) i)",
        "(arithmetic_relations st (transfer_live st i) i)",
    ),
    Rule(
        "midpoint scalar bounds",
        "project a search midpoint and its neighbours mid + c on the launch scalars that bound their terms",
        "GateDomain.v",
        "Definition sum_bounds (out : astate) (i : instr) : astate :=\n  match i with",
        "Definition sum_bounds (out : astate) (i : instr) : astate :=\n  match INop with",
    ),
    Rule(
        "search update selection bounds",
        "bound an in-place selection update by the launch scalars of its operands' forms in each branch",
        "GateDomain.v",
        "Definition selection_bounds (st out : astate) (i : instr) : astate :=\n  match i with",
        "Definition selection_bounds (st out : astate) (i : instr) : astate :=\n  match INop with",
    ),
    Rule(
        "loop register order and bound proposals",
        "propose the order of two search bounds and their start bounds by launch scalars at joins",
        "GateDomain.v",
        "++order_candidates old out else[])++bound_candidates old out loop_edge++",
        " else[])++",
    ),
    Rule(
        "sentinel bound proposals",
        "propose the bound of a search result that may be -1 by each launch scalar where its paths merge",
        "GateDomain.v",
        "++sentinel_candidates old out++sentinel_candidates out old",
        "",
    ),
    Rule(
        "proportional step proposals",
        "propose m * i - j constant for two registers that a loop steps in the ratio m",
        "GateDomain.v",
        "++proportion_candidates old out++",
        "++",
    ),
    Rule(
        "equality exit thresholds",
        "widen a counter to K - 1 or K + 1 for each constant K that the program tests it for equality with",
        "GateDomain.v",
        "join_states_with (program_thresholds prog) o s",
        "join_states_with [] o s",
    ),
    Rule(
        "loops that clear dynamic shared memory",
        "clear dynamic shared memory whose bytes depend on the sizes with a loop of the zeroing"
        " prologue whose count the size parameters give",
        "GateFront.v",
        "  else if negb (null dynamic) && negb (defined (size_parameters Mf)) then",
        "  else if negb (null dynamic) then",
    ),
)


RULE_NAMES = {rule.name: rule for rule in RULES}


@dataclass(frozen=True)
class Group:
    """Rules that are turned off together, their changes made in the order of
    `rules`. A rule of `subsumed` is one whose text a change of `rules`
    replaces: that change turns off at least what the rule does."""

    name: str
    description: str
    rules: tuple[str, ...]
    subsumed: tuple[str, ...] = ()


# Every rule of RULES is in exactly one group (tests/eval/test_ablation.py).
GROUPS = (
    Group(
        "termination",
        "the proofs that every loop of an accepted kernel ends",
        (
            "loop end at a counter value",
            "loop end by a counter",
            "loop end by a binary search",
            "loop end by a merge-path search",
            "loop end by a dead back edge",
            "loop end by a power-of-two counter step",
            "counter kept from one value by an order test",
            "counter update through conversions",
            "binary search bounds from the loop entry",
        ),
    ),
    Group(
        "barrier participation routes",
        "the proofs that every thread of a block reaches each barrier in the same order",
        (
            "ordered CTA barrier events",
            "finite CTA barrier phases",
            "CTA context event schedules",
            "mixed context event schedules",
            "deterministic uniform-prefix context values",
            "step-1 inequality counted loops",
            "acyclic event segments",
            "shared ranked prefix for straight event tails",
            "finite repeated event words",
            "structural straight-line event progress",
        ),
    ),
    Group(
        "event-free loop summaries",
        "the summaries of loops between barriers that let a participation proof pass them",
        (
            "checked loop summaries in occurrence segments",
            "checked signed-counter occurrence loops",
            "checked compound-predicate occurrence loops",
            "unrolled event-free range loops",
            "rank-ended loop summaries",
            "ranked CTA loop summaries",
            "checked forward-branch loop summaries",
            "local-cell straight-loop progress",
            "local-cell branched-loop progress",
            "loop increment source aliases",
        ),
        ("compared induction counter selection",),
    ),
    Group(
        "shared-value premises",
        "the values equal in every thread of a block, shared memory included, that decide a barrier's branches",
        (
            "shared value participation",
            "shared value regions with loops",
            "shared address facts across blocks",
        ),
    ),
    Group(
        "warp collectives",
        "the proofs that every lane of a warp reaches each warp collective",
        (
            "checked physical-warp event words",
            "warp-uniform entry exits",
            "mixed collective event markers",
        ),
    ),
    Group(
        "search bounds",
        "the bounds of binary-search and merge-path midpoints and their results",
        (
            "midpoint halving bounds",
            "midpoint scalar bounds",
            "search update selection bounds",
            "loop register order and bound proposals",
            "sentinel bound proposals",
            "affine quotient bounds",
            "shared search recovery",
        ),
    ),
    Group(
        "loop cursor relations",
        "the relations between two or three registers that a loop steps together",
        (
            "cursor sum implication",
            "weighted cursor implication",
            "constant-entry pair projection",
            "paired cursor inference",
            "exact pair cursor sums",
            "triple cursor invariants",
            "loop exit cursor sums",
            "counter range recovery",
            "proportional step proposals",
            "moving loop bounds",
            "conditional cursor steps",
            "variable affine steps",
            "projected cursor bounds",
        ),
        (
            "mixed width cursor projection",
            "forward cursor joins",
        ),
    ),
    Group(
        "loop invariants and widening",
        "counted-loop bounds, widening thresholds, pivots, and the joins of loop states",
        (
            "counted-loop exit bounds",
            "monotone affine start bounds",
            "symbolic counted-loop starts",
            "equality counted-loop exits",
            "numeric loop pivot priority",
            "loop pivot ordering",
            "bounded narrow loop pivots",
            "bounded pivot",
            "signed loop candidates",
            "stable widening constraints",
            "launch widening thresholds",
            "universal bounds before widening",
            "affine range widening",
            "equality exit thresholds",
            "nested loop relations",
            "bounded loop cases",
            "conditional range loop cases",
            "join through copies",
            "R6 implied",
            "per-edge revisit limits",
            "affine interval bound witnesses",
            "CFG liveness boundary trimming",
            "checked interval spatial certificates",
        ),
        ("layout-independent counted bounds",),
    ),
    Group(
        "relational constraints and projection",
        "the projection of linear constraints onto an access's atoms in a bound check",
        (
            "R7",
            "scaled inequalities",
            "projection pivots",
            "canonical projection",
            "typed projection ranges",
            "elimination by an equality",
            "exact forms in a check",
            "constraints through copies",
            "copy constraints",
            "target form equations",
            "equation dependency elimination",
            "bit-aware projection residuals",
            "fixed-goal projection components",
            "residue range equalities",
            "natural machine bounds for modular form equalities",
            "register implication checks",
            "predicate bit implication",
        ),
    ),
    Group(
        "evaluation shortcuts",
        "memoized and shallow range queries that make a check faster without changing what it proves",
        (
            "memoized range contexts",
            "memoized access projection",
            "shared lifting ranges",
            "shallow span gap checks",
            "shallow span offset checks",
            "exact masked result shortcut",
            "missing certificate cases preflight",
            "index load width eligibility",
            "stable range layers",
            "constant interval facts",
            "bounded narrow lifting",
            "bounded private ranges",
            "shallow constraint interval classification",
        ),
        ("query range caches",),
    ),
    Group(
        "selections and min/max",
        "the values and constraints of selp, min, and max",
        (
            "selp",
            "ordered scalar selection",
            "select of a register",
            "select through copied operands",
            "select symmetry",
            "constant selection atom interpretations",
            "in-place select constraints",
            "conditional select constraints",
            "conditional select values",
            "signed selection ordering",
            "selection comparison normalization",
            "selection machine ranges",
            "implied selection predicates",
            "min and max",
            "minmax relations",
            "minmax launch relations",
            "combined clamp bounds",
            "symbolic minmax common bounds",
        ),
        ("selected register constants",),
    ),
    Group(
        "predicates and branch guards",
        "the facts that comparisons, predicate logic, and branch guards give",
        (
            "R1 on the atom",
            "R1 on the atom of a constant",
            "bit-field bound",
            "R5 moves",
            "comparison alias transport",
            "comparison operand snapshots",
            "or.pred",
            "xor.pred",
            "predicate negation translation",
            "negated predicate facts",
            "predicate result folding",
            "compound false branches",
            "complementary predicate bits",
            "predicate bit joins",
            "semantic predicate joins",
            "copied predicate recovery",
        ),
    ),
    Group(
        "index loads and launch facts",
        "the declared ranges of index inputs, guarded loads, launch geometry, and size parameters",
        (
            "R3",
            "masked load",
            "conditional load guard recovery",
            "index value by bits",
            "launch geometry",
            "derived size entry constraints",
            "low bits of a size parameter",
        ),
        (
            "conditional load defaults",
            "masked sentinel implications",
        ),
    ),
    Group(
        "alignment",
        "the known low zero bits of addresses and offsets",
        (
            "left shift explicit alignment",
            "symbolic step alignment",
            "or alignment",
            "aligned integer lifting",
            "aligned private offsets",
        ),
    ),
    Group(
        "local and shared memory",
        "local-memory cells and the bounds of shared and local variables",
        (
            "register-indexed local stores",
            "R8 cells",
            "windows",
            "signed shared load",
            "shared load at a variable",
            "scalar shared variable",
            "private projected bounds",
            "symbolic private extent",
            "shared allocation launch cases",
            "loops that clear dynamic shared memory",
        ),
    ),
    Group(
        "integer arithmetic and conversions",
        "bitwise operations, shifts, quotients, remainders, sign and width conversions",
        (
            "R4",
            "or of disjoint bits",
            "OR already-set bits",
            "complemented contiguous OR",
            "xor",
            "constant immediate xor",
            "bounded immediate xor",
            "high bits",
            "mask to the width",
            "rounded mask relations",
            "modular low-mask residue",
            "quotient right shift",
            "quotient relations",
            "quotient complements",
            "shifted masked forms",
            "shift source relations",
            "low bitfield source bound",
            "variable remainder intervals",
            "symbolic remainder relations",
            "constraint bounded divisors",
            "absolute value",
            "immediate-first subtraction",
            "vector of one",
            "scalar-to-vector bit unpack lowering",
            "constant Boolean identities",
            "centered exact constants",
            "general integer conversion widths",
            "checked conversion form lift",
            "low half of a register",
            "signed conversion windows",
            "relational extension",
            "conversion residue facts",
            "signed unknown comparisons",
            "signed interval comparisons",
            "signed interval normalization",
            "signed modular steps",
            "range of a sign extension",
            "signed reading",
            "unsigned equality refinement",
            "integer constraint reduction",
        ),
    ),
)

BASELINE = Group("baseline", "unchanged checker", ())

# the files of the gate, in dependency order, as the Makefile builds them
GATE = re.compile(r"^FILES := (.*?)\n\n", re.S | re.M)
# Lemmas and theorems proved in a Section without `Proof using`, with the
# Section variables that each uses in the trusted library. -vos runs such a
# proof to find them, and a change that disables a decision rule may break the
# proof; with the variables declared, the proof does not run.
PROOF_USING = (
    ("GateSoundRange.v", "plain_range_sound", "F E rs Hrs"),
    ("GateSoundRange.v", "atom_range_with_sound", "F E HE rs Hrs"),
    ("GateSoundRange.v", "fact_range_sound", "F E HE t st rs Hst Hrs"),
    ("GateSoundRange.v", "rf_stable_layers_sound", "F E HE t st rs Hst Hrs"),
    ("GateSoundRange.v", "rft_on_sound", "F E HE t st rs Hst Hrs"),
    ("GateSoundRange.v", "form_range_nat_sound", "F E HE t st rs Hst Hrs"),
    ("GateSoundTransfer.v", "cons_prepare_exact", "F"),
    ("GateSoundTransfer.v", "branch_work_spec", "F"),
    ("GateSoundTransfer.v", "shared_check_exact", "F"),
    ("GateSoundTransfer.v", "branch_cons_exact", "F"),
    ("GateSoundTransfer.v", "sel_cons_exact", "F"),
    ("GateInterval.v", "branch_work_job_sound", "F prog E HE"),
    ("GateInterval.v", "sel_value_fact_sound", "F prog E HE"),
    (
        "GateContext.v",
        "warp_exit_reach",
        "F uniform r w before last rest E Hlaunch Hlf Hrange Hpure Hwf Hbranch",
    ),
    ("GateTerm.v", "te_step_facts", "F prog cert E l h d Hcert HE"),
    ("GateTerm.v", "te_alias_some", "F prog l h d a al dz Hrw Hal"),
    (
        "GateTerm.v",
        "ts_loop_proof_at",
        "F prog cert E l h d Hcert HE Hshape He Hok neg p c ra rb k0"
        " Hlast Hpc Hplo Hphi Hlw Hk1 Hk2 Hgm",
    ),
)


def _files(library: Path) -> list[str]:
    matched = GATE.search((library / "Makefile").read_text())
    if matched is None:
        raise ValueError("formal/Makefile has no FILES list")
    files = matched.group(1).replace("\\\n", " ").split()
    return files[: files.index("GateParse.v") + 1]


def _apply(text: str, rule: Rule) -> str:
    count = text.count(rule.old)
    if count != 1:
        raise ValueError(f"{rule.name}: the change's text occurs {count} times")
    return text.replace(rule.old, rule.new)


def validate_rules(rules: Sequence[Rule], library: Path) -> None:
    """Reject stale rule edits before spending time on builds or measurements."""
    for rule in rules:
        _apply((library / rule.file).read_text(), rule)


def variant_sources(group: Group, library: Path) -> dict[str, str]:
    """The files that the group's changes edit, with the changes made in order."""
    edited: dict[str, str] = {}
    for name in group.rules:
        rule = RULE_NAMES[name]
        text = edited[rule.file] if rule.file in edited else (library / rule.file).read_text()
        edited[rule.file] = _apply(text, rule)
    for name in group.subsumed:
        rule = RULE_NAMES[name]
        validate_rules([rule], library)
        if rule.file not in edited or rule.old in edited[rule.file]:
            raise ValueError(f"{group.name}: no change of the group replaces {name}")
    return edited


def slug(group: Group) -> str:
    """The group's name as a directory name."""
    return re.sub(r"[^a-z0-9]+", "-", group.name.lower()).strip("-")


def validate_groups(groups: Sequence[Group], library: Path) -> None:
    """Reject unknown, repeated, or stale changes before any build."""
    names = [g.name for g in groups]
    if BASELINE.name in names or len(set(names)) != len(names):
        raise ValueError("variant names must be distinct and differ from the baseline")
    if len({slug(g) for g in groups}) != len(groups):
        raise ValueError("variant names must give distinct directory names")
    for group in groups:
        members = [*group.rules, *group.subsumed]
        if not members or len(set(members)) != len(members):
            raise ValueError(f"{group.name}: a group has distinct rules")
        unknown = set(members) - set(RULE_NAMES)
        if unknown:
            raise ValueError(f"{group.name}: unknown rules {sorted(unknown)}")
        variant_sources(group, library)


def _build(group: Group, directory: Path, library: Path) -> None:
    """Build an isolated decision-only variant; it is never a runtime proof library."""
    files = _files(library)
    for name in files:
        shutil.copy(library / name, directory / name)
    for name, text in variant_sources(group, library).items():
        (directory / name).write_text(text)
    # The trusted library is untouched; a .vos-only variant cannot serve verify.
    for name, lemma, dependencies in PROOF_USING:
        path = directory / name
        pattern = rf"((?:Lemma|Theorem) {lemma} :.*?\n)Proof\."
        text, count = re.subn(
            pattern, rf"\1Proof using {dependencies}.", path.read_text(), flags=re.S
        )
        if count != 1:
            raise ValueError(f"cannot set ablation proof dependencies for {lemma}")
        path.write_text(text)
    for name in files:
        result = subprocess.run(
            ["coqc", "-vos", "-q", "-Q", ".", "OpGate", name],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode:
            raise RuntimeError(f"{group.name}: {name} failed to compile:\n{result.stderr[-2000:]}")


def _accepts(directory: Path, terms: str, timeout: float) -> Outcome:
    """Return accepted, refused, or timeout; failed queries are experiment errors."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Query.v"
        path.write_text(terms + QUERY)
        try:
            result = subprocess.run(
                ["coqc", "-vos", "-q", "-Q", str(directory), "OpGate", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "timeout"
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    answers = re.findall(r"= (true|false)\s*: bool", result.stdout)
    if len(answers) != 1:
        raise RuntimeError(f"query returned no unique boolean answer: {result.stdout[-500:]}")
    return "accepted" if answers[0] == "true" else "refused"


class Entry(NamedTuple):
    """An entry of a registration record."""

    id: str  # model/task/toolchain
    tasks: str  # the task set
    verdict: str  # safe, or the class of a defect
    accepted: bool  # the gate's decision on the kernel as written
    terms: str  # the exact exported Rocq terms that the gate decided


def same_library(registration: Path, library: Path) -> None:
    """Require the library's sources to be those that the registration retained."""
    root = registration.with_suffix(registration.suffix + ".inputs") / "sources" / "formal"
    recorded = {p.name: p.read_bytes() for p in root.glob("*.v")}
    current = {p.name: p.read_bytes() for p in library.glob("*.v")}
    if recorded != current:
        raise ValueError(f"{library}: the sources differ from those {registration} retained")


def registration_entries(path: Path, library: Path) -> list[Entry]:
    """The entries of a complete registration record of `library`, with their terms."""
    record = json.loads(path.read_text())
    provenance = record["provenance"]
    if (
        record["schema"] != 2
        or record["complete"] is not True
        or provenance["kind"] != REGISTRATION_KIND
        or provenance["query_suffix"] != CHECK_QUERY
    ):
        raise ValueError(f"{path}: a complete registration record of the current query is needed")
    if Path(provenance["library"]).resolve() != library:
        raise ValueError(f"{path}: registered with {provenance['library']}, not {library}")
    same_library(path, library)
    ids = [row["id"] for row in record["results"]]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(provenance["kernels"]):
        raise ValueError(f"{path}: the results are not the registered kernels")
    entries: list[Entry] = []
    for row in sorted(record["results"], key=lambda r: r["id"]):
        if row["query_artifact"] != row["id"] or type(row["accepted"]) is not bool:
            raise ValueError(f"{path}: an invalid result for {row['id']}")
        terms = read_artifact(path, row["query_artifact"], "v")
        entries.append(Entry(row["id"], row["tasks"], row["verdict"], row["accepted"], terms))
    return entries


def select(entries: Sequence[Entry], defects: int, seed: int) -> tuple[list[str], list[str]]:
    """Every safe entry that the gate accepts, and `defects` defects drawn with
    `seed`, from each task set and toolchain in proportion to its defects."""
    safe = sorted(e.id for e in entries if e.verdict == "safe" and e.accepted)
    wrong = [e for e in entries if e.verdict != "safe"]
    if any(e.accepted for e in wrong):
        raise ValueError("the gate accepts a defect")
    if not 0 < defects <= len(wrong):
        raise ValueError(f"a sample of 1 to {len(wrong)} defects is needed, not {defects}")
    strata: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for e in wrong:
        strata[(e.tasks, e.id.split("/")[2])].append(e.id)
    quotas = {k: defects * len(ids) / len(wrong) for k, ids in strata.items()}
    counts = {k: math.floor(q) for k, q in quotas.items()}
    largest = sorted(strata, key=lambda k: (counts[k] - quotas[k], k))
    for k in largest[: defects - sum(counts.values())]:
        counts[k] += 1
    generator = random.Random(seed)
    chosen = [i for k in sorted(strata) for i in generator.sample(sorted(strata[k]), counts[k])]
    return safe, sorted(chosen)


class Row(TypedDict):
    """The decisions of one variant of the checker."""

    id: str  # the group's name, or baseline
    decisions: dict[str, Outcome]  # by entry
    seconds: dict[str, float]  # by entry
    build_seconds: float


def library_files(library: Path) -> dict[str, Path]:
    return {name: library / name for name in _files(library)}


def _built(group: Group, directory: Path, library: Path, output: Path) -> float:
    """Build the variant and retain its sources that differ from the library."""
    started = time.perf_counter()
    _build(group, directory, library)
    seconds = time.perf_counter() - started
    changed = {
        name: directory / name
        for name, source in library_files(library).items()
        if (directory / name).read_bytes() != source.read_bytes()
    }
    root = output.with_suffix(output.suffix + ".inputs") / "variants" / slug(group)
    snapshot_inputs(output, f"variants/{slug(group)}", changed, create=not root.exists())
    return seconds


def _decided(directory: Path, terms: str, timeout: float) -> tuple[Outcome, float]:
    started = time.perf_counter()
    outcome = _accepts(directory, terms, timeout)
    return outcome, time.perf_counter() - started


class Run(NamedTuple):
    """The settings of a run: the library, the time limit, and the workers."""

    library: Path
    timeout: float
    workers: int


def ablate(groups: Sequence[Group], entries: Sequence[Entry], record: Checkpoint, run: Run) -> None:
    """Decide every entry with each group's variant that the record lacks, and
    add each variant's decisions once all are made."""
    pending = [g for g in groups if record.get(g.name) is None]
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(run.workers) as pool:
        directories = {g.name: Path(tmp) / slug(g) for g in pending}
        builds: dict[str, Future[float]] = {}
        decisions: dict[Future[tuple[Outcome, float]], tuple[str, str]] = {}
        made: defaultdict[str, dict[str, tuple[Outcome, float]]] = defaultdict(dict)
        try:
            for group in pending:
                directories[group.name].mkdir()
                builds[group.name] = pool.submit(
                    _built, group, directories[group.name], run.library, record.path
                )
            for group in pending:
                print(f"{group.name}: built in {builds[group.name].result():.0f} s", flush=True)
                for entry in entries:
                    future = pool.submit(
                        _decided, directories[group.name], entry.terms, run.timeout
                    )
                    decisions[future] = (group.name, entry.id)
            for future in as_completed(decisions):
                name, entry = decisions[future]
                made[name][entry] = future.result()
                if len(made[name]) == len(entries):
                    found = made.pop(name)
                    row: Row = {
                        "id": name,
                        "decisions": {e: found[e][0] for e in sorted(found)},
                        "seconds": {e: found[e][1] for e in sorted(found)},
                        "build_seconds": builds[name].result(),
                    }
                    record.add(dict(row))
                    counts = Counter(row["decisions"].values())
                    print(f"{name}: {dict(sorted(counts.items()))}", flush=True)
        except BaseException:
            for future in [*builds.values(), *decisions]:
                future.cancel()
            raise


def _group_record(group: Group) -> dict[str, object]:
    return {
        "name": group.name,
        "description": group.description,
        "rules": list(group.rules),
        "subsumed": list(group.subsumed),
    }


def validate_baseline(row: dict[str, object] | None, safe: list[str], defects: list[str]) -> None:
    """Require the unchanged checker to accept every safe entry and no defect."""
    if row is None or not isinstance(decisions := row["decisions"], dict):
        raise ValueError("the unchanged checker's decisions are missing")
    if any(decisions[s] != "accepted" for s in safe):
        raise RuntimeError("the unchanged checker does not accept every accepted entry")
    if any(decisions[d] == "accepted" for d in defects):
        raise RuntimeError("the unchanged checker accepts a defect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--defects", type=int, default=DEFECTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--group", action="append", default=[], choices=[g.name for g in GROUPS])
    parser.add_argument(
        "--rule", action="append", default=[], help="a rule to turn off alone", choices=RULE_NAMES
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.workers <= 0:
        parser.error("timeout and workers must be positive; timeout must be finite")
    cores = sorted(os.sched_getaffinity(0))
    if args.workers > len(cores):
        parser.error(f"{args.workers} workers need {args.workers} cores, not {len(cores)}")
    groups = [g for g in GROUPS if g.name in args.group] + [
        Group(r.name, r.description, (r.name,)) for r in RULES if r.name in args.rule
    ]
    groups = groups or list(GROUPS)
    library = formal_dir()
    validate_groups(groups, library)
    entries = registration_entries(args.registration, library)
    safe, defects = select(entries, args.defects, args.seed)
    planned = set(safe) | set(defects)
    entries = [e for e in entries if e.id in planned]
    coqc = subprocess.run(
        ["coqc", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    sources = library_files(library)
    provenance = {
        "kind": KIND,
        "registration": str(args.registration),
        "library": str(library),
        "coqc": coqc,
        "query": QUERY,
        "groups": [_group_record(g) for g in groups],
        "safe": safe,
        "defects": defects,
        "defect_seed": args.seed,
        "timeout": args.timeout,
        "workers": args.workers,
        "cores": cores,
    }
    inputs = {
        "rule_ablation.py": Path(__file__),
        "checkpoint.py": Path(__file__).parents[1] / "checkpoint.py",
    }
    registration = {"registration.json": args.registration}
    with Checkpoint(args.output, provenance, resume=args.resume) as record:
        snapshot_inputs(args.output, "sources", inputs, create=not args.resume)
        snapshot_inputs(args.output, "library", sources, create=not args.resume)
        snapshot_inputs(args.output, "registration", registration, create=not args.resume)
        if record.data["complete"]:
            print(f"{args.output}: already complete")
            return
        run = Run(library, args.timeout, args.workers)
        ablate([BASELINE], entries, record, run)
        validate_baseline(record.get(BASELINE.name), safe, defects)
        ablate(groups, entries, record, run)
        snapshot_inputs(args.output, "sources", inputs, create=False)
        snapshot_inputs(args.output, "library", sources, create=False)
        snapshot_inputs(args.output, "registration", registration, create=False)
        record.finish()
    print(args.output)


if __name__ == "__main__":
    main()
