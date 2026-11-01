# Lowering catalog

The checker proves safety of the lowered program under its formal semantics and launch premises. PTX parsing and lowering remain trusted and require instruction-effect fidelity; the model theorem alone does not establish that correspondence. Coverage depends on supported compiler code shapes and proved rules. This document records those shapes and the remaining obligations; the dated measurements in READINESS.md (`READINESS.md`, not included in this archive) distinguish historical results from the current source.

The instruction-effect audit (`../eval/records/effect_boundary_assessment_2026-09-30.json`, not included in this archive)
closes the unrestricted generic-instruction path. Only explicitly classified
register-result families may produce arbitrary registers; supported memory
instructions must retain their checked access obligations. Unknown effects,
asynchronous copies, tensor-map operations, and collectives without checked
prerequisites refuse. Ordinary modeled fences remain supported. The
narrow collective extension (`../eval/records/narrow_collective_qualification_2026-09-30.json`, not included in this archive)
checks explicit PTX 6.0–9.4 and target sm_70+ metadata, exact shuffle/vote
operand shapes, and full warp masks. Register masks generate independent
proof obligations. The matching and integer-reduction extension (`../eval/records/match_redux_qualification_2026-09-30.json`, not included in this archive)
uses these obligations for `match.any/all.sync.b32/b64` and the documented
integer `redux.sync` modes. Matching requires sm_70+ and PTX 6.0+; integer
reduction requires sm_80+ and PTX 7.0+. Both retain the reviewed PTX 9.4 upper
bound and arbitrary outputs; floating reductions are unsupported.
The source-control boundary runs before each reached instruction is rewritten
or inlined. Unpredicated fixed-target `bra.uni` and `call.uni` remain supported;
uniform indexed branches, uniform returns and predicated uniform branches/calls
refuse until their additional warp-uniformity promises are proved. Uncalled
function bodies do not impose obligations on the selected entry.
Fractional cache-policy creation retains lexical floating literals and checks
exact binary32 fractions, priorities, operands, PTX 7.4–9.4 and sm_80+ before
producing an arbitrary 64-bit result. Decimal support requires exact binary32
representability; rounded fractions and other policy forms remain unsupported.
`bar.sync` additionally requires a barrier-only straight-line inlined body ending in
one unconditional return, a proved single-event phase, or a checked ordered
event sequence. The distinct-event phase checker covers all modeled guarded
paths from launch to each exact barrier, proves no return to that event, and
excludes return edges and ambiguous source origins. Mixed CTA/warp bodies
include every supported collective in the ordered event sequence. Their
unpredicated source instructions retain literal full masks or the exact
lowered register-mask checks and result writes. A proved structural case
handles one actual IR block of fallthrough instructions followed by an
unconditional exit. It preserves the same progress and event-order guarantee
without symbolic traversal for each event. The
ranked-loop extension (`../eval/records/ranked_barrier_final_confirmation_2026-09-30.json`, not included in this archive)
checks increasing 64-bit loop traces using positive preserved strides,
no-wrap bounds and inductive upper bounds. A copied increment source must
provably equal the counter; comparisons may precede the increment and either
branch polarity is supported. It proves finite exit and common,
unique event order without enumerating the input-sized iteration count.
For a shared prefix followed by ordered events in one block, a separate
structural rule checks ranked progress to the first event once and proves
fallthrough through every intervening instruction. Any control instruction
in that span excludes the rule; marker and no-return checks remain required.
This avoids repeating the same prefix analysis for every event.
The ordinary memory checker still checks the original program. Other paths
consume a bounded total work budget and refuse if progress is unproved.
For actual IR that revisits an event, the occurrence checker proves a common
finite word of event visits, followed by a closed event-free suffix. It checks
all modeled successors, their arrival states and each event transfer. A
structural eligibility check rejects cycles between events before this bounded
collector runs. This rule permits repeated visits but still requires a common
word across all modeled launched threads; warp-local suffixes require further
scope reasoning. The suffix may diverge without revisiting an event, so this
is not a whole-kernel termination proof; that is the separate check of DESIGN
decision 40.
Historical acceptance counts
below predate these corrections and are not qualification of the current
lowering.

Arbitrary register values cover unspecified *results*. They do not discharge
an instruction's required participation, resource validity, or additional
memory effects. The supported warp-collective implementation assigns arbitrary
outputs; it does not transport bounds. Repeated schedules outside the checked
common-word form, optional event sequences, unsupported loop forms or collective arrangements,
and shared-value transport still need additional checked support.
The communication review (`../eval/records/collective_provenance_feasibility_2026-09-30.json`, not included in this archive)
describes the semantic work needed to use bounds transported between threads.

A rule is added only for a row of this table, and a row is added only when a kernel of a corpus needs it. Section 10.3 of the design (`DESIGN.md`, not included in this archive) records the counts that ordered the rows.

## Constructs and their lowering

| Construct | numba-cuda, level 1 | nvcc, level 1 | Rule | Status |
| --- | --- | --- | --- | --- |
| thread index `cuda.grid(1)`, `blockIdx.x * blockDim.x + threadIdx.x` | `cvt.s64.s32 tid; mul.wide.s32 ntid, ctaid; add.s64` (the block size is the first operand) | `mul.wide.u32 ctaid, ntid; cvt.u64.u32 tid; add.s64`, or `mad.lo.s32` at 32 bits then `cvt.u64.u32` | launch geometry (`%ntid` a constant), `mul.wide` with the constant as either operand, zero extension of a 32-bit index | done |
| bound guard `i < n` | `setp.ge.s64 p, i, n; @p bra` | same | R1 against a size symbol, on the compared register's normal form and on its own atom, so that a loop head keeps the guard of a counter whose form it forgets | done |
| loop whose counter starts at a constant, `for (j = 0; j < k; ++j)` | `mov j, 0`, then the bottom test | rotated: `k < 1` tested against an immediate at the entry, `j < k` at the bottom | R6 keeps the back edge's constraint `j - k <= -1` when the entry edge's own facts (`j = 0`, `k >= 1`) place it within its bounds | done |
| exact integer flags and conditional short loops | `xor.b32` of constants; selected or conditional counter increments | corresponding constant operations and branches | proved exact constant XOR transfer; untrusted discovery proposes disjoint short-loop cases, which the checker validates on all original instructions and edges | implemented; historical Haiku Numba `string_contains` and Opus Numba `rolling_sum` acceptances predate the stricter barrier boundary. The latter now refuses at a barrier. |
| element address, contiguous array | `shl.b64 i, log2 elem; add.s64 base` | same | forms | done |
| element address, Numba array of layout `A` | `shr.s64 s, i, 63; and.b64 s, shape; add.s64 i; mul.lo.s64 stride; add.s64 data`, or `setp.lt.s64 p, i, -1; selp.b64 shape, 0, p; add` | not emitted | `and` with the constant as either operand; `selp` refined by its predicate | done |
| narrow index load `ld.s32` into a 64-bit register | `ld.s32 rd, [addr]` | `ld.global.s32` | the load of its bits and a sign extension | done |
| narrow unsigned load `ld.u32` | `ld.u32` | `ld.global.u32` | the zero-extension range of a narrow register | done |
| grid-stride loop `i += blockDim.x * gridDim.x` | `mul.wide.u32 nctaid, ntid` once; `add.s64 i, i, stride`; pointers advanced by `stride * elem` | same, with `cvta.to.global` once | R5 with a register step; the analysis proposes `pointer = base + c * i` at the loop head from proportional steps | done |
| grid-stride loop, level 3 | unrolled four times, trip count by `div.u64` | same | none: the front ends compile at level 1 | by design |
| Numba `for i in range(start, stop, step)` with a register step | Numba's own lowering, also at level 1, counts down a trip count `t` computed by `div.s64` and never tests `i < stop`, so its bound `start + (t - 1) * step <= stop - 1` is a product of two registers. The front end lowers `range` itself (design 5.9, decision 25): before each iteration it tests `position < stop`, with both negated for a negative step, and it keeps the next position and whether its addition overflowed | not emitted | a copy has its source's constraints and range; a check reads a register's exact form as an equality; a join proposes a side's constraints read through that side's copies, and the check of an invariant reads constraints through copies too; a join reads the next position through the position that the loop test bounds; `abs.s64` of the step | done (decision 25) |
| warp-aligned start `grid(1) - lane` | Numba's `tid % 32` is `tid - (tid & -32)`, and `tid & -32` keeps bits 5 to 31 of the 32-bit `tid` | `tid` and `tid & 31` are two atoms | a range reads a register less its low `n` bits as `2^n (x >> n)`, and a register less its bits from `n` as `x mod 2^n`; a mask to `2^j` above a register's width keeps the bits up to the width (decision 24) | done |
| a guard against a constant on a bit field, `(tid & 31) <= 7` | | `setp.gt.u32` of the bit field | the guard's bound on a bit-field or shifted atom is a constraint of the atom (decision 24) | done |
| a loop over a string nested in a grid-stride loop | the inner count-down loop inside the outer loop; with the front end's lowering, a test of the inner position before each iteration | | the rules of the range loop above: the analysis had no fact for the inner count-down counter at the inner loop head, and the inner position is bounded by its test | done (decision 25) |
| a flag that holds a guard, tested after a warp collective: `pred = (i < n) && x`, `ballot(pred)`, `if (pred) out[pos] = i` | `mov.pred p, 0` on one path, a comparison on the other, joined | same | none: the domain keeps no path condition across a join, so after the join the flag does not imply the guard of the path that set it; that needs a path-sensitive domain. The refusal names the store, and testing the guard again there is accepted: Fable's round-2 revisions of its 4 round-1 kernels are. Triton's probe loop has the same shape, a `found` flag that only rows below `m` set (Opus `hash_probe` after its fix) | by design |
| hash slot `h & (capacity - 1)` and the probe loop `(slot + 1) & mask` | `add.s64 mask, capacity, -1; and.b64` | same | R4 with the constant as either operand; with a symbolic capacity, the size minimum 1 makes the mask exact and R4 adds `slot - capacity <= -1`; register ranges reduced modulo `2^w`; R6 hull of the two ranges at the loop head | done |
| atomics on a table `atom.global.cas`, `atom.global.add` | `atom.global.*` | a call of `__iAtomicCAS` or `__dAtomicAdd`, whose body the module defines; the translation inlines it | read-write span | done |
| shared memory histogram, `local[b]` in shared memory | `atom.shared.add` on a `.shared` symbol | `cvta.shared.u64` of the symbol, then a generic `atom.add` through the inlined call | the windows of shared and local memory: a generic address formed from a `.shared` or `.local` symbol lies in its window (design 5.11); every shared access lies inside a declared shared variable, and a local load inside a local variable (design 5.8, decision 20) | done (7 kernels use shared or local memory) |
| lane and warp arithmetic, `cuda.laneid`, `threadIdx.x >> 5` | `mov.u32 %r, %laneid`; Numba's negative-index term `(i & 2^31) >> 26` | `%laneid` | `%laneid` is its fixed register, `[0, 31]`; `x & k` is 0 when `k` has none of the bits of `x`'s range | done (decision 20) |
| a device function's result, Numba | the caller passes a generic pointer to a slot of its local frame, the callee stores through it (`st.u64 [p], v`), the caller reads it back (`ld.local.u64`) | not emitted: nvcc inlines the function | R8: a cell of the caller's variable carries the value's facts from the callee's store to the caller's load (design 5.11) | done |
| local store `st.local`, or a generic store in the local window | the result slot above | a local array | R8: a store must lie inside a live local variable; outside one it may overwrite a spilled register | done; outside a variable refused by design |
| loop over a string or segment `for j in range(start, end)`, or its `while j < end` form | the `while` form: count-down, `sub.s32 end, start; selp; add.s32 count, -1` while a pointer goes up; at level 1 a loop over the 64-bit copy of a 32-bit `j`. The `range` form, with the front end's lowering: a test of `j < end` before each iteration (decision 25) | `for (j = start; j < end; ++j)`: `add.s32 j, 1` with `j` sign-extended, pointer `add.s64 p, 1` | nvcc: R5 with a sign-extended 32-bit step and a stepped index-input element; Numba: R5 with comparisons and constraints that move with the step, R6 with constraints that follow through copies and normal forms | done (5 kernels) |
| offsets read as `uint32(offsets[i])` into a 64-bit register | `ld.u32 rd, [addr]` of an `int32` element | not emitted | R3: the signed element's range, when it is not negative, bounds the zero-extended register | done |
| length check before a prefix loop, `end - start >= k` then `j < k` | `sub; setp.lt; loop` | same | R7: the check projects `j < k`, `k <= end - start`, and `end <= C` onto `start + j` by Fourier-Motzkin elimination; the guard's `end - start` is a 32-bit form over zero-extended atoms, the address over sign-extended ones, which R7 reads as one atom when their range is below the sign bit | done (4 kernels); substituting `j < k` through `end - start - k >= 0` at the guard was tried and reverted (design 5.3, R1) |
| division and modulo by a power of two, `pos // 32`, `pos % 32` | `shr.s64`, `and.b64 31`; Numba's floor-division idiom with a `selp` on the sign | `shr`, `and` | R4 plus proved shift/remainder bounds and integer coefficient reduction; masked shifts retain their relation to the original bit position | strengthened: 22/24 current Bloom originals accepted and all 22 pass runtime references; Haiku's two Numba signed-remainder/fixup lowerings remain refused |
| `selp`, `min`, `max`, and masks in index arithmetic | `selp` of a register or of two constants; `min`, `max`; `and` with a register mask | same | a `selp` of a register copies it, and one of two constants on a 1-bit predicate is `b + (a - b) * p`; `min` and `max` decided by the operands' ranges; a mask covering the register's width keeps the value (design 5.5) | done |
| scatter by offsets `out[offsets[i]] = x` with the guard `pos < total` | a load from the offsets buffer, then the guard | same | R3: the loaded value gets the declared range of the index input; an offsets buffer's range is `[0, total]`, so the store without the guard is refused (verdict E) | done |
| compaction with an atomic counter `pos = atomicAdd(count, 1); out[pos] = i` | `atom.global.add` sign-extended, `cvt.s64.s32` | `atom.global.add.u32` zero-extended | accepted with a guard that bounds the position on both sides: an unsigned position `pos < n` (`setp.ge.u64` of the zero-extended counter), or the two comparisons of the next row; a one-sided signed guard is refused because the counter's start is the tenant's | by design (design 5.4) |
| a guard of two comparisons, `pos >= 0 && pos < n` | `setp.lt.s64`, `setp.ge.s64`, then `or.pred` and one branch | `setp.lt.s32` of the 32-bit counter, `setp.ge.s64` of its sign extension, then `or.pred` and one branch | `or.pred` is the instruction `IOrp`; its fact `FNeg` holds the negation of each comparison when the predicate fails, so the fall-through edge keeps both; a signed comparison of a copy of a 32-bit register reads that register signed, so both comparisons constrain one atom | done (the repairs of round 1) |
| index output store `sel[pos] = i` | `st.global.u32`; `st.global.u32` of a 64-bit register's low bits | same | the store writes one whole element, and the stored operand, at the element's width and signedness, lies in the declared range; a register wider than the element must lie in `[0, 2^(8w-1))`; an atomic to an index output is refused (design 5.4) | done |
| alignment of every access | | | every global, shared, and local access is aligned to its width, up to 32 bytes: the larger of the address form's alignment and the alignment of the base register and the offset; a buffer's base is a multiple of 256 bytes, the catalog's region alignment; the windows lie at multiples of `2^24` (design 5.8) | done |
| `trap`, `brkpt`, `__assertfail` | Numba `debug=True` only | `assert` | a store to address 0 under the instruction's guard, as for behavior that PTX leaves undefined; the check accepts it only where it proves the guard never holds (design 5.8) | done |
| 32-bit signed index that wraps at `2^31` | `add.s32` then `cvt.s64.s32` | same | refused under a claim of any grid: a real defect on inputs of 2 GiB or more (the unchecked corpus run). Under a claim that bounds the grid to `2^31 / blockDim` blocks (`b_grid`, decision 33), `%ctaid.x` is below the bound and the index does not wrap | provable under a grid claim |
| warp shuffles and ballots feeding an address | `shfl.sync`, `vote.sync` | same | arbitrary outputs after checking PTX 6.0–9.4, target sm_70+, operand widths and full member masks; register masks carry a separate full-mask proof obligation | narrow support implemented; downstream address guards must establish bounds independently |
| Triton: `tl.program_id`, `tl.arange`, masks `offs < n` on predicated loads and stores | | | a predicated access is checked in the state its guard refines; the offset `pid * BLOCK \| lane` is an `or` of disjoint bits, and a thread's later elements add constants whose bits lie below the lane offset's alignment or above its range, `(4 * tid & 508) \| 513` | done: the reference kernel with 64-bit offsets, `tl.program_id(0).to(tl.int64) * BLOCK`, is accepted |
| Triton: 32-bit offsets `tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)` | `mul`, `or` at 32 bits, then `mul.wide.s32` | | refused under a claim of any grid: the offset wraps on a grid of more than `2^31 / BLOCK` programs, a real defect as for the 32-bit signed index. Under a claim of at most `2^31 / BLOCK` programs (`b_grid`, decision 33), the offset stays below `2^31` and the mask bounds it | provable under a grid claim: the DataKernelBench plans' Triton kernels |
| Triton: vector loads and stores `ld.global.v4.b32` | | | a vector access is one access of its whole width; Triton emits it only for a pointer and a size that it knows to be multiples of 16, which the front end does not claim of a size, so the corpus's accesses are scalar | by design |
| Triton: a store `tl.store(p, v, mask=m)` (SB) | | | `@m st.global.b32 [a], { %r }`: the translation reads a vector of one register as that register, so a store to an index output has its stored value | done (decision 24) |
| Triton: a masked load of an index input `tl.load(sel + rows, mask=m, other=0)` (PL) | | | `mov r, 0; @m ld.global.b32 { r }, [a]`: R3 under the guard; d gets the join of its old fact and the input's range, and each constraint of the load that d's old value also meets; each other constraint holds under the mask through the mask's value, so an access under the same mask is bounded | done (decision 24) |
| a `selp` of two registers, `tl.where(active, (slot + 1) & mask, slot)` (WS), or of a register and a constant, `tl.where(active, slot, 0)` | | | a constraint on the register's atoms that also holds with the other operand in its place, a register or a constant, holds with the result in its place | done (decisions 24 and 26) |
| a loop mask updated by `xor.pred`, `pending := pending xor stored`, as in Triton's hash loops | only in Numba's own lowering of `range`, for the sign of a trip count, which the front end replaces (decision 25) | | `IXorp`: the comparisons that both operands' facts list hold when the result holds | done (decision 24) |
| Triton: a mask flag carried through a loop, `ok = valid & (end - begin >= k)`, with a layout change (FL) | | | none: Triton passes `begin`, `end`, and the mask between threads through shared memory before the loop, and the load of `chars[begin + j]` needs the relation of three values another thread stored, which one thread's check does not see | by design (Fable `starts_with`; Sonnet's `starts_with` and `string_length` are defects) |
| Triton: a reduction across lanes that bounds a loop or is a term of an address, `range(0, tl.max(lens), V)` (RD) | | | integer `redux.sync` can produce arbitrary results after checked version, operand and full-mask prerequisites; useful transported bounds additionally require communication semantics | narrow instruction support implemented; transported bounds remain open |
| Triton: a guarded warp-aggregated position `base + rank`, `base` broadcast through shared memory (PG) | | | a signed shared memory load narrower than its register, `ld.shared.s32` into a 64-bit register, is the load of its bits and their sign extension, as a global load is; a shared load's address is a register or a shared variable, such as `[global_smem]`, which had given an arbitrary value (decision 26) | done: the fixed `filter_select` of Fable and Opus and Fable's fixed `hash_probe` are accepted |
| Triton: the alignment of a grid-stride loop register, `base` of `tl.range(pid * 64, g, nprog * 64)` (GA) | | | none: the loop head keeps `base`'s range and drops its alignment, so `base \| (lane \| 32)` is not read as a sum | not needed: the two kernels that have it store at `sums + base + row` with `row` a reduction across lanes (RD), which stays refused by design |
| Triton: the swizzled shared-memory layout of `tl.sort`, an `xor` of thread-index bit fields (SW) | | | `xor.b32` is the operation `BXor`: `x ^ y` lies below the power of two above both operands and keeps their common alignment, and `x \| y` of two ranges keeps it too (decision 24) | done |
| Triton: a mask `(p >= 0) & (p < n)` on a 64-bit position `p = begin.to(tl.int64) + j` from 32-bit values (SE) | | | `p` is the sum of two sign extensions, and a 32-bit sum `j` whose own range is unknown gave its sign extension no range, so the comparison `p < n` related nothing; a sign extension of a register of unknown range lies in `[-2^(w-1), 2^(w-1) - 1]`, and the comparison relates `p` to `n` (decision 26) | done: Fable's round-1 revisions of `hash_probe`, `segmented_sum`, and `string_length` are accepted |
| Triton: a 32-bit hash mask `(capacity - 1).to(tl.int32)` (MT) | | | Triton loads the 64-bit capacity with `ld.param.b32`, its low 32 bits: a load of a size parameter's low bits gets those bits of the value computed from the sizes, and a conversion of a 64-bit register's low 32 bits is the register modulo `2^32`; R4 then relates the 32-bit mask to the capacity (decision 26) | done: Sonnet `groupby_sum` after its fix is accepted |
| Triton: masks packed into an integer, 32 elements per thread (PK) | | | proved residue, exact-copy and zero-test rules recover comparisons for supported packed predicate forms; arbitrary bit-vector relationships are not inferred | implemented subset: scoped historical examples are recorded in `packed_predicate_precision_2026-09-30.json` and `packed_predicate_refusal_groups_2026-09-30.json`; these do not establish acceptance of every packed-mask kernel under the current collective checks |
| held-out: a binary search, `mid = (lo + hi) >> 1`, `(lo + hi) / 2`, or a base-and-length form `pos + half`, with `lo` and `hi` updated by `selp` (BS) | Numba's floor division by 2 | `shr` of the sum, or `(x + (x >>> 63)) >> 1` for a signed `/ 2` | proved quotient/remainder constraints for nonnegative 64-bit halving, scaled inequality projection, fixed-parameter bounds, and constraints from either select operand or its copied source preserve midpoint/cursor bounds | strengthened: six additional original searches accepted across the versioned source replays; signed/unsigned, copied-operand and select-order regression variants pass. Remaining search lowerings remain open; the 12 cases are historical baseline limits |
| held-out: dynamic shared memory in Numba or CUDA C++, `cuda.shared.array(0)`, `extern __shared__`, with the bytes that the kernel's own launcher passes (DS) | `.extern .shared` | same | the frontends carry an affine shared allocation through the binding, formal private-memory facts and checked launch effect; Sonnet pattern staging declares `k` bytes with an enforced `k <= 49152` size bound | closed for these allocations: Haiku min/max, Sonnet CUDA and Numba rolling sum, and both Sonnet pattern arrays accepted; the two pattern bindings explicitly change from zero shared bytes to `k` bytes with a 48 KiB limit |
| held-out: a k-ary search whose lanes probe `lo + t * stride` with a register stride (KS) | | | none: a product of two registers | open: 2 |
| held-out: bounds through `min` and `max` of sizes and block offsets, `h = max(0, t0 - window + 1)` (MM) | `max.s64`, `min.s64` | same | proved transfer rules retain the result's order relative to both operands; constant interval facts are available to shallow range checks | partially implemented: signed/unsigned clamp tests pass; the complete rolling kernels still need loop invariants |
| held-out: a sequential merge whose cursors satisfy `i + j = k` (MC) | | | none: an equality of three registers across the loop | open: 2 |
| held-out: a shared work queue, a count zeroed before a barrier and incremented at most once by each thread (SQ) | `atom.shared.add` | same | none: shared memory holds any value; the count's bound needs a model of barriers | open: 2 |
| a local array written at a bounded register index (LA) | `st.local` at a register offset | same | R8 proves the entire store span inside one live local variable and invalidates every tracked cell that may overlap; exact-offset stores retain the precise cell update | implemented local-store support; the historical Gemini Q12 CUDA acceptance predates the collective checks. Its unchanged acceptance assertion currently fails on barrier progress. Local-store controls do not discharge that separate obligation |
| held-out: binary lifting, `j = where(cond, j + step, j)` with a halving step (BL) | | | WS keeps a constraint only when both operands meet it; `j >= 0` needs `step >= 1` on one side | open: 1 |
| held-out: Numba's count-down `range(K - 1, -1, -1)`, lowered as `j = -c` with an up-counter (CD) | `mov j, K - 1`, the counter from `1 - K` | | none: the invariant `j = 1 - c` holds on entry only after the check substitutes `c`'s constant | open: 1 |
| held-out: a bitmap of `(n + 7) / 8` bytes (QS) | | | `DerivedSize` can explicitly relate the separate byte/word size to rows through a checked floor quotient, including ceiling forms; adding this relation changes the contract | the historical affine-only limitation is lifted at the contract layer. Null-mask host integration and existing corpus acceptance require separate evidence; archived contracts remain unchanged |
| held-out: a Triton store with a cache-policy operand, `st.global.L1::evict_last.L2::cache_hint.b64 [a], {v}, policy` (CP) | | | the address and width are read; the stored value is not, so a store to an index output fails its value check | open: no safe kernel needs it |

## What the table does

- **Order of work.** A row's count of corpus kernels decides its place; the counts are in design section 10.3 and in `eval/corpus/synthesized/classification.md`.
- **The paper's coverage claim.** The table, with its counts, is the coverage figure of RQ2: which constructs of synthesized operators the gate proves, and which it refuses by design.
- **Preserve the held-out baseline.** Rows marked held-out describe the original unchanged-gate evaluation. Once a revised rule uses these tasks, they are development evidence for that revision; generalization requires new held-out tasks (design decision 28).
- **No rule without a row.** A rule that no row needs is not added, so the checker and its proofs stay as small as the corpora require.
