# Architecture

This document is the code design of the `opgate` package: what each package owns, what it may import, the types that cross package boundaries, and the path of one query through them. The design (`DESIGN.md`, not included in this archive) says what the system does and why; this document says where each part lives. When a type or a boundary here changes, change this document in the same commit. The terms are defined in the [glossary](GLOSSARY.md).

## Principles

- The core names no engine. `signature`, `binding`, `catalog`, and `gate` import only the standard library, so they run and are tested without a GPU; the gate also needs `coqc`. `executor` needs the CUDA driver, which it calls through `cuda.bindings`. Each front end imports its toolchain, and each host adapter imports its engine, in a module of its own. Nothing outside `opgate/hosts` imports cuDF, pylibcudf, RMM, Sirius, or Spark.
- The gate's `verify` and the catalog's `check_step` check proposals: PTX from front ends, bindings from `derive_binding`, planner steps, suballocator regions and argument-buffer selections. Proposals that fail these checks are refused. Correct PTX parsing/lowering, authenticated session selection, complete host mediation and faithful effect execution remain trusted; the two checks do not verify those implementations.
- Decisions are values: `verify` returns a `Decision`, `check_step` returns accepted with the next catalog and the effects, or refused with a reason. The executor turns a refusal into the exception `RefusedError` at its boundary. No function substitutes a default for a refusal.
- Records are frozen dataclasses; identifiers are `NewType`s (`TenantId`, `BufferId`, `OperatorId`); closed sets are enums or unions of dataclasses, matched exhaustively with `assert_never` in the last case.
- A check reads the catalog and does not change it. The changes of an accepted step are committed in one call, so a refused step leaves no trace, as `check_step` in Rocq is a function from a state and a step to a decision and a next state.

## Packages

| Package | Owns | Imports | Trusted |
| --- | --- | --- | --- |
| `signature` | `Signature`, its argument kinds, `SizeSymbol`, `DerivedSize`, `Affine`, `Direction`: the author's claim | nothing | no |
| `binding` | `Binding`, `Parameter` (`Base`, `Size`, `Constant`, `Bounded`, `Zero`), `Span`, `IndexRange`, `ElementType`, `Convention`; `derive_binding` | `signature` | no |
| `catalog` | `Catalog`, `Buffer`, `Status`, `Region`, `Grant`, the actions, `Step`, the effects; `check_step`, `check_launch`; `rocq`, the Rocq printer and replay of steps | `signature`, `binding`, `gate` (`rocq` only) | yes: the transcription of `formal/CatalogStep.v` |
| `gate` | `verify`, `Decision`, `export_terms`, the `coqc` runner, the in-process record of decisions; `zeroed` (`prologue`), the writer of the zeroing prologue that the gate requires | `binding` | yes: the export only; the parser, translation, and checker are in `formal/`, and `zeroed` is not trusted, since the gate checks the text it writes |
| `executor` | `Executor`, `Module`, `Graph`, `RefusedError`, the checked driver and NVRTC calls; graph capture and replay, fills, the trusted check kernels (`trusted`), the allocation wrapper, the tenant context | `binding`, `catalog`, `gate` | yes: loading the verified text, marshaling checked launches, clearing, fills, the check kernels |
| `frontends` | `Operator`, `Target`; `numba_operator`, `nvcc_operator`, `triton_operator`, one module each; `guard`, the untrusted pass that checks unproved accesses | `signature`, `binding`, `gate` (`guard` only, the decisions) | no |
| `hosts` | `CatalogResource` and `BufferView` (`rapids`, the RMM allocation wrapper and trusted operators on the pool); `install_cudf` (`cudf`, the `apply` hook and query-cache release); `Service` and `Client` (`service`, a package: the wire protocol, principals and opaque session handles with per-tenant quotas, the owner thread, registration records, and the plan-schema and query-node interfaces that a deployment implements); `TorchHost` (`torch`, PyTorch tensors in catalog segments); `register`, the registration that guards unproved accesses, and `execute` and `launch`, which give a guarded module its status buffer and raise `CheckFailedError` on a failed check; one module per engine | `signature`, `binding`, `catalog`, `gate`, `frontends`, `executor` | yes: the service boundary and engine interception are trusted; bindings, regions, and steps are still checked below them |

`tests/architecture/` checks these import rules and that only `executor/driver.py` imports the driver and NVRTC and each front end's module imports its toolchain.

## Types that cross package boundaries

`binding`: the symbolic statement the gate verifies against and the catalog checks launches against. It holds no data size. The binding is one record in three places: the Rocq record `binding` in `formal/GateFront.v` is the source; the Python record has the same fields in the same order; and `export_terms` prints the Python record field by field. `tests/gate` checks that a printed binding type-checks in Rocq. When the Rocq record changes, the Python record and the printer change in the same commit.

`DerivedSize(target, numerator, divisor)` states an exact floor quotient between
size symbols. For sizes `(n, blocks)`,
`DerivedSize(1, Affine(1023, ((0, 1),)), 1024)` states
`blocks = (n + 1023) // 1024`. The divisor is positive and numerator symbols
precede the target, so relations can be evaluated in declaration order.
`OperatorSchema` accepts only independent sizes from clients and computes the
dependent values using a plan prepared at registration. The catalog independently
checks every quotient on every launch, including launches submitted directly by
trusted host code. `StoreLink` proves that these checked relations imply the
constraints used by the kernel checker. Merely naming two sizes does not relate
them; adding a relation is an explicit contract change.

Raw binding records require `derived_sizes`, including an empty list when there
are no relations; exported Rocq bindings likewise require `b_derived`. Older wire
records fail schema validation. Regenerate active bindings from their authored
signatures when upgrading. Historical experiment records retain their original
bytes and schema; any migrated query must be recorded separately.

```python
@dataclass(frozen=True)
class Affine:  # constant + sum of coefficient * size symbol; Rocq: affine
    constant: int
    terms: tuple[tuple[int, int], ...]  # (index into Binding.sizes, coefficient)


@dataclass(frozen=True)
class ElementType:
    width: Literal[1, 2, 4, 8]
    signed: bool


@dataclass(frozen=True)
class IndexRange:  # Rocq: index
    element: ElementType
    lo: Affine
    hi: Affine


@dataclass(frozen=True)
class Span:  # Rocq: span
    direction: Direction  # read, write, read_write; an atomic needs read_write
    extent: Affine  # in bytes
    align: int  # the base is a multiple of this power of two; the pool guarantees it
    index: IndexRange | None  # an index input (read) or an index output (write)


# What one PTX parameter holds; Rocq: param.
Parameter = (
    Base | Size | Constant | Bounded | Zero
)  # Base carries a Span, Size an Affine of the sizes


@dataclass(frozen=True)
class SizeSymbol:  # from the signature; Rocq: one entry of b_sizes, its minimum
    name: str
    minimum: int = 0
    maximum: int = 2**31 - 1


@dataclass(frozen=True)
class Binding:
    kernel: str  # the entry name in the PTX text
    sizes: tuple[SizeSymbol, ...]  # size register k has this symbol's declared bounds
    parameters: tuple[tuple[str, Parameter], ...]  # each PTX parameter by name, in order
    block: tuple[int, int, int]
    grid: tuple[int, int, int]  # maximum blocks in each launch dimension
    shared: Affine  # dynamic shared bytes as a function of the launch sizes
```

The buffer arguments of a launch are the `Base` parameters in order. A Numba array of one dimension is seven parameters: `Zero`, `Zero`, `Size(k)`, `Constant(elem)`, `Base(span)`, `Size(k)`, `Constant(elem)`. A size symbol whose value is not a row count of the plan, such as the character count that a string column's last offset holds, is still a symbol: the catalog reads its value from the buffer's status at the launch, and the offsets buffer is an index input with the range `[0, S]`.

`frontends`: an `Operator` is the PTX text, the entry name, the signature, the toolchain's `Convention`, the block size, the grid maxima (the largest grid of any launch, the GPU's limit unless the claim states less), and the dynamic shared memory of every launch. The convention, which `binding` defines because `derive_binding` reads it, says how the toolchain lays out one signature argument as PTX parameters (Numba: seven parameters for a one-dimensional array; nvcc: one pointer; Triton: one pointer, and two zero scratch pointers after the arguments). `Operator.binding(element_types)` calls `derive_binding` with the signature, the element types of the plan's columns, the convention, the entry's PTX parameter names, the block size, and the grid maxima.

```python
@dataclass(frozen=True)
class Operator:
    text: str  # the PTX module
    kernel: str  # the entry kernel's name in the text
    signature: Signature
    convention: Convention
    block: tuple[int, int, int]
    grid: tuple[int, int, int]
    shared: Affine  # bytes of dynamic shared memory
```

`gate`: `Decision = Accepted | Refused`, where `Refused` carries the reason and, when identifiable, the kernel and PTX instruction for the failed check, and the first accesses that the walk of the certificate refuses, at most `LISTED` = 32, each with its reason and instruction (`unproved`; `text_diagnosis` in `GateParse.v` gives the reason and the list from one walk of each case of the certificate, which stops after the last listed access, and the check query prints both). Participation failures and timeouts may lack a specific instruction. `verify(text, bindings)` returns one decision for the module, accepted only if every entry kernel is, and records it in the process, keyed by the selected compiled-library identity, exact PTX text, and printed binding terms. The exporter requires exact built-in integer and Boolean primitives before cache lookup or query construction. Reuse compares the actual text and binding terms before packing PTX into Rocq integers; dataclass equality alone is insufficient because it can equate distinct printed numeric terms. The library is an immutable build snapshot, published only after the complete theorem chain compiles; each verification resolves and pins that snapshot. Cached decisions are invalidated when its compiled artifacts change, and an in-place change during a check fails immediately.

Acceptance means that Rocq proved `verify_certificate_sound` for the actual text and bindings: in every launch that establishes the binding's facts, every thread of every entry kernel meets the access, alignment, index-output, and trap obligations (`kernel_safe`) and ends (`kernel_ends`). The lowering (`front_lower` in `GateFront.v`) also requires that each entry kernel begins with exactly the zeroing prologue of its shared and local memory that `prologue` gives (`zeroing_check`) and that every declared register, and the carry flag, that an instruction may read is written on every path from the start to it (`defined_check`, on the kernel with its device functions inlined), so that a kernel cannot read data that an earlier kernel left in shared memory, local memory, or registers (decision 46); for shared memory, and for local memory when the kernel declares no local variables, the launch model proves it (`GatePrologue.v`, `launch_ni`, decision 50), and the certificate check requires the recognized prologue (`prologue_ok`). `opgate.gate.zeroed(text, binding)` writes the prologue: zero stores of every shared variable of the module by all threads of the block in rounds, `bar.sync 0`, and zero stores of every local variable of the entry kernel by each thread, after the body's leading declarations, and zero stores of the local variables of each device function that declares any, at the start of its body (`function_prologue`); it leaves a text without shared or local variables as it is.

One `coqc` process materializes the front end's result of each entry kernel and an untrusted invariant certificate, then proves that `check_text_certificate_ok` accepts the certificate for the actual text and bindings. The checker independently parses and translates the text, checks launch facts and the binding link, requires exact entry coverage and a valid entry case, and checks that every outgoing reachable state is covered by a successor case. Unreachable blocks may have empty case lists. Rocq checks an opaque auxiliary proof before the outer proof closes; printed output never establishes acceptance. On proof failure, diagnosis reads the same certificate without rerunning proposal. The query computes `text_front front_program` outside the kernel: the program and launch facts of each entry kernel without the participation check. The kernel checks that this result equals `text_front front`, which runs the participation check once, before the proposal; a kernel that the check refuses is refused there and diagnosed with the proposal from `front`, which gives it no certificate (`ScalarMissing`). The proposal (`propose_front`) and the check of the certificate (`check_front_ok`, which `check_text_certificate_ok` is on `text_front front`, `check_text_front`) read the checked result, so the front end runs once in the kernel and once outside it. Proposal, checking, and diagnosis share one wall-clock deadline. A check that does not finish in `timeout` seconds, 180 seconds by default, is refused with that reason and not recorded, since nothing is proved. `export_terms` exports only the inputs; `export_query` appends the exact production certificate and proof commands, also retained by registration measurements.

Inlining checks source uniform-control promises before erasing call and return
qualifiers. Unresolved uniform indexed branches, uniform returns and predicated
uniform branches/calls refuse; ordinary returns and unpredicated fixed-target
branches/calls remain supported. Only reached function bodies are expanded.
Lowering classifies instruction effects explicitly. Unknown effects cannot
inherit arbitrary-register handling. Supported vector local loads retain
their complete footprint and alignment, even though their result registers
are arbitrary. Narrow synchronous shuffle, vote, matching and integer-reduction
support checks the applicable PTX and target
metadata, operand shapes and full member masks; masks held in registers become
independent proof obligations. Aligned barriers require a barrier-only
straight-line body with a single final return, a checked single-event phase,
or a checked event sequence. Mixed CTA/warp bodies check every
supported waiting site in one common order. The shared-value check of
`GateShared.v` finds the waiting sites and collectives of a position in maps
keyed by its block and index (`pc_table`) rather than by comparing it with
every site; `phase_verify_at_same` and `dep_verify_at_same` prove the walks
with the maps equal to the walks with the lists, and its proposals use the
same maps. Source instructions must be
unpredicated and have unambiguous origins. Warp collectives are waiting
sites of the same check in a kernel without a CTA barrier too, so that every
lane that a member mask names reaches each collective; a predicated collective
is refused for participation. Register masks retain their complete
lowered precondition checks, including any comparison snapshot, and the
actual translator register map identifies the source operand. The original
collective result writes remain in the program.
Source-origin lookup builds ordered source and instruction lists once, indexed
by the existing injective origin key. Universal equality proofs preserve the
previous complete hit lists, including duplicates, so uniqueness checks retain
the same meaning. Common source conditions are checked once; each origin
still requires its own uniqueness check.
Phase checking proves finite progress from launch to each event and excludes
return edges. For one actual IR block ending in an unconditional exit, a
structural proof establishes progress from concrete fallthrough steps. Other
shapes use the bounded work queue, which checks every possible guarded
successor; budget exhaustion refuses. Dependency propagation stops at an exact
fixed point, with a proof that it returns the same retained facts as all the
original passes. For multiple events, a proved
loop rule can replace enumeration of a straight-line loop with a checked
64-bit increasing counter, a preserved positive stride, no wrap, and
an inductive invariant with a checked upper bound. The increment may read a
proved alias of the counter; the comparison can precede the increment and
the branch can use either predicate polarity. Straight-line instructions around the
increment must preserve the counter and stride. Every actual trace instruction and edge is checked;
the exit state comes from its transfer. Distinct-event proofs establish the
same event order and a unique visit for every modeled launched thread.
Adjacent ordering comparisons are proved equivalent to the previous all-pairs
check. Cheap structural prerequisites precede symbolic exploration, with
universal equality proofs preserving the original checker decisions.
Closure checks traverse actual instructions directly; control dependencies
use the exact instruction prefix before each event. Both replace indexed-list
scans with universally equal computations. When a non-straight program's events
all lie in one block with no intervening control flow, a structural rule proves
ranked progress to the first event once, then proves concrete fallthrough to
every later event. It retains all event markers, strict order and no-return
checks and the existing safety predicate. The older one-block fast path remains
selected for its original program shapes.
When actual IR can revisit an event, a separate occurrence checker proves one
common finite word of event visits for every modeled launched thread, followed
by a closed event-free suffix. It checks arrival states and the actual transfer
through every event. A preliminary graph walk rejects event-free cycles that
this collector cannot summarize. The proof form is selected from the actual
program before checking; a failed check is not retried through another route.
This common-word rule does not yet express a warp-local suffix reached by only
some warps after the final CTA barrier.
For a block whose x extent is a multiple of 32, a per-warp rule checks one
complete word for each warp, with one projection onto the CTA barriers. When
the compound loop descriptors leave an event-free loop unsupported, the
per-warp words walk a loop without a descriptor, or one whose step is a
`selp` of 1 or 0, one iteration at a time, and the abstract states must decide
each back edge. When a kernel without CTA barriers has an entry block that
ends with a conditional branch to blocks without a waiting site, the warp exit
rule of `GateContext.v` is selected instead: a walk of the entry block proves
that the branch's predicate is a function of the CTA's launch values and of
the warp index, and each way out of the branch needs a complete word for each
warp.
A CTA-uniform counted loop around barriers and warp collectives may test its
counter with `!=` against the limit when it steps by 1, and other instructions
may lie between the counter's update and its test. `GateEnds.v` adds two
routes to the fixed check. For a program whose segments between events hold
an event-free cycle, an event-free loop whose ways out all lead to one
position is summarized by the ranges of the registers that it does not write;
this route's guarantee holds for a kernel that ends (`kernel_ends`), which
the certificate check proves. For a program whose events lie outside loops
and whose segments are acyclic, one occurrence word replaces the separate
progress proof of each event.
The proofs use exactly the program and facts returned by the frontend.
These rules do not establish shared-value provenance. When they fail, the
shared-value route of `GateShared.v` proves participation from values that
the threads of a block share (decision 41). There, a branch whose predicate
may differ between threads leads into a set of blocks without waiting sites
or exits that control leaves only to one join block; the blocks may hold
loops, so this route's guarantee holds for a kernel that ends.
Whole-kernel termination
is the separate check of `GateTerm.v`. Its loops are back edges, a branch at
the end of block `l` to a block `h <= l`; they must be nested or disjoint and
start after block 0, a head may have several back edges, and a thread may
enter a loop past its head. Each
loop needs back edges that the certificate's states show are never taken, or a
checked descriptor: a counter written once in its head or latch block, whose step the certificate's states bound away from 0 and whose interval
at the head is inductive, a counter incremented once by 1 that the loop leaves
when it equals a constant, or the two bounds of a binary search, with the new
bounds evaluated as functions of the head's lower bound and distance along
every path through the loop. `check_entry_certificate` requires
`term_check` after the safety certificate, and `term_check_sound` gives
`kernel_ends`. A loop without a proved descriptor is refused at its back edge's
branch with the reason "the kernel is not proved to end".
Floating literals retain their original spelling separately from integers.
Fractional cache-policy creation checks exact binary32 representability and
the permitted fraction range before defining an opaque result; it does not
relax the checks on any later memory access.

Access checking tries the complete obligation under the ordinary guard before
computing conditional-load provenance. Both paths check permissions, bounds,
alignment and indexed-output values. A separate exact-equivalence lemma skips
selection constraint transport when both alternatives are constants, because
that computation cannot change its input state. A backward dependency closure
also identifies pure selection results outside address, guard and local-value
uses, when no writable indexed contents are declared. Read-only indexed inputs
retain their ordinary access and value semantics. It combines every definition's
dependencies, including division, and keeps multiply defined selection
destinations. Ordinary payload calculations and unused modeled results can
then omit selection precision. Transfer may forget those result facts; its
proof holds for any hint map. All original instructions,
memory accesses and indexed-value checks remain present. Projection retains the
query's connected non-fixed atoms and every constraint containing only fixed
atoms. This sound subset selection reduces unrelated elimination work; it can
lose bounds that a disconnected component imposes on a fixed launch parameter.
Register assignment traverses the existing fact-map trie once, pruning empty
branches; a lookup-equivalence theorem relates it to the former insertion fold.
The memoized range layers keep the cached registers' ranges in maps
(`pair_table`) and stop at the first layer whose cached ranges equal those of
the layer below (`rf_stable_layers`). Each range query caches the layers on
the registers that its ranges reach (`reach_regs`), whose layers read only
these registers and registers whose ranges read no layer below;
`rf_on_agree` proves the cached layers equal to the plain layers of the same
depth on these registers, and `form_range_exact` and `project_range_exact`
prove each query's result equal to that of the plain layers. A projection
reads its ranges through a cache on the registers of its goal and of the
constraints near it (`projection_regs`); `project_range_with_plain` proves it
equal to the projection of the uncached ranges. `terms_eqb` compares two term
lists by their structure, `plain_sum` and `cons_range` multiply a coefficient
by a bound with the coefficient first, since `Pos.mul` recurses on its first
argument, and `rep` and `treduce` read 2^m for the common widths from a table
(`pow2`, `pow2_eq`). The analysis allows a
block's entry state `LIMIT` changes for each edge into the block
(`revisit_limits`).
These optimizations preserve
the independent certificate checks; scoped results and remaining regressions
are recorded in READINESS.md (`READINESS.md`, not included in this archive).

`executor`: `Executor.register(text, binding)` puts the zeroing prologue at the start of the entry kernel (`zeroed`), has the gate verify that text, and returns a `Module`, the accepted text, which the prologue begins, and binding with the driver's handles of the loaded module and its entry kernel and the byte size of each parameter as the driver lays out the loaded function (`cuFuncGetParamInfo`), or raises `RefusedError`. Besides the gate's refusal, it refuses a binding that does not name the entry kernel's parameters in the order the text declares them (the gate finds a parameter by its name, the executor marshals it by its position), a text the driver does not load, a parameter size that cannot hold what the binding says (8 bytes for a buffer's base, 1, 2, 4, or 8 bytes that hold a constant or bounded scalar), and a block or dynamic shared memory the device cannot launch the loaded function with. Loading submits no work, so a refusal unloads the text and leaves the executor usable.

`catalog`: the public state model below corresponds to the Rocq state. The
Python implementation indexes live buffers and operators in maps and applies a
step through a transaction with an undo log.

```python
Status = Raw | Checked  # Checked(element, lo, hi), concrete integers


@dataclass(frozen=True)
class Buffer:
    id: BufferId
    offset: int  # byte offset in the pool
    nbytes: int
    owner: TenantId
    status: Status


@dataclass(frozen=True)
class Catalog:
    pool: int  # the bytes of the pool
    buffers: tuple[Buffer, ...]
    grants: tuple[Grant, ...]  # Grant(buffer, tenant)
    freed: tuple[Region, ...]  # each freed region and its last owner
    operators: tuple[tuple[OperatorId, Binding], ...]  # each verified operator's binding


@dataclass(frozen=True)
class Launch:
    operator: OperatorId  # its binding is the catalog's
    grid: tuple[int, int, int]
    buffers: tuple[BufferId | View, ...]  # one per buffer argument; View(buffer, offset)
    sizes: tuple[int, ...]  # one per size symbol
    scalars: tuple[int, ...]  # one per Bounded parameter
    guarded: bool  # an unchecked index input is checked when the launch runs


Action = Allocate | Free | Grant | Write | Fill | Launch


@dataclass(frozen=True)
class Step:
    tenant: TenantId
    actions: tuple[Action, ...]  # checked in order
```

`check_step(catalog, step)` returns `StepAccepted(next, effects)`, the next catalog and what the executor does (`Clear`, `Upload`, `Memset`, `Kernel`), or `StepRefused(reason)`, leaving the supplied catalog unchanged. The executor uses the equivalent `Catalog.begin` transaction directly: a refused step rolls back, and an accepted step supplies checked effects for submission before commit. `check_launch` is its clause for one launch; a `Kernel` effect holds, for each PTX parameter, a pool offset (`At`) or a value (`Value`), and the `Check`s of the index inputs that a guarded launch checks when it runs.

Installing a verified binding prepares its immutable argument layout once:
parameter positions, read/write directions, dynamic size and scalar positions,
and fixed `Value` arguments. Every launch checks the current grid, sizes,
shared-memory extent, scalar bounds, live buffers, allocation extents,
permissions, duplicate writable arguments, and index provenance. It then fills
the layout with that launch's checked offsets and values. Freed or reallocated
buffers and changed grants are therefore visible to the next launch; the
prepared layout retains no pointers or previous launch decisions.

The catalog cost record (`../eval/records/runtime_catalog_validation_2026-09-30.json`, not included in this archive)
measures the complete Q6 fill-plus-launch catalog transaction on one CPU core:
the median falls from 46.57 to 24.35 microseconds across three alternating
process pairs. Preparing and installing the binding metadata once rises from
0.78 to 29.54 microseconds; these measurements exclude verification and CUDA
execution. Separate Q6 service measurements at scale factor 0.1, with the owner
on core 8 and connections and clients on cores 9–15, give median paired
throughput ratios of 0.9995 and 1.0315 at one and eight tenants. The record
retains all 16 cells and 194,692 checked outputs; the first repeat is marked
diagnostic because CPU regression tests overlapped it, and an additional repeat
supplies three comparison pairs per tenant count. Substantial shared-host
variation limits the service conclusion to these observations.

The earlier combined candidate record (`../eval/records/runtime_hotpath_2026-09-30.json`, not included in this archive)
also changed eager 64-bit fills to a repeated 32-bit memset where the bytes
matched. That combined candidate had lower paired throughput, with median
ratios 0.875 and 0.914 at one and eight tenants. The fill substitution was
removed before the catalog-only measurements. Its raw results, including the
matched separate-service baseline, describe the discarded candidate rather
than the retained implementation. The executor continues to use its trusted
kernel for 8-byte fills.

`executor`: `Executor(device, pool_bytes)` owns the pool and the catalog's state. `register(text, binding)` gives a `Module` and records the operator; `execute(step)` checks and performs a step and returns whether each kernel ran; `capture(step)` performs a step as a CUDA graph, `replay(graph, step)` launches it again with the effects of a checked step of the same shape, `ran(graph)` says whether each kernel of its last launch ran, which `replay` requires of a graph with checks before it launches the graph again, and `release(graph)` destroys it; `tenant(t)` sets the session's tenant; `reserve(nbytes)` gives the session's tenant its reservation, a region of the pool that the executor's suballocator over the pool proposes, and `unreserve()` ends it once the tenant holds no live buffer; `allocate(nbytes)` gives a buffer whose id is the tenant and the tenant's own serial number, at a region of the reservation that the tenant's own first-fit suballocator proposes, and `free(buffer, *more)` frees buffers of the tenant in one step; `read(tenant, buffer)` copies out a buffer the tenant may read; `keep_raw(buffer, holder)` keeps a buffer raw while trusted host code writes it outside the catalog through `holder`. `executor/trusted.py` holds the trusted kernels that check an index input, set a graph's condition, and fill 8-byte elements; `executor/driver.py` compiles them with NVRTC.

## The path of one query

1. Trusted session code sets the tenant of the query in the executor's tenant context. Host adapters read it; they do not set it.
2. Trusted operators run and allocate. Each allocation goes through the allocation wrapper: the untrusted suballocator returns a region, and the wrapper submits a step with that one allocation, which `check_step` accepts only if the buffer id is the tenant's and the region lies in the tenant's reservation at a multiple of 256 bytes and overlaps no live buffer, and which clears the region unless the tenant is its last owner. Host data enters by a write action; with an element type, the buffer becomes checked with the range of the data. An output of a trusted engine operator remains raw; its index consumers use guarded launches. There is no trusted output-range declaration API.
3. For each untrusted operator, a front end produces an `Operator`, `Operator.binding` a `Binding`, and `Executor.register` calls `verify` unless the decision is already recorded, then loads the accepted text and records the operator's binding and prepared argument layout in the catalog. A refused operator raises `RefusedError`; the caller receives the refusal. There is no automatic alternative execution path.
4. The planner proposes a step: writes, fills of index outputs, launches with their buffers, sizes, and scalars, frees, and grants, in order.
5. `Executor.execute(step)` first requires the step tenant to equal the executor session tenant, begins the catalog transaction, performs the checked effects in order, and commits after successful submission: clears and narrow fills as memsets, 8-byte fills through the trusted fill kernel, uploads, and each launch as `cuLaunchKernel` with parameters built from the checked record at the sizes the driver gives for the loaded function, which `register` checked against the binding: the pool's base plus the offset of each buffer argument, the sizes, the constants, the scalars, and zero. No pointer comes from anywhere else. A kernel with checks runs only if the trusted check kernel finds every element of each checked index input in its range; the executor waits for the result.
6. In graph mode, `Executor.capture(step)` performs the step as one CUDA graph: clears and fills are memset nodes (8-byte fills are nodes of the trusted fill kernel), launches are kernel nodes, and a kernel with checks runs in the body of a conditional IF node whose condition the check kernels set, so no step waits on the device. `Executor.replay(graph, step)` has `check_step` check the next step, whose effects must have the graph's shape, and updates every node from its effects, the function too, so every replay is a checked step. A launch overwrites the flags of the graph's checks, so a graph with checks replays after `ran` has read its last launch.

The executor is confined to its creating thread; a host must serialize its
requests there. Its tenant context belongs to that executor, and `read` checks
it as well as the catalog's grants. A submission, graph update, or
synchronization error permanently fails the executor, because a prefix of a
step may already have executed. Every operation that commits to the catalog
first waits for the uses of prepared steps in progress. Effects outside a use
run on the legacy stream, and the constructor refuses a per-thread default
stream; the tenants' streams are blocking streams, which `prepare` checks when
it creates one (design decision 48). Further work fails immediately; `close` still
attempts every resource release and reports cleanup errors. A refused
registration or free submits no work and does not fail the executor. RMM releases a
region under the owner recorded at allocation, commits the free only after
the pool releases it, and removes its pointer mapping last. This permits
ordinary object destruction after a session ends without borrowing another
tenant's identity.

## What a host adapter provides

A host adapter is the only engine-specific code. For an engine to host OpGate it provides:

- the mapping from the engine's column objects to catalog buffers, which the allocation wrapper makes possible because every buffer was registered when it was allocated;
- an entry point where the engine hands over an untrusted kernel, such as cuDF's `ApplyKernelBase._compile_or_get_kernel`, replaced by a proxy that routes the kernel's PTX to a front end and its launch to `Executor.launch`;
- the engine's allocator, wrapped (for the RAPIDS engines, one RMM memory resource).

With the correct authenticated tenant and complete launch mediation, `check_step` rejects an argument buffer that the tenant lacks permission to use. This contains incorrect buffer selections. It does not establish isolation against arbitrary adapter bugs: session selection, interception, pointer correspondence and effect execution remain trusted.

For the RAPIDS engines, `opgate.hosts.rapids` has the three parts. `CatalogResource` is the RMM allocation wrapper. The engine allocates in its tenant's arena: an RMM pool whose upstream regions, its chunks, are catalog buffers of the tenant, and inside `resource.current()` RMM's current resource is the session tenant's arena, which the executor's session observer selects (`Executor.observe_sessions`), so an allocation takes no Python call and no catalog step. The catalog takes a step when an arena grows by a chunk and when `release(tenant)` returns the chunks of an arena that holds no allocation, in one step; a released arena serves no new allocation and returns at a later release or free once its last allocation is freed. `argument_at(pointer)` names an engine allocation as a launch argument, the view of its chunk from the allocation (decision 35). `allocate` and `free` give trusted host code a buffer of its own, and inside `dedicated()` every engine allocation is one: the unit that a session names, grants, and passes with an index range, whose free is a step; `buffer_at(pointer)` names it. The service exposes the release as an explicit authenticated request and releases a tenant's arena when its session closes. Byte and buffer quotas count chunks and buffers. Allocation fails when capacity is exhausted, without automatic eviction and retry. `BufferView.of(resource, buffer, typestr, count, access=...)` shows a buffer that `resource.allocate` gave to the engine through the CUDA array interface, so that a trusted pylibcudf operator runs on it. The view holds the buffer's RMM allocation, so the region stays allocated while the engine holds the view. A `read` view needs the session's tenant to be able to read the buffer, is read-only, and leaves the buffer's status, so a checked index range survives it; a `write` view needs the tenant to be able to write the buffer and first makes the buffer raw (a write of no data without an element type), since the engine may write any value through it, and the buffer stays raw while the view lives: the executor refuses a step or a use that would free it or record a checked range for it (`Executor.keep_raw`). The operator's output is raw, and a launch that takes it as an index input is a guarded launch. `opgate.hosts.cudf.install_cudf(executor, resource, target)` is the entry point for cuDF: within the block, `Series.apply` and `DataFrame.apply` compile cuDF's kernel with the Numba front end at level 1, with a signature from the frame (each input column a read span of `size + offset` rows, so a slice needs no sub-buffer), verify and load it through the executor, and launch it as a step of the session's tenant; CuPy allocates through RMM, so cuDF's outputs are pool buffers. A column with a null mask, a string column, a UDF that uses cuDF's string runtime, and a UDF aggregation (`Rolling.apply`, a function in `GroupBy.agg`), whose PTX libcudf compiles into a kernel of its own, are refused. The hook imports Numba and CuPy, which cuDF's UDF path uses, and reads the tenant that trusted session code set (`Executor.session_tenant`).

For PyTorch, `opgate.hosts.torch.TorchHost` has the same parts. Each tenant has a `torch.cuda.MemPool` whose allocator is the executor's (`torch._C._cuda_customAllocator` with ctypes callbacks): each segment that PyTorch's caching allocator takes is a buffer of the tenant, placed by the executor's suballocator and cleared unless the tenant last owned it, and inside `session(tenant)` PyTorch places the tenant's tensors in its segments without a catalog step. `argument_at(tensor)` names a tensor as the view of its segment, `buffer(count, dtype)` gives a buffer of its own as a tensor for a span with an index range, `launch` launches a verified operator on tensors through `Executor.launch`, and `release(tenant)` ends the tenant's pool, whose segments PyTorch gives back. The executor launches on CUDA's legacy stream, PyTorch's default stream, so a launch is ordered with PyTorch's work without a synchronization. `eval/tpch/torch_engine.py` is a tensor engine for TPC-H Q1, Q6, and Q14 whose row-wise operators are PyTorch's own, the corpus's Triton kernels that its `MODELS` names for each task (Fable's `q6_sum`, `projection`, and `gather`, and Opus's `scatter_add`, which sums Q1's groups) launched directly, or the same kernels through the gate and the executor.

`opgate.frontends.guard` checks, when they run, the accesses that the gate does not prove (decision 38): an untrusted pass rewrites each such access to run when its offset from its buffer's or shared variable's base, in whole elements, lies inside the region, and otherwise to store 1 to a status buffer; it appends a parameter for each guarded buffer's extent, which the binding declares as a size, then the status buffer, which the binding declares as a 4-byte buffer that the kernel writes, and has the gate check the result with its zeroing prologue, which `Executor.register` puts there when it loads the result. A loop that the gate does not prove ends gets an iteration limit (decision 43): a counter that the loop's head increments and, at the limit, a store of 1 to the status buffer and a branch out of the loop; the limit is the sum of the buffers' extents, which an appended parameter `opgate_limit` declared as a size holds, plus the grid's thread count. A checked load or atomic without a predicate runs into a register of the pass that holds zero, which the destination copies after it, and a register that the gate names as read before a write, as on a limit's path out of the loop, gets zero at the kernel's start (decision 46), in the same round as every register that the pass's analysis finds may be read before every path writes it. Each of the first `BATCH` rounds guards every access that the gate's refusal lists (decision 51). `opgate.hosts.register(executor, text, binding)` registers the text as written or guarded and returns a `Registered`; the cuDF hook and the PyTorch engine use it, and the service applies the same pass on its verification threads. `opgate.hosts.execute(executor, allocator, step)` and `opgate.hosts.launch(executor, allocator, launch)` give each launch of a guarded module a zeroed status buffer of the session's tenant and raise `CheckFailedError` naming the kernel when a check failed or a loop reached its limit; the service answers it to the tenant as `check_failed`.

`Executor.launch(launch)` decides one launch as `execute` decides the step of the launch alone. A launch without index checks goes through a prepared step that the executor keeps for the tenant and the launch without its scalars (at most `KEPT_LAUNCHES`): while the records of the buffers it names are those of its last full check, the catalog checks only the scalars, and the driver call built for that check takes them (decision 36).

`opgate.hosts.service.Service` is the deployed multi-tenant boundary, used by
the service-boundary and consolidation experiments. A Unix-socket connection
authenticates once, within a time limit for its complete frame; the server maps the credential, and
for a principal with a `uid` the peer's user, to a `TenantId`, and the
connection is one session. The JSON protocol carries opaque handles, host
bytes, scalar values, PTX, and the fields of a plan schema or a query node.
It has no device address, Python callable, pickle, caller-written binding,
caller-selected tenant, or unchecked launch. The operator and query interfaces
are deployment-defined; the shipped service backend requires RAPIDS, initializes
an RMM resource, and installs the cuDF hook. Integrating a different engine
requires its allocation and launch interception as well as query nodes.
The deployment passes the closed set of plan schemas (how a registered PTX's
binding is derived and how `run_operator` fields become a launch) and query
nodes (trusted engine operations, such as a cuDF query over a session's
buffers). `OperatorSchema` supplies a reusable declarative plan schema: the
deployment fixes a `Signature`, buffer element types, calling convention,
block, maximum grid, and shared-memory extent. Registration supplies only
PTX and `{"kind": "the_deployed_kind"}`. The schema derives its binding in
the C, Numba, or Triton ABI and requires exactly one entry kernel; the gate
must still prove that particular module. It is independent of corpus row
names and does not make unsupported PTX acceptable.

Its `run_operator` request supplies named `buffers`, `sizes`, and `scalars`
objects, a three-integer `grid`, and a required boolean `guarded`. Buffer
values are opaque handles. Sizes and bounded scalar values are checked against
the deployment's contract; `SizeOf` parameters are derived from sizes.
Scalars represent parameter bits. A guarded launch scans raw index inputs
through the existing catalog path; an unguarded launch requires their catalog
provenance. Both enforce current allocation bounds and authority. Custom
plans also return the required `LaunchRequest.guarded` field explicitly.

One owner thread holds the executor, the RMM resource, and the cuDF
hook, and performs every request in the server-selected tenant's context. A
`register` request names one or more kernels, each with its PTX and plan
(design decision 53). The owner counts each kernel against its tenant's
quotas, and the server submits its check to a pool of
`Limits.verification_workers` threads, a required setting that the
deployment derives from the machine's cores; each check has its own
deadline from the moment a thread takes it, and the owner loads or refuses
the kernels in the request's order. The checks run outside the owner thread,
so one tenant's check leaves the others' queries running, and the kernels of
one request are checked at once, so a query plan waits for its slowest
kernel while the pool has a free thread for each. The response gives each
kernel its own answer: an operator handle, or the refusal of its count, its
check, or its load. A principal has at most `Limits.registrations_in_flight`
kernels counted and not yet answered, over its sessions, and a request
beyond that is refused as a whole. A counted kernel holds a place among its
tenant's operator handles until the owner loads or refuses it, so kernels
checked at once stay within the operator quota. Per-tenant quotas bound
the pool bytes and regions at the RMM resource, including engine temporaries
and retained regions, as well as operator handles and registrations. Reusing
a retained region does not charge it twice. Exhaustion fails immediately;
the authenticated `release` request or session cleanup returns retained regions.
`release` affects only its principal's retained regions and leaves live handles
intact, allowing an existing session to reclaim quota before changing buffer sizes.
A read must fit in a response. Every outcome is decided by the type of
the exception that ends it: a malformed request is the client's input error, a
refusal of the catalog, the gate, the driver's load, or a quota is a refusal,
and anything else fails the service, which stops accepting connections for a
supervisor to restart it. A query node runs inside a query boundary: cuDF's
instance-keyed UDF caches and device-scalar cache are released afterwards, and
if the live-buffer set differs from before, the service fails. Disconnect work
follows all queued work and frees the session's buffers before their pool
regions can be reused. Registration records keep the submitted PTX text and
the derived binding for an administrator to compare.
Shutdown cancels running verification, stops accepting requests, and lets
connection handlers finish session cleanup before placing the owner's stop
message on its queue.

A session's repeated launch runs on the data plane (design decision 34). Its
`prepare` request names an operator handle, a grid, buffer handles, sizes,
initial scalars, fills, and the buffer to read back; the owner has the
catalog check the step once (`Executor.prepare`) and returns a small handle.
Its `data_plane` request makes the session's mailbox (`dataplane`): a sealed
memfd and two eventfds, passed over the session's authenticated socket after
the JSON response. The client writes a record of the handle and the launch's
scalars and rings; the owner copies the record, has the catalog decide the
use (`Executor.submit`), and answers with the read-back bytes when the use
ends. The owner's loop is compiled with Numba (`fast`): it serves the uses
whose steps the executor describes as hot (`Executor.hot`) on their tenants'
streams, and returns to Python, with no use in progress, for queued control
work and for any request that the catalog must decide in full. The compiled
loop is part of the trusted service boundary.

`Limits` requires positive integer quotas and positive finite time budgets.
The defaults are 5 seconds for the whole authentication frame, 300 seconds
for each whole authenticated request frame, and 30 seconds to send a whole
response. Request reception includes idle time between requests; verification
and query execution occur outside that receive deadline. A receive or send
timeout closes the connection and schedules its session cleanup. Fragmented
delivery does not renew the receive deadline. `Client` serializes requests on
its session and closes a stream after an incomplete or malformed response;
`Client.register` sends a query's kernels in one request.

Each registration audit row records the submitted PTX, derived binding,
decision, reason, module-load count, and successful launch count. The
administrator's wire request is `{"op":"audit","start":0,"count":1}`;
its response contains `registrations`, `total`, and `next` alongside `ok`.
An oversized requested page is refused before socket emission. `Client.audit`
reads explicit pages, one record per page by default, retaining all raw record
values. It collects the records present at the first page; launch counters
are observed at each page's retrieval time. It neither truncates nor retries
an oversized page. The existing experiment schemas `elementwise_f64`,
`read_one_i64`, and `tpch_q6_sum` remain concrete deployments alongside the
reusable `OperatorSchema` API. An unexpected driver submission
or synchronization failure poisons the executor, resolves pending requests as
fatal, and makes later requests fail immediately. The service does not retry a
query in that context; a process supervisor supplies restart in a deployment.

## Engineering method

The first part of the gate port (milestones 2 and 3 of design section 11) found rules by running the checker on one kernel, reading the failing access, and adding a rule. That found the rules, but it does not scale, and it is not how the paper can describe the work. From here on:

- **Soundness and precision are kept apart.** Soundness is the Rocq proofs: a rule enters the checker with its lemma, and the theorem chain never regresses. Precision is measured, not proved: `eval/corpus/synthesized/decisions.tsv` preserves the historical baseline. Corpus tests require its accepted kernels to remain accepted and dedicated regression tests require new capabilities. Revised measurements live in versioned JSON records with exact provenance; they do not rewrite the held-out baseline. A defect row that flips to accepted fails the test.
- **Rules come from the lowering catalog.** [LOWERING.md](LOWERING.md) lists each source construct, the PTX each toolchain emits for it at level 1, the rule, and its status. A rule is added for a row, and a row is added when a corpus kernel needs it. This is the SFI validator's discipline: sound for all input, precise for the compiler's output.
- **The product driver is the only harness.** `opgate.gate` exports and evaluates; the corpus runner, the diagnosis of a refusal, and the decision table use it. No script outside the tree is needed to reproduce a number in the paper.
- **A front end is a thin, pinned module with a canary.** Each front end is one function from source to `Operator`, its toolchain version is pinned in `pyproject.toml`, and a test compiles one reference grid-stride kernel and asserts the shape the catalog expects (no `div.u64`, no `call`). numba-cuda's public API offers only NVVM levels 3 and 0, so the Numba front end calls the same internal steps as `numba.cuda.compiler.compile` with `nvvm_options={"opt": 1}`; the canary is what detects a change in that API. The Numba front end also installs its own lowering of `range` iteration in numba-cuda's target (`RANGES`, design decision 25), written with llvmlite, the IR library that Numba lowers with; a canary compiles the reference kernel as a `range` loop, and `tests/frontends/test_numba_range.py` checks on the GPU that the lowering yields Python's values.
- **The trusted base is countable.** The Python trusted base includes `gate` (the export and the `coqc` runner), `catalog`, `executor`, and the host allocation, interception, and authenticated service boundary. The formal specification and its parser/translation also need review; the theorem chain checks the stated model. `tests/architecture` enforces the import table above, and the paper's implementation table counts these components separately. Conformance tests do not prove the Python implementation or its correspondence with device memory.

## Issues met so far and what fixes each

| Issue | Fix |
| --- | --- |
| The corpus harness lived outside the tree, held constant sizes, and read lengths with regular expressions | `opgate.gate` and the decision table in the tree; sizes are symbols, and a signature's lengths are affine in them |
| Two Numba compilers produced different PTX (the built-in target keeps calls and a stack frame; numba-cuda inlines) | one front end, numba-cuda 0.30.4, pinned; it reproduces the corpus PTX (done, milestone 6) |
| NVVM level 1 needs numba-cuda's internal compile steps | `opgate.frontends.numba_cuda` isolates the call; the canary test detects a change (done, milestone 6) |
| Triton appends two scratch pointers to a kernel's parameters | `Convention.TRITON` states them as zero, and the front end refuses a kernel that needs scratch memory (done, milestone 6) |
| The type checker reads numba's own `numba.cuda`, which numba-cuda replaces at import, and `cuda.bindings` has no type stubs | the two imports carry an ignore comment with the code and the reason; the canary and executor tests exercise them |
| `pyproject.toml` excluded all of `eval/corpus` from lint and type checks, so the corpus adapter was not checked | the exclusion names only `eval/corpus/synthesized`, the model-written files; four recorded experiment scripts, whose records cite their line numbers, are excluded from formatting only |
| The Rocq binding record and the Python types had drifted | one record, converged with size symbols; the round-trip test |
| Constants in forms are unreduced (`capacity - 1` is `2^64 + 2047`), so masks got no range | register ranges reduced modulo `2^w` (done) |
| NVVM strength-reduces loop pointers even at level 1 | the analysis proposes `pointer = base + c * index` from proportional steps (done) |
| cuDF 26.08 depends on numba-cuda-mlir | `apply` still compiles through NVVM; the adapter notes the backend it hooks, and the canary covers it |
| The precision work had no fixed order | the lowering catalog and the decision table |
| Recursive range queries repeat expansion of shared register dependencies; comparison snapshots exposed severe costs during loop recovery | each range query caches its layers on the registers that its ranges reach (`reach_regs`); the cache preserves the range calculation (`form_range_exact`, `project_range_exact`). Depth-1 join queries and bounded Fourier-Motzkin projection alone do not prevent this cost |
| A local store was not checked, though it can overwrite a register that `ptxas` spilled, and a callee's local variables had no register | a local store must lie in a live local variable, and each inlined call has its own variables and frame (R8, design section 5.11) |
| Numba returns a device function's result through a slot in the caller's local frame, so the result had no fact | cells of local variables carry a stored value's facts to its load (R8) |
| Rocq's kernel unfolds the translation of an unknown kernel at every inlining depth when it compares types, so `verify_ptx_sound` did not finish | `Strategy opaque [front]` before the theorem (design section 13) |
| A refusal says only that some access is not proved, which neither the repair loop nor a developer can act on | diagnosis names the failed check and, when identifiable, the PTX instruction with its operands; `GateFront.v` retains instruction origins. Participation failures and timeouts need not identify one instruction |
| The corpus adapter reads a model's free-text `signature.json`: a capacity is a power of two when its description says so, and an input is an offsets buffer when its name contains "offset" | the reviewed per-kernel tables `eval/corpus/synthesized/<model>/roles.tsv` of roles and minimums, which the adapter reads and checks against each kernel's arguments (done, milestone 5) |
| Only the gate existed: the front ends, executor, catalog, and host were docstrings, and the evidence supports gate-level prevention, not tenant isolation | the front ends and the executor's register path (done, milestone 6); the catalog, the launch path, and the RMM allocation wrapper (done, milestone 7); the theorem chain, guarded launches, and the graph path (done, milestone 8); the cuDF hook and the TPC-H pipelines (done, milestone 9) |
| The gate accepted a shared access, or a local load, anywhere in its 16 MiB window, but an access past the memory a block or thread has faults with `ILLEGAL_ADDRESS`, which ends the context for every tenant | every shared access lies inside a declared shared variable, and every local load inside a local variable; the launch fact that no buffer lies in a window; two precision rules keep the corpus decisions (done, decision 20) |
| The architecture promised `tests/architecture`, which did not exist | the import rules and the document links are tests (done, milestone 6) |

## Tests

The table describes implemented test coverage. It does not mean every test was
rerun against the latest revision or every currently accepted expanded-corpus
entry has a functional replay. Readiness (`READINESS.md`, not included in this archive) links the completed
measurements and their actual source snapshots. Corpus launch tests select the
entries marked accepted in the historical decision table; current acceptance
comes from the separately selected complete registration record.

| Directory | What it establishes | GPU | State |
| --- | --- | --- | --- |
| `tests/binding` | derivation on each convention, on constructed cases | no | done |
| `tests/formal`, `tests/executor` | the step check on generated and constructed steps | no, and yes | done |
| `tests/gate` | accepted kernels and refused variants; the decision table of the frozen corpus; a printed binding type-checks in Rocq | no | done |
| `tests/gate` | the zeroing prologue: DeepSeek's CUDA C++ Q6 kernel without the store that initializes its shared array is refused as written and accepted with the prologue; a prologue without one of its instructions, of another block size, entered by a label, or with its registers declared twice is refused, as is a device function without the stores of its local variables; a register read before it is written is refused; a loop clears dynamic shared memory whose bytes depend on the sizes, and a prologue that clears fewer dynamic bytes than the launch gives (a smaller count, the count of another size, a wider stride, a later first element, an earlier exit) is refused, as is a branch into the loop | no | done |
| `tests/gpu` | a kernel that reads a shared array that it never writes reads what the previous kernel of the context left there, and zeros with the prologue; DeepSeek's Q6 kernel gives NumPy's revenue with its prologue, and its negative control gives 0 after another tenant's kernel filled shared memory; a reader of dynamic shared memory reads the pattern as written and zeros with the loop | yes | done |
| `tests/gate` | the parser against `ptxas`: every corpus module that `ptxas` assembles parses, with the entry kernels its text declares | no | done |
| `tests/gate` | a grid-stride loop and a binary search end and are accepted; a loop whose counter does not move, a search whose lower bound can stay, and a shuffle that one lane branches around are refused | no | done |
| `tests/frontends` | each toolchain's canary kernel has the expected shape; the corpus from source through the front ends and the executor decides as the decision table | the corpus test: yes | done |
| `tests/frontends` | a loop that the gate does not prove ends is accepted with an iteration limit and gives its task's results; a probe of a full hash table reports a failed check; a limit that does not end its loop is refused | the runs: yes | done |
| `tests/formal` | Rocq evaluates `check_step` on generated steps and agrees with the Python transcription, on every refusal reason | no | done |
| `tests/formal` | `verify_certificate_sound` assumes only the primitive integers and arrays, and `tenant_isolation` and `term_check_sound` nothing | no | done |
| `tests/executor` | registration verifies and loads once, and a refusal raises; launches, refused launches, index inputs, clearing; historical decision-table acceptances on their task inputs | yes | done |
| `tests/executor` | guarded launches, eager and in graphs; graph capture, replays through the step check, and the shape of a replay | yes | done |
| `tests/hosts` | a cuDF query allocates only through the RMM allocation wrapper; `apply` UDFs run through the gate and the executor, and one with a nullable column is refused | yes | done |
| `tests/pipelines` | TPC-H Q6, the revenue pipeline on the static and the guarded path, eagerly and as graphs, and a join through pylibcudf give NumPy's results | yes | done |
| `tests/gpu` | the window facts that the gate assumes, on each GPU | yes | done |
| `tests/gpu` | the instruction semantics of the translation against the GPU: 47 probes of the integer, conversion, comparison, and predicate instructions, `abs` and `xor` among them, each on 296 operand pairs | yes | done |
| `tests/architecture` | the import rules above; links in the documents | no | done |
