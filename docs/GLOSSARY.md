# Glossary and naming

OpGate has its own vocabulary. Every concept has one name, used in the documents, the Python identifiers, the Rocq identifiers, and the paper.

## Terms

| Term | Definition |
| --- | --- |
| tenant | a principal the engine separates: a user under row-level security, a team, or a customer of a shared service |
| pool | the one device allocation per worker from which every buffer is carved |
| buffer | an allocation from the pool with a byte extent, an owner tenant, and a status |
| last owner | the tenant that owned a free region of the pool before it was freed; a region is cleared before it goes to another tenant |
| status | `raw`, or `checked` with an element type and a range: the catalog's knowledge of the values a buffer holds |
| range | an interval `[lo, hi]` of integer values; concrete in a status, affine in size symbols in a binding |
| column | a data buffer, a validity bitmask buffer when nullable, and for strings and lists an offsets buffer over a data buffer; cuDF's layout |
| catalog | the trusted state: tenants, tables, columns, buffers, owners, last owners, grants, statuses; and the check of each step, including each allocation the engine's allocator makes |
| allocation wrapper | the executor's wrapper around the engine's device allocator: each region the untrusted suballocator returns becomes a one-allocation step for the catalog |
| grant | a catalog entry that makes a buffer readable by a tenant other than its owner; a granted buffer is read-only for every tenant, and a grant declassifies its contents |
| step | one proposal from the planner: allocations, frees, grants, and launches, checked by `check_step` before any launch runs |
| milestone | one item of the project plan in design section 11, done when its test passes; never called a step, which is the planner's |
| launch | one run of an operator's kernel inside a step: the operator, its binding, the grid, and the arguments |
| guarded launch | a launch whose unchecked index inputs a validation kernel checks when it runs; the kernel runs only if every check holds |
| repair loop | the evaluation's return of a refused kernel to the model that wrote it, with the gate's reason, for a revision that the gate then decides |
| size register | the checker's register for a size symbol, fixed at launch with the range `[minimum, maximum]`, with `maximum <= Smax`; a parameter that holds the symbol reads it |
| window | the range of generic addresses that the hardware maps to shared or to local memory; its base is a fixed register of the checker |
| frame | the local memory of one call of a device function, named by the calls from the kernel to it; each inlined call has its own frame and its own copy of the callee's local variables |
| live variable | a local variable of the frame of an instruction or of a frame that called it; a local store must lie inside one |
| cell | a register of the checker that holds 4 or 8 bytes of a local variable, so that a value stored there keeps its facts until it is loaded back |
| bottom state | the abstract state of a block that no run reaches; no register values satisfy it, so its accesses need no proof |
| lowering catalog | the table of source constructs, the PTX each toolchain emits for them, and the checker rule for each; the order of the precision work |
| decision table | the expected decision of every corpus kernel, kept in the tree and asserted by a test; the measure of precision |
| operator | a plan node whose kernel is untrusted: the PTX text, a signature, and the plan position |
| signature | the author's claim about each argument: its role and what it refers to in the plan |
| role | what an argument is: span, index input, index output, size, or scalar |
| span | a role: a data buffer with an extent and a direction, read, write, or read-write; atomics only on read-write |
| index input | a role: a buffer whose elements the catalog has established to lie in a range; each loaded element carries the range |
| index output | a role: a buffer the launch fills with indices by plain stores, each value proven in the declared range |
| size | a role: an integer parameter equal to a size symbol |
| scalar | a role: an integer parameter with declared bounds |
| size symbol | a symbolic count, in `[minimum, maximum]` for bounds the signature states (defaults 0 and `2^31 - 1`), in which a binding states extents, ranges, and bounds; its value at a launch is usually a buffer's row count |
| binding | the symbolic statement of an operator's launches, derived from the signature by untrusted code: what each PTX parameter holds (a buffer base, a size, a constant, a bounded integer, or zero), each buffer argument's role, element type, extent, and range in size symbols, and the block size; what the gate verifies against and the launch check checks each launch against |
| arguments | the concrete values of one launch: a buffer for each buffer argument, a value for each size and scalar |
| launch check | the clause of `check_step` that checks one launch's arguments against its binding and the catalog; it covers every launch and every graph parameter update |
| fill | the executor's write of a value in range into an index output before its producer runs, as a memset node in the graph |
| gate | the component that verifies a kernel: parses the PTX text, translates it, and runs the checker in Rocq |
| verify | what the gate does to a kernel and a binding; the result is a decision |
| decision | `accepted`, or `refused` with a reason that names the access and the failed obligation |
| checker | the abstract interpreter in Rocq: `check_thread` for one thread, `check_block` for a thread block |
| provenance | the checker's proof of where an address or a stored value comes from |
| store provenance | the rule that every value stored to an index output carries a proven range fact |
| fault freedom | the property that an accepted operator raises no memory fault and executes no trap |
| executor | the only execute path: step check, verification, module loading, marshaling of checked launches, graph capture |
| validation kernel | the trusted kernel that scans an intermediate and sets the condition a consumer runs under |
| intermediate | a buffer one launch produces and a later launch consumes |
| front end (toolchain) | the code that obtains PTX text and the calling convention from Numba, nvcc, or Triton |
| host | an engine the executor is installed into: cuDF, later Sirius |
| counterexample | a small accepted kernel that leaks or faults when one catalog rule is dropped |
| tenant isolation | the end-to-end theorem: noninterference over a tenant's buffers and outputs |

Words that no longer name anything: sentinel page, zero page, discard page, write sink, publisher, program row, dependent row, block table, owner tag, bind check. A numerical sentinel value, such as the signed maximum used to initialize a masked load, retains its standard meaning.

## Identifiers

Python, in package `opgate`:

| Identifier | Module |
| --- | --- |
| `Signature`, `SizeSymbol`, `Span`, `IndexIn`, `Offsets`, `IndexOut`, `Scalar`, `SizeOf`, `Between`, `Affine`, `Direction` | `opgate.signature` |
| `Binding`, `Parameter`, `Base`, `Size`, `Constant`, `Bounded`, `Zero`, `Span`, `IndexRange`, `ElementType`, `Convention`, `derive_binding` | `opgate.binding` |
| `Catalog`, `Buffer`, `Status`, `Raw`, `Checked`, `Region`, `Grant`, `Allocate`, `Free`, `Write`, `Fill`, `Launch`, `Step`, `Clear`, `Upload`, `Memset`, `Kernel`, `TenantId`, `BufferId`, `OperatorId`, `check_step`, `check_launch` | `opgate.catalog` |
| `Decision`, `Accepted`, `Refused`, `verify`, `export_terms` | `opgate.gate` |
| `Executor`, `register`, `execute`, `capture`, `replay`, `Module`, `Graph`, `RefusedError` | `opgate.executor` |
| `Trusted`, `load`, the validation kernels `opgate_check` and `opgate_decide`, and `opgate_fill8` | `opgate.executor.trusted` |
| `Operator`, `Target`, `numba_operator`, `nvcc_operator`, `triton_operator` | `opgate.frontends` |
| `CatalogResource`, `install_cudf` | `opgate.hosts` |

Rocq, library `OpGate` in `formal/`:

| Identifier | File |
| --- | --- |
| `verify_text`, `verify_ptx` | `GateParse.v`, `GateFront.v` |
| `check_thread`, `check_block` | `GateDomain.v` |
| `check_thread_sound`, `check_block_sound`, `kernel_safe` | `GateThread.v` |
| `verify_ptx_sound`, `verify_text_sound` | `GateFront.v`, `GateParse.v` |
| `check_step`, `check_launch`, `check_step_inv`, `check_launch_sound` | `CatalogStep.v` |
| `tenant_isolation` | `PipelineIsolation.v` |

## Rules

- One term per concept, the same in prose, Python, Rocq, and LaTeX.
- `verify` and `verification` are the words for what the gate does. Do not use `admission`, which in a database paper means workload admission control.
- `accepted` and `refused` are the two outcomes. Do not use `pass` or `fail`, which collide with compiler passes and test results.
- Write the words out: `index input`, not `idx`; `binding`, not `bind` or `ctr`.
