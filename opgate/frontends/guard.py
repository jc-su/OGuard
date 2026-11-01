"""Bounds checks for the accesses that the gate does not prove.

The gate proves most of a kernel's accesses inside their buffers and shared
variables, and one that it cannot prove refuses the kernel. `guard` rewrites
such an access into one that checks its own bounds when it runs. Of the
access's offset from the base of its buffer or shared variable,
`d = address - base`, it keeps the whole elements of the access's width,
`t = (d >> s) << s` for a width of `2^s`; the access runs only if `t` lies in
`[0, extent - width]`, at the address `base + t`. A shared variable's extent
is its declared size; a buffer's is the binding's at the launch's sizes, as
is the dynamic shared memory's, and a parameter that the pass appends to
the kernel holds it: the binding gets the parameter as a size, which the
executor computes from the launch's sizes. The gate then checks the
rewritten text and binding as any other, with the zeroing prologue that
`Executor.register` puts at its start (`opgate.gate.zeroed`), so this pass
is untrusted: a wrong check is refused, or skips an access, and never lets
one leave its region. An access whose check fails does not run: a store or
an atomic writes nothing to memory, and a load or an atomic that the kernel
runs unconditionally writes zero to its registers, which would otherwise
hold what the hardware left there, a value that the gate refuses to read
(design decision 46); one under a predicate leaves them as they were. In
place of the access, the kernel stores the number 1 to the first word of a
status buffer (`STATUS`), the last parameter, which the pass appends after
the extents when it adds a check and which the binding declares as a flag
buffer of `STATUS_BYTES` that the kernel writes: the gate proves that every
store to a flag buffer stores 1 into one whole word, so a word that the host
zeroes is 0 or 1 after the launch, and 1 if any such store ran. A text that
itself declares the parameter `STATUS` is refused (`declares_status`). A host
gives each launch of such a text a status buffer of zeros and
raises an error after the launch if a check failed (`opgate.hosts.launch`),
so a defect's failed check is reported to the host. A kernel that is safe
makes aligned accesses inside their regions, which pass their checks at the
addresses they name, so its results do not change and its status stays
zero.

The region of an access is the one buffer parameter of the binding, or the
one shared variable, whose base its address is plus a value that no base
enters, read from the registers that compute it without regard to order; a
shared address that no such sum gives, as a swizzled one, has as its region
the one shared variable that it derives from, if it derives from no buffer.
An access whose region is none of these is not guarded, nor is a local
access. If the gate refuses the text for barrier participation, `guard`
first rewrites the block sums, loops, branches, and binary searches that
block the proof, as `opgate.frontends.participation` describes. If the gate
refuses the text, `guard` then inlines each device function that the kernel
calls and the module defines, so that an access in a callee is guarded as
the kernel's own, and renames apart the registers
that scopes declare by name, as nvcc's `{ .reg .b64 %tmp; ... }`, so that a
register's name has one meaning. A generic access whose address
`cvta.shared` makes from a shared variable is checked against that
variable, at its generic base. It then repeats: the gate names the
first access that it does not prove in bounds or aligned and lists the
first accesses that its walk of the certificate refuses, at most `LISTED`
in `GateParse.v` (`Refused.unproved`), the pass guards every access of the
same instruction and address as each one listed, or after `BATCH` rounds
every access that it can, and the gate
checks the text again, until it accepts, refuses for another reason, or
names an access that the pass does not guard.

The gate also proves that every loop ends, and refuses a loop that it does
not prove, such as a compare-and-swap that a thread retries until it
succeeds, which ends because another thread's swap succeeded, a fact about
other threads that the gate's model does not have. Once the barrier rewrite
has been tried, the pass gives such a loop an iteration limit (`_limit`): a
counter that the loop's head increments, and when it reaches the limit, the
kernel stores 1 to the status buffer and leaves the loop. The limit is the
sum of the buffers' extents in bytes at the launch's sizes, which a
parameter that the pass appends holds (`LIMIT`), plus the grid's thread
count. A safe kernel's loops stop before it: a compare-and-swap is retried
only after another call's swap changed the word, and each call changes it
at most once, so a call retries fewer times than there are calls, one for
each row or each thread; a hash table's probe visits each slot at most
once, or twice when a failed swap retries the slot, and a slot takes at
least 2 bytes; a search or a scan over a length that the kernel reads runs
at most once for each element of the buffer it searches or scans, plus one;
and a loop of a fixed count, 2 to 32 in the corpus, runs fewer times than
the grid has threads. The gate checks the limited loop as any other: the
counter rule of decision 40 proves that it ends.

`opgate.hosts.register` registers a text with the executor as written, or,
if the gate refuses such an access or loop (`applies`), the guarded text,
which a host launches with the arguments of the text as written: the
executor computes the extents and the limit's sum from the launch's sizes,
and `opgate.hosts.launch` supplies the status buffer (`reports`).

Imports `opgate.signature`, `opgate.binding`, `opgate.frontends.participation`,
and the decisions of `opgate.gate`.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from opgate.binding import BASE_ALIGN, Base, Binding, Direction, Size, Span
from opgate.frontends import entry_kernels, entry_parameters, participation
from opgate.gate import Accepted, Refused, zeroed
from opgate.signature import Affine

__all__ = [
    "BATCH",
    "LIMIT",
    "ROUNDS",
    "STATUS",
    "STATUS_BYTES",
    "STATUS_DECLARED",
    "UNPROVED",
    "Guarded",
    "applies",
    "declares_status",
    "guard",
    "guard_every",
    "reports",
]

ROUNDS = 32  # the most checks of one kernel by the gate
# the refusals of an access that a check makes provable: of its bounds, and of
# its alignment, since the check keeps whole elements of the access's width
UNPROVED = (
    "the access is not proved inside its buffer, a shared variable, or a local variable: ",
    "the shared memory access is not proved inside a shared variable: ",
    "the access is not proved aligned to its width: ",
)
_WIDTHS = {"8": 1, "16": 2, "32": 4, "64": 8, "128": 16}
_TYPE = re.compile(r"\.[bsuf](8|16|32|64|128)$|\.bf16$")
_ACCESS = re.compile(r"^(ld|st|atom|red)\.")
_ADDRESS = re.compile(r"\[\s*(%\w+)\s*(?:\+\s*(-?\d+))?\s*\]")
# an instruction, which may close a scope on its line, as nvcc's `cvta ...; }`
_LINE = re.compile(r"^(\s*)(@!?%\w+\s+)?([a-z][\w.:]*)(\s+.*?)?;\s*\}?\s*(//.*)?$")
_CLOSES = re.compile(r";\s*\}\s*(//.*)?$")
_PARAMETER = re.compile(r"\.param\s+\.\w+\s+(\w+)")
_REGISTERS = re.compile(r"\.reg\s+\.[bsuf](\d+)\s+(%[\w$]+)<\d+>")
_PREDICATES = re.compile(r"\.reg\s+\.pred\s+(%[\w$]+)<\d+>")
# the special registers, whose names an inlined callee keeps
_SPECIAL = frozenset(
    "tid ntid ctaid nctaid laneid warpid nwarpid smid nsmid gridid clock clock64 "
    "globaltimer dynamic_smem_size total_smem_size aggr_smem_size".split()
)
BATCH = 4  # the rounds of the listed accesses before a round checks every access it can
_SHARED = re.compile(
    r"^\s*(?:\.visible\s+)?(\.extern\s+)?\.shared\s+.*?\s([A-Za-z_$][\w$]*)\[(\d*)\]\s*;"
)
STATUS = "opgate_status"
"""The parameter of the status buffer, to whose first word a failed check stores 1."""
STATUS_BYTES = 4
STATUS_DECLARED = f"the text declares the parameter {STATUS}, which only the guard pass adds"
_FAIL = 1  # the number that a failed check stores to the status buffer's first word
ENDS = "the kernel is not proved to end"
# the refusal of one loop, which an iteration limit makes provable
_BACK_EDGE = f"{ENDS}: the check does not prove that the loop of this back edge ends: "
# the refusal of a register that an instruction may read before any writes it
_UNWRITTEN = re.compile(r"an instruction may read the register (%[\w$]+) before an instruction")
_BRANCH = re.compile(r"^bra(\.uni)?$")
# the opcodes that write no register
_NO_DESTINATION = ("st.", "red.", "bra", "ret", "exit", "bar", "membar", "fence", "trap", "call")
LIMIT = "opgate_limit"
"""The parameter that holds the sum of the buffers' extents, to which the kernel
adds the grid's thread count for the iteration limit of a loop."""
_STATUS = Base(Span(Direction.WRITE, Affine(STATUS_BYTES), BASE_ALIGN, None, True))
_LABEL = re.compile(r"^\s*([$\w][\w$]*):")


def declares_status(text: str) -> bool:
    """Whether an entry kernel of `text` itself declares the parameter `STATUS`,
    which only the pass appends: a host refuses such a text (`STATUS_DECLARED`),
    so that a status buffer is always the pass's."""
    return any(STATUS in entry_parameters(text, kernel) for kernel in entry_kernels(text))


def applies(reason: str) -> bool:
    """Whether the gate's refusal for `reason` is one that the pass addresses: of an
    access that it does not prove in bounds or aligned, of barrier
    participation, or of a loop that it does not prove ends."""
    return any(prefix in reason for prefix in (*UNPROVED, participation.PARTICIPATION, _BACK_EDGE))


def reports(binding: Binding) -> bool:
    """Whether `binding` is of a text with the pass's checks: its last parameter is
    the status buffer."""
    return binding.parameters[-1:] == ((STATUS, _STATUS),)


@dataclass(frozen=True)
class Guarded:
    """A kernel's text with bounds checks and iteration limits, its binding with
    the appended extent and limit parameters and, when the pass added a check
    or a limit, the status buffer, the accesses checked, the back edges of the
    loops limited, whether the pass rewrote the kernel's barrier structure
    (`opgate.frontends.participation`), and the gate's decision on the two."""

    text: str
    binding: Binding
    guarded: tuple[str, ...]
    limited: tuple[str, ...]
    restructured: bool
    decision: Accepted | Refused


@dataclass(frozen=True)
class _Instruction:
    indent: str
    predicate: str  # "", "@%p", or "@!%p"
    opcode: str
    operands: str


@dataclass(frozen=True)
class _Region:
    """Where an access lies: a buffer parameter with its extent at the launch's
    sizes, or a shared variable with its declared bytes, or dynamic shared
    memory with the binding's extent."""

    name: str
    shared: bool
    extent: Affine | int
    generic: bool = False  # a shared variable reached by a generic address


def _parse(line: str) -> _Instruction | None:
    match = _LINE.match(line)
    if match is None:
        return None
    predicate = (match.group(2) or "").strip()
    return _Instruction(match.group(1), predicate, match.group(3), match.group(4) or "")


def _operands(text: str) -> list[str]:
    """The comma-separated operands, a braced vector or a bracketed address being one."""
    parts, depth, current = [], 0, ""
    for character in text:
        if character in "{[":
            depth += 1
        elif character in "}]":
            depth -= 1
        if character == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        parts.append(current.strip())
    return parts


def _width(opcode: str) -> int | None:
    """The bytes that an access instruction reads or writes."""
    match = _TYPE.search(opcode)
    if match is None:
        return None
    width = 2 if match.group(1) is None else _WIDTHS[match.group(1)]
    lanes = re.search(r"\.v(2|4|8)\.", opcode)
    return width * (int(lanes.group(1)) if lanes else 1)


class _Kernel:
    """The lines of the text, where the entry kernel's parameter list closes and
    its body opens and closes, the bits of each register prefix, the module's
    shared variables, and each register's definitions in the body."""

    def __init__(self, text: str, binding: Binding) -> None:
        self.lines = text.split("\n")
        entry = re.compile(rf"\.entry\s+{re.escape(binding.kernel)}\s*\(")
        start = next(i for i, line in enumerate(self.lines) if entry.search(line))
        self.body = next(i for i in range(start, len(self.lines)) if self.lines[i].strip() == "{")
        self.close = max(
            i for i in range(start, self.body) if self.lines[i].strip().startswith(")")
        )
        self.parameters = _PARAMETER.findall("\n".join(self.lines[start : self.close]))
        depth, end = 0, None
        for i in range(self.body, len(self.lines)):
            depth += self.lines[i].count("{") - self.lines[i].count("}")
            if depth == 0:
                end = i
                break
        if end is None:
            raise ValueError("the entry kernel's body does not close")
        self.end = end
        self.bits = {prefix: int(bits) for bits, prefix in _REGISTERS.findall(text)}
        self.predicates = set(_PREDICATES.findall(text))
        # each shared variable's bytes, or None for dynamic shared memory
        self.shared: dict[str, int | None] = {}
        for line in self.lines:
            found = _SHARED.match(line)
            if found is not None:
                size = found.group(3)
                self.shared[found.group(2)] = int(size) if size and not found.group(1) else None
        self.definitions: dict[str, list[_Instruction]] = {}
        for number in range(self.body + 1, self.end):
            instruction = _parse(self.lines[number])
            if instruction is None or instruction.opcode.startswith(("st.", "red.", "bra")):
                continue
            operands = _operands(instruction.operands)
            for register in re.findall(r"%\w+", operands[0] if operands else ""):
                self.definitions.setdefault(register, []).append(instruction)

    def width(self, register: str) -> int | None:
        """The bits of `register`, from its prefix's declaration."""
        return self.bits.get(re.sub(r"\d+$", "", register))

    def coefficients(self, bases: set[str]) -> dict[str, dict[str, int] | None]:
        """Each register's value as a sum of the `bases` (parameters that hold a
        buffer's base, and shared variables) times integer coefficients, plus a
        value that no base enters, when every definition of the register gives
        the same sum; None when a definition gives none. A value from memory is
        data. The registers that no definition reaches are absent."""
        sums: dict[str, dict[str, int] | None] = {}

        def operand(token: str) -> tuple[bool, dict[str, int] | None]:
            """Whether the operand's sum is known yet, and the sum."""
            token = token.strip()
            if token.startswith("%"):
                if token in sums:
                    return True, sums[token]
                # a special register, such as %tid.x, is data; a register that
                # a definition reaches waits for it
                return not self.definitions.get(token), {}
            return True, {token: 1} if token in bases else {}

        def combine(opcode: str, values: list[dict[str, int] | None]) -> dict[str, int] | None:
            if any(v is None for v in values):
                return None
            known = [v for v in values if v is not None]
            match opcode.split(".")[0]:
                case "mov" | "cvta" | "cvt" if len(known) == 1:
                    return dict(known[0])
                case "add" if len(known) == 2:
                    return {
                        k: known[0].get(k, 0) + known[1].get(k, 0) for k in {*known[0], *known[1]}
                    }
                case "sub" if len(known) == 2:
                    return {
                        k: known[0].get(k, 0) - known[1].get(k, 0) for k in {*known[0], *known[1]}
                    }
                case "mad" if (
                    len(known) == 3 and not any(known[0].values()) and not any(known[1].values())
                ):
                    return dict(known[2])
                case "selp" if len(known) == 3 and known[0] == known[1]:
                    return dict(known[0])  # either value: they have the same sum
                case _:
                    return {} if not any(any(v.values()) for v in known) else None

        changed = True
        while changed:
            changed = False
            for register, instructions in self.definitions.items():
                results: list[dict[str, int] | None] = []
                for instruction in instructions:
                    operands = _operands(instruction.operands)[1:]
                    if instruction.opcode.startswith("ld.param"):
                        names = re.findall(r"\[\s*(\w+)", " ".join(operands))
                        results.append({names[0]: 1} if names and names[0] in bases else {})
                        continue
                    if _ACCESS.match(instruction.opcode) or instruction.opcode.startswith("setp"):
                        results.append({})
                        continue
                    known = [operand(token) for token in operands]
                    if not all(ready for ready, _ in known):
                        continue
                    results.append(combine(instruction.opcode, [value for _, value in known]))
                if not results:
                    continue
                joined: dict[str, int] | None = results[0]
                for result in results[1:]:
                    if result != joined:
                        joined = None
                if joined is not None:
                    joined = {k: c for k, c in joined.items() if c}
                    if any(abs(c) > 1 for c in joined.values()):
                        joined = None
                if register not in sums or sums[register] != joined:
                    if register in sums and sums[register] is None:
                        continue  # None is the top: it stays
                    sums[register] = joined
                    changed = True
        return sums

    def sources(self, register: str) -> tuple[set[str], set[str], bool, bool]:
        """The parameters and the shared variables that `register` derives from, and
        whether its derivation passes through `cvta.to.global` and through
        `cvta.shared`, which makes a shared address generic. A value from
        memory is data, not an address, and the derivation does not go through
        it."""
        parameters: set[str] = set()
        variables: set[str] = set()
        converted, generic, seen, pending = False, False, set(), [register]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            for instruction in self.definitions.get(current, []):
                operands = " ".join(_operands(instruction.operands)[1:])
                if instruction.opcode.startswith("ld.param"):
                    parameters.update(re.findall(r"\[\s*(\w+)", operands))
                elif not _ACCESS.match(instruction.opcode):
                    converted |= instruction.opcode.startswith("cvta.to.global")
                    generic |= instruction.opcode.startswith("cvta.shared")
                    pending.extend(re.findall(r"%\w+", operands))
                    words = re.findall(r"(?<![%\w$])([A-Za-z_$][\w$]*)", operands)
                    variables.update(w for w in words if w in self.shared)
        return parameters, variables, converted, generic


@dataclass
class _Rewrite:
    """The checks and limits added so far: the instructions at the kernel's start
    that hold each region's base, each bound, and the iteration limit, the
    appended extent and limit parameters, the registers that the checks use,
    the register that holds the iteration limit, once a loop is limited, and
    the register that holds the status buffer's address, once a check or a
    limit is added, and the registers that the pass writes zero to at the
    kernel's start."""

    kernel: _Kernel
    binding: Binding
    sums: dict[str, dict[str, int] | None]  # each register's sum of bases
    prologue: list[str]
    bases: dict[tuple[str, bool, int], str]
    bounds: dict[tuple[str, int, int], str]
    extents: dict[str, str]  # each region's extent parameter, by the region's name
    appended: list[tuple[str, Size]]
    counts: dict[int, int]  # registers used, by bits; 1 for predicates
    limit: str | None
    status: str | None
    zeroed: set[str]

    def register(self, bits: int) -> str:
        self.counts[bits] = self.counts.get(bits, 0) + 1
        return f"%ogp{self.counts[bits]}" if bits == 1 else f"%og{bits}_{self.counts[bits]}"

    def base(self, region: _Region, converted: bool, bits: int) -> str:
        key = (region.name, converted or region.generic, bits)
        if key not in self.bases:
            register = self.register(bits)
            if region.generic:
                if bits != 64:
                    raise ValueError("a generic address has 64 bits")
                shared = self.register(32)
                self.prologue.append(f"\tmov.u32 \t{shared}, {region.name};")
                self.prologue.append(f"\tcvt.u64.u32 \t{register}, {shared};")
                self.prologue.append(f"\tcvta.shared.u64 \t{register}, {register};")
            elif region.shared:
                self.prologue.append(f"\tmov.u{bits} \t{register}, {region.name};")
            else:
                self.prologue.append(f"\tld.param.u64 \t{register}, [{region.name}];")
                if converted:
                    self.prologue.append(f"\tcvta.to.global.u64 \t{register}, {register};")
            self.bases[key] = register
        return self.bases[key]

    def bound(self, region: _Region, width: int, bits: int) -> str:
        """`extent - width`, the largest offset of an access of `width` bytes: an
        immediate for a declared shared variable, or a register that an appended
        extent parameter fills."""
        if isinstance(region.extent, int):
            return str(region.extent - width)
        key = (region.name, width, bits)
        if key not in self.bounds:
            name = self.extents.get(region.name)
            if name is None:
                name = f"opgate_extent_{len(self.extents)}"
                if name in self.kernel.parameters:
                    raise ValueError(f"the kernel has a parameter {name}")
                self.extents[region.name] = name
                self.appended.append((name, Size(region.extent)))
            register = self.register(bits)
            self.prologue.append(f"\tld.param.u{bits} \t{register}, [{name}];")
            self.prologue.append(f"\tadd.s{bits} \t{register}, {register}, {-width};")
            self.bounds[key] = register
        return self.bounds[key]

    def iterations(self) -> str:
        """The register that holds the iteration limit of a loop: the sum of the
        buffers' extents at the launch's sizes, from the appended parameter
        `LIMIT`, plus the grid's thread count, computed in 64 bits."""
        if self.limit is None:
            if LIMIT in self.kernel.parameters:
                raise ValueError(f"the kernel has a parameter {LIMIT}")
            extents = [p.span.extent for _, p in self.binding.parameters if isinstance(p, Base)]
            self.appended.append((LIMIT, Size(_total(extents))))
            limit, threads, dimension = (self.register(64) for _ in range(3))
            blocks, width = self.register(32), self.register(32)
            self.prologue.append(f"\tld.param.u64 \t{limit}, [{LIMIT}];")
            for axis in "xyz":
                product = threads if axis == "x" else dimension
                self.prologue.append(f"\tmov.u32 \t{blocks}, %nctaid.{axis};")
                self.prologue.append(f"\tmov.u32 \t{width}, %ntid.{axis};")
                self.prologue.append(f"\tmul.wide.u32 \t{product}, {blocks}, {width};")
                if axis != "x":
                    self.prologue.append(f"\tmul.lo.u64 \t{threads}, {threads}, {dimension};")
            self.prologue.append(f"\tadd.u64 \t{limit}, {limit}, {threads};")
            self.limit = limit
        return self.limit

    def failure(self) -> str:
        """The register that holds the global address of the status buffer, from
        the appended parameter `STATUS`, to whose first word a failed check
        stores the number 1 (`_FAIL`)."""
        if self.status is None:
            if STATUS in self.kernel.parameters:
                raise ValueError(f"the kernel has a parameter {STATUS}")
            address = self.register(64)
            self.prologue.append(f"\tld.param.u64 \t{address}, [{STATUS}];")
            self.prologue.append(f"\tcvta.to.global.u64 \t{address}, {address};")
            self.status = address
        return self.status


def _region(rewrite: _Rewrite, shared: bool, register: str) -> _Region | None:
    """The one buffer or shared variable whose base an address is, plus a value that
    no base enters (`_Kernel.coefficients`)."""
    terms = rewrite.sums.get(register)
    if terms is None and shared:
        # an address that no sum of bases gives, as Triton's swizzled layouts:
        # derived from one shared variable and no buffer, that variable is its base
        parameters, variables, _, _ = rewrite.kernel.sources(register)
        buffers = {n for n, p in rewrite.binding.parameters if isinstance(p, Base)}
        if len(variables) == 1 and not parameters & buffers:
            terms = {next(iter(variables)): 1}
    if terms is None or len(terms) != 1:
        return None
    ((name, coefficient),) = terms.items()
    if coefficient != 1:
        return None
    if name in rewrite.kernel.shared:
        # a shared access, or a generic one whose address `cvta.shared` made
        generic = not shared and rewrite.kernel.sources(register)[3]
        if not shared and not generic:
            return None
        size = rewrite.kernel.shared[name]
        extent = rewrite.binding.shared if size is None else size
        return _Region(name, True, extent, generic)
    parameter = dict(rewrite.binding.parameters).get(name)
    if shared or not isinstance(parameter, Base):
        return None
    return _Region(name, False, parameter.span.extent)


def _bounds(
    rewrite: _Rewrite, register: str, offset: int, width: int, shared: bool, indent: str
) -> tuple[list[str], str, str] | None:
    """The lines that check an access of `width` bytes at `register + offset`, the
    register that holds its checked address, and the predicate that holds when
    the check passes; or None if the pass does not check it."""
    if width & (width - 1):
        return None
    bits = rewrite.kernel.width(register)
    if bits not in (32, 64):
        return None
    region = _region(rewrite, shared, register)
    if region is None:
        return None
    _, _, converted, _ = rewrite.kernel.sources(register)
    bound = rewrite.bound(region, width, bits)
    base = rewrite.base(region, converted, bits)
    shift = width.bit_length() - 1
    difference, whole, at = (rewrite.register(bits) for _ in range(3))
    low, high, inside = (rewrite.register(1) for _ in range(3))
    lines = [f"{indent}sub.s{bits} \t{difference}, {register}, {base};"]
    if offset:
        lines.append(f"{indent}add.s{bits} \t{difference}, {difference}, {offset};")
    # a shift, by 0 for a byte, makes the whole elements a value of their own for
    # the gate, whose bounds the check gives; a copy would keep the address's form
    lines += [
        f"{indent}shr.s{bits} \t{whole}, {difference}, {shift};",
        f"{indent}shl.b{bits} \t{whole}, {whole}, {shift};",
        f"{indent}setp.ge.s{bits} \t{low}, {whole}, 0;",
        f"{indent}setp.le.s{bits} \t{high}, {whole}, {bound};",
        f"{indent}and.pred \t{inside}, {low}, {high};",
        f"{indent}add.s{bits} \t{at}, {base}, {whole};",
    ]
    return lines, at, inside


def _predicated(
    rewrite: _Rewrite, lines: list[str], inside: str, predicate: str, indent: str
) -> tuple[str, str]:
    """The predicate under which the instruction runs, which holds when the check
    passes and the instruction's predicate, `@%p` or `@!%p`, holds; and the
    predicate under which the check fails and the instruction's predicate
    holds."""
    outside = rewrite.register(1)
    lines.append(f"{indent}not.pred \t{outside}, {inside};")
    if not predicate:
        return inside, outside
    if predicate.startswith("@!"):
        holds = rewrite.register(1)
        lines.append(f"{indent}not.pred \t{holds}, {predicate[2:]};")
    else:
        holds = predicate[1:]
    runs, fails = rewrite.register(1), rewrite.register(1)
    lines.append(f"{indent}and.pred \t{runs}, {inside}, {holds};")
    lines.append(f"{indent}and.pred \t{fails}, {outside}, {holds};")
    return runs, fails


def _check(rewrite: _Rewrite, instruction: _Instruction) -> list[str] | None:
    """The lines of `instruction` with its bounds check, which stores 1 to the
    status buffer when the check fails; or None if it is not guarded."""
    width = _width(instruction.opcode)
    address = _ADDRESS.search(instruction.operands)
    if width is None or address is None:
        return None
    if any(space in instruction.opcode for space in (".local", ".param", ".const")):
        return None
    indent = instruction.indent or "\t"
    register, offset = address.group(1), int(address.group(2) or 0)
    checked = _bounds(rewrite, register, offset, width, ".shared" in instruction.opcode, indent)
    if checked is None:
        return None
    lines, at, inside = checked
    operands = _ADDRESS.sub(f"[{at}]", instruction.operands, count=1)
    after: list[str] = []
    if not instruction.predicate and instruction.opcode.startswith(("ld.", "atom.")):
        received = _received(rewrite, operands, indent)
        if received is None:
            return None
        before, operands, after = received
        lines += before
    runs, fails = _predicated(rewrite, lines, inside, instruction.predicate, indent)
    status = rewrite.failure()
    lines.append(f"{indent}@{fails} st.global.u32 \t[{status}], {_FAIL};")
    lines.append(f"{indent}@{runs} {instruction.opcode}{operands};")
    return lines + after


def _received(
    rewrite: _Rewrite, operands: str, indent: str
) -> tuple[list[str], str, list[str]] | None:
    """For a load or an atomic that runs when its check passes: the lines that
    write zero to a register of the pass for each of its destinations, its
    operands with those registers as its destinations, and the lines that copy
    them into the destinations after it, so that a failed check leaves zero in
    the destinations; or None if the bits of a destination are not declared."""
    found = re.match(r"\s*(\{[^}]*\}|[^,{]+),", operands)
    if found is None:
        return None
    destination = found.group(1)
    before: list[str] = []
    after: list[str] = []
    for name in re.findall(r"%[\w$]+", destination):
        bits = rewrite.kernel.width(name)
        if bits is None:
            return None
        temporary = rewrite.register(bits)
        before.append(f"{indent}mov.b{bits} \t{temporary}, 0;")
        after.append(f"{indent}mov.b{bits} \t{name}, {temporary};")
        destination = re.sub(rf"{re.escape(name)}(?![\w$])", temporary, destination, count=1)
    replaced = operands[: found.start(1)] + destination + operands[found.end(1) :]
    return before, replaced, after


def _plain(opcode: str) -> str:
    """An opcode as the gate prints it, without the scope `::cta` of a shared access."""
    return opcode.replace("::cta", "")


def _guard_access(rewrite: _Rewrite, refused: str) -> int:
    """Guard every access of the refused instruction's opcode and address; how many.
    The gate names a register of a nested scope, as an inlined body's, `%sN$name`;
    the pass's inlined registers have names of their own, so the scope drops."""
    parsed = _parse(re.sub(r"%s\d+\$", "%", refused.strip()) + ";")
    if parsed is None:
        return 0
    address = _ADDRESS.search(parsed.operands)
    if address is None:
        return 0
    register, offset = address.group(1), int(address.group(2) or 0)
    kernel = rewrite.kernel
    changes: dict[int, list[str]] = {}
    for number in range(kernel.body + 1, kernel.end):
        instruction = _parse(kernel.lines[number])
        if instruction is None or _plain(instruction.opcode) != _plain(parsed.opcode):
            continue
        found = _ADDRESS.search(instruction.operands)
        if found is None or found.group(1) != register or int(found.group(2) or 0) != offset:
            continue
        lines = _check(rewrite, instruction)
        if lines is None:
            return 0
        changes[number] = lines
    for number, lines in changes.items():
        _replace(kernel, number, lines)
    return len(changes)


def _replace(kernel: _Kernel, number: int, lines: list[str]) -> None:
    """Replace line `number` by `lines`, which end with its instruction or the
    copies of its results, keeping the brace that closes a scope after them."""
    closes = _CLOSES.search(kernel.lines[number]) is not None
    kernel.lines[number] = "\n".join(lines) + (" }" if closes else "")


def _guard_every(rewrite: _Rewrite) -> int:
    """Guard every access that the pass can check and has not; how many."""
    kernel = rewrite.kernel
    changes: dict[int, list[str]] = {}
    for number in range(kernel.body + 1, kernel.end):
        instruction = _parse(kernel.lines[number])
        if instruction is None or not _ACCESS.match(instruction.opcode):
            continue
        if instruction.predicate.startswith("@%ogp"):
            continue  # checked already
        lines = _check(rewrite, instruction)
        if lines is not None:
            changes[number] = lines
    for number, lines in changes.items():
        _replace(kernel, number, lines)
    return len(changes)


def _total(affines: list[Affine]) -> Affine:
    """The sum of `affines`."""
    coefficients: dict[int, int] = {}
    for affine in affines:
        for symbol, coefficient in affine.terms:
            coefficients[symbol] = coefficients.get(symbol, 0) + coefficient
    terms = tuple(sorted((s, c) for s, c in coefficients.items() if c))
    return Affine(sum(affine.constant for affine in affines), terms)


def _limit(rewrite: _Rewrite, refused: str) -> bool:
    """Give the loop whose back edge the gate names an iteration limit; whether the
    pass found the loop. The loop is the lines from its head's label to the
    last branch back to it. Before the head, and before each branch from
    outside the loop to a label inside it, a counter is set to 0; at the
    head, it is incremented, and when it reaches the limit
    (`_Rewrite.iterations`), the kernel stores 1 to the status buffer and
    leaves the loop to the line after its last back edge. The gate names a
    register of a nested scope `%sN$name` and a label `sN$$name`, which drop
    the scope as in `_guard_access`."""
    plain = re.sub(r"(?<![\w$])s\d+\$(?=\$)", "", re.sub(r"%s\d+\$", "%", refused.strip()))
    parsed = _parse(plain + ";")
    if parsed is None or parsed.opcode.split(".")[0] != "bra":
        return False
    kernel = rewrite.kernel
    head = parsed.operands.strip()
    # each line of the body, as its entry and its line in the entry, since an
    # entry holds the lines that the pass put in its place
    positions = [
        (number, part)
        for number in range(kernel.body + 1, kernel.end)
        for part in range(len(kernel.lines[number].split("\n")))
    ]
    text = {(n, k): kernel.lines[n].split("\n")[k] for n, k in positions}
    labels = {m.group(1): at for at in positions if (m := _LABEL.match(text[at]))}
    start = labels.get(head)
    if start is None:
        return False
    branches = {
        at: ins
        for at in positions
        if (ins := _parse(text[at])) is not None and ins.opcode.split(".")[0] == "bra"
    }
    back = [at for at, ins in branches.items() if ins.operands.strip() == head and at > start]
    named = (parsed.predicate, parsed.opcode)
    if not any((branches[at].predicate, branches[at].opcode) == named for at in back):
        return False
    end = max(back)
    inside = {name for name, at in labels.items() if start <= at <= end}
    counter, over = rewrite.register(64), rewrite.register(1)
    limit = rewrite.iterations()
    status = rewrite.failure()
    leave = f"$og_exit_{counter[1:]}"
    reset = f"\tmov.u64 \t{counter}, 0;"
    before: dict[tuple[int, int], list[str]] = {start: [reset]}
    after: dict[tuple[int, int], list[str]] = {
        start: [
            f"\tadd.u64 \t{counter}, {counter}, 1;",
            f"\tsetp.ge.u64 \t{over}, {counter}, {limit};",
            f"\t@{over} st.global.u32 \t[{status}], {_FAIL};",
            f"\t@{over} bra \t{leave};",
        ],
        end: [f"{leave}:"],
    }
    for at, ins in branches.items():
        if ins.operands.strip() in inside and not start <= at <= end:
            before.setdefault(at, []).append(reset)
    for number in {n for n, _ in (*before, *after)}:
        parts = kernel.lines[number].split("\n")
        kernel.lines[number] = "\n".join(
            line
            for k, part in enumerate(parts)
            for line in (*before.get((number, k), []), part, *after.get((number, k), []))
        )
    return True


def _zero(rewrite: _Rewrite, register: str) -> bool:
    """Write zero to `register` at the kernel's start; whether the pass could, which
    needs a register of the kernel's scope whose width it declares, not yet
    zeroed. The gate refuses a kernel in which an instruction may read a
    register before an instruction writes it (design decision 46), as on the
    path that a loop's limit adds from its head to its exit, or in a region that
    the barrier rewrite runs under a predicate. On the limit's path the launch
    reports a failed check, and in the region the predicate keeps the value
    from every store, so a launch whose checks pass gives the same results
    with the zero."""
    if register in rewrite.zeroed or "$" in register:
        return False
    prefix = re.sub(r"\d+$", "", register)
    bits = rewrite.kernel.width(register)
    if bits is not None:
        rewrite.prologue.append(f"\tmov.b{bits} \t{register}, 0;")
    elif prefix in rewrite.kernel.predicates:
        rewrite.prologue.append(f"\tmov.pred \t{register}, 0;")
    else:
        return False
    rewrite.zeroed.add(register)
    return True


def _unwritten(rewrite: _Rewrite) -> list[str]:
    """The registers of the kernel's body that an instruction may read before every
    path from the body's start writes them, by a forward analysis of the
    registers that every path writes: an instruction under a predicate writes
    none for certain, and the registers that the prologue writes start as
    written. The pass zeroes them all in one round, in place of the one
    register that each check names; a zero of a register that the gate would
    not name changes no value that a path reads after a write."""
    kernel = rewrite.kernel
    blocks: list[list[_Instruction]] = [[]]
    labels: dict[str, int] = {}
    for number in range(kernel.body + 1, kernel.end):
        for line in kernel.lines[number].split("\n"):
            label = _LABEL.match(line)
            if label is not None:
                if blocks[-1]:
                    blocks.append([])
                labels[label.group(1)] = len(blocks) - 1
                continue
            instruction = _parse(line)
            if instruction is None:
                continue
            blocks[-1].append(instruction)
            if _BRANCH.match(instruction.opcode) or instruction.opcode.startswith(("ret", "exit")):
                blocks.append([])

    def successors(b: int) -> list[int]:
        last = blocks[b][-1] if blocks[b] else None
        if last is not None and _BRANCH.match(last.opcode):
            target = labels[last.operands.strip()]
            return [target, b + 1] if last.predicate else [target]
        if last is not None and last.opcode.startswith(("ret", "exit")) and not last.predicate:
            return []
        return [b + 1] if b + 1 < len(blocks) else []

    def effects(instruction: _Instruction) -> tuple[list[str], list[str]]:
        """The registers that `instruction` writes and those that it reads."""
        operands = _operands(instruction.operands)
        uses = re.findall(r"%[\w$.]+", instruction.predicate)
        if instruction.opcode.startswith(_NO_DESTINATION) or not operands:
            return [], uses + re.findall(r"%[\w$]+", " ".join(operands))
        writes = re.findall(r"%[\w$]+", operands[0])
        return writes, uses + re.findall(r"%[\w$]+", " ".join(operands[1:]))

    start = set(rewrite.zeroed)
    for line in rewrite.prologue:
        instruction = _parse(line)
        if instruction is not None:
            start.update(effects(instruction)[0])
    entry: dict[int, set[str]] = {0: start}
    work = [0]
    while work:
        b = work.pop()
        written = set(entry[b])
        for instruction in blocks[b]:
            if not instruction.predicate:
                written.update(effects(instruction)[0])
        for v in successors(b):
            joined = written if v not in entry else entry[v] & written
            if v not in entry or joined != entry[v]:
                entry[v] = set(joined)
                work.append(v)
    read: list[str] = []
    for b, known in sorted(entry.items()):
        written = set(known)
        for instruction in blocks[b]:
            writes, uses = effects(instruction)
            for register in uses:
                special = register[1:].split(".")[0] in _SPECIAL
                if not special and register not in written and register not in read:
                    read.append(register)
            if not instruction.predicate:
                written.update(writes)
    return read


def _functions(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Each device function that the module defines: the lines of its body."""
    functions: dict[str, tuple[int, int]] = {}
    for number, line in enumerate(lines):
        declared = re.match(r"^\s*(?:\.\w+\s+)*\.func\s+(?:\([^)]*\)\s*)?([\w$]+)", line)
        if declared is None:
            continue
        opening = next(
            (i for i in range(number, len(lines)) if lines[i].strip() in ("{", ";")), None
        )
        if opening is None or lines[opening].strip() == ";":
            continue  # a declaration, not a definition
        depth = 0
        for closing in range(opening, len(lines)):
            depth += lines[closing].count("{") - lines[closing].count("}")
            if depth == 0:
                functions[declared.group(1)] = (opening, closing)
                break
    return functions


def _called(lines: list[str], start: int, end: int) -> list[str]:
    """The functions that the calls between lines `start` and `end` name."""
    names = []
    for number in range(start, end):
        if re.match(r"^\s*(@!?%\w+\s+)?call(\.uni)?\b", lines[number]):
            rest = " ".join(lines[number : number + 2])
            found = re.search(r"call(?:\.uni)?\s*(?:\([^)]*\)\s*,\s*)?([\w$]+)", rest)
            if found is not None:
                names.append(found.group(1))
    return names


def _widths(lines: list[str]) -> dict[str, int]:
    """The bits of each register name prefix that `lines` declare."""
    return {name: int(bits) for bits, name in _REGISTERS.findall("\n".join(lines))}


def _move(kind: str, target: str, value: str, widths: dict[str, int]) -> str | None:
    """The instruction that moves a parameter's `value` of type `kind` into the
    register `target`, as `ld.param` and `st.param` would: by its bits when the
    registers have the same width, since the caller's register may hold another
    type of that width, as a double that a function takes as a 64-bit integer;
    extended, by the sign for a signed `kind`, into a wider register; and an
    immediate by its bits, a float literal's written as an integer; None for a
    register whose width no counted declaration states."""
    bits = widths.get(re.sub(r"\d+$", "", target.strip()))
    value = value.strip()
    if bits is None:
        return None
    if not value.startswith("%"):
        literal = re.match(r"^0[dDfF]([0-9A-Fa-f]+)$", value)
        return f"mov.b{bits} \t{target}, {'0x' + literal.group(1) if literal else value};"
    source = widths.get(re.sub(r"\d+$", "", value))
    if source is None:
        return None
    if source == bits:
        return f"mov.b{bits} \t{target}, {value};"
    sign = "s" if kind.startswith("s") else "u"
    return f"cvt.{sign}{bits}.{sign}{source} \t{target}, {value};"


def _renamed(prefix: str, line: str) -> str:
    """A line of an inlined body, its registers, labels, and local depot renamed
    with `prefix`; the special registers keep their names."""
    line = re.sub(
        r"%([A-Za-z_$][\w$]*)",
        lambda m: m.group(0) if m.group(1).split(".")[0] in _SPECIAL else f"%{prefix}{m.group(1)}",
        line,
    )
    line = re.sub(r"\$([A-Za-z_][\w$]*)", lambda m: f"${prefix}{m.group(1)}", line)
    return re.sub(r"\b(__local_depot\d+)\b", rf"{prefix}\1", line)


def _inlined(
    lines: list[str], site: int, start: int, functions: dict[str, tuple[int, int]], count: int
) -> tuple[int, int, list[str], list[str]] | None:
    """The first and last line of the call sequence around the call at `site`, the
    body that replaces them, and the callee's local variables, renamed, which
    the kernel declares at the start of its body, where the zeroing prologue
    clears them (a declaration in the body's scope is not visible there); None
    if the pass cannot map the call's arguments or result, as for an argument
    that the callee loads and the caller does not store, or a structure
    returned by value."""
    name = _called(lines, site, site + 1)[0]
    opening = max(i for i in range(start, site) if re.match(r"^\s*\{\s*//\s*callseq", lines[i]))
    closing = next(
        i for i in range(site, len(lines)) if re.match(r"^\s*\}\s*//\s*callseq", lines[i])
    )
    sequence = "\n".join(lines[opening : closing + 1])
    call = re.search(
        rf"call(?:\.uni)?\s*(?:\(\s*([\w$]+)\s*\)\s*,\s*)?{re.escape(name)}\s*,?\s*(?:\(([^)]*)\))?",
        sequence,
    )
    if call is None:
        return None
    result_param = call.group(1)
    arguments = [a.strip() for a in (call.group(2) or "").split(",") if a.strip()]
    values = dict(re.findall(r"st\.param\.\w+\s+\[(\w+)(?:\+0)?\],\s*([^;]+);", sequence))
    result = None
    if result_param is not None:
        found = re.search(rf"ld\.param\.\w+\s+(%\w+),\s*\[{result_param}(?:\+0)?\]", sequence)
        if found is None:
            return None
        result = found.group(1)
    header_end, body_end = functions[name]
    declared = max(
        i for i in range(header_end) if re.search(rf"\.func\b.*\b{re.escape(name)}\b", lines[i])
    )
    header = "\n".join(lines[declared:header_end])
    formals = re.findall(r"\.param\s+\.\w+\s+(?:\.\w+\s+)*([\w$]+)", header)
    # a return value is declared before the function's name, its parameters after
    returned = re.search(
        rf"\.func\s*\(\s*\.param\s+\.\w+\s+([\w$]+)\s*\)\s*{re.escape(name)}\b", header
    )
    returns = returned.group(1) if returned is not None else None
    parameters = [p for p in formals if p != returns]
    if len(parameters) != len(arguments) or (returns is None) != (result_param is None):
        return None
    argument_of = {
        p: values[a].strip() for p, a in zip(parameters, arguments, strict=True) if a in values
    }
    prefix = f"oi{count}_"
    end_label = f"$oi{count}_end"
    callee = lines[header_end + 1 : body_end]
    # the instructions after each line of the body
    later = [any(_parse(rest) for rest in callee[i + 1 :]) for i in range(len(callee))]
    widths = _widths(lines) | _widths([_renamed(prefix, line) for line in callee])
    body: list[str] = [f"\t{{ // inlined {name}"]
    local: list[str] = []
    branched = False
    for number, line in enumerate(callee):
        loaded = re.match(r"^(\s*)ld\.param\.(\w+)\s+(%[\w$]+),\s*\[([\w$]+)(?:\+0)?\]\s*;", line)
        stored = re.match(r"^(\s*)st\.param\.(\w+)\s+\[([\w$]+)(?:\+0)?\],\s*([^;]+);", line)
        returning = re.match(r"^(\s*)(@!?%[\w$]+\s+)?ret(\.uni)?\s*;", line)
        named = re.search(r"\[([\w$]+)", line)
        if loaded is not None and loaded.group(4) in argument_of:
            indent, kind, register, parameter = loaded.groups()
            moved = _move(kind, _renamed(prefix, register), argument_of[parameter], widths)
            if moved is None:
                return None
            body.append(indent + moved)
        elif stored is not None and stored.group(3) == returns and result is not None:
            indent, kind, _, value = stored.groups()
            moved = _move(kind, result, _renamed(prefix, value), widths)
            if moved is None:
                return None
            body.append(indent + moved)
        elif named is not None and named.group(1) in (*parameters, returns):
            return None  # an access of a parameter that the pass does not map
        elif returning is not None:
            indent, predicate = returning.group(1), returning.group(2) or ""
            if predicate or later[number]:
                body.append(_renamed(prefix, f"{indent}{predicate}bra \t") + f"{end_label};")
                branched = True
        elif re.match(r"^\s*\.local\b", line):
            local.append(_renamed(prefix, line))
        else:
            body.append(_renamed(prefix, line))
    body += [f"{end_label}:", "\t}"] if branched else ["\t}"]
    return opening, closing, body, local


def _inline(text: str, kernel: str) -> str:
    """`text` with each call in the entry kernel `kernel` of a function that the
    module defines replaced by the function's body, as the gate inlines it: the
    callee's registers and labels renamed, its parameter loads moving the call's
    arguments, its return value moving into the call's result, and each `ret`
    before the body's last instruction a branch past the body. Calls in an
    inlined body are inlined in turn. A call whose arguments or result the pass
    cannot map stays as written, and the gate inlines it."""
    skipped = 0
    for count in range(1, 1000):
        lines = text.split("\n")
        functions = _functions(lines)
        entry = re.compile(rf"\.entry\s+{re.escape(kernel)}\s*\(")
        start = next(i for i, line in enumerate(lines) if entry.search(line))
        sites = [
            i
            for i in range(start, len(lines))
            if re.match(r"^\s*call(\.uni)?\b", lines[i])
            and _called(lines, i, i + 1)
            and _called(lines, i, i + 1)[0] in functions
        ]
        if len(sites) <= skipped:
            return text
        replaced = _inlined(lines, sites[skipped], start, functions, count)
        if replaced is None:
            skipped += 1
            continue
        opening, closing, body, local = replaced
        lines[opening : closing + 1] = body
        # the entry's body opens at the first brace after its header
        opened = next(i for i in range(start, len(lines)) if re.match(r"^\s*\{\s*$", lines[i]))
        lines[opened + 1 : opened + 1] = local
        text = "\n".join(lines)
    raise ValueError("calls nest deeper than the pass inlines")


def _text(rewrite: _Rewrite) -> str:
    """The kernel's text with the checks so far, their registers, the prologue, and
    the appended parameters: the extents, then the status buffer."""
    kernel = rewrite.kernel
    lines = list(kernel.lines)
    # the first instruction, after the body's declarations and comments
    first = next(
        i
        for i in range(kernel.body + 1, kernel.end + 1)
        if lines[i].strip() and not lines[i].strip().startswith((".", "//"))
    )
    lines[first:first] = rewrite.prologue
    lines[kernel.body + 1 : kernel.body + 1] = [
        f"\t.reg .pred \t%ogp<{count + 1}>;"
        if bits == 1
        else f"\t.reg .b{bits} \t%og{bits}_<{count + 1}>;"
        for bits, count in sorted(rewrite.counts.items())
    ]
    appended = [name for name, _ in _appended(rewrite)]
    if appended:
        last = max(i for i in range(kernel.close) if ".param" in lines[i])
        lines[last] = lines[last].rstrip().rstrip(",") + ","
        lines[last + 1 : last + 1] = [",\n".join(f"\t.param .u64 {name}" for name in appended)]
    return "\n".join(lines)


def _appended(rewrite: _Rewrite) -> list[tuple[str, Size | Base]]:
    """The parameters that the checks so far append: the extents, then the status
    buffer once a check is added."""
    status: list[tuple[str, Size | Base]] = [] if rewrite.status is None else [(STATUS, _STATUS)]
    return [*rewrite.appended, *status]


def _binding(rewrite: _Rewrite) -> Binding:
    """The kernel's binding with the parameters that the checks so far append."""
    binding = rewrite.binding
    return replace(binding, parameters=(*binding.parameters, *_appended(rewrite)))


def _unscoped(text: str) -> str:
    """`text` with each register that a scope declares by name, as nvcc's
    `{ .reg .b64 %tmp; ... }`, renamed apart from the other scopes' registers
    of the same name, so that a register's name has one meaning in the kernel."""
    lines = text.split("\n")
    count = 0
    for number, line in enumerate(lines):
        opened = re.match(r"^\s*\{\s*((?:\.reg\s+\.\w+\s+%[A-Za-z_$][\w$]*\s*;\s*)+)", line)
        if opened is None:
            continue
        names = re.findall(r"(%[A-Za-z_$][\w$]*)\s*;", opened.group(1))
        depth, closing = 0, number
        for closing in range(number, len(lines)):
            depth += lines[closing].count("{") - lines[closing].count("}")
            if depth == 0:
                break
        count += 1
        for name in names:
            pattern = re.compile(rf"{re.escape(name)}(?![\w$])")
            for scoped in range(number, closing + 1):
                lines[scoped] = pattern.sub(f"{name}_os{count}", lines[scoped])
    return "\n".join(lines)


def _start(text: str, binding: Binding) -> _Rewrite:
    """A rewrite of `text`, with each device function that its kernel calls inlined
    and the registers that scopes declare by name renamed apart."""
    kernel = _Kernel(_unscoped(_inline(text, binding.kernel)), binding)
    bases = {n for n, p in binding.parameters if isinstance(p, Base)} | set(kernel.shared)
    return _Rewrite(
        kernel, binding, kernel.coefficients(bases), [], {}, {}, {}, [], {}, None, None, set()
    )


def guard(
    text: str, binding: Binding, check: Callable[[str, Binding], Accepted | Refused]
) -> Guarded:
    """`text` and `binding` with a bounds check on each access that the gate, as
    `check` calls it, does not prove, as long as the pass can check it, and an
    iteration limit on each loop that it does not prove ends; and the gate's
    decision on the result. The barrier rewrite applies to a text that the
    gate refuses for participation, or, once the checks hold, for a loop that
    it does not prove ends: the rewrite bounds searches and widens grid
    strides. A loop gets a limit only after the barrier rewrite has been
    tried. A register that the gate finds may be read before an instruction
    writes it gets zero at the kernel's start (`_zero`), in the same round as
    every register that the pass finds may be read so (`_unwritten`). The gate
    decides each text with its zeroing prologue
    (`opgate.gate.zeroed`), which `Executor.register` puts there when it
    loads the text."""

    def decide(candidate: str, bound: Binding) -> Accepted | Refused:
        return check(zeroed(candidate, bound), bound)

    result = _guard(text, binding, decide, False)
    if (
        isinstance(result.decision, Refused)
        and not result.restructured
        and ENDS in result.decision.reason
    ):
        return _guard(text, binding, decide, True)
    return result


def _guard(
    text: str,
    binding: Binding,
    check: Callable[[str, Binding], Accepted | Refused],
    restructure: bool,
) -> Guarded:
    decision = check(text, binding)
    if isinstance(decision, Accepted):
        return Guarded(text, binding, (), (), False, decision)
    restructured = False
    tried = restructure or participation.PARTICIPATION in decision.reason
    if tried:
        structured = participation.rewrite(text, binding.block)
        if structured is not None:
            text, restructured = structured, True
            decision = check(text, binding)
            if isinstance(decision, Accepted):
                return Guarded(text, binding, (), (), restructured, decision)
    rewrite = _start(text, binding)
    inlined = _text(rewrite)
    guarded: list[str] = []
    limited: list[str] = []
    current, bound = inlined, binding
    if inlined != text:
        decision = check(current, bound)
    rounds = 0  # the rounds that checked accesses
    for _ in range(ROUNDS):
        if isinstance(decision, Accepted):
            break
        refused = _unproved_access(decision.reason)
        if refused is not None:
            if refused in guarded:
                break
            # the first BATCH rounds check the access that the gate names and
            # every other access that it lists; each later round checks every
            # access
            if rounds < BATCH:
                if _guard_access(rewrite, refused) == 0:
                    break
                guarded.append(refused)
                for item in decision.unproved:
                    other = _unproved_access(item)
                    if (
                        other is not None
                        and other not in guarded
                        and _guard_access(rewrite, other) > 0
                    ):
                        guarded.append(other)
            else:
                if _guard_every(rewrite) == 0:
                    break
                guarded.append(refused)
            rounds += 1
        elif tried and _BACK_EDGE in decision.reason:
            at = decision.reason.find(_BACK_EDGE)
            edge = decision.reason[at + len(_BACK_EDGE) :]
            if edge in limited or not _limit(rewrite, edge):
                break
            limited.append(edge)
        elif (unwritten := _UNWRITTEN.search(decision.reason)) is not None:
            if not _zero(rewrite, unwritten.group(1)):
                break
            for register in _unwritten(rewrite):
                _zero(rewrite, register)
        else:
            break
        current = _text(rewrite)
        bound = _binding(rewrite)
        decision = check(current, bound)
    return Guarded(current, bound, tuple(guarded), tuple(limited), restructured, decision)


def _unproved_access(reason: str) -> str | None:
    """The instruction of a refusal of an access that a check makes provable."""
    found = [(reason.find(p), p) for p in UNPROVED if p in reason]
    if not found:
        return None
    at, prefix = found[0]
    return reason[at + len(prefix) :]


def guard_every(text: str, binding: Binding) -> tuple[str, Binding]:
    """`text` and `binding` with a bounds check on every global and shared access
    that the pass can check, whether or not the gate proves it: the checks of a
    design that checks each access when it runs instead of proving it, against
    which the evaluation measures `guard`. The gate checks the result as any
    other, with its zeroing prologue (`opgate.gate.zeroed`)."""
    rewrite = _start(text, binding)
    _guard_every(rewrite)
    return _text(rewrite), _binding(rewrite)
