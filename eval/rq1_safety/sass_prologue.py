"""RQ1: the zeroing prologue in the SASS that the driver's JIT compiles from each deployed text.

The gate checks PTX, and its trusted base includes the driver's JIT compiler,
which `cuModuleLoadData` runs on the PTX when the service registers a text,
and the SASS that it writes. This experiment checks, for every text that the
service deploys with a zeroing prologue, that the prologue that the gate
verified is still in the SASS. The texts come from two complete records of
one registration: the entries that the registration record
(`eval.rq4_cost.registration`) accepts as written, and the entries that the
guarded record (`eval.rq2_coverage.guarded`) accepts with checks, as the
guard pass's text and binding. A text is checked when `zeroed` gives it a
prologue, that is, when it declares shared variables or local variables. The
text with its prologue is `zeroed` of the archived text and binding, and it
must be the text of the archived Rocq query, the exact text that the gate
decided.

Each text is loaded as `Executor.register` loads it: `cuModuleLoadData` of the
text and its terminating zero byte, on the primary context of the device,
with the driver's default JIT options, which `cuModuleLoadData` applies. The
load runs in a fresh process whose JIT cache (`CUDA_CACHE_PATH`) is an empty
directory, so that the driver writes exactly one cache entry, of this text:
the entry holds the PTX bytes that the driver compiled and ends in the cubin,
of the length that its header states. The cubin, loaded by the driver, must
give the entry function the attributes
that the load of the text gives (registers, local memory per thread, static
shared memory, constant memory, and the most threads of a block). The
driver's linker (`cuLinkCreate`) compiles PTX as relocatable code and gives
other SASS (on one kernel 105 registers against the load's 56), so the
experiment reads the cubin of the load itself. `nvdisasm -c` disassembles the
cubin. The record keeps each text, cubin, and SASS listing, the cache
entry's bytes before the PTX, the function's attributes, the path of the JIT
compiler library that the driver mapped, whose name holds the driver
release, and the driver's API version. The GPU of the timed runs is an RTX
A6000, sm_86.

Shared memory. Each thread of the binding's block runs the entry kernel's
SASS from its first instruction with its own `%tid`, an interpreter of the
integer instructions that compute the prologue's addresses and predicates.
Values that the thread index does not determine (parameters, `%ctaid`,
loaded data) are unknown, and a branch on an unknown predicate follows both
successors, so the check covers every path of every thread. Every path must
reach a `BAR.SYNC 0x0` without a thread count, unpredicated, and before it
run only stores of zero to shared memory (`STS` of `RZ` or of registers that
hold zero) and execute no shared load or atomic (`LDS`, `LDSM`, `ATOMS`) and
no generic load or atomic (`LD`, `ATOM`), which could address shared memory;
a path that exits, calls, or returns first fails. The bytes that a thread
zeroes are those that it stores on all of its paths, and the block's zeroed
bytes are their union. The prologue zeroes every byte of each shared variable
(the PTX prologue's stores, read from the text, give the bytes per variable).
The cubin leaves out the placement of a module's non-visible shared
variables, so the check is that the zeroed bytes below the static shared size
of the loaded function are exactly the bytes of some aligned, disjoint
placement of the static variables in that size, and the zeroed bytes from
the static size on are exactly the bytes that the prologue clears of the
dynamic shared memory, which starts at the static size. A kernel whose
prologue clears dynamic shared memory in a loop over a count from launch
parameters is checked with the parameters' words in constant bank 0 (at
`PARAM_BASE` plus each parameter's offset) set to each of a sample of sizes
in the binding's range, `SIZES` around the block's thread count and the
range's ends, below the shared memory of an sm_86 block; the other loop
entries are checked at every launch.

Local memory. A forward must-analysis over the control-flow graph of the
entry kernel's SASS tracks the stack pointer `R1` relative to its value at
entry (`c[0x0][0x28]`), the registers that hold zero, and the frame bytes
that unpredicated `STL` stores of zero have written on every path. The
frame's zeroed bytes, those of every such store, must be exactly the bytes
of an aligned, disjoint placement of the entry kernel's local variables in
the frame, the local memory per thread of the loaded function. A local load
(`LDL`) at `[R1+imm]` must find the zeroed bytes among those it reads
written; any other local load, every generic load or atomic, and every call
must find all of them written. The loads of the registers that the compiler
spills read frame bytes outside the variables and need no zero store. A
device function with local variables also has a prologue (the guard pass
keeps the device functions that it inlines); the check reads the entry
kernel's SASS, so a `CALL` in the entry kernel's SASS fails such a text.

Outside the check: that the driver runs the cubin that it writes to its JIT
cache, of which the check compares the function's attributes; the
hardware's execution of the SASS; the placement of each named variable,
which the check takes as one that the compiler could choose; and the sampled
sizes of a loop's count. The interpreter models the integer instructions
that the prologues use; any other instruction makes the registers and
predicates it writes unknown, which can fail a kernel and never passes one.

Run from the repository root, on a GPU that is free of timed runs:

    CUDA_VISIBLE_DEVICES=<gpu> taskset -c <cores> uv run python -m eval.rq1_safety.sass_prologue \\
        --registration eval/records/registration_<date>.json \\
        --guarded eval/records/guarded_<date>.json \\
        --nvdisasm <nvdisasm of the same or a later CUDA> \\
        --output eval/records/sass_prologue_<date>.json --workers 8
"""

import argparse
import json
import multiprocessing
import os
import re
import struct
import subprocess
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict, assert_never

from eval import ROOT
from eval.checkpoint import Checkpoint, _publish, read_artifact, snapshot_inputs
from eval.rq2_coverage.guarded import KIND as GUARDED_KIND
from opgate.binding import Binding
from opgate.executor.driver import call, driver, integer
from opgate.gate import zeroed
from opgate.hosts.service.audit import parse_binding

__all__ = [
    "ARCH",
    "JIT_LIBRARY",
    "Attributes",
    "Compiled",
    "Launch",
    "Result",
    "Variable",
    "check",
    "compile_sass",
    "layout",
    "main",
    "measure",
    "parse_listing",
    "prologue_of",
]

KIND = "zeroing prologue in the SASS that the driver's JIT compiles from each deployed text"
ARCH = "sm_86"
WORKERS = 8
PARAM_BASE = 0x160
"""The offset of the first kernel parameter in constant bank 0 on sm_86."""
STACK = (0, 0x28)
"""The constant-bank word that holds the stack pointer at entry on sm_86."""
SHARED_LIMIT = 101376
"""The most shared memory of one block on sm_86 (99 KB)."""
PATHS = 64  # the most paths of one thread before its barrier
STEPS = 2_000_000  # the most instructions that one thread's paths run before the barrier
TIMEOUT = 600.0  # seconds of one nvdisasm run
SPAWN = multiprocessing.get_context("spawn")  # fresh processes, which load with an empty JIT cache

_MASK = 0xFFFFFFFF
_RZ = 255
_URZ = 63
_PT = 7

# ------------------------------------------------------------------
# The text: declarations and the PTX prologue.


class Variable(NamedTuple):
    name: str
    align: int
    bytes: int  # 0 for an array of unknown size, the dynamic shared memory


class Loop(NamedTuple):
    """A loop of the prologue that clears a dynamic shared variable: thread `t`
    stores `width` bytes at elements `t, t + N, ...` below the count, which is
    `constant` plus each coefficient times the 32-bit parameter's value."""

    name: str
    width: int
    constant: int
    terms: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Prologue:
    """What the PTX prologue of the entry kernel clears: the bytes of each shared
    variable (offsets in it) cleared in rounds, the loops, and the bytes of each
    local variable; and how many device functions have a prologue of their
    local variables."""

    shared: tuple[Variable, ...]
    rounds: dict[str, frozenset[int]]
    loops: tuple[Loop, ...]
    local: tuple[Variable, ...]
    local_bytes: dict[str, frozenset[int]]
    parameters: dict[str, int]  # the offset of each entry parameter
    functions: int


_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_DECLARATION = (
    r"\.{space}\s+\.align\s+(\d+)\s+\.[bsuf](8|16|32|64)\s+([A-Za-z_$][\w$]*)"
    r"\s*(?:\[\s*(\d*)\s*\])?\s*;"
)
_SHARED = re.compile(_DECLARATION.format(space="shared"))
_LOCAL = re.compile(_DECLARATION.format(space="local"))
_PARAMETER = re.compile(
    r"\.param\s+(?:\.align\s+(\d+)\s+)?\.([bsuf](?:8|16|32|64))\s+"
    r"(?:\.ptr\s+(?:\.(?:global|shared|local|const)\s+)?(?:\.align\s+\d+\s+)?)?"
    r"([A-Za-z_$][\w$]*)\s*(?:\[\s*(\d+)\s*\])?\s*$"
)


def _variables(pattern: re.Pattern[str], text: str) -> tuple[Variable, ...]:
    result = []
    for found in pattern.finditer(text):
        align, bits, name, count = found.groups()
        elements = 1 if count is None else int(count or "0")
        result.append(Variable(name, int(align), int(bits) // 8 * elements))
    return tuple(result)


def _closing(text: str, at: int, opening: str, closing: str) -> int:
    depth = 0
    for position in range(at, len(text)):
        if text[position] == opening:
            depth += 1
        elif text[position] == closing:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError(f"the {opening} at {at} does not close")


def _parameters(declarations: str) -> dict[str, int]:
    """The offset of each parameter in the parameter space, in declaration order."""
    offsets: dict[str, int] = {}
    at = 0
    for declaration in (d.strip() for d in declarations.split(",") if d.strip()):
        found = _PARAMETER.fullmatch(declaration)
        if found is None:
            raise ValueError(f"an entry parameter that the check does not read: {declaration}")
        align, kind, name, count = found.groups()
        width = int(kind[1:]) // 8
        alignment = int(align) if align is not None else width
        at = -(-at // alignment) * alignment
        offsets[name] = at
        at += width * (int(count) if count is not None else 1)
    return offsets


_ROUND_BASE = re.compile(r"mov\.u32 %opgate_z1, ([A-Za-z_$][\w$]*);")
_ROUND_WIDTH = re.compile(r"mad\.lo\.u32 %opgate_z1, %opgate_z0, (\d+), %opgate_z1;")
_LOOP_WIDTH = re.compile(r"mad\.lo\.u32 %opgate_z1, %opgate_d1, (\d+), %opgate_z1;")
_GUARD = re.compile(r"setp\.lt\.u32 %opgate_p, %opgate_z0, (\d+);")
_SHARED_STORE = re.compile(
    r"(@%opgate_p )?st\.shared\.u(8|16|32) \[%opgate_z1(?:\+(\d+))?\], %opgate_z2;"
)
_LOCAL_STORE = re.compile(
    r"st\.local(?:\.v(2|4))?\.u(8|16|32) \[([A-Za-z_$][\w$]*)\+(\d+)\], "
    r"(?:%opgate_z2|\{%opgate_z2(?:, %opgate_z2)+\});"
)
_COUNT = re.compile(r"mov\.u32 %opgate_d0, (\d+);")
_COUNT_PARAMETER = re.compile(r"ld\.param\.u32 %opgate_d1, \[([A-Za-z_$][\w$]*)\];")
_COUNT_TERM = re.compile(r"mad\.lo\.u32 %opgate_d0, %opgate_d1, (\d+), %opgate_d0;")
_IGNORED = re.compile(
    r"\.reg \.(b32|pred) \t%opgate_[zpd](<\d>)?;|mov\.u32 %opgate_z2, 0;"
    r"|mov\.u32 %opgate_z[01], %tid\.[xyz];"
    r"|mad\.lo\.u32 %opgate_z0, %opgate_z0, \d+, %opgate_z1;"
    r"|mov\.u32 %opgate_d1, %opgate_z0;|setp\.ge\.u32 %opgate_p, %opgate_d1, %opgate_d0;"
    r"|@%opgate_p bra \$opgate_zeroed_\d+;|add\.u32 %opgate_d1, %opgate_d1, \d+;"
    r"|bra \$opgate_zero_\d+;|\$opgate_zeroed_\d+:"
)


def prologue_of(text: str, kernel: str, threads: int) -> Prologue:
    """Read the declarations and the zeroing prologue of entry kernel `kernel`
    from `text`, for a block of `threads` threads."""
    masked = _COMMENT.sub(lambda found: re.sub(r"[^\n]", " ", found.group(0)), text)
    entries = list(re.finditer(rf"\.entry\s+{re.escape(kernel)}\s*\(", masked))
    if len(entries) != 1:
        raise ValueError(f"the module has {len(entries)} entry kernels named {kernel}")
    closed = _closing(masked, entries[0].end() - 1, "(", ")")
    parameters = _parameters(masked[entries[0].end() : closed])
    opened = masked.index("{", closed)
    body = masked[opened : _closing(masked, opened, "{", "}") + 1]
    functions = 0
    for found in re.finditer(r"\.func\b", masked):
        at = masked.find("{", found.end())
        semicolon = masked.find(";", found.end())
        if at != -1 and (semicolon == -1 or at < semicolon or "(" in masked[found.end() : at]):
            functions += "%opgate_z" in masked[at : _closing(masked, at, "{", "}") + 1]
    shared = _variables(_SHARED, masked)
    local = _variables(_LOCAL, body)
    rounds: dict[str, set[int]] = {}
    local_bytes: dict[str, set[int]] = {}
    loops: list[Loop] = []
    name: str | None = None
    width: int | None = None
    left: int | None = None
    constant: int | None = None
    count_parameter: str | None = None
    terms: list[tuple[str, int]] = []
    loop_name: str | None = None
    for line in (s.strip() for s in body.splitlines()):
        if "%opgate_" not in line and "$opgate_" not in line:
            continue
        if _IGNORED.fullmatch(line):
            continue
        if (found := _ROUND_BASE.fullmatch(line)) is not None:
            name, width, left = found.group(1), None, None
        elif (found := _ROUND_WIDTH.fullmatch(line)) is not None:
            width = int(found.group(1))
        elif (found := _LOOP_WIDTH.fullmatch(line)) is not None:
            width, loop_name = int(found.group(1)), name
        elif (found := _GUARD.fullmatch(line)) is not None:
            left = int(found.group(1))
        elif (found := _COUNT.fullmatch(line)) is not None:
            constant, terms = int(found.group(1)), []
        elif (found := _COUNT_PARAMETER.fullmatch(line)) is not None:
            count_parameter = found.group(1)
        elif (found := _COUNT_TERM.fullmatch(line)) is not None:
            if count_parameter is None:
                raise ValueError(f"{kernel}: a loop term without its parameter: {line}")
            terms.append((count_parameter, int(found.group(1))))
            count_parameter = None
        elif re.fullmatch(r"\$opgate_zero_\d+:", line):
            name = None
        elif (found := _SHARED_STORE.fullmatch(line)) is not None:
            guarded, bits, offset = found.groups()
            if name is None or width is None or width != int(bits) // 8:
                raise ValueError(f"{kernel}: a prologue store without its address: {line}")
            if loop_name == name and offset is None:
                if constant is None:
                    raise ValueError(f"{kernel}: a loop without its count: {line}")
                loops.append(Loop(name, width, constant, tuple(terms)))
                loop_name, constant = None, None
                continue
            if guarded is not None and left is None:
                raise ValueError(f"{kernel}: a guarded store without its guard: {line}")
            stores = threads if guarded is None else left
            assert stores is not None
            base = int(offset or "0")
            bytes_ = rounds.setdefault(name, set())
            for t in range(stores):
                bytes_.update(range(base + t * width, base + (t + 1) * width))
        elif (found := _LOCAL_STORE.fullmatch(line)) is not None:
            vector, bits, variable, offset = found.groups()
            size = int(bits) // 8 * (int(vector) if vector else 1)
            local_bytes.setdefault(variable, set()).update(range(int(offset), int(offset) + size))
        else:
            raise ValueError(f"{kernel}: a prologue line that the check does not read: {line}")
    return Prologue(
        shared=shared,
        rounds={k: frozenset(v) for k, v in rounds.items()},
        loops=tuple(loops),
        local=local,
        local_bytes={k: frozenset(v) for k, v in local_bytes.items()},
        parameters=parameters,
        functions=functions,
    )


# ------------------------------------------------------------------
# SASS listings.


@dataclass(frozen=True)
class Reg:
    index: int  # _RZ or _URZ for the zero register
    uniform: bool
    negate: bool = False
    invert: bool = False
    absolute: bool = False
    scale: int = 1


@dataclass(frozen=True)
class Pred:
    index: int  # _PT for the true predicate
    uniform: bool
    negated: bool = False


@dataclass(frozen=True)
class Imm:
    value: int


@dataclass(frozen=True)
class Const:
    bank: int
    offset: int


@dataclass(frozen=True)
class Special:
    name: str


@dataclass(frozen=True)
class Address:
    terms: tuple["Reg | Imm", ...]


@dataclass(frozen=True)
class Label:
    name: str


@dataclass(frozen=True)
class Other:
    text: str


Operand = Reg | Pred | Imm | Const | Special | Address | Label | Other


class Instruction(NamedTuple):
    address: int
    guard: Pred | None
    opcode: str
    operands: tuple[Operand, ...]
    text: str


@dataclass(frozen=True)
class Listing:
    instructions: tuple[Instruction, ...]
    labels: dict[str, int]  # the index of the instruction that each label precedes


_LINE = re.compile(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*(?:/\*.*\*/\s*)?$")
_LABEL_LINE = re.compile(r"\s*([.$\w]+):\s*$")
_REGISTER = re.compile(r"(-|~)?(\|)?(U?R)(\d+|Z)(?:\.(?:reuse|X(\d+)|U32|64))*(\|)?")
_PREDICATE = re.compile(r"(!)?(U?P)(\d|T)")
_IMMEDIATE = re.compile(r"-?0x[0-9a-f]+|-?\d+")
_CONSTANT = re.compile(r"c\[(0x[0-9a-f]+)\]\[(0x[0-9a-f]+)\]")


def _operand(text: str) -> Operand:
    text = text.strip()
    if (found := _REGISTER.fullmatch(text)) is not None:
        sign, bar, kind, number, scale, _ = found.groups()
        uniform = kind == "UR"
        index = (_URZ if uniform else _RZ) if number == "Z" else int(number)
        return Reg(
            index,
            uniform,
            negate=sign == "-",
            invert=sign == "~",
            absolute=bar is not None,
            scale=int(scale) if scale else 1,
        )
    if (found := _PREDICATE.fullmatch(text)) is not None:
        negated, kind, number = found.groups()
        return Pred(_PT if number == "T" else int(number), kind == "UP", negated is not None)
    if _IMMEDIATE.fullmatch(text):
        return Imm(int(text, 0) & _MASK)
    if (found := _CONSTANT.fullmatch(text)) is not None:
        return Const(int(found.group(1), 16), int(found.group(2), 16))
    if text.startswith("SR_") or text == "SRZ":
        return Special(text)
    if text.startswith("[") and text.endswith("]"):
        terms: list[Reg | Imm] = []
        for part in re.split(r"\+(?=[^+])", text[1:-1].replace("+-", "+-")):
            term = _operand(part)
            if not isinstance(term, Reg | Imm):
                return Other(text)
            terms.append(term)
        return Address(tuple(terms))
    if (found := re.fullmatch(r"`\(([.$\w]+)\)", text)) is not None:
        return Label(found.group(1))
    return Other(text)


def parse_listing(listing: str, kernel: str) -> Listing:
    """The instructions of entry kernel `kernel` in an `nvdisasm -c` listing."""
    section = f".text.{kernel}"
    lines = listing.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == f"{section}:"]
    if len(starts) != 1:
        raise ValueError(f"the listing has {len(starts)} sections {section}")
    instructions: list[Instruction] = []
    labels: dict[str, int] = {}
    for line in lines[starts[0] + 1 :]:
        if line.startswith("//---"):
            break
        if (label := _LABEL_LINE.fullmatch(line)) is not None:
            labels[label.group(1)] = len(instructions)
            continue
        found = _LINE.fullmatch(line)
        if found is None:
            if line.strip() and not line.strip().startswith("."):
                raise ValueError(f"a line of the listing that the check does not read: {line}")
            continue
        address, body = int(found.group(1), 16), found.group(2)
        guard: Pred | None = None
        if body.startswith("@"):
            head, body = body.split(None, 1)
            parsed = _operand(head[1:])
            if not isinstance(parsed, Pred):
                raise ValueError(f"a guard that the check does not read: {line}")
            guard = parsed
        opcode, _, rest = body.partition(" ")
        operands = tuple(_operand(o) for o in rest.split(",")) if rest.strip() else ()
        instructions.append(Instruction(address, guard, opcode, operands, body))
    return Listing(tuple(instructions), labels)


# ------------------------------------------------------------------
# The interpreter of one thread.


@dataclass
class _Machine:
    tid: tuple[int, int, int]
    constants: dict[tuple[int, int], int]
    r: list[int | None] = field(default_factory=lambda: [None] * 256)
    ur: list[int | None] = field(default_factory=lambda: [None] * 64)
    p: list[bool | None] = field(default_factory=lambda: [None] * 8)
    up: list[bool | None] = field(default_factory=lambda: [None] * 8)

    def copy(self) -> "_Machine":
        return _Machine(self.tid, self.constants, self.r[:], self.ur[:], self.p[:], self.up[:])

    def value(self, operand: Operand) -> int | None:
        match operand:
            case Reg(index=index, uniform=uniform, negate=negate, invert=invert, absolute=absolute):
                if index == (_URZ if uniform else _RZ):
                    v: int | None = 0
                else:
                    v = (self.ur if uniform else self.r)[index]
                if v is None:
                    return None
                if absolute:
                    v = abs(v - (1 << 32) if v & 0x80000000 else v) & _MASK
                if negate:
                    v = -v & _MASK
                if invert:
                    v = ~v & _MASK
                return v
            case Imm(value=value):
                return value
            case Const(bank=bank, offset=offset):
                return self.constants.get((bank, offset))
            case Pred() | Special() | Address() | Label() | Other():
                return None
            case _ as never:
                assert_never(never)

    def predicate(self, operand: Operand) -> bool | None:
        if not isinstance(operand, Pred):
            return None
        if operand.index == _PT:
            v: bool | None = True
        else:
            v = (self.up if operand.uniform else self.p)[operand.index]
        return None if v is None else v != operand.negated

    def write(self, operand: Operand, value: int | None) -> None:
        if isinstance(operand, Reg) and operand.index != (_URZ if operand.uniform else _RZ):
            (self.ur if operand.uniform else self.r)[operand.index] = (
                None if value is None else value & _MASK
            )

    def write_predicate(self, operand: Operand, value: bool | None) -> None:
        if isinstance(operand, Pred) and operand.index != _PT:
            (self.up if operand.uniform else self.p)[operand.index] = value

    def address(self, operand: Operand) -> int | None:
        if not isinstance(operand, Address):
            return None
        total = 0
        for term in operand.terms:
            v = self.value(term)
            if v is None:
                return None
            total += v * (term.scale if isinstance(term, Reg) else 1)
        return total & 0xFFFFFF

    def forget(self, ins: Instruction) -> None:
        """Make what an instruction that the interpreter does not model writes unknown:
        the registers from its first register operand (`_written`), and every
        predicate that it names."""
        first = next((o for o in ins.operands if not isinstance(o, Pred)), None)
        if isinstance(first, Reg):
            registers = self.ur if first.uniform else self.r
            zero = _URZ if first.uniform else _RZ
            for k in range(first.index, min(first.index + _written(ins), zero)):
                registers[k] = None
        self.forget_predicates(ins.operands)
        for operand in ins.operands:
            if isinstance(operand, Other) and operand.text == "PR":
                self.p = [None] * 8
            elif isinstance(operand, Other) and operand.text == "UPR":
                self.up = [None] * 8

    def forget_predicates(self, operands: Sequence[Operand]) -> None:
        for operand in operands:
            if isinstance(operand, Pred):
                self.write_predicate(operand, None)


_CONVERSIONS = frozenset({"F2F", "I2F", "F2I", "I2I", "FRND", "F2IP", "I2FP"})
_DOUBLE = frozenset({"DADD", "DMUL", "DFMA", "DMNMX", "DSETP"})


def _written(ins: Instruction) -> int:
    """How many registers an instruction writes from its first register operand,
    or at least as many."""
    parts = ins.opcode.split(".")
    base, mods = parts[0].removeprefix("U"), set(parts[1:])
    if base in ("HMMA", "IMMA", "DMMA", "BMMA"):
        return 256
    if "128" in mods or (base == "LDSM" and "X4" in mods):
        return 4
    if "64" in mods or "WIDE" in mods or (base == "LDSM" and "X2" in mods) or base in _DOUBLE:
        return 2
    if base in _CONVERSIONS and mods & {"F64", "U64", "S64"}:
        return 2
    if base == "CS2R" and "32" not in mods:
        return 2
    return 1


def _lut(lut: int, a: int, b: int, c: int) -> int:
    result = 0
    for i in range(8):
        if lut >> i & 1:
            result |= (a if i & 4 else ~a) & (b if i & 2 else ~b) & (c if i & 1 else ~c) & _MASK
    return result


def _signed(v: int) -> int:
    return v - (1 << 32) if v & 0x80000000 else v


def _compare(cmp: str, a: int, b: int, signed: bool) -> bool:
    if signed:
        a, b = _signed(a), _signed(b)
    match cmp:
        case "EQ":
            return a == b
        case "NE":
            return a != b
        case "LT":
            return a < b
        case "LE":
            return a <= b
        case "GT":
            return a > b
        case "GE":
            return a >= b
        case _:
            raise ValueError(f"a comparison that the check does not read: {cmp}")


def _combine(op: str, a: bool, b: bool) -> bool:
    match op:
        case "AND":
            return a and b
        case "OR":
            return a or b
        case "XOR":
            return a != b
        case _:
            raise ValueError(f"a predicate operation that the check does not read: {op}")


def _execute(m: _Machine, ins: Instruction) -> None:
    """Run one unconditional instruction that is not a store, a load or a branch."""
    parts = ins.opcode.split(".")
    base, mods = parts[0].removeprefix("U") if parts[0] != "UMOV" else "MOV", parts[1:]
    ops = ins.operands
    v = m.value
    if base == "MOV" and len(ops) >= 2 and (len(ops) == 2 or ops[2] == Imm(0xF)):
        m.write(ops[0], v(ops[1]))
    elif base == "IMAD" and not {"HI", "WIDE", "X"} & set(mods) and len(ops) == 4:
        a, b, c = v(ops[1]), v(ops[2]), v(ops[3])
        m.write(ops[0], None if a is None or b is None or c is None else a * b + c)
    elif base == "IADD3" and "X" not in mods:
        a, b, c = v(ops[-3]), v(ops[-2]), v(ops[-1])
        m.forget_predicates(ops[1:-3])
        m.write(ops[0], None if a is None or b is None or c is None else a + b + c)
    elif base == "LEA" and not {"HI", "X"} & set(mods):
        a, b, s = v(ops[-3]), v(ops[-2]), v(ops[-1])
        m.forget_predicates(ops[1:-3])
        m.write(ops[0], None if a is None or b is None or s is None else (a << s) + b)
    elif base == "LOP3" and mods == ["LUT"] and len(ops) >= 6:
        a, b, c, lut = v(ops[-5]), v(ops[-4]), v(ops[-3]), v(ops[-2])
        result = None if a is None or b is None or c is None or lut is None else _lut(lut, a, b, c)
        m.forget_predicates(ops[:-5])
        for dest in ops[:-5]:
            if isinstance(dest, Reg):
                m.write(dest, result)
            elif ops[-1] == Pred(_PT, isinstance(dest, Pred) and dest.uniform, True):
                m.write_predicate(dest, None if result is None else result != 0)
    elif base == "SHF" and len(ops) == 4 and mods[0] in ("L", "R"):
        a, s, b = v(ops[1]), v(ops[2]), v(ops[3])
        wide = "U64" in mods or "S64" in mods
        signed = "S32" in mods or "S64" in mods
        if a is None or s is None or b is None or (not wide and s > 31) or s > 63:
            m.write(ops[0], None)
        else:
            x = (_signed(b) << 32 if signed else b << 32) | a
            y = x << s if mods[0] == "L" else x >> s
            m.write(ops[0], (y >> 32 if "HI" in mods else y) & _MASK)
    elif base == "SEL" and len(ops) == 4:
        a, b, q = v(ops[1]), v(ops[2]), m.predicate(ops[3])
        m.write(ops[0], None if q is None else a if q else b)
    elif base == "ISETP" and "EX" not in mods and len(ops) == 5 and len(mods) >= 2:
        cmp, op = mods[0], mods[-1]
        a, b, c = v(ops[2]), v(ops[3]), m.predicate(ops[4])
        if a is None or b is None or c is None:
            m.write_predicate(ops[0], None)
            m.write_predicate(ops[1], None)
        else:
            result = _compare(cmp, a, b, "U32" not in mods)
            m.write_predicate(ops[0], _combine(op, result, c))
            m.write_predicate(ops[1], _combine(op, not result, c))
    elif base == "PLOP3" and mods == ["LUT"] and len(ops) == 7:
        a, b, c = m.predicate(ops[2]), m.predicate(ops[3]), m.predicate(ops[4])
        lut, lut2 = v(ops[5]), v(ops[6])
        if a is None or b is None or c is None or lut is None or lut2 is None:
            m.write_predicate(ops[0], None)
            m.write_predicate(ops[1], None)
        else:
            index = 4 * a + 2 * b + c
            m.write_predicate(ops[0], bool(lut >> index & 1))
            m.write_predicate(ops[1], bool(lut2 >> index & 1))
    elif base == "PRMT" and not mods and len(ops) == 4:
        a, selector, b = v(ops[1]), v(ops[2]), v(ops[3])
        if a is None or selector is None or b is None:
            m.write(ops[0], None)
        else:
            source = a | b << 32
            result = 0
            for k in range(4):
                nibble = selector >> 4 * k & 0xF
                byte = source >> 8 * (nibble & 7) & 0xFF
                if nibble & 8:
                    byte = 0xFF if byte & 0x80 else 0
                result |= byte << 8 * k
            m.write(ops[0], result)
    elif (
        base == "LDC"
        and mods in ([], ["64"])
        and len(ops) == 2
        and isinstance(ops[0], Reg)
        and isinstance(ops[1], Const)
    ):
        # `LDC` or `ULDC` of a word of a constant bank, or of two words into a
        # register pair with `.64`
        for k in range(2 if mods else 1):
            word = m.value(Const(ops[1].bank, ops[1].offset + 4 * k))
            m.write(Reg(ops[0].index + k, ops[0].uniform), word)
    elif ins.opcode == "S2R" and len(ops) == 2 and isinstance(ops[1], Special):
        axes = {"SR_TID.X": 0, "SR_TID.Y": 1, "SR_TID.Z": 2}
        axis = axes.get(ops[1].name)
        m.write(ops[0], None if axis is None else m.tid[axis])
    elif ins.opcode in ("CS2R", "CS2R.32") and len(ops) == 2 and isinstance(ops[0], Reg):
        zero = 0 if ops[1] == Special("SRZ") else None
        m.write(ops[0], zero)
        if ins.opcode == "CS2R" and ops[0].index + 1 < (_URZ if ops[0].uniform else _RZ):
            m.write(Reg(ops[0].index + 1, ops[0].uniform), zero)
    elif base in _NO_WRITE:
        pass
    else:
        m.forget(ins)


_NO_WRITE = frozenset(
    {"NOP", "BSSY", "BSYNC", "WARPSYNC", "YIELD", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL", "BREAK"}
)
_CONTROL = frozenset(
    {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BPT", "KILL", "JMXU", "BRXU"}
)
_SHARED_READS = frozenset({"LDS", "LDSM", "ATOMS", "LD", "ATOM"})
_LOCAL_READS = frozenset({"LDL", "LD", "ATOM"})
_SHARED_WRITES = frozenset({"STS", "LDGSTS", "ATOMS", "ST", "RED", "ATOM"})


def _width(opcode: str) -> int:
    """The bytes of a store."""
    mods = opcode.split(".")[1:]
    for mod, width in (("U8", 1), ("S8", 1), ("U16", 2), ("S16", 2), ("64", 8), ("128", 16)):
        if mod in mods:
            return width
    return 4


def _zero_data(m: _Machine, operand: Operand, width: int) -> bool:
    if not isinstance(operand, Reg) or operand.negate or operand.invert:
        return False
    if operand.index == (_URZ if operand.uniform else _RZ):
        return True
    registers = max(1, width // 4)
    return all(m.value(Reg(operand.index + k, operand.uniform)) == 0 for k in range(registers))


def _barrier(ins: Instruction) -> bool:
    return (
        ins.opcode in ("BAR.SYNC", "BAR.SYNC.DEFER_BLOCKING")
        and ins.guard is None
        and ins.operands == (Imm(0),)
    )


def _thread(listing: Listing, machine: _Machine) -> frozenset[int] | str:
    """The shared bytes that the thread stores zero to on every path before its first
    barrier, or why a path fails."""
    stored: frozenset[int] | None = None
    work: list[tuple[int, _Machine, frozenset[int]]] = [(0, machine, frozenset())]
    paths, steps = 1, 0
    instructions = listing.instructions
    while work:
        index, m, bytes_ = work.pop()
        while True:
            steps += 1
            if steps > STEPS:
                return f"a thread runs more than {STEPS} instructions before the barrier"
            if index >= len(instructions):
                return "a path runs past the end of the kernel"
            ins = instructions[index]
            base = ins.opcode.split(".")[0]
            guard = True if ins.guard is None else m.predicate(ins.guard)
            if base == "BAR":
                if not _barrier(ins):
                    return f"the first barrier of a path is `{ins.text}`"
                stored = bytes_ if stored is None else stored & bytes_
                break
            if base in _SHARED_READS and guard is not False:
                return f"a load or atomic before the barrier at {ins.address:#x}: `{ins.text}`"
            if base in _SHARED_WRITES:
                if guard is False:
                    index += 1
                    continue
                width = _width(ins.opcode)
                zero = base == "STS" and len(ins.operands) == 2
                zero = zero and _zero_data(m, ins.operands[1], width)
                if not zero:
                    return f"a store other than zero before the barrier at {ins.address:#x}: `{ins.text}`"
                address = m.address(ins.operands[0])
                if guard is True and address is not None:
                    bytes_ = bytes_ | frozenset(range(address, address + width))
                index += 1
                continue
            if base in _CONTROL:
                if guard is False:
                    index += 1
                    continue
                if base != "BRA":
                    return f"a path reaches `{ins.text}` at {ins.address:#x} before the barrier"
                targets = [o for o in ins.operands if isinstance(o, Label)]
                conditions = [o for o in ins.operands if not isinstance(o, Label)]
                if len(targets) != 1:
                    return (
                        f"a branch that the check does not read at {ins.address:#x}: `{ins.text}`"
                    )
                taken = guard
                for condition in conditions:
                    q = m.predicate(condition)
                    taken = None if taken is None or q is None else taken and q
                target = listing.labels[targets[0].name]
                if taken is True:
                    index = target
                elif taken is None:
                    paths += 1
                    if paths > PATHS:
                        return f"a thread has more than {PATHS} paths before the barrier"
                    work.append((target, m.copy(), bytes_))
                    index += 1
                else:
                    index += 1
                continue
            if guard is True:
                _execute(m, ins)
            elif guard is None:
                m.forget(ins)
            index += 1
    assert stored is not None
    return stored


def _runs(bytes_: frozenset[int]) -> list[list[int]]:
    """The maximal intervals `[start, end)` of a set of bytes, in order."""
    runs: list[list[int]] = []
    for b in sorted(bytes_):
        if runs and runs[-1][1] == b:
            runs[-1][1] = b + 1
        else:
            runs.append([b, b + 1])
    return runs


def layout(
    covered: frozenset[int], variables: Sequence[Variable], size: int
) -> list[list[int]] | None:
    """An aligned, disjoint placement of `variables` in `[0, size)` whose bytes are
    exactly `covered`, as `[offset, bytes]` of each variable in order, or None."""
    runs = _runs(covered)
    if any(v.bytes <= 0 for v in variables) or (runs and runs[-1][1] > size):
        return None
    if sum(v.bytes for v in variables) != len(covered):
        return None
    order = sorted(range(len(variables)), key=lambda k: (-variables[k].bytes, k))
    placed: dict[int, int] = {}

    def fill(run: int, at: int) -> bool:
        if run == len(runs):
            return len(placed) == len(variables)
        if at == runs[run][1]:
            return fill(run + 1, runs[run + 1][0] if run + 1 < len(runs) else 0)
        tried: set[tuple[int, int]] = set()
        for k in order:
            v = variables[k]
            if k in placed or (v.align, v.bytes) in tried:
                continue
            tried.add((v.align, v.bytes))
            if at % v.align == 0 and at + v.bytes <= runs[run][1]:
                placed[k] = at
                if fill(run, at + v.bytes):
                    return True
                del placed[k]
        return False

    if not fill(0, runs[0][0] if runs else 0):
        return None
    return [[placed[k], variables[k].bytes] for k in range(len(variables))]


# ------------------------------------------------------------------
# The checks.


class Launch(NamedTuple):
    """What the check reads of a binding: the entry kernel, its block, and the
    range of the size that each parameter holds."""

    kernel: str
    block: tuple[int, int, int]
    ranges: dict[str, tuple[int, int]]


class SharedCheck(TypedDict):
    values: dict[str, int]  # the loop parameters' values, empty without a loop
    covered: list[list[int]]  # the zeroed bytes as intervals [start, end)
    layout: list[list[int]] | None  # [offset, bytes] of each static variable
    dynamic: list[int]  # the dynamic shared bytes that the prologue clears, [start, end)


class LocalCheck(TypedDict):
    covered: list[list[int]]  # the frame bytes that zero stores write, [start, end)
    layout: list[list[int]] | None  # [offset, bytes] of each local variable
    reads: int  # the local loads, generic loads and calls that the analysis reached


class Result(TypedDict):
    shared: list[SharedCheck]
    local: LocalCheck | None
    failures: list[str]


SIZES = (0, 1, 2, 3, 4, 5, 7, 8)
"""The offsets of the sampled loop sizes from 0, from each multiple of the
block's threads up to eight, and below the range's maximum."""


def _sizes(threads: int, lo: int, hi: int) -> list[int]:
    candidates = {lo, hi}
    for multiple in range(9):
        for offset in SIZES:
            candidates.update((multiple * threads + offset, multiple * threads - offset))
    candidates.update(hi - offset for offset in SIZES)
    return sorted(s for s in candidates if lo <= s <= hi)


def _shared(
    listing: Listing, prologue: Prologue, launch: Launch, static: int
) -> tuple[list[SharedCheck], list[str]]:
    bx, by, bz = launch.block
    threads = bx * by * bz
    static_variables = [v for v in prologue.shared if v.bytes > 0]
    dynamic_variables = [v for v in prologue.shared if v.bytes == 0]
    for v in static_variables:
        if prologue.rounds.get(v.name) != frozenset(range(v.bytes)):
            raise ValueError(f"{launch.kernel}: the prologue does not clear {v.name} whole")
    dynamic_bytes = 0
    for v in dynamic_variables:
        cleared = prologue.rounds.get(v.name, frozenset())
        if cleared != frozenset(range(len(cleared))):
            raise ValueError(f"{launch.kernel}: the prologue clears part of {v.name}")
        dynamic_bytes = max(dynamic_bytes, len(cleared))
    parameters = sorted({p for loop in prologue.loops for p, _ in loop.terms})
    if parameters:
        ranges = [launch.ranges[p] for p in parameters]
        lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
        samples: list[dict[str, int]] = [
            dict.fromkeys(parameters, s) for s in _sizes(threads, lo, hi)
        ]
    else:
        samples = [{}]
    checks: list[SharedCheck] = []
    failures: list[str] = []
    for values in samples:
        cleared = dynamic_bytes
        for loop in prologue.loops:
            count = loop.constant + sum(k * values[p] for p, k in loop.terms)
            cleared = max(cleared, loop.width * count)
        if parameters and static + cleared > SHARED_LIMIT:
            continue
        constants = {(0, PARAM_BASE + prologue.parameters[p]): value for p, value in values.items()}
        covered: set[int] = set()
        for t in range(threads):
            tid = (t % bx, t // bx % by, t // (bx * by))
            outcome = _thread(listing, _Machine(tid, constants))
            if isinstance(outcome, str):
                failures.append(f"shared: thread {tid}{_at(values)}: {outcome}")
                return checks, failures
            covered |= outcome
        zeroed = frozenset(covered)
        below = frozenset(b for b in zeroed if b < static)
        dynamic = frozenset(b for b in zeroed if b >= static)
        placement = layout(below, static_variables, static)
        checks.append(
            SharedCheck(
                values=values,
                covered=_runs(zeroed),
                layout=placement,
                dynamic=[static, static + cleared],
            )
        )
        if placement is None:
            failures.append(
                f"shared{_at(values)}: the zeroed bytes below {static} are not a placement "
                f"of the static variables: {_runs(below)}"
            )
        if dynamic != frozenset(range(static, static + cleared)):
            failures.append(
                f"shared{_at(values)}: the zeroed bytes from {static} are {_runs(dynamic)}, "
                f"not [{static}, {static + cleared})"
            )
    return checks, failures


def _at(values: dict[str, int]) -> str:
    return f" at sizes {values}" if values else ""


class _Frame(NamedTuple):
    """The must-state of the local analysis at an instruction."""

    sp: int | None  # R1 minus the stack pointer at entry
    zero: frozenset[int]  # registers that hold zero
    stored: frozenset[int]  # frame bytes, relative to the stack pointer at entry


def _join(a: _Frame, b: _Frame) -> _Frame:
    return _Frame(a.sp if a.sp == b.sp else None, a.zero & b.zero, a.stored & b.stored)


def _successors(listing: Listing, index: int) -> list[int] | str:
    ins = listing.instructions[index]
    base = ins.opcode.split(".")[0]
    conditional = ins.guard is not None
    if base == "BRA":
        targets = [o for o in ins.operands if isinstance(o, Label)]
        if len(targets) != 1:
            return f"a branch that the check does not read at {ins.address:#x}: `{ins.text}`"
        conditional = conditional or len(ins.operands) > 1
        target = listing.labels[targets[0].name]
        return [target, index + 1] if conditional else [target]
    if base in ("EXIT", "RET", "BPT", "KILL"):
        return [index + 1] if conditional else []
    if base in ("BRX", "JMX", "JMP", "BRXU", "JMXU"):
        return f"an indirect jump at {ins.address:#x}: `{ins.text}`"
    return [index + 1]


def _frame_address(operand: Operand, sp: int | None) -> int | None:
    """The address, relative to the stack pointer at entry, of `[R1+imm]`."""
    if sp is None or not isinstance(operand, Address):
        return None
    registers = [t for t in operand.terms if isinstance(t, Reg)]
    if registers != [Reg(1, False)]:
        return None
    offset = sum(t.value for t in operand.terms if isinstance(t, Imm))
    return sp + _signed(offset & _MASK)


def _transfer(ins: Instruction, state: _Frame) -> tuple[_Frame, frozenset[int]]:
    """The state after `ins`, and the frame bytes that it stores zero to."""
    base = ins.opcode.split(".")[0]
    ops = ins.operands
    sp, zero, stored = state
    new: frozenset[int] = frozenset()
    if base == "STL":
        if ins.guard is None and len(ops) == 2 and sp is not None:
            width = _width(ins.opcode)
            address, data = ops
            zeroes = isinstance(data, Reg) and not data.uniform and not data.negate
            zeroes = zeroes and not data.invert
            if zeroes and isinstance(data, Reg) and data.index != _RZ:
                zeroes = all(data.index + k in zero for k in range(max(1, width // 4)))
            start = _frame_address(address, sp)
            if zeroes and start is not None:
                new = frozenset(range(start, start + width))
                stored = stored | new
        return _Frame(sp, zero, stored), new
    if base in _NO_WRITE or base in _CONTROL or base in ("STS", "STG", "ST", "RED", "BAR"):
        return state, new
    dest = next((o for o in ops if not isinstance(o, Pred)), None)
    if not isinstance(dest, Reg) or dest.uniform:
        return state, new
    written = set(range(dest.index, min(dest.index + _written(ins), _RZ)))
    value_zero = False
    if ins.guard is None:
        if ins.opcode in ("CS2R", "CS2R.32") and ops[1:] == (Special("SRZ"),):
            written = {dest.index, dest.index + 1} if ins.opcode == "CS2R" else {dest.index}
            value_zero = True
        elif ins.opcode == "MOV" and ops[1:2] == (Reg(_RZ, False),):
            written, value_zero = {dest.index}, True
        elif ins.opcode in ("IMAD.MOV.U32", "IMAD.MOV") and ops[1:] == (Reg(_RZ, False),) * 3:
            written, value_zero = {dest.index}, True
    zero = frozenset(zero - written) | (frozenset(written) if value_zero else frozenset())
    if 1 in written:
        if ins.guard is None and ins.opcode in ("MOV", "IMAD.MOV.U32") and ops[-1] == Const(*STACK):
            sp = 0
        elif (
            ins.guard is None
            and ins.opcode == "IADD3"
            and len(ops) == 4
            and ops[1] == Reg(1, False)
            and isinstance(ops[2], Imm)
            and ops[3] == Reg(_RZ, False)
            and sp is not None
        ):
            sp = sp + _signed(ops[2].value)
        else:
            sp = None
    return _Frame(sp, zero, stored), new


def _local(listing: Listing, prologue: Prologue, frame: int) -> tuple[LocalCheck, list[str]]:
    for v in prologue.local:
        if v.bytes > 0 and prologue.local_bytes.get(v.name) != frozenset(range(v.bytes)):
            raise ValueError(f"the prologue does not clear the local variable {v.name} whole")
    variables = [v for v in prologue.local if v.bytes > 0]
    instructions = listing.instructions
    states: dict[int, _Frame] = {0: _Frame(None, frozenset(), frozenset())}
    work = [0]
    failures: list[str] = []
    while work:
        index = work.pop()
        if index >= len(instructions):
            return LocalCheck(covered=[], layout=None, reads=0), [
                "local: a path runs past the end of the kernel"
            ]
        out, _ = _transfer(instructions[index], states[index])
        successors = _successors(listing, index)
        if isinstance(successors, str):
            return LocalCheck(covered=[], layout=None, reads=0), [f"local: {successors}"]
        for s in successors:
            joined = out if s not in states else _join(states[s], out)
            if s not in states or joined != states[s]:
                states[s] = joined
                work.append(s)
    zeroed: set[int] = set()
    for index, state in states.items():
        _, new = _transfer(instructions[index], state)
        zeroed |= new
    if not zeroed:
        return LocalCheck(covered=[], layout=None, reads=0), [
            "local: the SASS has no store of zero to the frame"
        ]
    bottom = -frame
    relative = frozenset(b - bottom for b in zeroed)
    placement = layout(relative, variables, frame)
    if placement is None:
        failures.append(
            f"local: the zeroed frame bytes {_runs(relative)} are not a placement of the "
            f"local variables in the frame of {frame} bytes"
        )
    reads = 0
    for index in sorted(states):
        ins = instructions[index]
        base = ins.opcode.split(".")[0]
        if base in _LOCAL_READS or base == "CALL":
            reads += 1
            read = frozenset(zeroed)
            start = _frame_address(ins.operands[1], states[index].sp) if base == "LDL" else None
            if start is not None and len(ins.operands) == 2:
                read = read & frozenset(range(start, start + _width(ins.opcode)))
            missing = read - states[index].stored
            if missing:
                failures.append(
                    f"local: `{ins.text}` at {ins.address:#x} runs before the stores of zero "
                    f"to frame bytes {_runs(frozenset(b - bottom for b in missing))}"
                )
    return LocalCheck(covered=_runs(relative), layout=placement, reads=reads), failures


def check(text: str, launch: Launch, listing: str, static: int, frame: int) -> Result:
    """Check the SASS listing of `text`, whose loaded entry function has `static`
    bytes of static shared memory and `frame` bytes of local memory per thread."""
    bx, by, bz = launch.block
    prologue = prologue_of(text, launch.kernel, bx * by * bz)
    parsed = parse_listing(listing, launch.kernel)
    shared: list[SharedCheck] = []
    local: LocalCheck | None = None
    failures: list[str] = []
    if any(v.bytes > 0 for v in prologue.shared) or prologue.rounds or prologue.loops:
        shared, found = _shared(parsed, prologue, launch, static)
        failures += found
    if any(v.bytes > 0 for v in prologue.local):
        local, found = _local(parsed, prologue, frame)
        failures += found
    if prologue.functions:
        calls = [i for i in parsed.instructions if i.opcode.split(".")[0] == "CALL"]
        if calls:
            failures.append(
                f"{prologue.functions} device functions have a zeroing prologue, and the "
                f"kernel calls at {calls[0].address:#x}: `{calls[0].text}`"
            )
    return Result(shared=shared, local=local, failures=failures)


# ------------------------------------------------------------------
# The experiment.


class Attributes(TypedDict):
    """The attributes of a loaded entry function that the check reads or compares."""

    registers: int
    local_bytes: int  # the local memory of a thread, its frame
    shared_bytes: int  # the static shared memory
    const_bytes: int
    max_threads: int  # the most threads of a block


def _attributes(handle: object, kernel: str) -> Attributes:
    (function,) = call(driver.cuModuleGetFunction(handle, kernel.encode("ascii")))
    attribute = driver.CUfunction_attribute

    def read(name: str) -> int:
        return integer(
            driver.cuFuncGetAttribute(getattr(attribute, f"CU_FUNC_ATTRIBUTE_{name}"), function)
        )

    return Attributes(
        registers=read("NUM_REGS"),
        local_bytes=read("LOCAL_SIZE_BYTES"),
        shared_bytes=read("SHARED_SIZE_BYTES"),
        const_bytes=read("CONST_SIZE_BYTES"),
        max_threads=read("MAX_THREADS_PER_BLOCK"),
    )


class Jit(NamedTuple):
    """What the driver's JIT gives for a text: the cubin of its cache entry, the
    entry's bytes before the PTX, the entry function's attributes, and the path
    of the JIT compiler library that the driver mapped."""

    cubin: bytes
    header: bytes
    attributes: Attributes
    compiler: str


JIT_LIBRARY = "libnvidia-ptxjitcompiler.so"
CUBIN_SIZE = 12
"""The offset in a JIT cache entry of the 32-bit length of its cubin."""


def _compiler() -> str:
    """The path of the driver's JIT compiler library that this process maps."""
    mapped = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if JIT_LIBRARY in line
    }
    if len(mapped) != 1:
        raise RuntimeError(f"the process maps {sorted(mapped)} as the JIT compiler")
    return mapped.pop()


def _architecture(device: object) -> str:
    attribute = driver.CUdevice_attribute
    major = integer(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    )
    minor = integer(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
    )
    return f"sm_{major}{minor}"


def _jit(text: str, kernel: str) -> Jit:
    """Load `text` as `Executor.register` loads it, `cuModuleLoadData` of the text
    and its terminating zero byte on the primary context of device 0, in this
    process, which must not have initialized the driver: the driver takes the
    directory of its JIT cache when it initializes, and here it is an empty
    one, so that the load writes exactly one cache entry, of this text."""
    with tempfile.TemporaryDirectory() as cache:
        os.environ["CUDA_CACHE_PATH"] = cache
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        call(driver.cuInit(0))
        (device,) = call(driver.cuDeviceGet(0))
        if _architecture(device) != ARCH:
            raise RuntimeError(f"the device is {_architecture(device)}, and the check reads {ARCH}")
        (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
        try:
            call(driver.cuCtxSetCurrent(context))
            data = text.encode("ascii") + b"\0"
            (handle,) = call(driver.cuModuleLoadData(data))
            attributes = _attributes(handle, kernel)
            entries = [p for p in Path(cache).rglob("*") if p.is_file() and p.name != "index"]
            if len(entries) != 1:
                raise RuntimeError(f"the load wrote {len(entries)} JIT cache entries")
            entry = entries[0].read_bytes()
            at = entry.find(data)
            if at < 0 or entry.count(data) != 1:
                raise RuntimeError("the JIT cache entry does not hold the loaded text once")
            (size,) = struct.unpack_from("<I", entry, CUBIN_SIZE)
            cubin = entry[len(entry) - size :]
            if len(entry) - size < at + len(data) or not cubin.startswith(b"\x7fELF"):
                raise RuntimeError("the JIT cache entry does not end in a cubin after the text")
            (reloaded,) = call(driver.cuModuleLoadData(cubin))
            if _attributes(reloaded, kernel) != attributes:
                raise RuntimeError(
                    f"the cubin of the JIT cache gives {_attributes(reloaded, kernel)}, "
                    f"and the load of the text {attributes}"
                )
            compiler = _compiler()
            call(driver.cuModuleUnload(reloaded))
            call(driver.cuModuleUnload(handle))
        finally:
            call(driver.cuDevicePrimaryCtxRelease(device))
    return Jit(cubin, entry[:at], attributes, compiler)


class Compiled(NamedTuple):
    jit: Jit
    listing: str  # the `nvdisasm -c` listing of the cubin


def compile_sass(text: str, kernel: str, nvdisasm: Path) -> Compiled:
    """The driver JIT's cubin of `text`, loaded in a fresh process (`_jit`), and
    its `nvdisasm -c` listing."""
    with ProcessPoolExecutor(1, mp_context=SPAWN) as pool:
        jit = pool.submit(_jit, text, kernel).result()
    with tempfile.TemporaryDirectory() as tmp:
        cubin = Path(tmp) / "module.cubin"
        cubin.write_bytes(jit.cubin)
        listing = subprocess.run(
            [str(nvdisasm), "-c", str(cubin)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=True,
        )
    return Compiled(jit, listing.stdout)


Group = Literal["as_written", "with_checks"]
"""The registration's entries that the gate accepts as written, and the guarded
record's entries that it accepts with the guard pass's checks."""


class Kernel(TypedDict):
    id: str
    group: Group
    tasks: str
    verdict: str
    shared: bool  # the text declares a shared variable
    local: bool  # the entry kernel declares a local variable
    kernel: str
    block: list[int]
    ranges: dict[str, list[int]]


class Row(TypedDict):
    id: str
    group: Group
    tasks: str
    verdict: str
    shared: bool
    local: bool
    block: list[int]
    ptx_artifact: str
    sass_artifact: str
    cubin_artifact: str
    cache_header: str  # the JIT cache entry's bytes before the PTX, in hex
    attributes: Attributes
    compiler: str
    result: Result
    passed: bool
    seconds: float


def _ranges(binding: dict[str, object]) -> dict[str, list[int]]:
    """The range of the size that each parameter that holds a size symbol holds."""
    sizes = binding["sizes"]
    parameters = binding["parameters"]
    assert isinstance(sizes, list) and isinstance(parameters, list)
    ranges: dict[str, list[int]] = {}
    for parameter in parameters:
        assert isinstance(parameter, dict)
        value = parameter.get("value")
        if parameter["kind"] == "size" and isinstance(value, dict):
            terms = value["terms"]
            if value["constant"] == 0 and len(terms) == 1 and terms[0][1] == 1:
                size = sizes[terms[0][0]]
                ranges[parameter["name"]] = [size["minimum"], size["maximum"]]
    return ranges


def _text(record: Path, key: str) -> str:
    """The text with its prologue that the archived Rocq query of `key` holds."""
    terms = read_artifact(record, key, "v")
    words = re.search(r"Definition text : array int := \[\|(.*?)\| 0 \|\]\.", terms, re.S)
    size = re.search(r"Definition size : int := (\d+)\.", terms)
    if words is None or size is None:
        raise ValueError(f"{key}: the archived query holds no text")
    data = bytearray()
    for word in words.group(1).split(";"):
        value = int(word)
        data += bytes(value >> (7 * k) & 0x7F for k in range(8))
    length = int(size.group(1))
    if len(data) < length or any(data[length:]) or len(data) - length >= 8 and length > 0:
        raise ValueError(f"{key}: the archived text does not have {length} characters")
    return data[:length].decode("ascii")


def _deployed(
    record: Path, key: str, binding: dict[str, object]
) -> tuple[str, Binding, Prologue] | None:
    """The text with the prologue that the service loads for the archived text
    `key` of `record` and `binding`, the binding, and the prologue, when
    `zeroed` gives the text one; the text must be that of the archived query of
    `key`."""
    text = read_artifact(record, key, "ptx")
    bound = parse_binding(binding)
    deployed = zeroed(text, bound)
    if deployed == text:
        return None
    if _text(record, key) != deployed:
        raise ValueError(f"{key}: the archived query holds another text than the deployed one")
    bx, by, bz = bound.block
    return deployed, bound, prologue_of(deployed, bound.kernel, bx * by * bz)


def _kernel(
    group: Group, entry: dict[str, object], found: tuple[str, Binding, Prologue], record: object
) -> tuple[Kernel, str]:
    """The kernel of the registration's `entry` in `group`, with the binding
    record `record`, and its deployed text."""
    text, binding, prologue = found
    if not isinstance(record, dict):
        raise ValueError(f"{entry['id']}: a binding record that is not an object")
    kernel = Kernel(
        id=str(entry["id"]),
        group=group,
        tasks=str(entry["tasks"]),
        verdict=str(entry["verdict"]),
        shared=bool(prologue.shared),
        local=bool(prologue.local),
        kernel=binding.kernel,
        block=list(binding.block),
        ranges=_ranges(record),
    )
    return kernel, text


def kernels(registration: Path, guarded: Path) -> list[tuple[Kernel, str]]:
    """Each text that the service deploys with a zeroing prologue, with its kernel:
    of the registration's entries that the gate accepts as written, and of the
    guarded record's entries that it accepts with checks."""
    record = json.loads(registration.read_text())
    checked = json.loads(guarded.read_text())
    if record["schema"] != 2 or record["complete"] is not True:
        raise ValueError(f"{registration}: a complete registration record is needed")
    if (
        checked["complete"] is not True
        or checked["provenance"]["kind"] != GUARDED_KIND
        or Path(checked["provenance"]["registration"]).name != registration.name
    ):
        raise ValueError(f"{guarded}: a complete guarded record of {registration} is needed")
    rows = {r["id"]: r for r in record["results"]}
    chosen: list[tuple[Kernel, str]] = []
    for row in sorted(record["results"], key=lambda r: r["id"]):
        if row["accepted"] is True:
            found = _deployed(registration, row["id"], row["binding"])
            if found is not None:
                chosen.append(_kernel("as_written", row, found, row["binding"]))
    for row in sorted(checked["results"], key=lambda r: r["id"]):
        if row["first"] == "refused" and row["decision"] == "accepted":
            entry = rows[row["id"]]
            if entry["accepted"] is not False:
                raise ValueError(f"{row['id']}: accepted as written and with checks")
            found = _deployed(guarded, row["id"] + "/final", row["binding"])
            if found is not None:
                chosen.append(_kernel("with_checks", entry, found, row["binding"]))
    return chosen


def _measured(kernel: Kernel, text: str, nvdisasm: Path) -> tuple[Row, Compiled]:
    started = time.perf_counter()
    compiled = compile_sass(text, kernel["kernel"], nvdisasm)
    bx, by, bz = kernel["block"]
    launch = Launch(
        kernel["kernel"], (bx, by, bz), {k: (v[0], v[1]) for k, v in kernel["ranges"].items()}
    )
    attributes = compiled.jit.attributes
    result = check(
        text, launch, compiled.listing, attributes["shared_bytes"], attributes["local_bytes"]
    )
    seconds = time.perf_counter() - started
    print(f"{kernel['id']}: {len(result['failures'])} failures, {seconds:.1f} s", flush=True)
    row = Row(
        id=kernel["id"],
        group=kernel["group"],
        tasks=kernel["tasks"],
        verdict=kernel["verdict"],
        shared=kernel["shared"],
        local=kernel["local"],
        block=kernel["block"],
        ptx_artifact=kernel["id"],
        sass_artifact=kernel["id"],
        cubin_artifact=kernel["id"],
        cache_header=compiled.jit.header.hex(),
        attributes=attributes,
        compiler=compiled.jit.compiler,
        result=result,
        passed=not result["failures"],
        seconds=seconds,
    )
    return row, compiled


def _artifact(output: Path, key: str, kind: str, data: bytes) -> None:
    """Publish the bytes of a SASS listing or a cubin beside the record's PTX
    artifact of the same key."""
    path = output.with_suffix(output.suffix + ".artifacts") / f"{key}.{kind}"
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"artifact already exists with different contents: {path}")
        return
    _publish(path, data)


def _version(tool: Path) -> str:
    return subprocess.run(
        [str(tool), "--version"], capture_output=True, text=True, check=True, timeout=60
    ).stdout


def _sources() -> dict[str, Path]:
    return {
        name: ROOT / name
        for name in (
            "eval/checkpoint.py",
            "eval/rq1_safety/sass_prologue.py",
            "opgate/gate/prologue.py",
            "opgate/executor/__init__.py",
        )
    }


def summary(rows: Sequence[Row]) -> dict[str, object]:
    """The counts of checked, passed and failed kernels, and each failure."""
    return {
        "kernels": len(rows),
        "as_written": sum(1 for r in rows if r["group"] == "as_written"),
        "with_checks": sum(1 for r in rows if r["group"] == "with_checks"),
        "with_shared_variables": sum(1 for r in rows if r["shared"]),
        "with_local_variables": sum(1 for r in rows if r["local"]),
        "with_sampled_loops": sum(
            1 for r in rows if any(c["values"] for c in r["result"]["shared"])
        ),
        "passed": sum(1 for r in rows if r["passed"]),
        "failed": {r["id"]: r["result"]["failures"] for r in rows if not r["passed"]},
    }


def _sample(chosen: list[tuple[Kernel, str]], count: int | None) -> list[tuple[Kernel, str]]:
    if count is None:
        return chosen
    step = max(1, len(chosen) // count)
    return chosen[::step][:count]


def _device() -> str:
    """The name of device 0, which the JIT processes load on."""
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (name,) = call(driver.cuDeviceGetName(64, device))
    if not isinstance(name, bytes):
        raise RuntimeError(f"the driver returned {name!r}, not a name")
    return name.split(b"\0")[0].decode()


def measure(
    registration: Path,
    guarded: Path,
    nvdisasm: Path,
    output: Path,
    *,
    workers: int,
    sample: int | None,
    resume: bool,
) -> None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    cores = sorted(os.sched_getaffinity(0))
    if workers > len(cores):
        raise ValueError(f"{workers} workers need at least {workers} cores")
    chosen = _sample(kernels(registration, guarded), sample)
    provenance = {
        "kind": KIND,
        "registration": str(registration),
        "guarded": str(guarded),
        "arch": ARCH,
        "device": _device(),
        "driver_api": integer(driver.cuDriverGetVersion()),
        "load": "cuModuleLoadData of the text and its terminating zero byte",
        "jit_options": [],
        "jit_cache": {"CUDA_CACHE_DISABLE": "0", "CUDA_CACHE_PATH": "empty, for each text"},
        "nvdisasm": str(nvdisasm),
        "nvdisasm_version": _version(nvdisasm),
        "param_base": PARAM_BASE,
        "shared_limit": SHARED_LIMIT,
        "paths": PATHS,
        "steps": STEPS,
        "sizes": list(SIZES),
        "cores": cores,
        "workers": workers,
        "kernels": [k["id"] for k, _ in chosen],
    }
    with Checkpoint(output, provenance, resume=resume) as record:
        snapshot_inputs(output, "sources", _sources(), create=not resume)
        pending = [(k, text) for k, text in chosen if record.get(k["id"]) is None]
        for kernel, text in pending:
            record.artifact(kernel["id"], text, "ptx")
        with ProcessPoolExecutor(max_workers=workers, mp_context=SPAWN) as pool:
            futures = [pool.submit(_measured, k, text, nvdisasm) for k, text in pending]
            for future in as_completed(futures):
                row, compiled = future.result()
                _artifact(output, row["id"], "sass", compiled.listing.encode("ascii"))
                _artifact(output, row["id"], "cubin", compiled.jit.cubin)
                record.add(dict(row))
        snapshot_inputs(output, "sources", _sources(), create=False)
        rows: list[Row] = [Row(**r) for r in record.data["results"]]
        record.finish(summary=summary(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--sample", type=int, help="check this many kernels, evenly spaced")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    measure(
        args.registration,
        args.guarded,
        args.nvdisasm,
        args.output,
        workers=args.workers,
        sample=args.sample,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
