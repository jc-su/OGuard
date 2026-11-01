"""The zeroing prologue that the gate requires of every kernel (design decision 46).

A kernel's shared memory and local memory hold, when it starts, what the
kernels that ran before it on the same multiprocessor left there, which may
be another tenant's data. `zeroed` puts a prologue at the start of the entry
kernel's body that writes zero to every byte that the kernel can read in
these memories before the kernel's other instructions run:

- every shared variable of the module, by all threads of the block. Thread
  `t`, the block's linear thread index, stores the element `t + k * N`, for
  `N` threads and `k = 0, 1, ...`. A dynamic variable (an `.extern` array, or
  an array of no elements) whose bytes, the binding's bytes of dynamic shared
  memory, depend on the launch's sizes is cleared first, by a loop over `k`
  below the count of elements, which the loop computes at run time from the
  parameters that hold the sizes: the binding's constant plus each
  coefficient times its size, divided by the width of a store. The other
  variables are cleared in rounds unrolled to their bytes, and in the last
  round the threads below the elements left store;
- then a barrier, `bar.sync 0`, so that every thread has stored before a
  thread reads shared memory;
- every local variable of the entry kernel, by each thread.

Each device function of the module that declares local variables gets the
stores of its local variables at the start of its body, which a call runs
when it enters its frame. A store to a local variable writes the largest of
16, 8, 4, 2 and 1 bytes that divides the variable's alignment and its bytes;
a store to a shared variable writes the largest of 4, 2 and 1 bytes that
divides the alignment and the bytes of every shared variable, and the
constant and the coefficients of the dynamic shared memory that a loop
clears, so that the shared stores are stores of the number 0 of one width,
which the gate's shared value analysis takes as ordered writes. The gate
recognizes exactly these instructions (`prologue` and `function_prologue` in
`formal/GateFront.v`) and refuses a kernel without them, and Rocq proves
that the loop stores zero to every byte of the dynamic shared memory of a
launch before the barrier (`formal/GateZero.v`). `Executor.register`
registers every text with its prologue, so every kernel that runs has one.
This module is not trusted: the gate checks the text that it writes, and
refuses a text that it does not complete, as one whose dynamic shared memory
has a size that no parameter holds, or a negative coefficient.

Imports `opgate.binding` only.
"""

import re
from dataclasses import dataclass

from opgate.binding import Affine, Binding, Size

__all__ = ["LOOP_REGISTERS", "PREDICATE", "REGISTERS", "zeroed"]

REGISTERS = "%opgate_z"
"""The prologue's three 32-bit registers: 0 the thread index, 1 an address, 2 zero."""
PREDICATE = "%opgate_p"
"""The prologue's predicate: whether a thread stores in a variable's last round,
or whether a loop's element lies past its count."""
LOOP_REGISTERS = "%opgate_d"
"""The two 32-bit registers of the loops that clear dynamic shared memory: 0 the
count of elements, 1 the element that a thread stores."""

_INDEX, _ADDRESS, _ZERO = (f"{REGISTERS}{k}" for k in range(3))
_COUNT, _ELEMENT = (f"{LOOP_REGISTERS}{k}" for k in range(2))
# a comment or a quoted string, as of a .file directive, which the pass does not read
_OPAQUE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:[^\"\\\n]|\\.)*\"", re.S)
# a declaration as the gate's parser reads it (`var_at`): an alignment, an
# element type, a name, and a count, which an array of unknown size lacks
_VARIABLE = r"\.{space}\s+\.align\s+(\d+)\s+\.[bsuf](8|16|32|64)\s+([A-Za-z_$][\w$]*)\s*(?:\[\s*(\d*)\s*\])?\s*;"
_SHARED = re.compile(_VARIABLE.format(space="shared"))
_LOCAL = re.compile(_VARIABLE.format(space="local"))
# a declaration or a directive of a body before its first instruction or label
_LEADING = re.compile(r"\s*(\.(reg|shared|local|pragma)\b[^;]*;|\.(loc|file)\b[^\n]*)")
_DECLARATIONS = (f".reg .b32 \t{REGISTERS}<3>;",)


@dataclass(frozen=True)
class _Variable:
    name: str
    align: int
    bytes: int  # 0 for an array of unknown size


def _variables(pattern: re.Pattern[str], text: str) -> list[_Variable]:
    result = []
    for found in pattern.finditer(text):
        align, bits, name, count = found.groups()
        elements = 1 if count is None else int(count or "0")
        result.append(_Variable(name, int(align), int(bits) // 8 * elements))
    return result


def _width(align: int, size: int) -> int:
    """The bytes of each store to a local variable: the largest power of two up
    to 16 that divides both the alignment and the bytes."""
    return next(w for w in (16, 8, 4, 2, 1) if align % w == 0 and size % w == 0)


def _shared_width(sized: list[tuple[int, int]]) -> int:
    """The bytes of each store to a shared variable: the largest of 4, 2 and 1
    that divides each alignment and byte count of `sized`."""
    return next(w for w in (4, 2, 1) if all(a % w == 0 and n % w == 0 for a, n in sized))


def _store(space: str, width: int, address: str) -> str:
    vector = {16: (".v4.u32", 4), 8: (".v2.u32", 2)}
    if width in vector:
        suffix, count = vector[width]
        return f"st.{space}{suffix} [{address}], {{{', '.join([_ZERO] * count)}}};"
    return f"st.{space}.u{8 * width} [{address}], {_ZERO};"


def _local(variables: list[_Variable]) -> list[str]:
    """The stores of the local variables, by each thread."""
    lines = []
    for variable in variables:
        width = _width(variable.align, variable.bytes)
        lines += [
            _store("local", width, f"{variable.name}+{width * k}")
            for k in range(variable.bytes // width)
        ]
    return lines


def _size_parameter(binding: Binding, symbol: int) -> str | None:
    """The first parameter of `binding` that holds size `symbol` itself."""
    return next(
        (
            name
            for name, parameter in binding.parameters
            if isinstance(parameter, Size) and parameter.value == Affine.symbol(symbol)
        ),
        None,
    )


def _loop(
    j: int, threads: int, width: int, name: str, shared: Affine, parameters: list[str]
) -> list[str]:
    """The loop that clears the `j`-th dynamic shared variable `name` in stores of
    `width` bytes: the count of elements is the constant of `shared` plus each
    coefficient times the size that its parameter holds, divided by `width`,
    and thread `t` stores the elements `t`, `t + threads`, ... below it."""
    lines = [f"mov.u32 {_COUNT}, {shared.constant // width};"]
    for (_, coefficient), parameter in zip(shared.terms, parameters, strict=True):
        lines += [
            f"ld.param.u32 {_ELEMENT}, [{parameter}];",
            f"mad.lo.u32 {_COUNT}, {_ELEMENT}, {coefficient // width}, {_COUNT};",
        ]
    return [
        *lines,
        f"mov.u32 {_ELEMENT}, {_INDEX};",
        f"$opgate_zero_{j}:",
        f"setp.ge.u32 {PREDICATE}, {_ELEMENT}, {_COUNT};",
        f"@{PREDICATE} bra $opgate_zeroed_{j};",
        f"mov.u32 {_ADDRESS}, {name};",
        f"mad.lo.u32 {_ADDRESS}, {_ELEMENT}, {width}, {_ADDRESS};",
        _store("shared", width, _ADDRESS),
        f"add.u32 {_ELEMENT}, {_ELEMENT}, {threads};",
        f"bra $opgate_zero_{j};",
        f"$opgate_zeroed_{j}:",
    ]


def _instructions(
    shared: list[_Variable], local: list[_Variable], binding: Binding
) -> tuple[list[str], bool]:
    """The prologue's instructions and labels, and whether it has a loop."""
    bx, by, bz = binding.block
    threads = bx * by * bz
    terms = binding.shared.terms
    sized: list[tuple[str, int, int]] = []  # the variables that rounds clear
    dynamic: list[_Variable] = []  # the variables that loops clear
    for variable in shared:
        if variable.bytes == 0 and terms:
            dynamic.append(variable)
            continue
        size = variable.bytes if variable.bytes > 0 else binding.shared.constant
        if size > 0:
            sized.append((variable.name, variable.align, size))
    parameters = [p for symbol, _ in terms if (p := _size_parameter(binding, symbol)) is not None]
    if (
        len(parameters) != len(terms)
        or binding.shared.constant < 0
        or any(c <= 0 for _, c in terms)
    ):
        dynamic = []  # the gate refuses the text, which lacks the loops
    locals_ = [v for v in local if v.bytes > 0]
    if not sized and not dynamic and not locals_:
        return [], False
    lines = [f"mov.u32 {_ZERO}, 0;"]
    if sized or dynamic:
        lines += [
            f"mov.u32 {_INDEX}, %tid.z;",
            f"mov.u32 {_ADDRESS}, %tid.y;",
            f"mad.lo.u32 {_INDEX}, {_INDEX}, {by}, {_ADDRESS};",
            f"mov.u32 {_ADDRESS}, %tid.x;",
            f"mad.lo.u32 {_INDEX}, {_INDEX}, {bx}, {_ADDRESS};",
        ]
        counts = (binding.shared.constant, *(c for _, c in terms))
        width = _shared_width(
            [(a, n) for _, a, n in sized] + [(v.align, c) for v in dynamic for c in counts]
        )
        for j, variable in enumerate(dynamic):
            lines += _loop(j, threads, width, variable.name, binding.shared, parameters)
        for name, _, size in sized:
            elements = size // width
            full, left = divmod(elements, threads)
            lines += [
                f"mov.u32 {_ADDRESS}, {name};",
                f"mad.lo.u32 {_ADDRESS}, {_INDEX}, {width}, {_ADDRESS};",
            ]
            lines += [
                _store("shared", width, f"{_ADDRESS}+{width * threads * k}") for k in range(full)
            ]
            if left:
                lines += [
                    f"setp.lt.u32 {PREDICATE}, {_INDEX}, {left};",
                    f"@{PREDICATE} "
                    + _store("shared", width, f"{_ADDRESS}+{width * threads * full}"),
                ]
        lines.append("bar.sync 0;")
    return lines + _local(locals_), bool(dynamic)


def _masked(text: str) -> str:
    """`text` with each character of its comments and quoted strings but the line
    ends a space, so that a position in it is the same position in `text`."""
    return _OPAQUE.sub(lambda found: re.sub(r"[^\n]", " ", found.group(0)), text)


def _closing(masked: str, at: int, opening: str, closing: str) -> int:
    """The position of the bracket that closes the bracket at `at`."""
    depth = 0
    for position in range(at, len(masked)):
        if masked[position] == opening:
            depth += 1
        elif masked[position] == closing:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError(f"the {opening} at {at} does not close")


def _functions(masked: str) -> list[tuple[int, int]]:
    """The opening and closing brace of the body of each device function."""
    bodies = []
    for found in re.finditer(r"\.func\b", masked):
        at, depth = found.end(), 0
        while at < len(masked) and not (depth == 0 and masked[at] in "{;"):
            depth += {"(": 1, ")": -1}.get(masked[at], 0)
            at += 1
        if at < len(masked) and masked[at] == "{":
            bodies.append((at, _closing(masked, at, "{", "}")))
    return bodies


def _start(masked: str, opened: int) -> int:
    """Where the first instruction or label of the body that opens at `opened`
    may go: after the body's leading declarations and directives."""
    at = opened + 1
    while (leading := _LEADING.match(masked, at)) is not None:
        at = leading.end()
    return at


def zeroed(text: str, binding: Binding) -> str:
    """`text` with the zeroing prologue at the start of the body of the entry kernel
    that `binding` names, and the stores of the local variables of each device
    function at the start of its body; `text` itself when there is no shared
    or local memory to clear."""
    masked = _masked(text)
    entry = re.compile(rf"\.entry\s+{re.escape(binding.kernel)}\s*\(")
    entries = list(entry.finditer(masked))
    if len(entries) != 1:
        raise ValueError(f"the module has {len(entries)} entry kernels named {binding.kernel}")
    opened = masked.index("{", _closing(masked, entries[0].end() - 1, "(", ")"))
    closed = _closing(masked, opened, "{", "}")
    shared = _variables(_SHARED, masked)
    local = _variables(_LOCAL, masked[opened : closed + 1])
    insertions: list[tuple[int, list[str]]] = []
    instructions, loops = _instructions(shared, local, binding)
    if instructions:
        loop = [f".reg .b32 \t{LOOP_REGISTERS}<2>;"] if loops else []
        lines = [*_DECLARATIONS, f".reg .pred \t{PREDICATE};", *loop, *instructions]
        insertions.append((_start(masked, opened), lines))
    for start, end in _functions(masked):
        variables = [v for v in _variables(_LOCAL, masked[start : end + 1]) if v.bytes > 0]
        if variables:
            lines = [*_DECLARATIONS, f"mov.u32 {_ZERO}, 0;", *_local(variables)]
            insertions.append((_start(masked, start), lines))
    for at, lines in sorted(insertions, reverse=True):
        text = text[:at] + "".join(f"\n\t{line}" for line in lines) + text[at:]
    return text
