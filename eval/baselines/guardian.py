"""Guardian's PTX address fencing, the baseline of the cost experiment.

A copy of `src/guardian/ptx.py` of the Guardian reimplementation (package
`guardian-ptx` 0.1.0, at `~/Dev/Guardian` on the evaluation machine),
which fences each global access as `(address & mask) | base`, for a
partition of `mask + 1` bytes at `base`, with the two values as parameters
appended to each entry. The copy differs from the original in its imports,
its formatting, and one reordered type test, which the project's lint and
type checks require, and in the accesses it fences. The original fences the
accesses of the `.global` state space only, and passes a generic access
(`ld`, `st`, `atom`, or `red` without a state space) through unfenced; Numba
reaches every array through generic accesses, so the original fenced none of
a Numba kernel's array accesses. Guardian fences every load and store
(Pavlidakis et al., Middleware 2024, section 5), and the copy fences each
generic access as a global one. A generic address is a global address
unless the function converts a local, shared, constant, or parameter address
to a generic one (`cvta.local` and the like); the fence would move such an
access into the partition, so the copy refuses a function that has both.
The copy also reads the header of a `.func` that returns a value, such as
nvcc's `.func (.param .b32 func_retval0) __iAtomicAdd(`, which the original
passes through as text outside any function, with its accesses unfenced.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


class UnsupportedPtxError(ValueError):
    """Raised when the patcher sees PTX it cannot instrument completely."""


@dataclass(frozen=True)
class EntryReport:
    name: str
    instrumented_accesses: int


@dataclass(frozen=True)
class PatchReport:
    entries: tuple[EntryReport, ...]

    @property
    def instrumented_accesses(self) -> int:
        return sum(entry.instrumented_accesses for entry in self.entries)


# a function's header, with the return parameters of a `.func` that returns a value
_FUNCTION_RE = re.compile(r"\.(?P<kind>entry|func)\s+(?:\([^)]*\)\s*)?(?P<name>[^\s(]+)\s*\(")
_PREDICATED_OPCODE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<predicate>@!?\s*%[A-Za-z0-9_$]+)?\s*"
    r"(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)\b"
)
_REGISTER_RE = re.compile(r"^(?P<reg>%[A-Za-z_][A-Za-z0-9_$]*)(?P<rest>.*)$")
_OFFSET_RE = re.compile(r"^(?P<sign>[+-])\s*(?P<value>0x[0-9A-Fa-f]+|\d+)$")
_ANY_REGISTER_RE = re.compile(r"%[A-Za-z_][A-Za-z0-9_$]*")

_PROTECTED_ROOT_OPS = frozenset({"ld", "st", "atom", "red"})
_STATE_SPACES = ("global", "shared", "local", "const", "param")
# a conversion of a local, shared, constant, or parameter address to a generic one
_TO_GENERIC_RE = re.compile(r"\bcvta\.(?P<space>local|shared|const|param)\b")
_HIDDEN_PARAM_TYPES = ".param .u64"


ProtectedParamSpec = Mapping[str, set[int]] | set[int] | None


def patch_ptx(
    ptx: str,
    *,
    protected_param_indices: ProtectedParamSpec = None,
    force_hidden_params: bool = False,
    trailing_toolchain_params: int = 0,
) -> tuple[str, PatchReport]:
    """Instrument global-memory PTX accesses with Guardian address fencing.

    The patcher only emits code it knows how to protect. Unsupported global
    memory address expressions raise :class:`UnsupportedPtxError`.
    """

    lines = ptx.splitlines()
    output: list[str] = []
    reports: list[EntryReport] = []
    cursor = 0
    function_index = 0

    while cursor < len(lines):
        match = _FUNCTION_RE.search(lines[cursor])
        if match is None:
            output.append(lines[cursor])
            cursor += 1
            continue

        kind = match.group("kind")
        name = match.group("name")
        body_open = _find_body_open(lines, cursor)
        body_close = _find_body_close(lines, body_open + 1)
        function_lines = lines[cursor : body_close + 1]

        if kind == "func":
            _reject_unsupported_func_with_global_memory(name, function_lines)
            output.extend(function_lines)
        else:
            patched, access_count = _patch_entry(
                function_lines=function_lines,
                function_name=name,
                function_index=function_index,
                protected_param_indices=_protected_params_for_entry(
                    protected_param_indices,
                    name,
                ),
                force_hidden_params=force_hidden_params,
                trailing_toolchain_params=trailing_toolchain_params,
            )
            output.extend(patched)
            reports.append(EntryReport(name=name, instrumented_accesses=access_count))
            function_index += 1

        cursor = body_close + 1

    if (
        protected_param_indices is None
        and reports
        and all(report.instrumented_accesses == 0 for report in reports)
    ):
        raise UnsupportedPtxError("no global or generic memory accesses were found to instrument")

    return "\n".join(output) + ("\n" if ptx.endswith("\n") else ""), PatchReport(tuple(reports))


def _patch_entry(
    function_lines: list[str],
    function_name: str,
    function_index: int,
    protected_param_indices: set[int] | None,
    force_hidden_params: bool,
    trailing_toolchain_params: int = 0,
) -> tuple[list[str], int]:
    if any("guardian_base_" in line or "guardian_mask_" in line for line in function_lines):
        raise UnsupportedPtxError(f"{function_name}: PTX already appears to be instrumented")

    body_open = _relative_body_open(function_lines)
    body_close = _relative_body_close(function_lines, body_open + 1)
    body = function_lines[body_open + 1 : body_close]

    instrumented_body, access_count = _instrument_body(
        function_name,
        body,
        protected_param_indices=protected_param_indices,
    )
    if protected_param_indices is not None and access_count == 0 and not force_hidden_params:
        return function_lines, access_count

    patched_header = _add_hidden_params(
        function_lines[: body_open + 1], function_index, trailing_toolchain_params
    )

    if access_count == 0:
        return patched_header + function_lines[body_open + 1 :], access_count

    declaration_lines = _guardian_declarations(access_count)
    load_lines = _guardian_param_loads(function_index)
    insertion = _declaration_insertion_index(instrumented_body)
    body_with_decls = (
        instrumented_body[:insertion] + declaration_lines + instrumented_body[insertion:]
    )

    load_insertion = _load_insertion_index(body_with_decls)
    body_with_loads = (
        body_with_decls[:load_insertion] + load_lines + body_with_decls[load_insertion:]
    )

    return patched_header + body_with_loads + function_lines[body_close:], access_count


def _instrument_body(
    function_name: str,
    body: Iterable[str],
    *,
    protected_param_indices: set[int] | None,
) -> tuple[list[str], int]:
    patched: list[str] = []
    access_index = 0
    tainted_registers: set[str] = set()
    body = list(body)
    converted = _TO_GENERIC_RE.search("\n".join(_split_comment(line)[0] for line in body))

    for line_number, line in enumerate(body, start=1):
        instruction = _parse_memory_instruction(line)
        if instruction is None:
            patched.append(line)
            if protected_param_indices is not None:
                _update_taint(function_name, line, protected_param_indices, tainted_registers)
            continue

        if converted is not None and _state_space(instruction.opcode) is None:
            raise UnsupportedPtxError(
                f"{function_name}:{line_number}: a generic access in a function that makes "
                f"generic addresses of {converted.group('space')} memory"
            )
        address = _parse_address_expression(function_name, line_number, instruction.address_expr)
        should_guard = protected_param_indices is None or address.base_register in tainted_registers
        if not should_guard:
            patched.append(line)
            _update_memory_taint(line, tainted_registers)
            continue

        guard = _guard_lines(
            indent=instruction.indent,
            predicate=instruction.predicate,
            source_reg=address.base_register,
            offset=address.offset,
            temp_reg="%grd_addr",
        )
        patched.extend(guard)
        patched.append(
            line[: instruction.address_start] + "[%grd_addr]" + line[instruction.address_end :]
        )
        access_index += 1
        if protected_param_indices is not None:
            _update_memory_taint(line, tainted_registers)

    return patched, access_index


def _protected_params_for_entry(
    spec: ProtectedParamSpec,
    function_name: str,
) -> set[int] | None:
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return set(spec.get(function_name, set()))
    return set(spec)


@dataclass(frozen=True)
class _MemoryInstruction:
    indent: str
    predicate: str
    opcode: str
    address_expr: str
    address_start: int
    address_end: int


@dataclass(frozen=True)
class _AddressExpression:
    base_register: str
    offset: int | None


def _parse_memory_instruction(line: str) -> _MemoryInstruction | None:
    code, _comment = _split_comment(line)
    opcode_match = _PREDICATED_OPCODE_RE.match(code)
    if opcode_match is None:
        return None

    opcode = opcode_match.group("opcode")
    if not _is_protected_opcode(opcode):
        return None

    address = _find_global_address_expression(code, opcode)
    if address is None:
        raise UnsupportedPtxError(
            f"global memory instruction has no address operand: {line.strip()}"
        )

    address_start, address_end, address_expr = address
    return _MemoryInstruction(
        indent=opcode_match.group("indent"),
        predicate=(opcode_match.group("predicate") or "").replace(" ", ""),
        opcode=opcode,
        address_expr=address_expr,
        address_start=address_start,
        address_end=address_end,
    )


def _state_space(opcode: str) -> str | None:
    """The state space that `opcode` names, such as `shared` for
    `ld.shared::cta.u32`, or None for a generic access."""
    for token in opcode.split(".")[1:]:
        space = token.split("::", 1)[0]
        if space in _STATE_SPACES:
            return space
    return None


def _is_protected_opcode(opcode: str) -> bool:
    """Whether `opcode` accesses global memory: a global or a generic access."""
    if opcode.startswith("cp.async") and ".global" in opcode:
        return True

    root = opcode.split(".", 1)[0]
    if root not in _PROTECTED_ROOT_OPS:
        return False
    return _state_space(opcode) in ("global", None)


def _find_global_address_expression(
    code: str,
    opcode: str,
) -> tuple[int, int, str] | None:
    if opcode.startswith("cp.async"):
        expressions = _find_bracketed_expressions(code)
        if len(expressions) < 2:
            raise UnsupportedPtxError(
                f"cp.async instruction has no global source operand: {code.strip()}"
            )
        return expressions[1]
    return _find_first_bracketed_expression(code)


def _find_first_bracketed_expression(code: str) -> tuple[int, int, str] | None:
    expressions = _find_bracketed_expressions(code)
    if not expressions:
        return None
    return expressions[0]


def _find_bracketed_expressions(code: str) -> list[tuple[int, int, str]]:
    expressions: list[tuple[int, int, str]] = []
    cursor = 0
    while True:
        start = code.find("[", cursor)
        if start == -1:
            return expressions
        end = code.find("]", start + 1)
        if end == -1:
            raise UnsupportedPtxError(f"unterminated PTX address expression: {code.strip()}")
        expressions.append((start, end + 1, code[start + 1 : end].strip()))
        cursor = end + 1


def _parse_address_expression(
    function_name: str,
    line_number: int,
    expression: str,
) -> _AddressExpression:
    compact = expression.replace(" ", "")
    register_match = _REGISTER_RE.match(compact)
    if register_match is None:
        raise UnsupportedPtxError(
            f"{function_name}:{line_number}: unsupported global address expression [{expression}]"
        )

    base_register = register_match.group("reg")
    rest = register_match.group("rest")
    if rest == "":
        return _AddressExpression(base_register=base_register, offset=None)

    offset_match = _OFFSET_RE.match(rest)
    if offset_match is None:
        raise UnsupportedPtxError(
            f"{function_name}:{line_number}: unsupported global address offset [{expression}]"
        )

    value = int(offset_match.group("value"), 0)
    if offset_match.group("sign") == "-":
        value = -value
    return _AddressExpression(base_register=base_register, offset=value)


def _guard_lines(
    indent: str,
    predicate: str,
    source_reg: str,
    offset: int | None,
    temp_reg: str,
) -> list[str]:
    pred = f"{predicate} " if predicate else ""
    lines: list[str] = []
    if offset is not None and offset != 0:
        op = "add.s64" if offset < 0 else "add.u64"
        lines.append(f"{indent}{pred}{op} {temp_reg}, {source_reg}, {offset};")
        source_reg = temp_reg
    lines.append(f"{indent}{pred}and.b64 {temp_reg}, {source_reg}, %grd_mask;")
    lines.append(f"{indent}{pred}or.b64 {temp_reg}, {temp_reg}, %grd_base;")
    return lines


def _update_taint(
    function_name: str,
    line: str,
    protected_param_indices: set[int],
    tainted_registers: set[str],
) -> None:
    code, _comment = _split_comment(line)
    opcode_match = _PREDICATED_OPCODE_RE.match(code)
    if opcode_match is None:
        return

    opcode = opcode_match.group("opcode")
    if opcode.startswith("ld.param"):
        _update_param_taint(function_name, code, protected_param_indices, tainted_registers)
        return

    if _is_protected_opcode(opcode):
        _update_memory_taint(line, tainted_registers)
        return

    dest = _first_destination_register(code)
    if dest is None:
        return

    sources = set(_ANY_REGISTER_RE.findall(code))
    sources.discard(dest)
    if sources & tainted_registers:
        tainted_registers.add(dest)
        return

    if opcode_match.group("predicate") is None:
        tainted_registers.discard(dest)


def _update_param_taint(
    function_name: str,
    code: str,
    protected_param_indices: set[int],
    tainted_registers: set[str],
) -> None:
    dest = _first_destination_register(code)
    if dest is None:
        return

    param_match = re.search(
        rf"\[\s*{re.escape(function_name)}_param_(?P<index>\d+)\s*\]",
        code,
    )
    if param_match is None:
        tainted_registers.discard(dest)
        return

    param_index = int(param_match.group("index"))
    if param_index in protected_param_indices:
        tainted_registers.add(dest)
    else:
        tainted_registers.discard(dest)


def _update_memory_taint(line: str, tainted_registers: set[str]) -> None:
    code, _comment = _split_comment(line)
    # Memory instructions load data, not trusted pointers. Do not taint loaded
    # values from a protected address. If the destination register is reused as
    # an address later, kill its old taint on unpredicated loads.
    opcode_match = _PREDICATED_OPCODE_RE.match(code)
    if opcode_match is None or opcode_match.group("predicate") is not None:
        return
    opcode = opcode_match.group("opcode")
    if opcode.split(".", 1)[0] not in {"ld"}:
        return
    dest = _first_destination_register(code)
    if dest is not None:
        tainted_registers.discard(dest)


def _first_destination_register(code: str) -> str | None:
    opcode_match = _PREDICATED_OPCODE_RE.match(code)
    if opcode_match is None:
        return None
    operand_text = code[opcode_match.end() :].lstrip()
    if not operand_text:
        return None
    match = _ANY_REGISTER_RE.search(operand_text)
    if match is None:
        return None
    return match.group(0)


def _guardian_declarations(access_count: int) -> list[str]:
    return [
        "\t.reg .b64 \t%grd_base;",
        "\t.reg .b64 \t%grd_mask;",
        "\t.reg .b64 \t%grd_addr;",
    ]


def _guardian_param_loads(function_index: int) -> list[str]:
    return [
        f"\tld.param.u64 \t%grd_base, [guardian_base_{function_index}];",
        f"\tld.param.u64 \t%grd_mask, [guardian_mask_{function_index}];",
    ]


def _add_hidden_params(
    header: list[str], function_index: int, trailing_toolchain_params: int = 0
) -> list[str]:
    """Add the guard's two scalars to one entry's parameter list.

    They go last, except before any parameters the toolchain appended after
    the kernel's own. Triton 3.8 adds a global-scratch and a profile-scratch
    pointer there and its launcher passes them after the arguments the
    signature names, so the guard's two have to sit between the two groups
    for the entry's order and the launcher's order to agree. Zero keeps the
    older behavior of appending after everything.
    """
    close_index = _signature_close_index(header)
    patched = list(header)
    insert_index = close_index
    for _ in range(trailing_toolchain_params):
        previous = _previous_signature_item_index(patched, insert_index)
        if previous is None:
            raise UnsupportedPtxError(
                "entry declares fewer parameters than the toolchain is said to append"
            )
        insert_index = previous

    previous_param_index = _previous_signature_item_index(patched, insert_index)
    if previous_param_index is not None:
        stripped = patched[previous_param_index].rstrip()
        if not stripped.endswith(",") and not stripped.endswith("("):
            patched[previous_param_index] = stripped + ","

    trails = insert_index < close_index
    hidden_params = [
        f"\t{_HIDDEN_PARAM_TYPES} guardian_base_{function_index},",
        f"\t{_HIDDEN_PARAM_TYPES} guardian_mask_{function_index}" + ("," if trails else ""),
    ]
    patched[insert_index:insert_index] = hidden_params
    return patched


def _signature_close_index(header: list[str]) -> int:
    paren_depth = 0
    saw_open = False
    for index, line in enumerate(header):
        paren_depth += line.count("(")
        if "(" in line:
            saw_open = True
        paren_depth -= line.count(")")
        if saw_open and paren_depth == 0:
            if line.strip() == "{":
                raise UnsupportedPtxError("function body opened before signature closed")
            return index
    raise UnsupportedPtxError("could not locate PTX function signature close")


def _previous_signature_item_index(lines: list[str], close_index: int) -> int | None:
    for index in range(close_index - 1, -1, -1):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("//"):
            continue
        return index
    return None


def _declaration_insertion_index(body: list[str]) -> int:
    for index, line in enumerate(body):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("//") or stripped.endswith(":"):
            continue
        return index
    return 0


def _load_insertion_index(body: list[str]) -> int:
    for index, line in enumerate(body):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.endswith(":"):
            continue
        if stripped.startswith(".") or _is_guardian_declaration(stripped):
            continue
        return index
    return len(body)


def _is_guardian_declaration(stripped: str) -> bool:
    return stripped.startswith(".reg") and "%grd_" in stripped


def _split_comment(line: str) -> tuple[str, str]:
    marker = line.find("//")
    if marker == -1:
        return line, ""
    return line[:marker], line[marker:]


def _find_body_open(lines: list[str], start: int) -> int:
    for index in range(start, len(lines)):
        if lines[index].strip() == "{":
            return index
    raise UnsupportedPtxError(f"could not locate body for function starting at line {start + 1}")


def _find_body_close(lines: list[str], start: int) -> int:
    for index in range(start, len(lines)):
        if lines[index].strip() == "}":
            return index
    raise UnsupportedPtxError(f"could not locate function close after line {start + 1}")


def _relative_body_open(function_lines: list[str]) -> int:
    return _find_body_open(function_lines, 0)


def _relative_body_close(function_lines: list[str], start: int) -> int:
    return _find_body_close(function_lines, start)


def _reject_unsupported_func_with_global_memory(name: str, function_lines: list[str]) -> None:
    for line in function_lines:
        if _parse_memory_instruction(line) is not None:
            raise UnsupportedPtxError(
                f"{name}: .func bodies with global or generic memory accesses are not patched yet; "
                "inline or entry-lift them"
            )
