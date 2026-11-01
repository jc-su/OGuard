"""Rewrites that make a kernel's barrier participation provable for the gate.

The gate proves that every thread of a block executes the same sequence of
barriers and warp collectives only if that sequence depends on values that
are the same for the block's threads at launch; it treats a value that a
thread reads from shared memory, or receives from a collective, as
arbitrary. It also needs every thread to reach each barrier, so it needs a
proof that each loop before a barrier ends, which it gives a loop that
counts in 64 bits by a positive register stride. `rewrite` changes the
forms below, which block these proofs, and keeps a safe kernel's results:

- Grid values in 32 bits. nvcc computes a grid index `%ctaid.x * %ntid.x +
  %tid.x` or a grid stride `%ntid.x * %nctaid.x` (as products, constant
  shifts, and sums) in 32 bits and converts it to 64; the 32-bit stride of a
  launch with 2^32 threads is zero, so the gate cannot prove that a loop that
  steps by it ends. The rewrite computes each such converted value in 64 bits
  (`mul.wide.u32`). A launch with fewer than 2^31 threads in all gets the
  same values; a larger one, whose 32-bit values wrap, gets the true ones.
- Counted loops in 32 bits. A loop `c = c + s; if (c < n) goto head` with
  `s` a positive constant or the block's width gets a 64-bit copy of `c`,
  which its test compares with a 64-bit copy of `n`. While `c + s` does not
  pass 2^31 (2^32 unsigned) the tests agree; past it the 32-bit counter
  wraps, a signed overflow that C leaves undefined, and the loop may not
  end, while the 64-bit test ends it.
- Immediate strides. A loop that counts in 64 bits by a constant counts by a
  register that holds it, which the gate's loop summary requires; the values
  are the same.

- Loops that end on a block reduction. Triton compiles `while
  tl.sum(pending) > 0:` (or `tl.max`) to a loop whose exit test is a block
  total: each thread contributes `selp 1, 0, C` for each of its own
  predicates `C`, the warps reduce the contributions (`redux.sync` or
  `shfl.sync`) and exchange them through shared memory between two
  `bar.sync`, and the loop leaves when the total is zero (or a test `Q` of
  the thread's own holds). The rewrite makes the loop leave when the
  thread's own contributions are all false (or `Q` holds), and removes the
  reduction and the loop's barriers. It applies only if, for every value of
  the predicates that the loop carries from one iteration to the next that
  makes the contributions false, the iteration writes no memory (each store
  and atomic is guarded by a predicate that is then false, or is a
  compare-and-swap whose new value equals the value it compares with),
  keeps the contributions false, and keeps every register that the code
  after the loop reads. A thread whose contributions are false then has no
  effect in a later iteration, so leaving at once gives the results that
  waiting for the other threads gives.
- Branches between barriers. A forward branch before the last barrier or
  collective, or inside the lines such a branch skips, that skips no jump,
  becomes guards: each load, store, and atomic that it skips, and each
  instruction that writes a register that is read outside the skipped
  lines, runs only if the branch would fall through, and the other
  instructions run on both paths. A load that the branch skips may lie
  outside its buffer or shared variable on the other path, as a block
  reduction's `smin[tid + s]` for `tid >= s` does, so it is guarded as well.
  It applies only if each barrier or collective among them names
  every thread and writes no register that is read elsewhere.
  If the lines hold no barrier or collective, the results are the same for
  every branch condition; if they do, the condition is the same for every
  thread of a safe kernel, which a barrier requires, and the barriers run
  where no thread skipped them, or for every thread.
- Binary searches. A loop `while lo < hi` that sets either `lo = mid + 1`
  or `hi = mid` in each iteration, or `while lo <= hi` that sets either
  `lo = mid + 1` or `hi = mid - 1`, with `mid` the midpoint of the two, at
  least halves `hi - lo` (or `hi - lo + 1`) each time, so it runs at most
  bits + 1 times for bounds of a given number of bits. The rewrite adds an iteration count
  that ends the loop after bits + 2 iterations, which the search never
  reaches, so that the gate can prove that it ends.

The pass is untrusted: the gate checks the rewritten text as any other, and
refuses it if it is unsafe.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

__all__ = ["CARRIED", "PARTICIPATION", "rewrite"]

# the gate's reason for a kernel whose barriers it does not prove every thread reaches
PARTICIPATION = "CTA barrier participation is not proved"
CARRIED = 10  # the most carried predicates whose values the check enumerates

_LABEL = re.compile(r"^\s*(\$[\w$]+):")
_INSTRUCTION = re.compile(r"^\s*(?:@(!?)(%?[\w$]+)\s+)?([a-z][\w.:]*)\b\s*([^;]*);")
_REGISTER = re.compile(r"^%?[A-Za-z_][\w$]*$")
_TOKEN = re.compile(r"%?[A-Za-z_][\w$]*")
_COMBINE = ("add", "max", "or")
_COLLECTIVE = ("bar", "barrier", "shfl", "redux", "vote", "match")


@dataclass(frozen=True)
class _Instruction:
    line: int
    negated: bool
    guard: str | None
    opcode: str
    operands: tuple[str, ...]

    @property
    def root(self) -> str:
        return self.opcode.split(".")[0]

    @property
    def destination(self) -> str | None:
        """The register that the instruction writes as its first operand."""
        if self.operands and _REGISTER.match(self.operands[0]):
            return self.operands[0]
        return None

    def reads(self) -> set[str]:
        """The registers that the instruction reads, its guard included."""
        sources = list(self.operands[1:]) + [o for o in self.operands[:1] if o.startswith("[")]
        used = {r for o in sources for r in _TOKEN.findall(o) if not r[0].isdigit()}
        if self.guard is not None:
            used.add(self.guard)
        return used

    def writes(self) -> set[str]:
        """The registers that the instruction may write."""
        if not self.operands:
            return set()
        first = self.operands[0]
        if first.startswith("{"):
            return {r for r in _TOKEN.findall(first)}
        return {first} if _REGISTER.match(first) else set()


def _operands(text: str) -> tuple[str, ...]:
    """The comma-separated operands, a vector `{...}` or an address `[...]` as one."""
    parts: list[str] = []
    depth, current = 0, ""
    for ch in text:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return tuple(parts)


def _parse(lines: list[str], number: int) -> _Instruction | None:
    code = lines[number].split("//", 1)[0]
    found = _INSTRUCTION.match(code)
    if found is None:
        return None
    negated, guard, opcode, rest = found.groups()
    return _Instruction(number, negated == "!", guard, opcode, _operands(rest))


def _integer(text: str) -> int | None:
    try:
        return int(text.strip(), 0)
    except ValueError:
        return None


def _width(opcode: str) -> int:
    found = re.search(r"\.[bsu](8|16|32|64)\b", opcode)
    return int(found.group(1)) if found else 32


def _signed(opcode: str) -> bool:
    return re.search(r"\.s(8|16|32|64)\b", opcode) is not None


@dataclass(frozen=True)
class _Unknown:
    """A value that the evaluation does not know: a register's value at the start
    of the iteration (`line` None), or the result of the instruction at `line`."""

    register: str
    line: int | None


Value = int | bool | _Unknown


class _State:
    """Each register's value; a register that the iteration has not written holds
    its value at the start of the iteration."""

    def __init__(self, known: dict[str, int | bool]) -> None:
        self.values: dict[str, Value] = dict(known)

    def get(self, operand: str) -> Value:
        immediate = _integer(operand)
        if immediate is not None:
            return immediate
        return self.values.get(operand, _Unknown(operand, None))

    def set(self, register: str, value: Value) -> None:
        self.values[register] = value


def _compare(kind: str, a: int, b: int) -> bool:
    match kind:
        case "eq":
            return a == b
        case "ne":
            return a != b
        case "lt" | "lo":
            return a < b
        case "le" | "ls":
            return a <= b
        case "gt" | "hi":
            return a > b
        case "ge" | "hs":
            return a >= b
        case _:
            raise ValueError(f"no comparison {kind}")


def _as_signed(value: int, width: int) -> int:
    value %= 1 << width
    return value - (1 << width) if value >= 1 << (width - 1) else value


def _kleene(op: str, x: bool | None, y: bool | None) -> bool | None:
    match op:
        case "and":
            if x is False or y is False:
                return False
            return None if x is None or y is None else True
        case "or":
            if x is True or y is True:
                return True
            return None if x is None or y is None else False
        case "xor":
            return None if x is None or y is None else x != y
        case _:
            raise ValueError(op)


def _arith(opcode: str, a: int, b: int) -> int:
    width = _width(opcode)
    if _signed(opcode):
        a, b = _as_signed(a, width), _as_signed(b, width)
    match opcode.split(".")[0]:
        case "and":
            return a & b
        case "or":
            return a | b
        case "xor":
            return a ^ b
        case "add":
            return a + b
        case "sub":
            return a - b
        case "mul":
            return a * b
        case "shl":
            return a << b if b < width else 0
        case "shr":
            return a >> b if b < width else (-1 if a < 0 else 0)
        case "min":
            return min(a, b)
        case "max":
            return max(a, b)
        case other:
            raise ValueError(other)


def _step(state: _State, ins: _Instruction) -> None:
    """The state after `ins` runs: a result that the evaluation does not compute
    is unknown."""
    ops = ins.operands
    width = _width(ins.opcode)
    mask = (1 << width) - 1
    pred = ".pred" in ins.opcode

    def boolean(operand: str) -> bool | None:
        value = state.get(operand.lstrip("!"))
        if type(value) is not bool:
            return None
        return not value if operand.startswith("!") else value

    def integers(*xs: str) -> list[int] | None:
        values = [state.get(x) for x in xs]
        found = [v for v in values if type(v) is int]
        return found if len(found) == len(values) else None

    destination = ins.destination
    result: Value | None = None
    match ins.root:
        case "mov" if len(ops) == 2 and destination is not None:
            value = state.get(ops[1])
            result = value & mask if type(value) is int and not pred else value
        case "selp" if len(ops) == 4:
            choice = boolean(ops[3])
            a, b = state.get(ops[1]), state.get(ops[2])
            value = (a if choice else b) if choice is not None else (a if a == b else None)
            result = value & mask if type(value) is int else value
        case "setp" if len(ops) == 3:
            values = integers(ops[1], ops[2])
            if values is not None:
                a, b = values
                if _signed(ins.opcode):
                    a, b = _as_signed(a, width), _as_signed(b, width)
                else:
                    a, b = a & mask, b & mask
                result = _compare(ins.opcode.split(".")[1], a, b)
        case "and" | "or" | "xor" if pred and len(ops) == 3:
            result = _kleene(ins.root, boolean(ops[1]), boolean(ops[2]))
        case "not" if pred and len(ops) == 2:
            value = boolean(ops[1])
            result = None if value is None else not value
        case "and" | "or" | "xor" | "add" | "sub" | "shl" | "shr" | "min" | "max" | "mul" if (
            not pred and len(ops) == 3
        ):
            values = integers(ops[1], ops[2])
            if values is not None:
                result = _arith(ins.opcode, *values) & mask
            else:
                result = _identity(ins.root, state.get(ops[1]), state.get(ops[2]))
    for register in ins.writes():
        if register == destination and result is not None:
            state.set(register, result)
        else:
            state.set(register, _Unknown(register, ins.line))


def _identity(root: str, a: Value, b: Value) -> Value | None:
    """The result of `root` on `a` and `b` when one of them is its identity element."""
    if b == 0 and root in ("add", "sub", "or", "xor", "shl", "shr"):
        return a
    if a == 0 and root in ("add", "or", "xor"):
        return b
    if b == 1 and root == "mul":
        return a
    if a == 1 and root == "mul":
        return b
    return None


def _writes_memory(ins: _Instruction) -> bool:
    return ins.root in ("st", "atom", "red") or ins.opcode.startswith("cp.")


def _runs(state: _State, ins: _Instruction) -> bool | None:
    if ins.guard is None:
        return True
    value = state.get(ins.guard)
    if type(value) is not bool:
        return None
    return value != ins.negated


@dataclass(frozen=True)
class _Loop:
    head: int  # the header label's line
    back: int  # the back edge's line
    exit: _Instruction  # the branch whose test ends the loop: an exit, or the back edge
    bottom: bool  # whether the loop ends when its conditional back edge is not taken


def _labels(lines: list[str]) -> dict[str, int]:
    return {m.group(1): i for i, line in enumerate(lines) if (m := _LABEL.match(line))}


def _loops(lines: list[str]) -> Iterator[_Loop]:
    """The loops with a straight body: an unconditional back edge and one exit, or
    a conditional back edge and no other jump."""
    labels = _labels(lines)
    for number in range(len(lines)):
        ins = _parse(lines, number)
        if ins is None or ins.root != "bra" or not ins.operands:
            continue
        head = labels.get(ins.operands[0])
        if head is None or head >= number:
            continue
        jumps = [
            b
            for i in range(head + 1, number)
            if (b := _parse(lines, i)) is not None
            and b.root in ("bra", "brx", "ret", "exit", "call", "trap")
        ]
        if ins.guard is not None:
            if not jumps:
                yield _Loop(head, number, ins, True)
            continue
        if len(jumps) != 1 or jumps[0].root != "bra" or jumps[0].guard is None:
            continue
        if labels.get(jumps[0].operands[0], -1) > number:
            yield _Loop(head, number, jumps[0], False)


def _definition(lines: list[str], start: int, end: int, register: str) -> _Instruction | None:
    """The last instruction in `[start, end)` that writes `register`."""
    for number in range(end - 1, start - 1, -1):
        ins = _parse(lines, number)
        if ins is not None and register in ins.writes():
            return ins
    return None


@dataclass(frozen=True)
class _Reduction:
    contributions: tuple[str, ...]  # each thread's own predicates
    own: str | None  # the thread's own exit test beside the total's, if any
    lines: frozenset[int]  # the lines that compute the total and test it


def _reduction(lines: list[str], loop: _Loop) -> _Reduction | None:
    """The block reduction of each thread's predicates whose zero total the exit
    tests; None if the exit is not such a test."""
    test = loop.exit
    assert test.guard is not None
    condition = _definition(lines, loop.head, test.line, test.guard)
    if condition is None:
        return None
    computed: set[int] = {condition.line}
    own: str | None = None
    # a back edge leaves the loop when it is not taken
    negated = test.negated != loop.bottom
    if (
        condition.opcode == "or.pred"
        and not negated
        and not loop.bottom
        and len(condition.operands) == 3
    ):
        # exit if the total is zero, or if the thread's own test holds
        a, b = condition.operands[1], condition.operands[2]
        found = [
            (d, other)
            for operand, other in ((a, b), (b, a))
            if not operand.startswith("!")
            and (d := _definition(lines, loop.head, condition.line, operand)) is not None
            and d.opcode.startswith("setp.")
        ]
        tested = [
            (d, other) for d, other in found if _total_test(lines, loop, d, negated) is not None
        ]
        if len(tested) != 1:
            return None
        condition, own = tested[0]
        computed.add(condition.line)
    if not condition.opcode.startswith("setp."):
        return None
    total = _total_test(lines, loop, condition, negated)
    if total is None:
        return None
    contributions, slice_lines = total
    return _Reduction(contributions, own, frozenset(computed | slice_lines))


def _total_test(
    lines: list[str], loop: _Loop, compare: _Instruction, negated: bool
) -> tuple[tuple[str, ...], set[int]] | None:
    """The contributions and the lines of a total that `compare` tests for zero."""
    if len(compare.operands) != 3:
        return None
    total, limit = compare.operands[1], _integer(compare.operands[2])
    if limit is None:
        return None
    kind = compare.opcode.split(".")[1]
    slice_lines: set[int] = {compare.line}
    contributions: list[str] = []
    sums = False  # whether the total adds contributions, or only takes their maximum
    pending = [(total, compare.line)]
    while pending:
        register, before = pending.pop()
        ins = _definition(lines, loop.head, before, register)
        if ins is None:
            return None
        if ins.line in slice_lines:
            continue
        ops = ins.operands
        if ins.root == "selp" and len(ops) == 4 and _integer(ops[1]) == 1 and _integer(ops[2]) == 0:
            # a contribution stays: the loop may use its value for more than the total
            if ops[3].startswith("!") or not ops[3].startswith("%p"):
                return None
            contributions.append(ops[3])
            continue
        slice_lines.add(ins.line)
        if ins.root in _COMBINE and len(ops) == 3:
            sums = sums or ins.root == "add"
            pending += [(o, ins.line) for o in ops[1:] if _integer(o) is None]
        elif ins.root in ("mov", "cvt") and len(ops) == 2:
            if _integer(ops[1]) is None:
                pending.append((ops[1], ins.line))
        elif ins.root == "shfl" and len(ops) >= 2:
            pending.append((ops[1], ins.line))
        elif ins.root == "redux" and len(ops) >= 2 and ins.opcode.split(".")[2] in _COMBINE:
            sums = sums or ins.opcode.split(".")[2] == "add"
            pending.append((ops[1], ins.line))
        elif ins.root == "ld" and ".shared" in ins.opcode and len(ops) == 2:
            # the values that the loop's shared stores wrote before a barrier
            stores = [
                s
                for i in range(loop.head, ins.line)
                if (s := _parse(lines, i)) is not None and s.opcode.startswith("st.shared")
            ]
            barriers = [
                i for i in range(loop.head, ins.line) if re.match(r"^\s*bar\.sync\b", lines[i])
            ]
            if not stores or not barriers or max(s.line for s in stores) > max(barriers):
                return None
            for store in stores:
                slice_lines.add(store.line)
                pending.append((store.operands[1], store.line))
        else:
            return None
    if not contributions:
        return None

    def exits(value: int) -> bool:
        return _compare(kind, value, limit) != negated

    # each contribution is 0 or 1, so a maximum is 0 or 1 and a sum at least 0;
    # the test must hold for a zero total only
    if not exits(0) or any(exits(v) for v in ((1, 2, 3) if sums else (1,))):
        return None
    return tuple(sorted(set(contributions))), slice_lines


def _carried(lines: list[str], loop: _Loop, removed: frozenset[int]) -> list[str]:
    """The predicates that the loop reads before it writes them in an iteration."""
    written: set[str] = set()
    carried: list[str] = []
    for number in range(loop.head, loop.back):
        ins = _parse(lines, number)
        if ins is None or number in removed or number == loop.exit.line:
            continue
        for name in sorted(ins.reads()):
            if name.startswith("%p") and name not in written and name not in carried:
                carried.append(name)
        written |= ins.writes()
    return carried


def _constants(lines: list[str]) -> dict[str, int | bool]:
    """Registers that the kernel sets once, to an immediate."""
    counts: dict[str, int] = {}
    values: dict[str, int] = {}
    for number in range(len(lines)):
        ins = _parse(lines, number)
        if ins is None:
            continue
        for register in ins.writes():
            counts[register] = counts.get(register, 0) + 1
        if ins.root == "mov" and len(ins.operands) == 2 and ins.guard is None and ins.destination:
            value = _integer(ins.operands[1])
            if value is not None:
                values[ins.destination] = value & ((1 << _width(ins.opcode)) - 1)
    return {r: v for r, v in values.items() if counts[r] == 1}


@dataclass(frozen=True)
class _Iteration:
    idle: bool | None  # whether every contribution is false at the exit
    quiet: bool  # whether the iteration provably writes no memory
    end: dict[str, Value]  # the registers' values at the back edge


def _iteration(
    lines: list[str], loop: _Loop, start: dict[str, int | bool], reduction: _Reduction
) -> _Iteration:
    state = _State(start)
    idle: bool | None = None
    quiet = True
    # the back edge too: a bottom-tested loop tests the contributions there
    for number in range(loop.head, loop.back + 1):
        if number == loop.exit.line:
            values = [state.get(c) for c in reduction.contributions]
            idle = None if any(type(v) is not bool for v in values) else not any(values)
        ins = _parse(lines, number)
        if ins is None or number in reduction.lines or number == loop.back:
            continue
        runs = _runs(state, ins)
        if _writes_memory(ins) and runs is not False:
            cas = ins.root == "atom" and ".cas" in ins.opcode and len(ins.operands) == 4
            if not (cas and state.get(ins.operands[2]) == state.get(ins.operands[3])):
                quiet = False
        if runs is True:
            _step(state, ins)
        elif runs is None:
            for register in ins.writes():
                state.set(register, _Unknown(register, number))
    return _Iteration(idle, quiet, state.values)


def _after(lines: list[str], loop: _Loop) -> set[str]:
    """The registers that the code after the loop reads."""
    return {
        r for n in range(loop.back + 1, len(lines)) if (i := _parse(lines, n)) for r in i.reads()
    }


def _equivalent(lines: list[str], loop: _Loop, reduction: _Reduction) -> bool:
    """Whether an iteration that starts with the thread's contributions false at the
    exit writes no memory, keeps them false, and keeps every register that the
    code after the loop reads, for every value of the carried predicates."""
    carried = _carried(lines, loop, reduction.lines)
    if len(carried) > CARRIED:
        return False
    constants = _constants(lines)
    after = _after(lines, loop)
    idle_states = 0
    for bits in range(1 << len(carried)):
        start = dict(constants) | {p: bool(bits >> k & 1) for k, p in enumerate(carried)}
        first = _iteration(lines, loop, start, reduction)
        if first.idle is None:
            return False
        if not first.idle:
            continue
        idle_states += 1
        following = {p: first.end.get(p) for p in carried}
        values = {p: v for p, v in following.items() if type(v) is bool}
        if len(values) != len(carried):
            return False
        second = _iteration(lines, loop, dict(constants) | values, reduction)
        # an idle thread leaves at the exit: the rest of a top-tested loop's
        # iteration, or the next iteration of a bottom-tested one, is what it skips
        skipped, begun = (second, dict(constants) | values) if loop.bottom else (first, start)
        if not skipped.quiet or second.idle is not True:
            return False
        for register in after:
            kept = skipped.end.get(register, _Unknown(register, None))
            if kept != begun.get(register, _Unknown(register, None)):
                return False
    return idle_states > 0


def _declare(lines: list[str], loop: _Loop) -> int:
    """The line of the kernel's predicate declaration, after which the rewrite
    declares its own."""
    for number in range(loop.head, -1, -1):
        if re.match(r"^\s*\.reg\s+\.pred\b", lines[number]):
            return number
    raise ValueError("the kernel declares no predicate registers")


def _rewrite_loop(lines: list[str], loop: _Loop, serial: int) -> list[str] | None:
    reduction = _reduction(lines, loop)
    if reduction is None:
        return None
    barriers = {i for i in range(loop.head, loop.back) if re.match(r"^\s*bar\.sync\b", lines[i])}
    removed = reduction.lines | barriers
    # the loop keeps no other shared memory access or collective once these go
    for number in range(loop.head, loop.back):
        ins = _parse(lines, number)
        if number in removed or ins is None:
            continue
        if ".shared" in ins.opcode or ins.root in _COLLECTIVE:
            return None
    # no instruction that stays reads a register that a removed one writes
    defined = {r for n in removed if (i := _parse(lines, n)) is not None for r in i.writes()}
    for number in range(len(lines)):
        ins = _parse(lines, number)
        if ins is None or number in removed or number == loop.exit.line:
            continue
        if defined & ins.reads():
            return None
    if not _equivalent(lines, loop, reduction):
        return None
    line = lines[loop.exit.line]
    indent = line[: len(line) - len(line.lstrip())]
    target = loop.exit.operands[0]
    if loop.bottom:
        # continue while some contribution holds
        if reduction.own is not None:
            return None
        contributions = list(reduction.contributions)
        current = contributions[0]
        code = []
        for k, c in enumerate(contributions[1:]):
            code.append(f"{indent}or.pred \t%opx{serial}_{k}, {current}, {c};")
            current = f"%opx{serial}_{k}"
        code.append(f"{indent}@{current} bra \t{target};")
        result = list(lines)
        for number in sorted(removed):
            result[number] = "\t// removed with the block reduction: " + lines[number].strip()
        result[loop.exit.line] = "\n".join(code)
        declared = _declare(lines, loop)
        count = max(1, len(contributions) - 1)
        result[declared] = result[declared] + f"\n\t.reg .pred \t%opx{serial}_<{count}>;"
        return "\n".join(result).split("\n")
    names = [f"%opx{serial}_{k}" for k in range(len(reduction.contributions) + 2)]
    # any: whether some contribution holds; the exit leaves when none holds, or own
    code: list[str] = []
    contributions = list(reduction.contributions)
    if len(contributions) == 1 and reduction.own is None:
        code.append(f"{indent}@!{contributions[0]} bra \t{target};")
    else:
        current = contributions[0]
        for k, c in enumerate(contributions[1:]):
            code.append(f"{indent}or.pred \t{names[k]}, {current}, {c};")
            current = names[k]
        if reduction.own is None:
            code.append(f"{indent}@!{current} bra \t{target};")
        else:
            code.append(f"{indent}not.pred \t{names[-2]}, {current};")
            code.append(f"{indent}or.pred \t{names[-1]}, {names[-2]}, {reduction.own};")
            code.append(f"{indent}@{names[-1]} bra \t{target};")
    result = list(lines)
    for number in sorted(removed):
        result[number] = "\t// removed with the block reduction: " + lines[number].strip()
    result[loop.exit.line] = "\n".join(code)
    declared = _declare(lines, loop)
    result[declared] = result[declared] + f"\n\t.reg .pred \t%opx{serial}_<{len(names)}>;"
    return "\n".join(result).split("\n")


_JUMPS = ("bra", "brx", "ret", "exit", "call", "trap", "brkpt")
_FULL_MASKS = ("-1", "0xffffffff", "0xFFFFFFFF", "4294967295")


def _event(ins: _Instruction) -> bool:
    return ins.root in _COLLECTIVE


def _full_mask(ins: _Instruction) -> bool:
    """Whether a barrier or warp collective names every thread: a `bar.sync` with
    an immediate id, or a collective whose member mask is all ones."""
    if ins.root == "bar":
        return (
            ins.opcode == "bar.sync"
            and len(ins.operands) == 1
            and _integer(ins.operands[0]) is not None
        )
    return bool(ins.operands) and ins.operands[-1] in _FULL_MASKS


def _region(lines: list[str], labels: dict[str, int], branch: int) -> tuple[int, int] | None:
    """The lines that a forward branch skips, if they hold no jump and no label
    that a branch from outside them targets."""
    ins = _parse(lines, branch)
    assert ins is not None and ins.operands
    end = labels.get(ins.operands[0])
    if end is None or end <= branch:
        return None
    inside = {name for name, line in labels.items() if branch < line < end}
    for number in range(len(lines)):
        other = _parse(lines, number)
        if other is None:
            continue
        if branch < number < end and other.root in _JUMPS:
            return None
        if (
            not branch < number < end
            and other.root in ("bra", "brx")
            and set(other.operands) & inside
        ):
            return None
    return branch + 1, end


def _straighten_one(lines: list[str], serial: int) -> list[str] | None:
    """`lines` with one forward branch before the last barrier or collective, or
    inside the lines that such a branch skips, replaced by guards on the loads,
    stores, and atomics that it skips; None if there is no branch that the pass can
    replace."""
    events = [n for n in range(len(lines)) if (i := _parse(lines, n)) is not None and _event(i)]
    if not events:
        return None
    labels = _labels(lines)
    forward = {
        n: labels[i.operands[0]]
        for n in range(len(lines))
        if (i := _parse(lines, n)) is not None
        and i.root == "bra"
        and i.guard is not None
        and labels.get(i.operands[0], -1) > n
    }
    # the branches before the last event, and those that their skipped lines hold
    early = [(n, end) for n, end in forward.items() if n < events[-1]]
    wanted = {n for n in forward if any(b <= n < end for b, end in early)}
    candidates: list[tuple[int, int, int]] = []
    for number in sorted(wanted):
        region = _region(lines, labels, number)
        if region is not None:
            candidates.append((region[1] - region[0], number, region[1]))
    for _, branch, end in sorted(candidates):
        replaced = _guard_region(lines, branch, end, serial)
        if replaced is not None:
            return replaced
    return None


def _guard_region(lines: list[str], branch: int, end: int, serial: int) -> list[str] | None:
    """The skipped lines run with each load, store, and atomic, and each instruction
    that writes a register that is read outside them, guarded by the condition under
    which the branch falls through; None if a register that a barrier or
    collective among them writes is read elsewhere, or such an instruction
    names fewer than every thread."""
    test = _parse(lines, branch)
    assert test is not None and test.guard is not None
    outside: set[str] = set()
    for number in range(len(lines)):
        if not branch < number < end and (ins := _parse(lines, number)) is not None:
            outside |= ins.reads()
    for number in range(branch + 1, end):
        ins = _parse(lines, number)
        if ins is None or not _event(ins):
            continue
        if ins.guard is not None or not _full_mask(ins) or ins.writes() & outside:
            return None
    line = lines[branch]
    indent = line[: len(line) - len(line.lstrip())]
    through = f"%opq{serial}"
    op = "mov.pred" if test.negated else "not.pred"
    result = list(lines)
    result[branch] = f"{indent}{op} \t{through}, {test.guard};"
    count = 0
    for number in range(branch + 1, end):
        ins = _parse(lines, number)
        if ins is None or _event(ins):
            continue
        if ins.root != "ld" and not _writes_memory(ins) and not ins.writes() & outside:
            continue
        text = lines[number]
        found = _INSTRUCTION.match(text.split("//", 1)[0])
        assert found is not None
        body = text[found.start(3) :]
        lead = text[: len(text) - len(text.lstrip())]
        if ins.guard is None:
            result[number] = f"{lead}@{through} {body}"
        else:
            guard = f"%opg{serial}_{count}"
            negate = f"%opn{serial}_{count}"
            count += 1
            source = ins.guard
            prefix = []
            if ins.negated:
                prefix.append(f"{lead}not.pred \t{negate}, {source};")
                source = negate
            prefix.append(f"{lead}and.pred \t{guard}, {through}, {source};")
            result[number] = "\n".join([*prefix, f"{lead}@{guard} {body}"])
    declared = next(n for n in range(branch, -1, -1) if re.match(r"^\s*\.reg\s+\.pred\b", lines[n]))
    names = [f"\t.reg .pred \t{through};"]
    if count:
        names += [
            f"\t.reg .pred \t%opg{serial}_<{count}>;",
            f"\t.reg .pred \t%opn{serial}_<{count}>;",
        ]
    result[declared] = "\n".join([result[declared], *names])
    return "\n".join(result).split("\n")


def _defined_by(
    lines: list[str], head: int, before: int, register: str, roots: tuple[str, ...]
) -> _Instruction | None:
    """The last definition of `register` in `[head, before)`, if its opcode's root
    is one of `roots`."""
    found = _definition(lines, head, before, register)
    if found is None or found.root not in roots:
        return None
    return found


def _midpoint(lines: list[str], head: int, line: int, mid: str, lo: str, hi: str) -> bool:
    """Whether `mid` is `(lo + hi) >> 1`, or `lo + ((hi - lo) >> 1)` with its sign
    correction, as each iteration computes it from the bounds it starts with."""
    shift = _defined_by(lines, head, line, mid, ("shr",))
    if shift is not None and _integer(shift.operands[2]) == 1:
        total = _defined_by(lines, head, shift.line, shift.operands[1], ("add",))
        return total is not None and set(total.operands[1:]) == {lo, hi}
    added = _defined_by(lines, head, line, mid, ("add",))
    if added is None or lo not in added.operands[1:]:
        return False
    (half,) = [o for o in added.operands[1:] if o != lo] or [""]
    shift = _defined_by(lines, head, added.line, half, ("shr",))
    if shift is None or _integer(shift.operands[2]) != 1:
        return False
    width = shift.operands[1]
    difference = _defined_by(lines, head, shift.line, width, ("sub", "add"))
    if difference is not None and difference.root == "add":
        # the difference plus its sign bit, which rounds the halving toward zero
        parts = [
            _defined_by(lines, head, difference.line, o, ("sub", "shr"))
            for o in difference.operands[1:]
        ]
        subs = [p for p in parts if p is not None and p.root == "sub"]
        difference = subs[0] if len(subs) == 1 else None
    return difference is not None and difference.operands[1:] == (hi, lo)


def _search(lines: list[str], head: int, back: int) -> tuple[str, int] | None:
    """For a binary search loop, `while lo < hi` with `mid` between and either
    `lo = mid + 1` or `hi = mid` each iteration, or `while lo <= hi` with
    either `lo = mid + 1` or `hi = mid - 1`, its continue predicate and the
    bits of its bounds: each iteration at least halves `hi - lo` (or
    `hi - lo + 1`), so it runs at most bits + 1 times."""
    test = _parse(lines, back)
    assert test is not None and test.guard is not None
    if test.negated:
        return None
    compare = _definition(lines, head, back, test.guard)
    if compare is None or not compare.opcode.startswith("setp.") or len(compare.operands) != 3:
        return None
    kind = compare.opcode.split(".")[1]
    a, b = compare.operands[1], compare.operands[2]
    inclusive = kind in ("le", "ls", "ge", "hs")
    if kind in ("lt", "lo", "le", "ls"):
        lo, hi = a, b
    elif kind in ("gt", "hi", "ge", "hs"):
        lo, hi = b, a
    else:
        return None
    new_lo = _defined_by(lines, head, compare.line, lo, ("selp",))
    new_hi = _defined_by(lines, head, compare.line, hi, ("selp",))
    if new_lo is None or new_hi is None or new_lo.operands[3] != new_hi.operands[3]:
        return None
    # one predicate picks either lo = mid + 1, or hi = mid
    if new_lo.operands[2] == lo and new_hi.operands[1] == hi:
        step, mid = new_lo.operands[1], new_hi.operands[2]
    elif new_lo.operands[1] == lo and new_hi.operands[2] == hi:
        step, mid = new_lo.operands[2], new_hi.operands[1]
    else:
        return None
    if inclusive:
        # the upper bound moves to mid - 1
        decrement = _defined_by(lines, head, new_hi.line, mid, ("add",))
        if decrement is None or decrement.operands[2] != "-1":
            return None
        mid = decrement.operands[1]
    increment = _defined_by(lines, head, new_lo.line, step, ("add",))
    if increment is None or sorted(increment.operands[1:]) != sorted((mid, "1")):
        return None
    if not _midpoint(lines, head, increment.line, mid, lo, hi):
        return None
    # the selections are the bounds' only writes in the loop
    for number in range(head, back):
        ins = _parse(lines, number)
        if ins is not None and {lo, hi} & ins.writes() and number not in (new_lo.line, new_hi.line):
            return None
    return test.guard, _width(compare.opcode)


def _bound_search(lines: list[str], serial: int) -> list[str] | None:
    """`lines` with one binary search loop that holds no barrier or collective
    given an iteration count that ends it after bits + 2 iterations, which a
    search never reaches, so that the gate can prove that the loop ends."""
    labels = _labels(lines)
    for number in range(len(lines)):
        ins = _parse(lines, number)
        if ins is None or ins.root != "bra" or ins.guard is None or not ins.operands:
            continue
        head = labels.get(ins.operands[0])
        if head is None or head >= number:
            continue
        inner = [_parse(lines, i) for i in range(head + 1, number)]
        if any(
            i is not None and (_event(i) or i.root in ("ret", "exit", "call", "brx")) for i in inner
        ):
            continue
        if any(
            i is not None and i.root == "bra" and labels.get(i.operands[0], -1) <= head
            for i in inner
        ):
            continue
        # the header is entered by falling into it, and by the back edge
        label = ins.operands[0]
        if any(
            (o := _parse(lines, n)) is not None
            and o.root == "bra"
            and label in o.operands
            and n != number
            for n in range(len(lines))
        ):
            continue
        found = _search(lines, head, number)
        if found is None:
            continue
        guard, bits = found
        count, below, both = f"%opk{serial}", f"%opkp{serial}", f"%opkq{serial}"
        result = list(lines)
        result[head] = f"\tmov.u32 \t{count}, 0;\n{lines[head]}"
        result[number] = "\n".join(
            [
                f"\tadd.s32 \t{count}, {count}, 1;",
                f"\tsetp.lt.s32 \t{below}, {count}, {bits + 2};",
                f"\tand.pred \t{both}, {guard}, {below};",
                f"\t@{both} bra \t{label};",
            ]
        )
        declared = next(
            n for n in range(head, -1, -1) if re.match(r"^\s*\.reg\s+\.pred\b", lines[n])
        )
        result[declared] = "\n".join(
            [
                result[declared],
                f"\t.reg .pred \t{below};",
                f"\t.reg .pred \t{both};",
                f"\t.reg .b32 \t{count};",
            ]
        )
        return "\n".join(result).split("\n")
    return None


def _entry(lines: list[str]) -> tuple[int, int]:
    """The lines of the entry kernel's body, between its braces."""
    head = next(n for n, line in enumerate(lines) if re.search(r"\.entry\s", line))
    opening = next(n for n in range(head, len(lines)) if lines[n].strip() == "{")
    depth = 0
    for closing in range(opening, len(lines)):
        depth += lines[closing].count("{") - lines[closing].count("}")
        if depth == 0:
            return opening, closing
    raise ValueError("the entry kernel's body does not close")


class _Wide:
    """Lines that compute 32-bit grid values in 64 bits, into fresh registers."""

    def __init__(self, serial: int, once: dict[str, _Instruction], special: dict[str, str]) -> None:
        self.serial = serial
        self.once = once
        self.special = special
        self.registers: list[str] = []

    def fresh(self) -> str:
        register = f"%opw{self.serial}_{len(self.registers)}"
        self.registers.append(register)
        return register

    def value(self, register: str, lead: str) -> tuple[list[str], str] | None:
        """The lines that compute `register`'s grid value in 64 bits, and the
        register that holds it: a special register, a product of two, a special
        register shifted left by a constant, and sums of these."""
        if register in self.special:
            wide = self.fresh()
            return [f"{lead}cvt.u64.u32 \t{wide}, {register};"], wide
        ins = self.once.get(register)
        if ins is None or ins.guard is not None:
            return None
        ops = ins.operands
        if ins.opcode in ("mul.lo.s32", "mul.lo.u32") and all(o in self.special for o in ops[1:3]):
            wide = self.fresh()
            return [f"{lead}mul.wide.u32 \t{wide}, {ops[1]}, {ops[2]};"], wide
        if (
            ins.opcode == "shl.b32"
            and ops[1] in self.special
            and (k := _integer(ops[2])) is not None
        ):
            wide = self.fresh()
            return [
                f"{lead}cvt.u64.u32 \t{wide}, {ops[1]};",
                f"{lead}shl.b64 \t{wide}, {wide}, {k};",
            ], wide
        if ins.opcode in ("mad.lo.s32", "mad.lo.u32") and all(o in self.special for o in ops[1:3]):
            addend = self.value(ops[3], lead)
            if addend is None:
                return None
            wide = self.fresh()
            return [
                *addend[0],
                f"{lead}mul.wide.u32 \t{wide}, {ops[1]}, {ops[2]};",
                f"{lead}add.s64 \t{wide}, {wide}, {addend[1]};",
            ], wide
        if ins.opcode in ("add.s32", "add.u32"):
            left, right = self.value(ops[1], lead), self.value(ops[2], lead)
            if left is None or right is None:
                return None
            wide = self.fresh()
            return [*left[0], *right[0], f"{lead}add.s64 \t{wide}, {left[1]}, {right[1]};"], wide
        return None


def _widen_indices(lines: list[str], serial: int) -> list[str] | None:
    """`lines` with each conversion to 64 bits of a grid stride or grid index
    that the kernel computes in 32 bits from `%ntid.x`, `%nctaid.x`,
    `%ctaid.x`, and `%tid.x` (products, constant shifts, and sums) replaced by
    its computation in 64 bits; None if there is none. A launch with fewer than
    2^31 threads in all gets the same values; with more, the 32-bit values
    wrap, and a grid stride may be zero, so that a loop that steps by it does
    not end."""
    opening, closing = _entry(lines)
    counts: dict[str, int] = {}
    defined: dict[str, _Instruction] = {}
    special: dict[str, str] = {}
    for number in range(opening, closing):
        ins = _parse(lines, number)
        if ins is None:
            continue
        for register in ins.writes():
            counts[register] = counts.get(register, 0) + 1
            defined[register] = ins
        if ins.root == "mov" and len(ins.operands) == 2 and ins.destination:
            if re.fullmatch(r"%(n?ctaid|n?tid)\.x", ins.operands[1]):
                special[ins.destination] = ins.operands[1]
    once = {r: d for r, d in defined.items() if counts[r] == 1}
    special = {r: v for r, v in special.items() if counts[r] == 1}
    wide = _Wide(serial, once, special)
    result = list(lines)
    for number in range(opening, closing):
        cvt = _parse(lines, number)
        if cvt is None or cvt.guard is not None or cvt.opcode not in ("cvt.u64.u32", "cvt.s64.s32"):
            continue
        if cvt.operands[1] in special:
            continue
        lead = lines[number][: len(lines[number]) - len(lines[number].lstrip())]
        before = len(wide.registers)
        found = wide.value(cvt.operands[1], lead)
        if found is None:
            del wide.registers[before:]
            continue
        code, register = found
        names = {
            special[r]
            for r in special
            if any(re.search(rf"{re.escape(r)}\b", line) for line in code)
        }
        if not names & {"%nctaid.x", "%ctaid.x"}:
            del wide.registers[before:]
            continue
        result[number] = "\n".join([*code, f"{lead}mov.b64 \t{cvt.operands[0]}, {register};"])
    if result == lines:
        return None
    at = next(n for n in range(opening, closing) if re.match(r"^\s*\.reg\s", lines[n]))
    result[at] = "\n".join([result[at], *(f"\t.reg .b64 \t{r};" for r in wide.registers)])
    return "\n".join(result).split("\n")


def _register_strides(lines: list[str], serial: int) -> list[str] | None:
    """`lines` with each loop that counts in 64 bits by an immediate, `c = c + k;
    if (c < n) goto head`, counting by a register that holds `k`; None if there
    is none. The values are the same; the gate summarizes a loop whose
    counter's stride is a register."""
    opening, closing = _entry(lines)
    labels = _labels(lines)
    result = list(lines)
    strides: list[tuple[str, int]] = []
    for number in range(opening, closing):
        back = _parse(lines, number)
        if back is None or back.root != "bra" or back.guard is None or back.negated:
            continue
        head = labels.get(back.operands[0], closing)
        if head >= number:
            continue
        start = max(n for n in range(head, number) if _LABEL.match(lines[n]))
        test = _definition(lines, start, number, back.guard)
        if test is None or test.opcode not in ("setp.lt.s64", "setp.lt.u64"):
            continue
        counter = test.operands[1]
        step = _definition(lines, start, test.line, counter)
        if step is None or step.opcode not in ("add.s64", "add.u64") or step.guard is not None:
            continue
        stride = _integer(step.operands[2])
        if stride is None or stride <= 0 or _integer(step.operands[1]) is not None:
            continue
        register = f"%ops{serial}_{len(strides)}"
        strides.append((register, stride))
        lead = lines[step.line][: len(lines[step.line]) - len(lines[step.line].lstrip())]
        result[step.line] = f"{lead}{step.opcode} \t{counter}, {step.operands[1]}, {register};"
    if not strides:
        return None
    at = next(n for n in range(opening, closing) if re.match(r"^\s*\.reg\s", lines[n]))
    result[at] = "\n".join([result[at], *(f"\t.reg .b64 \t{r};" for r, _ in strides)])
    first = next(
        n
        for n in range(opening + 1, closing)
        if _parse(lines, n) is not None and not lines[n].strip().startswith(".")
    )
    result[first] = "\n".join([*(f"\tmov.u64 \t{r}, {k};" for r, k in strides), result[first]])
    return "\n".join(result).split("\n")


def _widen_counters(lines: list[str], serial: int) -> list[str] | None:
    """`lines` with each loop that counts in 32 bits, `c = c + s; if (c < n) goto
    head` with `s` a positive constant or the block's width, given a 64-bit
    copy of `c` that the loop's test compares with a 64-bit copy of `n`; None
    if there is none. While `c + s` does not pass 2^31 (2^32 unsigned) the
    tests agree; past it the 32-bit counter wraps, a signed overflow that C
    leaves undefined, and the loop may not end, while the 64-bit test ends it."""
    opening, closing = _entry(lines)
    labels = _labels(lines)
    special = {
        i.destination: i.operands[1]
        for n in range(opening, closing)
        if (i := _parse(lines, n)) is not None
        and i.root == "mov"
        and len(i.operands) == 2
        and i.destination is not None
        and i.operands[1] == "%ntid.x"
    }
    result = list(lines)
    declared: list[str] = []
    for number in range(opening, closing):
        back = _parse(lines, number)
        if back is None or back.root != "bra" or back.guard is None or back.negated:
            continue
        label = back.operands[0]
        head = labels.get(label, closing)
        if head >= number or any(
            (o := _parse(lines, n)) is not None
            and o.root == "bra"
            and label in o.operands
            and n != number
            for n in range(opening, closing)
        ):
            continue
        start = max(n for n in range(head, number) if _LABEL.match(lines[n]))
        test = _definition(lines, start, number, back.guard)
        if test is None or test.opcode not in ("setp.lt.s32", "setp.lt.u32"):
            continue
        counter, limit = test.operands[1], test.operands[2]
        step = _definition(lines, start, test.line, counter)
        if step is None or step.opcode not in ("add.s32", "add.u32") or step.guard is not None:
            continue
        if step.operands[1] != counter:
            continue
        stride = step.operands[2]
        constant = _integer(stride)
        if not ((constant is not None and constant > 0) or stride in special):
            continue
        writes = [
            n
            for n in range(head, number)
            if (w := _parse(lines, n)) is not None and (w.writes() & {counter, limit, stride})
        ]
        if writes != [step.line]:
            continue
        signed = test.opcode.endswith("s32")
        widen = "cvt.s64.s32" if signed else "cvt.u64.u32"
        wide = [f"%opc{serial}_{len(declared) + k}" for k in range(3)]
        declared += wide
        count, bound, by = wide
        size = (
            f"mov.u64 \t{by}, {constant};"
            if constant is not None
            else f"cvt.u64.u32 \t{by}, {stride};"
        )
        result[head] = "\n".join(
            [
                f"\t{widen} \t{count}, {counter};",
                f"\t{widen} \t{bound}, {limit};",
                f"\t{size}",
                lines[head],
            ]
        )
        result[step.line] = "\n".join([lines[step.line], f"\tadd.s64 \t{count}, {count}, {by};"])
        result[test.line] = "\t// widened into the 64-bit test: " + lines[test.line].strip()
        kind = "s64" if signed else "u64"
        result[number] = "\n".join(
            [f"\tsetp.lt.{kind} \t{back.guard}, {count}, {bound};", lines[number]]
        )
    if not declared:
        return None
    at = next(n for n in range(opening, closing) if re.match(r"^\s*\.reg\s", lines[n]))
    result[at] = "\n".join([result[at], *(f"\t.reg .b64 \t{r};" for r in declared)])
    return "\n".join(result).split("\n")


def rewrite(text: str, block: tuple[int, int, int]) -> str | None:
    """`text`, of a kernel launched with blocks of `block` threads, with the
    rewrites above: grid values computed in 64 bits, 32-bit counted loops
    tested in 64 bits, immediate loop strides held in registers, loops
    that end on a block reduction ended on each thread's own predicates,
    forward branches before the last barrier or collective replaced by guards,
    and binary searches bounded; None if there is nothing to rewrite."""
    lines = text.split("\n")
    serial = 0
    if (widened := _widen_indices(lines, serial)) is not None:
        lines, serial = widened, serial + 1
    if (counted := _widen_counters(lines, serial)) is not None:
        lines, serial = counted, serial + 1
    if (strided := _register_strides(lines, serial)) is not None:
        lines, serial = strided, serial + 1
    while True:
        for loop in _loops(lines):
            rewritten = _rewrite_loop(lines, loop, serial)
            if rewritten is not None:
                lines, serial = rewritten, serial + 1
                break
        else:
            break
    while (straightened := _straighten_one(lines, serial)) is not None:
        lines, serial = straightened, serial + 1
    while (bounded := _bound_search(lines, serial)) is not None:
        lines, serial = bounded, serial + 1
    return "\n".join(lines) if serial else None
