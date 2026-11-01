"""From a corpus kernel's `signature.json` and its reviewed roles to a signature and a binding.

The models wrote each kernel's `signature.json` in a small vocabulary: each
argument has a kind (`input`, `output`, `inout`, `scalar`), a dtype, and a
length expression. A buffer's length is a sum of size names and constants,
each possibly times a constant factor, such as `n + 1`, `m + p`, or `2 * n`,
or a size symbol named after the buffer: for
`offsets[n]`, the count of values that the last offset of `offsets` holds,
and for a quotient such as `(n + 7) / 8`, the bytes of an Arrow bitmap,
which is not affine in the sizes, so the signature states nothing that
relates it to the other sizes. Buffers with the same such length, such as
a string column's characters and its output of the same bytes, share the
size symbol of the first of them.
A scalar's length is free text that can state its bounds as `[lo, hi]` or
`[lo, inf)`. The adapter reads the dtype and the length from this file.

The role of each argument, and each size symbol with its least value, come
from the reviewed table `roles.tsv` of the model's directory, not from the argument names or the
models' kinds: a model's `output` can be an accumulator that the kernel also
reads, and its `indexes_into` can name a buffer of another kernel. Each row
names a model, a task, an argument, a role, a target, and a minimum, with `-`
for a field that the role does not use:

- `read`, `write`, `readwrite`: a buffer that the kernel reads, writes, or both.
- `index-in`, `index-out`: a buffer of indices into the target buffer, which
  the kernel reads or writes.
- `offsets`: a buffer of offsets into the target buffer, whose last offset is
  the target's length.
- `size`: a scalar that is the size symbol of the same name.
- `scalar`: any other scalar, bounded by its length text or by its bits.
- `symbol`: a size symbol, with its least value in the minimum field. The
  target field `derived` states that the symbol, the length of a buffer
  given as a quotient `(a) / d` of an affine `a` of earlier size symbols and
  a positive constant `d`, equals floor(a / d) at every launch, as the plan
  computes it: the signature then has that derived size (design decision
  24), which the catalog checks at each launch. Without it the quotient is
  an independent symbol.

`claim_of` gives a kernel's signature and element types, and `binding_of`
derives its binding for one of the corpus's toolchains: Numba, nvcc, or
Triton.

The corpus has two collections, each with its own roles and decision tables:
the synthesized collection, whose kernels four Claude models wrote for the
task lists of this repository, and the DataKernelBench collection, whose
kernels come from the TPC-H query plans that seven other model families
wrote for that benchmark. A kernel is named by its model and its task, and
the model names of the two collections differ.
"""

import csv
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never

from opgate import signature as sig
from opgate.binding import (
    GRID_LIMIT,
    Binding,
    Convention,
    ElementType,
    validate_grid,
    validate_shared,
)
from opgate.frontends import Operator, entry_kernels
from opgate.signature import Affine, Direction

ELEMENTS: dict[str, ElementType] = {
    "float64": ElementType(8, False),
    "int64": ElementType(8, True),
    "uint64": ElementType(8, False),
    "int32": ElementType(4, True),
    "uint32": ElementType(4, False),
    "float32": ElementType(4, False),
    "int16": ElementType(2, True),
    "uint16": ElementType(2, False),
    "int8": ElementType(1, True),
    "uint8": ElementType(1, False),
    "bool": ElementType(1, False),
}

ROLE_COLUMNS = ("model", "task", "argument", "role", "target", "minimum")
Block = tuple[int, int, int]


@dataclass(frozen=True)
class Collection:
    """A collection of corpus kernels: `root/<model>/<task>/`, with the reviewed
    roles of each model's kernels in `root/<model>/roles.tsv` and the decisions
    in `root/decisions.tsv`.
    `block` is the block of every Numba and nvcc launch when the collection's
    protocol fixes it; otherwise each nvcc kernel states its block in its
    `launch.json`."""

    root: Path
    models: tuple[str, ...]
    block: Block | None

    @property
    def decisions(self) -> Path:
        return self.root / "decisions.tsv"


# the kernels that four Claude models wrote for the task lists of this
# repository, each test launching 256 threads per block
SYNTHESIZED = Collection(
    Path(__file__).resolve().parent / "synthesized",
    ("fable", "haiku", "opus", "sonnet"),
    (256, 1, 1),
)
# the kernels of the TPC-H query plans that seven other model families wrote
# for DataKernelBench, launched as their plans launch them
DATAKERNELBENCH = Collection(
    Path(__file__).resolve().parent / "datakernelbench",
    (
        "deepseek-v4-flash",
        "devstral-2-123b",
        "gemini-3.1-pro",
        "gpt-5.5",
        "gpt-oss-120b",
        "minimax-m2.5",
        "qwen3.5-397b",
    ),
    None,
)
COLLECTIONS = (SYNTHESIZED, DATAKERNELBENCH)
CONVENTIONS = {"numba": Convention.NUMBA, "nvcc": Convention.C, "triton": Convention.TRITON}
REQUIRED_THREADS = re.compile(r"^\s*\.reqntid\s+(\d+)(?:\s*,\s*1){0,2}\s*$", re.M)
TERM = re.compile(r"^\s*(?:(\d+)\s*\*\s*)?(?:([A-Za-z_]\w*)|(\d+))(?:\s*\*\s*(\d+))?\s*$")
VALUES_OF = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*[A-Za-z_]\w*\s*\]\s*$")
QUOTIENT = re.compile(r"^[\w\s+*()]*/[\w\s+*/()]*$")
FLOOR_QUOTIENT = re.compile(r"^\s*\(([^()]*)\)\s*/\s*(\d+)\s*$")
INTERVAL = re.compile(r"^\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$")
LOWER_ONLY = re.compile(r"^\s*\[\s*(-?\d+)\s*,\s*inf\s*\)\s*$")


class Role(Enum):
    """The role of an argument without a target, as `roles.tsv` names it."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "readwrite"
    SIZE = "size"
    SCALAR = "scalar"


class Pointer(Enum):
    """The role of a buffer that points into a target buffer, as `roles.tsv` names it."""

    INDEX_IN = "index-in"
    INDEX_OUT = "index-out"
    OFFSETS = "offsets"


@dataclass(frozen=True)
class Points:
    """A buffer of indices or offsets into the buffer `target`."""

    pointer: Pointer
    target: str


Assigned = Role | Points
POINTERS = {pointer.value for pointer in Pointer}


@dataclass(frozen=True)
class Roles:
    """The reviewed roles of one kernel: each argument's role, each size
    symbol, in the table's order, with its least value, and the symbols that
    equal the quotient their buffer's length states."""

    arguments: dict[str, Assigned]
    symbols: dict[str, int]
    derived: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RawArgument:
    name: str
    kind: str
    dtype: str
    length: str


def collection_of(model: str) -> Collection:
    """The collection whose kernels `model` wrote."""
    (collection,) = (c for c in COLLECTIONS if model in c.models)
    return collection


def directory_of(model: str, task: str) -> Path:
    """The directory of the kernel that `model` wrote for `task`."""
    return collection_of(model).root / model / task


def read_raw(path: Path) -> tuple[RawArgument, ...]:
    """The arguments of a `signature.json`, as written."""
    loaded: object = json.loads(path.read_text())
    if not isinstance(loaded, list):
        raise ValueError(f"{path}: not a list")
    raws: list[RawArgument] = []
    for entry in loaded:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: an entry is not an object")
        name, kind, dtype = entry["name"], entry["kind"], entry["dtype"]
        if not (isinstance(name, str) and isinstance(kind, str) and isinstance(dtype, str)):
            raise ValueError(f"{path}: {entry} has a field of the wrong type")
        raws.append(RawArgument(name, kind, dtype, str(entry["length"])))
    return tuple(raws)


def read_roles(model: str, task: str) -> Roles:
    """The rows of `roles.tsv` for the kernel of `model` and `task`."""
    arguments: dict[str, Assigned] = {}
    symbols: dict[str, int] = {}
    derived: set[str] = set()
    table = collection_of(model).root / model / "roles.tsv"
    with table.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ROLE_COLUMNS:
            raise ValueError(f"{table}: the columns are not {ROLE_COLUMNS}")
        for line in reader:
            if (line["model"], line["task"]) != (model, task):
                continue
            name, target, minimum = line["argument"], line["target"], line["minimum"]
            where = f"{table}: {model}/{task} {name} {line['role']}"
            if line["role"] == "symbol":
                if target not in ("-", "derived") or name in symbols:
                    raise ValueError(f"{where}: a symbol row with a target, or a second one")
                symbols[name] = int(minimum)
                if target == "derived":
                    derived.add(name)
                continue
            if minimum != "-" or name in arguments:
                raise ValueError(f"{where}: an argument row with a minimum, or a second one")
            pointer = line["role"] in POINTERS
            if (target != "-") != pointer:
                raise ValueError(f"{where}: the role and the target field disagree")
            if pointer:
                arguments[name] = Points(Pointer(line["role"]), target)
            else:
                arguments[name] = Role(line["role"])
    if not arguments:
        raise ValueError(f"{table}: no rows for {model}/{task}")
    return Roles(arguments, symbols, frozenset(derived))


def _affine(name: str, text: str) -> tuple[dict[str, int], int]:
    """A sum of terms: the size symbols it adds, each with its coefficient,
    and a constant to add. A term is a size name or a constant, either times a
    constant factor, such as `2 * n`."""
    symbols: dict[str, int] = {}
    constant = 0
    for term in text.split("+"):
        matched = TERM.match(term)
        if matched is None:
            raise ValueError(f"{name}: a length {text!r} the adapter does not read")
        factor = int(matched.group(1) or 1) * int(matched.group(4) or 1)
        if matched.group(2) is not None:
            symbols[matched.group(2)] = symbols.get(matched.group(2), 0) + factor
        else:
            constant += factor * int(matched.group(3))
    return symbols, constant


def _length(raw: RawArgument, named: dict[str, str]) -> tuple[dict[str, int], int]:
    """A buffer's length: an affine sum of size symbols and a constant. A
    length that is not affine is the size symbol that `named` gives it, the
    name of the first buffer with it."""
    if VALUES_OF.match(raw.length) or QUOTIENT.match(raw.length):
        return {named.setdefault(" ".join(raw.length.split()), raw.name): 1}, 0
    return _affine(raw.name, raw.length)


def _scalar(raw: RawArgument) -> sig.Between:
    """A scalar's bound on its bits, which the checker reads unsigned: the
    interval that the length states of an integer, and every bit pattern of
    any other integer and of a float."""
    element = ELEMENTS[raw.dtype]
    bits = 8 * element.width
    interval = INTERVAL.match(raw.length)
    lower = LOWER_ONLY.match(raw.length)
    if raw.dtype.startswith("float"):
        return sig.Between(0, 2**bits - 1)
    if interval is not None:
        lo, hi = int(interval.group(1)), int(interval.group(2))
    elif lower is not None and element.signed:
        lo, hi = int(lower.group(1)), 2 ** (bits - 1) - 1
    else:
        return sig.Between(0, 2**bits - 1)
    # a negative value's bits are 2^bits plus it
    if lo >= 0:
        return sig.Between(lo, hi)
    if hi < 0:
        return sig.Between(lo + 2**bits, hi + 2**bits)
    return sig.Between(0, 2**bits - 1)


def signature_of(
    raws: tuple[RawArgument, ...], roles: Roles
) -> tuple[sig.Signature, dict[str, ElementType]]:
    """The signature that `signature.json` and the reviewed roles state, and each buffer's element type."""
    names = sorted(raw.name for raw in raws)
    if names != sorted(roles.arguments):
        raise ValueError(f"the arguments {names} and the rows {sorted(roles.arguments)} differ")
    for raw in raws:
        assigned = roles.arguments[raw.name]
        if (raw.kind == "scalar") != (assigned in (Role.SIZE, Role.SCALAR)):
            raise ValueError(f"{raw.name}: the kind {raw.kind} and the role {assigned} disagree")
    shared: dict[str, str] = {}
    lengths = {raw.name: _length(raw, shared) for raw in raws if raw.kind != "scalar"}
    named = {symbol for added, _ in lengths.values() for symbol in added}
    named |= {raw.name for raw in raws if roles.arguments[raw.name] is Role.SIZE}
    if named != set(roles.symbols):
        raise ValueError(f"the named sizes {sorted(named)} and the symbol rows differ")
    symbols = tuple(roles.symbols)

    def rows(name: str) -> Affine:
        added, constant = lengths[name]
        return Affine(constant, tuple(sorted((symbols.index(s), c) for s, c in added.items())))

    arguments: list[sig.Argument] = []
    elements: dict[str, ElementType] = {}
    for raw in raws:
        assigned = roles.arguments[raw.name]
        if isinstance(assigned, Points) and assigned.target not in lengths:
            raise ValueError(f"{raw.name}: the target {assigned.target} is not a buffer argument")
        if raw.kind != "scalar":
            elements[raw.name] = ELEMENTS[raw.dtype]
        match assigned:
            case Role.READ:
                arguments.append(sig.Span(raw.name, raw.name, Direction.READ, rows(raw.name)))
            case Role.WRITE:
                arguments.append(sig.Span(raw.name, raw.name, Direction.WRITE, rows(raw.name)))
            case Role.READ_WRITE:
                arguments.append(sig.Span(raw.name, raw.name, Direction.READ_WRITE, rows(raw.name)))
            case Role.SIZE:
                size = Affine.symbol(symbols.index(raw.name))
                arguments.append(sig.Scalar(raw.name, sig.SizeOf(size)))
            case Role.SCALAR:
                arguments.append(sig.Scalar(raw.name, _scalar(raw)))
            case Points(pointer, target):
                match pointer:
                    case Pointer.INDEX_IN:
                        arguments.append(sig.IndexIn(raw.name, raw.name, rows(raw.name), target))
                    case Pointer.INDEX_OUT:
                        arguments.append(sig.IndexOut(raw.name, raw.name, rows(raw.name), target))
                    # an offsets buffer reaches the end of its target: the last
                    # offset of a string column is its character count, and an
                    # exclusive prefix sum reaches the total (design section 10.3)
                    case Pointer.OFFSETS:
                        arguments.append(sig.Offsets(raw.name, raw.name, rows(raw.name), target))
                    case _ as unreachable:
                        assert_never(unreachable)
            case _ as unreachable:
                assert_never(unreachable)
    sizes = tuple(sig.SizeSymbol(name, minimum) for name, minimum in roles.symbols.items())
    return sig.Signature(sizes, tuple(arguments), _derived(raws, roles, shared, symbols)), elements


def _derived(
    raws: tuple[RawArgument, ...], roles: Roles, shared: dict[str, str], symbols: tuple[str, ...]
) -> tuple[sig.DerivedSize, ...]:
    """The derived size of each symbol that the roles mark `derived`: the
    floor quotient `(a) / d` that the length of its buffer states."""
    quotients = {name: text for text, name in shared.items()}
    derived = []
    for name in sorted(roles.derived, key=symbols.index):
        if name not in quotients:
            raise ValueError(f"{name}: a derived symbol that no buffer's quotient length names")
        matched = FLOOR_QUOTIENT.match(quotients[name])
        if matched is None:
            raise ValueError(f"{name}: the length {quotients[name]!r} is not a quotient (a) / d")
        added, constant = _affine(name, matched.group(1))
        numerator = Affine(constant, tuple(sorted((symbols.index(s), c) for s, c in added.items())))
        derived.append(sig.DerivedSize(symbols.index(name), numerator, int(matched.group(2))))
    return tuple(derived)


def claim_of(model: str, task: str) -> tuple[sig.Signature, dict[str, ElementType]]:
    """The signature of the corpus kernel of `model` and `task`, and each buffer's element type."""
    raws = read_raw(directory_of(model, task) / "signature.json")
    return signature_of(raws, read_roles(model, task))


def _block(path: Path, value: object) -> Block:
    """A block that `launch.json` states: three positive dimensions of at most
    1024 threads in all."""
    match value:
        case [int(x), int(y), int(z)] if x > 0 and y > 0 and z > 0 and x * y * z <= 1024:
            return (x, y, z)
        case _:
            raise ValueError(f"{path}: {value!r} is not a block")


Launch = tuple[sig.Signature, Affine, Block, Block]
"""A kernel's launch claim: its signature, its dynamic shared memory, its block, and its grid maxima."""


def grid_of(path: Path, launch: dict[str, object]) -> Block:
    """The grid maxima that a launch declaration states, or the GPU's limit."""
    if "grid" not in launch:
        return GRID_LIMIT
    match launch["grid"]:
        case [int(x), int(y), int(z)]:
            grid = (x, y, z)
        case other:
            raise ValueError(f"{path}: {other!r} is not a grid")
    validate_grid(grid)
    return grid


def declared_launch(
    directory: Path, toolchain: str, signature: sig.Signature, fixed: Block | None
) -> Launch:
    """The launch of the Numba or nvcc kernel in `directory`: the signature with
    the size maxima that `launch.json` declares, its dynamic shared memory, its
    block, and its grid maxima, the GPU's limit unless `launch.json` states
    `grid`. `fixed` is the collection's block, which a declaration does not
    restate; without one, the declaration states the block."""
    path = directory / "launch.json"
    if not path.exists():
        if fixed is None:
            raise ValueError(f"{directory}: no launch.json states the block")
        return signature, Affine(0), fixed, GRID_LIMIT
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict) or not set(loaded) <= {"numba", "nvcc"}:
        raise ValueError(f"{path}: invalid launch toolchains")
    fields = {"shared", "size_maxima"} if fixed is not None else {"block", "shared", "size_maxima"}
    names = tuple(size.name for size in signature.sizes)
    contracts: dict[str, Launch] = {}
    for name, launch in loaded.items():
        if not isinstance(launch, dict) or not fields <= set(launch) <= fields | {"grid"}:
            raise ValueError(
                f"{path}: the declaration for {name} has not the fields {sorted(fields)}"
            )
        maxima = launch["size_maxima"]
        extent = launch["shared"]
        if not isinstance(maxima, dict) or not set(maxima) <= set(names):
            raise ValueError(f"{path}: undeclared size maximum for {name}")
        if not isinstance(extent, dict) or set(extent) != {"constant", "terms"}:
            raise ValueError(f"{path}: invalid shared extent for {name}")
        constant, terms = extent["constant"], extent["terms"]
        if type(constant) is not int or not isinstance(terms, dict) or not set(terms) <= set(names):
            raise ValueError(f"{path}: invalid shared expression for {name}")
        if any(type(c) is not int or c == 0 for c in terms.values()):
            raise ValueError(f"{path}: invalid shared coefficient for {name}")
        sizes = []
        for size in signature.sizes:
            maximum = maxima.get(size.name, size.maximum)
            if type(maximum) is not int or not size.minimum <= maximum <= size.maximum:
                raise ValueError(f"{path}: invalid maximum for {size.name}")
            sizes.append(sig.SizeSymbol(size.name, size.minimum, maximum))
        revised = sig.Signature(tuple(sizes), signature.arguments, signature.derived_sizes)
        shared = Affine(constant, tuple(sorted((names.index(k), c) for k, c in terms.items())))
        validate_shared(revised.sizes, shared)
        block = fixed if fixed is not None else _block(path, launch["block"])
        contracts[name] = revised, shared, block, grid_of(path, launch)
    if toolchain in contracts:
        return contracts[toolchain]
    if fixed is None:
        raise ValueError(f"{path}: no launch of {toolchain}")
    return signature, Affine(0), fixed, GRID_LIMIT


def launch_of(model: str, task: str, toolchain: str, signature: sig.Signature, text: str) -> Launch:
    """The launch of the corpus kernel of `model` and `task` whose PTX is
    `text`: for Numba and nvcc, the declared one; for Triton, the dynamic
    shared memory that Triton states in `triton.json`, the grid maxima it
    states as `grid` or the GPU's limit, and the `32 * num_warps` threads that
    its PTX requires."""
    directory = directory_of(model, task)
    match toolchain:
        case "numba" | "nvcc":
            return declared_launch(directory, toolchain, signature, collection_of(model).block)
        case "triton":
            path = directory / "triton.json"
            loaded = json.loads(path.read_text())
            if not isinstance(loaded, dict) or type(loaded.get("shared")) is not int:
                raise ValueError(f"{path}: no dynamic shared memory declaration")
            if not set(loaded) <= {"shared", "grid"}:
                raise ValueError(f"{path}: fields other than shared and grid")
            shared = Affine(loaded["shared"])
            validate_shared(signature.sizes, shared)
            required = REQUIRED_THREADS.findall(text)
            if len(required) != 1:
                raise ValueError(f"a Triton text with {len(required)} `.reqntid` directives")
            return signature, shared, (int(required[0]), 1, 1), grid_of(path, loaded)
        case _:
            raise ValueError(f"no toolchain {toolchain}")


def binding_of(model: str, task: str, toolchain: str, text: str) -> Binding:
    """The binding of the corpus kernel of `model` and `task`, the one entry
    kernel of `text`, whose toolchain is `numba`, `nvcc`, or `triton`."""
    kernels = entry_kernels(text)
    if len(kernels) != 1:
        raise ValueError(f"{model}/{task}: {len(kernels)} entry kernels, not one")
    signature, elements = claim_of(model, task)
    signature, shared, block, grid = launch_of(model, task, toolchain, signature, text)
    operator = Operator(text, kernels[0], signature, CONVENTIONS[toolchain], block, grid, shared)
    return operator.binding(elements)
