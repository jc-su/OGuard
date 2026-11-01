"""The binding: the symbolic statement of an operator's launches.

`derive_binding` turns a signature, the element types of the plan's columns,
and a toolchain's calling convention into a binding: what each PTX parameter
holds (a buffer base, an affine of the sizes, a constant, a bounded integer,
or zero), and for each buffer base its direction, extent, alignment, and
index range, stated in size symbols (design section 3.2), with the block
and the dynamic shared memory of every launch. The binding is
what the gate verifies against and what the catalog's launch check checks
each launch against. The derivation is untrusted: a wrong binding makes the
gate refuse the kernel or the catalog refuse the launch.

The Python records mirror the Rocq records of `formal/GateFront.v` field by
field: `Affine` is `affine`, `IndexRange` is `ixrange`, `Span` is `bspan`,
`Parameter` is `holding`, and `Binding` is `binding`. `opgate.gate` prints
them as those terms.

Imports `opgate.signature` only.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, assert_never

from opgate import signature as sig
from opgate.signature import Affine, Direction

__all__ = [
    "BASE_ALIGN",
    "GRID_LIMIT",
    "SHARED_MAX",
    "validate_grid",
    "shared_bounds",
    "NO_SHARED",
    "validate_shared",
    "Affine",
    "Base",
    "Binding",
    "Bounded",
    "Constant",
    "Convention",
    "Direction",
    "ElementType",
    "IndexRange",
    "Parameter",
    "Size",
    "Span",
    "Zero",
    "derive_binding",
]


@dataclass(frozen=True)
class ElementType:
    """The width in bytes of a buffer's elements, and whether an index is signed."""

    width: Literal[1, 2, 4, 8]
    signed: bool

    def __post_init__(self) -> None:
        if self.width not in (1, 2, 4, 8):
            raise ValueError(f"an element of {self.width} bytes, not 1, 2, 4, or 8")


BASE_ALIGN = 8
"""log2 of the alignment of every buffer base: a launch passes the start of a
catalog region, and the catalog places every region at a multiple of
`2**BASE_ALIGN` bytes (`opgate.catalog.ALIGNMENT`), the largest alignment of a
base that the link allows (`align_ok` in `formal/GateFront.v`)."""


@dataclass(frozen=True)
class IndexRange:
    """The elements of an index input or output lie in `[lo, hi]`."""

    element: ElementType
    lo: Affine
    hi: Affine


@dataclass(frozen=True)
class Span:
    """A buffer parameter: the base of `extent` bytes, a multiple of `2**align`.
    A flag span (`flag`) holds words of 4 bytes that a launch may only set to 1,
    as the status buffer of the guard pass's checks; the kernel only writes
    it, and it has no index range."""

    direction: Direction
    extent: Affine
    align: int
    index: IndexRange | None
    flag: bool

    def __post_init__(self) -> None:
        if type(self.flag) is not bool:
            raise TypeError("a span's flag must be a boolean")
        if self.flag and (self.direction is not Direction.WRITE or self.index is not None):
            raise ValueError("a flag span is written only and has no index range")


@dataclass(frozen=True)
class Base:
    span: Span


@dataclass(frozen=True)
class Size:
    """The parameter holds the value of an affine of the sizes."""

    value: Affine


@dataclass(frozen=True)
class Constant:
    value: int


@dataclass(frozen=True)
class Bounded:
    """The parameter holds an integer in `[lower, upper]`."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError(f"an empty interval [{self.lower}, {self.upper}]")


@dataclass(frozen=True)
class Zero:
    pass


Parameter = Base | Size | Constant | Bounded | Zero


NO_SHARED = Affine(0)

SHARED_MAX = 49152
"""Dynamic shared bytes available without a device-specific opt-in."""


def shared_bounds(sizes: Sequence[sig.SizeSymbol], shared: Affine) -> tuple[int, int]:
    """The fewest and the most bytes of dynamic shared memory over the sizes' bounds."""
    if not isinstance(shared, Affine):
        raise TypeError("dynamic shared memory must be an Affine byte extent")
    lo = hi = shared.constant
    for symbol, coefficient in shared.terms:
        if not 0 <= symbol < len(sizes):
            raise ValueError(f"undeclared shared-memory size symbol {symbol}")
        size = sizes[symbol]
        a, b = coefficient * size.minimum, coefficient * size.maximum
        lo += min(a, b)
        hi += max(a, b)
    return lo, hi


GRID_LIMIT = (2**31 - 1, 65535, 65535)
"""The largest grid, in blocks, of every GPU: a binding's grid maxima lie within it."""


def validate_grid(grid: tuple[int, int, int]) -> None:
    """A binding's grid maxima: at least one block and at most the GPU's limit in each dimension."""
    if not all(1 <= g <= limit for g, limit in zip(grid, GRID_LIMIT, strict=True)):
        raise ValueError(f"grid maxima {grid} lie outside [1, {GRID_LIMIT}]")


def validate_shared(sizes: Sequence[sig.SizeSymbol], shared: Affine) -> None:
    """Every size tuple in the binding must fit the shared-memory limit."""
    lo, hi = shared_bounds(sizes, shared)
    if not 0 <= lo <= hi <= SHARED_MAX:
        raise ValueError("dynamic shared memory must lie in [0, 48 KiB] over its size bounds")


@dataclass(frozen=True)
class Binding:
    """What every launch of a kernel establishes, in size symbols."""

    kernel: str
    sizes: tuple[sig.SizeSymbol, ...]
    parameters: tuple[tuple[str, Parameter], ...]
    block: tuple[int, int, int]
    grid: tuple[int, int, int]  # the largest grid of any launch, in blocks
    shared: Affine  # the bytes of dynamic shared memory of every launch
    derived_sizes: tuple[sig.DerivedSize, ...] = ()

    def __post_init__(self) -> None:
        sig.validate_derived_sizes(self.sizes, self.derived_sizes)
        names = [name for name, _ in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"a parameter is named twice in {names}")
        for _, parameter in self.parameters:
            for affine in _affines(parameter):
                for symbol, _ in affine.terms:
                    if not 0 <= symbol < len(self.sizes):
                        raise ValueError(
                            f"size symbol {symbol} of {len(self.sizes)} in {parameter}"
                        )
        if any(extent < 1 for extent in self.block):
            raise ValueError(f"an empty block {self.block}")
        validate_grid(self.grid)
        validate_shared(self.sizes, self.shared)

    @property
    def bases(self) -> tuple[tuple[str, Span], ...]:
        """The buffer parameters, in order: one per buffer argument of a launch."""
        return tuple((name, p.span) for name, p in self.parameters if isinstance(p, Base))


def _affines(parameter: Parameter) -> tuple[Affine, ...]:
    match parameter:
        case Base(span=span):
            affines = [span.extent]
            if span.index is not None:
                affines += [span.index.lo, span.index.hi]
            return tuple(affines)
        case Size(value=value):
            return (value,)
        case Constant() | Bounded() | Zero():
            return ()
        case _ as never:
            assert_never(never)


class Convention(Enum):
    """How a toolchain lays out one signature argument as PTX parameters."""

    NUMBA = (
        "numba"  # a one-dimensional array: meminfo, parent, nitems, itemsize, data, shape, stride
    )
    C = "c"  # a pointer, or the scalar itself
    # as C, then the global and the profile scratch pointers that Triton 3.8
    # appends; the launch passes zero for both, as a kernel without scratch
    # memory needs
    TRITON = "triton"


def _trailing(convention: Convention) -> list[Parameter]:
    """The parameters that a convention appends after the arguments."""
    match convention:
        case Convention.NUMBA | Convention.C:
            return []
        case Convention.TRITON:
            return [Zero(), Zero()]
        case _ as never:
            assert_never(never)


def _span(argument: sig.Buffer, element: ElementType, targets: Mapping[str, Affine]) -> Span:
    extent = argument.rows.times(element.width)
    match argument:
        case sig.Span(direction=direction):
            return Span(direction, extent, BASE_ALIGN, None, False)
        case sig.IndexIn(target=target):
            hi = targets[target].plus(-1)
            return Span(
                Direction.READ, extent, BASE_ALIGN, IndexRange(element, Affine(0), hi), False
            )
        case sig.Offsets(target=target):
            return Span(
                Direction.READ,
                extent,
                BASE_ALIGN,
                IndexRange(element, Affine(0), targets[target]),
                False,
            )
        case sig.IndexOut(target=target):
            hi = targets[target].plus(-1)
            return Span(
                Direction.WRITE, extent, BASE_ALIGN, IndexRange(element, Affine(0), hi), False
            )
        case _ as never:
            assert_never(never)


def _parameters(
    argument: sig.Argument,
    element_types: Mapping[str, ElementType],
    targets: Mapping[str, Affine],
    convention: Convention,
) -> list[Parameter]:
    if isinstance(argument, sig.Scalar):
        match argument.bound:
            case sig.SizeOf(value=value):
                return [Size(value)]
            case sig.Between(lower=lower, upper=upper):
                return [Bounded(lower, upper)]
            case _ as never:
                assert_never(never)
    element = element_types[argument.name]
    base = Base(_span(argument, element, targets))
    match convention:
        case Convention.NUMBA:
            rows = Size(argument.rows)
            width = Constant(element.width)
            return [Zero(), Zero(), rows, width, base, rows, width]
        case Convention.C | Convention.TRITON:
            return [base]
        case _ as never:
            assert_never(never)


def derive_binding(
    signature: sig.Signature,
    element_types: Mapping[str, ElementType],
    convention: Convention,
    kernel: str,
    parameter_names: Sequence[str],
    block: tuple[int, int, int],
    grid: tuple[int, int, int],
    shared: Affine,
) -> Binding:
    """The binding of `kernel`, whose PTX parameters are `parameter_names` in order,
    launched in blocks of `block` threads, in grids of at most `grid` blocks, with
    `shared` bytes of dynamic shared memory.

    `element_types` gives the element type of each buffer argument, by argument
    name, from the plan's columns.
    """
    targets = {a.name: a.rows for a in signature.arguments if not isinstance(a, sig.Scalar)}
    parameters: list[Parameter] = []
    for argument in signature.arguments:
        parameters += _parameters(argument, element_types, targets, convention)
    parameters += _trailing(convention)
    if len(parameters) != len(parameter_names):
        raise ValueError(
            f"{kernel}: {len(parameter_names)} PTX parameters, the signature lays out {len(parameters)}"
        )
    return Binding(
        kernel,
        signature.sizes,
        tuple(zip(parameter_names, parameters, strict=True)),
        block,
        grid,
        shared,
        signature.derived_sizes,
    )
