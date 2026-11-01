"""The author's claim about an operator's arguments.

A signature names the size symbols of a launch and says what each kernel
argument is in terms of the plan: a span over a named column or intermediate
whose row count is affine in the symbols, an index input that gathers from a
target, an offsets buffer into a target, an index output a later launch will
gather by, or a scalar that holds a size or lies in an interval. The catalog
checks each claim against the buffer it names at every launch, and the gate
verifies the kernel against the binding derived from it (design section 3.2).
Nothing in this module is trusted on its own.
"""

from dataclasses import dataclass
from enum import Enum
from typing import assert_never


@dataclass(frozen=True)
class Affine:
    """A constant plus a sum of coefficient times size symbol.

    A symbol is its index into `Signature.sizes`; each symbol occurs at most
    once, with a nonzero coefficient.
    """

    constant: int
    terms: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        symbols = [symbol for symbol, _ in self.terms]
        if len(set(symbols)) != len(symbols):
            raise ValueError(f"a symbol occurs twice in {self}")
        if any(coefficient == 0 for _, coefficient in self.terms):
            raise ValueError(f"a zero coefficient in {self}")

    def plus(self, constant: int) -> "Affine":
        return Affine(self.constant + constant, self.terms)

    def times(self, factor: int) -> "Affine":
        if factor == 0:
            return Affine(0)
        return Affine(
            self.constant * factor,
            tuple((symbol, coefficient * factor) for symbol, coefficient in self.terms),
        )

    @staticmethod
    def symbol(index: int) -> "Affine":
        return Affine(0, ((index, 1),))


SIZE_MAX = 2**31 - 1
"""The largest value of a size symbol: the row limit of a 32-bit signed count."""


@dataclass(frozen=True)
class SizeSymbol:
    """A size symbol of a launch: its name and the inclusive bounds it takes.

    A symbol takes a value in `[minimum, maximum]` at every launch; the
    launch check refuses a launch outside these bounds. A hash table's capacity
    has minimum 1, so that `capacity - 1` is a mask.
    """

    name: str
    minimum: int = 0
    maximum: int = SIZE_MAX

    def __post_init__(self) -> None:
        if type(self.minimum) is not int or type(self.maximum) is not int:
            raise TypeError("size bounds must be integers")
        if not 0 <= self.minimum <= self.maximum <= SIZE_MAX:
            raise ValueError(f"the bounds of {self.name} are outside [0, {SIZE_MAX}]")


@dataclass(frozen=True)
class DerivedSize:
    """The target size equals floor(numerator / divisor) at every launch.

    The divisor is a positive constant. A ceiling by d is expressed as
    floor((n + d - 1) / d). Numerator symbols precede the target symbol,
    so service schemas can derive sizes in declaration order.
    """

    target: int
    numerator: Affine
    divisor: int

    def __post_init__(self) -> None:
        if type(self.target) is not int or self.target < 0:
            raise ValueError("a derived size target must be a nonnegative symbol index")
        if type(self.divisor) is not int or self.divisor <= 0:
            raise ValueError("a derived size divisor must be a positive integer")
        a = self.numerator
        if (
            not isinstance(a, Affine)
            or type(a.constant) is not int
            or type(a.terms) is not tuple
            or any(
                type(t) is not tuple or len(t) != 2 or any(type(v) is not int for v in t)
                for t in a.terms
            )
        ):
            raise TypeError("a derived size numerator must be an immutable integer affine")
        if any(not 0 <= index < self.target for index, _ in a.terms):
            raise ValueError("derived size numerators must refer only to earlier size symbols")


def validate_derived_sizes(sizes: tuple[SizeSymbol, ...], derived: tuple[DerivedSize, ...]) -> None:
    if type(derived) is not tuple or any(not isinstance(r, DerivedSize) for r in derived):
        raise TypeError("derived sizes must be an immutable tuple of DerivedSize relations")
    targets = [r.target for r in derived]
    if targets != sorted(set(targets)) or any(target >= len(sizes) for target in targets):
        raise ValueError("derived size targets must be distinct declared symbols in order")


class Direction(Enum):
    """How a span is accessed. An atomic needs `READ_WRITE`."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class Span:
    """A flat buffer of `rows` elements with a direction.

    `ref` names the column or intermediate in the plan whose buffer this
    argument is.
    """

    name: str
    ref: str
    direction: Direction
    rows: Affine


@dataclass(frozen=True)
class IndexIn:
    """A buffer of indices, checked by the catalog, used to gather from `target`.

    Its elements lie in `[0, rows(target) - 1]`.
    """

    name: str
    ref: str
    rows: Affine
    target: str


@dataclass(frozen=True)
class Offsets:
    """A buffer of offsets into `target`, as string and list columns hold.

    Its elements lie in `[0, rows(target)]`: an offset may reach the end.
    """

    name: str
    ref: str
    rows: Affine
    target: str


@dataclass(frozen=True)
class IndexOut:
    """A buffer of indices this launch produces; a consumer gathers from `target` by them.

    The gate proves every stored value in `[0, rows(target) - 1]`.
    """

    name: str
    ref: str
    rows: Affine
    target: str


@dataclass(frozen=True)
class SizeOf:
    """A scalar bound: the parameter holds the value of an affine of the sizes."""

    value: Affine


@dataclass(frozen=True)
class Between:
    """A scalar bound: a closed integer interval on the parameter's bits."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError(f"an empty interval [{self.lower}, {self.upper}]")


ScalarBound = SizeOf | Between


@dataclass(frozen=True)
class Scalar:
    """An integer argument with a declared bound."""

    name: str
    bound: ScalarBound


Argument = Span | IndexIn | Offsets | IndexOut | Scalar
Buffer = Span | IndexIn | Offsets | IndexOut


@dataclass(frozen=True)
class Signature:
    """The size symbols of a launch and the arguments of one kernel, in parameter order."""

    sizes: tuple[SizeSymbol, ...]
    arguments: tuple[Argument, ...]
    derived_sizes: tuple[DerivedSize, ...] = ()

    def __post_init__(self) -> None:
        validate_derived_sizes(self.sizes, self.derived_sizes)
        names = [size.name for size in self.sizes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate size symbols: {names}")
        names = [argument.name for argument in self.arguments]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate argument names: {duplicates}")
        buffers = {argument.name for argument in self.arguments if not isinstance(argument, Scalar)}
        for argument in self.arguments:
            for affine in self._affines(argument):
                for symbol, _ in affine.terms:
                    if not 0 <= symbol < len(self.sizes):
                        raise ValueError(
                            f"{argument.name} names size symbol {symbol} of {len(self.sizes)}"
                        )
            if (
                isinstance(argument, IndexIn | Offsets | IndexOut)
                and argument.target not in buffers
            ):
                raise ValueError(
                    f"{argument.name} targets {argument.target}, which is not a buffer argument"
                )

    @staticmethod
    def _affines(argument: Argument) -> tuple[Affine, ...]:
        match argument:
            case Span(rows=rows) | IndexIn(rows=rows) | Offsets(rows=rows) | IndexOut(rows=rows):
                return (rows,)
            case Scalar(bound=bound):
                match bound:
                    case SizeOf(value=value):
                        return (value,)
                    case Between():
                        return ()
                    case _ as never:
                        assert_never(never)
            case _ as never:
                assert_never(never)

    def buffer(self, name: str) -> Buffer:
        """The buffer argument called `name`."""
        for argument in self.arguments:
            if argument.name == name and not isinstance(argument, Scalar):
                return argument
        raise KeyError(f"no buffer argument {name}")
