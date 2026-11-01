"""Declarative, deployment-owned operator contracts for the service.

One schema accepts any single-entry PTX module that the gate proves under
its signature and calling convention. Clients supply buffer handles and
bounded launch values, never contracts, layouts, or device addresses.
"""

from dataclasses import dataclass, field

from opgate import signature as sig
from opgate.binding import (
    GRID_LIMIT,
    NO_SHARED,
    Binding,
    Convention,
    ElementType,
    validate_grid,
    validate_shared,
)
from opgate.frontends import Operator, entry_kernels
from opgate.hosts.service.plans import LaunchRequest
from opgate.hosts.service.protocol import Fields, RequestError

__all__ = ["OperatorSchema"]


def _dimension(value: object, limit: int) -> int:
    if type(value) is not int or not 1 <= value <= limit:
        raise RequestError(f"a grid dimension must be an integer within [1, {limit}]")
    return value


def _immutable_affine(value: sig.Affine) -> None:
    if (
        not isinstance(value, sig.Affine)
        or type(value.constant) is not int
        or type(value.terms) is not tuple
        or any(
            type(term) is not tuple or len(term) != 2 or any(type(v) is not int for v in term)
            for term in value.terms
        )
    ):
        raise TypeError("an affine must have an integer constant and immutable integer terms")


@dataclass(frozen=True)
class OperatorSchema:
    """A reusable service plan schema, configured by the deployment.

    ``element_types`` names every buffer argument exactly once. The launch
    request has named ``buffers``, ``sizes``, and ``scalars`` objects, a
    three-dimensional ``grid``, and an explicit ``guarded`` boolean. Scalars
    described by ``SizeOf`` are derived from sizes; dependent sizes are also
    derived and must not be supplied by clients. Only ``Between`` scalars
    appear in the request. Integer scalar values represent parameter bits.

    ``guarded=True`` requests the existing checked index scan for raw index
    inputs. With it false, those inputs must already have catalog provenance.
    Both paths check the launch's actual authority, allocation, and bounds.
    """

    kind: str
    signature: sig.Signature
    element_types: tuple[tuple[str, ElementType], ...]
    convention: Convention
    block: tuple[int, int, int]
    grid: tuple[int, int, int] = GRID_LIMIT
    shared: sig.Affine = NO_SHARED
    _buffers: tuple[str, ...] = field(init=False, repr=False)
    _scalars: tuple[tuple[str, sig.Between], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.kind) is not str or not self.kind:
            raise ValueError("an operator schema needs a nonempty kind")
        if not isinstance(self.convention, Convention):
            raise TypeError("the calling convention must be a Convention")
        if (
            not isinstance(self.signature, sig.Signature)
            or type(self.signature.sizes) is not tuple
            or type(self.signature.arguments) is not tuple
        ):
            raise TypeError("the signature must have immutable tuples of sizes and arguments")
        for argument in self.signature.arguments:
            if isinstance(argument, sig.Scalar):
                if isinstance(argument.bound, sig.SizeOf):
                    _immutable_affine(argument.bound.value)
                elif not isinstance(argument.bound, sig.Between):
                    raise TypeError("a scalar needs a SizeOf or Between bound")
            elif isinstance(argument, sig.Span | sig.IndexIn | sig.IndexOut | sig.Offsets):
                _immutable_affine(argument.rows)
                if isinstance(argument, sig.Span) and not isinstance(
                    argument.direction, sig.Direction
                ):
                    raise TypeError("a span needs a Direction")
            else:
                raise TypeError("unsupported signature argument")
        _immutable_affine(self.shared)
        for name, dimensions in (("block", self.block), ("grid", self.grid)):
            if (
                type(dimensions) is not tuple
                or len(dimensions) != 3
                or any(type(d) is not int or d <= 0 for d in dimensions)
            ):
                raise ValueError(f"{name} must be a tuple of three positive integers")
        validate_grid(self.grid)
        validate_shared(self.signature.sizes, self.shared)
        names = [s.name for s in self.signature.sizes]
        names += [a.name for a in self.signature.arguments]
        if any(type(name) is not str or not name for name in names):
            raise ValueError("signature names must be nonempty strings")
        buffers = tuple(a.name for a in self.signature.arguments if not isinstance(a, sig.Scalar))
        if type(self.element_types) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 for pair in self.element_types
        ):
            raise TypeError("element_types must be a tuple of (argument name, ElementType) pairs")
        elements = dict(self.element_types)
        if len(elements) != len(self.element_types) or set(elements) != set(buffers):
            raise ValueError("element_types must name every buffer argument exactly once")
        if any(
            not isinstance(element, ElementType)
            or type(element.width) is not int
            or type(element.signed) is not bool
            for element in elements.values()
        ):
            raise TypeError("buffer element types need an integer width and boolean signedness")
        scalars = tuple(
            (a.name, a.bound)
            for a in self.signature.arguments
            if isinstance(a, sig.Scalar) and isinstance(a.bound, sig.Between)
        )
        if any(type(b.lower) is not int or type(b.upper) is not int for _, b in scalars):
            raise TypeError("scalar bounds must be integers")
        if any(not 0 <= b.lower <= b.upper < 2**64 for _, b in scalars):
            raise ValueError("scalar bounds must fit unsigned 64-bit parameter bits")
        object.__setattr__(self, "_buffers", buffers)
        object.__setattr__(self, "_scalars", scalars)

    def plan(self, text: str, fields: Fields) -> "_OperatorPlan":
        fields.exact("kind")
        if fields.text("kind") != self.kind:
            raise RequestError(f"the plan kind must be {self.kind!r}")
        kernels = entry_kernels(text)
        if len(kernels) != 1:
            raise RequestError(f"the module must define one entry kernel, not {len(kernels)}")
        operator = Operator(
            text, kernels[0], self.signature, self.convention, self.block, self.grid, self.shared
        )
        return _OperatorPlan(self, operator.binding(dict(self.element_types)))


@dataclass(frozen=True)
class _OperatorPlan:
    schema: OperatorSchema
    binding: Binding
    _size_names: tuple[str, ...] = field(init=False, repr=False)
    _size_plan: tuple[tuple[sig.SizeSymbol, sig.DerivedSize | None], ...] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        derived = {relation.target: relation for relation in self.binding.derived_sizes}
        object.__setattr__(
            self,
            "_size_names",
            tuple(
                size.name for index, size in enumerate(self.binding.sizes) if index not in derived
            ),
        )
        object.__setattr__(
            self,
            "_size_plan",
            tuple((size, derived.get(index)) for index, size in enumerate(self.binding.sizes))
            if derived
            else (),
        )

    def _derived_values(self, sizes: Fields) -> tuple[int, ...]:
        values: list[int] = []
        for symbol, relation in self._size_plan:
            if relation is None:
                value = sizes.integer(symbol.name, minimum=symbol.minimum, maximum=symbol.maximum)
            else:
                value = (
                    relation.numerator.constant
                    + sum(
                        coefficient * values[source]
                        for source, coefficient in relation.numerator.terms
                    )
                ) // relation.divisor
                if not symbol.minimum <= value <= symbol.maximum:
                    raise RequestError(
                        f"derived size {symbol.name!r} lies outside its declared bounds"
                    )
            values.append(value)
        return tuple(values)

    @property
    def kind(self) -> str:
        return self.schema.kind

    def launch(self, request: Fields) -> LaunchRequest:
        request.exact("op", "operator", "grid", "buffers", "sizes", "scalars", "guarded")
        if request.text("op") != "run_operator":
            raise RequestError("an operator launch requires op='run_operator'")
        request.text("operator")
        dimensions = request.values("grid")
        if len(dimensions) != 3:
            raise RequestError("grid must have three integer dimensions")
        grid = (
            _dimension(dimensions[0], self.binding.grid[0]),
            _dimension(dimensions[1], self.binding.grid[1]),
            _dimension(dimensions[2], self.binding.grid[2]),
        )
        buffers, sizes, scalars = (
            request.fields("buffers"),
            request.fields("sizes"),
            request.fields("scalars"),
        )
        buffers.exact(*self.schema._buffers)
        sizes.exact(*self._size_names)
        size_values = (
            self._derived_values(sizes)
            if self._size_plan
            else tuple(
                sizes.integer(symbol.name, minimum=symbol.minimum, maximum=symbol.maximum)
                for symbol in self.binding.sizes
            )
        )
        scalars.exact(*(name for name, _ in self.schema._scalars))
        return LaunchRequest(
            grid,
            tuple(buffers.text(name) for name in self.schema._buffers),
            size_values,
            tuple(
                scalars.integer(name, minimum=bound.lower, maximum=bound.upper)
                for name, bound in self.schema._scalars
            ),
            request.boolean("guarded"),
        )
