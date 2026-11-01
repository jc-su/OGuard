"""The service's registration records and counters.

A registration record keeps the submitted PTX text and the server-derived
binding, the guard pass's result when the gate refused them for a reason
that the pass addresses (its text and binding, which the gate checked and
the executor loaded with the zeroing prologue, the accesses that it checks
when they run, and whether
it rewrote the kernel's barrier structure), the text that the executor
loaded, which begins with the zeroing prologue of its shared and local
memory (`opgate.gate.zeroed`), the decision, and the number of module loads
and launches, so an administrator can compare the actual bytes. Each tenant's
registrations are limited (`Limits.tenant_registrations`), which bounds the
records' memory.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, assert_never

from opgate.binding import (
    Base,
    Binding,
    Bounded,
    Constant,
    Direction,
    ElementType,
    IndexRange,
    Size,
    Span,
    Zero,
)
from opgate.frontends.guard import Guarded
from opgate.hosts.service.protocol import Fields, RequestError
from opgate.signature import Affine, DerivedSize, SizeSymbol

__all__ = [
    "Decision",
    "Registration",
    "Telemetry",
    "binding_record",
    "parse_binding",
    "parse_element",
]

Decision = Literal["accepted", "refused"]


def _affine(value: Affine) -> dict[str, object]:
    return {"constant": value.constant, "terms": [list(term) for term in value.terms]}


def binding_record(binding: Binding) -> dict[str, object]:
    """A binding as JSON, field by field."""
    parameters: list[dict[str, object]] = []
    for name, parameter in binding.parameters:
        match parameter:
            case Base(span=span):
                index = (
                    None
                    if span.index is None
                    else {
                        "element": {
                            "width": span.index.element.width,
                            "signed": span.index.element.signed,
                        },
                        "lo": _affine(span.index.lo),
                        "hi": _affine(span.index.hi),
                    }
                )
                parameters.append(
                    {
                        "name": name,
                        "kind": "base",
                        "direction": span.direction.value,
                        "extent": _affine(span.extent),
                        "align": span.align,
                        "index": index,
                        "flag": span.flag,
                    }
                )
            case Size(value=value):
                parameters.append({"name": name, "kind": "size", "value": _affine(value)})
            case Constant(value=value):
                parameters.append({"name": name, "kind": "constant", "value": value})
            case Bounded(lower=lower, upper=upper):
                parameters.append({"name": name, "kind": "bounded", "lower": lower, "upper": upper})
            case Zero():
                parameters.append({"name": name, "kind": "zero"})
            case _ as never:
                assert_never(never)
    return {
        "kernel": binding.kernel,
        "sizes": [
            {"name": s.name, "minimum": s.minimum, "maximum": s.maximum} for s in binding.sizes
        ],
        "derived_sizes": [
            {"target": r.target, "numerator": _affine(r.numerator), "divisor": r.divisor}
            for r in binding.derived_sizes
        ],
        "parameters": parameters,
        "block": list(binding.block),
        "grid": list(binding.grid),
        "shared": _affine(binding.shared),
    }


def _parse_affine(fields: Fields) -> Affine:
    fields.exact("constant", "terms")
    terms: list[tuple[int, int]] = []
    for index, term in enumerate(fields.values("terms")):
        match term:
            case [int() as symbol, int() as coefficient] if (
                type(symbol) is int and type(coefficient) is int and symbol >= 0
            ):
                terms.append((symbol, coefficient))
            case _:
                raise RequestError(
                    f"{fields.where}.terms[{index}] must be [symbol >= 0, coefficient]"
                )
    return Affine(fields.integer("constant"), tuple(terms))


def parse_element(fields: Fields) -> ElementType:
    """An element type, `{"width": 1 | 2 | 4 | 8, "signed": bool}`."""
    fields.exact("width", "signed")
    match fields.integer("width"):
        case 1 | 2 | 4 | 8 as width:
            return ElementType(width, fields.boolean("signed"))
        case other:
            raise RequestError(f"{fields.where}.width is {other}, not 1, 2, 4, or 8")


def _parse_parameter(fields: Fields) -> tuple[str, Base | Size | Constant | Bounded | Zero]:
    name, kind = fields.text("name"), fields.text("kind")
    match kind:
        case "base":
            fields.exact("name", "kind", "direction", "extent", "align", "index", "flag")
            try:
                direction = Direction(fields.text("direction"))
            except ValueError as error:
                raise RequestError(f"{fields.where}.direction names no direction") from error
            index: IndexRange | None = None
            if fields.raw("index") is not None:
                bounds = fields.fields("index")
                bounds.exact("element", "lo", "hi")
                index = IndexRange(
                    parse_element(bounds.fields("element")),
                    _parse_affine(bounds.fields("lo")),
                    _parse_affine(bounds.fields("hi")),
                )
            span = Span(
                direction,
                _parse_affine(fields.fields("extent")),
                fields.integer("align", minimum=0),
                index,
                fields.boolean("flag"),
            )
            return name, Base(span)
        case "size":
            fields.exact("name", "kind", "value")
            return name, Size(_parse_affine(fields.fields("value")))
        case "constant":
            fields.exact("name", "kind", "value")
            return name, Constant(fields.integer("value"))
        case "bounded":
            fields.exact("name", "kind", "lower", "upper")
            return name, Bounded(fields.integer("lower"), fields.integer("upper"))
        case "zero":
            fields.exact("name", "kind")
            return name, Zero()
        case _:
            raise RequestError(f"{fields.where}.kind names no parameter kind")


def _triple(record: Fields, name: str) -> tuple[int, int, int]:
    """Three positive integers, a block or a grid."""
    values = record.values(name)
    if len(values) != 3 or any(type(v) is not int or v < 1 for v in values):
        raise RequestError(f"binding.{name} must hold three positive integers")
    x, y, z = (int(v) for v in values if type(v) is int)
    return x, y, z


def parse_binding(value: object) -> Binding:
    """The binding of a record that `binding_record` wrote, to replay a registration;
    a `RequestError` (a `ValueError`) for a record of another shape, and the
    binding's own `ValueError` for a contract it does not allow."""
    record = Fields(value, "binding")
    record.exact("kernel", "sizes", "parameters", "block", "grid", "shared", "derived_sizes")
    sizes: list[SizeSymbol] = []
    for index, item in enumerate(record.values("sizes")):
        size = Fields(item, f"binding.sizes[{index}]")
        size.exact("name", "minimum", "maximum")
        sizes.append(
            SizeSymbol(size.text("name"), size.integer("minimum"), size.integer("maximum"))
        )
    derived = []
    for index, item in enumerate(record.values("derived_sizes")):
        relation = Fields(item, f"binding.derived_sizes[{index}]")
        relation.exact("target", "numerator", "divisor")
        derived.append(
            DerivedSize(
                relation.integer("target", minimum=0),
                _parse_affine(relation.fields("numerator")),
                relation.integer("divisor", minimum=1),
            )
        )
    parameters = tuple(
        _parse_parameter(Fields(item, f"binding.parameters[{index}]"))
        for index, item in enumerate(record.values("parameters"))
    )
    try:
        return Binding(
            record.text("kernel"),
            tuple(sizes),
            parameters,
            _triple(record, "block"),
            _triple(record, "grid"),
            _parse_affine(record.fields("shared")),
            tuple(derived),
        )
    except TypeError as error:
        raise RequestError(f"binding: {error}") from error


@dataclass
class Registration:
    """One registration: the actual PTX and binding as submitted, the guard pass's
    result when it ran, the text that the executor loaded, with its zeroing
    prologue, when the gate accepted it, and what became of them."""

    index: int
    principal: str
    plan_kind: str  # a plan schema's kind, or the query node whose engine registered it
    ptx: str
    binding: Binding
    guarded: Guarded | None
    loaded: str | None
    decision: Decision
    reason: str | None
    module_load_count: int
    launch_count: int

    def record(self) -> dict[str, object]:
        return {
            "registration": self.index,
            "principal": self.principal,
            "plan_kind": self.plan_kind,
            "ptx": self.ptx,
            "ptx_bytes": len(self.ptx.encode("ascii")),
            "binding": binding_record(self.binding),
            "guarded": None if self.guarded is None else _guarded_record(self.guarded),
            "loaded": self.loaded,
            "decision": self.decision,
            "reason": self.reason,
            "module_load_count": self.module_load_count,
            "launch_count": self.launch_count,
        }


def _guarded_record(guarded: Guarded) -> dict[str, object]:
    """The guard pass's text and binding, the accesses it checks, the back edges of
    the loops it limits, and whether it rewrote the kernel's barrier structure."""
    return {
        "ptx": guarded.text,
        "ptx_bytes": len(guarded.text.encode("ascii")),
        "binding": binding_record(guarded.binding),
        "checks": list(guarded.guarded),
        "limits": list(guarded.limited),
        "restructured": guarded.restructured,
    }


@dataclass
class Telemetry:
    """Counts of what the service did."""

    requests: int = 0
    refused_requests: int = 0
    input_errors: int = 0
    internal_errors: int = 0
    fatal_errors: int = 0
    register_requests: int = 0
    accepted_registrations: int = 0
    module_loads: int = 0
    launch_requests: int = 0
    launch_submissions: int = 0
    failed_checks: int = 0
    closed_sessions: int = 0
    node_requests: Counter[str] = field(default_factory=Counter)

    def record(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "refused_requests": self.refused_requests,
            "input_errors": self.input_errors,
            "internal_errors": self.internal_errors,
            "fatal_errors": self.fatal_errors,
            "register_requests": self.register_requests,
            "accepted_registrations": self.accepted_registrations,
            "module_loads": self.module_loads,
            "launch_requests": self.launch_requests,
            "launch_submissions": self.launch_submissions,
            "failed_checks": self.failed_checks,
            "closed_sessions": self.closed_sessions,
            "node_requests": dict(self.node_requests),
        }
