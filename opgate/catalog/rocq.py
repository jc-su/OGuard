"""The catalog's Rocq terms, and Rocq's decisions on them (`formal/CatalogStep.v`).

`catalog_term` and `step_term` print a catalog and a step as Rocq terms of
`formal/CatalogStep.v`, with each verified operator's binding named
`OPS_<id>`, and `show` prints a decision as Rocq's `show_decision` does.
`decide` has Rocq evaluate `check_step` on (catalog, step) pairs. `replay`
has Rocq fold `check_step` over what a traced catalog recorded
(`Catalog.trace`), from the catalog at the trace's start: each step's
decision decides the catalog of the next, and each verified operator joins
the catalog where the trace recorded it. Both compare Rocq's printed
decisions with Python's, after `as_sets` orders the buffers, grants, and
freed regions, which are sets. This module is not on the path that decides
a step; the tests and the evaluation's audit use it.

Imports `opgate.catalog`, `opgate.gate`, and runs `coqc`.
"""

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from opgate.binding import Binding, ElementType
from opgate.catalog import (
    Action,
    Allocate,
    At,
    Buffer,
    BufferId,
    Catalog,
    Checked,
    Clear,
    Decided,
    Effect,
    Fill,
    Free,
    Grant,
    Kernel,
    Launch,
    Memset,
    OperatorId,
    Raw,
    Region,
    Reservation,
    Reserve,
    Status,
    Step,
    StepAccepted,
    StepRefused,
    Traced,
    Unreserve,
    Upload,
    Verified,
    View,
    Write,
    check_step,
)
from opgate.gate import binding_term, formal_dir

__all__ = ["Replayed", "as_sets", "catalog_term", "decide", "replay", "show", "step_term"]


def _z(value: int) -> str:
    return f"({value})%Z"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _status(status: Status) -> str:
    match status:
        case Raw():
            return "SRaw"
        case Checked(element=e, lo=lo, hi=hi):
            return f"(SChecked {_z(e.width)} {_bool(e.signed)} {_z(lo)} {_z(hi)})"
        case _ as never:
            raise TypeError(never)


def _list(items: list[str]) -> str:
    return "[" + "; ".join(items) + "]"


def _id(buffer: BufferId) -> str:
    return f"({_z(buffer.tenant)}, {_z(buffer.serial)})"


def _reservation(r: Reservation) -> str:
    return f"{{| rs_tenant := {_z(r.tenant)}; rs_offset := {_z(r.offset)}; rs_bytes := {_z(r.nbytes)} |}}"


def catalog_term(c: Catalog) -> str:
    buffers = [
        f"{{| bf_id := {_id(b.id)}; bf_offset := {_z(b.offset)}; bf_bytes := {_z(b.nbytes)}; "
        f"bf_owner := {_z(b.owner)}; bf_status := {_status(b.status)} |}}"
        for b in c.buffers
    ]
    grants = [f"({_id(g.buffer)}, {_z(g.tenant)})" for g in c.grants]
    freed = [
        f"{{| rg_offset := {_z(r.offset)}; rg_bytes := {_z(r.nbytes)}; rg_owner := {_z(r.owner)} |}}"
        for r in c.freed
    ]
    reserved = [_reservation(r) for r in c.reserved]
    operators = [f"({_z(o)}, OPS_{o})" for o, _ in c.operators]
    return (
        f"{{| c_pool := {_z(c.pool)}; c_buffers := {_list(buffers)}; c_grants := {_list(grants)}; "
        f"c_freed := {_list(freed)}; c_reserved := {_list(reserved)}; "
        f"c_operators := {_list(operators)} |}}"
    )


def _bytes(data: bytes) -> str:
    """The bytes as a list of Z, written as lists of at most 256 joined by `++`,
    since Rocq's parser nests a list literal one level for each element."""
    chunks = [data[i : i + 256] for i in range(0, len(data), 256)] or [b""]
    return "(" + " ++ ".join(_list([str(x) for x in chunk]) + "%Z" for chunk in chunks) + ")"


def _action(a: Action) -> str:
    match a:
        case Allocate(buffer=b, offset=o, nbytes=n):
            return f"(Allocate {_id(b)} {_z(o)} {_z(n)})"
        case Free(buffer=b):
            return f"(Free {_id(b)})"
        case Grant(buffer=b, tenant=t):
            return f"(Grant {_id(b)} {_z(t)})"
        case Write(buffer=b, data=data, element=None):
            return f"(Write {_id(b)} {_bytes(data)} None)"
        case Write(buffer=b, data=data, element=ElementType() as e):
            return f"(Write {_id(b)} {_bytes(data)} (Some ({_z(e.width)}, {_bool(e.signed)})))"
        case Fill(buffer=b, element=e, value=v):
            return f"(Fill {_id(b)} {_z(e.width)} {_bool(e.signed)} {_z(v)})"
        case Launch(operator=op, grid=(gx, gy, gz), buffers=bs, sizes=ss, scalars=xs, guarded=g):
            views = _list([_view(b) for b in bs])
            lists = " ".join(_list([_z(v) for v in values]) for values in (ss, xs))
            return f"(Launch {_z(op)} {_z(gx)} {_z(gy)} {_z(gz)} {views} {lists} {_bool(g)})"
        case Reserve(offset=o, nbytes=n):
            return f"(Reserve {_z(o)} {_z(n)})"
        case Unreserve():
            return "Unreserve"
        case _:
            raise TypeError(a)


def _view(argument: BufferId | View) -> str:
    match argument:
        case View(buffer=b, offset=o):
            return f"({_id(b)}, {_z(o)})"
        case BufferId():
            return f"({_id(argument)}, {_z(0)})"
        case _ as never:
            assert_never(never)


def step_term(s: Step) -> str:
    return (
        f"{{| s_tenant := {_z(s.tenant)}; s_actions := {_list([_action(a) for a in s.actions])} |}}"
    )


def _show_status(status: Status) -> str:
    match status:
        case Raw():
            return "raw"
        case Checked(element=e, lo=lo, hi=hi):
            return f"checked({e.width},{'s' if e.signed else 'u'},{lo},{hi})"
        case _ as never:
            raise TypeError(never)


def _show_id(buffer: BufferId) -> str:
    return f"({buffer.tenant},{buffer.serial})"


def _show_buffer(b: Buffer) -> str:
    return f"({_show_id(b.id)},{b.offset},{b.nbytes},{b.owner},{_show_status(b.status)})"


def _show_region(r: Region) -> str:
    return f"({r.offset},{r.nbytes},{r.owner})"


def _show_reservation(r: Reservation) -> str:
    return f"({r.tenant},{r.offset},{r.nbytes})"


def _show_effect(e: Effect) -> str:
    match e:
        case Clear(offset=o, nbytes=n):
            return f"clear({o},{n})"
        case Upload(offset=o, data=data):
            return f"upload({o},{len(data)})"
        case Memset(offset=o, nbytes=n, element=el, value=v):
            return f"memset({o},{n},{el.width},{'s' if el.signed else 'u'},{v})"
        case Kernel(operator=op, grid=(gx, gy, gz), shared=shared, arguments=args, checks=checks):
            shown = [f"at({a.offset})" if isinstance(a, At) else f"val({a.value})" for a in args]
            tests = [
                f"({k.offset},{k.nbytes},{k.element.width},{'s' if k.element.signed else 'u'},"
                f"{k.lo},{k.hi})"
                for k in checks
            ]
            return f"kernel({op},{gx},{gy},{gz},{shared},[{';'.join(shown)}],[{';'.join(tests)}])"
        case _ as never:
            raise TypeError(never)


def show(decision: StepAccepted | StepRefused) -> str:
    match decision:
        case StepRefused(reason=reason):
            return f"refused {reason}"
        case StepAccepted(next=c, effects=effects):
            return (
                f"accepted pool={c.pool} buffers=[{';'.join(_show_buffer(b) for b in c.buffers)}]"
                f" grants=[{';'.join(f'({_show_id(g.buffer)},{g.tenant})' for g in c.grants)}]"
                f" freed=[{';'.join(_show_region(r) for r in c.freed)}]"
                f" reserved=[{';'.join(_show_reservation(r) for r in c.reserved)}]"
                f" effects=[{';'.join(_show_effect(e) for e in effects)}]"
            )
        case _ as never:
            raise TypeError(never)


def as_sets(shown: str) -> str:
    """A printed decision with its buffers, grants, freed regions, and reservations
    in increasing order: they are sets, which no check reads in order; the Python
    catalog keeps them by id, offset, and tenant, where Rocq keeps lists in the
    order of their changes."""

    def ordered(match: re.Match[str]) -> str:
        items = [item for item in match.group(2).split(";") if item]
        numbers = [tuple(int(v) for v in re.findall(r"-?\d+", item)) for item in items]
        return f"{match.group(1)}[{';'.join(item for _, item in sorted(zip(numbers, items, strict=True)))}]"

    return re.sub(r"( (?:buffers|grants|freed|reserved)=)\[([^]]*)\]", ordered, shown)


def _coqc(name: str, source: str, library: Path, keep: Path | None = None) -> list[str]:
    """The strings that `coqc` prints for the one `Eval` of `source` against the
    compiled `library`; `keep`, if given, receives the source."""
    if keep is not None:
        keep.write_text(source)
    coqc = shutil.which("coqc")
    if coqc is None:
        raise RuntimeError("coqc is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.v"
        path.write_text(source)
        result = subprocess.run(
            [coqc, "-q", "-Q", str(library), "OpGate", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    if result.returncode != 0:
        raise RuntimeError(f"coqc failed: {result.stderr[-2000:]}")
    printed = result.stdout[result.stdout.index("=") :]
    return [s.replace('""', '"') for s in re.findall(r'"((?:[^"]|"")*)"', printed)]


_PRELUDE = """From Coq Require Import ZArith List Strings.String.
Import ListNotations.
Require Import OpGate.GateDomain OpGate.GateFront OpGate.GateParse OpGate.CatalogStep.
Local Open Scope string_scope.
"""


def _operators(bindings: dict[OperatorId, Binding]) -> str:
    return "\n".join(f"Definition OPS_{o} := {binding_term(b)}." for o, b in bindings.items())


def _add(bindings: dict[OperatorId, Binding], operator: OperatorId, binding: Binding) -> None:
    if operator in bindings and bindings[operator] != binding:
        raise ValueError(f"operator {operator} has two bindings")
    bindings[operator] = binding


def decide(cases: Sequence[tuple[Catalog, Step]], library: Path | None = None) -> list[str]:
    """Rocq's printed decision of `check_step` on each (catalog, step) pair, with
    the compiled `library`, by default `formal_dir()`."""
    bindings: dict[OperatorId, Binding] = {}
    for catalog, _ in cases:
        for operator, binding in catalog.operators:
            _add(bindings, operator, binding)
    terms = ";\n  ".join(f"({catalog_term(c)}, {step_term(s)})" for c, s in cases)
    source = f"""{_PRELUDE}{_operators(bindings)}
Definition cases : list (catalog * step) := [
  {terms}].
Eval vm_compute in map (fun x => show_decision (check_step (fst x) (snd x))) cases.
"""
    return _coqc("Conformance", source, formal_dir() if library is None else library)


@dataclass(frozen=True)
class Replayed:
    """A recorded step: Rocq's printed decision, the printed decision of Python's
    `check_step` on the same catalog, and whether that decision is the one the
    run applied (its refusal, or its effects)."""

    step: Step
    rocq: str
    python: str
    as_run: bool


def replay(
    start: Catalog,
    trace: Sequence[Traced],
    *,
    library: Path | None = None,
    keep: Path | None = None,
) -> list[Replayed]:
    """Each recorded step of `trace`, decided by Rocq and by Python from `start`,
    the catalog when the trace began, with the compiled `library`, by default
    `formal_dir()`; `keep`, if given, receives the Rocq source."""
    bindings: dict[OperatorId, Binding] = {}
    for operator, binding in start.operators:
        _add(bindings, operator, binding)
    events: list[str] = []
    python: list[tuple[Step, str, bool]] = []
    catalog = start.copy()
    for traced in trace:
        match traced:
            case Verified(operator=operator, binding=binding):
                _add(bindings, operator, binding)
                catalog.add_operator(operator, binding)
                events.append(f"EOperator {_z(operator)} OPS_{operator}")
            case Decided(step=step, decision=recorded):
                decision = check_step(catalog, step)
                match decision, recorded:
                    case StepRefused(reason=reason), StepRefused(reason=as_run):
                        same = reason == as_run
                    case StepAccepted(effects=effects), tuple() as as_run:
                        same = effects == as_run
                    case _:
                        same = False
                if isinstance(decision, StepAccepted):
                    catalog = decision.next
                python.append((step, show(decision), same))
                events.append(f"EStep {step_term(step)}")
            case _ as never:
                raise TypeError(never)
    if not events:
        return []
    source = f"""{_PRELUDE}{_operators(bindings)}
Inductive event := EStep (s : step) | EOperator (op : Z) (b : binding).
Definition add_operator (c : catalog) (op : Z) (b : binding) : catalog :=
  {{| c_pool := c_pool c; c_buffers := c_buffers c; c_grants := c_grants c;
     c_freed := c_freed c; c_reserved := c_reserved c; c_operators := c_operators c ++ [(op, b)] |}}.
Fixpoint replay (c : catalog) (es : list event) : list string :=
  match es with
  | [] => []
  | EStep s :: rest =>
      let d := check_step c s in
      show_decision d :: replay (match d with Accepted c' _ => c' | Refused _ => c end) rest
  | EOperator op b :: rest => replay (add_operator c op b) rest
  end.
Definition start := {catalog_term(start)}.
Definition events : list event := [
  {(";" + chr(10) + "  ").join(events)}].
Eval vm_compute in replay start events.
"""
    rocq = _coqc("Replay", source, formal_dir() if library is None else library, keep)
    if len(rocq) != len(python):
        raise RuntimeError(f"Rocq printed {len(rocq)} decisions for {len(python)} steps")
    return [Replayed(step, r, p, same) for (step, p, same), r in zip(python, rocq, strict=True)]
