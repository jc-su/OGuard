"""The gate: PTX text and bindings to a decision.

`verify` exports the module text and the binding of each entry kernel as Rocq
terms and has Rocq check an explicit invariant certificate. The result is a
`Decision`: accepted, or refused with the reason Rocq reports. The gate keeps
no trusted logic of its own beyond the export; the parser, the translation, and
the checker live in `formal/Gate*.v`, which must be built (`make -C formal
build`). A decision is recorded in the process, keyed by the compiled-library
identity, exact PTX text, and printed binding terms. Each check pins one complete
build snapshot. One process and one deadline cover proposal, diagnosis and proof.
A check that does not finish in its time is a refusal that is not recorded.
The gate refuses a kernel that does not begin with the zeroing prologue of its
shared and local memory, which `zeroed` (`opgate.gate.prologue`) writes.

Imports `opgate.binding` only.
"""

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from threading import Event
from typing import assert_never

from opgate.binding import (
    Affine,
    Base,
    Binding,
    Bounded,
    Constant,
    Direction,
    IndexRange,
    Parameter,
    Size,
    Span,
    Zero,
)
from opgate.gate.prologue import zeroed

__all__ = [
    "Accepted",
    "CHECK_QUERY",
    "Decision",
    "TIMEOUT",
    "TIMEOUT_REASON",
    "Refused",
    "VerificationBudget",
    "REGISTRATION_SECONDS",
    "timed_out",
    "binding_term",
    "export_terms",
    "export_query",
    "formal_dir",
    "verify",
    "zeroed",
]

_REFUSAL = re.compile(r'OPGATE_REFUSAL\s*\(?Some\s*"((?:[^"]|"")*)"')
_UNPROVED = re.compile(r'OPGATE_UNPROVED\s*\[((?:\s*"(?:[^"]|"")*"\s*;?)*)\s*\]')
_STRING = re.compile(r'"((?:[^"]|"")*)"')


@dataclass(frozen=True)
class Accepted:
    pass


@dataclass(frozen=True)
class Refused:
    """The reason of a refusal, and the first `LISTED` accesses that the check's
    walk of the certificate refuses (`text_diagnosis`, `GateParse.v`), each as
    its reason and instruction: what the guard pass checks in one round."""

    reason: str
    unproved: tuple[str, ...]


Decision = Accepted | Refused


def formal_dir() -> Path:
    """The explicitly selected library, or this installation's packaged proofs."""
    override = os.environ.get("OPGATE_FORMAL")
    directory = (
        Path(override) if override else Path(str(files("opgate_formal"))) / "build" / "current"
    )
    if not (directory / "GateParse.vo").is_file():
        command = f"make -C {directory} build" if override else "python -m opgate_formal"
        raise RuntimeError(f"the gate is not built: run {command}")
    return directory.resolve()


# ------------------------------------------------------------------
# Rocq terms.


def _z(value: int) -> str:
    if type(value) is not int:
        raise TypeError("Rocq integer fields must be integers")
    return f"({value})%Z"


def _packed(text: str) -> tuple[str, int]:
    """The text as an array of primitive integers, eight 7-bit characters in each, the
    first in the low bits, and its length. An empty text has one element past its end."""
    data = text.encode("ascii")
    words = [
        sum(b << (7 * k) for k, b in enumerate(data[i : i + 8])) for i in range(0, len(data), 8)
    ]
    return "[| " + "; ".join(map(str, words or [0])) + " | 0 |]", len(data)


def _affine(a: Affine) -> str:
    terms = "; ".join(f"({_z(symbol)}, {_z(coefficient)})" for symbol, coefficient in a.terms)
    return f"{{| af_const := {_z(a.constant)}; af_terms := [{terms}] |}}"


def _direction(d: Direction) -> str:
    match d:
        case Direction.READ:
            return "DRead"
        case Direction.WRITE:
            return "DWrite"
        case Direction.READ_WRITE:
            return "DReadWrite"
        case _ as never:
            assert_never(never)


def _index(ix: IndexRange | None) -> str:
    if ix is None:
        return "None"
    if type(ix.element.signed) is not bool:
        raise TypeError("index signedness must be a boolean")
    signed = "true" if ix.element.signed else "false"
    return (
        f"Some {{| ir_elem := {_z(ix.element.width)}; ir_signed := {signed}; "
        f"ir_lo := {_affine(ix.lo)}; ir_hi := {_affine(ix.hi)} |}}"
    )


def _span(span: Span) -> str:
    if type(span.flag) is not bool:
        raise TypeError("a span's flag must be a boolean")
    flag = "true" if span.flag else "false"
    return (
        f"{{| bs_dir := {_direction(span.direction)}; bs_ext := {_affine(span.extent)}; "
        f"bs_align := {_z(span.align)}; bs_index := {_index(span.index)}; bs_flag := {flag} |}}"
    )


def _parameter(p: Parameter) -> str:
    match p:
        case Base(span=span):
            return f"PBase {_span(span)}"
        case Size(value=value):
            return f"PSize {_affine(value)}"
        case Constant(value=value):
            return f"PConst {_z(value)}"
        case Bounded(lower=lower, upper=upper):
            return f"PBounded {_z(lower)} {_z(upper)}"
        case Zero():
            return "PZero"
        case _ as never:
            assert_never(never)


class _Strings:
    """Each distinct string is one definition that the terms name: Rocq reads a
    string literal much more slowly than a name."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def __call__(self, text: str) -> str:
        if not text.isascii():
            raise ValueError(f"a name outside ASCII: {text!r}")
        if text not in self.names:
            self.names[text] = f"str_{len(self.names)}"
        return self.names[text]

    def definitions(self) -> str:
        quoted = ((text.replace('"', '""'), name) for text, name in self.names.items())
        return "\n".join(f'Definition {name} := "{text}".' for text, name in quoted)


def _quoted(text: str) -> str:
    """A string literal of Rocq."""
    if not text.isascii():
        raise ValueError(f"a name outside ASCII: {text!r}")
    return '"' + text.replace('"', '""') + '"'


def binding_term(b: Binding) -> str:
    """The Rocq term of a binding, with its names as string literals."""
    return _binding(_quoted, b)


def _binding(strings: Callable[[str], str], b: Binding) -> str:
    bx, by, bz = b.block
    gx, gy, gz = b.grid
    sizes = "; ".join(f"({_z(size.minimum)}, {_z(size.maximum)})" for size in b.sizes)
    params = ";\n    ".join(f"({strings(name)}, {_parameter(p)})" for name, p in b.parameters)
    derived = "; ".join(
        f"{{| ds_target := {_z(r.target)}; ds_numerator := {_affine(r.numerator)}; ds_divisor := {_z(r.divisor)} |}}"
        for r in b.derived_sizes
    )
    return (
        f"{{| b_sizes := [{sizes}];\n   b_params := [\n    {params}];\n"
        f"   b_block := ({_z(bx)}, {_z(by)}, {_z(bz)});\n   b_grid := ({_z(gx)}, {_z(gy)}, {_z(gz)});\n"
        f"   b_shared := {_affine(b.shared)}; b_derived := [{derived}] |}}"
    )


def export_terms(text: str, bindings: Sequence[Binding]) -> str:
    """A Rocq file that defines the module's text, `text` of `size` characters, and its
    bindings, `mfs`, keyed by entry kernel."""
    if not text.isascii():
        raise ValueError("PTX text outside ASCII")
    array, size = _packed(text)
    strings = _Strings()
    mfs = ";\n  ".join(f"({strings(b.kernel)}, {_binding(strings, b)})" for b in bindings)
    # The text comes before the list notations, which would read its array literal as a list.
    return f"""From Coq Require Import ZArith NArith List Strings.String.
From Coq Require Import Numbers.Cyclic.Int63.Uint63 Array.PArray.
Local Open Scope uint63_scope.
Definition text : array int := {array}.
Definition size : int := {size}.
Close Scope uint63_scope.
Import ListNotations.
Require Import OpGate.GateIsa OpGate.GateForm OpGate.GateDomain OpGate.GateFront OpGate.GateParse.
Local Open Scope Z_scope.
Local Open Scope string_scope.

{strings.definitions()}

Definition mfs : list (string * binding) := [
  {mfs}].
"""


# ------------------------------------------------------------------
# Rocq evaluation.


@contextmanager
def _coqc(source: str, directory: Path) -> Iterator[subprocess.Popen[str]]:
    """A running `coqc` on `source`, stopped when the block ends."""
    coqc = shutil.which("coqc")
    if coqc is None:
        raise RuntimeError("coqc is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Query.v"
        path.write_text(source)
        process = subprocess.Popen(
            [coqc, "-q", "-Q", str(directory), "OpGate", str(path)],
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


# The front end's result of each entry kernel, without the participation
# check (front_program), and the proposal become concrete terms exactly once.
# The kernel checks that the result is what front gives, which runs the
# participation check once, and then checks the certificate on that result;
# the checked goal refers only to these terms, the original text and
# bindings, so kernel conversion cannot re-run the front end or proposal.
# abstract checks its opaque auxiliary proof before returning. Only failure
# runs the diagnosis: one evaluation gives the reason and the first refused
# accesses, which are never evidence of acceptance. A kernel that front
# refuses is diagnosed with the proposal from front, which gives it no
# certificate. coqc must close both lemmas with Qed.
CHECK_QUERY = """
Definition lowered := Eval vm_compute in text_front front_program text size mfs.
Lemma lowered_front : text_front front text size mfs = lowered.
Proof.
  first [ abstract (vm_cast_no_check (eq_refl lowered))
        | let diagnosis := eval vm_compute in
            (text_diagnosis text size mfs (propose_text_certificate text size mfs)) in
          lazymatch diagnosis with
          | (?reason, ?unproved) => idtac "OPGATE_REFUSAL" reason; idtac "OPGATE_UNPROVED" unproved
          end;
          fail 1 "front end refused" ].
Qed.
Definition certificate : text_certificate := Eval vm_compute in propose_front lowered.
Lemma accepted : check_text_certificate_ok text size mfs certificate = true.
Proof.
  rewrite check_text_front, lowered_front.
  first [ abstract (vm_cast_no_check (eq_refl true))
        | let diagnosis := eval vm_compute in (text_diagnosis text size mfs certificate) in
          lazymatch diagnosis with
          | (?reason, ?unproved) => idtac "OPGATE_REFUSAL" reason; idtac "OPGATE_UNPROVED" unproved
          end;
          fail 1 "certificate refused" ].
Qed.
"""


def export_query(text: str, bindings: Sequence[Binding]) -> str:
    """The exact complete Rocq query used to propose, diagnose and prove a module."""
    return export_terms(text, bindings) + CHECK_QUERY


TIMEOUT = 180.0  # one verify's time for proposal, proof, and diagnosis, outside a registration
TIMEOUT_REASON = "the check did not finish in"

Library = tuple[str, tuple[tuple[str, int, int, int, int], ...]]


def _library(directory: Path) -> Library:
    """Invalidate cached decisions when the selected compiled library changes."""
    stamps = []
    for path in sorted(directory.glob("Gate*.vo")):
        stat = path.stat()
        stamps.append((path.name, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return str(directory), tuple(stamps)


_decisions: dict[tuple[Library, str, tuple[tuple[str, str], ...]], Decision] = {}


def _communicate(
    process: subprocess.Popen[str], deadline: float, cancel: Event | None
) -> tuple[str, str]:
    """Wait within the common deadline; cancellation stops the verification process."""
    if cancel is None:
        return process.communicate(timeout=max(0.0, deadline - time.monotonic()))
    while True:
        if cancel.is_set():
            raise InterruptedError("verification cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, 0)
        try:
            return process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise


def verify(
    text: str,
    bindings: Sequence[Binding],
    *,
    timeout: float = TIMEOUT,
    cancel: Event | None = None,
) -> Decision:
    """Verify the module's text against the binding of each of its entry kernels.

    Accepted means Rocq checked `verify_certificate_sound` for this text and these
    bindings. In every launch that establishes the bindings' facts, every
    thread of every entry kernel keeps each global access in the region its
    binding allows and each local store in a live local variable, aligns
    every access to its width, stores only values in the range to an index
    output, runs no trap, and ends. Refused carries the failed check and the
    PTX instruction where it fails, with the first accesses that the check
    refuses, or says that the check did not finish in `timeout` seconds.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if cancel is not None and cancel.is_set():
        raise InterruptedError("verification cancelled")
    deadline = time.monotonic() + timeout
    directory = formal_dir().resolve()
    library = _library(directory)
    bindings = tuple(bindings)
    # Compare the original text and the exact printed binding terms before
    # packing PTX. Export validates primitive values before consulting the
    # cache; malformed inputs must never reuse an earlier decision.
    key = (library, text, tuple((_quoted(b.kernel), binding_term(b)) for b in bindings))
    if key in _decisions:
        return _decisions[key]
    query = export_query(text, bindings)
    # A failed or timed-out process cannot accept a module. Refusal diagnostics
    # run only when the checked auxiliary proof fails, using the materialized
    # certificate. An unexplained compiler failure is an internal error.
    with _coqc(query, directory) as process:
        try:
            diagnosis, errors = _communicate(process, deadline, cancel)
            if process.returncode == 0:
                decision: Decision = Accepted()
            else:
                found = _REFUSAL.search(diagnosis)
                listed = _UNPROVED.search(diagnosis)
                if found is None or listed is None:
                    raise RuntimeError(
                        f"Rocq refused the module without a reason:\n{errors[-2000:]}"
                    )
                decision = Refused(
                    found.group(1).replace('""', '"'),
                    tuple(m.replace('""', '"') for m in _STRING.findall(listed.group(1))),
                )
        except subprocess.TimeoutExpired:
            return Refused(f"{TIMEOUT_REASON} {timeout:g} seconds", ())
    if _library(directory) != library:
        raise RuntimeError("the compiled gate changed during verification; use a stable build")
    _decisions[key] = decision
    return decision


REGISTRATION_SECONDS = 600.0


def timed_out(decision: Decision) -> bool:
    """Whether `decision` is a refusal because a check did not finish in its time."""
    return isinstance(decision, Refused) and decision.reason.startswith(TIMEOUT_REASON)


class VerificationBudget:
    """One wall-clock budget for the checks of one registration: the check of the
    text as written and every round of the guard pass.

    Each check may take the time that remains of the budget; `TIMEOUT` bounds
    only a single `verify` outside a registration. A check that does not finish
    before the budget ends, and every check after it, is a refusal for the
    budget's seconds. Cache hits consume the same deadline; cancellation is
    checked before every round.
    """

    def __init__(
        self,
        seconds: float = REGISTRATION_SECONDS,
        *,
        cancel: Event | None = None,
        checker: Callable[..., Decision] = verify,
    ) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("registration timeout must be finite and positive")
        self.seconds = seconds
        self.deadline = time.monotonic() + seconds
        self.cancel = cancel
        self.checker = checker

    def __call__(self, text: str, binding: Binding) -> Decision:
        if self.cancel is not None and self.cancel.is_set():
            raise InterruptedError("verification cancelled")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            return Refused(f"{TIMEOUT_REASON} {self.seconds:g} seconds", ())
        decision = self.checker(text, (binding,), timeout=remaining, cancel=self.cancel)
        if timed_out(decision):
            return Refused(f"{TIMEOUT_REASON} {self.seconds:g} seconds", ())
        return decision
