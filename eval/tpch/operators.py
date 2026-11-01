"""The operators of the TPC-H pipelines.

Every operator of the pipelines is model-written: sonnet's Numba kernels of
the corpus that the gate accepts (`q6_sum`, `gather`, `projection`,
`segmented_sum`), registered from their recorded PTX with the reviewed roles
(`eval/corpus/adapter.py`), and the filter, sonnet's `filter_select` of the
first repair round (`repair1/numba.py`). The first `filter_select` stores a
row id at the position an atomic counter returns without checking it, a
defect the gate refuses; the revision checks the position against the
output's length, and the gate accepts it. The filter is registered twice:
with `sel` as an index output, which the gate proves in range (the static
path), and with `sel` as a plain output, which the catalog records as raw,
so that a consumer must be a guarded launch (the run-time check, design
section 6.4).
"""

import dataclasses
from dataclasses import dataclass
from typing import assert_never

from eval.corpus.adapter import binding_of, claim_of, directory_of
from eval.corpus.run import TOOLCHAINS
from eval.corpus.sources import operator_of
from opgate import signature as sig
from opgate.binding import ElementType
from opgate.executor import Executor, Module
from opgate.signature import Direction

__all__ = ["BLOCK", "INDEX", "MODEL", "Operators", "register"]

MODEL = "sonnet"
TOOLCHAIN = "numba"
FILTER = "filter_select"
BLOCK = (256, 1, 1)
INDEX = ElementType(4, True)


@dataclass(frozen=True)
class Operators:
    """The registered operators of the pipelines."""

    q6: Module
    gather: Module
    projection: Module
    segmented_sum: Module
    select_static: Module  # the filter whose output is an index output
    select_raw: Module  # the same filter whose output the catalog records as raw


def _corpus(executor: Executor, task: str) -> Module:
    text = (directory_of(MODEL, task) / TOOLCHAINS[TOOLCHAIN]).read_text()
    return executor.register(text, binding_of(MODEL, task, TOOLCHAIN, text))


def _raw(argument: sig.Argument) -> sig.Argument:
    """The argument with an index output as a plain output."""
    match argument:
        case sig.IndexOut(name=name, ref=ref, rows=rows):
            return sig.Span(name, ref, Direction.WRITE, rows)
        case sig.Span() | sig.IndexIn() | sig.Offsets() | sig.Scalar():
            return argument
        case _ as never:
            assert_never(never)


def _filters(executor: Executor) -> tuple[Module, Module]:
    """The filter with `sel` as an index output, and with `sel` as a plain output."""
    static = operator_of(MODEL, FILTER, TOOLCHAIN, directory_of(MODEL, FILTER) / "repair1")
    arguments = tuple(map(_raw, static.signature.arguments))
    raw = dataclasses.replace(
        static, signature=dataclasses.replace(static.signature, arguments=arguments)
    )
    _, elements = claim_of(MODEL, FILTER)
    return (
        executor.register(static.text, static.binding(elements)),
        executor.register(raw.text, raw.binding(elements)),
    )


def register(executor: Executor) -> Operators:
    """Verify and load every operator of the pipelines."""
    static, raw = _filters(executor)
    return Operators(
        _corpus(executor, "q6_sum"),
        _corpus(executor, "gather"),
        _corpus(executor, "projection"),
        _corpus(executor, "segmented_sum"),
        static,
        raw,
    )
