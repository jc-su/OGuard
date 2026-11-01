"""The guard pass's status buffer is a flag buffer (DESIGN decision 47).

The binding declares the status buffer as a flag span, and the gate proves
that every store to a flag span stores the number 1 into one whole word, so
a status word that the host zeroes is 0 or 1 after a launch, and 1 if a
failed check stored to it. A kernel stores to word `at` of a buffer of four
words, with `at` in `[0, 7]`: the gate does not prove the store in bounds,
and the pass checks it. The guarded text is accepted; the same text with a
store of 0, of 2, of a register that holds 1, or of two bytes to the status
buffer is refused. A text that itself declares the status parameter is
refused by `opgate.hosts.register` and by the service's check, so only the
pass adds it.
"""

from threading import Event
from typing import cast

import pytest

from opgate.binding import BASE_ALIGN, Base, Binding, Bounded, ElementType, IndexRange, Span
from opgate.executor import Executor, RefusedError
from opgate.frontends.guard import STATUS, STATUS_DECLARED, declares_status, guard, reports
from opgate.gate import Accepted, Refused, verify
from opgate.hosts import register
from opgate.signature import Affine, Direction

FLAG = "the write to the flag buffer is not a store of the number 1 into one whole word"

TEXT = """.version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(
	.param .u64 out,
	.param .u32 at
)
{
	.reg .b32 %r<2>;
	.reg .b64 %rd<5>;
	ld.param.u64 %rd1, [out];
	ld.param.u32 %r1, [at];
	cvta.to.global.u64 %rd2, %rd1;
	mul.wide.u32 %rd3, %r1, 4;
	add.s64 %rd4, %rd2, %rd3;
	st.global.u32 [%rd4], 7;
	ret;
}
"""
BINDING = Binding(
    "kernel",
    (),
    (
        ("out", Base(Span(Direction.WRITE, Affine(16), BASE_ALIGN, None, False))),
        ("at", Bounded(0, 7)),
    ),
    (1, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def _check_text(text: str, binding: Binding) -> Accepted | Refused:
    return verify(text, [binding])


def test_the_status_buffer_is_a_flag_span_that_the_checks_set_to_1() -> None:
    guarded = guard(TEXT, BINDING, _check_text)
    assert guarded.decision == Accepted(), guarded.decision
    assert reports(guarded.binding)
    name, parameter = guarded.binding.parameters[-1]
    assert name == STATUS
    assert parameter == Base(Span(Direction.WRITE, Affine(4), BASE_ALIGN, None, True))
    assert guarded.text.count("], 1;") == 1


@pytest.mark.parametrize(
    "store",
    ["], 0;", "], 2;", "], %one;", "u16"],
    ids=["zero", "two", "a register that holds 1", "two bytes"],
)
def test_a_status_store_of_anything_but_the_number_1_is_refused(store: str) -> None:
    guarded = guard(TEXT, BINDING, _check_text)
    assert guarded.decision == Accepted(), guarded.decision
    text = guarded.text
    if store == "u16":
        failing = next(line for line in text.split("\n") if line.endswith("], 1;"))
        text = text.replace(failing, failing.replace("st.global.u32", "st.global.u16"))
    else:
        text = text.replace("], 1;", store)
    if "%one" in text:
        text = text.replace("\t.reg .b64 %rd<5>;", "\t.reg .b64 %rd<5>;\n\t.reg .b32 %one;", 1)
        text = text.replace(
            "\tld.param.u64 %rd1, [out];", "\tld.param.u64 %rd1, [out];\n\tmov.u32 %one, 1;", 1
        )
    decision = verify(text, [guarded.binding])
    assert isinstance(decision, Refused), decision
    assert FLAG in decision.reason, decision.reason


def test_a_flag_span_is_written_only_and_has_no_index_range() -> None:
    with pytest.raises(ValueError, match="flag span"):
        Span(Direction.READ_WRITE, Affine(4), BASE_ALIGN, None, True)
    index = IndexRange(ElementType(4, False), Affine(0), Affine(1))
    with pytest.raises(ValueError, match="flag span"):
        Span(Direction.WRITE, Affine(4), BASE_ALIGN, index, True)


DECLARED = TEXT.replace("\t.param .u32 at\n", f"\t.param .u32 at,\n\t.param .u64 {STATUS}\n")


def test_a_text_that_declares_the_status_parameter_is_refused() -> None:
    assert declares_status(DECLARED)
    assert not declares_status(TEXT)
    # the refusal comes before the executor registers anything
    with pytest.raises(RefusedError, match=STATUS_DECLARED):
        register(cast(Executor, object()), DECLARED, BINDING)


@pytest.mark.gpu
def test_the_service_refuses_a_text_that_declares_the_status_parameter() -> None:
    # the service's modules initialize the driver when they are imported
    from opgate.hosts.service.server import _check

    checked = _check(DECLARED, BINDING, 60.0, Event())
    assert checked.decision == Refused(STATUS_DECLARED, ())
    assert checked.guarded is None
