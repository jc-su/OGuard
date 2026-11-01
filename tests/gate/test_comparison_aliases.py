"""Saved comparison operands remain valid even when their source can wrap."""

from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.binding import Base, Binding, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction


@pytest.mark.parametrize("upper,accepted", [(8, True), (9, False)])
def test_saved_raw_index_survives_possible_wrap(upper, accepted):
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 input, .param .u64 output) {{
.reg .pred %p;
.reg .b64 %a, %b;
.reg .b64 %input, %base, %index, %address;
ld.param.u64 %input, [input];
ld.param.u64 %base, [output];
ld.global.u64 %a, [%input];
cvt.u64.u64 %index, %a;
mov.u64 %b, {upper};
setp.lt.u64 %p, %a, %b;
add.u64 %a, %a, 1;
@!%p bra END;
add.u64 %address, %base, %index;
st.global.u8 [%address], 1;
END:
ret;
}}
"""
    binding = Binding(
        "kernel",
        (),
        (
            ("input", Base(Span(Direction.READ, Affine(8), 8, None, False))),
            ("output", Base(Span(Direction.WRITE, Affine(8), 8, None, False))),
        ),
        (1, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    result = verify(text, [binding], timeout=60)
    if accepted:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result


@pytest.mark.slow
def test_saved_loop_comparison_retains_constant_step_induction():
    archive = Path(__file__).with_name("fixtures") / "signed_guard_frozen_probe_2026-09-30.json"
    text = read_artifact(archive, "frozen/fable/string_contains/numba/0", "ptx")
    binding = binding_of("fable", "string_contains", "numba", text)
    assert verify(text, [binding], timeout=180) == Accepted()
