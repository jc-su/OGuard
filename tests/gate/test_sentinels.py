"""A masked search load's sentinel retains the predicate needed by its cursor."""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify


@pytest.mark.parametrize("reverse_comparison", [False, True])
@pytest.mark.parametrize("past_end", [None, "probe", "final"])
def test_sentinel_search_checks_every_member_access(reverse_comparison, past_end):
    text = (directory_of("fable", "sorted_contains") / "triton.ptx").read_bytes().decode("ascii")
    if reverse_comparison:
        for predicate, loaded, key in (("%p9", "%rd16", "%rd7"), ("%p10", "%rd18", "%rd9")):
            before = f"setp.lt.s64 \t{predicate}, {loaded}, {key};"
            assert text.count(before) == 1
            text = text.replace(before, f"setp.gt.s64 {predicate}, {key}, {loaded};")
    if past_end is not None:
        # Static negative controls; never launch the modified PTX.
        loaded, address = ("%rd16", "%rd17") if past_end == "probe" else ("%rd26", "%rd27")
        before = f"ld.global.b64 {{ {loaded} }}, [ {address} + 0 ];"
        assert text.count(before) == 1
        text = text.replace(before, f"ld.global.b64 {{ {loaded} }}, [ {address} + 8 ];")
    binding = binding_of("fable", "sorted_contains", "triton", text)
    decision = verify(text, [binding], timeout=90)
    if past_end is None:
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
