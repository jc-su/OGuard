"""Compiler-lowered short loops and constant XOR keep proved address bounds."""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify, zeroed


@pytest.mark.slow
@pytest.mark.parametrize(("model", "task"), [("opus", "rolling_sum"), ("haiku", "string_contains")])
def test_current_numba_range_loops_are_accepted(model: str, task: str) -> None:
    text = (directory_of(model, task) / "numba.O1.ptx").read_text()
    binding = binding_of(model, task, "numba", text)
    assert verify(zeroed(text, binding), [binding], timeout=180) == Accepted()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("number", "nbytes", "reduction", "accepted"),
    [
        (9, 16384, 8, True),
        (9, 16384, 16, False),
        (10, 8192, 8, False),
        (11, 2048, 8, False),
        (12, 2048, 8, False),
        (13, 1024, 4, False),
        (14, 1024, 4, False),
    ],
)
def test_current_numba_rolling_allocation_bounds(
    number: int, nbytes: int, reduction: int, accepted: bool
) -> None:
    text = (directory_of("opus", "rolling_sum") / "numba.O1.ptx").read_text()
    binding = binding_of("opus", "rolling_sum", "numba", text)
    original = f"_cudapy_smem_{number}[{nbytes}]"
    assert text.count(original) == 1
    # Static verification only: never execute modified kernels. Tile 9 has
    # prefix/final indices 4*t+w-1+j and t+w-1+256*j for t=0..255, w=1..1024,
    # j=0..3, reaching 2046. Fill/suffix guards also keep indices < 2047.
    # All indices are nonnegative, so Numba's negative-index correction is 0.
    # The exact extent is 2047*8=16376; 16368 removes an accessed element.
    text = text.replace(original, f"_cudapy_smem_{number}[{nbytes - reduction}]")
    decision = verify(zeroed(text, binding), [binding], timeout=180)
    if accepted:
        assert decision == Accepted(), decision
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision


@pytest.mark.slow
@pytest.mark.parametrize("register", ["rd51", "rd56", "rd63"])
def test_current_numba_string_access_bounds(register: str) -> None:
    text = (directory_of("haiku", "string_contains") / "numba.O1.ptx").read_text()
    binding = binding_of("haiku", "string_contains", "numba", text)
    original = f"[%{register}]"
    assert text.count(original) == 1
    # Static verification only: this can access one byte past its allocation.
    text = text.replace(original, f"[%{register}+1]")
    decision = verify(zeroed(text, binding), [binding], timeout=180)
    assert isinstance(decision, Refused), decision
    assert not decision.reason.startswith(TIMEOUT_REASON), decision
