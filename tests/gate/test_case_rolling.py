"""Thread partitions and loop cases preserve each shared allocation boundary."""

from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify, zeroed


@pytest.mark.parametrize(
    "allocation, reduction, accepted",
    [
        pytest.param(None, 0, True, id="original"),
        pytest.param("xs", 8, True, id="xs-exact-boundary"),
        pytest.param("xs", 16, False, id="xs"),
        pytest.param("suffix", 8, False, id="suffix"),
        pytest.param("prefix_carry", 8, False, id="prefix_carry"),
        pytest.param("suffix_carry", 8, False, id="suffix_carry"),
        pytest.param("prefix_flag", 1, False, id="prefix_flag"),
        pytest.param("suffix_flag", 1, False, id="suffix_flag"),
    ],
)
@pytest.mark.slow
def test_case_rolling_allocation_boundaries(allocation, reduction, accepted):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29z.json"
    text = read_artifact(archive, "opus/rolling_sum/nvcc/0", "ptx")
    if allocation is not None:
        # Static checks only: constrained allocations are never executed.
        # xs has one unused tail element. With t=tid.x in [0,255], window w
        # in [1,1024], and j in [0,3], its prefix scan accesses 4*t+w-1+j
        # and final scan accesses t+w-1+256*j. Both reach index 2046 exactly.
        # Fill indices are < w+1023 <= 2047. The descending suffix scan
        # loads only below r9: 1023 when 1024%w=0, otherwise w+1023-1024%w,
        # also <= 2047. Every xs access is nonnegative and eight bytes wide,
        # so the exact extent is (2046+1)*8=16376 bytes. Shrinking by eight
        # is safe; shrinking by sixteen removes an accessed element.
        size = {
            "xs": 16384,
            "suffix": 8192,
            "prefix_carry": 2048,
            "suffix_carry": 2048,
            "prefix_flag": 256,
            "suffix_flag": 256,
        }[allocation]
        name = f"_ZZ6kernelE{len(allocation)}{allocation}"
        before, after = f"{name}[{size}]", f"{name}[{size - reduction}]"
        assert text.count(before) == 1
        text = text.replace(before, after)
    binding = binding_of("opus", "rolling_sum", "nvcc", text)
    result = verify(zeroed(text, binding), [binding], timeout=180)
    if accepted:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result
