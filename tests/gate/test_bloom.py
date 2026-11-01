"""A bound carried through a shift or a remainder: the word index of a Bloom filter.

The bit position `p` is `h & (32 * words - 1)` or `h % (32 * words)`, and
the kernel reads word `p >> 5` of `bits`, which has `words` elements. Rule R4
bounds `p` below `32 * words`, and the check carries that bound to the atom
`p >> 5` through the shift link `0 <= p - 32 (p >> 5) <= 31` and the
reduction of the eliminated constraint `32 (p >> 5) - 32 words <= -1` to
`(p >> 5) - words <= -1`. nvcc writes the word's byte offset as
`(p >> 3) & 0x1ffffffc`, which is `4 * (p >> 5)` for a `p` below `2^32`.

Each variant reads one word further than the bound allows and must be
refused: a shift by 4 instead of 5 in the Numba kernel, a shift by 2
instead of 3 in the CUDA C++ kernel, and a divisor `64 * words` in place of
`32 * words` in the CUDA C++ kernel that uses a remainder.
"""

from dataclasses import replace

import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify

UNPROVED = "the access is not proved inside its buffer, a shared variable, or a local variable"


def _text(model: str, task: str, toolchain: str) -> str:
    return (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()


@pytest.mark.parametrize(
    ("model", "toolchain", "before", "after", "instruction"),
    [
        (
            "sonnet",
            "numba",
            "shr.u64 \t%rd29, %rd25, 5;",
            "shr.u64 \t%rd29, %rd25, 4;",
            "ld.u32 %rd32",
        ),
        (
            "sonnet",
            "nvcc",
            "shr.u64 \t%rd22, %rd18, 3;",
            "shr.u64 \t%rd22, %rd18, 2;",
            "ld.global.nc.u32 %r6",
        ),
        (
            "haiku",
            "nvcc",
            "shl.b64 \t%rd4, %rd14, 5;",
            "shl.b64 \t%rd4, %rd14, 6;",
            "ld.global.u32 %r17",
        ),
    ],
    ids=["Numba mask and shift", "CUDA C++ mask and masked shift", "CUDA C++ remainder"],
)
@pytest.mark.parametrize("past_words", [False, True])
def test_a_bloom_probe_is_accepted_and_a_read_past_words_is_refused(
    model: str, toolchain: str, before: str, after: str, instruction: str, past_words: bool
) -> None:
    text = _text(model, "bloom_probe", toolchain)
    if past_words:
        # A static negative control; never launch the modified PTX.
        assert text.count(before) == 1
        text = text.replace(before, after)
    binding = binding_of(model, "bloom_probe", toolchain, text)
    decision = verify(text, [binding], timeout=120)
    if past_words:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
        assert f"{UNPROVED}: {instruction}" in decision.reason, decision
    else:
        assert decision == Accepted()


@pytest.mark.parametrize("extra_block", [False, True])
def test_triton_signed_index_respects_the_launch_grid_bound(extra_block: bool) -> None:
    """The final permitted block fits signed32; one additional block does not."""
    text = _text("haiku", "bloom_probe", "triton")
    binding = binding_of("haiku", "bloom_probe", "triton", text)
    assert binding.grid == (8388608, 1, 1)
    if extra_block:
        # Only a static contract check: never execute this wider launch.
        binding = replace(binding, grid=(8388609, 1, 1))
    decision = verify(text, [binding], timeout=120)
    if extra_block:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
        assert UNPROVED in decision.reason, decision
    else:
        assert decision == Accepted()
