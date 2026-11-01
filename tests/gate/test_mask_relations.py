"""Relations between rounded masks, complemented masks, and symbolic clamps."""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine


def _assert(text, binding, accepted):
    result = verify(text, [binding], timeout=180)
    if accepted:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result


@pytest.mark.parametrize("tile_bits", [4, 10])
@pytest.mark.parametrize("displacement", [-1, 0, 1])
def test_complement_mask_tile_reverse_requires_both_bounds(tile_bits, displacement):
    signature = sig.Signature(
        sizes=(sig.SizeSymbol("n", 0, 2**20),),
        arguments=(
            sig.Span("out", "out", sig.Direction.WRITE, Affine.symbol(0)),
            sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
        ),
    )
    binding = derive_binding(
        signature,
        {"out": ElementType(4, False)},
        Convention.C,
        "kernel",
        ("out", "n"),
        (128, 1, 1),
        GRID_LIMIT,
        Affine(0),
    )
    tile = 2**tile_bits
    # For i<n, b=i rounded down to the tile boundary and h=min(b+tile,n),
    # the index h-(i-b)-1 is in [0,n-1]. The two displaced forms can
    # access before zero or at n. All cases are static; none are executed.
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 out, .param .u64 n) {{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .b64 %rd<12>;
ld.param.u64 %rd1, [out];
ld.param.u64 %rd2, [n];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mad.lo.u32 %r4, %r2, %r3, %r1;
cvt.u64.u32 %rd3, %r4;
setp.ge.s64 %p1, %rd3, %rd2;
@%p1 bra DONE;
not.b64 %rd4, %rd3;
and.b64 %rd5, %rd3, {2**32 - tile};
add.s64 %rd6, %rd5, {tile};
min.s64 %rd7, %rd6, %rd2;
or.b64 %rd8, %rd4, {2**62 - tile};
add.s64 %rd9, %rd7, %rd8;
add.s64 %rd9, %rd9, {displacement};
shl.b64 %rd10, %rd9, 2;
add.s64 %rd11, %rd1, %rd10;
st.global.u32 [%rd11], 0;
DONE:
ret;
}}"""
    _assert(text, binding, displacement == 0)


@pytest.mark.parametrize("minimum", [False, True])
@pytest.mark.parametrize("displacement", [-1, 0, 1])
def test_symbolic_clamp_transports_a_shared_relative_bound(minimum, displacement):
    signature = sig.Signature(
        sizes=(),
        arguments=(
            sig.Span("out", "out", sig.Direction.WRITE, Affine(16)),
            sig.Scalar("start", sig.Between(16, 2**30)),
            sig.Scalar("u", sig.Between(1, 16)),
            sig.Scalar("v", sig.Between(1, 16)),
        ),
    )
    binding = derive_binding(
        signature,
        {"out": ElementType(8, True)},
        Convention.C,
        "kernel",
        ("out", "start", "u", "v"),
        (1, 1, 1),
        GRID_LIMIT,
        Affine(0),
    )
    op = "min" if minimum else "max"
    direction = "add" if minimum else "sub"
    difference = "%rd7, %rd2" if minimum else "%rd2, %rd7"
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 out, .param .u64 start, .param .u64 u, .param .u64 v) {{
.reg .b64 %rd<11>;
ld.param.u64 %rd1, [out];
ld.param.u64 %rd2, [start];
ld.param.u64 %rd3, [u];
ld.param.u64 %rd4, [v];
{direction}.s64 %rd5, %rd2, %rd3;
{direction}.s64 %rd6, %rd2, %rd4;
{op}.s64 %rd7, %rd5, %rd6;
sub.s64 %rd8, {difference};
add.s64 %rd8, %rd8, {displacement - 1};
shl.b64 %rd9, %rd8, 3;
add.s64 %rd10, %rd1, %rd9;
st.global.u64 [%rd10], 0;
ret;
}}"""
    _assert(text, binding, displacement == 0)


@pytest.mark.parametrize("variant", ["original", "before", "after", "inclusive"])
def test_current_cuda_tile_reverse_mask_relations(variant):
    from eval.corpus.adapter import binding_of, directory_of

    text = (directory_of("haiku", "tile_reverse") / "kernel.O1.ptx").read_text()
    binding = binding_of("haiku", "tile_reverse", "nvcc", text)
    changes = {
        "before": ("[%rd13]", "[%rd13-4]"),
        "after": ("[%rd13]", "[%rd13+4]"),
        "inclusive": ("setp.ge.s64", "setp.gt.s64"),
    }
    if variant != "original":
        before, after = changes[variant]
        assert text.count(before) == 1
        text = text.replace(before, after)
    _assert(text, binding, variant == "original")
