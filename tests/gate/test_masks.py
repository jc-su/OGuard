"""Masked loads, selects of two registers, and exclusive-or masks (design decision 24).

Triton writes a masked load `tl.load(p, mask=m, other=0)` as `mov r, 0;
@m ld r, [a]`. A masked load of an index input bounds a later access under
the same mask: the gather `col[sel[i]]` with both loads masked is accepted,
and the same gather with the load of `col` unmasked, which reads `col[0]`
of an empty `col` when the mask fails, is refused.

`tl.where(active, (slot + 1) & mask, slot)` is a `selp` of two registers
that both lie below the capacity, so the slot does; with the unmasked hash in
place of the old slot, it does not.

A loop mask `pending := pending xor stored`, where `stored` implies
`pending`, keeps the comparisons of `pending`; the exclusive or of two
unrelated predicates keeps none.

Triton loads a 64-bit capacity that it uses at 32 bits with `ld.param.b32`,
its low half (design decision 26): the 32-bit mask `capacity - 1` bounds the
slot, and a mask from the parameter's high half does not.
"""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Binding, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Refused, verify
from opgate.signature import Affine, Direction

INT32 = ElementType(4, True)
INT64 = ElementType(8, True)

GATHER = sig.Signature(
    sizes=(sig.SizeSymbol("n"), sig.SizeSymbol("m")),
    arguments=(
        sig.Span("col", "col", Direction.READ, Affine.symbol(0)),
        sig.IndexIn("sel", "sel", Affine.symbol(1), "col"),
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(1)),
        sig.Scalar("m", sig.SizeOf(Affine.symbol(1))),
    ),
)
TABLE = sig.Signature(
    sizes=(sig.SizeSymbol("capacity", 1),),
    arguments=(
        sig.Span("table", "table", Direction.READ, Affine.symbol(0)),
        sig.Scalar("capacity", sig.SizeOf(Affine.symbol(0))),
    ),
)


def _binding(signature: sig.Signature, elements: dict[str, ElementType], params: int) -> Binding:
    names = tuple(f"kernel_param_{k}" for k in range(params))
    return derive_binding(
        signature, elements, Convention.C, "kernel", names, (256, 1, 1), GRID_LIMIT, Affine(0)
    )


GATHER_BINDING = _binding(GATHER, {"col": INT32, "sel": INT32, "out": INT32}, 4)
TABLE_BINDING = _binding(TABLE, {"table": INT64}, 2)


def kernel(params: int, body: str) -> str:
    """A kernel of `params` 64-bit parameters, with `%r1` the thread index."""
    declared = ",\n".join(f"\t.param .u64 kernel_param_{k}" for k in range(params))
    return f""".version 8.0
.target sm_80
.address_size 64

.visible .entry kernel(
{declared}
)
{{
	.reg .pred 	%p<5>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<13>;

	mov.u32 	%r1, %tid.x;
{body}
	ret;
}}
"""


# col, sel, out, m; %p1 is the mask i < m, %rd6 the byte offset of row i
GATHER_HEAD = """	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	ld.param.u64 	%rd3, [kernel_param_2];
	ld.param.u64 	%rd4, [kernel_param_3];
	cvt.u64.u32 	%rd5, %r1;
	setp.lt.u64 	%p1, %rd5, %rd4;
	shl.b64 	%rd6, %rd5, 2;
	add.s64 	%rd7, %rd2, %rd6;
	mov.u32 	%r2, 0x0;
	@%p1 ld.global.b32 { %r2 }, [ %rd7 + 0 ];
	mul.wide.s32 	%rd8, %r2, 4;
	add.s64 	%rd9, %rd1, %rd8;
	mov.u32 	%r3, 0x0;
"""
GATHER_TAIL = """	add.s64 	%rd10, %rd3, %rd6;
	@%p1 st.global.b32 [ %rd10 + 0 ], { %r3 };"""
MASKED_GATHER = GATHER_HEAD + "\t@%p1 ld.global.b32 { %r3 }, [ %rd9 + 0 ];\n" + GATHER_TAIL
UNMASKED_GATHER = GATHER_HEAD + "\tld.global.b32 { %r3 }, [ %rd9 + 0 ];\n" + GATHER_TAIL

# table, capacity; %rd5 = tid & mask and %rd7 = (%rd5 + 1) & mask lie below the capacity
SLOT_HEAD = """	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	cvt.u64.u32 	%rd3, %r1;
	add.s64 	%rd4, %rd2, -1;
	and.b64 	%rd5, %rd3, %rd4;
	add.s64 	%rd6, %rd5, 1;
	and.b64 	%rd7, %rd6, %rd4;
	setp.eq.u32 	%p1, %r1, 0;
"""
SLOT_TAIL = """	shl.b64 	%rd9, %rd8, 3;
	add.s64 	%rd10, %rd1, %rd9;
	ld.global.u64 	%rd11, [%rd10];"""
SELECT_OF_SLOTS = SLOT_HEAD + "\tselp.b64 \t%rd8, %rd7, %rd5, %p1;\n" + SLOT_TAIL
SELECT_OF_HASH = SLOT_HEAD + "\tselp.b64 \t%rd8, %rd7, %rd3, %p1;\n" + SLOT_TAIL


def mask32(offset: int) -> str:
    """table, capacity; %rd8 = tid & mask, with the 32-bit mask from the capacity
    parameter's bytes at `offset`."""
    return (
        f"""	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.b32 	%r2, [kernel_param_1+{offset}];
	add.s32 	%r3, %r2, -1;
	and.b32 	%r4, %r1, %r3;
	cvt.u64.u32 	%rd8, %r4;
"""
        + SLOT_TAIL
    )


def high_half_mask() -> str:
    """Exercise high-half mask arithmetic using supported scalar instructions."""
    body = mask32(4)
    load = "\tld.param.b32 \t%r2, [kernel_param_1+4];"
    assert body.count(load) == 1
    return body.replace(
        load,
        "ld.param.u64 %rd2, [kernel_param_1];\nshr.u64 %rd12, %rd2, 32;\ncvt.u32.u64 %r2, %rd12;",
    )


# col, sel, out, m; %p2 implies %p1, the mask i < m
XOR_HEAD = """	ld.param.u64 	%rd3, [kernel_param_2];
	ld.param.u64 	%rd4, [kernel_param_3];
	cvt.u64.u32 	%rd5, %r1;
	setp.lt.u64 	%p1, %rd5, %rd4;
	setp.eq.u32 	%p4, %r1, 7;
	and.pred 	%p2, %p1, %p4;
"""
XOR_TAIL = """	shl.b64 	%rd6, %rd5, 2;
	add.s64 	%rd10, %rd3, %rd6;
	@%p3 st.global.b32 [ %rd10 + 0 ], { %r1 };"""
XOR_OF_IMPLIED = XOR_HEAD + "\txor.pred \t%p3, %p1, %p2;\n" + XOR_TAIL
XOR_OF_UNRELATED = XOR_HEAD + "\txor.pred \t%p3, %p1, %p4;\n" + XOR_TAIL

UNPROVED = "the access is not proved inside its buffer, a shared variable, or a local variable"


@pytest.mark.parametrize(
    ("text", "binding"),
    [
        (kernel(4, MASKED_GATHER), GATHER_BINDING),
        (kernel(2, SELECT_OF_SLOTS), TABLE_BINDING),
        (kernel(4, XOR_OF_IMPLIED), GATHER_BINDING),
        (kernel(2, mask32(0)), TABLE_BINDING),
    ],
    ids=[
        "masked gather",
        "select of two slots",
        "exclusive or of an implied mask",
        "32-bit mask of the capacity",
    ],
)
def test_a_masked_or_selected_access_that_stays_inside_is_accepted(
    text: str, binding: Binding
) -> None:
    assert verify(text, [binding]) == Accepted()


@pytest.mark.parametrize(
    ("text", "binding", "instruction"),
    [
        (kernel(4, UNMASKED_GATHER), GATHER_BINDING, "ld.global.b32 {%r3}"),
        (kernel(2, SELECT_OF_HASH), TABLE_BINDING, "ld.global.u64 %rd11"),
        (kernel(4, XOR_OF_UNRELATED), GATHER_BINDING, "@%p3 st.global.b32"),
        (kernel(2, high_half_mask()), TABLE_BINDING, "ld.global.u64 %rd11"),
    ],
    ids=[
        "unmasked load of the gathered row",
        "select of the hash",
        "exclusive or of unrelated masks",
        "32-bit mask of the capacity's high half",
    ],
)
def test_an_access_that_the_mask_or_select_does_not_bound_is_refused(
    text: str, binding: Binding, instruction: str
) -> None:
    decision = verify(text, [binding])
    assert isinstance(decision, Refused)
    assert f"{UNPROVED}: {instruction}" in decision.reason


def test_a_partial_parameter_load_at_an_unsupported_offset_fails_fast() -> None:
    # Keep the original compiler form as a separate frontend boundary control.
    decision = verify(kernel(2, mask32(4)), [TABLE_BINDING])
    assert isinstance(decision, Refused)
    assert (
        "unmodeled instruction effects: ld: ld.param.b32 %r2, [kernel_param_1+4]" in decision.reason
    )
