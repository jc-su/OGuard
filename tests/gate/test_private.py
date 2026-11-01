"""Fault freedom in shared and local memory: every access lies inside a variable.

An access past the memory a block or a thread has faults, and the fault
ends the context for every tenant, though the address lies in the window
of that memory (design section 5.8). Each kernel counts into `counts`, a
shared array of 256 32-bit elements, at `tid + k` in a block of 256
threads, reads a local array of 8 bytes, or stores to and loads from
`flag`, a scalar shared variable, as nvcc declares a demoted `__shared__`
variable, with a load that sign-extends into a 64-bit register. The
accepted kernels stay inside their variable; each refused variant reaches
one element past it in shared memory, through a generic address in the
shared window, or in local memory, or loads constant memory, which the
translation does not model.
"""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.signature import Affine, Direction

SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"),),
    arguments=(
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
BINDING = derive_binding(
    SIGNATURE,
    {"out": ElementType(4, True)},
    Convention.C,
    "kernel",
    ("kernel_param_0", "kernel_param_1"),
    (256, 1, 1),
    GRID_LIMIT,
    Affine(0),
)


def kernel(body: str) -> str:
    """The kernel with `counts`, 1024 shared bytes, `flag`, 4 shared bytes, `slot`, 8 local
    bytes, and `%r1 = tid`."""
    return f""".version 8.0
.target sm_80
.address_size 64

.const .align 4 .b8 table[16];

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1
)
{{
	.shared .align 4 .b8 	counts[1024];
	.shared .align 4 .u32 	flag;
	.local .align 8 .b8 	slot[8];
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<8>;

	mov.u32 	%r1, %tid.x;
{body}
	ret;
}}
"""


def shared_count(k: int) -> str:
    return f"""	add.s32 	%r2, %r1, {k};
	shl.b32 	%r3, %r2, 2;
	mov.u32 	%r4, counts;
	add.s32 	%r5, %r4, %r3;
	atom.shared.add.u32 	%r6, [%r5], 1;"""


def generic_count(k: int) -> str:
    return f"""	add.s32 	%r2, %r1, {k};
	mul.wide.u32 	%rd1, %r2, 4;
	mov.u64 	%rd2, counts;
	cvta.shared.u64 	%rd3, %rd2;
	add.s64 	%rd4, %rd3, %rd1;
	atom.add.u32 	%r6, [%rd4], 1;"""


def local_load(offset: int) -> str:
    return f"""	mov.u64 	%rd5, slot;
	st.local.u32 	[%rd5], %r1;
	ld.local.u32 	%r7, [%rd5+{offset}];"""


def flag_access(offset: int) -> str:
    return f"""	st.shared.u32 	[flag], %r1;
	ld.shared.s32 	%rd6, [flag+{offset}];"""


CONSTANT = "\tld.const.u32 \t%r7, [table];"


@pytest.mark.parametrize(
    "body",
    [shared_count(0), generic_count(0), local_load(4), flag_access(0)],
    ids=["shared", "generic in the shared window", "local", "scalar shared"],
)
def test_an_access_inside_its_variable_is_accepted(body: str) -> None:
    assert verify(zeroed(kernel(body), BINDING), [BINDING]) == Accepted()


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            shared_count(1),
            "the shared memory access is not proved inside a shared variable: atom.shared.add.u32",
        ),
        (
            generic_count(1),
            "not proved inside its buffer, a shared variable, or a local variable: atom.add.u32",
        ),
        (local_load(8), "the local load is not proved inside a local variable: ld.local.u32"),
        (
            flag_access(4),
            "the shared memory access is not proved inside a shared variable: ld.shared.s32",
        ),
        (CONSTANT, "memory instruction the translation does not model"),
    ],
    ids=["shared", "generic in the shared window", "local", "scalar shared", "constant"],
)
def test_an_access_past_its_variable_is_refused(body: str, reason: str) -> None:
    decision = verify(zeroed(kernel(body), BINDING), [BINDING])
    assert isinstance(decision, Refused)
    assert reason in decision.reason
