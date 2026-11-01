"""The zeroing prologue of shared and local memory, and registers written before
they are read (design decision 46); CPU only.

The gate refuses a kernel that does not begin with exactly the prologue that
`opgate.gate.zeroed` writes, and a kernel in which an instruction may read a
register before an instruction writes it. The negative control is DeepSeek's CUDA C++
Q6 kernel without the store that initializes its shared array: as written its
threads would sum what an earlier kernel left in shared memory.
"""

from dataclasses import replace

import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.binding import Base, Binding, Size, Span
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.signature import Affine, Direction, SizeSymbol

MISSING = "the kernel does not begin with the zeroing prologue of its shared and local memory"
UNWRITTEN = "an instruction may read the register "

PROBE = ("deepseek-v4-flash", "q06_core_cuda_q6_kernel", "nvcc")
INITIALIZING = "\tst.shared.f64 \t[%r3], %fd16;\n"

# Each thread stores its index in a static shared array, waits, reads it
# back through a local variable, and writes it to the output.
KERNEL = """.version 8.0
.target sm_80
.address_size 64
{module}
.visible .entry kernel(.param .u64 out)
.reqntid 128
{{
.reg .pred %p<4>;
.reg .b32 %r<16>;
.reg .b64 %rd<4>;
.shared .align 4 .b8 buf[{shared}];
.local .align 8 .b8 slot[8];
ld.param.u64 %rd1, [out];
mov.u32 %r1, %tid.x;
shl.b32 %r2, %r1, 2;
mov.u32 %r3, buf;
add.s32 %r4, %r3, %r2;
st.shared.u32 [%r4], %r1;
bar.sync 0;
ld.shared.u32 %r5, [%r4];
st.local.u32 [slot+0], %r5;
ld.local.u32 %r6, [slot+0];
{body}
mul.wide.u32 %rd2, %r1, 4;
add.s64 %rd3, %rd1, %rd2;
st.global.u32 [%rd3], %r6;
ret;
}}
"""

BINDING = Binding(
    "kernel",
    (SizeSymbol("n", 0, 2147483647),),
    (("out", Base(Span(Direction.WRITE, Affine(512), 4, None, False))),),
    (128, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def kernel(body: str = "", module: str = "", shared: int = 512) -> str:
    return KERNEL.format(body=body, module=module, shared=shared)


def refused(text: str, binding: Binding = BINDING) -> str:
    decision = verify(text, [binding])
    assert isinstance(decision, Refused), text
    return decision.reason


def probe() -> tuple[str, str, Binding]:
    text = (directory_of(*PROBE[:2]) / TOOLCHAINS[PROBE[2]]).read_text()
    without = text.replace(INITIALIZING, "")
    assert without != text
    return text, without, binding_of(*PROBE, text)


@pytest.mark.slow
def test_the_probe_is_refused_without_the_prologue_and_accepted_with_it() -> None:
    text, without, binding = probe()
    for written in (text, without):
        assert MISSING in refused(written, binding)
        assert isinstance(verify(zeroed(written, binding), [binding]), Accepted)


def test_the_prologue_clears_the_shared_and_local_variables() -> None:
    text = zeroed(kernel(), BINDING)
    assert "st.shared.u32 [%opgate_z1+0], %opgate_z2;" in text
    assert "st.shared.v4.u32" not in text  # alignment 4 allows 4 bytes per store
    assert text.count("st.local.v2.u32 [slot+0], {%opgate_z2, %opgate_z2};") == 1
    assert text.index("bar.sync 0;") < text.index("st.local.v2.u32")
    assert isinstance(verify(text, [BINDING]), Accepted)
    assert MISSING in refused(kernel())


def test_a_kernel_without_shared_or_local_memory_needs_no_prologue() -> None:
    text = """.version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 out)
.reqntid 128
{
.reg .b32 %r<4>;
.reg .b64 %rd<4>;
ld.param.u64 %rd1, [out];
mov.u32 %r1, %tid.x;
mul.wide.u32 %rd2, %r1, 4;
add.s64 %rd3, %rd1, %rd2;
st.global.u32 [%rd3], %r1;
ret;
}
"""
    assert zeroed(text, BINDING) == text
    assert isinstance(verify(text, [BINDING]), Accepted)


def test_comments_and_quoted_strings_do_not_place_the_prologue() -> None:
    before = '// .entry kernel( {\n.file 1 ".entry kernel( {"\n'
    text = kernel().replace(".visible", before + ".visible")
    assert zeroed(text, BINDING) == zeroed(kernel(), BINDING).replace(
        ".visible", before + ".visible"
    )


def test_a_last_round_stores_only_below_the_elements_left() -> None:
    # 600 bytes are 150 words: one full round of 128 threads and 22 left
    text = zeroed(kernel(shared=600), BINDING)
    assert "setp.lt.u32 %opgate_p, %opgate_z0, 22;" in text
    assert "@%opgate_p st.shared.u32 [%opgate_z1+512], %opgate_z2;" in text
    assert isinstance(verify(text, [BINDING]), Accepted)


@pytest.mark.parametrize(
    "line",
    [
        "st.shared.u32 [%opgate_z1+0], %opgate_z2;",
        "bar.sync 0;",
        "st.local.v2.u32 [slot+0], {%opgate_z2, %opgate_z2};",
        "mad.lo.u32 %opgate_z0, %opgate_z0, 128, %opgate_z1;",
    ],
)
def test_a_prologue_without_one_of_its_instructions_is_refused(line: str) -> None:
    text = zeroed(kernel(), BINDING)
    assert line in text
    assert MISSING in refused(text.replace(line, "", 1))


def test_a_prologue_of_another_block_size_is_refused() -> None:
    wider = replace(
        BINDING,
        parameters=(("out", Base(Span(Direction.WRITE, Affine(1024), 4, None, False))),),
        block=(256, 1, 1),
    )
    text = zeroed(kernel(shared=1024).replace(".reqntid 128", ".reqntid 256"), wider)
    assert isinstance(verify(text, [wider]), Accepted)
    assert MISSING in refused(text.replace(".reqntid 256", ".reqntid 128"), BINDING)


def test_a_label_inside_the_prologue_is_refused() -> None:
    text = zeroed(kernel(), BINDING)
    entered = text.replace("\tbar.sync 0;", "INSIDE:\n\tbar.sync 0;", 1)
    assert "a label inside the zeroing prologue" in refused(entered)


def test_a_second_declaration_of_the_prologue_registers_is_refused() -> None:
    text = zeroed(kernel(), BINDING)
    twice = text.replace(".reg .b64 %rd<4>;", ".reg .b64 %rd<4>;\n.reg .b32 %opgate_z<3>;", 1)
    assert "registers are not declared once" in refused(twice)


def test_a_device_function_clears_its_local_variables_when_a_call_enters_it() -> None:
    function = """.func (.param .b32 ret) helper(.param .b32 x)
{
.reg .b32 %h<4>;
.local .align 4 .b8 tmp[4];
ld.param.b32 %h1, [x];
st.local.u32 [tmp+0], %h1;
ld.local.u32 %h2, [tmp+0];
st.param.b32 [ret+0], %h2;
ret;
}"""
    call = """{
.param .b32 a;
.param .b32 b;
st.param.b32 [a+0], %r6;
call.uni (b), helper, (a);
ld.param.b32 %r7, [b+0];
}"""
    text = zeroed(kernel(body=call, module=function), BINDING)
    stores = "\tmov.u32 %opgate_z2, 0;\n\tst.local.u32 [tmp+0], %opgate_z2;"
    assert text.count(stores) == 1 and text.index(stores) < text.index(".entry")
    assert isinstance(verify(text, [BINDING]), Accepted)
    reason = refused(text.replace(stores, ""))
    assert "the device function helper does not begin with the zeroing prologue" in reason


@pytest.mark.parametrize(("offset", "accepted"), [(0, True), (1, False)])
def test_a_callee_reads_the_low_byte_of_a_parameter_that_its_caller_passes_in_32_bits(
    offset: int, accepted: bool
) -> None:
    # cuDF's UDFs pass a byte in 32 bits and load it as .u8; the store sets the
    # cells of its low bytes at its offset, and a byte at another offset stays
    # unwritten
    function = f""".func (.param .b32 ret) low(.param .b32 x)
{{
.reg .b16 %hs<2>;
.reg .b32 %h<2>;
ld.param.u8 %hs1, [x+{offset}];
cvt.u32.u16 %h1, %hs1;
st.param.b32 [ret+0], %h1;
ret;
}}"""
    call = """{
.param .b32 a;
.param .b32 b;
st.param.b32 [a+0], %r6;
call.uni (b), low, (a);
ld.param.b32 %r6, [b+0];
}"""
    text = zeroed(kernel(body=call, module=function), BINDING)
    if accepted:
        assert isinstance(verify(text, [BINDING]), Accepted)
    else:
        assert UNWRITTEN in refused(text)


def test_dynamic_shared_memory_of_a_size_that_no_parameter_holds_is_refused() -> None:
    dynamic = replace(BINDING, sizes=(SizeSymbol("n", 0, 1024),), shared=Affine(0, ((0, 4),)))
    text = kernel(module=".extern .shared .align 16 .b8 dynamic[];")
    assert "needs a parameter that holds each size of the dynamic shared memory" in refused(
        zeroed(text, dynamic), dynamic
    )


# Each thread reads its word of dynamic shared memory, of `bytes` bytes for a
# launch of n elements, after the prologue, and writes it to the output: as
# written it would read what an earlier kernel left there.
DYNAMIC = """.version 8.0
.target sm_80
.address_size 64
.extern .shared .align 4 .b8 dyn[];
.visible .entry kernel(.param .u64 out, .param .u32 n, .param .u32 m)
.reqntid 128
{
.reg .pred %p<4>;
.reg .b32 %r<16>;
.reg .b64 %rd<4>;
ld.param.u64 %rd1, [out];
mov.u32 %r1, %tid.x;
shl.b32 %r2, %r1, 2;
mov.u32 %r3, dyn;
add.s32 %r4, %r3, %r2;
ld.param.u32 %r7, [n];
setp.ge.u32 %p1, %r1, %r7;
@%p1 bra $done;
ld.shared.u32 %r5, [%r4];
mul.wide.u32 %rd2, %r1, 4;
add.s64 %rd3, %rd1, %rd2;
st.global.u32 [%rd3], %r5;
$done:
ret;
}
"""


def dynamic_binding(shared: Affine) -> Binding:
    """Sizes n and m of at most 128 elements, held by the parameters of the same names."""
    return Binding(
        "kernel",
        (SizeSymbol("n", 0, 128), SizeSymbol("m", 0, 128)),
        (
            ("out", Base(Span(Direction.WRITE, Affine(512), 4, None, False))),
            ("n", Size(Affine.symbol(0))),
            ("m", Size(Affine.symbol(1))),
        ),
        (128, 1, 1),
        (1, 1, 1),
        shared,
    )


TWO_WORDS = dynamic_binding(Affine(0, ((0, 8),)))  # 8 bytes, two words, for each element


def test_a_loop_clears_dynamic_shared_memory_whose_bytes_depend_on_the_sizes() -> None:
    text = zeroed(DYNAMIC, TWO_WORDS)
    for line in (
        ".reg .b32 \t%opgate_d<2>;",
        "mov.u32 %opgate_d0, 0;",
        "ld.param.u32 %opgate_d1, [n];",
        "mad.lo.u32 %opgate_d0, %opgate_d1, 2, %opgate_d0;",
        "mov.u32 %opgate_d1, %opgate_z0;",
        "$opgate_zero_0:",
        "setp.ge.u32 %opgate_p, %opgate_d1, %opgate_d0;",
        "@%opgate_p bra $opgate_zeroed_0;",
        "mov.u32 %opgate_z1, dyn;",
        "mad.lo.u32 %opgate_z1, %opgate_d1, 4, %opgate_z1;",
        "st.shared.u32 [%opgate_z1], %opgate_z2;",
        "add.u32 %opgate_d1, %opgate_d1, 128;",
        "bra $opgate_zero_0;",
        "$opgate_zeroed_0:",
    ):
        assert line in text, line
    assert text.index("$opgate_zeroed_0:") < text.index("bar.sync 0;")
    assert isinstance(verify(text, [TWO_WORDS]), Accepted)
    assert MISSING in refused(DYNAMIC, TWO_WORDS)


def test_the_loop_comes_before_the_rounds_of_the_other_shared_variables() -> None:
    module = ".shared .align 4 .b8 fixed[64];"
    text = zeroed(DYNAMIC.replace(".extern", f"{module}\n.extern"), TWO_WORDS)
    assert text.index("$opgate_zeroed_0:") < text.index("mov.u32 %opgate_z1, fixed;")
    assert text.index("mov.u32 %opgate_z1, fixed;") < text.index("bar.sync 0;")
    assert isinstance(verify(text, [TWO_WORDS]), Accepted)


@pytest.mark.parametrize(
    ("written", "changed"),
    [
        # a count of one word for each element, where the launch gives two
        (
            "mad.lo.u32 %opgate_d0, %opgate_d1, 2, %opgate_d0;",
            "mad.lo.u32 %opgate_d0, %opgate_d1, 1, %opgate_d0;",
        ),
        # the count of another size, which may be smaller
        ("ld.param.u32 %opgate_d1, [n];", "ld.param.u32 %opgate_d1, [m];"),
        # a stride that skips every other block's elements
        ("add.u32 %opgate_d1, %opgate_d1, 128;", "add.u32 %opgate_d1, %opgate_d1, 256;"),
        # a first element past the thread's index
        ("mov.u32 %opgate_d1, %opgate_z0;", "add.u32 %opgate_d1, %opgate_z0, 1;"),
        # an exit before the last element
        (
            "setp.ge.u32 %opgate_p, %opgate_d1, %opgate_d0;",
            "setp.gt.u32 %opgate_p, %opgate_d0, %opgate_d1;",
        ),
    ],
)
def test_a_prologue_that_clears_fewer_dynamic_bytes_than_the_launch_gives_is_refused(
    written: str, changed: str
) -> None:
    text = zeroed(DYNAMIC, TWO_WORDS)
    assert written in text
    assert MISSING in refused(text.replace(written, changed, 1), TWO_WORDS)


def test_a_constant_part_of_the_dynamic_bytes_is_counted() -> None:
    # 64 bytes plus 4 for each element: the count starts at 16 words
    binding = dynamic_binding(Affine(64, ((0, 4),)))
    text = zeroed(DYNAMIC, binding)
    assert "mov.u32 %opgate_d0, 16;" in text
    assert isinstance(verify(text, [binding]), Accepted)
    assert MISSING in refused(
        text.replace("mov.u32 %opgate_d0, 16;", "mov.u32 %opgate_d0, 0;"), binding
    )


def test_a_branch_into_the_loop_is_refused() -> None:
    text = zeroed(DYNAMIC, TWO_WORDS).replace("$done:\nret;", "$done:\nbra $opgate_zeroed_0;\nret;")
    assert "a branch into the zeroing prologue" in refused(text, TWO_WORDS)


def test_the_loop_registers_must_be_declared_once() -> None:
    text = zeroed(DYNAMIC, TWO_WORDS).replace(".reg .b32 \t%opgate_d<2>;\n", "", 1)
    assert "registers are not declared once" in refused(text, TWO_WORDS)


def test_dynamic_bytes_with_a_negative_coefficient_are_refused() -> None:
    binding = dynamic_binding(Affine(512, ((0, -4),)))
    reason = refused(zeroed(DYNAMIC, binding), binding)
    assert "constant and coefficients that are not negative" in reason


@pytest.mark.parametrize(
    ("body", "register"),
    [
        # read before any write
        ("add.s32 %r6, %r6, %r9;", "%r9"),
        # written only under a guard
        ("setp.eq.s32 %p1, %r1, 0;\n@%p1 mov.u32 %r9, 1;\nadd.s32 %r6, %r6, %r9;", "%r9"),
        # written on one path only
        (
            "setp.eq.s32 %p1, %r1, 0;\n@%p1 bra SKIP;\nmov.u32 %r9, 1;\nSKIP:\n"
            "add.s32 %r6, %r6, %r9;",
            "%r9",
        ),
        # the carry flag before an instruction sets it
        ("addc.u32 %r6, %r6, 0;", "carry flag"),
        # a guard predicate that nothing sets
        ("@%p3 add.s32 %r6, %r6, 1;", "%p3"),
    ],
)
def test_a_register_read_before_it_is_written_is_refused(body: str, register: str) -> None:
    reason = refused(zeroed(kernel(body=body), BINDING))
    assert f"{UNWRITTEN}{register} before an instruction writes it" in reason


@pytest.mark.parametrize(
    "body",
    [
        "setp.eq.s32 %p1, %r1, 0;\n@%p1 bra OTHER;\nmov.u32 %r9, 1;\nbra JOIN;\n"
        "OTHER:\nmov.u32 %r9, 2;\nJOIN:\nadd.s32 %r6, %r6, %r9;",
        "add.cc.u32 %r9, %r6, 1;\naddc.u32 %r6, %r6, 0;",
    ],
)
def test_a_register_written_on_every_path_is_accepted(body: str) -> None:
    assert isinstance(verify(zeroed(kernel(body=body), BINDING), [BINDING]), Accepted)


def test_dynamic_shared_memory_of_constant_bytes_is_cleared_in_stores_of_one_width() -> None:
    # 256 bytes aligned to 16 next to `buf`, aligned to 4: every shared store
    # writes 4 bytes, so the stores are of the number 0 and of one width, which
    # the shared value analysis takes as ordered writes
    dynamic = replace(BINDING, shared=Affine(256))
    text = zeroed(kernel(module=".extern .shared .align 16 .b8 dynamic[];"), dynamic)
    assert "mov.u32 %opgate_z1, dynamic;" in text
    assert "@%opgate_p st.shared.u32 [%opgate_z1+0], %opgate_z2;" in text
    assert "st.shared.v" not in text
    assert isinstance(verify(text, [dynamic]), Accepted)
