"""The SASS prologue check on small texts that `zeroed` completes, loaded with the
driver's JIT as the service loads them (on an sm_86 GPU) and disassembled with
`nvdisasm`, and on listings that differ from a compiled one in one instruction."""

import importlib.util
import re
from pathlib import Path

import pytest

from eval.rq1_safety import sass_prologue as sp
from opgate.binding import Binding, Size
from opgate.gate import zeroed
from opgate.hosts.service.audit import binding_record
from opgate.signature import Affine, SizeSymbol


def _nvdisasm() -> Path:
    spec = importlib.util.find_spec("triton")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent / "backends" / "nvidia" / "bin" / "nvdisasm"


HEADER = ".version 8.8\n.target sm_86\n.address_size 64\n"

# A static shared array of 256 bytes and a local array of 2048 bytes, indexed by
# a loaded value so that the compiler keeps it in local memory.
STATIC = (
    HEADER
    + """
.visible .entry k(.param .u64 out)
{
	.local .align 16 .b8 	__local_depot0[2048];
	.shared .align 4 .b8 s[256];
	.reg .b32 %r<8>;
	.reg .b64 %rd<6>;
	mov.u32 %r1, %tid.x;
	ld.param.u64 %rd1, [out];
	cvta.to.global.u64 %rd2, %rd1;
	ld.global.u32 %r2, [%rd2];
	and.b32 %r3, %r2, 2044;
	mov.u32 %r4, __local_depot0;
	add.u32 %r4, %r4, %r3;
	st.local.u32 [%r4], %r1;
	ld.global.u32 %r5, [%rd2+4];
	and.b32 %r5, %r5, 2044;
	mov.u32 %r6, __local_depot0;
	add.u32 %r6, %r6, %r5;
	ld.local.u32 %r7, [%r6];
	bar.sync 0;
	shl.b32 %r3, %r1, 2;
	mov.u32 %r4, s;
	add.u32 %r4, %r4, %r3;
	ld.shared.u32 %r5, [%r4];
	add.u32 %r5, %r5, %r7;
	mul.wide.u32 %rd3, %r1, 4;
	add.s64 %rd4, %rd2, %rd3;
	st.global.u32 [%rd4], %r5;
	ret;
}
"""
)

# Two static arrays and an extern array whose bytes a constant gives.
PLACED = (
    HEADER
    + """
.extern .shared .align 16 .b8 dyn[];
.shared .align 4 .b8 sa[12];
.shared .align 8 .b8 sb[20];

.visible .entry k(.param .u64 out)
{
	.reg .b32 %r<8>;
	.reg .b64 %rd<4>;
	mov.u32 %r1, %tid.x;
	bar.sync 0;
	ld.shared.u32 %r5, [sa+4];
	ld.shared.u32 %r6, [sb+8];
	ld.shared.u32 %r7, [dyn+12];
	add.u32 %r5, %r5, %r6;
	add.u32 %r5, %r5, %r7;
	ld.param.u64 %rd1, [out];
	cvta.to.global.u64 %rd2, %rd1;
	st.global.u32 [%rd2], %r5;
	ret;
}
"""
)

# An extern array whose bytes the size parameter `n` gives, cleared in a loop.
LOOP = (
    HEADER
    + """
.extern .shared .align 4 .b8 dyn[];

.visible .entry k(.param .u64 out, .param .u32 n)
{
	.reg .b32 %r<8>;
	.reg .b64 %rd<4>;
	mov.u32 %r1, %tid.x;
	bar.sync 0;
	shl.b32 %r2, %r1, 2;
	mov.u32 %r3, dyn;
	add.u32 %r3, %r3, %r2;
	ld.shared.u32 %r5, [%r3];
	ld.param.u64 %rd1, [out];
	cvta.to.global.u64 %rd2, %rd1;
	st.global.u32 [%rd2], %r5;
	ret;
}
"""
)


# A static shared array, and a device function with a local array that the
# entry kernel calls when `CALLS` is in the text, or never.
FUNCTION = (
    HEADER
    + """
.func (.param .b32 r) f(.param .b32 i)
{
	.local .align 4 .b8 	__local_depot1[64];
	.reg .b32 %r<4>;
	ld.param.u32 %r1, [i];
	and.b32 %r1, %r1, 60;
	mov.u32 %r2, __local_depot1;
	add.u32 %r2, %r2, %r1;
	st.local.u32 [%r2], %r1;
	ld.local.u32 %r3, [%r2+4];
	st.param.b32 [r], %r3;
	ret;
}

.visible .entry k(.param .u64 out)
{
	.shared .align 4 .b8 s[128];
	.reg .b32 %r<6>;
	.reg .b64 %rd<4>;
	mov.u32 %r1, %tid.x;
	bar.sync 0;
	shl.b32 %r2, %r1, 2;
	mov.u32 %r3, s;
	add.u32 %r3, %r3, %r2;
	ld.shared.u32 %r4, [%r3];
	CALLS
	ld.param.u64 %rd1, [out];
	cvta.to.global.u64 %rd2, %rd1;
	st.global.u32 [%rd2], %r4;
	ret;
}
"""
)
CALL = """{
	.param .b32 a;
	st.param.b32 [a], %r4;
	.param .b32 b;
	call.uni (b), f, (a);
	ld.param.b32 %r4, [b];
	}"""


def _binding(block: int, shared: Affine, parameters: tuple[tuple[str, Size], ...] = ()) -> Binding:
    sizes = (SizeSymbol("n", 0, 300),) if parameters else ()
    return Binding("k", sizes, parameters, (block, 1, 1), (1, 1, 1), shared)


def _launch(binding: Binding) -> sp.Launch:
    ranges = sp._ranges(binding_record(binding))
    return sp.Launch(binding.kernel, binding.block, {k: (v[0], v[1]) for k, v in ranges.items()})


Static = tuple[str, sp.Launch, str, sp.Attributes]


def _compiled(text: str) -> sp.Compiled:
    return sp.compile_sass(text, "k", _nvdisasm())


def _check(text: str, launch: sp.Launch, listing: str, attributes: sp.Attributes) -> sp.Result:
    return sp.check(text, launch, listing, attributes["shared_bytes"], attributes["local_bytes"])


@pytest.fixture(scope="module")
def static() -> Static:
    binding = _binding(64, Affine(0))
    text = zeroed(STATIC, binding)
    compiled = _compiled(text)
    return text, _launch(binding), compiled.listing, compiled.jit.attributes


@pytest.mark.gpu
def test_the_static_prologue_survives_the_jit(static: Static) -> None:
    text, launch, listing, attributes = static
    result = _check(text, launch, listing, attributes)
    assert result["failures"] == []
    assert result["shared"][0]["covered"] == [[0, 256]]
    local = result["local"]
    assert local is not None and local["covered"] == [[0, 2048]] and local["reads"] >= 1
    assert attributes["shared_bytes"] == 256 and attributes["local_bytes"] == 2048


@pytest.mark.gpu
def test_the_jit_cache_entry_holds_the_text_and_ends_in_the_cubin() -> None:
    binding = _binding(64, Affine(0))
    text = zeroed(STATIC, binding)
    jit = _compiled(text).jit
    assert jit.cubin.startswith(b"\x7fELF")
    assert sp.JIT_LIBRARY in Path(jit.compiler).name
    assert b"sm_86" in jit.header


def _without(listing: str, pattern: str) -> str:
    lines = listing.splitlines(keepends=True)
    matching = [i for i, line in enumerate(lines) if re.search(pattern, line)]
    assert matching
    del lines[matching[0]]
    return "".join(lines)


def _before(listing: str, pattern: str, inserted: str) -> str:
    """`listing` with `inserted` before the first line that matches `pattern`."""
    lines = listing.splitlines(keepends=True)
    at = next(i for i, line in enumerate(lines) if re.search(pattern, line))
    return "".join([*lines[:at], f"        /*0000*/                   {inserted} ;\n", *lines[at:]])


@pytest.mark.gpu
def test_a_missing_shared_store_fails(static: Static) -> None:
    text, launch, listing, attributes = static
    result = _check(text, launch, _without(listing, r"STS .*, RZ"), attributes)
    assert any("not a placement" in f for f in result["failures"])


@pytest.mark.gpu
def test_a_barrier_before_the_stores_fails(static: Static) -> None:
    text, launch, listing, attributes = static
    moved = _before(listing, r"STS .*, RZ", "BAR.SYNC.DEFER_BLOCKING 0x0")
    result = _check(text, launch, moved, attributes)
    assert any(f.startswith("shared:") for f in result["failures"])


@pytest.mark.gpu
def test_a_shared_load_before_the_barrier_fails(static: Static) -> None:
    text, launch, listing, attributes = static
    result = _check(text, launch, _before(listing, r"BAR\.SYNC", "LDS R9, [RZ]"), attributes)
    assert any("a load or atomic before the barrier" in f for f in result["failures"])


@pytest.mark.gpu
def test_a_local_load_before_the_local_stores_fails(static: Static) -> None:
    text, launch, listing, attributes = static
    result = _check(text, launch, _before(listing, r"STL.*, RZ", "LDL R9, [R1]"), attributes)
    assert any("runs before the stores of zero" in f for f in result["failures"])


@pytest.mark.gpu
def test_static_variables_and_dynamic_memory_have_a_placement() -> None:
    binding = _binding(32, Affine(32))
    text = zeroed(PLACED, binding)
    compiled = _compiled(text)
    result = _check(text, _launch(binding), compiled.listing, compiled.jit.attributes)
    assert result["failures"] == []
    (shared,) = result["shared"]
    assert shared["dynamic"] == [48, 80]
    assert sorted(b for _, b in shared["layout"] or []) == [12, 20]


@pytest.mark.gpu
def test_a_loop_over_a_size_parameter_is_checked_at_each_sampled_size() -> None:
    binding = _binding(64, Affine(0, ((0, 4),)), (("n", Size(Affine.symbol(0))),))
    text = zeroed(LOOP, binding)
    assert "$opgate_zero_0" in text
    compiled = _compiled(text)
    result = _check(text, _launch(binding), compiled.listing, compiled.jit.attributes)
    assert result["failures"] == []
    sizes = [c["values"]["n"] for c in result["shared"]]
    assert 0 in sizes and 300 in sizes and 64 in sizes and 65 in sizes
    for check in result["shared"]:
        assert check["covered"] == ([[0, 4 * check["values"]["n"]]] if check["values"]["n"] else [])


def test_a_layout_tiles_runs_with_aligned_variables() -> None:
    variables = [sp.Variable("a", 4, 12), sp.Variable("b", 8, 20)]
    assert sp.layout(frozenset([*range(12), *range(16, 36)]), variables, 48) == [[0, 12], [16, 20]]
    assert sp.layout(frozenset([*range(12), *range(14, 34)]), variables, 48) is None
    assert sp.layout(frozenset(range(36)), variables, 48) is None


@pytest.mark.gpu
def test_a_local_load_outside_the_variables_needs_no_zero_store(static: Static) -> None:
    text, launch, listing, attributes = static
    moved = _before(listing, r"STL.*, RZ", "LDL R9, [R1+0x800]")
    assert _check(text, launch, moved, attributes)["failures"] == []


@pytest.mark.gpu
@pytest.mark.parametrize("calls", [False, True])
def test_a_device_function_prologue_needs_an_entry_kernel_without_calls(calls: bool) -> None:
    binding = _binding(32, Affine(0))
    text = zeroed(FUNCTION.replace("CALLS", CALL if calls else ""), binding)
    assert sp.prologue_of(text, "k", 32).functions == 1
    compiled = _compiled(text)
    failures = _check(text, _launch(binding), compiled.listing, compiled.jit.attributes)["failures"]
    if calls:
        assert len(failures) == 1 and "device functions have a zeroing prologue" in failures[0]
    else:
        assert failures == []


# The SASS that the driver's JIT gives for the dynamic shared memory loop of
# GPT-5.5's CUDA C++ `q22_final_group_kernel`, with the guard pass's checks, up
# to its barrier: the count, in words, is the parameter at `c[0x0][0x190]`,
# which `ULDC` loads, and a block of 1024 threads clears it four stores at a
# time, then two, then one.
ULDC_LOOP = """.text.kernel:
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.Z ;
        /*0020*/                   ULDC UR4, c[0x0][0x190] ;
        /*0030*/                   BSSY B0, `(.L_x_0) ;
        /*0040*/                   USHF.L.U32 UR5, UR4, 0x2, URZ ;
        /*0050*/                   S2R R3, SR_TID.Y ;
        /*0060*/                   S2R R5, SR_TID.X ;
        /*0070*/                   IMAD.IADD R0, R0, 0x1, R3 ;
        /*0080*/                   IMAD R0, R0, 0x400, R5 ;
        /*0090*/                   ISETP.GE.U32.AND P0, PT, R0, UR5, PT ;
        /*00a0*/               @P0 BRA `(.L_x_1) ;
        /*00b0*/                   ISETP.GE.U32.AND P1, PT, R0, UR5, PT ;
        /*00c0*/                   BSSY B1, `(.L_x_2) ;
        /*00d0*/               @P1 STS [R0.X4], RZ ;
        /*00e0*/               @P1 IADD3 R0, R0, 0x400, RZ ;
        /*00f0*/                   PLOP3.LUT P1, PT, P1, PT, PT, 0x8, 0x0 ;
        /*0100*/                   ISETP.LT.U32.AND P0, PT, R0.reuse, UR5, PT ;
        /*0110*/                   IADD3 R2, -R0, UR5, RZ ;
        /*0120*/                   ISETP.LE.U32.OR P0, PT, R2, 0xc00, !P0 ;
        /*0130*/               @P0 BRA `(.L_x_3) ;
        /*0140*/                   IMAD.U32 R3, RZ, RZ, UR5 ;
        /*0150*/                   PLOP3.LUT P1, PT, PT, PT, PT, 0x8, 0x0 ;
        /*0160*/                   IADD3 R3, R3, -0xc00, RZ ;
.L_x_4:
        /*0170*/                   STS [R0.X4], RZ ;
        /*0180*/                   STS [R0.X4+0x1000], RZ ;
        /*0190*/                   STS [R0.X4+0x2000], RZ ;
        /*01a0*/                   STS [R0.X4+0x3000], RZ ;
        /*01b0*/                   IADD3 R0, R0, 0x1000, RZ ;
        /*01c0*/                   ISETP.GE.U32.AND P0, PT, R0, R3, PT ;
        /*01d0*/              @!P0 BRA `(.L_x_4) ;
.L_x_3:
        /*01e0*/                   BSYNC B1 ;
.L_x_2:
        /*01f0*/                   ISETP.LT.U32.AND P0, PT, R0.reuse, UR5, PT ;
        /*0200*/                   IADD3 R2, -R0, UR5, RZ ;
        /*0210*/                   ISETP.LE.U32.OR P0, PT, R2, 0x400, !P0 ;
        /*0220*/              @!P0 PLOP3.LUT P1, PT, PT, PT, PT, 0x8, 0x0 ;
        /*0230*/              @!P0 STS [R0.X4], RZ ;
        /*0240*/              @!P0 STS [R0.X4+0x1000], RZ ;
        /*0250*/              @!P0 IADD3 R0, R0, 0x800, RZ ;
        /*0260*/                   ISETP.GE.U32.AND P1, PT, R0, UR5, !P1 ;
        /*0270*/              @!P1 STS [R0.X4], RZ ;
.L_x_1:
        /*0280*/                   BSYNC B0 ;
.L_x_0:
        /*0290*/                   S2R R7, SR_CTAID.X ;
        /*02a0*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;
//--------------------- .nv.shared.kernel --------------------------
"""


@pytest.mark.parametrize("count", [0, 1, 300, 1000, 1024, 3000])
@pytest.mark.parametrize("thread", [0, 5, 1023])
def test_a_loop_over_a_count_that_uldc_loads_has_one_path(thread: int, count: int) -> None:
    listing = sp.parse_listing(ULDC_LOOP, "kernel")
    stored = sp._thread(listing, sp._Machine((thread, 0, 0), {(0, 0x190): count}))
    words = range(thread, 4 * count, 1024)
    assert stored == frozenset(b for w in words for b in range(4 * w, 4 * w + 4))
