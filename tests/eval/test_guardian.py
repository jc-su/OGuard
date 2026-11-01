"""Guardian's fence covers each global and generic access of a kernel.

A generic access (no state space) is fenced as a global one; a function that
converts a local, shared, constant, or parameter address to a generic one and
makes a generic access is refused, as is a `.func` that returns a value and
accesses global memory. On every patched corpus defect kernel, the fence
count equals the kernel's global and generic accesses.
"""

import gzip
import json
import re
from pathlib import Path

import pytest

from eval.baselines.guardian import UnsupportedPtxError, patch_ptx
from eval.corpus.adapter import directory_of
from eval.corpus.run import TOOLCHAINS

HEADER = """.version 9.0
.target sm_86
.address_size 64
"""

GENERIC = (
    HEADER
    + """
.visible .entry kernel(
	.param .u64 kernel_param_0
)
{
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;
	ld.param.u64 	%rd1, [kernel_param_0];
	ld.u32 	%r1, [%rd1+4];
	st.u32 	[%rd1], %r1;
	ret;
}
"""
)

ACCESS = re.compile(r"^\s*(?:@!?%\w+\s+)?((?:ld|st|atom|red)(?:\.[\w:]+)+)", re.M)
OTHER_SPACES = {"shared", "local", "const", "param"}


def test_a_generic_load_and_store_are_fenced() -> None:
    fenced, report = patch_ptx(GENERIC)
    assert report.instrumented_accesses == 2
    assert "add.u64 %grd_addr, %rd1, 4;" in fenced
    assert "and.b64 %grd_addr, %rd1, %grd_mask;" in fenced
    assert "ld.u32 	%r1, [%grd_addr];" in fenced
    assert "st.u32 	[%grd_addr], %r1;" in fenced
    assert ".param .u64 guardian_base_0," in fenced


def test_a_generic_access_beside_a_generic_local_address_is_refused() -> None:
    text = GENERIC.replace(
        "\tld.u32 \t%r1, [%rd1+4];", "\tcvta.local.u64 \t%rd1, %rd1;\n\tld.u32 \t%r1, [%rd1+4];"
    )
    with pytest.raises(UnsupportedPtxError, match="generic addresses of local memory"):
        patch_ptx(text)


def test_a_func_that_returns_a_value_and_accesses_global_memory_is_refused() -> None:
    func = """
.func  (.param .b32 func_retval0) add(
	.param .b64 add_param_0
)
{
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<2>;
	ld.param.u64 	%rd1, [add_param_0];
	atom.global.add.u32 	%r1, [%rd1], 1;
	st.param.b32 	[func_retval0+0], %r1;
	ret;
}
"""
    with pytest.raises(UnsupportedPtxError, match="add: .func bodies"):
        patch_ptx(GENERIC.replace(HEADER, HEADER + func))


def _defects() -> list[str]:
    raw = gzip.decompress(
        (Path(__file__).with_name("fixtures") / "e3-results.jsonl.gz").read_bytes()
    )
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    return sorted(f"{r['model']}/{r['task']}/{r['toolchain']}" for r in rows)


@pytest.mark.parametrize("key", _defects())
def test_each_global_and_generic_access_of_a_patched_defect_is_fenced(key: str) -> None:
    model, task, toolchain = key.split("/")
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    try:
        _, report = patch_ptx(text)
    except UnsupportedPtxError:
        return
    accesses = [
        opcode
        for opcode in ACCESS.findall(text)
        if not {token.split("::")[0] for token in opcode.split(".")[1:]} & OTHER_SPACES
    ]
    assert report.instrumented_accesses == len(accesses)
