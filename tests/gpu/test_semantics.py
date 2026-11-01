"""The translation's semantics of each integer instruction is the GPU's.

The gate trusts the translation of `GateFront.v` to state what each PTX
instruction does. For each instruction the translation models exactly, a
probe kernel applies it, or a short sequence of instructions, to two 64-bit
scalar parameters and their low 32 bits, and stores the 64-bit result. The
GPU runs the probe on every pair of edge values and on random pairs; Rocq
translates the same probe and runs its one block with the IR's `result`,
from the same parameters, up to the store. The stored values agree.
"""

import ctypes
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eval.corpus.diagnose import COMMON
from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, Direction, ElementType, derive_binding
from opgate.executor.driver import address, call, driver
from opgate.gate import export_terms, formal_dir
from opgate.signature import Affine

# Each probe computes %rd4 from %rd1 and %rd2 (the parameters), %r1 and %r2
# (their low 32 bits), and %r3 (a 32-bit result, which the probe zero-extends).
PROBES: dict[str, str] = {
    "add.s32": "add.s32 %r3, %r1, %r2;",
    "sub.s32": "sub.s32 %r3, %r1, %r2;",
    "mul.lo.s32": "mul.lo.s32 %r3, %r1, %r2;",
    "mad.lo.s32": "mad.lo.s32 %r3, %r1, %r2, %r1;",
    "shl.b32": "shl.b32 %r3, %r1, %r2;",
    "shr.s32": "shr.s32 %r3, %r1, %r2;",
    "shr.u32": "shr.u32 %r3, %r1, %r2;",
    "and.b32": "and.b32 %r3, %r1, %r2;",
    "or.b32": "or.b32 %r3, %r1, %r2;",
    "min.s32": "min.s32 %r3, %r1, %r2;",
    "max.u32": "max.u32 %r3, %r1, %r2;",
    "neg.s32": "neg.s32 %r3, %r1;",
    "not.b32": "not.b32 %r3, %r1;",
    "bfe.u32": "bfe.u32 %r3, %r1, 5, 7;",
    "div.s32": "div.s32 %r3, %r1, 7;",
    "rem.s32": "rem.s32 %r3, %r1, 7;",
    "div.u32": "div.u32 %r3, %r1, 7;",
    "add.s64": "add.s64 %rd4, %rd1, %rd2;",
    "sub.s64": "sub.s64 %rd4, %rd1, %rd2;",
    "mul.lo.s64": "mul.lo.s64 %rd4, %rd1, %rd2;",
    "mad.lo.s64": "mad.lo.s64 %rd4, %rd1, %rd2, %rd1;",
    "shl.b64": "shl.b64 %rd4, %rd1, %r2;",
    "shr.s64": "shr.s64 %rd4, %rd1, %r2;",
    "shr.u64": "shr.u64 %rd4, %rd1, %r2;",
    "and.b64": "and.b64 %rd4, %rd1, %rd2;",
    "or.b64": "or.b64 %rd4, %rd1, %rd2;",
    "min.s64": "min.s64 %rd4, %rd1, %rd2;",
    "max.u64": "max.u64 %rd4, %rd1, %rd2;",
    "mul.wide.s32": "mul.wide.s32 %rd4, %r1, %r2;",
    "mul.wide.u32": "mul.wide.u32 %rd4, %r1, %r2;",
    "cvt.s64.s32": "cvt.s64.s32 %rd4, %r1;",
    "cvt.u64.u32": "cvt.u64.u32 %rd4, %r1;",
    "div.s64": "div.s64 %rd4, %rd1, 7;",
    "rem.u64": "rem.u64 %rd4, %rd1, 7;",
    "setp.lt.s32": "setp.lt.s32 %p1, %r1, %r2; selp.u32 %r3, 7, 9, %p1;",
    "setp.ge.u32": "setp.ge.u32 %p1, %r1, %r2; selp.u32 %r3, 7, 9, %p1;",
    "setp.le.s64": "setp.le.s64 %p1, %rd1, %rd2; selp.b64 %rd4, %rd1, %rd2, %p1;",
    "setp.gt.u64": "setp.gt.u64 %p1, %rd1, %rd2; selp.b64 %rd4, %rd1, %rd2, %p1;",
    "setp.eq.s32": "setp.eq.s32 %p1, %r1, %r2; selp.u32 %r3, 7, 9, %p1;",
    "setp.ne.s64": "setp.ne.s64 %p1, %rd1, %rd2; selp.b64 %rd4, 7, 9, %p1;",
    "and.pred": (
        "setp.lt.s32 %p1, %r1, 0; setp.ge.s64 %p2, %rd1, %rd2; and.pred %p3, %p1, %p2;"
        " selp.b64 %rd4, 1, 0, %p3;"
    ),
    "or.pred": (
        "setp.lt.s32 %p1, %r1, 0; setp.ge.s64 %p2, %rd1, %rd2; or.pred %p3, %p1, %p2;"
        " selp.b64 %rd4, 1, 0, %p3;"
    ),
    "xor.pred": (
        "setp.lt.s32 %p1, %r1, 0; setp.ge.s64 %p2, %rd1, %rd2; xor.pred %p3, %p1, %p2;"
        " selp.b64 %rd4, 1, 0, %p3;"
    ),
    "xor.b32": "xor.b32 %r3, %r1, %r2;",
    "xor.b64": "xor.b64 %rd4, %rd1, %rd2;",
    "abs.s32": "abs.s32 %r3, %r1;",
    "abs.s64": "abs.s64 %rd4, %rd1;",
}

EDGES = (
    0,
    1,
    2,
    7,
    31,
    32,
    33,
    63,
    64,
    2**31 - 1,
    2**31,
    2**32 - 1,
    2**32,
    2**63 - 1,
    2**63,
    2**64 - 7,
)
RANDOM = 40

SIGNATURE = sig.Signature(
    sizes=(),
    arguments=(
        sig.Scalar("a", sig.Between(0, 2**64 - 1)),
        sig.Scalar("b", sig.Between(0, 2**64 - 1)),
        sig.Span("out", "out", Direction.WRITE, Affine(1)),
    ),
)
BINDING = derive_binding(
    SIGNATURE,
    {"out": ElementType(8, False)},
    Convention.C,
    "probe",
    ("probe_a", "probe_b", "probe_out"),
    (1, 1, 1),
    GRID_LIMIT,
    Affine(0),
)

QUERY = """
Definition probe (a b : Z) : option Z :=
  match kernel_of with
  | Some (k, mf) =>
      match front k mf with
      | Ok (prog, F) =>
          let init := fun r => if String.eqb (name_of r) "param:probe_a" then a
                               else if String.eqb (name_of r) "param:probe_b" then b else 0 in
          let fix go (is : list instr) (rs : reg -> Z) : option Z :=
            match is with
            | [] => None
            | IMem _ KStore _ _ _ _ (Some v) _ _ :: _ => Some (opv rs v)
            | i :: rest =>
                go rest (match dest i with
                         | Some d => upd (wdn F) rs d
                                       (result (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => 0) 0 rs i)
                         | None => rs
                         end)
            end in
          go (List.concat prog) init
      | Err _ _ => None
      end
  | None => None
  end.
Eval vm_compute in map (fun p => probe (fst p) (snd p)) VECTORS.
"""


def _text(body: str) -> str:
    widen = "" if "%rd4" in body else "cvt.u64.u32 %rd4, %r3;"
    return f""".version 8.8
.target sm_86
.address_size 64

.visible .entry probe(
\t.param .u64 probe_a,
\t.param .u64 probe_b,
\t.param .u64 probe_out
)
{{
\t.reg .pred \t%p<4>;
\t.reg .b32 \t%r<4>;
\t.reg .b64 \t%rd<6>;
\tld.param.u64 \t%rd1, [probe_a];
\tld.param.u64 \t%rd2, [probe_b];
\tld.param.u64 \t%rd3, [probe_out];
\tcvt.u32.u64 \t%r1, %rd1;
\tcvt.u32.u64 \t%r2, %rd2;
\t{body}
\t{widen}
\tcvta.to.global.u64 \t%rd5, %rd3;
\tst.global.u64 \t[%rd5], %rd4;
\tret;
}}
"""


def _vectors() -> list[tuple[int, int]]:
    rng = random.Random(3)
    pairs = [(a, b) for a in EDGES for b in EDGES]
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(RANDOM)]
    return pairs


def _rocq(text: str, vectors: list[tuple[int, int]]) -> list[int]:
    listed = "; ".join(f"(({a})%Z, ({b})%Z)" for a, b in vectors)
    query = QUERY.replace("VECTORS", f"[{listed}]")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Probe.v"
        path.write_text(export_terms(text, [BINDING]) + COMMON + query)
        result = subprocess.run(
            ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=600,
        )
    assert result.returncode == 0, result.stderr[-2000:]
    printed = result.stdout.split("=", 1)[1].split(":", 1)[0]
    values = []
    for item in printed.replace("\n", " ").strip(" []").split(";"):
        item = item.strip()
        assert item.startswith("Some"), (text, item)
        values.append(int(item.removeprefix("Some").strip(" ()%Z")))
    return values


def _gpu(texts: dict[str, str], vectors: list[tuple[int, int]]) -> dict[str, list[int]]:
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    call(driver.cuCtxSetCurrent(context))
    try:
        (out,) = call(driver.cuMemAlloc(8))
        results: dict[str, list[int]] = {}
        for name, text in texts.items():
            (module,) = call(driver.cuModuleLoadData(text.encode("ascii") + b"\0"))
            (function,) = call(driver.cuModuleGetFunction(module, b"probe"))
            values = []
            for a, b in vectors:
                arguments = ((a, b, address(out)), (ctypes.c_uint64,) * 3)
                call(driver.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, arguments, 0))
                word = bytearray(8)
                call(driver.cuMemcpyDtoH(word, out, 8))
                values.append(int.from_bytes(word, "little"))
            results[name] = values
            call(driver.cuModuleUnload(module))
        call(driver.cuMemFree(out))
        return results
    finally:
        call(driver.cuDevicePrimaryCtxRelease(device))


@pytest.mark.slow
@pytest.mark.gpu
def test_each_instruction_computes_on_the_gpu_what_the_translation_states() -> None:
    vectors = _vectors()
    texts = {name: _text(body) for name, body in PROBES.items()}
    device = _gpu(texts, vectors)
    with ThreadPoolExecutor(max_workers=4) as pool:
        stated = dict(
            zip(texts, pool.map(lambda t: _rocq(t, vectors), texts.values()), strict=True)
        )
    differ = {
        name: [
            (a, b, g, r)
            for (a, b), g, r in zip(vectors, device[name], stated[name], strict=True)
            if g != r
        ][:3]
        for name in texts
    }
    assert {name: found for name, found in differ.items() if found} == {}
