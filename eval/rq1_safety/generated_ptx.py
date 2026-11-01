"""Gate a compiler-emitted PTX variant from the independent E6 replay.

The checked PTX and Triton metadata were copied from E6's isolated replay
cache after the commit-pinned Claude Opus Q3 module reproduced its upstream
illegal-address failure.  The defect is across autotuner launches: a counter
is mutated and not reset.  OpGate receives the kernel's most permissive honest
binding, including a read-write counter with no assumed initial value.
"""

import argparse
import gzip
import json
import re
import time
from pathlib import Path
from typing import Literal, TypedDict, assert_never

from eval import RECORDS
from eval.motivation.generated_kernel import Provenance, verify_upstream
from opgate.binding import GRID_LIMIT, Base, Binding, Bounded, Size, Span
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.signature import Affine, Direction, SizeSymbol

DEFAULT_PTX = RECORDS / "e6_q3_filter_sm86_2026-09-27.ptx.gz"
DEFAULT_METADATA = RECORDS / "e6_q3_filter_sm86_2026-09-27.metadata.json"
E6_RECORD = RECORDS / "e6_datakernelbench_2026-09-27.json"
ENTRY = "_q3_filter_kernel"
PARAMETERS = tuple(f"{ENTRY}_param_{index}" for index in range(16))

Outcome = Literal["accepted", "refused"]


class TritonTarget(TypedDict):
    backend: str
    arch: int
    warp_size: int


class TritonMetadata(TypedDict):
    """The fields of Triton's compilation metadata that the experiment reads."""

    target: TritonTarget
    num_warps: int
    num_stages: int
    triton_version: str
    shared: int


class MetadataSummary(TypedDict):
    target: TritonTarget
    num_warps: int
    num_stages: int
    triton_version: str


class DerivedArtifact(TypedDict):
    ptx_gzip: str
    ptx_bytes: int
    metadata: str
    metadata_summary: MetadataSummary
    captured_from: str


class E6Replay(TypedDict):
    path: str
    success: bool


class SizeSymbolRecord(TypedDict):
    name: str
    minimum: int


class BindingSummary(TypedDict):
    entry: str
    parameters: int
    block: list[int]
    grid: list[int]
    size_symbols: list[SizeSymbolRecord]
    counter: str
    outputs: str


class GateRecord(TypedDict):
    decision: Outcome
    reason: str | None
    seconds: float


class Result(TypedDict):
    experiment: str
    success: bool
    upstream: Provenance
    derived_artifact: DerivedArtifact
    e6_record: E6Replay
    binding: BindingSummary
    gate: GateRecord
    meaning: str
    scope: str


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, found {value!r}")
    return value


def _string(record: dict[str, object], key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} is not a string: {value!r}")
    return value


def _integer(record: dict[str, object], key: str) -> int:
    value = record[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} is not an integer: {value!r}")
    return value


def _base(width: int, direction: Direction, extent: Affine) -> Base:
    return Base(Span(direction, extent.times(width), {1: 0, 2: 1, 4: 2, 8: 3}[width], None, False))


def _binding(shared: int) -> Binding:
    n = Affine.symbol(0)
    parameters = (
        _base(8, Direction.READ, n),  # l_orderkey
        _base(8, Direction.READ, n),  # l_extendedprice
        _base(8, Direction.READ, n),  # l_discount
        _base(4, Direction.READ, n),  # l_shipdate
        _base(4, Direction.READ, n),  # o_orderdate
        _base(4, Direction.READ, n),  # o_shippriority
        Bounded(0, 2**32 - 1),  # shipdate_lo
        Bounded(0, 2**32 - 1),  # orderdate_hi
        _base(8, Direction.WRITE, n),  # out_lk
        _base(4, Direction.WRITE, n),  # out_od
        _base(4, Direction.WRITE, n),  # out_sp
        _base(8, Direction.WRITE, n),  # out_rev
        _base(4, Direction.READ_WRITE, Affine(1)),  # atomic counter
        Size(n),
        Bounded(0, 2**64 - 1),  # unused Triton global scratch pointer
        Bounded(0, 2**64 - 1),  # unused Triton profile scratch pointer
    )
    return Binding(
        ENTRY,
        (SizeSymbol("n", 1),),
        tuple(zip(PARAMETERS, parameters, strict=True)),
        (128, 1, 1),
        GRID_LIMIT,
        Affine(shared),
    )


def _verify_derived_artifact(ptx_path: Path, metadata_path: Path) -> tuple[str, TritonMetadata]:
    metadata = _object(json.loads(metadata_path.read_text()))
    expected_metadata: dict[str, object] = {
        "name": ENTRY,
        "num_warps": 4,
        "num_stages": 2,
        "warp_size": 32,
        "arch": "sm86",
        "triton_version": "3.8.0",
    }
    actual_metadata = {key: metadata.get(key) for key in expected_metadata}
    if actual_metadata != expected_metadata:
        raise RuntimeError(
            f"unexpected Triton compilation metadata: expected {expected_metadata}, "
            f"found {actual_metadata}"
        )
    target = metadata.get("target")
    if target != {"backend": "cuda", "arch": 86, "warp_size": 32}:
        raise RuntimeError(f"unexpected Triton target: {target}")

    with gzip.open(ptx_path, "rt", encoding="ascii") as source:
        text = source.read()
    entries = re.findall(r"\.entry\s+([\w$.]+)\s*\(", text)
    if entries != [ENTRY]:
        raise RuntimeError(f"unexpected PTX entries: {entries}")
    parameter_block = text[
        text.index(f".entry {ENTRY}(") : text.index(")", text.index(f".entry {ENTRY}("))
    ]
    declared = tuple(re.findall(rf"\b{re.escape(ENTRY)}_param_\d+\b", parameter_block))
    if declared != PARAMETERS:
        raise RuntimeError(f"unexpected PTX parameters: {declared}")
    witnesses = (
        ".reqntid 128",
        "atom.global.gpu.acq_rel.add.u32",
        "st.global.b64",
        "st.global.b32",
    )
    missing_witnesses = [witness for witness in witnesses if witness not in text]
    if missing_witnesses:
        raise RuntimeError(f"compiled PTX lacks expected witnesses: {missing_witnesses}")
    target_record = _object(target)
    return text, {
        "target": {
            "backend": _string(target_record, "backend"),
            "arch": _integer(target_record, "arch"),
            "warp_size": _integer(target_record, "warp_size"),
        },
        "num_warps": _integer(metadata, "num_warps"),
        "num_stages": _integer(metadata, "num_stages"),
        "triton_version": _string(metadata, "triton_version"),
        "shared": _integer(metadata, "shared"),
    }


def run(upstream: Path, ptx_path: Path, metadata_path: Path) -> Result:
    provenance = verify_upstream(upstream)
    recorded_e6 = _object(json.loads(E6_RECORD.read_text()))
    recorded_success = recorded_e6.get("success")
    if recorded_success is not True:
        raise RuntimeError("the recorded E6 replay was not successful")

    text, metadata = _verify_derived_artifact(ptx_path, metadata_path)
    # the dynamic shared memory of every launch, which Triton's metadata states
    binding = _binding(metadata["shared"])
    started = time.perf_counter()
    decision = verify(zeroed(text, binding), [binding], timeout=600)
    seconds = time.perf_counter() - started
    match decision:
        case Accepted():
            outcome: Outcome = "accepted"
            reason = None
        case Refused(reason=refused):
            outcome = "refused"
            reason = refused
        case _ as never:
            assert_never(never)
    return {
        "experiment": "E6c: gate an exact Triton PTX variant from the independent replay",
        "success": outcome == "refused",
        "upstream": provenance,
        "derived_artifact": {
            "ptx_gzip": str(ptx_path),
            "ptx_bytes": len(text.encode("ascii")),
            "metadata": str(metadata_path),
            "metadata_summary": {
                "target": metadata["target"],
                "num_warps": metadata["num_warps"],
                "num_stages": metadata["num_stages"],
                "triton_version": metadata["triton_version"],
            },
            "captured_from": "the local E6 replay's Triton cache",
        },
        "e6_record": {
            "path": str(E6_RECORD),
            "success": recorded_success,
        },
        "binding": {
            "entry": binding.kernel,
            "parameters": len(binding.parameters),
            "block": list(binding.block),
            "grid": list(binding.grid),
            "size_symbols": [
                {"name": size.name, "minimum": size.minimum} for size in binding.sizes
            ],
            "counter": "4-byte read-write span; no initial-value assumption",
            "outputs": "four n-row write spans",
        },
        "gate": {"decision": outcome, "reason": reason, "seconds": seconds},
        "meaning": (
            "The implemented gate refuses compiler-emitted PTX from an independently "
            "generated SQL kernel that E6 observed poison a shared cuDF worker."
        ),
        "scope": (
            "This is one sm_86 autotune variant with a manually constructed binding. "
            "A Triton frontend and complete mediation of every autotuner launch remain "
            "implementation work."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--ptx", type=Path, default=DEFAULT_PTX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.upstream, args.ptx.resolve(), args.metadata.resolve())
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
