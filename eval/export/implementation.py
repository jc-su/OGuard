"""The implementation: the lines of the trusted code, the checker, and its proofs."""

from pathlib import Path

from eval.export.records import ROOT, Numbers

# each part that the paper sizes, as a macro's name and the files of the part
PARTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # the PTX parser and translation, and the instruction set and the meaning of facts
    (
        "valTrustedRocqLines",
        ("formal/GateParse.v", "formal/GateFront.v", "formal/GateIsa.v", "formal/GateMeaning.v"),
    ),
    # the export of text and binding, the catalog, the executor, and the host
    # integration and service boundary
    (
        "valTrustedPythonLines",
        (
            "opgate/gate/*.py",
            "opgate/catalog/*.py",
            "opgate/executor/*.py",
            "opgate/hosts/*.py",
            "opgate/hosts/*/*.py",
        ),
    ),
    # the checker, its domain and certificate checks, and its proofs of soundness,
    # termination, and participation
    (
        "valCheckerLines",
        (
            "formal/GateForm.v",
            "formal/GateDomain.v",
            "formal/GateInterval.v",
            "formal/GateContext.v",
            "formal/GateLoop.v",
            "formal/GateCertificate.v",
            "formal/GateTerm.v",
            "formal/GateEnds.v",
            "formal/GateBarrier.v",
            "formal/GateShared.v",
            "formal/GateThread.v",
            "formal/GateSound*.v",
            "formal/GateZero.v",
            "formal/GateParticipation.v",
        ),
    ),
    # the catalog, the launch model, the link, the zeroing prologue, and isolation
    (
        "valProofLines",
        (
            "formal/GatePrologue.v",
            "formal/CatalogStep.v",
            "formal/Store*.v",
            "formal/Pipeline*.v",
        ),
    ),
)


def export() -> Numbers:
    """The lines of each part of the implementation."""
    numbers: Numbers = {}
    counted: list[Path] = []
    for name, patterns in PARTS:
        files = sorted({path for pattern in patterns for path in ROOT.glob(pattern)})
        if not files:
            raise ValueError(f"{name}: no file matches {patterns}")
        counted += files
        lines = sum(len(path.read_text().splitlines()) for path in files)
        numbers[name] = f"{lines:,}"
    # every Rocq file lies in exactly one part
    rocq = [path for path in counted if path.suffix == ".v"]
    if len(set(rocq)) != len(rocq) or set(rocq) != set(ROOT.glob("formal/*.v")):
        raise ValueError("the Rocq parts do not partition formal/*.v")
    return numbers
