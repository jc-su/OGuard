"""Evaluation code: the corpora, the experiments of each research question, and
the exporter of their records. Not part of the `opgate` package."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "eval" / "records"


def module_of(file: str) -> str:
    """The module name of an evaluation source file, for `python -m` from `ROOT`."""
    return ".".join(Path(file).resolve().relative_to(ROOT).with_suffix("").parts)
