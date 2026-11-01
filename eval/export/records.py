"""Reading records, aggregating runs, and writing CSV files, tables, and numbers."""

import csv
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from eval import RECORDS, ROOT
from eval.timing import RUNS

FIGURES = ROOT / "paper" / "figures"
DATA = FIGURES / "data"
_output: ContextVar[Path] = ContextVar("export_output", default=FIGURES)


@contextmanager
def output_to(directory: Path):
    """Direct every generated artifact to one staging directory."""
    token = _output.set(directory)
    try:
        yield
    finally:
        _output.reset(token)


__all__ = [
    "DATA",
    "FIGURES",
    "RECORDS",
    "ROOT",
    "SCALE_NAMES",
    "Across",
    "Numbers",
    "across",
    "field",
    "items",
    "load",
    "median",
    "number",
    "obj",
    "runs",
    "table",
    "text",
    "values",
    "write",
]

Numbers = dict[str, str]  # LaTeX macro name, without the backslash, to its text
SCALE_NAMES = {  # TPC-H scale factors in macro names
    0.1: "Tenth",
    0.3: "ThreeTenths",
    1.0: "One",
    3.0: "Three",
    10.0: "Ten",
    30.0: "Thirty",
}


def load(path: Path) -> dict[str, object]:
    loaded: object = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: not an object")
    return {str(k): v for k, v in loaded.items()}


def runs(directory: Path, count: int = RUNS) -> list[dict[str, object]]:
    """The `count` runs of a timed experiment, all on the same core and GPU."""
    found = sorted(directory.glob("run*.json"))
    expected = [directory / f"run{i}.json" for i in range(1, count + 1)]
    if found != expected:
        raise ValueError(
            f"{directory}: runs {[p.name for p in found]}, expected run1 to run{count}"
        )
    loaded = [load(path) for path in found]
    machines = {(json.dumps(r["core"]), json.dumps(field(r, "environment", "gpu"))) for r in loaded}
    if len(machines) != 1:
        raise ValueError(f"{directory}: the runs ran on different cores or GPUs: {machines}")
    return loaded


def obj(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: not an object")
    return {str(k): v for k, v in value.items()}


def values(value: object, where: str) -> list[object]:
    """A list of values of any kind."""
    if not isinstance(value, list):
        raise ValueError(f"{where}: not a list")
    return list(value)


def items(value: object, where: str) -> list[dict[str, object]]:
    """A list of objects."""
    if not isinstance(value, list):
        raise ValueError(f"{where}: not a list")
    return [obj(item, where) for item in value]


def number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where}: not a finite number")
    return result


def text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: not a string")
    return value


def field(record: dict[str, object], *path: str | int) -> object:
    """The value at `path`: a key of an object, or a position in a list."""
    value: object = record
    for step in path:
        where = ".".join(str(s) for s in path)
        match step:
            case int() if isinstance(value, list):
                value = value[step]
            case str():
                value = obj(value, where)[step]
            case _:
                raise ValueError(
                    f"{where}: {step!r} is not a key of an object or a position in a list"
                )
    return value


def median(record: dict[str, object], *path: str | int) -> float:
    """The median in microseconds of one timing of one run."""
    return number(field(record, *path, "median_us"), ".".join(str(s) for s in path))


@dataclass(frozen=True)
class Across:
    """One timing across the runs of an experiment, in microseconds: the median
    of the runs' medians, and the lowest and highest of them."""

    median: float
    low: float
    high: float

    def columns(self, scale: float = 1.0, digits: int = 1) -> tuple[str, str, str]:
        """The median, low, and high, divided by `scale`, for a CSV row."""
        middle, low, high = (f"{v / scale:.{digits}f}" for v in (self.median, self.low, self.high))
        return middle, low, high


def across(records: Sequence[dict[str, object]], *path: str | int) -> Across:
    """A timing at `path` in every run."""
    values = [median(r, *path) for r in records]
    return Across(statistics.median(values), min(values), max(values))


def write(name: str, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """A CSV file of `paper/figures/data/`."""
    directory = _output.get() / "data"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def table(name: str, columns: str, header: str, rows: Iterable[str], caption: str) -> None:
    """A generated table `paper/figures/<name>.tex`, labeled `tab:<name without tab_>`."""
    label = name.removeprefix("tab_")
    body = "\n".join(rows)
    directory = _output.get()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.tex").write_text(
        "% generated by `python -m eval.export` from the records; do not edit\n"
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{label}}}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{2.5pt}\n"
        f"\\begin{{tabular}}{{{columns}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
