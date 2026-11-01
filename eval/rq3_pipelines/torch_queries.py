"""TPC-H Q1, Q6, and Q14 on a PyTorch tensor engine with model-written operators (RQ3, RQ4).

Run from the repository root, on an otherwise idle GPU, pinned to one core of
the GPU's NUMA node as `eval/timing.py` requires:

    taskset -c <core> uv run python -m eval.timing eval.rq3_pipelines.torch_queries \\
        eval/records/torch_queries_<date> --scale S

It generates the tables at scale factor S with `eval/tpch/database.py` and
runs each query of `eval/tpch/torch_engine.py` in three configurations:

- `torch`: PyTorch's own operators;
- `direct`: the corpus's Triton kernels that the engine's `MODELS` names
  (Fable's `q6_sum`, `projection`, and `gather`, and Opus's `scatter_add`),
  launched unchecked by Triton through their `run` functions;
- `opgate`: the same kernels' recorded PTX, verified by the gate and launched
  through the executor on the tensors of one tenant, whose PyTorch segments
  are catalog buffers (`opgate.hosts.torch`); the tables are loaded in the
  tenant's session.

It records each configuration's result of each query, whether it equals
pandas' (up to scale factor `REFERENCE_LIMIT`) and the `torch`
configuration's, its host time from the start of the query to its result on
the host (the median, minimum, and maximum of `REPEATS` runs after a warm-up
by time, as `eval/timing.py` measures), and, from one run in the `opgate`
configuration, the steps that the catalog accepted: the launches, the
allocation and free of the buffers of the gather's row ids and the
`scatter_add`'s group numbers, and the segments that PyTorch takes. After
the queries the tenant's pool is released, and the command fails if a
catalog buffer is still live. The record is written before a mismatch makes
the command fail.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import pandas as pd

from eval.rq3_pipelines.tpch_queries import (
    REFERENCE_LIMIT,
    SEED,
    CountingExecutor,
    Counts,
    table_bytes,
)
from eval.timing import REPEATS, Timing, environment, host_time, pinned_core
from eval.tpch.database import Tables, generate
from eval.tpch.queries import QUERIES as PANDAS
from eval.tpch.queries import host, mismatch
from eval.tpch.torch_engine import (
    MODELS,
    QUERIES,
    DirectOperators,
    GatedOperators,
    Operators,
    TorchOperators,
    TorchTables,
    load,
)
from opgate.catalog import TenantId
from opgate.hosts.torch import TorchHost

__all__ = ["main"]

TENANT = TenantId(1)
POOL_FACTOR = 3  # the executor's pool: this many times the tables' bytes, and the margin
POOL_MARGIN = 1 << 30
CONFIGURATIONS = ("torch", "direct", "opgate")


class Measured(TypedDict):
    """One configuration's result of a query and its host time."""

    result: dict[str, list[object]]
    time: Timing


class QueryRecord(TypedDict):
    query: str
    equals_reference: dict[str, bool] | None  # None: no reference above REFERENCE_LIMIT
    equals_torch: dict[str, bool]
    mismatches: list[str]
    measured: dict[str, Measured]
    opgate_counts: dict[str, int]


def _measure(
    query: Callable[[Operators, TorchTables], pd.DataFrame], ops: Operators, tables: TorchTables
) -> tuple[pd.DataFrame, Timing]:
    return query(ops, tables), host_time(lambda: query(ops, tables))


def _same(expected: pd.DataFrame, actual: pd.DataFrame, what: str, notes: list[str]) -> bool:
    difference = mismatch(expected, actual)
    if difference is not None:
        notes.append(f"{what}: {difference}")
    return difference is None


def _columns(frame: pd.DataFrame) -> dict[str, list[object]]:
    """A result as its columns' values, as the record keeps it."""
    return {name: frame[name].tolist() for name in frame.columns}


def _unchecked(
    database: Tables[pd.DataFrame],
) -> dict[str, dict[str, tuple[pd.DataFrame, Timing]]]:
    """Each query with PyTorch's operators and with the model's kernels, unchecked."""
    tables = load(database)
    measured = {
        name: {
            configuration: _measure(query, ops, tables)
            for configuration, ops in (("torch", TorchOperators()), ("direct", DirectOperators()))
        }
        for name, query in QUERIES.items()
    }
    del tables
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scale: float = args.scale
    database = generate(scale, SEED)
    record: dict[str, object] = {
        "environment": environment(),
        "core": pinned_core(),
        "scale": scale,
        "seed": SEED,
        "models": MODELS,
        "repeats": REPEATS,
        "table_bytes": table_bytes(database),
    }
    reference: dict[str, pd.DataFrame] | None = None
    if scale <= REFERENCE_LIMIT:
        reference = {name: host(PANDAS[name](pd, database)) for name in QUERIES}
    unchecked = _unchecked(database)
    pool = POOL_FACTOR * table_bytes(database) + POOL_MARGIN
    record["pool_bytes"] = pool
    gated: dict[str, tuple[pd.DataFrame, Timing, Counts]] = {}
    with CountingExecutor(0, pool) as executor, TorchHost(executor) as torch_host:
        with executor.tenant(TENANT):
            executor.reserve(pool)
            modules = GatedOperators.register(executor)
        with torch_host.session(TENANT):
            tables = load(database)
            ops = GatedOperators(torch_host, modules)
            record["table_segments"] = len(torch_host.segments(TENANT))
            for name, query in QUERIES.items():
                with executor.counting() as counts:
                    query(ops, tables)
                result, timing = _measure(query, ops, tables)
                gated[name] = (result, timing, counts)
            del tables, ops
        torch_host.release(TENANT)
        if executor.catalog.buffers:
            raise RuntimeError(
                f"{len(executor.catalog.buffers)} catalog buffers outlive the queries"
            )
    rows: list[QueryRecord] = []
    for name in QUERIES:
        results = {
            "torch": unchecked[name]["torch"][0],
            "direct": unchecked[name]["direct"][0],
            "opgate": gated[name][0],
        }
        times = {
            "torch": unchecked[name]["torch"][1],
            "direct": unchecked[name]["direct"][1],
            "opgate": gated[name][1],
        }
        notes: list[str] = []
        rows.append(
            {
                "query": name,
                "equals_reference": None
                if reference is None
                else {
                    c: _same(reference[name], results[c], f"{c} against pandas", notes)
                    for c in results
                },
                "equals_torch": {
                    c: _same(results["torch"], results[c], f"{c} against torch", notes)
                    for c in results
                },
                "mismatches": notes,
                "measured": {
                    c: {"result": _columns(results[c]), "time": times[c]} for c in CONFIGURATIONS
                },
                "opgate_counts": asdict(gated[name][2]),
            }
        )
    record["queries"] = rows
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    wrong = [row["query"] for row in rows if row["mismatches"]]
    if wrong:
        raise SystemExit(f"results differ in {', '.join(wrong)}; see {args.output}")


if __name__ == "__main__":
    main()
