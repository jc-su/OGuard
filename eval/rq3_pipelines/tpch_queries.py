"""The 22 TPC-H queries on cuDF, with every allocation through the catalog and without (RQ3, RQ4).

Run from the repository root, on an otherwise idle GPU, pinned to one core of
the GPU's NUMA node as `eval/timing.py` requires:

    taskset -c <core> uv run python -m eval.timing eval.rq3_pipelines.tpch_queries \\
        eval/records/tpch_queries_<date> --scale S

It generates the eight tables at scale factor S with `eval/tpch/database.py`
and runs each query of `eval/tpch/queries.py` in three configurations:

- `reference`: pandas on the host, once; skipped above scale factor 1, and
  the record says so;
- `cudf`: cuDF with an RMM pool of the catalog configuration's size as RMM's
  current resource, which the tables are loaded into;
- `catalog`: cuDF inside `install_cudf`, in a session of one tenant, with
  `CatalogResource` as RMM's current resource, so that every allocation of the
  query lies in a chunk of the tenant's arena, a catalog buffer of the
  tenant. The arena grows by a chunk, a catalog step, when the tables or a
  query need more than it holds. The tables are loaded in the same session,
  so they lie in the tenant's chunks in the executor's pool, whose size is
  `POOL_FACTOR` times the tables' bytes plus `POOL_MARGIN`.

It checks that each cuDF result equals the reference and that the two cuDF
results are equal: the same columns and rows in the same order, a
floating-point value within a relative tolerance of 1e-9 and any other value
exactly. Per query it records the correctness, the host time of each cuDF
configuration from the start of the query to its result in host memory (the
median, minimum, and maximum of `REPEATS` runs after a warm-up by time, as
`eval/timing.py` measures), and, from one run in the catalog configuration
after cuDF's query caches are released, the steps that the catalog accepted
(the arena's growth by chunks), the allocations and frees in them, the
regions it cleared and their bytes, and the most live buffers and bytes in
the pool, the tables' included. After the last query the tables are freed,
the arena returns to the pool in one step, and the command fails if any
catalog buffer is still live. The record is written before a mismatch makes
the command fail.
"""

import argparse
import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import cudf
import pandas as pd
import rmm

from eval.timing import REPEATS, Timing, environment, host_time, pinned_core
from eval.tpch.database import Tables, generate
from eval.tpch.queries import QUERIES, Frame, Query, host, mismatch
from opgate.catalog import (
    Allocate,
    BufferId,
    CatalogView,
    Clear,
    Launch,
    Step,
    StepRefused,
    TenantId,
)
from opgate.executor import Executor
from opgate.frontends import Target
from opgate.hosts.cudf import install_cudf, release_cudf_query_state
from opgate.hosts.rapids import CatalogResource

__all__ = ["Counts", "CountingExecutor", "in_catalog", "main", "pool_bytes"]

# the database's seed: with it every query gives rows at each scale factor from
# 0.01 to 1 that eval/rq3_pipelines/tpch_queries.md lists
SEED = 154
REFERENCE_LIMIT = 1.0  # the largest scale factor that pandas runs the reference at
TENANT = TenantId(1)
TARGET = Target(8, 6)  # the RTX A6000; no query has a UDF, so no kernel is compiled
POOL_FACTOR = 3
POOL_MARGIN = 256 << 20
POOL_ALIGNMENT = 2 << 20


@dataclass
class Counts:
    """The steps that the catalog accepted during a run, the launches among them
    that a kept prepared step decided (`Executor.launch`), the regions it
    cleared, and the most live buffers and bytes in the pool at any point of
    the run."""

    steps: int
    kept_launches: int
    allocations: int
    frees: int
    clears: int
    cleared_bytes: int
    peak_buffers: int
    peak_bytes: int


def _live(catalog: CatalogView) -> tuple[int, int]:
    """The live buffers and their bytes."""
    return len(catalog.buffers), sum(buffer.nbytes for buffer in catalog.buffers)


class CountingExecutor(Executor):
    """The executor, which counts the steps that the catalog accepts inside `counting`."""

    def __init__(self, device: int, pool_bytes: int) -> None:
        super().__init__(device, pool_bytes)
        self._counts: Counts | None = None

    @contextmanager
    def counting(self) -> Iterator[Counts]:
        """Count the accepted steps of the block."""
        if self._counts is not None:
            raise RuntimeError("the executor is counting already")
        buffers, nbytes = _live(self.catalog)
        counts = Counts(0, 0, 0, 0, 0, 0, buffers, nbytes)
        self._counts = counts
        try:
            yield counts
        finally:
            self._counts = None

    def execute(self, step: Step) -> tuple[bool, ...]:
        counts = self._counts
        if counts is None:
            return super().execute(step)
        # the step's effects, which the executor computes again and does not return
        effects = self.catalog.check(step)
        ran = super().execute(step)
        if isinstance(effects, StepRefused):
            raise AssertionError("the executor performed a step that the catalog refuses")
        counts.steps += 1
        for effect in effects:
            if isinstance(effect, Clear):
                counts.clears += 1
                counts.cleared_bytes += effect.nbytes
        if any(isinstance(action, Allocate) for action in step.actions):
            counts.allocations += 1
            buffers, nbytes = _live(self.catalog)
            counts.peak_buffers = max(counts.peak_buffers, buffers)
            counts.peak_bytes = max(counts.peak_bytes, nbytes)
        return ran

    def launch(self, launch: Launch) -> bool:
        counts = self._counts
        if counts is None:
            return super().launch(launch)
        steps = counts.steps
        ran = super().launch(launch)
        if counts.steps == steps:
            # no step through `execute`: a kept prepared step decided it
            counts.steps += 1
            counts.kept_launches += 1
        return ran

    def free(self, buffer: BufferId, *more: BufferId) -> None:
        super().free(buffer, *more)
        counts = self._counts
        if counts is not None:
            counts.steps += 1
            counts.frees += 1 + len(more)


def table_bytes(database: Tables[pd.DataFrame]) -> int:
    """The bytes of the tables' columns: a string column's characters and offsets."""
    return sum(
        int(frame.memory_usage(deep=True, index=False).sum()) for _, frame in database.frames()
    )


def pool_bytes(database: Tables[pd.DataFrame]) -> int:
    """The executor's pool for the queries over these tables."""
    size = POOL_FACTOR * table_bytes(database) + POOL_MARGIN
    return math.ceil(size / POOL_ALIGNMENT) * POOL_ALIGNMENT


def in_catalog[T](
    database: Tables[pd.DataFrame],
    work: Callable[[CountingExecutor, CatalogResource, Tables[Frame]], T],
) -> T:
    """Run `work` in the catalog configuration: an executor on device 0 with a
    pool sized from the tables, `CatalogResource` as RMM's current resource, a
    session of `TENANT` inside `install_cudf`, and the tables loaded in it.
    Afterwards the tables are freed; a catalog buffer that is still live then
    is an error."""
    pool = pool_bytes(database)
    with (
        CountingExecutor(0, pool) as executor,
        CatalogResource(executor) as resource,
    ):
        with resource.current():
            with executor.tenant(TENANT), install_cudf(executor, resource, TARGET):
                executor.reserve(pool)
                tables = database.map(cudf.from_pandas)
                result = work(executor, resource, tables)
                del tables
                release_cudf_query_state()
                resource.release(TENANT)
                live, nbytes = _live(executor.catalog)
                if live:
                    raise RuntimeError(
                        f"{live} catalog buffers of {nbytes} bytes outlive the queries"
                    )
    return result


class Measured(TypedDict):
    """One configuration's result of a query and its host time."""

    result: pd.DataFrame
    time: Timing


class QueryRecord(TypedDict):
    """The record of one query."""

    query: str
    rows: int
    cudf_equals_reference: bool | None  # None: no reference above REFERENCE_LIMIT
    catalog_equals_reference: bool | None
    catalog_equals_cudf: bool
    mismatches: list[str]
    cudf: Timing
    catalog: Timing
    catalog_counts: dict[str, int]


def _time(query: Query, tables: Tables[Frame]) -> Timing:
    return host_time(lambda: host(query(cudf, tables)))


def _pooled(database: Tables[pd.DataFrame]) -> dict[str, Measured]:
    """Each query on cuDF with an RMM pool of `pool_bytes` as the current resource."""
    size = pool_bytes(database)
    previous = rmm.mr.get_current_device_resource()
    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(), initial_pool_size=size, maximum_pool_size=size
        )
    )
    try:
        tables = database.map(cudf.from_pandas)
        measured: dict[str, Measured] = {}
        for name, query in QUERIES.items():
            measured[name] = {"result": host(query(cudf, tables)), "time": _time(query, tables)}
            print(f"cudf {name}: {measured[name]['time']['median_us']:.0f} us", flush=True)
        del tables
        release_cudf_query_state()
        return measured
    finally:
        rmm.mr.set_current_device_resource(previous)


def _catalog(
    executor: CountingExecutor, resource: CatalogResource, tables: Tables[Frame]
) -> tuple[dict[str, tuple[Measured, Counts]], tuple[int, int]]:
    """Each query in the catalog configuration, with the counts of one run; and
    the live buffers and bytes of the tables."""
    loaded = _live(executor.catalog)
    measured: dict[str, tuple[Measured, Counts]] = {}
    for name, query in QUERIES.items():
        release_cudf_query_state()
        with executor.counting() as counts:
            result = host(query(cudf, tables))
        timing = _time(query, tables)
        measured[name] = ({"result": result, "time": timing}, counts)
        print(f"catalog {name}: {measured[name][0]['time']['median_us']:.0f} us", flush=True)
    return measured, loaded


def _compare(expected: pd.DataFrame, actual: pd.DataFrame, what: str, notes: list[str]) -> bool:
    difference = mismatch(expected, actual)
    if difference is not None:
        notes.append(f"{what}: {difference}")
    return difference is None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scale: float = args.scale
    record: dict[str, object] = {
        "environment": environment(),
        "core": pinned_core(),
        "scale": scale,
        "seed": SEED,
    }
    database = generate(scale, SEED)
    record["tables"] = {name: len(frame) for name, frame in database.frames()}
    record["table_bytes"] = table_bytes(database)
    reference: dict[str, pd.DataFrame] | None = None
    if scale <= REFERENCE_LIMIT:
        reference = {name: host(query(pd, database)) for name, query in QUERIES.items()}
        record["reference"] = "pandas"
    else:
        record["reference"] = (
            f"skipped: pandas runs the reference up to scale factor {REFERENCE_LIMIT}"
        )
    default = _pooled(database)
    catalog, loaded = in_catalog(database, _catalog)
    record["pool_bytes"] = pool_bytes(database)
    record["table_buffers"], record["table_buffer_bytes"] = loaded
    record["repeats"] = REPEATS
    rows: list[QueryRecord] = []
    for name in QUERIES:
        (measured, counts), plain = catalog[name], default[name]
        notes: list[str] = []
        against: tuple[bool, bool] | None = None
        if reference is not None:
            against = (
                _compare(reference[name], plain["result"], "cudf against the reference", notes),
                _compare(
                    reference[name], measured["result"], "catalog against the reference", notes
                ),
            )
        rows.append(
            {
                "query": name,
                "rows": len(plain["result"]),
                "cudf_equals_reference": None if against is None else against[0],
                "catalog_equals_reference": None if against is None else against[1],
                "catalog_equals_cudf": _compare(
                    plain["result"], measured["result"], "catalog against cudf", notes
                ),
                "mismatches": notes,
                "cudf": plain["time"],
                "catalog": measured["time"],
                "catalog_counts": asdict(counts),
            }
        )
    record["queries"] = rows
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    wrong = [row["query"] for row in rows if row["mismatches"]]
    if wrong:
        raise SystemExit(f"results differ in {', '.join(wrong)}; see {args.output}")


if __name__ == "__main__":
    main()
