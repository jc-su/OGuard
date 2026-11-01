"""RQ1: Rocq decides every step of the evaluation's workloads as the deployed catalog did.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq1_safety.trace_replay \\
        --output eval/records/trace_replay_<date>.json

The catalog that decides each step at run time is the Python transcription
of `check_step` in `formal/CatalogStep.v`, which the conformance test
compares with Rocq on generated steps. Here each workload runs with the
executor recording its catalog's decisions (`Executor.trace`), and
`opgate.catalog.rocq.replay` has Rocq fold `check_step` over the record from
the catalog at the trace's start: a translation validation of the run.
For each recorded step the record keeps Rocq's printed decision and
Python's, which must be equal after the buffers, grants, and freed regions
are put in order, and whether Python's decision is the one the run applied:
a launch that a kept prepared step decided from its memo, or a use with new
scalars, is compared with the full check of the same step. The Rocq source
of each replay is kept with the record, as the run's inputs.

The workloads:

- `cudf_tpch`: two tenants each load the TPC-H tables at scale factor
  `CUDF_SCALE` into their arenas through `CatalogResource` and run queries
  on cuDF (the first all 22, the second Q1 and Q6), each releasing its arena
  afterwards, so the second tenant's chunks reuse the first's regions.
- `cudf_udfs`: the `Series.apply` and `DataFrame.apply` examples of
  `eval.rq2_coverage.cudf_udfs`, verified and launched through the hook on
  views of the tenant's chunks.
- `torch_tpch`: TPC-H Q1, Q6, and Q14 on the PyTorch engine at scale factor
  `TORCH_SCALE`, twice each, with the Fable Triton kernels through the gate:
  segments, kept launches, and guarded gathers.
- `pipelines`: the executor's TPC-H pipelines (`eval.tpch.pipelines`) at
  scale factor `PIPELINE_SCALE`: uploads, index outputs, fills, static and
  guarded gathers, and graphs.
- `refusals`: on the PyTorch host, a second tenant's requests that name the
  first tenant's tensors (a launch that writes one, a launch that reads one,
  a free, a grant), and the first tenant's launches of the corpus's Fable
  projection with a view past its segment's end, a view off a multiple of
  256 bytes, and a size larger than its buffers, and of the Q6 kernel with a
  scalar outside its bounds, each refused, beside the accepted launch they
  vary.

`--catalog-library` names the compiled Rocq library whose `check_step` the
replay evaluates, by default the one that verifies the operators
(`formal_dir()`); the catalog's decisions do not depend on the gate's rules,
so the two may differ while the gate changes, and the record names both.
The command fails if any decision differs; the record is written first.
"""

import argparse
import gc
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import cudf
import torch

from eval.rq2_coverage import cudf_udfs
from eval.rq3_pipelines.tpch_queries import SEED, pool_bytes
from eval.tpch import data as tpch_data
from eval.tpch.database import generate
from eval.tpch.operators import register
from eval.tpch.pipelines import Q6, THRESHOLD, Session, load, q6, revenue, unload
from eval.tpch.pipelines import Path as IndexPath
from eval.tpch.queries import QUERIES, host
from eval.tpch.torch_engine import QUERIES as TORCH_QUERIES
from eval.tpch.torch_engine import GatedOperators
from eval.tpch.torch_engine import load as torch_load
from opgate.catalog import Catalog, Decided, Free, Grant, Launch, Step, TenantId, Traced, View
from opgate.catalog.rocq import as_sets, replay
from opgate.executor import Executor, RefusedError
from opgate.frontends import Target
from opgate.gate import formal_dir
from opgate.hosts.cudf import install_cudf, release_cudf_query_state
from opgate.hosts.rapids import CatalogResource
from opgate.hosts.torch import TorchHost

__all__ = ["main"]

ALICE, BOB = TenantId(1), TenantId(2)
TARGET = Target(8, 6)
CUDF_SCALE = 0.1
TORCH_SCALE = 0.1
PIPELINE_SCALE = 0.001
WORKLOADS = ("cudf_tpch", "cudf_udfs", "torch_tpch", "pipelines", "refusals")


@dataclass(frozen=True)
class Run:
    """A workload's record: the catalog at the trace's start and what it recorded."""

    start: Catalog
    trace: list[Traced]


@contextmanager
def _traced(executor: Executor, runs: dict[str, Run], name: str) -> Iterator[None]:
    with executor.trace() as (start, trace):
        yield
    runs[name] = Run(start, list(trace))


def _cudf_tpch(runs: dict[str, Run]) -> None:
    database = generate(CUDF_SCALE, SEED)
    pool = pool_bytes(database)
    with (
        Executor(0, pool) as executor,
        CatalogResource(executor) as resource,
        resource.current(),
        _traced(executor, runs, "cudf_tpch"),
    ):
        # the tenants run in turn, each with the whole pool as its reservation,
        # so that BOB's allocations take the regions that ALICE freed
        for tenant, names in ((ALICE, tuple(QUERIES)), (BOB, ("Q1", "Q6"))):
            with executor.tenant(tenant), install_cudf(executor, resource, TARGET):
                executor.reserve(pool)
                tables = database.map(cudf.from_pandas)
                for name in names:
                    host(QUERIES[name](cudf, tables))
                del tables
                release_cudf_query_state()
            if resource.release(tenant) == 0:
                raise RuntimeError(f"tenant {tenant}'s arena holds allocations after its queries")
            with executor.tenant(tenant):
                executor.unreserve()


def _cudf_udfs(runs: dict[str, Run]) -> None:
    with (
        Executor(0, 1 << 28) as executor,
        CatalogResource(executor, chunk_bytes=1 << 22) as resource,
        resource.current(),
        _traced(executor, runs, "cudf_udfs"),
    ):
        with executor.tenant(ALICE), install_cudf(executor, resource, TARGET):
            executor.reserve(1 << 28)
            for example in cudf_udfs.APPLIES:
                cudf_udfs.run_apply(example)
            gc.collect()
        resource.release(ALICE)


def _torch_tpch(runs: dict[str, Run]) -> None:
    database = generate(TORCH_SCALE, SEED)
    with Executor(0, 1 << 30) as executor, TorchHost(executor) as torch_host:
        with _traced(executor, runs, "torch_tpch"):
            with executor.tenant(ALICE):
                executor.reserve(1 << 30)
                modules = GatedOperators.register(executor)
            with torch_host.session(ALICE):
                tables = torch_load(database)
                ops = GatedOperators(torch_host, modules)
                for query in TORCH_QUERIES.values():
                    for _ in range(2):
                        query(ops, tables)
                del tables, ops
            torch_host.release(ALICE)


def _pipelines(runs: dict[str, Run]) -> None:
    _, lineitem = tpch_data.generate(PIPELINE_SCALE, seed=11)
    query = Q6(tpch_data.day(date(1994, 1, 1)), tpch_data.day(date(1995, 1, 1)), 0.05, 0.07, 24.0)
    with Executor(0, 1 << 26) as executor, _traced(executor, runs, "pipelines"):
        with executor.tenant(ALICE):
            executor.reserve(1 << 26)
        s = Session(executor, ALICE, executor)
        ops = register(executor)
        columns = load(s, lineitem)
        q6(s, ops, columns, query)
        for path in IndexPath:
            for graph in (False, True):
                revenue(s, ops, columns, path, THRESHOLD, graph)
        unload(s, columns)


def _refused(action: Callable[[], object]) -> None:
    try:
        action()
    except RefusedError:
        return
    raise RuntimeError("a request that the catalog must refuse was accepted")


def _refusals(runs: dict[str, Run]) -> None:
    rows = 4096
    with Executor(0, 1 << 28) as executor, TorchHost(executor) as torch_host:
        with _traced(executor, runs, "refusals"):
            for tenant in (ALICE, BOB):
                with executor.tenant(tenant):
                    executor.reserve(1 << 27)
            with executor.tenant(ALICE):
                modules = GatedOperators.register(executor)
            projection, q6_sum = modules["projection"], modules["q6_sum"]
            grid = (-(-rows // 1024), 1, 1)
            with torch_host.session(ALICE):
                price = torch.ones(rows, dtype=torch.float64, device="cuda")
                out = torch.empty(rows, dtype=torch.float64, device="cuda")
                arguments = tuple(torch_host.argument_at(t) for t in (price, price, out))
                launch = Launch(projection.operator, grid, arguments, (rows,), (), False)
                assert executor.launch(launch)
                (segment,) = torch_host.segments(ALICE)
                found = executor.catalog.buffer(segment)
                if found is None:
                    raise RuntimeError("the tenant's segment is not live")
                for wrong in (
                    replace(launch, buffers=(*arguments[:2], View(segment, found.nbytes - 256))),
                    replace(launch, buffers=(*arguments[:2], View(segment, 128))),
                    replace(launch, sizes=(found.nbytes,)),
                ):
                    _refused(lambda wrong=wrong: executor.launch(wrong))
                shipdate = torch.zeros(rows, dtype=torch.int32, device="cuda")
                result = torch.zeros(1, dtype=torch.float64, device="cuda")
                columns = tuple(
                    torch_host.argument_at(t) for t in (shipdate, price, price, price, result)
                )
                scalars = (0, 2**32, 0, 0, 0)  # a date past the int32 bound
                outside = Launch(q6_sum.operator, grid, columns, (rows,), scalars, False)
                _refused(lambda: executor.launch(outside))
                own = torch_host.buffer(rows, torch.float64)
                buffer = torch_host.argument_at(own)
            if isinstance(buffer, View):
                raise RuntimeError("a buffer of its own is a view")
            with torch_host.session(BOB):
                theirs = torch.zeros(rows, dtype=torch.float64, device="cuda")
                mine = torch_host.argument_at(theirs)
                writes = Launch(
                    projection.operator, grid, (mine, mine, arguments[2]), (rows,), (), False
                )
                _refused(lambda: executor.launch(writes))
                reads = Launch(
                    projection.operator, grid, (arguments[0], mine, mine), (rows,), (), False
                )
                _refused(lambda: executor.launch(reads))
                _refused(lambda: executor.execute(Step(BOB, (Free(buffer),))))
                _refused(lambda: executor.execute(Step(BOB, (Grant(buffer, BOB),))))
                del theirs
            with torch_host.session(ALICE):
                del price, out, shipdate, result, own
            torch_host.release(ALICE)
            torch_host.release(BOB)


RUNNERS: dict[str, Callable[[dict[str, Run]], None]] = {
    "cudf_tpch": _cudf_tpch,
    "cudf_udfs": _cudf_udfs,
    "torch_tpch": _torch_tpch,
    "pipelines": _pipelines,
    "refusals": _refusals,
}


def _replayed(name: str, run: Run, library: Path, inputs: Path) -> dict[str, object]:
    replayed = replay(run.start, run.trace, library=library, keep=inputs / f"{name}.v")
    steps = [t for t in run.trace if isinstance(t, Decided)]
    return {
        "workload": name,
        "operators_verified": len(run.trace) - len(steps),
        "steps": len(steps),
        "refused": sum(r.python.startswith("refused") for r in replayed),
        "launches": sum("kernel(" in r.python for r in replayed),
        "rocq_equals_python": sum(as_sets(r.rocq) == as_sets(r.python) for r in replayed),
        "python_equals_run": sum(r.as_run for r in replayed),
        "decisions": [{"rocq": r.rocq, "python": r.python, "as_run": r.as_run} for r in replayed],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-library", type=Path, default=None)
    args = parser.parse_args()
    gate = formal_dir()
    library = gate if args.catalog_library is None else args.catalog_library.resolve()
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    inputs = args.output.with_suffix(args.output.suffix + ".inputs")
    inputs.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    runs: dict[str, Run] = {}
    for name in WORKLOADS:
        RUNNERS[name](runs)
        print(f"{name}: {len(runs[name].trace)} recorded", flush=True)
    workloads = []
    for name in WORKLOADS:
        workloads.append(_replayed(name, runs[name], library, inputs))
        print({k: v for k, v in workloads[-1].items() if k != "decisions"}, flush=True)
    args.output.write_text(
        json.dumps(
            {
                "experiment": "Rocq's check_step on every recorded step of the workloads",
                "started_at_utc": started,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "scales": {"cudf": CUDF_SCALE, "torch": TORCH_SCALE, "pipelines": PIPELINE_SCALE},
                "gate_library": str(gate),
                "catalog_library": str(library),
                "workloads": workloads,
            },
            indent=2,
        )
        + "\n"
    )
    wrong = [
        w["workload"]
        for w in workloads
        if w["rocq_equals_python"] != w["steps"] or w["python_equals_run"] != w["steps"]
    ]
    if wrong:
        raise SystemExit(f"decisions differ in {', '.join(map(str, wrong))}; see {args.output}")


if __name__ == "__main__":
    main()
