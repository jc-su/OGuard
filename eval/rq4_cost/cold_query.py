"""RQ4: the cold start of TPC-H queries with model-written kernels, phase by phase,
beside the same queries warm.

    taskset -c <core> uv run python -m eval.timing eval.rq4_cost.cold_query \\
        eval/records/cold_query_<date> --scale 1

Each run is a fresh process (`eval.timing`) pinned to one core of the GPU's
NUMA node, so the gate's decisions start uncached and every compile has an
empty Triton cache. The run draws the tables at the scale factor before it
times anything, and then times, in order:

- `context`: the executor's context and pool, and the PyTorch host;
- for each kernel of the PyTorch engine (`torch_engine.MODELS`): `compile`,
  the front end's compile of the model's source to PTX
  (`generation.compile_seconds`); `decide`, the service's decision on the
  corpus's PTX (`opgate.hosts.service.server._check`: the check with the
  zeroing prologue and, when the guard pass applies, its rounds, under the
  registration budget); and `load`, `opgate.hosts.register` of the same text,
  whose checks the gate's cache answers, so that it times the prologue, the
  module load, and the catalog's record of the operator;
- `tables`: the tables' columns copied into the tenant's session;
- for each query of `torch_engine.QUERIES`: `first`, its first run on the
  registered kernels, from the start of the query to its result on the host,
  which includes each kernel's first launch; and `warm`, its host time after
  the warm-up (`eval.timing.host_time`).

The record keeps whether each compile's PTX bytes equal the corpus's text
(`eval.rq4_cost.generation`), and each query's result and its first
difference from pandas' result up to scale factor `REFERENCE_LIMIT`. The
command fails after writing the record if the gate refuses a kernel or a
result differs.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import TypedDict, assert_never

import pandas as pd

from eval.corpus.adapter import binding_of, directory_of
from eval.rq3_pipelines.tpch_queries import REFERENCE_LIMIT, SEED, table_bytes
from eval.rq4_cost.generation import compile_seconds
from eval.timing import Timing, environment, host_time, pinned_core
from eval.tpch.database import Tables, generate
from eval.tpch.queries import QUERIES as PANDAS
from eval.tpch.queries import host, mismatch
from eval.tpch.torch_engine import (
    MODELS,
    QUERIES,
    GatedOperators,
    Operators,
    TorchTables,
    load,
)
from opgate.catalog import TenantId
from opgate.executor import Executor, Module
from opgate.gate import REGISTRATION_SECONDS, Accepted, Refused
from opgate.hosts import register
from opgate.hosts.service.server import _check
from opgate.hosts.torch import TorchHost

__all__ = ["main"]

TENANT = TenantId(1)
POOL_FACTOR = 3  # the executor's pool: this many times the tables' bytes, and the margin
POOL_MARGIN = 1 << 30


class KernelRecord(TypedDict):
    """One kernel's cold phases, in seconds."""

    kernel: str
    compile_seconds: float
    same_as_corpus: bool
    decide_seconds: float
    refusal: str | None  # the gate's reason, None when it accepts
    load_seconds: float | None  # None when the gate refuses


class QueryRecord(TypedDict):
    query: str
    first_seconds: float
    warm: Timing
    result: dict[str, list[object]]
    mismatch: str | None  # None: equal to pandas', or no reference above REFERENCE_LIMIT


def _kernel(executor: Executor, task: str, model: str) -> tuple[KernelRecord, Module | None]:
    """The cold phases of the engine's kernel of `task`, and its module when the
    gate accepts it."""
    key = (model, task, "triton")
    corpus = (directory_of(model, task) / "triton.ptx").read_text()
    generated, compiled = compile_seconds(key)
    binding = binding_of(*key, corpus)
    before = time.perf_counter()
    decision = _check(corpus, binding, REGISTRATION_SECONDS, Event()).decision
    decided = time.perf_counter() - before
    module: Module | None = None
    loaded: float | None = None
    refusal: str | None = None
    match decision:
        case Accepted():
            with executor.tenant(TENANT):
                before = time.perf_counter()
                module = register(executor, corpus, binding).module
                loaded = time.perf_counter() - before
        case Refused(reason=reason):
            refusal = reason
        case _ as never:
            assert_never(never)
    found: KernelRecord = {
        "kernel": "/".join(key),
        "compile_seconds": compiled,
        "same_as_corpus": generated.encode("ascii") == corpus.encode("ascii"),
        "decide_seconds": decided,
        "refusal": refusal,
        "load_seconds": loaded,
    }
    return found, module


def _queries(
    torch_host: TorchHost,
    modules: dict[str, Module],
    database: Tables[pd.DataFrame],
    reference: dict[str, pd.DataFrame] | None,
) -> tuple[float, list[QueryRecord]]:
    """The time to copy the tables into the tenant's session, and each query's
    first run and warm time on the registered kernels."""
    found: list[QueryRecord] = []
    with torch_host.session(TENANT):
        before = time.perf_counter()
        tables = load(database)
        copied = time.perf_counter() - before
        ops = GatedOperators(torch_host, modules)
        for name, query in QUERIES.items():
            result, first, warm = _run(query, ops, tables)
            found.append(
                {
                    "query": name,
                    "first_seconds": first,
                    "warm": warm,
                    "result": {c: result[c].tolist() for c in result.columns},
                    "mismatch": None if reference is None else mismatch(reference[name], result),
                }
            )
        del tables, ops
    return copied, found


def _run(
    query: Callable[[Operators, TorchTables], pd.DataFrame], ops: Operators, tables: TorchTables
) -> tuple[pd.DataFrame, float, Timing]:
    """A query's first result and the time of its first run, and its warm time."""
    before = time.perf_counter()
    result = query(ops, tables)
    first = time.perf_counter() - before
    return result, first, host_time(lambda: query(ops, tables))


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
        "registration_seconds": REGISTRATION_SECONDS,
        "table_bytes": table_bytes(database),
    }
    reference: dict[str, pd.DataFrame] | None = None
    if scale <= REFERENCE_LIMIT:
        reference = {name: host(PANDAS[name](pd, database)) for name in QUERIES}
    pool = POOL_FACTOR * table_bytes(database) + POOL_MARGIN
    kernels: list[KernelRecord] = []
    queries: list[QueryRecord] = []
    started = time.perf_counter()
    with Executor(0, pool) as executor, TorchHost(executor) as torch_host:
        record["context_seconds"] = time.perf_counter() - started
        with executor.tenant(TENANT):
            executor.reserve(pool)
        modules: dict[str, Module] = {}
        for task, model in MODELS.items():
            found, module = _kernel(executor, task, model)
            kernels.append(found)
            if module is not None:
                modules[task] = module
        if len(modules) == len(MODELS):
            record["tables_seconds"], queries = _queries(torch_host, modules, database, reference)
        torch_host.release(TENANT)
        if executor.catalog.buffers:
            raise RuntimeError(
                f"{len(executor.catalog.buffers)} catalog buffers outlive the queries"
            )
    record["kernels"] = kernels
    record["queries"] = queries
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    refused = [k["kernel"] for k in kernels if k["refusal"] is not None]
    differ = [q["query"] for q in queries if q["mismatch"] is not None]
    if refused or differ:
        raise RuntimeError(f"refused kernels {refused}; results that differ {differ}")


if __name__ == "__main__":
    main()
