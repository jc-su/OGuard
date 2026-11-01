"""RQ4: how many tenants of one large table each configuration holds on one GPU.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq4_cost.density \\
        --output eval/records/density_<date>.json

At TPC-H scale factor 10, three configurations of `consolidation` set up 8 to
48 tenants, and every tenant runs one Q6 query as a prepared step, checked
against NumPy, while all of them stay resident:

* ``service``: one service holds one lineitem table, which a loader grants to
  every tenant; each tenant's client prepares its step and runs it through
  its mailbox.
* ``process``: each tenant has a process, a CUDA context, an executor, and its
  own copy of the table.
* ``mps``: the same processes under default CUDA MPS.

A cell records the peak device memory above the idle baseline and each
tenant's result, or the driver's out-of-memory failure that stopped a
tenant's setup, which is the measured outcome. Any other failure stops the run.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import secrets
import struct
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty
from typing import Literal

from eval.corpus.adapter import binding_of
from eval.rq4_cost.consolidation import (
    MODEL_PTX,
    TENANT_BYTES,
    Mps,
    Sampler,
    _checked,
    _device,
    _direct_q6,
    _direct_run,
    _environment,
    _memory,
    _pool,
    _read_inputs,
    _reference,
    _require_ok,
    _save_inputs,
    _scalars,
    _source_texts,
    _upload,
    _write_json,
)
from eval.service_plans import NODES, SCHEMAS, VERIFICATION_WORKERS
from eval.tpch.pipelines import Session, load
from opgate.catalog import TenantId
from opgate.executor import Executor
from opgate.executor.driver import DriverError
from opgate.frontends import Target
from opgate.hosts.service import Client, Limits, Principal, Service

Configuration = Literal["service", "process", "mps"]
CONFIGURATIONS: tuple[Configuration, ...] = ("service", "process", "mps")
TENANTS = (8, 16, 24, 32, 48)
SCALE = 10.0
MESSAGE = 1 << 30  # a column of the table in one write, as base64
SETUP_TIMEOUT = 1800.0
OUT_OF_MEMORY = "CUDA_ERROR_OUT_OF_MEMORY"


@dataclass(frozen=True)
class Report:
    """A tenant's outcome: its query's result, or why its setup stopped."""

    tenant: int
    result: float | None
    failure: str | None
    out_of_memory: bool


def _process_tenant(inputs: Path, tenant: int, results: mp.Queue[Report], done: Event) -> None:
    """A tenant with its own context and table: one checked Q6 use, then stay resident."""
    lineitem = _read_inputs(inputs)
    expected = _reference(lineitem)
    try:
        with Executor(0, _pool(len(lineitem.quantity))) as executor:
            with executor.tenant(TenantId(tenant)):
                executor.reserve(_pool(len(lineitem.quantity)))
            session = Session(executor, TenantId(tenant), executor)
            text = MODEL_PTX.read_text()
            module = executor.register(text, binding_of("sonnet", "q6_sum", "numba", text))
            table = load(session, lineitem)
            observed = _direct_q6(session, _direct_run(session, module, table))
            _checked(observed, expected, tenant)
            results.put(Report(tenant, observed, None, False))
            done.wait()
    except DriverError as error:
        results.put(Report(tenant, None, str(error), OUT_OF_MEMORY in str(error)))
    except BaseException:
        results.put(Report(tenant, None, traceback.format_exc(), False))


def _service_tenant(
    socket_path: str,
    token: str,
    tenant: int,
    handles: dict[str, str],
    rows: int,
    expected: float,
    results: mp.Queue[Report],
    done: Event,
) -> None:
    """A tenant of the shared service: prepare Q6 over the granted columns, one use."""
    try:
        with Client(socket_path, token, message_bytes=MESSAGE) as client:
            text = MODEL_PTX.read_text()
            (registration,) = client.register((text, {"kind": "tpch_q6_sum"}))
            operator = _require_ok(registration, "registration").text("operator")
            result = _require_ok(client.request("allocate", nbytes=8), "allocation").text("buffer")
            columns = ("shipdate", "discount", "quantity", "extendedprice")
            prepared = client.prepare(
                operator=operator,
                grid=[max(1, -(-rows // 256)), 1, 1],
                buffers=[*(handles[name] for name in columns), result],
                sizes=[rows],
                scalars=list(_scalars()),
                fills=[{"buffer": result, "element": {"width": 8, "signed": False}, "value": 0}],
                read=result,
            )
            with client.data_plane() as mailbox:
                (observed,) = struct.unpack("<d", mailbox.run(prepared, _scalars()))
            _checked(observed, expected, tenant)
            results.put(Report(tenant, observed, None, False))
            done.wait()
    except BaseException:
        results.put(Report(tenant, None, traceback.format_exc(), False))


def _service_cell(inputs: Path, tenants: int, results: mp.Queue[Report], done: Event) -> None:
    """One service with a loader that grants one table to `tenants` client processes."""
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    tokens = {name: secrets.token_hex(16) for name in ("loader", *map(str, range(tenants)))}
    principals = (
        Principal(
            "loader",
            tokens["loader"],
            TenantId(10_000),
            administrator=True,
            reservation=_pool(rows),
        ),
        *(Principal(f"t{t}", tokens[str(t)], TenantId(t + 1)) for t in range(tenants)),
    )
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="opgate-density-") as temporary:
        socket_path = Path(temporary) / "service.sock"
        limits = Limits(
            verification_workers=VERIFICATION_WORKERS,
            message_bytes=MESSAGE,
            tenant_bytes=TENANT_BYTES,
            chunk_bytes=TENANT_BYTES,
            connections=tenants + 4,
        )
        with (
            Service(
                socket_path,
                principals,
                device=0,
                pool_bytes=_pool(rows) + tenants * TENANT_BYTES,
                target=Target(8, 6),
                schemas=SCHEMAS,
                nodes=NODES,
                limits=limits,
            ),
            Client(socket_path, tokens["loader"], message_bytes=MESSAGE) as loader,
        ):
            handles = {
                name: _upload(loader, getattr(lineitem, name))
                for name in ("shipdate", "discount", "quantity", "extendedprice", "orderkey")
            }
            for t in range(tenants):
                for handle in handles.values():
                    granted = loader.request("grant_read", buffer=handle, principal=f"t{t}")
                    _require_ok(granted, "grant")
            clients = [
                context.Process(
                    target=_service_tenant,
                    args=(
                        str(socket_path),
                        tokens[str(t)],
                        t + 1,
                        handles,
                        rows,
                        _reference(lineitem),
                        results,
                        done,
                    ),
                )
                for t in range(tenants)
            ]
            for client in clients:
                client.start()
            done.wait()
            for client in clients:
                client.join()


def _cell(
    configuration: Configuration, tenants: int, inputs: Path, device: str
) -> dict[str, object]:
    """A cell on the GPU `device` names; under MPS, `CUDA_VISIBLE_DEVICES` names the
    clients' view of it, so the caller reads `device` before the daemon starts."""
    context = mp.get_context("spawn")
    results: mp.Queue[Report] = context.Queue()
    done = context.Event()
    baseline = _memory(device)
    with Sampler(device) as sampler:
        if configuration == "service":
            workers = [context.Process(target=_service_cell, args=(inputs, tenants, results, done))]
        else:
            workers = [
                context.Process(target=_process_tenant, args=(inputs, t + 1, results, done))
                for t in range(tenants)
            ]
        for worker in workers:
            worker.start()
        reports: list[Report] = []
        deadline = time.monotonic() + SETUP_TIMEOUT
        while len(reports) < tenants:
            try:
                reports.append(results.get(timeout=1.0))
            except Empty:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{configuration} with {tenants} tenants did not set up"
                    ) from None
                if all(not worker.is_alive() for worker in workers):
                    raise RuntimeError(
                        f"{configuration} with {tenants} tenants stopped early"
                    ) from None
        time.sleep(1.0)  # every tenant is resident: let the sampler see the peak
        done.set()
        for worker in workers:
            worker.join(timeout=120)
            if worker.is_alive():
                raise TimeoutError(f"a {configuration} worker did not exit")
    other = [r for r in reports if r.failure is not None and not r.out_of_memory]
    if other:
        raise RuntimeError(f"{configuration} with {tenants} tenants failed: {other[0].failure}")
    full = [r for r in reports if r.out_of_memory]
    return {
        "configuration": configuration,
        "scale": SCALE,
        "tenants": tenants,
        "fits": not full,
        "tenants_out_of_memory": len(full),
        "failure": full[0].failure if full else None,
        "results": sorted((r.tenant, r.result) for r in reports if r.result is not None),
        "memory_mib": sampler.peak - baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    environment = _environment()
    sources = {**_source_texts(), "eval/rq4_cost/density.py": Path(__file__).read_text()}
    inputs = args.output.with_suffix(args.output.suffix + ".inputs") / "tables"
    _save_inputs(inputs, (SCALE,), create=True)
    started = datetime.now(UTC).isoformat()
    cells: list[dict[str, object]] = []
    device = _device()
    for configuration in CONFIGURATIONS:
        for tenants in TENANTS:
            if configuration == "mps":
                with Mps():
                    cell = _cell(configuration, tenants, inputs / f"sf{SCALE:g}", device)
            else:
                cell = _cell(configuration, tenants, inputs / f"sf{SCALE:g}", device)
            print(json.dumps({k: v for k, v in cell.items() if k != "results"}), flush=True)
            cells.append(cell)
            if not cell["fits"]:
                break  # more tenants of this configuration do not fit either
    if {**_source_texts(), "eval/rq4_cost/density.py": Path(__file__).read_text()} != sources:
        raise RuntimeError("an evaluation source changed during the run")
    _write_json(
        args.output,
        {
            "schema": 1,
            "experiment": "tenants of one TPC-H table that each configuration holds on one GPU",
            "complete": True,
            "started_at_utc": started,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "environment": environment,
            "sources": sources,
            "scale": SCALE,
            "tenant_counts": list(TENANTS),
            "configurations": list(CONFIGURATIONS),
            "cells": cells,
        },
    )


if __name__ == "__main__":
    main()
