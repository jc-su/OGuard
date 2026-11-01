"""RQ4: what a query costs through OpGate, against direct launches and cuDF, at six scales.

    taskset -c <core> uv run python -m eval.timing eval.rq4_cost.query_overhead \\
        eval/records/query_overhead_<date>

At TPC-H scale factors 0.1, 0.3, 1, 3, 10, and 30 on one GPU, each run records:

- `pipelines`: Q6 and the revenue pipeline through the executor, with their
  allocations, fills, and step checks, on the static and the guarded path,
  against direct launches of the same verified kernels with the same
  parameters on buffers allocated beforehand (`eval.direct`).
- `cudf`: the same two queries written with cuDF's own operators on a copy of
  the same columns, outside OpGate, allocating from an RMM pool: how the
  engine computes them with its own kernels.
- `phases`: the revenue pipeline's phases on each path, each followed by a
  synchronization: the allocation steps, the filter's step (its check, the
  fill of its index output, and the filter), the host's read of the count,
  the scratch buffers' allocation, the upload of the sum's segments, the sum's
  step, the read of the total, and the frees; and the step check of the
  filter's step and of the sum's step alone.

The measurement protocol is `eval.timing`'s.
"""

import argparse
import json
import struct
import time
from pathlib import Path
from typing import TypedDict

import cudf
import rmm

from eval.direct import Direct, checked_kernels
from eval.timing import (
    REPEATS,
    WARMUP_SECONDS,
    Environment,
    Timing,
    environment,
    host_time,
    pinned_core,
    summary,
)
from eval.tpch.data import Lineitem, generate
from eval.tpch.operators import INDEX, Operators, register
from eval.tpch.pipelines import (
    Q6_PARAMETERS,
    THRESHOLD,
    Columns,
    Session,
    free,
    load,
    q6,
    q6_launch,
    revenue,
    scratch_buffers,
    select,
    select_launch,
    sum_modules,
    sum_step,
    unload,
    write_segments,
)
from eval.tpch.pipelines import Path as Route
from opgate.catalog import Fill, Step, TenantId
from opgate.executor import Executor
from opgate.executor.driver import call, driver

SCALES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
POOL = 16 << 30  # the lineitem columns and every buffer of the pipelines at scale factor 30
CUDF_POOL = 12 << 30  # cuDF's four columns and its queries' temporaries at scale factor 30
TENANT = TenantId(1)
PHASES = (
    "allocate",
    "filter step",
    "read count",
    "allocate scratch",
    "write segments",
    "sum step",
    "read total",
    "free",
    "check filter step",
    "check sum step",
)


class Pipelines(TypedDict):
    q6_opgate: Timing
    q6_direct: Timing
    revenue_static: Timing
    revenue_guarded: Timing
    revenue_direct: Timing


class Cudf(TypedDict):
    q6: Timing
    revenue: Timing


class Scale(TypedDict):
    scale: float
    rows: int
    selected: int
    pipelines: Pipelines
    cudf: Cudf
    phases: dict[str, dict[str, Timing]]  # by path, then by phase


class Record(TypedDict):
    environment: Environment
    core: int
    scales: list[Scale]


def cudf_queries(lineitem: Lineitem) -> Cudf:
    """Q6 and the revenue query with cuDF's own operators on the same columns,
    allocating from an RMM pool of `CUDF_POOL` bytes."""
    previous = rmm.mr.get_current_device_resource()
    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=CUDF_POOL,
            maximum_pool_size=CUDF_POOL,
        )
    )
    try:
        frame = cudf.DataFrame(
            {
                "shipdate": lineitem.shipdate,
                "discount": lineitem.discount,
                "quantity": lineitem.quantity,
                "price": lineitem.extendedprice,
            }
        )
        query = Q6_PARAMETERS

        def q6_cudf() -> float:
            keep = (
                (frame.shipdate >= query.date_lo)
                & (frame.shipdate < query.date_hi)
                & (frame.discount >= query.disc_lo)
                & (frame.discount <= query.disc_hi)
                & (frame.quantity < query.qty_max)
            )
            return float((frame.price[keep] * frame.discount[keep]).sum())

        def revenue_cudf() -> float:
            keep = frame.quantity < THRESHOLD
            return float((frame.price[keep] * (1 - frame.discount[keep])).sum())

        return {"q6": host_time(q6_cudf), "revenue": host_time(revenue_cudf)}
    finally:
        rmm.mr.set_current_device_resource(previous)


def pipelines(executor: Executor, s: Session, ops: Operators, t: Columns) -> Pipelines:
    """Q6 and the revenue pipeline through the executor against direct launches."""
    direct = Direct(executor)
    # Q6: a zeroed result and one launch
    with executor.tenant(s.tenant):
        result = executor.allocate(8)
    q6_step = Step(s.tenant, (q6_launch(ops, t, result, Q6_PARAMETERS),))
    (q6_kernel,) = checked_kernels(executor, q6_step)

    def q6_direct() -> None:
        call(driver.cuMemsetD8(direct.address(result), 0, 8))
        direct.launch(q6_kernel, ops.q6)
        direct.read(result, 8)

    # the revenue pipeline: the filter, the count, the segments, and the sum
    sel, count, rows = select(s, ops, t, Route.STATIC, THRESHOLD)
    scratch = scratch_buffers(s, rows)
    write_segments(s, scratch)
    select_step = Step(s.tenant, (select_launch(ops, t, sel, count, Route.STATIC, THRESHOLD),))
    (select_kernel,) = checked_kernels(executor, select_step)
    sum_kernels = checked_kernels(executor, sum_step(s, ops, t, sel, scratch, False))
    offsets, whole = scratch.segments()

    def revenue_direct() -> None:
        call(driver.cuMemsetD8(direct.address(count), 0, 4))
        direct.launch(select_kernel, ops.select_static)
        direct.read(count, 4)
        call(driver.cuMemcpyHtoD(direct.address(scratch.offsets), offsets, len(offsets)))
        call(driver.cuMemcpyHtoD(direct.address(scratch.whole), whole, len(whole)))
        for kernel, module in zip(sum_kernels, sum_modules(ops), strict=True):
            direct.launch(kernel, module)
        direct.read(scratch.total, 8)

    timings: Pipelines = {
        "q6_opgate": host_time(lambda: q6(s, ops, t, Q6_PARAMETERS)),
        "q6_direct": host_time(q6_direct),
        "revenue_static": host_time(lambda: revenue(s, ops, t, Route.STATIC, THRESHOLD, False)),
        "revenue_guarded": host_time(lambda: revenue(s, ops, t, Route.GUARDED, THRESHOLD, False)),
        "revenue_direct": host_time(revenue_direct),
    }
    free(s, result, sel, count, *scratch.buffers)
    return timings


def phases(
    executor: Executor, s: Session, ops: Operators, t: Columns, path: Route
) -> dict[str, Timing]:
    """The revenue pipeline's phases, as `revenue` performs them, each followed
    by a synchronization; and the step checks of its two launches' steps alone."""
    samples: dict[str, list[float]] = {phase: [] for phase in PHASES[:-2]}

    def run(record: bool) -> None:
        with executor.tenant(s.tenant):
            last = time.perf_counter()

            def mark(phase: str) -> None:
                nonlocal last
                executor.synchronize()
                now = time.perf_counter()
                if record:
                    samples[phase].append(now - last)
                last = now

            sel, count = s.memory.allocate(4 * t.rows), s.memory.allocate(4)
            mark("allocate")
            launch = select_launch(ops, t, sel, count, path, THRESHOLD)
            executor.execute(Step(s.tenant, (Fill(sel, INDEX, 0), Fill(count, INDEX, 0), launch)))
            mark("filter step")
            (rows,) = struct.unpack("<i", executor.read(s.tenant, count))
            mark("read count")
            scratch = scratch_buffers(s, rows)
            mark("allocate scratch")
            write_segments(s, scratch)
            mark("write segments")
            if not all(executor.execute(sum_step(s, ops, t, sel, scratch, path is Route.GUARDED))):
                raise RuntimeError("a guarded launch did not run")
            mark("sum step")
            executor.read(s.tenant, scratch.total)
            mark("read total")
            free(s, sel, count, *scratch.buffers)
            mark("free")

    until = time.perf_counter() + WARMUP_SECONDS
    runs = 0
    while runs < 2 or time.perf_counter() < until:
        run(False)
        runs += 1
    for _ in range(REPEATS):
        run(True)
    timings = {phase: summary(values) for phase, values in samples.items()}
    with executor.tenant(s.tenant):
        sel, count = s.memory.allocate(4 * t.rows), s.memory.allocate(4)
        launch = select_launch(ops, t, sel, count, path, THRESHOLD)
        filter_step = Step(s.tenant, (Fill(sel, INDEX, 0), Fill(count, INDEX, 0), launch))
        executor.execute(filter_step)
        (rows,) = struct.unpack("<i", executor.read(s.tenant, count))
        scratch = scratch_buffers(s, rows)
        write_segments(s, scratch)
        step = sum_step(s, ops, t, sel, scratch, path is Route.GUARDED)
        timings["check filter step"] = host_time(lambda: executor.catalog.check(filter_step))
        timings["check sum step"] = host_time(lambda: executor.catalog.check(step))
        free(s, sel, count, *scratch.buffers)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record: Record = {"environment": environment(), "core": pinned_core(), "scales": []}
    with Executor(0, POOL) as executor:
        with executor.tenant(TENANT):
            executor.reserve(POOL)
        s = Session(executor, TENANT, executor)
        ops = register(executor)
        for scale in SCALES:
            _, lineitem = generate(scale, seed=5)
            t = load(s, lineitem)
            sel, count, selected = select(s, ops, t, Route.STATIC, THRESHOLD)
            free(s, sel, count)
            record["scales"].append(
                {
                    "scale": scale,
                    "rows": t.rows,
                    "selected": selected,
                    "pipelines": pipelines(executor, s, ops, t),
                    "cudf": cudf_queries(lineitem),
                    "phases": {path.value: phases(executor, s, ops, t, path) for path in Route},
                }
            )
            unload(s, t)
            print(f"scale {scale}", flush=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
