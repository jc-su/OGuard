"""RQ3: the static index path, whose range the gate proves, against the guarded path, which checks it at run time.

    taskset -c <core> uv run python -m eval.timing eval.rq3_pipelines.index_path \\
        eval/records/index_path_<date>

At TPC-H scale factors 0.1, 1, and 10, each run times the revenue pipeline's
sum step (two gathers of the selected rows, the projection, and the two-level
sum) for filters that keep the lines with a quantity below 2, 6, 13, 26, and
51, about 2, 10, 24, 50, and 100 percent of them:

- on the static path, where the filter's output is an index output that the
  gate proves in range, so the gathers are plain launches;
- on the guarded path, where the same output is raw and each gather is a
  guarded launch whose index input a trusted kernel checks when it runs;
- eagerly and as CUDA graph replays, and the step check of each path's step;
- against direct launches of the same kernels (`eval.direct`).

The filter's output holds one row id per line, so the guarded check reads as
many elements at every selectivity. The measurement protocol is `eval.timing`'s.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from eval.direct import Direct, checked_kernels
from eval.timing import Environment, Timing, device_time, environment, host_time, pinned_core
from eval.tpch.data import generate
from eval.tpch.operators import Operators, register
from eval.tpch.pipelines import (
    Columns,
    Session,
    free,
    load,
    scratch_buffers,
    select,
    sum_modules,
    sum_step,
    unload,
    write_segments,
)
from eval.tpch.pipelines import Path as Route
from opgate.catalog import Step, TenantId
from opgate.executor import Executor

SCALES = (0.1, 1.0, 10.0)
THRESHOLDS = (2.0, 6.0, 13.0, 26.0, 51.0)
POOL = 12 << 30  # the lineitem columns and both paths' buffers at scale factor 10
TENANT = TenantId(1)


class Filter(TypedDict):
    threshold: float
    selected: int
    check_step_static: Timing
    check_step_guarded: Timing
    eager_static: Timing
    eager_guarded: Timing
    graph_static: Timing
    graph_guarded: Timing
    direct: Timing
    direct_device: Timing


class Scale(TypedDict):
    scale: float
    rows: int
    filters: list[Filter]


class Record(TypedDict):
    environment: Environment
    core: int
    scales: list[Scale]


def paths(executor: Executor, s: Session, ops: Operators, t: Columns, threshold: float) -> Filter:
    """The sum step over the lines with `quantity < threshold` on the static and
    the guarded path, eagerly and as graph replays, against direct launches of
    its kernels."""
    with executor.tenant(s.tenant):
        direct = Direct(executor)
        static_sel, static_count, rows = select(s, ops, t, Route.STATIC, threshold)
        raw_sel, raw_count, raw_rows = select(s, ops, t, Route.GUARDED, threshold)
        if raw_rows != rows:
            raise RuntimeError(f"the two filters selected {rows} and {raw_rows} rows")
        scratch = scratch_buffers(s, rows)
        write_segments(s, scratch)
        static = sum_step(s, ops, t, static_sel, scratch, False)
        guarded = sum_step(s, ops, t, raw_sel, scratch, True)
        kernels_static = checked_kernels(executor, static)

        def eager(step: Step) -> Callable[[], None]:
            def run() -> None:
                if not all(executor.execute(step)):
                    raise RuntimeError("a guarded kernel did not run")
                executor.synchronize()

            return run

        def replays(step: Step) -> Timing:
            graph = executor.capture(step)
            executor.ran(graph)

            def run() -> None:
                executor.replay(graph, step)
                if not all(executor.ran(graph)):
                    raise RuntimeError("a guarded kernel did not run")

            timing = host_time(run)
            executor.release(graph)
            return timing

        def launch_direct() -> None:
            for kernel, module in zip(kernels_static, sum_modules(ops), strict=True):
                direct.launch(kernel, module)

        def run_direct() -> None:
            launch_direct()
            executor.synchronize()

        timings: Filter = {
            "threshold": threshold,
            "selected": rows,
            "check_step_static": host_time(lambda: executor.catalog.check(static), 1000),
            "check_step_guarded": host_time(lambda: executor.catalog.check(guarded), 1000),
            "eager_static": host_time(eager(static)),
            "eager_guarded": host_time(eager(guarded)),
            "graph_static": replays(static),
            "graph_guarded": replays(guarded),
            "direct": host_time(run_direct),
            "direct_device": device_time(launch_direct, None),
        }
        free(s, static_sel, static_count, raw_sel, raw_count, *scratch.buffers)
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
            filters = [paths(executor, s, ops, t, threshold) for threshold in THRESHOLDS]
            record["scales"].append({"scale": scale, "rows": t.rows, "filters": filters})
            unload(s, t)
            print(f"scale {scale}", flush=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
