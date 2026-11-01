"""RQ4: the catalog's step check against the number of live buffers.

    taskset -c <core> uv run python -m eval.timing eval.rq4_cost.step_check \\
        eval/records/step_check_<date>

`check_step` of the revenue pipeline's sum step (five launches over the
lineitem columns at TPC-H scale factor 1 and seven scratch buffers) as the
catalog holds 16 to 2,048 live buffers, powers of 2: the pipeline's 14
buffers and more live buffers of 256 bytes. The check is
the Python transcription of `formal/CatalogStep.v`. The measurement protocol
is `eval.timing`'s.
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

from eval.timing import Environment, Timing, environment, host_time, pinned_core
from eval.tpch.data import generate
from eval.tpch.operators import register
from eval.tpch.pipelines import (
    THRESHOLD,
    Session,
    free,
    load,
    scratch_buffers,
    select,
    sum_step,
    unload,
    write_segments,
)
from eval.tpch.pipelines import Path as Route
from opgate.catalog import BufferId, TenantId
from opgate.executor import Executor

SCALE = 1.0
BUFFERS = (16, 32, 64, 128, 256, 512, 1024, 2048)  # the live buffers of each point
POOL = 3 << 30
TENANT = TenantId(1)


class Point(TypedDict):
    buffers: int  # the live buffers of the catalog
    check_step: Timing


class Record(TypedDict):
    environment: Environment
    core: int
    points: list[Point]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record: Record = {"environment": environment(), "core": pinned_core(), "points": []}
    with Executor(0, POOL) as executor:
        with executor.tenant(TENANT):
            executor.reserve(POOL)
        s = Session(executor, TENANT, executor)
        ops = register(executor)
        _, lineitem = generate(SCALE, seed=5)
        t = load(s, lineitem)
        sel, count, rows = select(s, ops, t, Route.STATIC, THRESHOLD)
        scratch = scratch_buffers(s, rows)
        write_segments(s, scratch)
        step = sum_step(s, ops, t, sel, scratch, False)
        extra: list[BufferId] = []
        for buffers in BUFFERS:
            if len(executor.catalog.buffers) > buffers:
                raise RuntimeError(
                    f"the catalog holds {len(executor.catalog.buffers)} live buffers, "
                    f"more than {buffers}"
                )
            with executor.tenant(TENANT):
                while len(executor.catalog.buffers) < buffers:
                    extra.append(executor.allocate(256))
            timing = host_time(lambda: executor.catalog.check(step), repeats=1000)
            record["points"].append(
                {"buffers": len(executor.catalog.buffers), "check_step": timing}
            )
        free(s, sel, count, *scratch.buffers, *extra)
        unload(s, t)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
