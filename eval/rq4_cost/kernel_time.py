"""RQ4: an accepted operator's kernel time: NVVM optimization level 1, the front ends' setting, against level 3.

    taskset -c <core> uv run python -m eval.timing eval.rq4_cost.kernel_time \\
        eval/records/kernel_time_<date>

An accepted operator runs unchanged, so its kernel time is the time of the same
PTX without OpGate; the one difference from a default build is that the front
ends compile at NVVM level 1 (design section 5.9). For the pipelines' four
corpus operators on one million rows, the device time of the level-1 kernel
against the level-3 kernel that the corpus records, launched directly with the
same checked parameters. The filter is a repair round's revision, and the
corpus records level-3 PTX only of first kernels, so it is not among them. The
measurement protocol is `eval.timing`'s.
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from eval.corpus.adapter import directory_of
from eval.corpus.workloads import corpus_kernel, prepare, workload
from eval.direct import Direct, checked_kernels
from eval.timing import Environment, Timing, device_time, environment, pinned_core
from eval.tpch.operators import MODEL
from opgate.catalog import Step, TenantId
from opgate.executor import Executor
from opgate.executor.driver import call, driver, parameter_sizes
from opgate.frontends import entry_kernels

TASKS = ("gather", "projection", "q6_sum", "segmented_sum")
ROWS = 1_000_000
POOL = 3 << 30
TENANT = TenantId(1)


class Operator(TypedDict):
    task: str
    level1: Timing
    level3: Timing


class Record(TypedDict):
    environment: Environment
    core: int
    rows: int
    operators: list[Operator]


def _levels(executor: Executor, task: str, rng: np.random.Generator) -> Operator:
    """The device time of `task`'s level-1 kernel and of its level-3 kernel, on
    the same checked parameters."""
    direct = Direct(executor)
    loaded = corpus_kernel(executor, MODEL, task, "numba")
    prepared = prepare(executor, TENANT, loaded, workload(task, ROWS, rng))
    (kernel,) = checked_kernels(executor, Step(TENANT, (prepared.launch,)))
    text = (directory_of(MODEL, task) / "numba.ptx").read_text()
    # the level-3 text names the kernel by another mangled name
    (name,) = entry_kernels(text)
    (handle,) = call(driver.cuModuleLoadData(text.encode("ascii") + b"\0"))
    (function,) = call(driver.cuModuleGetFunction(handle, name.encode("ascii")))
    if parameter_sizes(function) != prepared.module.widths:
        raise RuntimeError(f"{task}: the two levels' parameters differ")
    level1 = device_time(lambda: direct.launch(kernel, prepared.module), None)
    level3 = device_time(lambda: direct.launch(kernel, prepared.module, function), None)
    call(driver.cuModuleUnload(handle))
    with executor.tenant(TENANT):
        for buffer in prepared.buffers:
            executor.free(buffer)
    return {"task": task, "level1": level1, "level3": level3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(3)
    record: Record = {
        "environment": environment(),
        "core": pinned_core(),
        "rows": ROWS,
        "operators": [],
    }
    with Executor(0, POOL) as executor:
        with executor.tenant(TENANT):
            executor.reserve(POOL)
        for task in TASKS:
            record["operators"].append(_levels(executor, task, rng))
            print(task, flush=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
