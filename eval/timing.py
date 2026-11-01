"""The measurement protocol of the timed experiments (design section 9.3).

A timed experiment runs on an otherwise idle GPU, pinned with `taskset -c
<core>` to one core of the GPU's NUMA node: a host time depends on the clock
of the core that runs the host's work and on the process moving between
cores, so `pinned_core` refuses a process that may run on more than one core.

Each measurement repeats its action for `WARMUP_SECONDS`, and at least once,
before it measures (warm up by time, not by a count), and then records
`REPEATS` samples: their median, minimum, and maximum. A device time stops
earlier, after `MIN_SAMPLES` samples, once its samples add up to
`SAMPLE_SECONDS`, so a launch that takes seconds is timed in minutes; a launch
shorter than `SAMPLE_SECONDS / REPEATS` gets all `REPEATS` samples. A host time runs from
before the action to after the device finishes it; a device time is between
two CUDA events on the launch stream.

An experiment's record is `RUNS` runs, each in a fresh process, or the
number that `--runs` gives, as the guard-cost experiment's three:

    taskset -c <core> uv run python -m eval.timing [--runs N] <module> eval/records/<name>_<date> [options]

runs `python -m <module> --output <directory>/run<i>.json [options]` for each
run, and the exporter requires all of them.
"""

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import TypedDict

import numpy as np

from opgate.executor.driver import call, driver

REPEATS = 20
MIN_SAMPLES = 3
SAMPLE_SECONDS = 5.0
WARMUP_SECONDS = 1.0
RUNS = 5


class Timing(TypedDict):
    """Samples of one measurement in microseconds."""

    median_us: float
    min_us: float
    max_us: float
    samples: int


class Environment(TypedDict):
    """The machine and the software versions of a record."""

    gpu: str
    driver_api: str
    python: str
    numpy: str
    coqc: str
    date: str


def summary(samples: list[float]) -> Timing:
    """Seconds as microseconds: the median, minimum, and maximum."""
    return {
        "median_us": statistics.median(samples) * 1e6,
        "min_us": min(samples) * 1e6,
        "max_us": max(samples) * 1e6,
        "samples": len(samples),
    }


def warm(action: Callable[[], object]) -> None:
    """Run `action` for `WARMUP_SECONDS`, and at least once."""
    until = time.perf_counter() + WARMUP_SECONDS
    runs = 0
    while runs < 1 or time.perf_counter() < until:
        action()
        runs += 1


def host_time(action: Callable[[], object], repeats: int = REPEATS) -> Timing:
    """The host time of `action`, which returns after the device finishes it."""
    warm(action)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        samples.append(time.perf_counter() - start)
    return summary(samples)


def device_time(
    launch: Callable[[], None], reset: Callable[[], None] | None, repeats: int = REPEATS
) -> Timing:
    """The device time of `launch` on the launch stream, between two events.
    `reset`, for a kernel whose launch changes what the next one does, as a
    hash table that fills, restores its buffers before each launch, outside
    the events; None for a kernel whose launches are alike."""
    (start,) = call(driver.cuEventCreate(0))
    (end,) = call(driver.cuEventCreate(0))

    def restore() -> None:
        if reset is not None:
            reset()
            call(driver.cuCtxSynchronize())

    def once() -> None:
        restore()
        launch()
        call(driver.cuCtxSynchronize())

    warm(once)
    samples: list[float] = []
    while len(samples) < repeats and (len(samples) < MIN_SAMPLES or sum(samples) < SAMPLE_SECONDS):
        restore()
        call(driver.cuEventRecord(start, 0))
        launch()
        call(driver.cuEventRecord(end, 0))
        call(driver.cuEventSynchronize(end))
        (milliseconds,) = call(driver.cuEventElapsedTime(start, end))
        if not isinstance(milliseconds, float):
            raise RuntimeError(f"the driver returned {milliseconds!r}, not a time")
        samples.append(milliseconds / 1e3)
    call(driver.cuEventDestroy(start))
    call(driver.cuEventDestroy(end))
    return summary(samples)


def pinned_core() -> int:
    """The one CPU core this process may run on."""
    cores = os.sched_getaffinity(0)
    if len(cores) != 1:
        raise RuntimeError(
            f"the process may run on {len(cores)} cores; "
            "pin it to one core of the GPU's NUMA node with taskset -c <core>"
        )
    (core,) = cores
    return core


def environment() -> Environment:
    """The GPU, the driver, and the versions that a record's numbers depend on."""
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (name,) = call(driver.cuDeviceGetName(64, device))
    (version,) = call(driver.cuDriverGetVersion())
    coqc = subprocess.run(["coqc", "--version"], check=True, capture_output=True, text=True)
    if not isinstance(name, bytes):
        raise RuntimeError(f"the driver returned {name!r}, not a name")
    return {
        "gpu": name.split(b"\0")[0].decode(),
        "driver_api": str(version),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "coqc": coqc.stdout.splitlines()[0],
        "date": date.today().isoformat(),
    }


def repeat(module: str, directory: Path, count: int, options: list[str]) -> None:
    """Run `count` runs of the experiment `module`, each in a fresh process."""
    pinned_core()
    directory.mkdir(parents=True, exist_ok=False)
    for run in range(1, count + 1):
        output = directory / f"run{run}.json"
        command = [sys.executable, "-m", module, "--output", str(output), *options]
        print(f"run {run} of {count}: {' '.join(command[1:])}", flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment's runs in fresh processes.")
    parser.add_argument("--runs", type=int, default=RUNS, help="the number of runs")
    parser.add_argument("module", help="the experiment, for example eval.rq4_cost.step_check")
    parser.add_argument("directory", type=Path, help="the record's directory, which must not exist")
    parser.add_argument("options", nargs=argparse.REMAINDER, help="options of the experiment")
    args = parser.parse_args()
    repeat(args.module, args.directory, args.runs, args.options)


if __name__ == "__main__":
    main()
