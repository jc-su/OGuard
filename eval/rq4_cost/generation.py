"""RQ4: the toolchain's time to compile each corpus kernel from its source to PTX.

    taskset -c <cores> uv run python -m eval.rq4_cost.generation \\
        --output eval/records/generation_<date>.json [--workers N]

Each CUDA C++ and Triton kernel of the corpus (`eval.rq4_cost.registration.inputs`)
is compiled again from the model's source by its front end
(`eval.corpus.sources.operator_of`: nvcc at NVVM level 1, or Triton with the
module's types, constants, and warps), for sm_86, in a fresh process with an
empty Triton cache, so that no compile reuses another's work. `--workers`
processes run at once on the run's cores. The time is the front end's call,
after the process has imported the toolchains. These two front ends compile
without a GPU; numba-cuda's typing of some kernels initializes the CUDA
driver, so its kernels are left to a run on a machine with a device.

The record keeps each compile's PTX text and whether its bytes equal the
corpus's text, which the gate checks. nvcc names internal functions after a
temporary path and Triton's debug lines name the source's path, so a text can
differ from the corpus's in these names.
"""

import argparse
import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

from eval import ROOT
from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.corpus.adapter import directory_of
from eval.corpus.run import TOOLCHAINS, Key, kernels
from eval.corpus.sources import SOURCES, TARGET, operator_of

__all__ = [
    "COMPILED",
    "KIND",
    "WORKERS",
    "compile_seconds",
    "corpus_texts",
    "main",
    "measure",
    "source_files",
]

WORKERS = 16
KIND = "front-end compile of each CUDA C++ and Triton corpus kernel in a fresh process"
COMPILED = ("nvcc", "triton")  # the toolchains whose front ends compile without a GPU


def corpus_texts() -> dict[Key, bytes]:
    """The PTX of each corpus kernel of the toolchains `COMPILED`, as the corpus
    holds it and the registration record checks it
    (`eval.rq4_cost.registration.inputs`)."""
    return {
        key: (directory_of(key[0], key[1]) / TOOLCHAINS[key[2]]).read_bytes()
        for key in kernels(set())
        if key[2] in COMPILED
    }


def source_files(keys: list[Key]) -> dict[str, Path]:
    """The experiment's code, the front ends, and each kernel's source."""
    files = {str(p.relative_to(ROOT)): p for p in (ROOT / "opgate" / "frontends").glob("*.py")}
    for name in (
        "eval/rq4_cost/generation.py",
        "eval/corpus/sources.py",
        "eval/corpus/adapter.py",
        "eval/corpus/run.py",
    ):
        files[name] = ROOT / name
    for model, task, toolchain in keys:
        path = directory_of(model, task) / SOURCES[toolchain]
        files[str(path.relative_to(ROOT))] = path
    return files


def compile_seconds(key: Key) -> tuple[str, float]:
    """The PTX text of `key`'s source and the front end's time to compile it,
    with an empty Triton cache of its own."""
    with tempfile.TemporaryDirectory(prefix="opgate-generation-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        started = time.perf_counter()
        operator = operator_of(*key)
        seconds = time.perf_counter() - started
    return operator.text, seconds


def measure(output: Path, *, resume: bool, workers: int) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < workers:
        raise ValueError(f"{workers} workers need at least {workers} cores")
    captured = corpus_texts()
    keys = sorted(captured)
    provenance: dict[str, object] = {
        "kind": KIND,
        "cores": cores,
        "workers": workers,
        "target": [TARGET.major, TARGET.minor],
        "versions": {p: version(p) for p in ("numba-cuda", "triton", "cuda-bindings")},
        "kernels": ["/".join(k) for k in keys],
    }
    with Checkpoint(output, provenance, resume=resume) as record:
        snapshot_inputs(output, "sources", source_files(keys), create=not resume)
        pending = [k for k in keys if record.get("/".join(k)) is None]
        # one fresh process for each compile, so that no compile warms another
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1
        ) as pool:
            futures = {pool.submit(compile_seconds, key): key for key in pending}
            for future in as_completed(futures):
                key = futures[future]
                ptx, seconds = future.result()
                name = "/".join(key)
                record.artifact(name, ptx, "ptx")
                record.add(
                    {
                        "id": name,
                        "seconds": seconds,
                        "ptx_artifact": name,
                        "same_as_corpus": ptx.encode("ascii") == captured[key],
                    }
                )
                print(f"{name}: {seconds:.2f} s", flush=True)
        snapshot_inputs(output, "sources", source_files(keys), create=False)
        if corpus_texts() != captured:
            raise ValueError("corpus inputs changed during measurement; use a new record")
        record.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    measure(args.output, resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
