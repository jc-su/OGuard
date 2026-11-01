"""Measure cold verification and same-process reuse on explicitly selected inputs.

Pin this experiment to exactly two otherwise available CPU cores. Each cold
sample runs in a fresh Python process and invokes the public verifier, including
its certificate proposal, checking, and diagnosis in one subprocess. Reuse
samples follow in that process. The two-core affinity is retained for direct
comparison with archived measurements of the former paired-process verifier.
This is a registration experiment; it does not measure GPU execution, data
loading, or query latency. Retain its raw record beside the input directories.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import platform
import random
import resource
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.rq4_cost.registration import Input, inputs, source_files
from opgate.gate import (
    CHECK_QUERY,
    TIMEOUT,
    TIMEOUT_REASON,
    Accepted,
    Refused,
    export_terms,
    formal_dir,
    verify,
    zeroed,
)
from opgate.hosts.service.audit import binding_record


def _decision(decision: Accepted | Refused) -> dict:
    if isinstance(decision, Accepted):
        return {"decision": "accepted", "reason": ""}
    return {
        "decision": "timeout" if decision.reason.startswith(TIMEOUT_REASON) else "refused",
        "reason": decision.reason,
    }


def _sample(item: Input, library: str, timeout: float, warm_repeats: int) -> dict:
    """One worker is used for exactly one cold sample and its reuse samples."""
    os.environ["OPGATE_FORMAL"] = library
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    decision = verify(zeroed(item.ptx, item.binding), [item.binding], timeout=timeout)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = _decision(decision)
    warm = []
    # A timeout is deliberately not cached by the verifier. Repeating it would
    # measure another cold check rather than the reuse cost this field names.
    if result["decision"] != "timeout":
        for _ in range(warm_repeats):
            started = time.perf_counter()
            repeated = verify(zeroed(item.ptx, item.binding), [item.binding], timeout=timeout)
            seconds = time.perf_counter() - started
            if repeated != decision:
                raise RuntimeError("the same-process decision changed during the measurement")
            warm.append({"seconds": seconds, **_decision(repeated)})
    return {
        **result,
        "worker_pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cold_seconds": elapsed,
        "child_user_seconds": after.ru_utime - before.ru_utime,
        "child_system_seconds": after.ru_stime - before.ru_stime,
        "largest_child_peak_rss_kib": after.ru_maxrss,
        "reuse": warm,
    }


def measure(
    output: Path,
    keys: list[str],
    *,
    repeats: int = 3,
    warm_repeats: int = 10,
    timeout: float = TIMEOUT,
    seed: int = 30,
    resume: bool = False,
) -> None:
    if repeats < 1 or warm_repeats < 1:
        raise ValueError("repeats and warm repeats must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("select at least one kernel, without duplicate keys")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) != 2:
        raise ValueError("pin the registration cost experiment to exactly two CPU cores")
    available = {"/".join(item.key): item for item in inputs()}
    unknown = set(keys) - available.keys()
    if unknown:
        raise ValueError(f"unknown corpus kernels: {sorted(unknown)}")
    chosen = [available[key] for key in keys]
    library = formal_dir()
    sources = source_files(library)
    sources["eval/rq4_cost/registration_cost.py"] = Path(__file__).resolve()
    order = [(key, repeat) for key in keys for repeat in range(repeats)]
    random.Random(seed).shuffle(order)
    provenance = {
        "kind": "isolated cold verification and same-process reuse",
        "scope": "registration only; no GPU work or query latency",
        "library": str(library),
        "query_suffix": CHECK_QUERY,
        "processes_per_verification": 1,
        "cores": cores,
        "python": platform.python_version(),
        "coqc": subprocess.run(
            ["coqc", "--version"], check=True, capture_output=True, text=True
        ).stdout,
        "kernels": keys,
        "repeats": repeats,
        "warm_repeats": warm_repeats,
        "timeout_seconds": timeout,
        "seed": seed,
        "order": [[key, repeat] for key, repeat in order],
        "inputs": [
            {
                "key": list(item.key),
                "verdict": item.verdict,
                "binding": binding_record(item.binding),
            }
            for item in chosen
        ],
    }
    with Checkpoint(output, provenance, resume=resume) as record:
        snapshot_inputs(output, "sources", sources, create=not resume)
        for item in chosen:
            key = "/".join(item.key)
            record.artifact(key, item.ptx, "ptx")
            record.artifact(key, export_terms(zeroed(item.ptx, item.binding), [item.binding]), "v")
        # A one-task child lifetime prevents earlier checks from warming the
        # verifier's process-local cache. Tasks are submitted strictly in order.
        with ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"), max_tasks_per_child=1
        ) as pool:
            for key, repeat in order:
                row_id = f"{key}/repeat-{repeat}"
                if record.get(row_id) is not None:
                    continue
                started = time.perf_counter()
                result = pool.submit(
                    _sample, available[key], str(library), timeout, warm_repeats
                ).result()
                roundtrip = time.perf_counter() - started
                record.add(
                    {
                        "id": row_id,
                        "kernel": key,
                        "repeat": repeat,
                        "fresh_worker_roundtrip_seconds": roundtrip,
                        "ptx_artifact": key,
                        "query_artifact": key,
                        **result,
                    }
                )
                print(f"{row_id}: {result['decision']}, {result['cold_seconds']:.3f} s", flush=True)
        snapshot_inputs(output, "sources", sources, create=False)
        current = {"/".join(item.key): item for item in inputs()}
        if any(current.get(key) != available[key] for key in keys):
            raise ValueError("selected corpus inputs changed during measurement; use a new record")
        record.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", action="append", required=True, dest="keys")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    measure(
        args.output,
        args.keys,
        repeats=args.repeats,
        warm_repeats=args.warm_repeats,
        timeout=args.timeout,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
