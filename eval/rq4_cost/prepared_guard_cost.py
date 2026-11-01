"""Time an actual guarded operator through RPC and the prepared mailbox.

This diagnostic uses the archived Sonnet as-of join kernel on ordinary
in-bounds data. Both paths execute and return the complete output column,
which is compared byte for byte with an independent CPU reference.
All requests, inputs, responses, registration inputs and timing samples are kept.
"""

import argparse
import base64
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.checkpoint import _publish, read_artifact, snapshot_inputs
from eval.corpus.adapter import Role, directory_of, read_raw, read_roles
from eval.corpus.heldout_workloads import reference
from eval.corpus.workloads import workload
from eval.rq4_cost.guard_cost import source_files
from eval.service_plans import VERIFICATION_WORKERS
from eval.timing import environment, pinned_core
from opgate.binding import Binding
from opgate.catalog import TenantId
from opgate.frontends import Target
from opgate.gate import export_terms, zeroed
from opgate.hosts.service import Client, Fields, LaunchRequest, Limits, Principal, Service
from opgate.hosts.service.audit import parse_binding

KEY = "sonnet/asof_join/nvcc"
TOKEN = "benchmark-tenant-token-0123456789"


@dataclass(frozen=True)
class Plan:
    binding: Binding
    kind: str = "frozen_asof_join"

    def plan(self, text: str, fields: Fields) -> "Plan":
        fields.exact("kind")
        return self

    def launch(self, request: Fields) -> LaunchRequest:
        request.exact("op", "operator", "left", "right", "out", "n", "r", "tolerance")
        return LaunchRequest(
            (max(1, -(-request.integer("n") // self.binding.block[0])), 1, 1),
            tuple(request.text(k) for k in ("left", "right", "out")),
            (request.integer("n"), request.integer("r")),
            (request.integer("tolerance"),),
            False,
        )


def measure(registration: Path, output: Path, *, rows: int = 257, samples: int = 200) -> None:
    if output.exists() or rows <= 0 or samples <= 0:
        raise ValueError("use a new output record and positive rows and samples")
    core = pinned_core()
    (entry,) = [
        e
        for e in json.loads(registration.read_text())["results"]
        if e["id"] == KEY and e["verdict"] == "safe"
    ]
    text = read_artifact(registration, KEY, "ptx")
    binding = parse_binding(entry["binding"])
    if read_artifact(registration, KEY, "v") != export_terms(zeroed(text, binding), (binding,)):
        raise ValueError("archived input differs from its query")
    sources = source_files() | {"eval/rq4_cost/prepared_guard_cost.py": Path(__file__)}
    snapshot_inputs(output, "sources", sources, create=True)
    snapshot_inputs(output, "registration", {"registration.json": registration}, create=True)
    w = workload("asof_join", rows, np.random.default_rng(7))
    roles = read_roles("sonnet", "asof_join")
    raws = read_raw(directory_of("sonnet", "asof_join") / "signature.json")
    record = {
        "complete": False,
        "environment": environment(),
        "core": core,
        "kernel": KEY,
        "rows": rows,
        "samples": samples,
        "original_ptx": text,
        "binding": entry["binding"],
        "inputs": {
            name: {
                "dtype": array.dtype.str,
                "bytes": base64.b64encode(array.tobytes()).decode("ascii"),
            }
            for name, array in w.arrays.items()
        },
        "scalars": w.scalars,
        "sizes": w.sizes,
    }
    _publish(output, (json.dumps(record, indent=1) + "\n").encode())
    principal = Principal("tenant", TOKEN, TenantId(1), administrator=True)
    with tempfile.TemporaryDirectory(prefix="prepared-cost-") as directory:
        socket = Path(directory) / "service.sock"
        with Service(
            socket,
            (principal,),
            device=0,
            pool_bytes=1 << 26,
            target=Target(8, 6),
            schemas=(Plan(binding),),
            nodes=(),
            limits=Limits(
                verification_workers=VERIFICATION_WORKERS, tenant_bytes=1 << 24, chunk_bytes=1 << 20
            ),
        ):
            with Client(socket, TOKEN) as client:
                started = time.perf_counter()
                (registration_response,) = client.register((text, {"kind": "frozen_asof_join"}))
                record["cold_verification_seconds"] = time.perf_counter() - started
                operator = registration_response.text("operator")
                if not registration_response.values("checks"):
                    raise ValueError("the selected kernel did not require runtime checks")
                handles = {}
                for raw in raws:
                    if roles.arguments[raw.name] in (Role.SIZE, Role.SCALAR):
                        continue
                    array = w.arrays[raw.name]
                    handles[raw.name] = client.request(
                        "allocate", nbytes=max(1, array.nbytes)
                    ).text("buffer")
                    client.request(
                        "write",
                        buffer=handles[raw.name],
                        data=base64.b64encode(array.tobytes()).decode("ascii"),
                        element=None,
                    )
                tolerance = int(w.scalars["tolerance"])
                request = {
                    "operator": operator,
                    **handles,
                    "n": rows,
                    "r": w.sizes["r"],
                    "tolerance": tolerance,
                }
                timings, responses, outputs = {}, {}, {}
                expected = reference("asof_join", w.arrays, w.scalars, w.sizes)["out"].tobytes()

                def collect():
                    raw = base64.b64decode(
                        client.request("read", buffer=handles["out"]).text("data"), validate=True
                    )
                    if raw != expected:
                        raise ValueError("the result differs from the CPU reference bytes")
                    return base64.b64encode(raw).decode("ascii")

                rpc_responses = []
                for index in range(samples + 1):
                    started = time.perf_counter()
                    result = client.request("run_operator", **request)
                    raw = base64.b64decode(
                        client.request("read", buffer=handles["out"]).text("data"), validate=True
                    )
                    elapsed = time.perf_counter() - started
                    if result.values("ran") != [True] or raw != expected:
                        record["failure"] = {
                            "path": "rpc",
                            "observed": base64.b64encode(raw).decode("ascii"),
                            "expected": base64.b64encode(expected).decode("ascii"),
                        }
                        _publish(output, (json.dumps(record, indent=1) + "\n").encode())
                        raise ValueError("RPC returned an incorrect output")
                    if index:
                        timings.setdefault("rpc", []).append(elapsed * 1e6)
                        rpc_responses.append(base64.b64encode(raw).decode("ascii"))
                responses["rpc"] = rpc_responses
                outputs["rpc"] = collect()
                prepared_request = dict(
                    operator=operator,
                    grid=[max(1, -(-rows // binding.block[0])), 1, 1],
                    buffers=list(handles.values()),
                    sizes=[w.sizes[name] for name in roles.symbols],
                    scalars=[tolerance],
                    fills=[],
                    read=handles["out"],
                )
                prepared = client.prepare(**prepared_request)
                with client.data_plane() as mailbox:
                    for index in range(samples + 1):
                        started = time.perf_counter()
                        raw = mailbox.run(prepared, (tolerance,))
                        elapsed = time.perf_counter() - started
                        if raw != expected:
                            record["failure"] = {
                                "path": "prepared",
                                "observed": base64.b64encode(raw).decode("ascii"),
                                "expected": base64.b64encode(expected).decode("ascii"),
                            }
                            _publish(output, (json.dumps(record, indent=1) + "\n").encode())
                            raise ValueError("prepared execution returned an incorrect output")
                        if index:
                            timings.setdefault("prepared", []).append(elapsed * 1e6)
                            responses.setdefault("prepared", []).append(
                                base64.b64encode(raw).decode("ascii")
                            )
                outputs["prepared"] = collect()
                record |= {
                    "requests": {"rpc": request, "prepared": prepared_request},
                    "timing_us_samples": timings,
                    "responses": responses,
                    "outputs": outputs,
                    "audit": client.audit(),
                    "median_us": {
                        name: statistics.median(values) for name, values in timings.items()
                    },
                }
            with Client(socket, TOKEN) as client:
                telemetry = client.request("telemetry").fields("telemetry")
                if (
                    telemetry.integer("live_buffers")
                    or telemetry.integer("failed_checks")
                    or telemetry.integer("fatal_errors")
                ):
                    raise ValueError("the service did not complete and clean up normally")
                record["cleanup"] = {
                    name: telemetry.integer(name)
                    for name in ("live_buffers", "failed_checks", "fatal_errors")
                }
    snapshot_inputs(output, "sources", sources, create=False)
    snapshot_inputs(output, "registration", {"registration.json": registration}, create=False)
    record["complete"] = True
    _publish(output, (json.dumps(record, indent=1) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=257)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    measure(args.registration, args.output, rows=args.rows, samples=args.samples)


if __name__ == "__main__":
    main()
