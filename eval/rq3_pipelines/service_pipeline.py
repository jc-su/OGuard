"""A generic registered index pipeline followed by trusted cuDF aggregation.

Run with ``CUDA_VISIBLE_DEVICES=1 python -m eval.rq3_pipelines.service_pipeline
--output eval/records/service_pipeline_<date>.json``. Only safe kernels and
in-range data execute. A raw input lacking provenance is refused before
submission, then checked by the guarded path. Every request, response,
submitted PTX, binding, and input/output byte is retained without a digest.
"""

import argparse
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.service_plans import VERIFICATION_WORKERS
from opgate import signature as sig
from opgate.binding import Convention, Direction, ElementType
from opgate.catalog import TenantId
from opgate.frontends import Target
from opgate.frontends.nvcc import nvcc_operator
from opgate.gate import formal_dir
from opgate.hosts.service import (
    Client,
    Fields,
    Limits,
    NodeContext,
    OperatorSchema,
    Principal,
    Service,
    binding_record,
)

N = sig.Affine.symbol(0)
BLOCK = (128, 1, 1)
GRID = (8192, 1, 1)
LONG = ElementType(8, True)
SIZES = (sig.SizeSymbol("n", 1, 1 << 20),)
SOURCE = sig.Span("source", "source", Direction.READ, N)
COUNT = sig.Scalar("n", sig.SizeOf(N))
PRODUCER = OperatorSchema(
    "reverse_indices",
    sig.Signature(SIZES, (SOURCE, sig.IndexOut("index", "index", N, "source"), COUNT)),
    (("source", LONG), ("index", LONG)),
    Convention.C,
    BLOCK,
    GRID,
)
CONSUMER = OperatorSchema(
    "gather_i64",
    sig.Signature(
        SIZES,
        (
            SOURCE,
            sig.IndexIn("index", "index", N, "source"),
            sig.Span("output", "output", Direction.WRITE, N),
            COUNT,
        ),
    ),
    (("source", LONG), ("index", LONG), ("output", LONG)),
    Convention.C,
    BLOCK,
    GRID,
)
KERNELS = {
    "reverse_indices": """extern "C" __global__ void reverse_indices(
        const long long* source, long long* index, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) index[i] = n - 1 - i;
    }""",
    "gather_i64": """extern "C" __global__ void gather_i64(
        const long long* source, const long long* index, long long* output, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) output[i] = source[index[i]];
    }""",
}


@dataclass(frozen=True)
class GroupedSum:
    """A deployment-owned engine node reading the resulting catalog column."""

    op: str = "grouped_sum"

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        import cudf

        from opgate.hosts.rapids import BufferView

        request.exact("op", "buffer", "rows")
        rows = request.integer("rows", minimum=1, maximum=SIZES[0].maximum)
        buffer = context.buffer(request, "buffer")
        view = BufferView.of(context.resource, buffer, "<i8", rows, access="read")
        values = cudf.Series(view)
        frame = cudf.DataFrame({"group": values % 7, "value": values})
        grouped = frame.groupby("group")["value"].sum().sort_index()
        return {
            "groups": grouped.index.to_numpy().tolist(),
            "sums": grouped.to_numpy().tolist(),
        }


def compile_operators(target: Target) -> dict[str, str]:
    return {
        schema.kind: nvcc_operator(
            KERNELS[schema.kind], schema.kind, schema.signature, BLOCK, target, grid=GRID
        ).text
        for schema in (PRODUCER, CONSUMER)
    }


def run_pipeline(path: Path, *, rows: int = 4097) -> dict[str, object]:
    if type(rows) is not int or not 1 <= rows <= SIZES[0].maximum:
        raise ValueError("rows is outside the deployed signature")
    target = Target(8, 6)
    ptx = compile_operators(target)
    trace: list[dict[str, object]] = []

    def request(client: Client, op: str, *, refused: bool = False, **fields: object) -> Fields:
        response = client.request(op, **fields)
        trace.append({"request": {"op": op, **fields}, "response": response.record()})
        if refused:
            if (
                response.raw("ok") is not False
                or response.fields("error").text("kind") != "refused"
            ):
                raise AssertionError(response.record())
        elif response.raw("ok") is not True:
            raise AssertionError(response.record())
        return response

    path.parent.chmod(0o700)
    principals = (
        Principal("worker", "worker-example-token-0123456789", TenantId(1)),
        Principal("admin", "admin-example-token-0123456789", TenantId(2), administrator=True),
    )
    with Service(
        path,
        principals,
        device=0,
        pool_bytes=128 << 20,
        target=target,
        schemas=(PRODUCER, CONSUMER),
        nodes=(GroupedSum(),),
        limits=Limits(
            verification_workers=VERIFICATION_WORKERS, tenant_bytes=32 << 20, chunk_bytes=8 << 20
        ),
    ):
        with Client(path, principals[1].token) as admin:
            with Client(path, principals[0].token) as client:
                # every operator in one request, checked at once
                answers = request(
                    client,
                    "register",
                    kernels=[{"ptx": text, "plan": {"kind": kind}} for kind, text in ptx.items()],
                ).values("registrations")
                modules: dict[str, str] = {}
                for kind, answer in zip(ptx, answers, strict=True):
                    registered = Fields(answer, f"register.{kind}")
                    if registered.raw("ok") is not True:
                        raise AssertionError(registered.record())
                    modules[kind] = registered.text("operator")
                handles = {
                    name: request(client, "allocate", nbytes=8 * rows).text("buffer")
                    for name in ("source", "index", "output")
                }
                values = np.arange(rows, dtype=np.int64) * 3 + 1

                def write(name: str, data: np.ndarray) -> None:
                    request(
                        client,
                        "write",
                        buffer=handles[name],
                        data=base64.b64encode(data.tobytes()).decode("ascii"),
                        element=None,
                    )

                def launch(kind: str, guarded: bool, *, refused: bool = False) -> None:
                    names = ("source", "index") if kind == PRODUCER.kind else tuple(handles)
                    response = request(
                        client,
                        "run_operator",
                        operator=modules[kind],
                        grid=[-(-rows // BLOCK[0]), 1, 1],
                        buffers={name: handles[name] for name in names},
                        sizes={"n": rows},
                        scalars={},
                        guarded=guarded,
                        refused=refused,
                    )
                    if not refused and response.raw("ran") != [True]:
                        raise AssertionError("the checked operator did not execute")

                def read_output(expected: np.ndarray) -> None:
                    response = request(client, "read", buffer=handles["output"])
                    actual = base64.b64decode(response.text("data"), validate=True)
                    if actual != expected.tobytes():
                        raise AssertionError("the pipeline output differs from the reference bytes")

                write("source", values)
                # Index outputs preserve a range for the entire allocation,
                # including cells a kernel may leave unwritten. Establish the
                # initial in-range contents through the checked typed write.
                request(
                    client,
                    "write",
                    buffer=handles["index"],
                    data=base64.b64encode(bytes(rows * 8)).decode("ascii"),
                    element={"width": 8, "signed": True},
                )
                launch(PRODUCER.kind, False)
                launch(CONSUMER.kind, False)
                read_output(values[::-1])
                # Replacing the indices with safe raw data clears static provenance.
                # The same kernel is refused before launch until a checked scan is requested.
                indices = np.arange(rows, dtype=np.int64)
                write("index", indices)
                launch(CONSUMER.kind, False, refused=True)
                launch(CONSUMER.kind, True)
                read_output(values)
                grouped = request(client, "grouped_sum", buffer=handles["output"], rows=rows)
                groups = sorted(set((values % 7).tolist()))
                expected_sums = [int(values[values % 7 == group].sum()) for group in groups]
                if grouped.raw("groups") != groups or grouped.raw("sums") != expected_sums:
                    raise AssertionError("cuDF aggregation differs from the NumPy reference")
                for handle in handles.values():
                    request(client, "free", buffer=handle)
            audit = admin.audit()
            if len(audit) != 2:
                raise AssertionError("the two registrations were not retained")
            for raw in audit:
                entry = Fields(raw, "registration")
                kind = entry.text("plan_kind")
                if entry.text("ptx").encode("ascii") != ptx[kind].encode("ascii"):
                    raise AssertionError("the audit differs from the submitted PTX bytes")
                if entry.integer("launch_count") != (1 if kind == PRODUCER.kind else 2):
                    raise AssertionError("a refused launch executed or a successful one was lost")
            telemetry = request(admin, "telemetry").fields("telemetry").record()
            if telemetry["live_buffers"] != 0 or telemetry["internal_errors"] != 0:
                raise AssertionError(telemetry)
    bindings = {
        schema.kind: binding_record(
            schema.plan(ptx[schema.kind], Fields({"kind": schema.kind}, "plan")).binding
        )
        for schema in (PRODUCER, CONSUMER)
    }
    return {
        "rows": rows,
        "requests": trace,
        "audit": audit,
        "bindings": bindings,
        "telemetry": telemetry,
        "groups": groups,
        "sums": expected_sums,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4097)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    library = formal_dir()
    files = {str(p.relative_to(root)): p for p in (root / "opgate").rglob("*.py")}
    files.update({f"formal/{p.name}": p for p in library.glob("*.v")})
    files["eval/rq3_pipelines/service_pipeline.py"] = Path(__file__)
    with Checkpoint(
        args.output,
        {
            "scope": "safe generic service pipeline plus cuDF grouped aggregation",
            "rows": args.rows,
            "kernels": KERNELS,
            "refusal": "in-range raw index data without required provenance; no unsafe kernel executes",
        },
    ) as record:
        snapshot_inputs(args.output, "sources", files, create=True)
        with tempfile.TemporaryDirectory(prefix="opgate-pipeline-") as directory:
            record.add(
                {
                    "id": "service-pipeline",
                    **run_pipeline(Path(directory) / "service.sock", rows=args.rows),
                }
            )
        snapshot_inputs(args.output, "sources", files, create=False)
        record.finish()
    print(args.output)


if __name__ == "__main__":
    main()
