"""Complete mixed generated/engine TPC-H Q3 through the authenticated service.

All generated execution uses the unchanged archived safe Sonnet projection.
The result reference is the repository's pandas Q3. Input/output bytes and
individual request wall times are retained; these are integration measurements,
not a controlled performance comparison or evidence for arbitrary corpus plans.
"""

import argparse
import base64
import tempfile
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.corpus.adapter import binding_of
from eval.rq3_pipelines.service_queries import (
    Q3_COLUMNS,
    Q3_OUTPUT,
    REVENUE,
    ShippingPriority,
    result_columns,
)
from eval.service_plans import VERIFICATION_WORKERS
from eval.tpch.database import Tables, generate
from opgate.catalog import TenantId
from opgate.frontends import Target
from opgate.gate import export_terms, formal_dir, zeroed
from opgate.hosts.service import Client, Fields, Limits, Principal, Service, binding_record

ROOT = Path(__file__).resolve().parents[2]
PROJECTION = ROOT / "eval/corpus/synthesized/sonnet/projection/numba.O1.ptx"
STAGES = (
    {"operation": "construct numeric views and UTF-8 strings", "implementation": "trusted cuDF"},
    {"operation": "segment/date filters and two inner joins", "implementation": "trusted cuDF"},
    {"operation": "price * (1 - discount)", "implementation": "verified archived generated PTX"},
    {"operation": "grouped sum, ordering and top 10", "implementation": "trusted cuDF"},
)


def cases(*, include_generated: bool) -> dict[str, Tables[pd.DataFrame]]:
    """Exact-date boundaries, multiplicity, ties/top-10, and an empty join."""
    cutoff = (date(1995, 3, 15) - date(1970, 1, 1)).days
    empty = pd.DataFrame()
    keys = list(range(1, 15))
    customer = pd.DataFrame({"c_custkey": [1, 2], "c_mktsegment": ["BUILDING", "AUTOMOBILE"]})
    orders = pd.DataFrame(
        {
            "o_orderkey": keys + [15, 16, 17],
            "o_custkey": [1] * 16 + [2],
            "o_orderdate": [cutoff - 2] * 7 + [cutoff - 1] * 7 + [cutoff, cutoff - 1, cutoff - 1],
            "o_shippriority": [0] * 17,
        }
    )
    lineitem = pd.DataFrame(
        {
            "l_orderkey": keys + keys + [15, 16, 17],
            "l_extendedprice": [64.0] * 14 + [32.0] * 14 + [4096.0] * 3,
            "l_discount": [0.25] * 14 + [0.5] * 14 + [0.0] * 3,
            "l_shipdate": [cutoff + 1] * 29 + [cutoff, cutoff + 1],
        }
    )
    tables = Tables(0.01, empty, empty, empty, customer, empty, empty, orders, lineitem)
    result = {
        "date_boundaries_duplicates_ties_top10": tables,
        "empty_join": replace(tables, customer=customer.assign(c_mktsegment="AUTOMOBILE")),
        "empty_customer_table": replace(tables, customer=customer.iloc[:0]),
    }
    if include_generated:
        result["generated_sf001_seed154"] = generate(0.01, 154)
    return result


def decode_result(response: Fields) -> pd.DataFrame:
    """Validate the complete typed response before comparing its values."""
    rows = response.integer("rows", minimum=0, maximum=10)
    columns = response.values("columns")
    if len(columns) != len(Q3_OUTPUT):
        raise AssertionError("Q3 returned the wrong columns")
    result = {}
    for value, (name, dtype) in zip(columns, Q3_OUTPUT, strict=True):
        column = Fields(value, "column")
        column.exact("name", "dtype", "data")
        if (column.text("name"), column.text("dtype")) != (name, dtype):
            raise AssertionError("Q3 returned the wrong column type or order")
        raw = base64.b64decode(column.text("data"), validate=True)
        if len(raw) != rows * np.dtype(dtype).itemsize:
            raise AssertionError("Q3 returned the wrong output length")
        result[name] = np.frombuffer(raw, dtype=dtype)
    return pd.DataFrame(result)


def run_queries(path: Path, *, include_generated: bool = True, repeats: int = 3) -> dict:
    from eval.tpch.queries import host, mismatch, q3

    if type(repeats) is not int or not 1 <= repeats <= 100:
        raise ValueError("repeats must be in [1, 100]")
    datasets = cases(include_generated=include_generated)
    ptx = PROJECTION.read_bytes().decode("ascii")
    binding = REVENUE.plan(ptx, Fields({"kind": REVENUE.kind}, "plan")).binding
    if binding != binding_of("sonnet", "projection", "numba", ptx):
        raise AssertionError("the deployed schema differs from the archived projection contract")
    trace = []

    def request(
        client: Client, op: str, expected_error: str | None = None, /, **fields: object
    ) -> Fields:
        start = time.perf_counter()
        response = client.request(op, **fields)
        trace.append(
            {
                "request": {"op": op, **fields},
                "response": response.record(),
                "wall_seconds": time.perf_counter() - start,
            }
        )
        if expected_error is not None:
            if (
                response.raw("ok") is not False
                or response.fields("error").text("kind") != expected_error
            ):
                raise AssertionError(response.record())
        elif response.raw("ok") is not True:
            raise AssertionError(response.record())
        return response

    path.parent.chmod(0o700)
    principals = (
        Principal("worker", "q3-worker-example-token-0123456789", TenantId(1)),
        Principal(
            "admin",
            "q3-admin-example-token-0123456789",
            TenantId(2),
            administrator=True,
            reservation=64 << 20,
        ),
    )
    start = time.perf_counter()
    service = Service(
        path,
        principals,
        device=0,
        pool_bytes=512 << 20,
        target=Target(8, 6),
        schemas=(REVENUE,),
        nodes=(ShippingPriority(),),
        limits=Limits(verification_workers=VERIFICATION_WORKERS, tenant_bytes=384 << 20),
    )
    startup_seconds = time.perf_counter() - start
    results = []
    total_launches = 0
    with service, Client(path, principals[1].token) as admin:
        with Client(path, principals[0].token) as client:
            (answer,) = request(
                client, "register", kernels=[{"ptx": ptx, "plan": {"kind": REVENUE.kind}}]
            ).values("registrations")
            registered = Fields(answer, "register.registrations[0]")
            if registered.raw("ok") is not True:
                raise AssertionError(registered.record())
            module = registered.text("operator")
            registration_seconds = trace[-1]["wall_seconds"]
            for name, tables in datasets.items():
                reference = host(q3(pd, tables))
                expected = result_columns(reference, Q3_OUTPUT)
                payload = {}
                handles = []
                upload_start = time.perf_counter()
                for table_name, columns in Q3_COLUMNS.items():
                    table = getattr(tables, table_name)
                    wire = {}
                    for column, dtype in columns:
                        if dtype is None:
                            wire[column] = table[column].tolist()
                        else:
                            raw = np.ascontiguousarray(
                                table[column].to_numpy(), dtype=dtype
                            ).tobytes()
                            # The service has no zero-byte allocations. Empty
                            # columns use a one-byte allocation and a zero-row view.
                            handle = request(client, "allocate", nbytes=max(1, len(raw))).text(
                                "buffer"
                            )
                            handles.append(handle)
                            request(
                                client,
                                "write",
                                buffer=handle,
                                data=base64.b64encode(raw).decode("ascii"),
                                element=None,
                            )
                            wire[column] = handle
                    payload[table_name] = {"rows": len(table), "columns": wire}
                upload_seconds = time.perf_counter() - upload_start
                # An administrator has no implicit authority to another session's
                # module or columns. This is refused before engine work begins.
                request(admin, "tpch_q3", "refused", operator=module, **payload)
                samples = []
                for sample in range(repeats + 1):
                    response = request(client, "tpch_q3", operator=module, **payload)
                    actual = decode_result(response)
                    error = mismatch(reference, actual)
                    if error is not None:
                        raise AssertionError(f"{name}: {error}")
                    launches = response.integer("generated_launches", minimum=0, maximum=1)
                    projected = response.integer("projection_rows", minimum=0)
                    if (projected > 0) != bool(launches) or bool(len(reference)) != bool(launches):
                        raise AssertionError(
                            "the nonempty query must execute its generated operator"
                        )
                    total_launches += launches
                    raw_equal = response.raw("columns") == expected["columns"]
                    if name != "generated_sf001_seed154" and not raw_equal:
                        raise AssertionError("the exact binary fixture differs in its output bytes")
                    samples.append(
                        {
                            "sample": sample,
                            "wall_seconds": trace[-1]["wall_seconds"],
                            "reference_bytes_equal": raw_equal,
                            "response": response.record(),
                        }
                    )
                for handle in handles:
                    request(client, "free", buffer=handle)
                results.append(
                    {
                        "case": name,
                        "upload_seconds": upload_seconds,
                        "reference": expected,
                        "samples": samples,
                    }
                )
        audit = admin.audit()
        if len(audit) != 1:
            raise AssertionError("the projection registration was not retained")
        entry = Fields(audit[0], "audit")
        if entry.text("ptx").encode("ascii") != PROJECTION.read_bytes():
            raise AssertionError("the audit PTX differs from the archived bytes")
        if entry.integer("launch_count") != total_launches:
            raise AssertionError("the generated-launch audit count is wrong")
        telemetry = request(admin, "telemetry").fields("telemetry").record()
        if telemetry["live_buffers"] != 0 or telemetry["internal_errors"] != 0:
            raise AssertionError(telemetry)
    return {
        "stages": STAGES,
        "formal_library": str(formal_dir()),
        "startup_seconds": startup_seconds,
        "cold_registration_seconds": registration_seconds,
        "binding": binding_record(binding),
        "registration_query": export_terms(zeroed(ptx, binding), [binding]),
        "cases": results,
        "requests": trace,
        "audit": audit,
        "telemetry": telemetry,
        "generated_launches": total_launches,
        "comparison": "all columns/rows/order; exact integer and fixture bytes; generated-data revenue relative tolerance 1e-9",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {str(p.relative_to(ROOT)): p for p in (ROOT / "opgate").rglob("*.py")}
    files.update({f"formal/{p.name}": p for p in formal_dir().glob("*.v")})
    for name in (
        "eval/rq3_pipelines/service_q3.py",
        "eval/rq3_pipelines/service_queries.py",
        "eval/tpch/database.py",
        "eval/tpch/data.py",
        "eval/tpch/queries.py",
        "eval/checkpoint.py",
        "eval/corpus/adapter.py",
        "tests/hosts/test_service_queries.py",
    ):
        files[name] = ROOT / name
    for file in PROJECTION.parent.iterdir():
        if file.is_file():
            files[str(file.relative_to(ROOT))] = file
    with Checkpoint(
        args.output,
        {
            "scope": "complete mixed generated/engine Q3; four data cases",
            "stages": STAGES,
            "repeats": 3,
        },
    ) as record:
        snapshot_inputs(args.output, "sources", files, create=True)
        with tempfile.TemporaryDirectory(prefix="opgate-q3-") as directory:
            record.add({"id": "complete-q3", **run_queries(Path(directory) / "service.sock")})
        snapshot_inputs(args.output, "sources", files, create=False)
        record.finish()
    print(args.output)


if __name__ == "__main__":
    main()
