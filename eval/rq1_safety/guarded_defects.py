"""RQ1: the corpus defects that the guard pass makes acceptable, run on their triggers.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq1_safety.guarded_defects \\
        --output eval/records/guarded_defects_<date>.json

The unchecked corpus run (`e3_2026-09-26.tar.gz`) found an input or a
launch grid that drives each of 27 defective corpus kernels outside its
buffers: a store into a neighbouring buffer, a load whose value reaches an
output, or a fault. This experiment applies the guard pass
(`opgate.frontends.guard`) with the current gate to each of those kernels'
recorded PTX, and runs each kernel that the guarded text makes acceptable on
the same trigger, rebuilt from the archive's record (`TRIGGERS`).

The tenant's buffers lie in its reservation, which starts the executor's
pool, with neighbouring bytes that no buffer of the tenant holds after each
of them and before the first, as another tenant's buffers lie next to them
in a pool that one suballocator shares: the tenant allocates a filler buffer
for each such region, in address order, and frees it, so its first-fit
suballocator places each buffer as the layout states. Before each launch,
the whole span holds one fill byte, which differs with the secret; the bytes
that follow an input buffer hold the archive's values for that buffer's
neighbour, of which there are two (`secret` 0 and 1), as the archive's read
tests had; each of the tenant's buffers, outputs too, is written through the
catalog with the trigger's values, an index output with values in its
declared range, since the catalog checks them. Each kernel runs twice with
each secret:

- `guarded`: the guarded text and binding, registered with the executor
  and launched through it, after the catalog checks the launch;
- `as_written`: the kernel as written, loaded with the driver without the
  gate, and launched directly on the same buffers with the checked launch's
  parameters (`eval.direct`), which it reads as its own, since the appended
  extent and status parameters come last.

A guarded text with the pass's checks takes a status buffer of the tenant,
after the span, to which a failed check stores 1; it holds zeros before
each launch. For each run the record keeps the bytes of the span that
changed outside the tenant's output buffers, with their offsets and their
values before and after, each output's bytes, and, for a guarded run of a
text with checks, whether a check failed (the status buffer's first word
after the launch). A guarded kernel changes no byte outside its outputs and
reports a failed check, since its trigger drives an access outside its
region, and for a trigger with neighbour values, a read test, its outputs
are the same bytes under both secrets; the outputs of the other triggers
may differ between runs, as `hash_probe`'s order of matches does. The
command fails if a guarded run does otherwise; the record is written
first.

The archive's three far triggers (a 32-bit index that wraps near
`INT_MAX`, in buffers of 2 and 16 GiB) fault the context as written, which
ends the process, so they run guarded only, after the others, in a pool of
`LARGE_POOL` bytes, each in a process of its own that may run for
`LARGE_SECONDS`. Guarded, a wrapped index's accesses do not run, but a loop
whose index wraps may not reach its end, so the kernel may not end; the
record states it, since a kernel's run time is outside the design's
guarantee (design section 3). Their large input holds zeros, which the
record states as its element type and count.
"""

import argparse
import base64
import json
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import numpy as np

from eval import RECORDS
from eval.corpus.adapter import (
    ELEMENTS,
    Pointer,
    Points,
    Role,
    binding_of,
    directory_of,
    read_raw,
    read_roles,
)
from eval.corpus.run import TOOLCHAINS
from eval.corpus.workloads import encode
from eval.direct import Direct
from opgate.binding import Binding
from opgate.catalog import (
    ALIGNMENT,
    BufferId,
    Kernel,
    Launch,
    Step,
    StepRefused,
    TenantId,
    Write,
)
from opgate.executor import Executor, Module, RefusedError
from opgate.executor.driver import call, driver
from opgate.frontends.guard import STATUS_BYTES, Guarded, guard, reports
from opgate.gate import Accepted, Refused, formal_dir, verify
from opgate.hosts.service.audit import binding_record

__all__ = ["main"]

E3_ARCHIVE = RECORDS / "e3_2026-09-26.tar.gz"
TENANT = TenantId(1)
NEIGHBOUR_BYTES = 4096  # each neighbouring region
LAST_NEIGHBOUR_BYTES = 1 << 16  # the one after the tenant's last buffer
FILL = (0xAB, 0x5C)  # the span's fill byte, for secret 0 and 1
POOL = 64 << 20
LARGE_POOL = 20 << 30
LARGE_SECONDS = 600  # a far trigger's process
INT_MAX = 2**31 - 1

Mode = Literal["guarded", "as_written"]


@dataclass(frozen=True)
class Trigger:
    """A defect's trigger: the value of each argument and size symbol, the grid,
    and the bytes that follow an input buffer, for secret 0 and 1."""

    arrays: dict[str, np.ndarray]
    scalars: dict[str, int | float]
    sizes: dict[str, int]
    grid: int
    after: dict[str, tuple[bytes, bytes]] = field(default_factory=dict)
    large: bool = False


def _splitmix64(key: int, mask: int) -> int:
    """Sonnet's hash of `hash_build` and `hash_probe`."""
    m64 = (1 << 64) - 1
    h = key & m64
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & m64
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & m64
    h = (h ^ (h >> 31)) & m64
    return h & mask


def _mix64(key: int, mask: int) -> int:
    """Fable's hash of `hash_build` and `hash_probe`."""
    m64 = (1 << 64) - 1
    h = key & m64
    h = (h ^ (h >> 33)) & m64
    h = (h * 0xFF51AFD7ED558CCD) & m64
    h = (h ^ (h >> 33)) & m64
    h = (h * 0xC4CEB9FE1A85EC53) & m64
    h = (h ^ (h >> 33)) & m64
    return h & mask


def _filter_select(count: int) -> Trigger:
    """One row passes the filter; the counter starts at `count`."""
    n = 64
    quantity = np.ones(n)
    quantity[0] = -1.0
    return Trigger(
        {
            "quantity": quantity,
            "sel": np.zeros(n, dtype=np.int32),
            "count": np.array([count], dtype=np.int32),
        },
        {"threshold": 0.0},
        {"n": n},
        1,
    )


def _hash_probe_capacity() -> Trigger:
    """1,000 matches into an output of 100, bounded by the table's capacity."""
    capacity, out_cap, m = 2048, 100, 1000
    table_keys = np.full(capacity, -1, dtype=np.int64)
    table_rows = np.zeros(capacity, dtype=np.int32)
    table_keys[:m] = np.arange(m)
    table_rows[:m] = np.arange(m)
    return Trigger(
        {
            "probe_keys": np.arange(m, dtype=np.int64),
            "table_keys": table_keys,
            "table_rows": table_rows,
            "out_build": np.full(out_cap, -7, dtype=np.int32),
            "out_probe": np.zeros(out_cap, dtype=np.int32),
            "out_count": np.zeros(1, dtype=np.int32),
        },
        {},
        {"m": m, "capacity": capacity, "out_cap": out_cap},
        (m + 255) // 256,
    )


def _hash_probe_negative(count: int, hash: Callable[[int, int], int]) -> Trigger:
    """One match, with the output counter starting at `count`."""
    capacity, out_cap, key = 16, 100, 12345
    table_keys = np.full(capacity, -1, dtype=np.int64)
    table_rows = np.full(capacity, -1, dtype=np.int32)
    slot = hash(key, capacity - 1)
    table_keys[slot], table_rows[slot] = key, 42
    return Trigger(
        {
            "probe_keys": np.array([key], dtype=np.int64),
            "table_keys": table_keys,
            "table_rows": table_rows,
            "out_build": np.full(out_cap, -7, dtype=np.int32),
            "out_probe": np.zeros(out_cap, dtype=np.int32),
            "out_count": np.array([count], dtype=np.int32),
        },
        {},
        {"m": 1, "capacity": capacity, "out_cap": out_cap},
        1,
    )


def _scatter_by_offsets() -> Trigger:
    """A flagged row whose offset is the output's length."""
    n, total = 16, 8
    flags = np.zeros(n, dtype=np.int8)
    offsets = np.zeros(n, dtype=np.int32)
    flags[5], offsets[5] = 1, total
    return Trigger(
        {"flags": flags, "offsets": offsets, "out": np.zeros(total, dtype=np.int32)},
        {},
        {"n": n, "total": total},
        1,
    )


def _secret(
    dtype: type[np.generic], values: tuple[int | float, int | float]
) -> tuple[bytes, bytes]:
    """64 elements of a neighbour's column, for secret 0 and 1."""
    return (
        np.full(64, values[0], dtype=dtype).tobytes(),
        np.full(64, values[1], dtype=dtype).tobytes(),
    )


def _groupby_sum() -> Trigger:
    """4 blocks over 1,000 rows, with no row count: the last 24 threads read the
    group and value columns' neighbours."""
    n, capacity = 1000, 4096
    return Trigger(
        {
            "group": np.zeros(n, dtype=np.int32),
            "value": np.ones(n),
            "table_keys": np.full(capacity, -1, dtype=np.int32),
            "table_sums": np.zeros(capacity),
        },
        {},
        {"n": n, "capacity": capacity},
        4,
        {"group": _secret(np.int32, (111, 222)), "value": _secret(np.float64, (9.0, 9.0))},
    )


def _hash_build() -> Trigger:
    """4 blocks over 1,000 keys, with no row count: the table holds the keys'
    neighbour."""
    n, capacity = 1000, 4096
    return Trigger(
        {
            "keys": np.arange(n, dtype=np.int64),
            "table_keys": np.full(capacity, -1, dtype=np.int64),
            "table_rows": np.full(capacity, -1, dtype=np.int32),
        },
        {},
        {"n": n, "capacity": capacity},
        4,
        {"keys": _secret(np.int64, (700000001, 700000002))},
    )


def _histogram() -> Trigger:
    """4 blocks over 1,000 keys, with no row count: the counts reveal the keys'
    neighbour."""
    n = 1000
    return Trigger(
        {"keys": np.zeros(n, dtype=np.uint32), "counts": np.zeros(256, dtype=np.int32)},
        {"shift": 0},
        {"n": n},
        4,
        {"keys": _secret(np.uint32, (5, 9))},
    )


def _q6_sum() -> Trigger:
    """4 blocks over 1,000 rows that fail the filter: the neighbours' rows pass it
    and add their price to the result."""
    n = 1000
    return Trigger(
        {
            "shipdate": np.zeros(n, dtype=np.int32),
            "discount": np.zeros(n),
            "quantity": np.full(n, 100.0),
            "price": np.zeros(n),
            "result": np.zeros(1),
        },
        {"date_lo": 100, "date_hi": 200, "disc_lo": 0.0, "disc_hi": 0.1, "qty_max": 50.0},
        {"n": n},
        4,
        {
            "shipdate": _secret(np.int32, (150, 150)),
            "discount": _secret(np.float64, (0.05, 0.05)),
            "quantity": _secret(np.float64, (1.0, 1.0)),
            "price": _secret(np.float64, (1000.0, 2000.0)),
        },
    )


def _strings_past_count(output: str, dtype: type[np.generic], initial: int) -> Trigger:
    """4 blocks over 1,000 strings: the kernel takes 1,023 rows, reads the offsets'
    neighbour, and writes past its output."""
    n = 1000
    arrays = {
        "offsets": np.arange(n + 1, dtype=np.int32),
        "chars": np.zeros(n, dtype=np.uint8),
        output: np.full(n, initial, dtype=dtype),
    }
    if output == "match":
        arrays["prefix"] = np.zeros(1, dtype=np.uint8)
    sizes = {"n": n, "chars": n} | ({"k": 1} if output == "match" else {})
    return Trigger(
        arrays, {}, sizes, 4, {"offsets": _secret(np.int32, (-2_000_000_000, -1_900_000_000))}
    )


def _haiku_string_length_numba() -> Trigger:
    """One string: the kernel stores 8 bytes into a column of 4-byte lengths."""
    return Trigger(
        {
            "offsets": np.array([0, 5], dtype=np.int32),
            "chars": np.frombuffer(b"hello", dtype=np.uint8).copy(),
            "lengths": np.full(1, -7, dtype=np.int32),
        },
        {},
        {"n": 1, "chars": 5},
        1,
    )


def _fable_starts_with_numba() -> Trigger:
    """Decreasing offsets: the unsigned length wraps, and the kernel compares the
    prefix with the 5 bytes after the characters."""
    prefix = np.array([0x11, 0x22, 0x33, 0x44, 0x55], dtype=np.uint8)
    return Trigger(
        {
            "offsets": np.array([10, 0, 10], dtype=np.int32),
            "chars": np.zeros(10, dtype=np.uint8),
            "prefix": prefix,
            "match": np.full(2, 9, dtype=np.int8),
        },
        {},
        {"n": 2, "chars": 10, "k": 5},
        1,
        {"chars": (prefix.tobytes(), bytes([0x66] * 5))},
    )


def _wrap(task: str) -> Trigger:
    """One segment that ends at `INT_MAX`: a 32-bit index wraps."""
    big, out, symbol = (
        ("values", "sums", "values") if task == "segmented_sum" else ("chars", "lengths", "chars")
    )
    dtype = np.float64 if task == "segmented_sum" else np.uint8
    out_dtype = np.float64 if task == "segmented_sum" else np.int32
    rows = "g" if task == "segmented_sum" else "n"
    return Trigger(
        {
            "offsets": np.array([INT_MAX - 1, INT_MAX], dtype=np.int32),
            big: np.zeros(INT_MAX, dtype=dtype),
            out: np.zeros(1, dtype=out_dtype),
        },
        {},
        {rows: 1, symbol: INT_MAX},
        1,
        large=True,
    )


TRIGGERS: dict[str, Callable[[], Trigger]] = {
    "haiku/filter_select/numba": lambda: _filter_select(-65),
    "haiku/filter_select/nvcc": lambda: _filter_select(-1),
    "sonnet/filter_select/numba": lambda: _filter_select(64),
    "sonnet/filter_select/nvcc": lambda: _filter_select(64),
    "fable/filter_select/numba": lambda: _filter_select(64),
    "fable/filter_select/nvcc": lambda: _filter_select(64),
    "haiku/hash_probe/numba": _hash_probe_capacity,
    "haiku/hash_probe/nvcc": _hash_probe_capacity,
    "sonnet/hash_probe/numba": lambda: _hash_probe_negative(-101, _splitmix64),
    "sonnet/hash_probe/nvcc": lambda: _hash_probe_negative(-1, _splitmix64),
    "fable/hash_probe/numba": lambda: _hash_probe_negative(-101, _mix64),
    "fable/hash_probe/nvcc": lambda: _hash_probe_negative(-1, _mix64),
    "haiku/scatter_by_offsets/numba": _scatter_by_offsets,
    "haiku/scatter_by_offsets/nvcc": _scatter_by_offsets,
    "fable/scatter_by_offsets/numba": _scatter_by_offsets,
    "fable/scatter_by_offsets/nvcc": _scatter_by_offsets,
    "haiku/string_length/numba": _haiku_string_length_numba,
    "haiku/groupby_sum/nvcc": _groupby_sum,
    "haiku/hash_build/nvcc": _hash_build,
    "haiku/histogram/nvcc": _histogram,
    "haiku/q6_sum/nvcc": _q6_sum,
    "haiku/starts_with/nvcc": lambda: _strings_past_count("match", np.int8, 2),
    "haiku/string_length/nvcc": lambda: _strings_past_count("lengths", np.int32, -7),
    "fable/starts_with/numba": _fable_starts_with_numba,
    "sonnet/segmented_sum/nvcc": lambda: _wrap("segmented_sum"),
    "fable/string_length/nvcc": lambda: _wrap("string_length"),
    "fable/segmented_sum/nvcc": lambda: _wrap("segmented_sum"),
}


class Change(TypedDict):
    offset: int
    region: str
    before: int
    after: int


class Run(TypedDict):
    mode: Mode
    secret: int
    launched: bool
    refused: str | None
    changed: list[Change]
    outputs: dict[str, str]
    check_failed: bool | None  # None for a run without the pass's checks


def _archive() -> list[dict[str, object]]:
    with tarfile.open(E3_ARCHIVE) as archive:
        member = archive.extractfile("e3/results.jsonl")
        if member is None:
            raise ValueError(f"{E3_ARCHIVE} has no e3/results.jsonl")
        return [json.loads(line) for line in member.read().decode().splitlines()]


def _up(offset: int) -> int:
    return -(-offset // ALIGNMENT) * ALIGNMENT


@dataclass(frozen=True)
class _Layout:
    """Where each buffer of the tenant and each neighbouring region lies in the pool."""

    names: tuple[str, ...]  # the tenant's array arguments, in order
    offsets: dict[str, int]
    sizes: dict[str, int]
    neighbours: tuple[tuple[int, int], ...]  # (offset, bytes)
    end: int

    def region(self, offset: int) -> str:
        for name in self.names:
            if self.offsets[name] <= offset < self.offsets[name] + self.sizes[name]:
                return name
        for at, nbytes in self.neighbours:
            if at <= offset < at + nbytes:
                return "neighbour"
        return "gap"


def _layout(arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> _Layout:
    neighbours = [(0, NEIGHBOUR_BYTES)]
    cursor = NEIGHBOUR_BYTES
    offsets: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for number, name in enumerate(names):
        offsets[name] = _up(cursor)
        sizes[name] = max(1, arrays[name].nbytes)
        cursor = _up(offsets[name] + sizes[name])
        nbytes = LAST_NEIGHBOUR_BYTES if number == len(names) - 1 else NEIGHBOUR_BYTES
        neighbours.append((cursor, nbytes))
        cursor += nbytes
    return _Layout(names, offsets, sizes, tuple(neighbours), cursor)


def _output(role: Role | Points) -> bool:
    """Whether the kernel may write an argument of `role`."""
    match role:
        case Points(pointer=pointer):
            return pointer is Pointer.INDEX_OUT
        case Role.WRITE | Role.READ_WRITE:
            return True
        case Role.READ | Role.SIZE | Role.SCALAR:
            return False
        case _ as never:
            assert_never(never)


def _offset(executor: Executor, buffer: BufferId) -> int:
    found = executor.catalog.buffer(buffer)
    if found is None:
        raise RuntimeError(f"buffer {buffer} is not live")
    return found.offset


def _span(executor: Executor, layout: _Layout) -> np.ndarray:
    call(driver.cuCtxSynchronize())
    data = bytearray(layout.end)
    call(driver.cuMemcpyDtoH(data, driver.CUdeviceptr(executor.pool_address), layout.end))
    return np.frombuffer(data, dtype=np.uint8)


def _run(
    executor: Executor,
    module: Module,
    function: object | None,
    key: str,
    trigger: Trigger,
    mode: Mode,
    secret: int,
) -> Run:
    model, task, toolchain = key.split("/")
    raws = read_raw(directory_of(model, task) / "signature.json")
    roles = read_roles(model, task)
    arrays = [r for r in raws if roles.arguments[r.name] not in (Role.SIZE, Role.SCALAR)]
    names = tuple(r.name for r in arrays)
    values = {
        r.name: np.ascontiguousarray(trigger.arrays[r.name], dtype=np.dtype(r.dtype))
        for r in arrays
    }
    layout = _layout(values, names)
    outputs = {r.name for r in arrays if _output(roles.arguments[r.name])}
    pool = driver.CUdeviceptr(executor.pool_address)
    call(driver.cuMemsetD8(pool, FILL[secret], layout.end))
    checks = reports(module.binding)
    with executor.tenant(TENANT):
        # the tenant's first-fit suballocator places each buffer at the layout's
        # offset once a filler takes each neighbouring region before it
        fillers = [executor.allocate(layout.neighbours[0][1])]
        placed: list[BufferId] = []
        for name, (_, nbytes) in zip(names, layout.neighbours[1:], strict=True):
            placed.append(executor.allocate(layout.sizes[name]))
            fillers.append(executor.allocate(nbytes))
        ours = tuple(placed)
        # the status buffer of the pass's checks, after the span
        status = executor.allocate(STATUS_BYTES)
        executor.free(*fillers)
        offsets = [_offset(executor, b) for b in ours]
        if offsets != [layout.offsets[n] for n in names] or _offset(executor, status) != _up(
            layout.end
        ):
            raise RuntimeError(f"{key}: the buffers do not lie where the layout states")
        writes = []
        for raw, buffer in zip(arrays, ours, strict=True):
            role, data = roles.arguments[raw.name], values[raw.name].tobytes()
            # every buffer, outputs too, so that no byte of a run comes from the fill
            points = isinstance(role, Points)
            writes.append(Write(buffer, data, ELEMENTS[raw.dtype] if points else None))
        writes.append(Write(status, bytes(STATUS_BYTES), None))
        executor.execute(Step(TENANT, tuple(writes)))
    # the neighbour's values after each input, over the gap and into its buffer
    for name, secrets in trigger.after.items():
        at = layout.offsets[name] + layout.sizes[name]
        call(
            driver.cuMemcpyHtoD(
                driver.CUdeviceptr(executor.pool_address + at),
                secrets[secret],
                len(secrets[secret]),
            )
        )
    sizes = tuple(trigger.sizes[name] for name in roles.symbols)
    scalars = tuple(
        encode(r.dtype, trigger.scalars[r.name])
        for r in raws
        if roles.arguments[r.name] is Role.SCALAR
    )
    buffers = (*ours, status) if checks else ours
    launch = Launch(module.operator, (trigger.grid, 1, 1), buffers, sizes, scalars, False)
    before = _span(executor, layout)
    refused: str | None = None
    with executor.tenant(TENANT):
        match mode:
            case "guarded":
                try:
                    executor.execute(Step(TENANT, (launch,)))
                except RefusedError as error:
                    refused = str(error)
            case "as_written":
                effects = executor.catalog.check(Step(TENANT, (launch,)))
                if isinstance(effects, StepRefused):
                    refused = effects.reason
                else:
                    (kernel,) = (e for e in effects if isinstance(e, Kernel))
                    Direct(executor).launch(kernel, module, function)
            case _ as never:
                assert_never(never)
    after = _span(executor, layout)
    failed: bool | None = None
    if mode == "guarded" and checks and refused is None:
        with executor.tenant(TENANT):
            failed = any(executor.read(TENANT, status))
    legal = np.zeros(layout.end, dtype=bool)
    for name in outputs:
        legal[layout.offsets[name] : layout.offsets[name] + layout.sizes[name]] = True
    changed = [
        Change(
            offset=int(o), region=layout.region(int(o)), before=int(before[o]), after=int(after[o])
        )
        for o in np.nonzero((before != after) & ~legal)[0]
    ]
    out = {
        name: base64.b64encode(
            after[layout.offsets[name] : layout.offsets[name] + layout.sizes[name]].tobytes()
        ).decode("ascii")
        for name in sorted(outputs)
    }
    with executor.tenant(TENANT):
        executor.free(*ours, status)
    return Run(
        mode=mode,
        secret=secret,
        launched=refused is None,
        refused=refused,
        changed=changed,
        outputs=out,
        check_failed=failed,
    )


def _inputs(trigger: Trigger) -> dict[str, dict[str, object]]:
    """Each array of the trigger as its bytes, or a large one of zeros as its
    element type and count."""
    inputs: dict[str, dict[str, object]] = {}
    for name, array in trigger.arrays.items():
        if trigger.large and array.nbytes > 1 << 20:
            if np.any(array):
                raise ValueError(f"the large array {name} is not zeros")
            inputs[name] = {"dtype": str(array.dtype), "count": int(array.size), "zeros": True}
        else:
            inputs[name] = {
                "dtype": str(array.dtype),
                "bytes": base64.b64encode(array.tobytes()).decode("ascii"),
            }
    return inputs


def _decided(row: dict[str, object], artifacts: Path) -> tuple[dict[str, object], Guarded]:
    """A defect's entry with the gate's decisions on its text as written and
    guarded; the guarded text and binding kept with the artifacts."""
    key = f"{row['model']}/{row['task']}/{row['toolchain']}"
    model, task, toolchain = key.split("/")
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    decisions: list[Accepted | Refused] = []

    def check(candidate: str, bound: Binding) -> Accepted | Refused:
        decisions.append(verify(candidate, (bound,)))
        return decisions[-1]

    guarded = guard(text, binding, check)
    first = decisions[0]
    entry: dict[str, object] = {
        "id": key,
        "archive": row,
        "as_written": "accepted" if isinstance(first, Accepted) else first.reason,
        "decision": "accepted" if isinstance(guarded.decision, Accepted) else "refused",
        "reason": guarded.decision.reason if isinstance(guarded.decision, Refused) else None,
        "guarded": list(guarded.guarded),
    }
    if isinstance(guarded.decision, Accepted):
        out = artifacts / key
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{toolchain}.guarded.ptx").write_text(guarded.text)
        (out / f"{toolchain}.guarded.binding.json").write_text(
            json.dumps(binding_record(guarded.binding), indent=1) + "\n"
        )
    return entry, guarded


def _defect(row: dict[str, object], artifacts: Path) -> dict[str, object]:
    """A defect's entry, with its runs on its trigger if the guarded text is accepted."""
    entry, guarded = _decided(row, artifacts)
    if not isinstance(guarded.decision, Accepted):
        return entry
    key = str(entry["id"])
    model, task, toolchain = key.split("/")
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    trigger = TRIGGERS[key]()
    entry["inputs"] = _inputs(trigger)
    entry["grid"] = trigger.grid
    entry["block"] = list(guarded.binding.block)
    entry["after"] = {
        name: [base64.b64encode(b).decode("ascii") for b in secrets]
        for name, secrets in trigger.after.items()
    }
    runs: list[Run] = []
    pool_bytes = LARGE_POOL if trigger.large else POOL
    with Executor(0, pool_bytes) as executor:
        with executor.tenant(TENANT):
            # the whole pool, so the reservation starts at the pool's first byte
            executor.reserve(pool_bytes)
            module = executor.register(guarded.text, guarded.binding)
        modes: tuple[Mode, ...] = ("guarded",) if trigger.large else ("guarded", "as_written")
        function: object | None = None
        if "as_written" in modes:
            (handle,) = call(driver.cuModuleLoadData(text.encode("ascii") + b"\0"))
            (function,) = call(driver.cuModuleGetFunction(handle, binding.kernel.encode("ascii")))
        for mode in modes:
            for secret in (0, 1):
                runs.append(_run(executor, module, function, key, trigger, mode, secret))
    entry["runs"] = runs
    return entry


def _far(row: dict[str, object], artifacts: Path) -> dict[str, object]:
    """A far trigger's entry, from a process of its own that may run for
    `LARGE_SECONDS`, since a guarded kernel may not end."""
    key = f"{row['model']}/{row['task']}/{row['toolchain']}"
    with tempfile.TemporaryDirectory() as scratch:
        part = Path(scratch) / "entry.json"
        command = [sys.executable, "-m", "eval.rq1_safety.guarded_defects", "--defect", key]
        command += ["--artifacts", str(artifacts), "--output", str(part)]
        try:
            subprocess.run(command, check=True, timeout=LARGE_SECONDS)
        except subprocess.TimeoutExpired:
            entry, _ = _decided(row, artifacts)
            entry["runs"] = []
            entry["unfinished"] = (
                f"a guarded launch did not end in {LARGE_SECONDS} s, and its process was ended"
            )
            return entry
        loaded = json.loads(part.read_text())
        if not isinstance(loaded, dict):
            raise TypeError(f"{key}: the entry is not an object")
        return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--defect", help="run one defect and write its entry: a far trigger's process"
    )
    parser.add_argument("--artifacts", type=Path, help="with --defect, the record's artifacts")
    args = parser.parse_args()
    if args.defect is not None:
        if args.artifacts is None:
            parser.error("--defect needs --artifacts")
        (row,) = (
            r for r in _archive() if f"{r['model']}/{r['task']}/{r['toolchain']}" == args.defect
        )
        args.output.write_text(json.dumps(_defect(row, args.artifacts)))
        return
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    rows = _archive()
    keys = [f"{r['model']}/{r['task']}/{r['toolchain']}" for r in rows]
    if sorted(keys) != sorted(TRIGGERS):
        raise ValueError("the triggers do not match the archive's defects")
    # the far triggers last: their unchecked launch faults, so they run guarded only
    rows.sort(key=lambda r: TRIGGERS[f"{r['model']}/{r['task']}/{r['toolchain']}"]().large)
    artifacts = args.output.with_suffix(args.output.suffix + ".artifacts")
    artifacts.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    entries = []
    escaped: list[str] = []
    for row in rows:
        far = TRIGGERS[f"{row['model']}/{row['task']}/{row['toolchain']}"]().large
        entries.append(_far(row, artifacts) if far else _defect(row, artifacts))
        e = entries[-1]
        runs = e.get("runs", [])
        if not isinstance(runs, list):
            raise TypeError("an entry's runs are a list")
        print(
            e["id"],
            e["decision"],
            [
                (r["mode"], r["secret"], r["launched"], len(r["changed"]), r["check_failed"])
                for r in runs
            ],
            e.get("unfinished", ""),
            flush=True,
        )
        guarded = [r for r in runs if r["mode"] == "guarded"]
        if any(not r["launched"] or r["changed"] for r in guarded):
            escaped.append(f"{e['id']}: a guarded run did not launch or changed a byte")
        if any(r["check_failed"] is False for r in guarded):
            escaped.append(f"{e['id']}: a guarded run on its trigger reported no failed check")
        if e.get("after") and guarded and guarded[0]["outputs"] != guarded[1]["outputs"]:
            escaped.append(f"{e['id']}: the guarded outputs depend on the neighbour's values")
    args.output.write_text(
        json.dumps(
            {
                "experiment": "corpus defects that the guard pass makes acceptable, on their triggers",
                "archive": str(E3_ARCHIVE),
                "library": str(formal_dir()),
                "started_at_utc": started,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "fill": list(FILL),
                "defects": entries,
            },
            indent=1,
        )
        + "\n"
    )
    if escaped:
        raise SystemExit("\n".join(escaped))


if __name__ == "__main__":
    main()
