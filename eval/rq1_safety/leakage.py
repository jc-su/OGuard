"""RQ1: does each system expose one tenant's data to another tenant's kernel on one GPU?

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq1_safety.leakage \\
        --output eval/records/leakage_<date>.json

Tenant A writes a known secret, then tenant B runs an attack kernel. The
experiment runs every attack under four systems and records the raw outcome of
each: the bytes B read compared with the secret, A's buffer after the attack,
the errors, whether data leaked, and, for OpGate, the system's own decision.

Systems:

* ``shared_unchecked``: one process, one CUDA context, one pool, with no
  checks. A and B are two tenants whose buffers share the pool; B launches its
  kernel directly with the driver. This is an engine that runs a tenant's
  kernel in its own context.
* ``mps``: A and B are separate client processes of one private CUDA MPS
  daemon (its own pipe and log directories under the record's inputs, static
  SM partitions, stopped at the end), so each has its own address space.
* ``guardian``: one process, one context, one pool, with each tenant's buffers
  in its own power-of-two aligned partition; B's kernels are patched by
  Guardian's address fencing (`eval.baselines.guardian`) and launched with the
  partition's base and mask, which confine every global access to the
  partition. Guardian fences global memory only; it has no notion of a buffer
  and does not clear shared memory, local memory, or a freed region.
* ``opgate``: the executor, catalog, and gate. A and B are two tenants of one
  executor; B's kernel is registered through `opgate.hosts.register`, which the
  gate refuses or the guard pass checks, and run through `opgate.hosts.execute`,
  which reports a failed check; `Executor.register` puts the zeroing prologue
  (design decision 46) at the start of each kernel that it loads. The
  configuration uses whatever the hosts and the gate do at run time, so a rerun
  after a later change shows the change.

Attacks, each a small CUDA C++ kernel that B runs after A wrote the secret:

* ``read_oob``: B reads one element past its buffer, into A's adjacent buffer.
* ``write_oob``: B writes one element past its buffer, into A's buffer.
* ``shared_leftover``: A fills its shared memory with the secret; B reads its
  shared memory without writing it first.
* ``local_leftover``: each thread of A's kernel fills a local array, in the
  thread's local memory, with the secret; each thread of B's kernel reads a
  local array without writing it first.

A's two fill kernels run on ``FILL_BLOCKS`` blocks, enough to reach every
multiprocessor and warp slot of the GPU, and the shared fill writes all the
static shared memory a block may declare, so that B's one block reads what A
left wherever the hardware places it, on an idle GPU or a busy one.
* ``freed_buffer``: A frees a buffer holding the secret; B allocates and reads
  before writing.
* ``context_fault``: B forms an illegal address while A runs; does A fail?
* ``index_race``: B's gather reads a data buffer through an out-of-range index
  that a racing write left in an index buffer. The gather must not read outside
  the data buffer; under OpGate the index input is a guarded launch whose check
  catches the out-of-range index before the gather runs.
* ``revoked_grant``: A grants a buffer that holds the secret read-only to B,
  then frees it (revoking the grant). B reads through the stale grant: under
  OpGate the catalog no longer holds the buffer, so B's read is refused.
* ``infinite_loop``: B runs a kernel that does not terminate (it spins on a
  flag). The question is availability: does it hold the GPU and block other
  work? Under OpGate the gate refuses the kernel, or the guard pass bounds its
  loop with an iteration limit so that it ends and answers ``check_failed``.
  Under the raw systems the host releases the flag after ``SPIN_WINDOW`` so the
  kernel always terminates.

Each cell runs in its own process (two for MPS) so that a context fault of one
attack does not reach another.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import multiprocessing as mp
import os
import platform
import shutil
import struct
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Barrier, Event
from pathlib import Path
from queue import Empty
from typing import Literal, assert_never, cast

import numpy as np

from eval import ROOT
from eval.baselines.guardian import patch_ptx
from eval.rq4_cost.consolidation import Mps, guardian_layout, guardian_partition_size
from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType
from opgate.catalog import (
    ALIGNMENT,
    BufferId,
    Fill,
    Grant,
    Launch,
    OperatorId,
    Step,
    TenantId,
    Write,
)
from opgate.executor import CheckFailedError, Executor, RefusedError
from opgate.executor.driver import address, call, compile_ptx, driver, integer
from opgate.frontends import Operator
from opgate.gate import formal_dir
from opgate.hosts import execute, register

__all__ = ["ATTACKS", "SYSTEMS", "determine_leak", "main", "matches_secret", "validate_cell"]

System = Literal["shared_unchecked", "mps", "guardian", "opgate"]
Attack = Literal[
    "read_oob",
    "write_oob",
    "shared_leftover",
    "local_leftover",
    "freed_buffer",
    "context_fault",
    "index_race",
    "revoked_grant",
    "infinite_loop",
]
SYSTEMS: tuple[System, ...] = ("shared_unchecked", "mps", "guardian", "opgate")
ATTACKS: tuple[Attack, ...] = (
    "read_oob",
    "write_oob",
    "shared_leftover",
    "local_leftover",
    "freed_buffer",
    "context_fault",
    "index_race",
    "revoked_grant",
    "infinite_loop",
)
# an availability attack is judged by whether the attacker affected other work,
# not by a leak: the context fault, and the non-terminating kernel
AVAILABILITY: tuple[Attack, ...] = ("context_fault", "infinite_loop")
SPIN_WINDOW = 1.0  # seconds a non-terminating kernel runs before the host releases it

WORDS = 32  # the 64-bit words of each tenant's buffer
FILL_BLOCKS = 512  # the blocks of A's fill kernels: every multiprocessor and warp slot
FILL_THREADS = 1024  # the threads of each block of A's local fill: every warp slot
SHARED_FILL_WORDS = 6144  # 48 KiB, all the static shared memory a block may declare
WORD = 8
BUFFER_BYTES = WORDS * WORD
RESERVATION = 1 << 19  # each tenant's half of the executor's pool of 1 MiB
SECRET = 0x5EC0FFEECAFEF00D  # A's known secret, in every word of its buffer
ATTACKER_VALUE = 0x1234567890ABCDEF  # the value B writes in the write attack
B_PATTERN = 0xB0B0B0B0B0B0B0B0  # B's own buffers, so a leak of A's secret is distinct
WILD = 1 << 30  # an index far past any buffer, for the context fault
SOURCES = (
    "eval/rq1_safety/leakage.py",
    "eval/baselines/guardian.py",
)

# every kernel takes 64-bit values, so each parameter is eight bytes and a raw
# launch builds its parameter list directly, without a toolchain's array ABI; each
# kernel is its own module, so the gate verifies one entry against one binding
_SOURCES: dict[str, str] = {
    "fill": (
        'extern "C" __global__ void fill(long long* buffer, long long value, long long n) {'
        " long long i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) buffer[i] = value; }\n"
    ),
    "read_one": (
        'extern "C" __global__ void read_one('
        "const long long* source, long long* out, long long index) {"
        " if (threadIdx.x == 0) out[0] = source[index]; }\n"
    ),
    "write_one": (
        'extern "C" __global__ void write_one('
        "long long* target, long long value, long long index) {"
        " if (threadIdx.x == 0) target[index] = value; }\n"
    ),
    "read_copy": (
        'extern "C" __global__ void read_copy('
        "const long long* buffer, long long* out, long long n) {"
        " long long i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = buffer[i]; }\n"
    ),
    # A's block fills all the static shared memory a block may declare, so the
    # secret is in whatever slice of the multiprocessor's shared memory B's
    # block receives
    "fill_shared": (
        'extern "C" __global__ void fill_shared(long long value, long long* sink) {'
        f" __shared__ long long store[{SHARED_FILL_WORDS}];"
        f' _Pragma("unroll") for (int j = 0; j < {SHARED_FILL_WORDS // WORDS}; ++j)'
        f" store[threadIdx.x + {WORDS} * j] = value;"
        " __syncthreads(); if (threadIdx.x == 0) sink[0] = store[0]; }\n"
    ),
    "read_shared": (
        'extern "C" __global__ void read_shared(long long* out) {'
        f" __shared__ long long store[{WORDS}]; __syncthreads();"
        " out[threadIdx.x] = store[threadIdx.x]; }\n"
    ),
    # each local array is volatile and read at the thread's index, so that NVRTC
    # keeps it in local memory: fill_local stores every word with st.local, and
    # read_local loads a word that its kernel never wrote with ld.local
    "fill_local": (
        'extern "C" __global__ void fill_local(long long value, long long* sink) {'
        f" volatile long long store[{WORDS}];"
        f" for (int k = 0; k < {WORDS}; ++k) store[k] = value;"
        f" sink[threadIdx.x % {WORDS}] = store[threadIdx.x % {WORDS}]; }}\n"
    ),
    "read_local": (
        'extern "C" __global__ void read_local(long long* out) {'
        f" volatile long long store[{WORDS}];"
        " out[threadIdx.x] = store[threadIdx.x]; }\n"
    ),
    "fault": (
        'extern "C" __global__ void fault(long long* buffer, long long index) {'
        " buffer[index] = buffer[index] + 1; }\n"
    ),
    "gather": (
        'extern "C" __global__ void gather('
        "const long long* data, const long long* index, long long* out, long long n) {"
        " long long i = blockIdx.x * blockDim.x + threadIdx.x;"
        " if (i < n) out[i] = data[index[i]]; }\n"
    ),
    "spin": (
        'extern "C" __global__ void spin(const volatile long long* flag) {'
        " while (flag[0] == 0) {} }\n"
    ),
}


# ---------------------------------------------------------------- pure logic


def matches_secret(data: bytes) -> bool:
    """Whether any 64-bit word of `data` equals the secret: a leak of A's bytes."""
    if len(data) % WORD:
        raise ValueError("the read bytes are not whole 64-bit words")
    return SECRET in struct.unpack(f"<{len(data) // WORD}Q", data)


def _all_secret(data: bytes) -> bool:
    return all(word == SECRET for word in struct.unpack(f"<{len(data) // WORD}Q", data))


def determine_leak(attack: Attack, *, read: bytes | None, victim_after: bytes | None) -> bool:
    """Whether the cell leaked A's data to B, or let B corrupt A's buffer: for a
    read, that B read the secret; for a write, that A's buffer no longer holds
    only the secret; a context fault is availability, not a leak."""
    match attack:
        case (
            "read_oob"
            | "shared_leftover"
            | "local_leftover"
            | "freed_buffer"
            | "index_race"
            | "revoked_grant"
        ):
            if read is None:
                raise ValueError(f"{attack}: no bytes were read")
            return matches_secret(read)
        case "write_oob":
            if victim_after is None:
                raise ValueError(f"{attack}: A's buffer after the write was not read")
            return not _all_secret(victim_after)
        case "context_fault" | "infinite_loop":
            return False
        case _ as unreachable:
            assert_never(unreachable)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def cell(
    system: System,
    attack: Attack,
    *,
    read: bytes | None = None,
    victim_after: bytes | None = None,
    a_failed: bool | None = None,
    error: str | None = None,
    decision: str | None = None,
    note: str,
) -> dict[str, object]:
    """One (system, attack) outcome as a raw record cell, with its leak decided."""
    return {
        "system": system,
        "attack": attack,
        "secret": f"{SECRET:016x}",
        "bytes_read": None if read is None else _b64(read),
        "matches_secret": None if read is None else matches_secret(read),
        "victim_after": None if victim_after is None else _b64(victim_after),
        "victim_unchanged": None if victim_after is None else _all_secret(victim_after),
        "a_failed": a_failed,
        "error": error,
        "leaked": determine_leak(attack, read=read, victim_after=victim_after),
        "decision": decision,
        "note": note,
    }


_CELL_FIELDS = frozenset(
    {
        "system",
        "attack",
        "secret",
        "bytes_read",
        "matches_secret",
        "victim_after",
        "victim_unchanged",
        "a_failed",
        "error",
        "leaked",
        "decision",
        "note",
    }
)


def _as_attack(value: object) -> Attack:
    if value not in ATTACKS:
        raise ValueError(f"unknown attack {value!r}")
    return cast(Attack, value)


def validate_cell(entry: dict[str, object]) -> None:
    """Check one cell's shape and that its summary agrees with its raw bytes: the
    leak flag, the secret match, and the integrity flag are what the raw bytes
    give. Used by the exporter and the tests."""
    if not _CELL_FIELDS <= set(entry) <= _CELL_FIELDS | {"mps"}:
        raise ValueError(f"a leakage cell has the fields {sorted(entry)}")
    if entry["system"] not in SYSTEMS:
        raise ValueError(f"a leakage cell names the system {entry['system']!r}")
    attack = _as_attack(entry["attack"])
    if entry["secret"] != f"{SECRET:016x}":
        raise ValueError("a leakage cell records another secret")
    read_field, victim_field = entry["bytes_read"], entry["victim_after"]
    read = base64.b64decode(read_field) if isinstance(read_field, str) else None
    victim = base64.b64decode(victim_field) if isinstance(victim_field, str) else None
    if entry["matches_secret"] != (None if read is None else matches_secret(read)):
        raise ValueError("a leakage cell's secret match differs from its read bytes")
    unchanged = None if victim is None else _all_secret(victim)
    if entry["victim_unchanged"] != unchanged:
        raise ValueError("a leakage cell's integrity flag differs from A's bytes")
    if entry["leaked"] != determine_leak(attack, read=read, victim_after=victim):
        raise ValueError("a leakage cell's leak flag differs from its raw bytes")
    if not isinstance(entry["note"], str) or not entry["note"]:
        raise ValueError("a leakage cell has no note")


# ---------------------------------------------------------------- the device and the kernels


def _device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to the index of exactly one GPU")
    return visible


def _arch(device: object) -> str:
    attribute = driver.CUdevice_attribute
    major = integer(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    )
    minor = integer(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
    )
    return f"compute_{major}{minor}"


def _compiled(name: str, arch: str) -> str:
    """One kernel compiled to the PTX of a single-entry module, without NVRTC's
    trailing null byte, which the driver tolerates but the gate's parser does
    not read."""
    return compile_ptx(_SOURCES[name], name, arch).decode("ascii").split("\x00", 1)[0]


def _secret_bytes() -> bytes:
    return np.full(WORDS, SECRET, dtype=np.uint64).tobytes()


def _pattern_bytes() -> bytes:
    return np.full(WORDS, B_PATTERN, dtype=np.uint64).tobytes()


class _Raw:
    """One raw CUDA context over one pool, for the single-process systems. A
    faulted context is poisoned, so its cleanup is best effort."""

    def __init__(self, pool_bytes: int) -> None:
        call(driver.cuInit(0))
        (self.device,) = call(driver.cuDeviceGet(0))
        (self.context,) = call(driver.cuDevicePrimaryCtxRetain(self.device))
        call(driver.cuCtxSetCurrent(self.context))
        (pointer,) = call(driver.cuMemAlloc(pool_bytes))
        self.pool = address(pointer)
        self._pointer = pointer
        self._arch = _arch(self.device)
        self._modules: list[object] = []
        self._functions: dict[tuple[str, bool], object] = {}

    def function(self, name: str, *, fenced: bool) -> object:
        if (name, fenced) not in self._functions:
            text = _compiled(name, self._arch)
            if fenced:
                text, _ = patch_ptx(text)
            (handle,) = call(driver.cuModuleLoadData(text.encode("ascii") + b"\0"))
            self._modules.append(handle)
            (function,) = call(driver.cuModuleGetFunction(handle, name.encode("ascii")))
            self._functions[(name, fenced)] = function
        return self._functions[(name, fenced)]

    def upload(self, offset: int, data: bytes) -> None:
        call(driver.cuMemcpyHtoD(driver.CUdeviceptr(self.pool + offset), data, len(data)))

    def download(self, offset: int, nbytes: int) -> bytes:
        data = bytearray(nbytes)
        call(driver.cuMemcpyDtoH(data, driver.CUdeviceptr(self.pool + offset), nbytes))
        return bytes(data)

    def clear(self, offset: int, nbytes: int) -> None:
        call(driver.cuMemsetD8(driver.CUdeviceptr(self.pool + offset), 0, nbytes))

    def launch(
        self,
        function: object,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        values: tuple[int, ...],
    ) -> None:
        types = tuple(ctypes.c_uint64 for _ in values)
        call(driver.cuLaunchKernel(function, *grid, *block, 0, 0, (values, types), 0))

    def synchronize(self) -> None:
        call(driver.cuCtxSynchronize())

    def run_until_released(self, function: object, flag_offset: int, window: float) -> bool:
        """Launch the spin kernel on a stream of its own, with the flag at
        `flag_offset` set to zero, and watch it for `window` seconds. The kernel
        does not end on its own, so the stream stays busy; return whether it was
        still running at the window's end (it was: a non-terminating kernel holds
        the GPU). Then set the flag on a second stream, which runs beside the
        spinning kernel and ends it, and wait for both, so the launch always
        terminates whatever happened on the device."""
        flag = driver.CUdeviceptr(self.pool + flag_offset)
        call(driver.cuMemsetD8(flag, 0, WORD))
        call(driver.cuCtxSynchronize())
        (stream,) = call(driver.cuStreamCreate(0))
        try:
            call(
                driver.cuLaunchKernel(
                    function,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    stream,
                    ((self.pool + flag_offset,), (ctypes.c_uint64,)),
                    0,
                )
            )
            blocked = True
            deadline = time.monotonic() + window
            while time.monotonic() < deadline:
                (status,) = driver.cuStreamQuery(stream)
                if status == driver.CUresult.CUDA_SUCCESS:
                    blocked = False
                    break
            return blocked
        finally:
            (release,) = call(driver.cuStreamCreate(0))
            try:
                call(driver.cuMemsetD8Async(flag, 1, WORD, release))
                call(driver.cuStreamSynchronize(release))
                call(driver.cuStreamSynchronize(stream))
            finally:
                call(driver.cuStreamDestroy(release))
                call(driver.cuStreamDestroy(stream))

    def close(self) -> None:
        for release in (
            *(lambda handle=handle: driver.cuModuleUnload(handle) for handle in self._modules),
            lambda: driver.cuMemFree(self._pointer),
            lambda: driver.cuDevicePrimaryCtxRelease(self.device),
        ):
            try:
                call(release())
            except Exception:  # noqa: BLE001 - a faulted context rejects cleanup
                pass


def _append(values: tuple[int, ...], fence: tuple[int, int] | None) -> tuple[int, ...]:
    """A launch's parameters with the partition's base and mask appended when the
    kernel is fenced."""
    return values if fence is None else (*values, fence[0], fence[1])


# ---------------------------------------------------------------- single-process attacks


def _raw_attack(attack: Attack, guardian: bool) -> dict[str, object]:
    """Run one attack under the shared-context or the guardian system in a raw
    context and return its outcome cell. A and B are in one context; for
    guardian each has a power-of-two aligned partition and B's kernels are
    fenced into it."""
    system: System = "guardian" if guardian else "shared_unchecked"
    size = guardian_partition_size(4 * ALIGNMENT) if guardian else 0
    raw = _Raw(max(1 << 20, size * 4))
    try:
        if guardian:
            _, bases = guardian_layout(raw.pool, size, 2)
            a_base, b_base = bases[0] - raw.pool, bases[1] - raw.pool
            a_fence: tuple[int, int] | None = (bases[0], size - 1)
            b_fence: tuple[int, int] | None = (bases[1], size - 1)
            far = size // WORD  # an index that leaves B's partition; the fence wraps it back
        else:
            a_base, b_base = ALIGNMENT, 0  # A is adjacent to B, just past it
            a_fence = b_fence = None
            far = WORDS  # one element past B's buffer, into A's
        return _run_raw(raw, attack, system, a_base, b_base, a_fence, b_fence, far)
    finally:
        raw.close()


def _run_raw(
    raw: _Raw,
    attack: Attack,
    system: System,
    a_base: int,
    b_base: int,
    a_fence: tuple[int, int] | None,
    b_fence: tuple[int, int] | None,
    far: int,
) -> dict[str, object]:
    one, words = (1, 1, 1), (WORDS, 1, 1)
    fill_grid, fill_block = (FILL_BLOCKS, 1, 1), (FILL_THREADS, 1, 1)
    match attack:
        case "read_oob":
            out_base = b_base + 2 * BUFFER_BYTES
            raw.upload(a_base, _secret_bytes())
            raw.upload(b_base, _pattern_bytes())
            raw.clear(out_base, WORD)
            read_one = raw.function("read_one", fenced=b_fence is not None)
            raw.launch(
                read_one, one, one, _append((raw.pool + b_base, raw.pool + out_base, far), b_fence)
            )
            raw.synchronize()
            note = (
                "the fence confined B's read to its partition"
                if b_fence is not None
                else "B read one element past its buffer into A's adjacent buffer"
            )
            return cell(system, attack, read=raw.download(out_base, WORD), note=note)
        case "write_oob":
            raw.upload(a_base, _secret_bytes())
            raw.upload(b_base, _pattern_bytes())
            write_one = raw.function("write_one", fenced=b_fence is not None)
            raw.launch(
                write_one, one, one, _append((raw.pool + b_base, ATTACKER_VALUE, far), b_fence)
            )
            raw.synchronize()
            note = (
                "the fence confined B's write to its partition; A's buffer is unchanged"
                if b_fence is not None
                else "B wrote one element past its buffer into A's buffer"
            )
            return cell(system, attack, victim_after=raw.download(a_base, BUFFER_BYTES), note=note)
        case "shared_leftover":
            # both modules are loaded before either launch, as an engine loads a
            # tenant's kernel at registration; a module loaded between the two
            # launches can make the driver allocate the context's memory anew
            out_base = b_base + 2 * BUFFER_BYTES
            fill_shared = raw.function("fill_shared", fenced=a_fence is not None)
            read_shared = raw.function("read_shared", fenced=b_fence is not None)
            raw.clear(out_base, BUFFER_BYTES)
            raw.launch(fill_shared, fill_grid, words, _append((SECRET, raw.pool + a_base), a_fence))
            raw.synchronize()
            raw.launch(read_shared, one, words, _append((raw.pool + out_base,), b_fence))
            raw.synchronize()
            note = "B read its shared memory without writing it first; it is not cleared"
            return cell(system, attack, read=raw.download(out_base, BUFFER_BYTES), note=note)
        case "local_leftover":
            # Guardian fences global accesses alone, so the local accesses run
            # unfenced; both modules are loaded before either launch, since the
            # driver sizes the context's local memory by the kernels it loads
            out_base = b_base + 2 * BUFFER_BYTES
            fill_local = raw.function("fill_local", fenced=a_fence is not None)
            read_local = raw.function("read_local", fenced=b_fence is not None)
            raw.clear(out_base, BUFFER_BYTES)
            raw.launch(
                fill_local, fill_grid, fill_block, _append((SECRET, raw.pool + a_base), a_fence)
            )
            raw.synchronize()
            raw.launch(read_local, one, words, _append((raw.pool + out_base,), b_fence))
            raw.synchronize()
            note = "B read its local memory without writing it first; it is not cleared"
            return cell(system, attack, read=raw.download(out_base, BUFFER_BYTES), note=note)
        case "freed_buffer":
            out_base = b_base + 2 * BUFFER_BYTES
            raw.upload(a_base, _secret_bytes())
            if a_fence is not None:
                raw.upload(b_base, _pattern_bytes())
                reuse, note = b_base, "B's partition never held A's freed buffer"
            else:
                # the pool allocator reuses A's freed region for B with no clear
                reuse, note = a_base, "B allocated A's freed region and read it before writing"
            raw.clear(out_base, BUFFER_BYTES)
            read_copy = raw.function("read_copy", fenced=b_fence is not None)
            raw.launch(
                read_copy,
                one,
                words,
                _append((raw.pool + reuse, raw.pool + out_base, WORDS), b_fence),
            )
            raw.synchronize()
            return cell(system, attack, read=raw.download(out_base, BUFFER_BYTES), note=note)
        case "context_fault":
            raw.upload(a_base, _secret_bytes())
            fill = raw.function("fill", fenced=a_fence is not None)
            raw.launch(fill, one, words, _append((raw.pool + a_base, SECRET, WORDS), a_fence))
            raw.synchronize()
            fault = raw.function("fault", fenced=b_fence is not None)
            index = far if b_fence is not None else WILD
            error: str | None = None
            try:
                raw.launch(fault, one, one, _append((raw.pool + b_base, index), b_fence))
                raw.synchronize()
            except Exception as caught:  # noqa: BLE001 - the fault is the observation
                error = str(caught)
            a_failed = False
            try:
                raw.launch(fill, one, words, _append((raw.pool + a_base, SECRET, WORDS), a_fence))
                raw.synchronize()
            except Exception as caught:  # noqa: BLE001 - whether A survives is the observation
                a_failed, error = True, error or str(caught)
            note = (
                "the fence confined B's illegal address to its partition; A kept running"
                if b_fence is not None
                else "B's illegal address faulted the shared context"
            )
            return cell(system, attack, a_failed=a_failed, error=error, note=note)
        case "index_race":
            out_base = b_base + 2 * BUFFER_BYTES
            index_base = b_base + 3 * BUFFER_BYTES
            raw.upload(a_base, _secret_bytes())
            raw.upload(b_base, _pattern_bytes())
            # the racing write left an out-of-range index in B's index buffer,
            # pointing at A's adjacent buffer; the fence wraps it back into B's
            raw.upload(index_base, struct.pack("<q", far))
            raw.clear(out_base, WORD)
            gather = raw.function("gather", fenced=b_fence is not None)
            raw.launch(
                gather,
                one,
                one,
                _append(
                    (raw.pool + b_base, raw.pool + index_base, raw.pool + out_base, 1), b_fence
                ),
            )
            raw.synchronize()
            note = (
                "the fence confined B's gather of an out-of-range index to its partition"
                if b_fence is not None
                else "B's gather read an out-of-range index into A's adjacent buffer"
            )
            return cell(system, attack, read=raw.download(out_base, WORD), note=note)
        case "revoked_grant":
            out_base = b_base + 2 * BUFFER_BYTES
            raw.upload(a_base, _secret_bytes())
            if b_fence is not None:
                raw.upload(b_base, _pattern_bytes())
                note = "the fence confined B's read of the revoked grant to its partition"
            else:
                note = "B read A's secret through a grant that A had revoked"
            raw.clear(out_base, BUFFER_BYTES)
            read_copy = raw.function("read_copy", fenced=b_fence is not None)
            # the stale grant still names A's buffer; without a buffer notion,
            # nothing stops B's read of it, and the fence only wraps the address
            raw.launch(
                read_copy,
                one,
                words,
                _append((raw.pool + a_base, raw.pool + out_base, WORDS), b_fence),
            )
            raw.synchronize()
            return cell(system, attack, read=raw.download(out_base, BUFFER_BYTES), note=note)
        case "infinite_loop":
            spin = raw.function("spin", fenced=False)
            blocked = raw.run_until_released(spin, b_base, SPIN_WINDOW)
            note = (
                "B's non-terminating kernel held the GPU until the host released it"
                if blocked
                else "B's kernel ended on its own"
            )
            return cell(system, attack, a_failed=blocked, note=note)
        case _ as unreachable:
            assert_never(unreachable)


def _raw_worker(attack: str, guardian: bool, results: mp.Queue[dict[str, object]]) -> None:
    results.put(_raw_attack(_as_attack(attack), guardian))


# ---------------------------------------------------------------- OpGate


_LONG = ElementType(8, True)
_N = sig.Affine.symbol(0)


def _operator(
    kernel: str,
    arch: str,
    arguments: tuple[sig.Argument, ...],
    sizes: tuple[sig.SizeSymbol, ...],
    block: tuple[int, int, int],
) -> Operator:
    signature = sig.Signature(sizes, arguments)
    text = _compiled(kernel, arch)
    return Operator(text, kernel, signature, Convention.C, block, GRID_LIMIT, sig.Affine(0))


def _elements(operator: Operator) -> dict[str, ElementType]:
    return {
        argument.name: _LONG
        for argument in operator.signature.arguments
        if not isinstance(argument, sig.Scalar)
    }


def _run_safe(
    executor: Executor, tenant: TenantId, operator: Operator, build: Callable[[OperatorId], Launch]
) -> None:
    """Register and run a kernel that must be accepted and whose checks, if the
    pass added any, must pass: A's own benign kernels."""
    registered = register(executor, operator.text, operator.binding(_elements(operator)))
    with executor.tenant(tenant):
        execute(executor, executor, Step(tenant, (build(registered.module.operator),)))


def _gate_launch(
    executor: Executor, tenant: TenantId, operator: Operator, build: Callable[[OperatorId], Launch]
) -> tuple[str, bool | None, str | None]:
    """Register B's kernel through the hosts, which the gate refuses or the guard
    pass checks, and run it through `execute`; the decision tag, whether a check
    failed, and the first error."""
    try:
        registered = register(executor, operator.text, operator.binding(_elements(operator)))
    except RefusedError as refused:
        return "refused", None, str(refused)
    guarded = registered.guarded is not None
    try:
        with executor.tenant(tenant):
            execute(executor, executor, Step(tenant, (build(registered.module.operator),)))
    except CheckFailedError as failed:
        return "accepted_with_checks:check_failed", True, str(failed)
    return ("accepted_with_checks" if guarded else "accepted"), (False if guarded else None), None


def _read(executor: Executor, tenant: TenantId, buffer: BufferId) -> bytes:
    with executor.tenant(tenant):
        return executor.read(tenant, buffer)


def _opgate_attack(attack: Attack) -> dict[str, object]:
    """Run one attack through the executor, catalog, and gate, A as tenant 1 and
    B as tenant 2, and return its outcome cell."""
    call(driver.cuInit(0))
    with Executor(0, 1 << 20) as executor:
        (device,) = call(driver.cuDeviceGet(0))
        return _run_opgate(executor, attack, _arch(device), TenantId(1), TenantId(2))


def _run_opgate(
    executor: Executor, attack: Attack, arch: str, a: TenantId, b: TenantId
) -> dict[str, object]:
    system: System = "opgate"
    # A's reservation is the pool's first half, B's the second
    for tenant in (a, b):
        with executor.tenant(tenant):
            executor.reserve(RESERVATION)
    match attack:
        case "read_oob":
            with executor.tenant(a):
                secret = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(a, (Write(secret, _secret_bytes(), None),)))
            with executor.tenant(b):
                source = executor.allocate(BUFFER_BYTES)
                out = executor.allocate(WORD)
                executor.execute(
                    Step(b, (Write(source, _pattern_bytes(), None), Fill(out, _LONG, 0)))
                )
            operator = _operator(
                "read_one",
                arch,
                (
                    sig.Span("source", "source", sig.Direction.READ, _N),
                    sig.Span("out", "out", sig.Direction.WRITE, sig.Affine(1)),
                    sig.Scalar("index", sig.Between(0, 2**63 - 1)),
                ),
                (sig.SizeSymbol("n", 1),),
                (1, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor,
                b,
                operator,
                lambda op: Launch(op, (1, 1, 1), (source, out), (WORDS,), (WORDS,), False),
            )
            note = f"the gate {_phrase(tag)} B's out-of-bounds read"
            return cell(
                system, attack, read=_read(executor, b, out), decision=tag, error=error, note=note
            )
        case "write_oob":
            with executor.tenant(a):
                secret = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(a, (Write(secret, _secret_bytes(), None),)))
            with executor.tenant(b):
                target = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(b, (Write(target, _pattern_bytes(), None),)))
            operator = _operator(
                "write_one",
                arch,
                (
                    sig.Span("target", "target", sig.Direction.WRITE, _N),
                    sig.Scalar("value", sig.Between(0, 2**64 - 1)),
                    sig.Scalar("index", sig.Between(0, 2**63 - 1)),
                ),
                (sig.SizeSymbol("n", 1),),
                (1, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor,
                b,
                operator,
                lambda op: Launch(
                    op, (1, 1, 1), (target,), (WORDS,), (ATTACKER_VALUE, WORDS), False
                ),
            )
            note = f"the gate {_phrase(tag)} B's out-of-bounds write; A's buffer is unchanged"
            return cell(
                system,
                attack,
                victim_after=_read(executor, a, secret),
                decision=tag,
                error=error,
                note=note,
            )
        case "shared_leftover":
            with executor.tenant(a):
                sink = executor.allocate(WORD)
            _run_safe(
                executor,
                a,
                _operator(
                    "fill_shared",
                    arch,
                    (
                        sig.Scalar("value", sig.Between(0, 2**64 - 1)),
                        sig.Span("sink", "sink", sig.Direction.WRITE, sig.Affine(1)),
                    ),
                    (),
                    (WORDS, 1, 1),
                ),
                lambda op: Launch(op, (FILL_BLOCKS, 1, 1), (sink,), (), (SECRET,), False),
            )
            with executor.tenant(b):
                out = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(b, (Fill(out, _LONG, 0),)))
            read_shared = _operator(
                "read_shared",
                arch,
                (sig.Span("out", "out", sig.Direction.WRITE, sig.Affine(WORDS)),),
                (),
                (WORDS, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor, b, read_shared, lambda op: Launch(op, (1, 1, 1), (out,), (), (), False)
            )
            read = _read(executor, b, out)
            if tag == "refused":
                note = "the gate refused B's kernel"
            elif matches_secret(read):
                note = "B read A's secret from shared memory that its kernel did not write"
            else:
                note = "the zeroing prologue cleared B's shared memory before B read it"
            return cell(system, attack, read=read, decision=tag, error=error, note=note)
        case "local_leftover":
            with executor.tenant(a):
                sink = executor.allocate(BUFFER_BYTES)
            _run_safe(
                executor,
                a,
                _operator(
                    "fill_local",
                    arch,
                    (
                        sig.Scalar("value", sig.Between(0, 2**64 - 1)),
                        sig.Span("sink", "sink", sig.Direction.WRITE, sig.Affine(WORDS)),
                    ),
                    (),
                    (FILL_THREADS, 1, 1),
                ),
                lambda op: Launch(op, (FILL_BLOCKS, 1, 1), (sink,), (), (SECRET,), False),
            )
            with executor.tenant(b):
                out = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(b, (Fill(out, _LONG, 0),)))
            read_local = _operator(
                "read_local",
                arch,
                (sig.Span("out", "out", sig.Direction.WRITE, sig.Affine(WORDS)),),
                (),
                (WORDS, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor, b, read_local, lambda op: Launch(op, (1, 1, 1), (out,), (), (), False)
            )
            read = _read(executor, b, out)
            if tag == "refused":
                note = "the gate refused B's kernel"
            elif matches_secret(read):
                note = "B read A's secret from local memory that its kernel did not write"
            else:
                note = "the zeroing prologue cleared B's local memory before B read it"
            return cell(system, attack, read=read, decision=tag, error=error, note=note)
        case "freed_buffer":
            with executor.tenant(a):
                freed = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(a, (Write(freed, _secret_bytes(), None),)))
                executor.free(freed)
                executor.unreserve()
            with executor.tenant(b):
                # B's new reservation is the region that A's held
                executor.unreserve()
                executor.reserve(RESERVATION)
                reused = executor.allocate(BUFFER_BYTES)
                out = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(b, (Fill(out, _LONG, 0),)))
            read_copy = _operator(
                "read_copy",
                arch,
                (
                    sig.Span("buffer", "buffer", sig.Direction.READ, _N),
                    sig.Span("out", "out", sig.Direction.WRITE, _N),
                    sig.Scalar("n", sig.SizeOf(_N)),
                ),
                (sig.SizeSymbol("n", 1),),
                (WORDS, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor,
                b,
                read_copy,
                lambda op: Launch(op, (1, 1, 1), (reused, out), (WORDS,), (), False),
            )
            note = "the catalog cleared the region before B read it"
            return cell(
                system, attack, read=_read(executor, b, out), decision=tag, error=error, note=note
            )
        case "context_fault":
            with executor.tenant(a):
                secret = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(a, (Write(secret, _secret_bytes(), None),)))
            with executor.tenant(b):
                buffer = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(b, (Write(buffer, _pattern_bytes(), None),)))
            fault = _operator(
                "fault",
                arch,
                (
                    sig.Span("buffer", "buffer", sig.Direction.READ_WRITE, _N),
                    sig.Scalar("index", sig.Between(0, 2**63 - 1)),
                ),
                (sig.SizeSymbol("n", 1),),
                (1, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor,
                b,
                fault,
                lambda op: Launch(op, (1, 1, 1), (buffer,), (WORDS,), (WILD,), False),
            )
            a_failed = False
            try:
                _read(executor, a, secret)
            except Exception as caught:  # noqa: BLE001 - whether A survives is the observation
                a_failed, error = True, error or str(caught)
            note = f"the gate {_phrase(tag)} B's illegal access; A kept running"
            return cell(system, attack, a_failed=a_failed, decision=tag, error=error, note=note)
        case "index_race":
            with executor.tenant(a):
                secret = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(a, (Write(secret, _secret_bytes(), None),)))
            with executor.tenant(b):
                data = executor.allocate(BUFFER_BYTES)
                index = executor.allocate(BUFFER_BYTES)
                out = executor.allocate(BUFFER_BYTES)
                # the racing write left an out-of-range index in B's index buffer
                poisoned = np.zeros(WORDS, dtype=np.int64)
                poisoned[0] = 1 << 20
                executor.execute(
                    Step(
                        b,
                        (
                            Write(data, _pattern_bytes(), None),
                            Write(index, poisoned.tobytes(), None),
                            Fill(out, _LONG, 0),
                        ),
                    )
                )
            gather = _operator(
                "gather",
                arch,
                (
                    sig.Span("data", "data", sig.Direction.READ, _N),
                    sig.Span("index", "index", sig.Direction.READ, _N),
                    sig.Span("out", "out", sig.Direction.WRITE, _N),
                    sig.Scalar("n", sig.SizeOf(_N)),
                ),
                (sig.SizeSymbol("n", 1),),
                (1, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor,
                b,
                gather,
                lambda op: Launch(op, (1, 1, 1), (data, index, out), (WORDS,), (), False),
            )
            note = f"the gate {_phrase(tag)} B's gather of an out-of-range index"
            return cell(
                system, attack, read=_read(executor, b, out), decision=tag, error=error, note=note
            )
        case "revoked_grant":
            with executor.tenant(a):
                granted = executor.allocate(BUFFER_BYTES)
                executor.execute(Step(a, (Write(granted, _secret_bytes(), None),)))
                executor.execute(Step(a, (Grant(granted, b),)))
                executor.free(granted)  # revoking the grant: the buffer leaves the catalog
            read = bytes(BUFFER_BYTES)
            decision = "accepted"
            error = None
            with executor.tenant(b):
                try:
                    read = executor.read(b, granted)
                except RefusedError as refused:
                    decision, error = "refused", str(refused)
            note = (
                "the catalog refused B's read through the revoked grant"
                if decision == "refused"
                else "B read through the revoked grant"
            )
            return cell(system, attack, read=read, decision=decision, error=error, note=note)
        case "infinite_loop":
            with executor.tenant(b):
                flag = executor.allocate(WORD)
                executor.execute(Step(b, (Fill(flag, _LONG, 0),)))
            spin = _operator(
                "spin",
                arch,
                (sig.Span("flag", "flag", sig.Direction.READ, sig.Affine(1)),),
                (),
                (1, 1, 1),
            )
            tag, failed, error = _gate_launch(
                executor, b, spin, lambda op: Launch(op, (1, 1, 1), (flag,), (), (), False)
            )
            note = f"the gate {_phrase(tag)} B's non-terminating kernel; it did not block A"
            return cell(system, attack, a_failed=False, decision=tag, error=error, note=note)
        case _ as unreachable:
            assert_never(unreachable)


def _phrase(tag: str) -> str:
    if tag == "refused":
        return "refused"
    if tag == "accepted_with_checks:check_failed":
        return "checked and failed"
    return "accepted"


def _opgate_worker(attack: str, results: mp.Queue[dict[str, object]]) -> None:
    results.put(_opgate_attack(_as_attack(attack)))


# ---------------------------------------------------------------- MPS


def _mps_a_worker(
    attack: str, ready: Barrier, done: Event, results: mp.Queue[dict[str, object]]
) -> None:
    """A under MPS, in its own address space: write the secret, then after B is
    done read A's own buffer and confirm A still runs. A's buffer and survival
    are what the write and the fault cells need from A."""
    call(driver.cuInit(0))
    with Executor(0, 1 << 20) as executor:
        tenant = TenantId(1)
        with executor.tenant(tenant):
            executor.reserve(RESERVATION)
            buffer = executor.allocate(BUFFER_BYTES)
            executor.execute(Step(tenant, (Write(buffer, _secret_bytes(), None),)))
        ready.wait()
        done.wait(timeout=120)
        survived = True
        after = _secret_bytes()
        try:
            after = _read(executor, tenant, buffer)
        except Exception:  # noqa: BLE001 - whether A survives is the observation
            survived = False
        results.put({"victim_after": _b64(after), "survived": survived})


def _mps_b_worker(
    attack: str, ready: Barrier, done: Event, results: mp.Queue[dict[str, object]]
) -> None:
    """B under MPS, in its own address space with no secret present: run the
    attack on its own buffers. A's secret is unreachable."""
    outcome: dict[str, object]
    try:
        ready.wait()
        outcome = _mps_b_observation(_as_attack(attack))
    finally:
        done.set()
    results.put(outcome)


def _mps_b_observation(attack: Attack) -> dict[str, object]:
    """B's own observation under MPS: the bytes it read from its own memory, or
    the error of a fault of its own context."""
    raw = _Raw(1 << 20)
    try:
        one, words = (1, 1, 1), (WORDS, 1, 1)
        b_base, out_base, neighbour = 0, BUFFER_BYTES, 2 * BUFFER_BYTES
        match attack:
            case "read_oob":
                raw.upload(b_base, _pattern_bytes())
                raw.upload(neighbour, _pattern_bytes())
                raw.clear(out_base, WORD)
                read_one = raw.function("read_one", fenced=False)
                raw.launch(read_one, one, one, (raw.pool + b_base, raw.pool + out_base, WORDS))
                raw.synchronize()
                return {"read": _b64(raw.download(out_base, WORD))}
            case "write_oob":
                raw.upload(b_base, _pattern_bytes())
                write_one = raw.function("write_one", fenced=False)
                raw.launch(write_one, one, one, (raw.pool + b_base, ATTACKER_VALUE, WORDS))
                raw.synchronize()
                return {"read": None}
            case "shared_leftover":
                raw.clear(out_base, BUFFER_BYTES)
                read_shared = raw.function("read_shared", fenced=False)
                raw.launch(read_shared, one, words, (raw.pool + out_base,))
                raw.synchronize()
                return {"read": _b64(raw.download(out_base, BUFFER_BYTES))}
            case "local_leftover":
                raw.clear(out_base, BUFFER_BYTES)
                read_local = raw.function("read_local", fenced=False)
                raw.launch(read_local, one, words, (raw.pool + out_base,))
                raw.synchronize()
                return {"read": _b64(raw.download(out_base, BUFFER_BYTES))}
            case "freed_buffer":
                raw.upload(b_base, _pattern_bytes())
                raw.clear(out_base, BUFFER_BYTES)
                read_copy = raw.function("read_copy", fenced=False)
                raw.launch(read_copy, one, words, (raw.pool + b_base, raw.pool + out_base, WORDS))
                raw.synchronize()
                return {"read": _b64(raw.download(out_base, BUFFER_BYTES))}
            case "context_fault":
                raw.upload(b_base, _pattern_bytes())
                fault = raw.function("fault", fenced=False)
                error: str | None = None
                try:
                    raw.launch(fault, one, one, (raw.pool + b_base, WILD))
                    raw.synchronize()
                except Exception as caught:  # noqa: BLE001 - the fault is the observation
                    error = str(caught)
                return {"read": None, "error": error}
            case "index_race":
                index_base = b_base + 3 * BUFFER_BYTES
                raw.upload(b_base, _pattern_bytes())
                raw.upload(index_base, struct.pack("<q", 0))  # B's own in-range index
                raw.clear(out_base, WORD)
                gather = raw.function("gather", fenced=False)
                raw.launch(
                    gather,
                    one,
                    one,
                    (raw.pool + b_base, raw.pool + index_base, raw.pool + out_base, 1),
                )
                raw.synchronize()
                return {"read": _b64(raw.download(out_base, WORD))}
            case "revoked_grant":
                raw.upload(b_base, _pattern_bytes())
                raw.clear(out_base, BUFFER_BYTES)
                read_copy = raw.function("read_copy", fenced=False)
                raw.launch(read_copy, one, words, (raw.pool + b_base, raw.pool + out_base, WORDS))
                raw.synchronize()
                return {"read": _b64(raw.download(out_base, BUFFER_BYTES))}
            case "infinite_loop":
                spin = raw.function("spin", fenced=False)
                raw.run_until_released(spin, b_base, SPIN_WINDOW)
                return {"read": None}
            case _ as unreachable:
                assert_never(unreachable)
    finally:
        raw.close()


def _mps_cell(attack: Attack, partitions: tuple[str, ...]) -> dict[str, object]:
    """One MPS cell: A and B as separate client processes of the private daemon,
    each in its own static SM partition. B's attack reaches only its own address
    space, so A's secret does not leak; A's buffer and survival come from A's own
    process."""
    context = mp.get_context("spawn")
    ready = context.Barrier(2)
    done = context.Event()
    a_results: mp.Queue[dict[str, object]] = context.Queue()
    b_results: mp.Queue[dict[str, object]] = context.Queue()
    a = context.Process(target=_mps_a_worker, args=(attack, ready, done, a_results))
    b = context.Process(target=_mps_b_worker, args=(attack, ready, done, b_results))
    # a static MPS client names its partition before its first CUDA call, which
    # importing the executor makes as the spawned process starts
    if "CUDA_MPS_SM_PARTITION" in os.environ:
        raise RuntimeError("CUDA_MPS_SM_PARTITION was already set")
    for number, worker in enumerate((a, b)):
        if partitions:
            os.environ["CUDA_MPS_SM_PARTITION"] = partitions[number]
        try:
            worker.start()
        finally:
            os.environ.pop("CUDA_MPS_SM_PARTITION", None)
    b_outcome = _collect_one(b_results, [a, b], "MPS B client")
    a_outcome = _collect_one(a_results, [a, b], "MPS A client")
    a.join(timeout=60)
    b.join(timeout=60)
    if a.exitcode != 0 or b.exitcode != 0:
        raise RuntimeError(f"an MPS client exited with {a.exitcode}/{b.exitcode}")
    note = "under MPS A and B have separate address spaces; B's attack reached only its own memory"
    read_field = b_outcome.get("read")
    read = base64.b64decode(read_field) if isinstance(read_field, str) else None
    victim_field = a_outcome.get("victim_after")
    victim = base64.b64decode(victim_field) if isinstance(victim_field, str) else None
    a_failed = (a_outcome.get("survived") is not True) if attack in AVAILABILITY else None
    if attack in AVAILABILITY:
        note = (
            note
            if attack == "context_fault"
            else "under MPS B's non-terminating kernel ran in its own context and partition; "
            "A kept running"
        )
        return cell(
            "mps",
            attack,
            a_failed=a_failed,
            error=cast(str | None, b_outcome.get("error")),
            note=note,
        )
    if attack == "write_oob":
        return cell("mps", attack, victim_after=victim, note=note)
    return cell("mps", attack, read=read, note=note)


# ---------------------------------------------------------------- the run


def _environment() -> dict[str, object]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
            "-i",
            _device(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu": query.stdout.strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "formal_library": str(formal_dir()),
    }


def _collect_one(
    results: mp.Queue[dict[str, object]],
    workers: list[BaseProcess],
    where: str,
    timeout: float = 300.0,
) -> dict[str, object]:
    """One result from `results`, failing fast if a worker dies without producing
    it rather than waiting the whole timeout."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return results.get(timeout=1.0)
        except Empty:
            if all(worker.exitcode is not None for worker in workers):
                raise RuntimeError(f"{where}: a worker exited without a result") from None
            if time.monotonic() >= deadline:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                raise TimeoutError(f"{where}: no result within {timeout:g} seconds") from None


def _spawned(target: Callable[..., None], *args: object) -> dict[str, object]:
    """Run `target` in a fresh spawned process and collect its one cell, so a
    context fault of one attack does not reach another."""
    context = mp.get_context("spawn")
    results: mp.Queue[dict[str, object]] = context.Queue()
    worker = context.Process(target=target, args=(*args, results))
    worker.start()
    outcome = _collect_one(results, [worker], "leakage worker")
    worker.join(timeout=60)
    if worker.exitcode not in (0, None):
        raise RuntimeError(f"a leakage worker exited with {worker.exitcode}")
    return outcome


def run(system: System, attack: Attack, mps_logs: Path) -> dict[str, object]:
    """Run one (system, attack) cell and return its outcome."""
    match system:
        case "shared_unchecked":
            return _spawned(_raw_worker, attack, False)
        case "guardian":
            return _spawned(_raw_worker, attack, True)
        case "opgate":
            return _spawned(_opgate_worker, attack)
        case "mps":
            with Mps(static_tenants=2) as mps:
                try:
                    outcome = _mps_cell(attack, mps.partitions)
                finally:
                    shutil.copytree(mps.directory / "log", mps_logs / attack)
                return {**outcome, "mps": mps.metadata}
        case _ as unreachable:
            assert_never(unreachable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", action="append", choices=SYSTEMS, dest="systems")
    parser.add_argument("--attack", action="append", choices=ATTACKS, dest="attacks")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    systems = tuple(args.systems or SYSTEMS)
    attacks = tuple(args.attacks or ATTACKS)
    mps_logs = args.output.with_suffix(args.output.suffix + ".inputs") / "mps_logs"
    mps_logs.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    cells: list[dict[str, object]] = []
    for system in systems:
        for attack in attacks:
            outcome = run(system, attack, mps_logs)
            validate_cell(outcome)
            cells.append(outcome)
            print(
                json.dumps(
                    {key: outcome[key] for key in ("system", "attack", "leaked", "decision")}
                ),
                flush=True,
            )
    record = {
        "experiment": "leakage: one tenant's data reaching another tenant's kernel on one GPU",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        "secret": f"{SECRET:016x}",
        "words": WORDS,
        "systems": list(systems),
        "attacks": list(attacks),
        "sources": {name: (ROOT / name).read_text() for name in SOURCES},
        "cells": cells,
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
