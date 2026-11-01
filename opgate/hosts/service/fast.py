"""The owner's loop over the mailboxes, compiled with Numba to run without the interpreter.

The owner thread enters `serve` with tables that it builds from its state
(`Tables.of`): each mailbox's shared memory, doorbells, last sequence number,
and prepared steps, and, for each run whose next use the executor can describe
as `Hot`, the bounds of its scalars and its driver calls as integers. The loop
then answers the uses that end, takes each new request whose step is hot and
whose scalars lie in their bounds, and submits its driver calls on the
tenant's stream, as `Executor.submit` would after the catalog's memo decides
it. It returns to Python, with no use in progress, when the server queues
work, when a request needs Python (a step that is not hot, a scalar out of
its bounds, a malformed record), or when the driver fails. Nothing in the
catalog changes while the loop runs, since only the owner changes it, so the
tables stay true until the loop returns.

A request is copied out of the shared memory with `memcpy` before any field
is read, and a response is written with its sequence number last. The loop
calls the driver and the C library through function pointers, so Numba
compiles it when the service starts.
"""

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from opgate.executor import Hot
from opgate.executor.driver import procedure
from opgate.hosts.service.dataplane import (
    ACQUIRE,
    MAX_SCALARS,
    RELEASE,
    REQUEST,
    RESPONSE_AT,
    load_sequence,
    publish_sequence,
)

if TYPE_CHECKING:
    from opgate.hosts.service.dataplane import Mailbox

__all__ = ["Exit", "Tables", "serve"]

MAX_PREPARED = 64  # the steps one session may prepare
MAX_FILLS = 5  # the fills of a hot step
_RECORD = REQUEST.size  # a request record's bytes
_WORDS = _RECORD // 8
_NOT_READY = 600  # CUDA_ERROR_NOT_READY
_ACCEPTED = 0  # dataplane.Status.ACCEPTED
_CHECK_FAILED = 4  # dataplane.Status.CHECK_FAILED
_QUEUED, _PYTHON, _DRIVER, _IPC = 0, 1, 2, 3  # Exit's values, which the compiled loop returns

_u64, _u32, _ptr, _int = ctypes.c_uint64, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int
_size, _ssize = ctypes.c_size_t, ctypes.c_ssize_t
_libc = ctypes.CDLL(None, use_errno=True)


_Argument = type[ctypes._SimpleCData]


def _driver(name: str, *arguments: _Argument) -> Callable[..., int]:
    return ctypes.CFUNCTYPE(_int, *arguments)(procedure(name))


def _c(name: str, result: _Argument, *arguments: _Argument) -> Callable[..., int]:
    found = ctypes.cast(getattr(_libc, name), ctypes.c_void_p).value
    if found is None:
        raise RuntimeError(f"the C library has no {name}")
    return ctypes.CFUNCTYPE(result, *arguments)(found)


# the driver's functions of a use, and libc's, which the compiled loop calls directly
cuMemsetD8Async = _driver("cuMemsetD8Async", _u64, ctypes.c_ubyte, _size, _ptr)
cuMemsetD16Async = _driver("cuMemsetD16Async", _u64, ctypes.c_ushort, _size, _ptr)
cuMemsetD32Async = _driver("cuMemsetD32Async", _u64, _u32, _size, _ptr)
cuLaunchKernel = _driver(
    "cuLaunchKernel", _ptr, _u32, _u32, _u32, _u32, _u32, _u32, _u32, _ptr, _ptr, _ptr
)
cuMemcpyDtoHAsync = _driver("cuMemcpyDtoHAsync", _ptr, _u64, _size, _ptr)
cuEventRecord = _driver("cuEventRecord", _ptr, _ptr)
cuEventQuery = _driver("cuEventQuery", _ptr)
cuEventSynchronize = _driver("cuEventSynchronize", _ptr)
memcpy = _c("memcpy", _ptr, _ptr, _ptr, _size)
read = _c("read", _ssize, _int, _ptr, _size)
write = _c("write", _ssize, _int, _ptr, _size)
poll = _c("poll", _int, _ptr, ctypes.c_ulong, _int)


class Exit(IntEnum):
    """Why `serve` returned."""

    QUEUED = 0  # the server queued work
    PYTHON = 1  # a mailbox's request needs the Python path
    DRIVER = 2  # a driver call failed with the returned code
    IPC = 3  # a response notification could not be written


# the columns of `Tables.mail`
ADDRESS, REQUEST_FD, RESPONSE_FD, SEQUENCE, RUNNING, RUNNING_SEQUENCE = range(6)
# the columns of `Tables.runs`
(SCALARS, FILLS, FUNCTION, GX, GY, GZ, BX, BY, BZ, SHARED, STREAM, EVENT, HOST, SOURCE) = range(14)
NBYTES, POINTERS, STATUS_HOST, STATUS_SOURCE = 14, 15, 16, 17


@dataclass(frozen=True)
class Tables:
    """The loop's view of the owner's state, as arrays of integers."""

    mail: np.ndarray  # int64 [mailboxes, 6]
    steps: np.ndarray  # int64 [mailboxes, MAX_PREPARED]: a hot run's row, or -1
    runs: np.ndarray  # int64 [runs, 18]
    fills: np.ndarray  # int64 [runs, MAX_FILLS, 4]: width, address, value, count
    signed: np.ndarray  # uint8 [runs, MAX_SCALARS]
    lower: np.ndarray  # int64 [runs, MAX_SCALARS], two's complement for an unsigned bound
    upper: np.ndarray  # int64 [runs, MAX_SCALARS], likewise
    slots: np.ndarray  # int64 [runs, MAX_SCALARS]: the address of each scalar's slot
    masks: np.ndarray  # int64 [runs, MAX_SCALARS], two's complement
    uses: np.ndarray  # int64 [runs]: the uses that the loop submitted
    failed_checks: np.ndarray  # int64 [runs]: failed runtime checks
    polled: np.ndarray  # int32 [2 * (mailboxes + 1)]: struct pollfd, the wakeup first
    scratch: np.ndarray  # uint64 [_WORDS + 2]

    @staticmethod
    def of(
        wakeup: int, mailboxes: Sequence["Mailbox"], steps: Sequence[Sequence[Hot | None]]
    ) -> "Tables":
        """The tables of `mailboxes`, each with the hot form of each step it prepared,
        or None; a run's row in `runs` is its place among the hot forms, in order."""
        hot = [(m, k, h) for m, forms in enumerate(steps) for k, h in enumerate(forms) if h]
        mail = np.zeros((len(mailboxes), 6), np.int64)
        table = np.full((len(mailboxes), MAX_PREPARED), -1, np.int64)
        count = len(hot)
        runs = np.zeros((count, 18), np.int64)
        fills = np.zeros((count, MAX_FILLS, 4), np.int64)
        signed = np.zeros((count, MAX_SCALARS), np.uint8)
        lower = np.zeros((count, MAX_SCALARS), np.int64)
        upper = np.zeros((count, MAX_SCALARS), np.int64)
        slots = np.zeros((count, MAX_SCALARS), np.int64)
        masks = np.zeros((count, MAX_SCALARS), np.int64)
        for row, mailbox in enumerate(mailboxes):
            mail[row] = (
                mailbox.address,
                mailbox.request,
                mailbox.response,
                _word(mailbox.sequence),  # a client's 64-bit number, as int64 bits
                -1,
                0,
            )
        for row, (m, k, h) in enumerate(hot):
            if len(h.fills) > MAX_FILLS or len(h.bounds) > MAX_SCALARS:
                continue  # its uses take the Python path
            table[m, k] = row
            runs[row] = (
                len(h.bounds),
                len(h.fills),
                h.function,
                *h.grid,
                *h.threads,
                h.shared,
                h.stream,
                h.event,
                h.host,
                h.source,
                h.nbytes,
                h.pointers,
                h.status_host,
                h.status_source,
            )
            for f, fill in enumerate(h.fills):
                fills[row, f] = (fill.width, fill.address, fill.value, fill.count)
            base = ctypes.addressof(h.block)
            for j, ((low, high), parameter) in enumerate(zip(h.bounds, h.parameters, strict=True)):
                signed[row, j] = low < 0
                lower[row, j] = _word(low)
                upper[row, j] = _word(high)
                slots[row, j] = base + 8 * parameter
                masks[row, j] = _word(h.masks[parameter])
        polled = np.zeros(2 * (len(mailboxes) + 1), np.int32)
        polled[0], polled[1] = wakeup, 1  # POLLIN
        for row, mailbox in enumerate(mailboxes):
            polled[2 * row + 2], polled[2 * row + 3] = mailbox.request, 1
        return Tables(
            mail,
            table,
            runs,
            fills,
            signed,
            lower,
            upper,
            slots,
            masks,
            np.zeros(count, np.int64),
            np.zeros(count, np.int64),
            polled,
            np.zeros(_WORDS + 2, np.uint64),
        )


def _word(value: int) -> int:
    """`value`, of 64 bits signed or unsigned, as the int64 of the same bits."""
    word = value % 2**64
    return word - 2**64 if word >= 2**63 else word


def serve(tables: Tables, wakeup: int) -> tuple[Exit, int]:
    """Serve the mailboxes until Python is needed; the reason, and the mailbox's
    row for `Exit.PYTHON` or the driver's code for `Exit.DRIVER`."""
    t = tables
    code, index = _serve(
        t.mail,
        t.steps,
        t.runs,
        t.fills,
        t.signed,
        t.lower,
        t.upper,
        t.slots,
        t.masks,
        t.uses,
        t.failed_checks,
        t.polled,
        t.scratch,
        wakeup,
    )
    return Exit(code), int(index)


@njit(nogil=True, cache=True)
def in_bounds(word: np.uint64, signed: bool, lower: np.int64, upper: np.int64) -> np.bool_:
    """Whether a scalar's 8-byte `word` lies in `[lower, upper]`, read as signed or
    unsigned, as the catalog compares the scalar's integer."""
    if signed:
        value = np.int64(word)
        return lower <= value <= upper
    return np.uint64(lower) <= word <= np.uint64(upper)


@njit(nogil=True)
def _respond(
    mail: np.ndarray,
    m: int,
    runs: np.ndarray,
    r: int,
    scratch: np.ndarray,
    failed_checks: np.ndarray,
) -> bool:
    """Answer mailbox `m`'s use of run `r` with the run's read-back bytes."""
    address = mail[m, ADDRESS]
    header = address + RESPONSE_AT
    failed = False
    if runs[r, STATUS_HOST] != 0:
        memcpy(scratch.ctypes.data, runs[r, STATUS_HOST], 4)
        failed = np.uint32(scratch[0]) != 0
    if failed:
        failed_checks[r] += 1
        scratch[0] = np.uint64(_CHECK_FAILED)
    else:
        memcpy(header + 16, runs[r, HOST], runs[r, NBYTES])
        scratch[0] = np.uint64(_ACCEPTED) | (np.uint64(runs[r, NBYTES]) << np.uint64(32))
    memcpy(header + 8, scratch.ctypes.data, 8)
    publish_sequence(header, np.uint64(mail[m, RUNNING_SEQUENCE]), RELEASE)
    scratch[0] = np.uint64(1)
    return write(mail[m, RESPONSE_FD], scratch.ctypes.data, 8) == 8


@njit(nogil=True)
def _drain(
    mail: np.ndarray, runs: np.ndarray, scratch: np.ndarray, failed_checks: np.ndarray
) -> int:
    """Wait for every use in progress and answer it; a driver failure's code."""
    for m in range(mail.shape[0]):
        r = mail[m, RUNNING]
        if r >= 0:
            code = cuEventSynchronize(runs[r, EVENT])
            if code != 0:
                return code
            if not _respond(mail, m, runs, r, scratch, failed_checks):
                return -1
            mail[m, RUNNING] = -1
    return 0


@njit(nogil=True)
def _serve(
    mail: np.ndarray,
    steps: np.ndarray,
    runs: np.ndarray,
    fills: np.ndarray,
    signed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    slots: np.ndarray,
    masks: np.ndarray,
    uses: np.ndarray,
    failed_checks: np.ndarray,
    polled: np.ndarray,
    scratch: np.ndarray,
    wakeup: int,
) -> tuple[np.int64, np.int64]:
    record = scratch[:_WORDS]
    word = scratch[_WORDS:]
    while True:
        busy = False
        for m in range(mail.shape[0]):
            r = mail[m, RUNNING]
            if r >= 0:
                code = cuEventQuery(runs[r, EVENT])
                if code == _NOT_READY:
                    busy = True
                    continue
                if code != 0:
                    return np.int64(_DRIVER), np.int64(code)
                if not _respond(mail, m, runs, r, word, failed_checks):
                    return np.int64(_IPC), np.int64(m)
                mail[m, RUNNING] = -1
            address = mail[m, ADDRESS]
            sequence = load_sequence(address, ACQUIRE)
            if sequence == np.uint64(mail[m, SEQUENCE]):
                continue
            # a new record: copy it out, then read only the copy
            memcpy(record.ctypes.data, address, _RECORD)
            fields = record[1]
            step = fields & np.uint64(0xFFFFFFFF)
            count = fields >> np.uint64(32)
            r = -1
            following = np.uint64(mail[m, SEQUENCE]) + np.uint64(1)  # modulo 2**64
            if record[0] == sequence and sequence == following:
                if step < np.uint64(MAX_PREPARED):
                    r = steps[m, step]
            if r >= 0 and count != np.uint64(runs[r, SCALARS]):
                r = -1
            if r >= 0:
                for j in range(runs[r, SCALARS]):
                    if not in_bounds(record[2 + j], signed[r, j] != 0, lower[r, j], upper[r, j]):
                        r = -1
                        break
            if r < 0:
                code = _drain(mail, runs, word, failed_checks)
                if code != 0:
                    return np.int64(_DRIVER if code > 0 else _IPC), np.int64(code)
                return np.int64(_PYTHON), np.int64(m)
            for j in range(runs[r, SCALARS]):
                word[0] = record[2 + j] & np.uint64(masks[r, j])
                memcpy(slots[r, j], word.ctypes.data, 8)
            stream = runs[r, STREAM]
            failed = 0
            for f in range(runs[r, FILLS]):
                width = fills[r, f, 0]
                target, value, count = fills[r, f, 1], fills[r, f, 2], fills[r, f, 3]
                if width == 1:
                    failed = cuMemsetD8Async(target, np.uint8(value), count, stream)
                elif width == 2:
                    failed = cuMemsetD16Async(target, np.uint16(value), count, stream)
                else:
                    failed = cuMemsetD32Async(target, np.uint32(value), count, stream)
                if failed != 0:
                    return np.int64(_DRIVER), np.int64(failed)
            failed = cuLaunchKernel(
                runs[r, FUNCTION],
                runs[r, GX],
                runs[r, GY],
                runs[r, GZ],
                runs[r, BX],
                runs[r, BY],
                runs[r, BZ],
                runs[r, SHARED],
                stream,
                runs[r, POINTERS],
                0,
            )
            if failed != 0:
                return np.int64(_DRIVER), np.int64(failed)
            failed = cuMemcpyDtoHAsync(runs[r, HOST], runs[r, SOURCE], runs[r, NBYTES], stream)
            if failed != 0:
                return np.int64(_DRIVER), np.int64(failed)
            if runs[r, STATUS_HOST] != 0:
                failed = cuMemcpyDtoHAsync(runs[r, STATUS_HOST], runs[r, STATUS_SOURCE], 4, stream)
                if failed != 0:
                    return np.int64(_DRIVER), np.int64(failed)
            failed = cuEventRecord(runs[r, EVENT], stream)
            if failed != 0:
                return np.int64(_DRIVER), np.int64(failed)
            mail[m, SEQUENCE] = np.int64(sequence)
            mail[m, RUNNING] = r
            mail[m, RUNNING_SEQUENCE] = np.int64(sequence)
            uses[r] += 1
            busy = True
        if read(wakeup, word.ctypes.data, 8) == 8:
            code = _drain(mail, runs, word, failed_checks)
            if code != 0:
                return np.int64(_DRIVER if code > 0 else _IPC), np.int64(code)
            return np.int64(_QUEUED), np.int64(0)
        if not busy:
            # wait for a doorbell, then clear each request doorbell that rang
            poll(polled.ctypes.data, polled.shape[0] // 2, -1)
            for k in range(1, polled.shape[0] // 2):
                if polled[2 * k + 1] >> 16:
                    read(polled[2 * k], word.ctypes.data, 8)
                    polled[2 * k + 1] = 1
            polled[1] = 1
