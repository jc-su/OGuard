"""The data plane: a session runs its prepared steps through a shared-memory mailbox.

A session asks for its mailbox with the `data_plane` request. The server
creates a sealed memfd of `MAILBOX_BYTES`, which no process can shrink or
grow, and two eventfds, a request doorbell and a response doorbell, and
passes the three descriptors over the session's authenticated socket only
(`SCM_RIGHTS`), after the JSON response. The client writes one request record
and rings the request doorbell. The owner thread copies the record out of the
shared memory before it reads any field, so a client that changes the record
meanwhile changes nothing the owner checks; it has the executor submit the
use of the prepared step, and when the use ends it writes the response
record, its sequence number last, and rings the response doorbell. The client
waits for the sequence number for up to `SPIN_SECONDS`, as a client of the
engine in its own process would wait for its kernel, and then for the
doorbell. A session has at most one request in progress: a record written
while one runs is taken after it ends.

A request record is the sequence number (one more than the last), the handle
of a step that `prepare` gave this session, the count of scalars, and the
scalars, each 8 bytes: the two's complement of a signed scalar. A response
record is the request's sequence number, a status, and the payload's length;
the payload is the read-back buffer's bytes, or a refusal's reason.
"""

import ctypes
import fcntl
import math
import mmap
import os
import select
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Self

from opgate.executor import READ_BYTES, CheckFailedError, RefusedError
from opgate.hosts.service.protocol import RequestError

__all__ = [
    "MAILBOX_BYTES",
    "MAX_SCALARS",
    "ClientMailbox",
    "Mailbox",
    "Status",
    "decode_request",
    "encode_request",
]

MAX_SCALARS = 32
REQUEST = struct.Struct(f"<QII{MAX_SCALARS}Q")  # sequence, prepared step, count, scalars
RESPONSE = struct.Struct("<QII")  # sequence, status, payload length
RESPONSE_AT = 4096  # the response record's offset
MAILBOX_BYTES = RESPONSE_AT + RESPONSE.size + READ_BYTES
SPIN_SECONDS = 0.0002  # how long a client polls the response before it blocks
_SEQUENCE = struct.Struct("<Q")
_OUTCOME = struct.Struct("<II")  # status, payload length

# Publish a record with one release store, then consume it with an acquire load.
# struct.pack_into clears its destination before packing; it cannot publish a
# shared sequence word. Lock-free operations also work across processes.
ACQUIRE, RELEASE = 2, 3  # GCC __ATOMIC_ACQUIRE / __ATOMIC_RELEASE
_ATOMIC = ctypes.CDLL("libatomic.so.1")
load_sequence = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int)(
    ("__atomic_load_8", _ATOMIC)
)
publish_sequence = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)(
    ("__atomic_store_8", _ATOMIC)
)
_lock_free = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_size_t, ctypes.c_void_p)(
    ("__atomic_is_lock_free", _ATOMIC)
)
if sys.byteorder != "little" or not _lock_free(8, None):
    raise RuntimeError("the mailbox requires little-endian lock-free 64-bit atomics")


# Linux's memfd and file-seal interface (<linux/memfd.h>, <fcntl.h>), which this
# Python's `os` and `fcntl` modules do not wrap
_MFD_CLOEXEC, _MFD_ALLOW_SEALING = 0x1, 0x2
_F_ADD_SEALS = 1024 + 9
_F_SEAL_SEAL, _F_SEAL_SHRINK, _F_SEAL_GROW = 0x1, 0x2, 0x4
_LIBC = ctypes.CDLL(None, use_errno=True)


def _sealed_memory(nbytes: int) -> int:
    """A memfd of `nbytes` that no process can shrink or grow: a client that
    could shrink it would make the owner's reads of the mapping fault."""
    descriptor = _LIBC.memfd_create(b"opgate-mailbox", _MFD_CLOEXEC | _MFD_ALLOW_SEALING)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        os.ftruncate(descriptor, nbytes)
        fcntl.fcntl(descriptor, _F_ADD_SEALS, _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_SEAL)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


class Status(IntEnum):
    ACCEPTED = 0  # the payload is the read-back buffer's bytes
    REFUSED = 1  # the catalog refused the use; the payload is its reason
    INPUT = 2  # the record is malformed; the payload says why
    FATAL = 3  # the service failed
    CHECK_FAILED = 4  # a runtime check failed; no result bytes are returned


@dataclass(frozen=True)
class Record:
    """A request record, copied out of the mailbox."""

    sequence: int
    prepared: int
    count: int  # of scalars, as the client wrote it
    scalars: tuple[int, ...]  # the first `count`, at most `MAX_SCALARS`, each as 8 unsigned bytes


def encode_request(sequence: int, prepared: int, scalars: Sequence[int]) -> bytes:
    if len(scalars) > MAX_SCALARS:
        raise ValueError(f"a request holds at most {MAX_SCALARS} scalars")
    words = [value % 2**64 for value in scalars]
    return REQUEST.pack(sequence, prepared, len(words), *words, *(0,) * (MAX_SCALARS - len(words)))


def decode_request(data: bytes) -> Record:
    """Any record of `REQUEST.size` bytes decodes; the owner checks its fields."""
    sequence, prepared, count, *words = REQUEST.unpack(data)
    return Record(sequence, prepared, count, tuple(words[:count]))


class Mailbox:
    """The owner's side of a session's mailbox."""

    def __init__(self) -> None:
        self.memory = _sealed_memory(MAILBOX_BYTES)
        self.request = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self.response = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        try:
            self.view = mmap.mmap(self.memory, MAILBOX_BYTES)
        except BaseException:
            self._close_descriptors()
            raise
        # the mapping's address, for the compiled loop; the export ends at once
        exported = ctypes.c_char.from_buffer(self.view)
        self.address = ctypes.addressof(exported)
        del exported
        self.sequence = 0  # of the last request taken

    @property
    def descriptors(self) -> list[int]:
        """What the client receives: the memory, the request and the response doorbells."""
        return [self.memory, self.request, self.response]

    def take(self) -> Record:
        """Copy the request record out; the owner's loop clears the doorbell."""
        load_sequence(self.address, ACQUIRE)
        return decode_request(bytes(self.view[: REQUEST.size]))

    def respond(self, sequence: int, status: Status, payload: bytes) -> None:
        """Write the response, its sequence number last, and ring the doorbell."""
        payload = payload[:READ_BYTES]
        end = RESPONSE_AT + RESPONSE.size
        self.view[end : end + len(payload)] = payload
        _OUTCOME.pack_into(self.view, RESPONSE_AT + _SEQUENCE.size, status, len(payload))
        publish_sequence(self.address + RESPONSE_AT, sequence, RELEASE)
        os.eventfd_write(self.response, 1)

    def close(self) -> None:
        self.view.close()
        self._close_descriptors()

    def _close_descriptors(self) -> None:
        for descriptor in (self.memory, self.request, self.response):
            os.close(descriptor)


class ClientMailbox:
    """The client's side of its session's mailbox; one thread uses it."""

    def __init__(self, descriptors: Sequence[int], *, timeout: float) -> None:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("mailbox timeout must be finite and positive")
        memory, self._request, self._response = descriptors
        try:
            self._view = mmap.mmap(memory, MAILBOX_BYTES)
        finally:
            os.close(memory)
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(self._view))
        self._poll = select.poll()
        self._poll.register(self._response, select.POLLIN)
        self._timeout = timeout
        self._sequence = 0
        self._failure: str | None = None  # the service's failure, once it answered with it

    def run(self, prepared: int, scalars: Sequence[int]) -> bytes:
        """Run a use of the prepared step with the launch's scalars; the bytes it read back."""
        if self._failure is not None:
            raise RuntimeError(f"the service failed: {self._failure}")
        sequence = (self._sequence + 1) % 2**64
        record = encode_request(sequence, prepared, scalars)
        self._sequence = sequence
        deadline = time.monotonic() + self._timeout
        # the record, then its sequence number, which tells the owner that it is whole
        self._view[_SEQUENCE.size : REQUEST.size] = record[_SEQUENCE.size :]
        publish_sequence(self._address, self._sequence, RELEASE)
        os.eventfd_write(self._request, 1)
        spin = min(time.monotonic() + SPIN_SECONDS, deadline)
        while load_sequence(self._address + RESPONSE_AT, ACQUIRE) != self._sequence:
            if time.monotonic() >= spin:
                break
        # Publication precedes notification. Even if the sequence is visible,
        # notification must arrive within the same request deadline.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._failure = "the service did not answer the request before its deadline"
                raise TimeoutError(self._failure)
            try:
                os.eventfd_read(self._response)
                break
            except BlockingIOError:
                events = self._poll.poll(math.ceil(remaining * 1000))
                if any(
                    flags & (select.POLLERR | select.POLLHUP | select.POLLNVAL)
                    for _, flags in events
                ):
                    self._failure = "the service response descriptor failed"
                    raise RuntimeError(self._failure) from None
        end = RESPONSE_AT + RESPONSE.size
        load_sequence(self._address + RESPONSE_AT, ACQUIRE)
        sequence, status, length = RESPONSE.unpack(self._view[RESPONSE_AT:end])
        if length > READ_BYTES or status not in Status._value2member_map_:
            self._failure = "the service returned an invalid response header"
            raise RequestError(self._failure)
        payload = bytes(self._view[end : end + length])
        if status == Status.FATAL:
            # the failed owner answers with its last number, whatever was pending
            self._failure = payload.decode("utf-8")
            raise RuntimeError(f"the service failed: {self._failure}")
        if sequence != self._sequence:
            self._failure = f"the response answers request {sequence}, not {self._sequence}"
            raise RequestError(self._failure)
        match Status(status):
            case Status.ACCEPTED:
                return payload
            case Status.REFUSED:
                raise RefusedError(payload.decode("utf-8"))
            case Status.CHECK_FAILED:
                raise CheckFailedError(f"prepared step {prepared}")
            case Status.INPUT:
                raise RequestError(payload.decode("utf-8"))
            case Status.FATAL:
                raise AssertionError("a failure is answered above")

    def close(self) -> None:
        self._view.close()
        os.close(self._request)
        os.close(self._response)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
