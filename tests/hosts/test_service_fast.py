"""The compiled loop's check of a scalar's bounds agrees with the catalog's, and
the loop orders a use's driver calls as `Executor.submit` does.

The loop runs as Python over a recording driver: a use's copies precede the
record of its event, and the loop reads the result and the check status only
after the event ends, through a wait or a query that finds it complete
(design decision 48).
"""

import ctypes
import os
import random
from collections.abc import Callable
from typing import SupportsInt

import numpy as np
import pytest

pytest.importorskip("numba")

from opgate.executor import Hot, _Fill  # noqa: E402
from opgate.hosts.service import fast  # noqa: E402
from opgate.hosts.service.dataplane import (  # noqa: E402
    RESPONSE,
    RESPONSE_AT,
    Mailbox,
    Status,
    encode_request,
)
from opgate.hosts.service.fast import _word, in_bounds  # noqa: E402

BOUNDS = [
    (-4, 7),
    (-(2**63), 2**63 - 1),
    (-1, 0),
    (0, 2**32 - 1),
    (0, 2**64 - 1),
    (5, 5),
    (2**63, 2**64 - 1),
]


def _values(lower: int, upper: int, rng: random.Random) -> list[int]:
    """Scalars at and beside the bounds, and random ones, as 64-bit integers."""
    edges = [lower - 1, lower, lower + 1, upper - 1, upper, upper + 1, 0, -1, 2**63, 2**64 - 1]
    edges += [rng.randrange(-(2**63), 2**64) for _ in range(200)]
    return [v for v in edges if -(2**63) <= v < 2**64]


@pytest.mark.parametrize(("lower", "upper"), BOUNDS)
def test_in_bounds_agrees_with_the_catalogs_comparison(lower: int, upper: int) -> None:
    rng = random.Random(lower ^ upper)
    signed = lower < 0
    for value in _values(lower, upper, rng):
        word = value % 2**64
        # the owner reads a signed scalar's word as two's complement, as the catalog's integer
        integer = word - 2**64 if signed and word >= 2**63 else word
        expected = lower <= integer <= upper
        observed = in_bounds(
            np.uint64(word), signed, np.int64(_word(lower)), np.int64(_word(upper))
        )
        assert bool(observed) == expected, (lower, upper, value)


STREAM, EVENT = 21, 31
_NOT_READY = 600  # CUDA_ERROR_NOT_READY


@pytest.mark.parametrize("ended", ["wait", "query"])
def test_the_loop_reads_a_use_only_after_its_event(
    monkeypatch: pytest.MonkeyPatch, ended: str
) -> None:
    """A use ends through `_drain`'s wait when work is queued, or through a query
    that finds its event complete after one that does not."""
    host = ctypes.create_string_buffer(b"resultok" + bytes(4))  # the result, then the status
    hot = Hot(
        bounds=((0, 100),),
        parameters=(5,),
        masks=(*(2**64 - 1,) * 5, 2**32 - 1),
        block=(ctypes.c_uint64 * 6)(),
        pointers=0x5000,
        fills=(_Fill(4, 0x9000, 0, 16),),
        function=0x7000,
        grid=(1, 1, 1),
        threads=(256, 1, 1),
        shared=0,
        stream=STREAM,
        event=EVENT,
        host=ctypes.addressof(host),
        source=0xA000,
        nbytes=8,
        status_host=ctypes.addressof(host) + 8,
        status_source=0xB000,
    )
    calls: list[tuple[str, tuple[int, ...]]] = []
    wakeup = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
    mailbox = Mailbox()
    try:

        def recorded(name: str) -> Callable[..., int]:
            def call(*arguments: SupportsInt) -> int:
                calls.append((name, tuple(int(a) for a in arguments)))
                return 0

            return call

        for name in (
            "cuMemsetD8Async",
            "cuMemsetD16Async",
            "cuMemsetD32Async",
            "cuLaunchKernel",
            "cuMemcpyDtoHAsync",
            "cuEventRecord",
            "cuEventSynchronize",
        ):
            monkeypatch.setattr(fast, name, recorded(name))
        codes = iter((_NOT_READY, 0))

        def query(event: SupportsInt) -> int:
            code = next(codes)
            calls.append(("cuEventQuery", (int(event), code)))
            if code == 0:
                os.eventfd_write(wakeup, 1)  # queued work, once the use has ended
            return code

        monkeypatch.setattr(fast, "cuEventQuery", query)
        copy = fast.memcpy

        def memcpy(destination: SupportsInt, source: SupportsInt, size: SupportsInt) -> int:
            calls.append(("memcpy", (int(destination), int(source), int(size))))
            return copy(int(destination), int(source), int(size))

        monkeypatch.setattr(fast, "memcpy", memcpy)
        # the C functions take Python integers when the loop runs as Python
        for name in ("load_sequence", "publish_sequence", "read", "write"):
            function = getattr(fast, name)
            monkeypatch.setattr(
                fast, name, lambda *arguments, f=function: f(*(int(a) for a in arguments))
            )
        monkeypatch.setattr(fast, "poll", lambda *_: pytest.fail("the loop waited for a doorbell"))
        monkeypatch.setattr(fast, "_respond", fast._respond.py_func)
        monkeypatch.setattr(fast, "_drain", fast._drain.py_func)
        tables = fast.Tables.of(wakeup, [mailbox], [[hot]])
        mailbox.view[: len(encode_request(1, 0, (7,)))] = encode_request(1, 0, (7,))
        if ended == "wait":
            os.eventfd_write(wakeup, 1)  # work is queued while the use runs
        t = tables
        left, _ = fast._serve.py_func(
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
        assert left == fast.Exit.QUEUED
        assert t.mail[0, fast.RUNNING] == -1 and t.uses.tolist() == [1]
        assert hot.block[5] == 7
        end = RESPONSE_AT + RESPONSE.size
        assert RESPONSE.unpack(mailbox.view[RESPONSE_AT:end]) == (1, Status.ACCEPTED, 8)
        assert mailbox.view[end : end + 8] == b"resultok"
    finally:
        mailbox.close()
        os.close(wakeup)
    names = [name for name, _ in calls]
    record = names.index("cuEventRecord")
    assert calls[record] == ("cuEventRecord", (EVENT, STREAM))
    copies = [args for name, args in calls if name == "cuMemcpyDtoHAsync"]
    assert copies == [
        (hot.host, hot.source, hot.nbytes, STREAM),
        (hot.status_host, hot.status_source, 4, STREAM),
    ]
    assert names.index("cuMemcpyDtoHAsync") < record and names.count("cuEventRecord") == 1
    waits = [
        i
        for i, (name, args) in enumerate(calls)
        if name == "cuEventSynchronize" or (name == "cuEventQuery" and args[1] == 0)
    ]
    reads = [
        (i, args[1])
        for i, (name, args) in enumerate(calls)
        if name == "memcpy" and args[1] in (hot.host, hot.status_host)
    ]
    # the status first, then the result, both after the wait that ended the use
    assert [source for _, source in reads] == [hot.status_host, hot.host]
    assert record < waits[0] < reads[0][0]
    assert names[waits[0]] == ("cuEventSynchronize" if ended == "wait" else "cuEventQuery")
