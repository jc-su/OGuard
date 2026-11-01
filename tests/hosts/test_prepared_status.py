"""Both mailbox paths deliver runtime check outcomes without returning partial results."""

import ctypes
import os
import threading
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from opgate.executor import CheckFailedError, Executor, Run
from opgate.hosts.service import fast
from opgate.hosts.service.dataplane import ClientMailbox, Mailbox, Status


@pytest.mark.parametrize("status", [0, 1, 256])
def test_compiled_response_checks_the_entire_status_word(status):
    result = ctypes.create_string_buffer(b"result" + status.to_bytes(4, "little"))
    mail = Mailbox()
    try:
        runs = np.zeros((1, 18), dtype=np.int64)
        runs[0, fast.HOST] = ctypes.addressof(result)
        runs[0, fast.NBYTES] = 6
        runs[0, fast.STATUS_HOST] = ctypes.addressof(result) + 6
        rows = np.zeros((1, 6), dtype=np.int64)
        rows[0, fast.ADDRESS] = mail.address
        rows[0, fast.RESPONSE_FD] = mail.response
        rows[0, fast.RUNNING_SEQUENCE] = 1
        failures = np.zeros(1, dtype=np.int64)
        with ClientMailbox([os.dup(d) for d in mail.descriptors], timeout=1) as client:
            assert fast._respond(rows, 0, runs, 0, np.zeros(2, np.uint64), failures)
            if status:
                with pytest.raises(CheckFailedError, match="prepared step 0"):
                    client.run(0, ())
                assert failures.tolist() == [1]
            else:
                assert client.run(0, ()) == b"result"
                assert failures.tolist() == [0]
    finally:
        mail.close()


@pytest.mark.parametrize("status", [0, 1, 256])
def test_executor_collects_the_status_before_returning_output(monkeypatch, status):
    import opgate.executor as runtime

    result = ctypes.create_string_buffer(b"result" + status.to_bytes(4, "little"))
    run = SimpleNamespace(
        running=True,
        event=1,
        status=object(),
        host=ctypes.addressof(result),
        nbytes=6,
        module=SimpleNamespace(binding=SimpleNamespace(kernel="ordinary_kernel")),
    )
    executor = SimpleNamespace(_require_run=Mock())
    monkeypatch.setattr(runtime.driver, "cuEventSynchronize", Mock(return_value=(0,)))
    if status:
        with pytest.raises(CheckFailedError, match="ordinary_kernel"):
            Executor.result(cast(Executor, executor), cast(Run, run))
    else:
        assert Executor.result(cast(Executor, executor), cast(Run, run)) == b"result"
    assert not run.running


def test_a_timed_out_mailbox_cannot_submit_another_request():
    mail = Mailbox()
    try:
        assert not os.get_blocking(mail.response)
        with ClientMailbox([os.dup(d) for d in mail.descriptors], timeout=0.01) as client:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                client.run(0, ())
            assert time.monotonic() - started < 1
            with pytest.raises(RuntimeError, match="deadline"):
                client.run(0, ())
            assert mail.take().sequence == 1
    finally:
        mail.close()


def test_mailbox_normal_response_and_next_use():
    mail = Mailbox()
    try:
        with ClientMailbox([os.dup(d) for d in mail.descriptors], timeout=1) as client:
            for sequence in (1, 2):

                def answer(sequence=sequence):
                    time.sleep(0.005)
                    mail.respond(sequence, Status.ACCEPTED, b"ordinary output")

                worker = threading.Thread(target=answer)
                worker.start()
                try:
                    assert client.run(0, ()) == b"ordinary output"
                finally:
                    worker.join()
    finally:
        mail.close()


def test_mailbox_publishes_the_complete_request_before_notifying(monkeypatch):
    import opgate.hosts.service.dataplane as protocol

    mail = Mailbox()
    try:
        with ClientMailbox([os.dup(d) for d in mail.descriptors], timeout=1) as client:
            observed = []
            write = os.eventfd_write

            def notify(descriptor, value):
                if descriptor == client._request:
                    record = mail.take()
                    observed.append((record.sequence, record.prepared, record.scalars))
                    mail.respond(record.sequence, Status.ACCEPTED, b"complete")
                else:
                    write(descriptor, value)

            # A packing operation on the published word must never be used.
            monkeypatch.setattr(protocol, "_SEQUENCE", SimpleNamespace(size=8))
            monkeypatch.setattr(os, "eventfd_write", notify)
            for sequence, scalars in ((1, (7, 9)), (2, (11, 13))):
                assert client.run(3, scalars) == b"complete"
                assert observed[-1] == (sequence, 3, scalars)
    finally:
        mail.close()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_mailbox_requires_a_finite_positive_deadline(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        ClientMailbox([], timeout=timeout)


@pytest.mark.gpu
def test_service_prepares_status_reporting_kernels_and_clears_each_use(tmp_path):
    """A kernel stores to word `signal` of a buffer of one word: the gate does not
    prove the store in bounds, and the guard pass checks it, so a launch with
    signal 1 reports a failed check through the pass's status buffer."""
    from opgate.binding import BASE_ALIGN, Base, Binding, Bounded, Direction, Span
    from opgate.catalog import TenantId
    from opgate.frontends import Target
    from opgate.hosts.service import Client, Fields, Limits, Principal, Service
    from opgate.hosts.service.plans import LaunchRequest
    from opgate.signature import Affine

    text = """.version 8.0
.target sm_86
.address_size 64
.visible .entry report_status(
 .param .u64 output,
 .param .u32 signal
)
{
 .reg .b32 %r<2>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output];
 ld.param.u32 %r1, [signal];
 mul.wide.u32 %rd2, %r1, 4;
 add.s64 %rd3, %rd1, %rd2;
 st.global.u32 [%rd3], 42;
 ret;
}
"""
    bound = Binding(
        "report_status",
        (),
        (
            ("output", Base(Span(Direction.WRITE, Affine(4), BASE_ALIGN, None, False))),
            ("signal", Bounded(0, 1)),
        ),
        (1, 1, 1),
        (1, 1, 1),
        Affine(0),
    )

    class Schema:
        kind = "status_test"
        binding = bound

        def plan(self, text: str, fields: Fields):
            fields.exact("kind")
            return self

        def launch(self, request: Fields) -> LaunchRequest:
            raise AssertionError("this test exercises prepared execution")

    path = tmp_path / "service.sock"
    tmp_path.chmod(0o700)
    token = "ordinary-test-token-0123456789"
    principal = Principal("tenant", token, TenantId(1), administrator=True)
    with Service(
        path,
        (principal,),
        device=0,
        pool_bytes=1 << 25,
        target=Target(8, 6),
        schemas=(Schema(),),
        nodes=(),
        limits=Limits(verification_workers=2, tenant_bytes=1 << 20, chunk_bytes=1 << 16),
    ):
        with Client(path, token) as client:
            operator = client.register((text, {"kind": "status_test"}))[0].text("operator")
            output = client.request("allocate", nbytes=4).text("buffer")
            step = client.prepare(
                operator=operator,
                grid=[1, 1, 1],
                buffers=[output],
                sizes=[],
                scalars=[1],
                fills=[],
                read=output,
            )
            with client.data_plane() as mailbox:
                for signal in (1, 0, 1, 0):
                    if signal:
                        with pytest.raises(CheckFailedError):
                            mailbox.run(step, (signal,))
                    else:
                        assert mailbox.run(step, (signal,)) == (42).to_bytes(4, "little")
            telemetry = client.request("telemetry").fields("telemetry")
            assert telemetry.integer("failed_checks") == 2
            assert telemetry.integer("fatal_errors") == 0
        with Client(path, token) as admin:
            assert admin.request("telemetry").fields("telemetry").integer("live_buffers") == 0
