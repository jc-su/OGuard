"""Protocol depth and syntax checks run before any request reaches the owner."""

import json
import socket
from typing import cast

import pytest

import opgate.hosts.service.protocol as protocol
from opgate.hosts.service.protocol import (
    HEADER,
    MAX_DEPTH,
    RequestError,
    read_message,
    write_message,
)


def receive(payload: bytes):
    left, right = socket.socketpair()
    try:
        left.sendall(HEADER.pack(len(payload)) + payload)
        return read_message(right, 65536, "test")
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("arrays", [0, MAX_DEPTH - 1, MAX_DEPTH])
@pytest.mark.parametrize("text", ["[{]}", '\\"[{}]', '"\\\\"', "é☃\\\n["])
def test_escaped_string_contents_do_not_change_container_depth(arrays, text):
    value = text
    for _ in range(arrays):
        value = [value]
    payload = json.dumps({"value": value}, ensure_ascii=False).encode()
    if arrays + 1 > MAX_DEPTH:
        with pytest.raises(RequestError, match="deeper"):
            receive(payload)
    else:
        assert receive(payload).raw("value") == value


@pytest.mark.parametrize(
    "payload",
    [b'{"x":"\\q"}', b'{"x":"unfinished}', b'{"x":1,}', b'{"x":1,"x":2}'],
)
def test_invalid_json_is_rejected_after_depth_scan(payload):
    with pytest.raises(RequestError):
        receive(payload)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-32", "utf-32-be"])
def test_only_utf8_json_is_accepted(encoding):
    with pytest.raises(RequestError):
        receive('{"text":"valid text"}'.encode(encoding))


@pytest.mark.parametrize("number", [b"1e999", b"-1e999", b"NaN", b"Infinity"])
def test_nonfinite_numbers_are_rejected_at_the_protocol_boundary(number):
    with pytest.raises(RequestError):
        receive(b'{"nested":{"number":' + number + b"}}")


def test_finite_numbers_remain_valid():
    assert receive(b'{"values":[-1.25,1e308,1e-308]}').raw("values") == [
        -1.25,
        1e308,
        1e-308,
    ]


@pytest.mark.parametrize("expired", [False, True])
def test_one_deadline_covers_header_and_payload_and_restores_timeout(monkeypatch, expired):
    """Advance a fake clock between receives; no delayed traffic or live server."""
    now = [100.0]
    monkeypatch.setattr(protocol, "monotonic", lambda: now[0])
    frame = HEADER.pack(2) + b"{}"

    class Connection:
        timeout = 9.0
        position = 0
        waits = []

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value

        def recv(self, count):
            self.waits.append(self.timeout)
            chunk = frame[self.position : self.position + 1]
            self.position += 1
            now[0] += 0.4 if expired else 0.1
            return chunk

    connection = Connection()
    transport = cast(socket.socket, connection)
    if expired:
        with pytest.raises(TimeoutError, match="deadline"):
            read_message(transport, 100, "test", timeout=1.0)
        assert connection.position == 3
    else:
        assert read_message(transport, 100, "test", timeout=1.0).record() == {}
        assert connection.waits == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    assert connection.timeout == 9.0


@pytest.mark.parametrize("expired", [False, True])
def test_a_response_deadline_is_applied_to_sendall_and_restored(expired):
    class Connection:
        timeout = 9.0
        sent = None

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value

        def sendall(self, frame):
            assert self.timeout == 0.25
            if expired:
                raise TimeoutError("the receiver did not consume the response")
            self.sent = frame

    connection = Connection()
    transport = cast(socket.socket, connection)
    if expired:
        with pytest.raises(TimeoutError):
            write_message(transport, {"ok": True}, 100, timeout=0.25)
    else:
        write_message(transport, {"ok": True}, 100, timeout=0.25)
        assert connection.sent == HEADER.pack(11) + b'{"ok":true}'
    assert connection.timeout == 9.0


def test_many_shallow_containers_remain_valid():
    value = {"rows": [[number] for number in range(MAX_DEPTH * 2)]}
    assert receive(json.dumps(value).encode()).record() == value


def test_many_opening_bytes_inside_a_string_remain_valid():
    value = {"value": '[{\\"' * (MAX_DEPTH * 2)}
    assert receive(json.dumps(value).encode()).record() == value


@pytest.mark.parametrize("depth", [MAX_DEPTH, MAX_DEPTH + 1])
def test_container_depth_limit_with_no_quoted_brackets(depth):
    value = 0
    for _ in range(depth - 1):
        value = [value]
    payload = json.dumps({"value": value}).encode()
    if depth == MAX_DEPTH:
        assert receive(payload).raw("value") == value
    else:
        with pytest.raises(RequestError, match="deeper"):
            receive(payload)
