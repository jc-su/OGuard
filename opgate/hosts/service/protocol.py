"""The service's wire protocol: length-prefixed JSON objects over a stream socket.

A message is a four-byte big-endian length and that many bytes of UTF-8 JSON,
one object with no duplicate key and no non-finite number. `Fields` reads a
received object's fields with their types checked; every failure is a
`RequestError`, which the service reports to the client as its error, before
any state changes.
"""

import json
import math
import re
import socket
import struct
from collections.abc import Mapping
from time import monotonic

HEADER = struct.Struct("!I")
MAX_DEPTH = 32  # the deepest nesting of arrays and objects that a message may hold
# String runs are possessive: no backtracking over their contents. Consuming
# an unterminated string through EOF also prevents retrying at its escaped
# quotes; the JSON decoder rejects that syntax after the depth check.
_STRUCTURE = re.compile(rb'"[^"\\]*+(?:\\.[^"\\]*+)*+(?:"|\\?$)|([\[\]{}])', re.DOTALL)

__all__ = ["HEADER", "Fields", "RequestError", "encode", "read_message", "write_message"]


class RequestError(ValueError):
    """A malformed request, which is the client's error."""


class Fields:
    """The fields of one JSON object of a request, named by their path for errors."""

    def __init__(self, value: object, where: str) -> None:
        if type(value) is not dict:
            raise RequestError(f"{where} must be an object")
        self._fields: dict[str, object] = {str(k): v for k, v in value.items()}
        self.where = where

    def keys(self) -> frozenset[str]:
        return frozenset(self._fields)

    def record(self) -> dict[str, object]:
        """A copy of the object."""
        return dict(self._fields)

    def exact(self, *keys: str) -> None:
        """Require exactly these fields."""
        expected = set(keys)
        if set(self._fields) != expected:
            missing = sorted(expected - set(self._fields))
            extra = sorted(set(self._fields) - expected)
            raise RequestError(
                f"{self.where} has missing fields {missing} and extra fields {extra}"
            )

    def raw(self, name: str) -> object:
        if name not in self._fields:
            raise RequestError(f"{self.where} has no field {name!r}")
        return self._fields[name]

    def integer(self, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
        value = self.raw(name)
        if type(value) is not int:
            raise RequestError(f"{self.where}.{name} must be an integer")
        if minimum is not None and value < minimum:
            raise RequestError(f"{self.where}.{name} must be at least {minimum}")
        if maximum is not None and value > maximum:
            raise RequestError(f"{self.where}.{name} must be at most {maximum}")
        return value

    def optional_integer(self, name: str, *, minimum: int, maximum: int) -> int | None:
        """An integer field whose value may be null."""
        if self.raw(name) is None:
            return None
        return self.integer(name, minimum=minimum, maximum=maximum)

    def number(self, name: str) -> float:
        """A finite number."""
        value = self.raw(name)
        if type(value) is int:
            try:
                result = float(value)
            except OverflowError as error:
                raise RequestError(f"{self.where}.{name} must be a finite number") from error
        elif type(value) is float:
            result = value
        else:
            raise RequestError(f"{self.where}.{name} must be a number")
        if not math.isfinite(result):
            raise RequestError(f"{self.where}.{name} must be finite")
        return result

    def boolean(self, name: str) -> bool:
        value = self.raw(name)
        if type(value) is not bool:
            raise RequestError(f"{self.where}.{name} must be a boolean")
        return value

    def text(self, name: str) -> str:
        value = self.raw(name)
        if type(value) is not str:
            raise RequestError(f"{self.where}.{name} must be a string")
        return value

    def fields(self, name: str) -> "Fields":
        return Fields(self.raw(name), f"{self.where}.{name}")

    def values(self, name: str) -> list[object]:
        value = self.raw(name)
        if type(value) is not list:
            raise RequestError(f"{self.where}.{name} must be an array")
        return list(value)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RequestError(f"a duplicate JSON key {key!r}")
        result[key] = value
    return result


def _no_constant(token: str) -> object:
    raise RequestError(f"a non-finite JSON number {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise RequestError("a JSON number is outside the finite floating-point range")
    return value


def _depth(payload: bytes) -> int:
    """The deepest nesting of arrays and objects, outside strings."""
    depth = deepest = 0
    # The scanner consumes each string as one token, including escapes.
    # Inspect its first byte without copying potentially large string data.
    for match in _STRUCTURE.finditer(payload):
        if match.lastindex is None:
            continue
        byte = payload[match.start()]
        if byte in (0x5B, 0x7B):  # [ {
            depth += 1
            deepest = max(deepest, depth)
        elif byte in (0x5D, 0x7D):  # ] }
            depth -= 1
    return deepest


def encode(value: Mapping[str, object], maximum: int) -> bytes:
    """One message; a response larger than `maximum` bytes is the server's error."""
    payload = json.dumps(dict(value), allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > maximum:
        raise ValueError(f"a message of {len(payload)} bytes exceeds the limit of {maximum}")
    return HEADER.pack(len(payload)) + payload


def _read_exact(connection: socket.socket, count: int, deadline: float | None) -> bytes:
    # Store bytes, not one Python object per recv: fragmented delivery must not
    # turn a bounded frame into an unbounded list of tiny allocations.
    payload = bytearray()
    remaining = count
    while remaining:
        if deadline is not None:
            left = deadline - monotonic()
            if left <= 0:
                raise TimeoutError("the message deadline expired")
            connection.settimeout(left)
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError
        payload.extend(chunk)
        remaining -= len(chunk)
    return bytes(payload)


def read_message(
    connection: socket.socket, maximum: int, where: str, *, timeout: float | None = None
) -> Fields:
    """The next message, optionally within one deadline for its entire frame.

    The socket's prior timeout is restored; receiving part of a frame does not
    restart the deadline. `EOFError` means that the peer closed the connection.
    """
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("the message timeout must be finite and positive")
    deadline = None if timeout is None else monotonic() + timeout
    previous_timeout = connection.gettimeout() if deadline is not None else None
    try:
        (length,) = HEADER.unpack(_read_exact(connection, HEADER.size, deadline))
        if length == 0 or length > maximum:
            raise RequestError(f"a message of {length} bytes is outside the limit of {maximum}")
        payload = _read_exact(connection, length, deadline)
    finally:
        if deadline is not None:
            connection.settimeout(previous_timeout)
    try:
        document = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RequestError("the message is not UTF-8") from error
    # Every nested container needs an opening byte. Counting openings inside
    # strings only raises this bound; short flat messages need no token scan.
    openings = payload.count(b"[") + payload.count(b"{")
    if openings > MAX_DEPTH and _depth(payload) > MAX_DEPTH:
        raise RequestError(f"the message nests arrays and objects deeper than {MAX_DEPTH}")
    try:
        value = json.loads(
            document,
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constant,
            parse_float=_finite_float,
        )
    except RequestError:
        raise
    except ValueError as error:
        raise RequestError("the message is not JSON") from error
    return Fields(value, where)


def write_message(
    connection: socket.socket,
    value: Mapping[str, object],
    maximum: int,
    *,
    timeout: float | None = None,
) -> None:
    """Send one frame; `sendall` applies its timeout to the complete send."""
    payload = encode(value, maximum)
    if timeout is None:
        connection.sendall(payload)
        return
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("the message timeout must be finite and positive")
    previous_timeout = connection.gettimeout()
    try:
        connection.settimeout(timeout)
        connection.sendall(payload)
    finally:
        connection.settimeout(previous_timeout)
