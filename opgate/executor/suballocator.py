"""The executor's own suballocator: untrusted, as every suballocator is.

It proposes a region of `[base, base + size)` for each allocation, first fit,
at a multiple of 256 bytes. The executor keeps one over the pool, which
proposes each tenant's reservation, and one over each reservation, which
proposes the tenant's buffers, so that where a tenant's buffers lie depends
only on the tenant's own allocations and frees. The catalog checks every
proposal (design section 6.2): a region outside the pool, outside the
tenant's reservation, or over a live buffer is refused, never used.
"""

from opgate.catalog import ALIGNMENT

__all__ = ["Suballocator"]


class Suballocator:
    """First fit over `[base, base + size)`, with every region at a multiple of 256 bytes."""

    def __init__(self, base: int, size: int) -> None:
        if base % ALIGNMENT != 0 or size < 0:
            raise ValueError("a suballocator's base is a multiple of 256 and its size not negative")
        self._base = base
        self._end = base + size
        self._used: dict[int, int] = {}  # offset to bytes, of each proposed region

    def propose(self, nbytes: int) -> int:
        """The offset of a free region of `nbytes`."""
        offset = self._base
        for start, size in sorted(self._used.items()):
            if offset + nbytes <= start:
                break
            offset = max(offset, _round_up(start + size))
        if offset + nbytes > self._end:
            raise MemoryError(f"no free region of {nbytes} bytes")
        self._used[offset] = nbytes
        return offset

    @property
    def base(self) -> int:
        return self._base

    def release(self, offset: int) -> None:
        del self._used[offset]


def _round_up(offset: int) -> int:
    return -(-offset // ALIGNMENT) * ALIGNMENT
