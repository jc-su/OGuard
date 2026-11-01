"""The PyTorch host: tensors in the executor's pool, and verified operators on them.

PyTorch allocates device memory through its caching allocator, which takes
segments from an allocator and places tensors inside them. `TorchHost` gives
each tenant a `torch.cuda.MemPool` whose allocator is the executor's: each
segment is a buffer of the tenant, which the executor's suballocator places
in the tenant's reservation (`Executor.reserve`, which the tenant makes
first) and the catalog clears unless the tenant last owned it (design
section 6.2). Inside `session(tenant)`, PyTorch allocates the tenant's
tensors from its pool, so the caching allocator, which stays untrusted,
places them inside the tenant's segments without a catalog step; a segment
is a step when PyTorch takes it and when it gives it back. Outside a session
PyTorch allocates from its own allocator, outside the executor's pool, which
no launch reaches.

A launch takes a tensor as the view of its segment from the tensor's start
(`argument_at`). A span with an index range takes a buffer of its own, which
`buffer` gives as a tensor over the whole buffer; trusted host code copies
an index column into it, and a guarded launch checks it when it runs. A
module with the guard pass's checks takes a status buffer that the executor
allocates for the launch (`opgate.hosts.launch`), and a failed check raises
`CheckFailedError`.
`release(tenant)` ends the tenant's pool: PyTorch gives its segments back
once no tensor holds them, and each leaves the catalog.

PyTorch calls the allocator through C function pointers, which ctypes makes
from the methods here. An exception cannot cross such a call: a refusal of
the catalog or a full reservation gives PyTorch no segment, and PyTorch raises its
out-of-memory error; any other failure is kept, and the host's next
operation raises it.

Imports PyTorch, `opgate.catalog`, `opgate.executor`, and `opgate.hosts`.
"""

import bisect
import ctypes
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Self

import torch

from opgate.catalog import Launch, View
from opgate.executor import BufferId, Executor, Module, RefusedError, TenantId
from opgate.hosts import launch

__all__ = ["TorchHost"]

# void* alloc(ssize_t size, int device, cudaStream_t stream)
_ALLOCATE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p)
# void free(void* pointer, size_t size, int device, cudaStream_t stream)
_FREE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)


def _address(callback: "ctypes._CData") -> int:
    """The C function pointer of a ctypes callback."""
    address = ctypes.cast(callback, ctypes.c_void_p).value
    if address is None:
        raise RuntimeError("a ctypes callback has no address")
    return address


@dataclass(frozen=True)
class _Region:
    """A buffer that the executor's suballocator placed: its owner and bytes."""

    buffer: BufferId
    owner: TenantId
    nbytes: int


class _Interface:
    """The CUDA array interface of a buffer of its own, which a tensor holds."""

    def __init__(self, address: int, count: int, typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape": (count,),
            "strides": None,
            "typestr": typestr,
            "data": (address, False),
            "version": 3,
        }


class TorchHost:
    """PyTorch's allocator over the executor's pool, a pool for each tenant."""

    def __init__(self, executor: Executor) -> None:
        self._executor = executor
        self._device = torch.cuda.current_device()
        self._segments: dict[int, _Region] = {}  # each segment by its address
        self._starts: list[int] = []  # the segments' addresses, in increasing order
        self._buffers: dict[int, _Region] = {}  # each buffer of its own by its address
        self._pools: dict[TenantId, torch.cuda.MemPool] = {}
        self._failure: BaseException | None = None
        self._closed = False
        # the callbacks live as long as the host, since PyTorch keeps their addresses
        self._allocate_callback = _ALLOCATE(self._allocate)
        self._free_callback = _FREE(self._free)
        self._allocator = torch._C._cuda_customAllocator(
            _address(self._allocate_callback), _address(self._free_callback)
        )

    def close(self) -> None:
        """End the host with its executor: each tenant's pool gives its segments
        back, and a segment that a tensor still holds afterwards changes nothing
        when PyTorch frees it."""
        self._release(tuple(self._pools))
        self._closed = True
        self._segments.clear()
        self._starts.clear()
        self._buffers.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def session(self, tenant: TenantId) -> Iterator[None]:
        """Run the block as `tenant`'s session, with PyTorch allocating its tensors
        from the tenant's pool."""
        self._raise()
        pool = self._pools.get(tenant)
        if pool is None:
            pool = self._pools[tenant] = torch.cuda.MemPool(self._allocator)
        with self._executor.tenant(tenant), torch.cuda.use_mem_pool(pool):
            yield
        self._raise()

    def argument_at(self, tensor: torch.Tensor) -> BufferId | View:
        """The launch argument of a contiguous tensor: the buffer of its own, or the
        view of the segment that holds it from the tensor's start."""
        self._raise()
        if not tensor.is_contiguous():
            raise RefusedError("a launch argument must be a contiguous tensor")
        address = tensor.data_ptr()
        own = self._buffers.get(address)
        if own is not None:
            return own.buffer
        at = bisect.bisect_right(self._starts, address) - 1
        if at >= 0:
            start = self._starts[at]
            region = self._segments[start]
            if address < start + region.nbytes:
                return View(region.buffer, address - start)
        raise RefusedError(f"the tensor at {address:#x} lies in no buffer of the pool")

    def buffer(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        """A buffer of its own of `count` elements of `dtype` for the session's
        tenant, as a tensor over the whole buffer; the buffer is freed when no
        tensor holds it."""
        self._raise()
        tenant = self._executor.session_tenant
        nbytes = count * torch.empty((), dtype=dtype).element_size()
        if nbytes < 1:
            raise ValueError("a buffer has at least one byte")
        buffer = self._executor.allocate(nbytes)
        found = self._executor.catalog.buffer(buffer)
        if found is None:
            raise RuntimeError(f"buffer {buffer} is not live after its allocation")
        address = self._executor.pool_address + found.offset
        self._buffers[address] = _Region(buffer, tenant, nbytes)
        typestr = {torch.int32: "<i4", torch.int64: "<i8", torch.float64: "<f8"}[dtype]
        interface = _Interface(address, count, typestr)
        weakref.finalize(interface, self._free_buffer, address)
        return torch.as_tensor(interface, device="cuda")

    def launch(
        self,
        module: Module,
        grid: tuple[int, int, int],
        tensors: Sequence[torch.Tensor],
        sizes: Sequence[int],
        scalars: Sequence[int],
        *,
        guarded: bool = False,
    ) -> bool:
        """Launch `module` on the tensors, one for each buffer argument of the
        kernel as written in order, as a step of the session's tenant; whether
        the kernel ran, which a guarded launch's index checks decide. A module
        with the guard pass's checks takes a status buffer that the executor
        allocates, and raises `CheckFailedError` if a check failed
        (`opgate.hosts.launch`). The executor launches on CUDA's legacy stream,
        PyTorch's default stream, so the launch follows PyTorch's work on the
        tensors and precedes its later work."""
        if torch._C._cuda_getCurrentRawStream(self._device) != 0:
            raise RuntimeError("PyTorch's current stream is not the legacy stream")
        arguments = tuple(self.argument_at(tensor) for tensor in tensors)
        request = Launch(module.operator, grid, arguments, tuple(sizes), tuple(scalars), guarded)
        return launch(self._executor, self._executor, request)

    def release(self, tenant: TenantId) -> None:
        """End `tenant`'s pool: PyTorch gives back each segment that no tensor holds,
        now, and each other one when its last tensor is freed."""
        self._raise()
        self._release((tenant,))
        self._raise()

    def segments(self, tenant: TenantId) -> tuple[BufferId, ...]:
        """The segments of `tenant`'s pool."""
        return tuple(r.buffer for r in self._segments.values() if r.owner == tenant)

    def buffers(self, tenant: TenantId) -> tuple[BufferId, ...]:
        """The live buffers of their own of `tenant` that `buffer` gave."""
        return tuple(r.buffer for r in self._buffers.values() if r.owner == tenant)

    def _release(self, tenants: tuple[TenantId, ...]) -> None:
        for tenant in tenants:
            del self._pools[tenant]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _raise(self) -> None:
        if self._failure is not None:
            raise RuntimeError("the PyTorch allocator failed") from self._failure

    def _allocate(self, size: int, device: int, stream: int | None) -> int | None:
        """PyTorch's segment of `size` bytes: a buffer of the session's tenant."""
        if self._closed:
            return None
        try:
            tenant = self._executor.session_tenant
            buffer = self._executor.allocate(size)
            found = self._executor.catalog.buffer(buffer)
            if found is None:
                raise RuntimeError(f"buffer {buffer} is not live after its allocation")
            address = self._executor.pool_address + found.offset
            self._segments[address] = _Region(buffer, tenant, size)
            bisect.insort(self._starts, address)
            return address
        except (MemoryError, RefusedError):
            return None  # PyTorch raises its out-of-memory error
        except BaseException as error:
            self._failure = error
            return None

    def _free(self, address: int | None, size: int, device: int, stream: int | None) -> None:
        """PyTorch gives a segment back: it leaves the catalog under its owner."""
        if self._closed or address is None:
            return
        try:
            region = self._segments.pop(address)
            self._starts.remove(address)
            with self._executor.tenant(region.owner):
                self._executor.free(region.buffer)
        except BaseException as error:
            self._failure = error

    def _free_buffer(self, address: int) -> None:
        if self._closed:
            return
        region = self._buffers.pop(address)
        with self._executor.tenant(region.owner):
            self._executor.free(region.buffer)
