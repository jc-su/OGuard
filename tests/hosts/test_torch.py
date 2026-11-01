"""PyTorch allocates a tenant's tensors in segments that are the tenant's catalog
buffers, and verified Triton operators run on them through the executor.

A tenant's tensors lie in its segments, which the catalog takes a step for
only when PyTorch takes or gives back one; a segment reaches another tenant
cleared. A model-written Triton projection runs on views of the segments and
gives PyTorch's result; a gather takes its index column, and Opus's
`scatter_add`, which the gate accepts as written, its group numbers, as a
buffer of its own, which a guarded launch checks when it runs; and a launch
that writes another tenant's tensor is refused.
"""

from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.corpus.adapter import binding_of, directory_of  # noqa: E402
from opgate.catalog import TenantId, View  # noqa: E402
from opgate.executor import Executor, Module, RefusedError  # noqa: E402
from opgate.hosts.torch import TorchHost  # noqa: E402

ALICE, BOB = TenantId(1), TenantId(2)
BLOCK = 1024  # the rows of a program of each kernel here


def register(executor: Executor, model: str, task: str) -> Module:
    """The model's Triton kernel for the task, as written: the gate must accept it."""
    text = (directory_of(model, task) / "triton.ptx").read_text()
    return executor.register(text, binding_of(model, task, "triton", text))


def programs(rows: int) -> tuple[int, int, int]:
    return (max(1, -(-rows // BLOCK)), 1, 1)


@pytest.mark.gpu
def test_a_tenants_tensors_lie_in_its_segments(monkeypatch):
    with Executor(0, 1 << 26) as executor, TorchHost(executor) as host:
        for tenant in (
            ALICE,
            BOB,
        ):
            with executor.tenant(tenant):
                executor.reserve((1 << 26) // 2)
        begin = Mock(wraps=executor._begin)
        monkeypatch.setattr(executor, "_begin", begin)
        with host.session(ALICE):
            tensors = [torch.empty(n, dtype=torch.float64, device="cuda") for n in (1, 100, 1000)]
            assert begin.call_count == 1  # one segment holds them
            (segment,) = host.segments(ALICE)
            found = executor.catalog.buffer(segment)
            assert found is not None and found.owner == ALICE
            for tensor in tensors:
                view = host.argument_at(tensor)
                assert isinstance(view, View) and view.buffer == segment
                assert tensor.data_ptr() == executor.pool_address + found.offset + view.offset
            del tensors, tensor
        outside = torch.empty(16, device="cuda")
        with pytest.raises(RefusedError, match="lies in no buffer"):
            host.argument_at(outside)
        host.release(ALICE)
        assert host.segments(ALICE) == () and not executor.catalog.buffers
        assert begin.call_count == 2


@pytest.mark.gpu
def test_a_segment_reaches_another_tenant_cleared():
    with Executor(0, 1 << 26) as executor, TorchHost(executor) as host:
        for tenant in (
            ALICE,
            BOB,
        ):
            with executor.tenant(tenant):
                executor.reserve((1 << 26) // 2)
        with host.session(ALICE):
            secret = torch.full((1 << 18,), 7, dtype=torch.int64, device="cuda")
            del secret
        host.release(ALICE)
        with executor.tenant(ALICE):
            executor.unreserve()
        with executor.tenant(BOB):
            # BOB's new reservation is the region that ALICE's held
            executor.unreserve()
            executor.reserve((1 << 26) // 2)
        with host.session(BOB):
            theirs = torch.empty(1 << 18, dtype=torch.int64, device="cuda")
            assert int(theirs.count_nonzero()) == 0
            del theirs
        host.release(BOB)


@pytest.mark.gpu
def test_a_verified_projection_runs_on_pytorch_tensors():
    rows = 100_003
    with Executor(0, 1 << 26) as executor, TorchHost(executor) as host:
        for tenant in (
            ALICE,
            BOB,
        ):
            with executor.tenant(tenant):
                executor.reserve((1 << 26) // 2)
        with executor.tenant(ALICE):
            module = register(executor, "fable", "projection")
        with host.session(ALICE):
            generator = torch.Generator(device="cuda").manual_seed(5)
            price = torch.rand(rows, dtype=torch.float64, device="cuda", generator=generator)
            discount = torch.rand(rows, dtype=torch.float64, device="cuda", generator=generator)
            out = torch.empty(rows, dtype=torch.float64, device="cuda")
            assert host.launch(module, programs(rows), (price, discount, out), (rows,), ())
            assert torch.equal(out, price * (1 - discount))
            del price, discount, out
        host.release(ALICE)


@pytest.mark.gpu
def test_a_gathers_index_column_is_a_checked_buffer_of_its_own():
    n, m = 1000, 4096
    with Executor(0, 1 << 26) as executor, TorchHost(executor) as host:
        for tenant in (
            ALICE,
            BOB,
        ):
            with executor.tenant(tenant):
                executor.reserve((1 << 26) // 2)
        with executor.tenant(ALICE):
            module = register(executor, "fable", "gather")
        with host.session(ALICE):
            column = torch.arange(n, dtype=torch.float64, device="cuda")
            out = torch.zeros(m, dtype=torch.float64, device="cuda")
            rows = torch.randint(0, n, (m,), dtype=torch.int32, device="cuda")
            # an index column inside a segment is refused, since its view does not
            # start a buffer
            with pytest.raises(RefusedError, match="index span's view"):
                host.launch(module, programs(m), (column, rows, out), (n, m), (), guarded=True)
            index = host.buffer(m, torch.int32)
            index.copy_(rows)
            assert host.launch(module, programs(m), (column, index, out), (n, m), (), guarded=True)
            assert torch.equal(out, column[rows.long()])
            # a row id out of range: the check fails and the kernel does not run
            index[17] = n
            out.zero_()
            assert not host.launch(
                module, programs(m), (column, index, out), (n, m), (), guarded=True
            )
            assert int(out.count_nonzero()) == 0
            del column, out, rows, index
        host.release(ALICE)
        assert not executor.catalog.buffers


@pytest.mark.gpu
def test_a_scatter_adds_group_numbers_are_a_checked_buffer_of_their_own():
    m, n = 100_003, 6
    with Executor(0, 1 << 26) as executor, TorchHost(executor) as host:
        for tenant in (
            ALICE,
            BOB,
        ):
            with executor.tenant(tenant):
                executor.reserve((1 << 26) // 2)
        with executor.tenant(ALICE):
            module = register(executor, "opus", "scatter_add")
        with host.session(ALICE):
            generator = torch.Generator(device="cuda").manual_seed(7)
            values = torch.rand(m, dtype=torch.float64, device="cuda", generator=generator)
            groups = torch.randint(
                0, n, (m,), dtype=torch.int32, device="cuda", generator=generator
            )
            out = torch.zeros(n, dtype=torch.float64, device="cuda")
            index = host.buffer(m, torch.int32)
            index.copy_(groups)
            assert host.launch(module, programs(m), (index, values, out), (m, n), (), guarded=True)
            expected = torch.zeros(n, dtype=torch.float64, device="cuda")
            expected.index_add_(0, groups.long(), values)
            # the order of the atomic additions changes the last bits of a sum
            assert torch.allclose(out, expected, rtol=1e-12, atol=0.0)
            # a group number out of range: the check fails and the kernel does not run
            index[17] = n
            out.zero_()
            assert not host.launch(
                module, programs(m), (index, values, out), (m, n), (), guarded=True
            )
            assert int(out.count_nonzero()) == 0
            del values, groups, out, index, expected
        host.release(ALICE)
        assert not executor.catalog.buffers


@pytest.mark.gpu
def test_a_launch_that_writes_another_tenants_tensor_is_refused():
    rows = 4096
    with Executor(0, 1 << 26) as executor, TorchHost(executor) as host:
        for tenant in (
            ALICE,
            BOB,
        ):
            with executor.tenant(tenant):
                executor.reserve((1 << 26) // 2)
        with executor.tenant(ALICE):
            module = register(executor, "fable", "projection")
        with host.session(BOB):
            theirs = torch.zeros(rows, dtype=torch.float64, device="cuda")
        with host.session(ALICE):
            price = torch.ones(rows, dtype=torch.float64, device="cuda")
            with pytest.raises(RefusedError, match="may not read"):
                host.launch(module, programs(rows), (price, price, theirs), (rows,), ())
            del price
        assert np.all(theirs.cpu().numpy() == 0)
        del theirs
        host.release(ALICE)
        host.release(BOB)
