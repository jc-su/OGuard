"""A host launches a module with the guard pass's checks through `opgate.hosts.launch`.

`opgate.hosts.register` registers Sonnet's CUDA C++ `filter_select`, which the
gate refuses as written, as the guard pass's text, whose last buffer
parameter is the status buffer. `launch` takes the launch of the kernel as
written and gives it a status buffer of the session's tenant, which the
launch's step fills with zeros and which is freed after it: a launch whose
position counter starts at zero runs and reports no failed check, and the
defect's trigger, a counter that starts at `n`, raises `CheckFailedError`
naming the kernel after the kernel completes. A module that the gate accepts
as written has no status buffer and launches through the executor's kept
step.
"""

import re
from unittest.mock import Mock

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from eval.corpus.workloads import Loaded, prepare, workload
from opgate.catalog import TenantId
from opgate.executor import Executor
from opgate.frontends.guard import reports
from opgate.hosts import CheckFailedError, launch, register

TENANT = TenantId(1)


def _text(model: str, task: str, toolchain: str) -> str:
    return (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()


@pytest.mark.slow
@pytest.mark.gpu
def test_a_failed_check_raises_after_the_launch() -> None:
    n = 64
    text = _text("sonnet", "filter_select", "nvcc")
    binding = binding_of("sonnet", "filter_select", "nvcc", text)
    w = workload("filter_select", n, np.random.default_rng(3))
    w.arrays["quantity"][:] = 100.0
    w.arrays["quantity"][17] = 1.0  # one row passes the filter
    w.scalars["threshold"] = 24.0
    with Executor(0, 1 << 24) as executor, executor.tenant(TENANT):
        executor.reserve(1 << 24)
        registered = register(executor, text, binding)
        assert registered.text == text and registered.binding == binding
        assert registered.guarded is not None and registered.guarded.guarded
        assert reports(registered.module.binding) and not reports(binding)
        kernel = Loaded("sonnet", "filter_select", "nvcc", registered.module)
        for start in (0, n):
            w.arrays["count"][:] = start
            prepared = prepare(executor, TENANT, kernel, w)
            # the launch of the kernel as written: one buffer fewer than the module's
            assert len(prepared.launch.buffers) == len(registered.module.binding.bases) - 1
            if start == n:
                with pytest.raises(CheckFailedError, match=re.escape(binding.kernel)) as raised:
                    launch(executor, executor, prepared.launch)
                assert raised.value.kernel == binding.kernel
            else:
                assert launch(executor, executor, prepared.launch)
            count = executor.read(TENANT, prepared.buffers[2])
            assert np.frombuffer(count, dtype=np.int32).tolist() == [start + 1]
            # the status buffer was freed after the launch
            assert {b.id for b in executor.catalog.buffers} == set(prepared.buffers)
            for buffer in prepared.buffers:
                executor.free(buffer)


@pytest.mark.gpu
def test_a_module_without_checks_launches_through_the_kept_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _text("fable", "projection", "triton")
    binding = binding_of("fable", "projection", "triton", text)
    w = workload("projection", 1000, np.random.default_rng(4))
    with Executor(0, 1 << 24) as executor, executor.tenant(TENANT):
        executor.reserve(1 << 24)
        registered = register(executor, text, binding)
        assert registered.guarded is None and registered.module.text == text
        assert not reports(registered.module.binding)
        prepared = prepare(
            executor, TENANT, Loaded("fable", "projection", "triton", registered.module), w
        )
        kept = Mock(wraps=executor.launch)
        monkeypatch.setattr(executor, "launch", kept)
        assert launch(executor, executor, prepared.launch)
        kept.assert_called_once_with(prepared.launch)
        assert {b.id for b in executor.catalog.buffers} == set(prepared.buffers)
        for buffer in prepared.buffers:
            executor.free(buffer)
