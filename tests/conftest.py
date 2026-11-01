"""A deterministic driver for testing executor failures without device effects."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import opgate.executor as runtime


@pytest.fixture
def fake_executor(monkeypatch):
    driver = SimpleNamespace(
        CUdeviceptr=int, CUDA_MEMSET_NODE_PARAMS=SimpleNamespace, CU_STREAM_LEGACY=1
    )
    for name in (
        "cuInit",
        "cuCtxSetCurrent",
        "cuCtxSynchronize",
        "cuMemsetD8",
        "cuMemsetD16",
        "cuMemsetD32",
        "cuMemcpyHtoD",
        "cuMemcpyDtoH",
        "cuGraphLaunch",
        "cuGraphExecMemsetNodeSetParams",
        "cuGraphExecDestroy",
        "cuGraphDestroy",
        "cuModuleUnload",
        "cuMemFree",
        "cuDevicePrimaryCtxRelease",
    ):
        setattr(driver, name, Mock(return_value=(0,)))
    for name in (
        "cuDeviceGet",
        "cuDevicePrimaryCtxRetain",
        "cuGraphCreate",
        "cuGraphInstantiate",
        "cuGraphAddMemsetNode",
    ):
        setattr(driver, name, Mock(return_value=(0, 1)))
    # stream 0 is the legacy stream: both have the legacy stream's id
    driver.cuStreamGetId = Mock(return_value=(0, 7))
    driver.cuMemAlloc = Mock(side_effect=[(0, 4096), (0, 8192)])
    monkeypatch.setattr(runtime, "driver", driver)
    monkeypatch.setattr(
        runtime,
        "_MEMSET",
        {
            1: driver.cuMemsetD8,
            2: driver.cuMemsetD16,
            4: driver.cuMemsetD32,
        },
    )
    monkeypatch.setattr(runtime.trusted, "load", lambda _: SimpleNamespace(handle=1))
    executor = runtime.Executor(0, 4096)
    yield executor, driver
    executor.close()
