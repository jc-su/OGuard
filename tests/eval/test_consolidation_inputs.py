"""Consolidation replays actual table bytes and preserves measured outputs."""

import ctypes
import re
import struct
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from eval.rq4_cost import consolidation as experiment
from eval.tpch.data import Lineitem
from opgate.catalog import At, OperatorId
from opgate.hosts.service.protocol import Fields


@pytest.fixture
def table(monkeypatch):
    value = Lineitem(
        **{name: np.array([1, 2, 3], dtype=dtype) for name, dtype in experiment.INPUT_TYPES.items()}
    )
    monkeypatch.setattr(experiment, "generate", lambda scale, seed: (None, value))
    return value


def test_tables_are_saved_as_actual_bytes_and_read_without_regeneration(
    tmp_path, table, monkeypatch
):
    directory = tmp_path / "tables"
    experiment._save_inputs(directory, (0.1,), create=True)
    experiment._save_inputs(directory, (0.1,), create=False)

    def unexpected(*args, **kwargs):
        raise AssertionError("a worker regenerated its inputs")

    monkeypatch.setattr(experiment, "generate", unexpected)
    replayed = experiment._read_inputs(directory / "sf0.1")
    for name in experiment.INPUT_TYPES:
        actual = getattr(table, name).tobytes()
        assert (directory / "sf0.1" / f"{name}.bin").read_bytes() == actual
        assert getattr(replayed, name).tobytes() == actual


def test_changed_table_bytes_reject_resume(tmp_path, table):
    directory = tmp_path / "tables"
    experiment._save_inputs(directory, (1.0,), create=True)
    path = directory / "sf1" / "quantity.bin"
    path.write_bytes(np.array([1, 2, 4], dtype="<f8").tobytes())
    with pytest.raises(RuntimeError, match="recorded input changed"):
        experiment._save_inputs(directory, (1.0,), create=False)


def test_mismatched_column_lengths_fail_before_execution(tmp_path, table):
    directory = tmp_path / "tables"
    experiment._save_inputs(directory, (1.0,), create=True)
    (directory / "sf1" / "quantity.bin").write_bytes(np.array([1], dtype="<f8").tobytes())
    with pytest.raises(RuntimeError, match="different lengths"):
        experiment._read_inputs(directory / "sf1")


@pytest.mark.parametrize("service", [False, True])
def test_each_measured_latency_retains_its_actual_output(monkeypatch, service):
    monkeypatch.setattr(experiment, "WARMUP", 0)
    monkeypatch.setattr(experiment, "DURATION", 10)
    times = iter((1.0, 2.0, 11.0))
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(times))
    if service:
        prepared = []

        def prepare(**values):
            prepared.append(values)
            return 0

        mailbox = Mock(run=Mock(return_value=struct.pack("<d", 12.0)))
        plane = MagicMock()
        plane.__enter__.return_value = mailbox
        client = Mock(
            spec=experiment.Client,
            request=lambda op, **values: Fields({"ok": True, "buffer": "result"}, "response"),
            prepare=prepare,
            data_plane=Mock(return_value=plane),
        )
        handles = {name: name for name in ("shipdate", "discount", "quantity", "extendedprice")}
        stream = experiment._service_queries(client, 1, "operator", handles, 3, 12.0, 0.0)
        assert stream.outputs == [12.0]
        assert stream.request == {"op": "prepare", **prepared[0]}
        mailbox.run.assert_called_with(0, experiment._scalars())
    else:
        monkeypatch.setattr(experiment, "_direct_run", lambda *args: None)
        monkeypatch.setattr(experiment, "_direct_q6", lambda *args: 12.0)
        stream = experiment._direct_queries(
            Mock(spec=experiment.Session, tenant=1),
            Mock(spec=experiment.Module),
            Mock(spec=experiment.Columns, rows=3),
            12.0,
            0.0,
        )
        assert stream.outputs == [12.0]
    assert stream.q6 == [1.0]


def test_an_ipc_handle_round_trips_its_actual_bytes():
    raw = bytes(range(experiment.driver.CU_IPC_HANDLE_SIZE))
    assert experiment._handle_bytes(experiment._handle_of(raw)) == raw
    with pytest.raises(ValueError, match="IPC handle has"):
        experiment._handle_of(raw[:-1])
    with pytest.raises(experiment.DriverError, match="not an IPC handle"):
        experiment._handle_bytes(raw)


def _q6_module():
    text = experiment.MODEL_PTX.read_text()
    binding = experiment.binding_of("sonnet", "q6_sum", "numba", text)
    # each entry parameter's width in bytes, as the driver lays out the loaded function
    widths = tuple(int(bits) // 8 for bits in re.findall(r"\.param \.[usfb](\d+) ", text))
    return experiment.Module(OperatorId(0), text, binding, None, None, widths)


def test_the_shared_launch_points_at_the_mapped_columns_and_the_private_result():
    module = _q6_module()
    columns = tuple(0x7F00_0000_0000 + (index << 24) for index in range(4))
    result = 0x7F00_1000_0000
    kernel, values = experiment._shared_values(module, 3, 1000, columns, result)
    assert kernel.grid == (4, 1, 1) and len(values) == len(module.widths) == 41
    pointers = [i for i, argument in enumerate(kernel.arguments) if isinstance(argument, At)]
    assert [values[i] for i in pointers] == [*columns, result]
    assert values[35:40] == experiment._scalars() and values[40] == 1000
    with pytest.raises(ValueError):
        experiment._shared_values(module, 3, 1000, columns[:3], result)


def test_a_direct_launch_loop_records_its_plan_and_actual_outputs(monkeypatch):
    monkeypatch.setattr(experiment, "WARMUP", 0)
    monkeypatch.setattr(experiment, "DURATION", 10)
    times = iter((1.0, 2.0, 11.0))
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(experiment, "driver", Mock())
    monkeypatch.setattr(experiment, "call", lambda result: ())
    host = ctypes.create_string_buffer(struct.pack("<d", 12.0), 8)
    plan = experiment._LaunchPlan(
        1, None, (4, 1, 1), (256, 1, 1), (), (), None, ctypes.addressof(host), 0, False
    )
    stream = experiment._launch_queries(plan, 12.0, 0.0)
    assert stream.outputs == [12.0] and stream.q6 == [1.0]
    assert stream.request["fenced"] is False and stream.request["grid"] == [4, 1, 1]


@pytest.mark.parametrize("owners", [0, 1, 2])
def test_a_shared_cell_runs_one_owner_beside_its_tenants_and_keeps_its_evidence(
    tmp_path, monkeypatch, owners
):
    (tmp_path / "quantity.bin").write_bytes(bytes(8 * 3))
    started = []

    class Context:
        Barrier = Queue = staticmethod(lambda *args: Mock())

        @staticmethod
        def Process(target, args):
            started.append(target)
            return Mock()

    class Sampler:
        peak = 900

        def __init__(self, device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    evidence: dict[str, object] = {"compared_bytes": 108}
    streams = [experiment.Stream(t, [1.0], [12.0], {"fenced": False}) for t in (1, 2)]
    reports = [experiment.WorkerReport([s], None) for s in streams]
    reports += [experiment.WorkerReport([], None, None, evidence)] * owners
    monkeypatch.setattr(experiment.mp, "get_context", lambda method: Context)
    monkeypatch.setattr(experiment, "Sampler", Sampler)
    monkeypatch.setattr(experiment, "_collect", lambda queue, workers, count, where: reports)
    if owners != 1:
        with pytest.raises(RuntimeError, match="owners' evidence"):
            experiment.run("mps_shared", 0.1, 2, "1", inputs=tmp_path, baseline_memory=100)
        return
    cell = experiment.run("mps_shared", 0.1, 2, "1", inputs=tmp_path, baseline_memory=100)
    assert started == [experiment._shared_owner, *[experiment._shared_tenant] * 2]
    assert cell["shared"] == evidence and cell["memory_mib"] == 800 and cell["queries"] == 2
    assert cell["guardian"] is None and cell["services"] == []
