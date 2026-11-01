"""The mixed-operator experiment's kernels, decisions, inputs, requests, direct launches, and loop.

Every test but those marked `gpu` runs without a GPU: they use `eval.rq4_cost.mix`
and `eval.rq4_cost.fallback`, which import neither the service nor the driver's
functions. The `gpu` tests import `eval.rq4_cost.mixed`, whose service binds the
driver's functions when it is imported.
"""

import os
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from eval import ROOT
from eval.corpus.adapter import ELEMENTS, Points, Role, binding_of, claim_of, launch_of
from eval.corpus.workloads import Workload, workload
from eval.rq4_cost import mix
from eval.rq4_cost.fallback import BUILTINS, NODES, DeviceArray
from opgate.binding import Base
from opgate.catalog import At
from opgate.executor import READ_BYTES

REGISTRATION = ROOT / "eval/records/registration_2026-10-09.json"
GUARDED = ROOT / "eval/records/guarded_2026-10-09.json"


def _limit(kernel: mix.Kernel, text: str) -> mix.Block:
    """The grid limit of the kernel's schema, `mixed.schema_of`'s grid."""
    signature, _ = claim_of(kernel.model, kernel.task)
    *_, grid = launch_of(kernel.model, kernel.task, "numba", signature, text)
    return grid


def _workload(task: mix.Task) -> Workload:
    return workload(task, mix.ROWS[task], np.random.default_rng(mix.SEED))


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", mix.every_kernel(), ids=lambda k: k.kind)
def test_the_schema_gives_the_corpus_binding_and_grid(kernel: mix.Kernel) -> None:
    from eval.rq4_cost import mixed

    text = kernel.path.read_text()
    schema = mixed.schema_of(kernel, text)
    binding = mixed.binding_of_schema(kernel, text, schema)
    assert binding == binding_of(kernel.model, kernel.task, "numba", text)
    assert schema.grid == _limit(kernel, text)


@pytest.mark.gpu
@pytest.mark.slow
def test_the_live_decisions_equal_the_recorded_ones() -> None:
    from eval.rq4_cost import mixed

    kernels = mix.every_kernel()
    texts = {kernel: kernel.path.read_text() for kernel in kernels}
    schemas = {kernel: mixed.schema_of(kernel, texts[kernel]) for kernel in kernels}
    decided = mixed.decide(texts, schemas, os.sched_getaffinity(0))
    decisions = mix.recorded(REGISTRATION, GUARDED)
    live = {kernel.kind: decision.status for kernel, decision in decided.items()}
    assert live == {kind: decision["status"] for kind, decision in decisions.items()}
    for kernel, decision in decided.items():
        if decision.status == "refused":
            assert decision.reason == decisions[kernel.kind]["guarded"]["reason"]


def test_the_records_refuse_exactly_the_two_opus_kernels_of_the_mix() -> None:
    decisions = mix.recorded(REGISTRATION, GUARDED)
    refused = {kind for kind, decision in decisions.items() if decision["status"] == "refused"}
    assert refused == {"opus/string_contains/numba", "opus/groupby_sum/numba"}
    assert decisions["sonnet/groupby_sum/numba"]["status"] == "with_checks"
    opus = {k.kind for k in mix.kernels_of("service_fallback")} - refused
    assert {decisions[kind]["status"] for kind in opus} == {"as_written"}
    skipped = {}
    for configuration in mix.CONFIGURATIONS:
        runnable, names = mix.runnable_templates(
            configuration, lambda k: decisions[k.kind]["status"] != "refused"
        )
        assert runnable == [t for t in mix.MIXES[configuration] if t.name not in names]
        skipped[configuration] = names
    assert skipped == {
        "service": [],
        "service_refusals": ["like_sum", "group_by"],
        "service_fallback": [],
        "mps": [],
        "mps_shared": [],
    }


def test_a_refused_kernel_is_served_by_its_tasks_builtin_or_skipped() -> None:
    contains, groupby = mix.Kernel("opus", "string_contains"), mix.Kernel("opus", "groupby_sum")
    for configuration in ("service_fallback", "mps", "mps_shared"):
        assert mix.served_by(configuration, contains, False) == "builtin_string_contains"
        assert mix.served_by(configuration, groupby, False) == "builtin_groupby_sum"
        assert mix.served_by(configuration, contains, True) == contains.kind
    assert mix.served_by("service_refusals", contains, False) is None
    assert mix.served_by("service", groupby, True) == groupby.kind
    with pytest.raises(RuntimeError, match="no built-in"):
        mix.served_by("service_fallback", mix.Kernel("opus", "q6_sum"), False)
    assert [mix.host_of(c) for c in mix.CONFIGURATIONS] == [
        "service",
        "service",
        "service",
        "mps",
        "mps",
    ]
    assert [c for c in mix.CONFIGURATIONS if mix.serves_refused(c)] == [
        "service_fallback",
        "mps",
        "mps_shared",
    ]
    assert {c for c in mix.CONFIGURATIONS if c != "service"} == {
        c for c, templates in mix.MIXES.items() if templates == mix.AS_WRITTEN
    }


def test_the_saved_inputs_are_the_workloads(tmp_path: Path) -> None:
    directory = tmp_path / "workloads"
    mix.save_inputs(directory, create=True)
    mix.save_inputs(directory, create=False)
    for task in mix.TASKS:
        expected = _workload(task)
        read = mix.read_workload(directory, task)
        assert read.scalars == expected.scalars and read.sizes == expected.sizes
        assert read.arrays.keys() == expected.arrays.keys()
        for name, array in expected.arrays.items():
            assert read.arrays[name].dtype == array.dtype
            assert read.arrays[name].tobytes() == array.tobytes()
    (directory / "q6_sum" / "result.bin").write_bytes(b"\1" * 8)
    with pytest.raises(RuntimeError, match="changed"):
        mix.save_inputs(directory, create=False)


@pytest.mark.parametrize("kernel", mix.every_kernel(), ids=lambda k: k.kind)
def test_each_prepare_request_reads_back_within_the_mailbox(kernel: mix.Kernel) -> None:
    text = kernel.path.read_text()
    limit = _limit(kernel, text)
    w = _workload(kernel.task)
    buffers = mix.buffers_of(kernel)
    handles = {argument.name: f"handle-{argument.name}" for argument in buffers}
    request = mix.request_of(kernel, limit, w, "operator", handles)
    binding = binding_of(kernel.model, kernel.task, "numba", text)
    read, _ = mix.READS[kernel.task]
    assert request["read"] == handles[read]
    assert 0 < w.arrays[read].nbytes <= READ_BYTES
    assert request["buffers"] == [handles[argument.name] for argument in buffers]
    assert len(request["buffers"]) == sum(isinstance(p, Base) for _, p in binding.parameters)
    assert len(request["sizes"]) == len(binding.sizes)
    assert all(1 <= g <= m for g, m in zip(request["grid"], limit, strict=True))
    outputs = [a.name for a in buffers if mix.placement(a) == "output"]
    assert [f["buffer"] for f in request["fills"]] == [handles[name] for name in outputs]
    for fill, name in zip(request["fills"], outputs, strict=True):
        initial = w.arrays[name]
        assert fill["element"]["width"] == initial.itemsize
        assert np.all(initial.view(f"i{initial.itemsize}") == fill["value"]) or np.all(
            initial.view(f"u{initial.itemsize}") == fill["value"]
        )


def test_masked_sum_reads_the_first_operators_match() -> None:
    masked = [t for t in mix.MIXES["service"] if t.masked]
    assert [t.name for t in masked] == ["prefix_sum", "like_sum", "semi_join_sum"]
    for template in masked:
        first = {a.name: a for a in mix.buffers_of(template.first)}
        assert mix.placement(first[mix.LINK[1]]) == "output"
        assert first[mix.LINK[1]].dtype == "int8"
        assert mix.ROWS[template.first.task] == mix.ROWS["masked_sum"]
    shared = [a.name for a in mix.shared_arguments(mix.MASKED_SUM)]
    assert shared == ["values"]


def test_a_fill_needs_one_initial_value() -> None:
    argument = mix.Argument("out", "int32", Role.WRITE)
    with pytest.raises(ValueError, match="one initial value"):
        mix.fill_of(argument, np.array([0, 1], dtype=np.int32), "buffer")
    fill = mix.fill_of(argument, np.full(4, -1, dtype=np.int32), "buffer")
    assert fill == {"buffer": "buffer", "element": {"width": 4, "signed": True}, "value": -1}


@pytest.mark.parametrize("task", sorted(BUILTINS))
def test_a_builtin_takes_its_kernels_arguments(task: mix.Task) -> None:
    builtin = BUILTINS[task]
    assert builtin.read == mix.READS[task][0]
    assert {node.op for node in NODES} == {b.op for b in BUILTINS.values()} == set(mix.BY_OP)
    w = _workload(task)
    for model in ("opus", "sonnet"):
        arguments = mix.buffers_of(mix.Kernel(model, task))
        assert [a.name for a in arguments] == [o.name for o in builtin.operands]
        for argument, operand in zip(arguments, builtin.operands, strict=True):
            assert np.dtype(operand.typestr) == np.dtype(argument.dtype)
            wanted = "read" if mix.placement(argument) == "shared" else "write"
            assert operand.access == wanted
    handles = {o.name: f"handle-{o.name}" for o in builtin.operands}
    request = mix.builtin_request(builtin, w, handles)
    assert request == {
        "op": builtin.op,
        "buffers": handles,
        "sizes": {s: w.sizes[s] for s in builtin.sizes},
    }
    name = builtin.operands[0].name
    short = Workload({**w.arrays, name: w.arrays[name][:-1]}, w.scalars, w.sizes)
    with pytest.raises(ValueError, match="elements"):
        mix.builtin_request(builtin, short, handles)


def test_a_device_array_describes_its_memory() -> None:
    array = DeviceArray(1 << 32, "<f8", 5, True)
    assert array.__cuda_array_interface__ == {
        "shape": (5,),
        "strides": None,
        "typestr": "<f8",
        "data": (1 << 32, True),
        "version": 3,
    }


def test_the_shared_inputs_include_those_of_the_refused_kernels() -> None:
    labels = [mix.label_of(task, a.name) for task, a in mix.shared_inputs("service_fallback")]
    assert len(labels) == len(set(labels))
    for task in (t for t in mix.TASKS if t in BUILTINS):
        reads = [o.name for o in BUILTINS[task].operands if o.access == "read"]
        assert {mix.label_of(task, name) for name in reads} <= set(labels)
    assert "masked_sum/mask" not in labels
    assert mix.shared_inputs("mps") == mix.shared_inputs("service_fallback")


Step = tuple[mix.Kernel, str, mix.PrepareRequest, dict[str, int], dict]


def _placed(template: mix.Template) -> list[Step]:
    """Each kernel step of a template as an MPS tenant places it."""
    steps: list[Step] = []
    first: dict[str, str] = {}
    nbytes: dict[str, int] = {}
    for position, kernel in enumerate(template.kernels):
        text = kernel.path.read_text()
        w = _workload(kernel.task)
        handles: dict[str, str] = {}
        for argument in mix.buffers_of(kernel):
            if mix.linked(kernel, argument):
                handles[argument.name] = first[mix.LINK[1]]
            elif mix.placement(argument) == "shared":
                handles[argument.name] = mix.label_of(kernel.task, argument.name)
            else:
                handles[argument.name] = f"{template.name}/{kernel.kind}/{argument.name}"
                if not position:
                    first[argument.name] = handles[argument.name]
            nbytes.setdefault(handles[argument.name], int(w.arrays[argument.name].nbytes))
        request = mix.request_of(kernel, _limit(kernel, text), w, kernel.kind, handles)
        indexes = {
            handles[a.name]: (w.arrays[a.name].tobytes(), ELEMENTS[a.dtype])
            for a in mix.buffers_of(kernel)
            if isinstance(a.role, Points)
        }
        steps.append((kernel, text, request, dict(nbytes), indexes))
    return steps


@pytest.mark.parametrize("template", mix.AS_WRITTEN, ids=lambda t: t.name)
def test_the_catalog_checks_each_direct_launch_without_checks(template: mix.Template) -> None:
    decisions = mix.recorded(REGISTRATION, GUARDED)
    for kernel, text, request, nbytes, indexes in _placed(template):
        if decisions[kernel.kind]["status"] == "refused":
            assert kernel.task in BUILTINS
            continue
        binding = binding_of(kernel.model, kernel.task, "numba", text)
        checked, layout = mix.checked_launch(binding, request, nbytes, indexes)
        assert not checked.checks
        assert checked.grid == tuple(request["grid"])
        addresses = {h: (index + 1) << 32 for index, h in enumerate(layout.offsets)}
        values = mix.marshal(checked, [8] * len(checked.arguments), layout, addresses)
        pointed = [
            v
            for argument, v in zip(checked.arguments, values, strict=True)
            if isinstance(argument, At)
        ]
        assert pointed
        for value in pointed:
            (handle,) = (h for h, a in addresses.items() if a <= value <= a + nbytes[h])
            assert handle in request["buffers"]


def test_a_direct_launch_needs_the_range_of_its_index_inputs() -> None:
    template = mix.Template("string_equal", mix.Kernel("opus", "string_equal"), False)
    ((kernel, text, request, nbytes, indexes),) = _placed(template)
    assert indexes
    binding = binding_of(kernel.model, kernel.task, "numba", text)
    with pytest.raises(RuntimeError, match="refuses the direct launch|need no checks"):
        mix.checked_launch(binding, request, nbytes, {})


def test_a_layout_gives_every_address_one_buffer() -> None:
    layout = mix.Layout.of(["a", "b", "c"], {"a": 256, "b": 1, "c": 300})
    assert layout.offsets == {"a": 0, "b": 512, "c": 1024}
    assert layout.end == 1324
    assert [layout.owner(o) for o in (0, 256, 512, 513, 1024, 1324)] == list("aabbcc")
    with pytest.raises(ValueError, match="0 buffers"):
        layout.owner(300)
    with pytest.raises(ValueError, match="twice"):
        mix.Layout.of(["a", "a"], {"a": 8})


def test_a_fill_is_the_shortest_memset() -> None:
    def fill(width: int, signed: bool, value: int) -> mix.FillRequest:
        return {"buffer": "b", "element": {"width": width, "signed": signed}, "value": value}

    assert mix.memset_of(fill(8, False, 0), 64) == mix.Memset(1, 0, 64)
    assert mix.memset_of(fill(4, True, -1), 64) == mix.Memset(1, 0xFF, 64)
    assert mix.memset_of(fill(4, True, 1), 64) == mix.Memset(4, 1, 16)
    assert mix.memset_of(fill(2, False, 0x0101), 64) == mix.Memset(1, 1, 64)
    assert mix.memset_of(fill(8, False, 0x0000_0007_0000_0007), 64) == mix.Memset(4, 7, 16)
    assert mix.memset_of(fill(1, True, -2), 3) == mix.Memset(1, 0xFE, 3)
    with pytest.raises(ValueError, match="no memset"):
        mix.memset_of(fill(8, False, 1 << 40), 64)
    with pytest.raises(ValueError, match="whole elements"):
        mix.memset_of(fill(4, True, 0), 6)


def _masked_sum_plan() -> tuple[mix.Plan, dict[mix.Task, Workload]]:
    w = _workload("masked_sum")
    expected = struct.pack("<d", float(np.sum(w.arrays["values"][w.arrays["mask"] != 0])))
    workloads: dict[mix.Task, Workload] = {"masked_sum": w}
    plan, reads = mix.verify(
        "q6",
        [lambda: expected],
        [mix.MASKED_SUM],
        ["opus/masked_sum/numba"],
        lambda: [{"result": expected}],
        workloads,
        "tenant",
    )
    assert reads == [{"result": expected}]
    return plan, workloads


def test_the_verified_use_must_be_the_checked_output() -> None:
    plan, workloads = _masked_sum_plan()
    assert plan.verified == (plan.uses[0](),)
    wrong = struct.pack("<d", 0.5)
    with pytest.raises(RuntimeError, match="differ from the reference"):
        mix.verify(
            "q6",
            [lambda: wrong],
            [mix.MASKED_SUM],
            ["k"],
            lambda: [{"result": wrong}],
            workloads,
            "tenant",
        )
    with pytest.raises(RuntimeError, match="is not the output"):
        mix.verify(
            "q6",
            [lambda: wrong],
            [mix.MASKED_SUM],
            ["k"],
            lambda: [{"result": plan.verified[0]}],
            workloads,
            "tenant",
        )


def test_the_closed_loop_measures_the_queries_in_its_window() -> None:
    plan, _ = _masked_sum_plan()
    builtin = mix.Plan(
        "group_by",
        (lambda: struct.pack("<2d", 2.0, 1.0),),
        (mix.Kernel("opus", "groupby_sum"),),
        ("builtin_groupby_sum",),
        (struct.pack("<2d", 1.0, 2.0),),
    )
    loop = mix.closed_loop([plan, builtin], time.perf_counter(), 0.0, 0.05, "t")
    assert loop.latency and len(loop.latency) == len(loop.template) == len(loop.steps)
    assert set(loop.template) == {0, 1}
    assert all(len(steps) == 1 for steps in loop.steps)
    assert loop.calls["opus/masked_sum/numba"] + loop.calls["builtin_groupby_sum"] >= len(
        loop.latency
    )
    assert mix.calls_of([plan, builtin], loop)["builtin_groupby_sum"] == (
        loop.calls["builtin_groupby_sum"] + 1
    )
    share = mix.builtin_share(
        [loop],
        ["q6", "group_by"],
        {"q6": ["opus/masked_sum/numba"], "group_by": list(builtin.served)},
    )
    assert 0 < share < 1
    drifted = mix.Plan(
        "q6", (lambda: struct.pack("<d", 3.0),), plan.kernels, plan.served, plan.verified
    )
    with pytest.raises(RuntimeError, match="differs from"):
        mix.closed_loop([drifted], time.perf_counter(), 0.0, 0.05, "t")


def test_the_builtin_share_is_the_builtins_part_of_the_queries_time() -> None:
    loop = mix.Loop(
        template=[0, 1, 0],
        latency=[3.0, 1.0, 4.0],
        steps=[[1.0, 2.0], [1.0], [3.0, 1.0]],
        values=[[], [], []],
        calls={},
    )
    served = {"like_sum": ["builtin_string_contains", "opus/masked_sum/numba"], "q6": ["q6"]}
    assert mix.builtin_share([loop], ["like_sum", "q6"], served) == pytest.approx(4.0 / 8.0)
