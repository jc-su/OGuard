"""One tenant runs the mixed-operator templates in each configuration, with checked outputs."""

import base64
import os
from pathlib import Path
from typing import assert_never

import numpy as np
import pytest

from eval import ROOT
from eval.corpus.workloads import Workload, check
from eval.rq4_cost import mix

REGISTRATION = ROOT / "eval/records/registration_2026-10-09.json"
GUARDED = ROOT / "eval/records/guarded_2026-10-09.json"


def _check_outputs(template: mix.Template, outputs: list[dict[str, str]], inputs: Path) -> None:
    """The retained outputs of each operator of `template` pass the task's reference."""
    reads = [
        {name: base64.b64decode(data, validate=True) for name, data in read.items()}
        for read in outputs
    ]
    assert len(reads) == len(template.kernels)
    for position, (kernel, read) in enumerate(zip(template.kernels, reads, strict=True)):
        w = mix.read_workload(inputs, kernel.task)
        if position:
            mask = np.frombuffer(reads[0][mix.LINK[1]], dtype=np.int8)
            w = Workload({**w.arrays, mix.LINK[0]: mask}, w.scalars, w.sizes)
        arrays = {
            name: np.frombuffer(data, dtype=w.arrays[name].dtype) for name, data in read.items()
        }
        assert check(kernel.task, w, arrays) is None, (template.name, kernel.kind)


@pytest.mark.gpu
@pytest.mark.parametrize("configuration", mix.CONFIGURATIONS)
def test_one_tenant_runs_the_mix(tmp_path: Path, configuration: mix.Configuration) -> None:
    from eval.rq4_cost import mixed
    from eval.rq4_cost.consolidation import _device

    inputs = tmp_path / "workloads"
    mix.save_inputs(inputs, create=True)
    decisions = mix.recorded(REGISTRATION, GUARDED)
    expected = {kind: decision["status"] for kind, decision in decisions.items()}
    cell = mixed.run_cell(
        configuration,
        1,
        _device(),
        inputs=inputs,
        expected=expected,
        proof_cores=os.sched_getaffinity(0),
        warmup=0.5,
        duration=1.0,
    )
    refused = ["like_sum", "group_by"] if configuration == "service_refusals" else []
    templates = {t.name: t for t in mix.MIXES[configuration]}
    assert cell["refused_templates"] == refused
    assert cell["templates"] == [name for name in templates if name not in refused]
    served = cell["served"]
    assert isinstance(served, dict)
    builtins = {op for steps in served.values() for op in steps if op in mix.BY_OP}
    assert builtins == (
        {"builtin_string_contains", "builtin_groupby_sum"}
        if mix.serves_refused(configuration)
        else set()
    )
    share = cell["builtin_share"]
    assert isinstance(share, float) and (share > 0) == bool(builtins)
    queries = cell["queries"]
    assert isinstance(queries, int) and queries > 0
    per_tenant, names = cell["per_tenant"], cell["templates"]
    assert isinstance(per_tenant, list) and isinstance(names, list)
    (tenant,) = per_tenant
    assert isinstance(tenant, dict)
    for name in names:
        _check_outputs(templates[name], tenant["verified_outputs"][name], inputs)
        _check_outputs(templates[name], tenant["final_outputs"][name], inputs)
    rows = {row["kind"]: row for row in tenant["registrations"]}
    for kernel in mix.kernels_of(configuration):
        refusal = expected[kernel.kind] == "refused"
        match mix.host_of(configuration):
            case "service":
                response = rows[kernel.kind]["response"]
                assert response["ok"] is not refusal
                if refusal:
                    assert response["error"]["kind"] == "refused"
                    assert (
                        response["error"]["reason"] == decisions[kernel.kind]["guarded"]["reason"]
                    )
            case "mps":
                # an MPS tenant loads only the kernels that the gate accepts
                assert (kernel.kind in rows) is not refusal
                if not refusal:
                    assert rows[kernel.kind]["decision"]["status"] == expected[kernel.kind]
            case _ as never:
                assert_never(never)
