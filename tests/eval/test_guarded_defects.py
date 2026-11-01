"""The guarded-defects experiment's triggers and layout, and the cost experiment's partition.

Every defect that the unchecked corpus run triggered has a trigger, whose
arrays name each array argument of the kernel's signature, with a buffer of
the second tenant before the tenant's first buffer and after each of them;
each buffer starts at a multiple of the catalog's alignment. Guardian's
partition for the executor's pool holds the whole pool.
"""

import gzip
import io
import tarfile
from pathlib import Path

import pytest

from eval.corpus.adapter import Role, directory_of, read_raw, read_roles
from eval.rq1_safety.guarded_defects import (
    LAST_NEIGHBOUR_BYTES,
    NEIGHBOUR_BYTES,
    TRIGGERS,
    _archive,
    _layout,
)
from eval.rq4_cost.guard_cost import POOL, _partition
from opgate.catalog import ALIGNMENT


def test_every_triggered_defect_has_a_trigger(tmp_path, monkeypatch) -> None:
    raw = gzip.decompress(
        (Path(__file__).with_name("fixtures") / "e3-results.jsonl.gz").read_bytes()
    )
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("e3/results.jsonl")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    monkeypatch.setattr("eval.rq1_safety.guarded_defects.E3_ARCHIVE", path)
    keys = sorted(f"{r['model']}/{r['task']}/{r['toolchain']}" for r in _archive())
    assert keys == sorted(TRIGGERS)
    assert len(keys) == 27


@pytest.mark.parametrize("key", sorted(k for k, t in TRIGGERS.items() if not t().large))
def test_a_trigger_gives_each_array_argument(key: str) -> None:
    model, task, _ = key.split("/")
    raws = read_raw(directory_of(model, task) / "signature.json")
    roles = read_roles(model, task)
    arrays = {r.name for r in raws if roles.arguments[r.name] not in (Role.SIZE, Role.SCALAR)}
    scalars = {r.name for r in raws if roles.arguments[r.name] is Role.SCALAR}
    trigger = TRIGGERS[key]()
    assert set(trigger.arrays) == arrays
    assert set(trigger.scalars) == scalars
    assert set(trigger.sizes) == set(roles.symbols)
    assert set(trigger.after) <= arrays


def test_the_layout_puts_a_neighbour_around_each_buffer() -> None:
    trigger = TRIGGERS["sonnet/filter_select/nvcc"]()
    names = ("quantity", "sel", "count")
    layout = _layout(trigger.arrays, names)
    assert layout.neighbours[0] == (0, NEIGHBOUR_BYTES)
    assert layout.neighbours[-1][1] == LAST_NEIGHBOUR_BYTES
    for number, name in enumerate(names):
        offset, size = layout.offsets[name], layout.sizes[name]
        assert offset % ALIGNMENT == 0
        before, after = layout.neighbours[number], layout.neighbours[number + 1]
        assert before[0] + before[1] <= offset
        assert offset + size <= after[0]
        assert layout.region(offset) == name
        assert layout.region(after[0]) == "neighbour"
    assert layout.end == layout.neighbours[-1][0] + LAST_NEIGHBOUR_BYTES


@pytest.mark.parametrize("address", [0, 1 << 21, (1 << 32) + (1 << 21), (7 << 34) - (1 << 21)])
def test_the_partition_holds_the_pool(address: int) -> None:
    base, mask = _partition(address)
    assert mask & (mask + 1) == 0
    assert base & mask == 0
    assert base <= address and address + POOL <= base + mask + 1
