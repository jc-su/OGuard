"""Allocation bounds and service records preserve the complete symbolic contract."""

from dataclasses import replace

import pytest

from opgate.binding import GRID_LIMIT, Binding
from opgate.hosts.service import binding_record, parse_binding
from opgate.signature import Affine, SizeSymbol

BINDING = Binding(
    "kernel", (SizeSymbol("k", 0, 49152),), (), (32, 1, 1), GRID_LIMIT, Affine.symbol(0)
)


def test_service_round_trip_keeps_the_extent_and_maximum():
    record = binding_record(BINDING)
    assert record["sizes"] == [{"name": "k", "minimum": 0, "maximum": 49152}]
    assert record["shared"] == {"constant": 0, "terms": [[0, 1]]}
    assert parse_binding(record) == BINDING


@pytest.mark.parametrize("field", ["maximum", "shared"])
def test_incomplete_allocation_records_fail_immediately(field):
    record = binding_record(BINDING)
    if field == "maximum":
        record["sizes"] = [{"name": "k", "minimum": 0}]
    else:
        record["shared"] = 0
    with pytest.raises(ValueError):
        parse_binding(record)


@pytest.mark.parametrize("extent", [Affine(-1), Affine(49153), Affine.symbol(1)])
def test_invalid_allocation_contract_fails_immediately(extent):
    with pytest.raises(ValueError):
        replace(BINDING, shared=extent)


def test_device_limit_cannot_be_hidden_by_a_larger_size_bound():
    with pytest.raises(ValueError, match="48 KiB"):
        replace(BINDING, sizes=(SizeSymbol("k", 0, 49153),))
