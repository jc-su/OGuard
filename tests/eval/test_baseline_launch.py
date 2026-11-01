"""The parameters with which the MPS and Guardian baselines launch each corpus defect.

For every defect kernel of the unchecked corpus run, the values that
`eval.baselines.launch.parameters` gives, packed as the PTX `.param`
declarations of the kernel's entry lay them out (each at a multiple of its
alignment, in its declared bytes), fill every declared parameter, and each
value fits its declared bytes. For a Numba kernel, each array argument reads
back as Numba's array structure: meminfo and parent zero, the row count, the
element width, the buffer's address, the shape, and the stride. `launch`
passes the values with the driver's byte sizes, and refuses a value list of
another length, as one pointer per array was.
"""

import gzip
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

import eval.baselines.launch as baseline
from eval.corpus.adapter import Role, binding_of, claim_of, directory_of, read_raw, read_roles
from eval.corpus.run import TOOLCHAINS
from opgate import signature as sig
from opgate.binding import Base, Binding, Bounded
from opgate.catalog import evaluate

DECLARATION = re.compile(
    r"\.param\s+(?:\.align\s+(\d+)\s+)?\.([a-z]+)(\d+)\s+([\w$]+)(?:\[(\d+)\])?"
)
ENTRY = re.compile(r"\.entry\s+([\w$]+)\s*\(([^)]*)\)", re.S)
BASE_ADDRESS = 0x7F00_0000_0000


def _defects() -> list[str]:
    raw = gzip.decompress(
        (Path(__file__).with_name("fixtures") / "e3-results.jsonl.gz").read_bytes()
    )
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    return sorted(f"{r['model']}/{r['task']}/{r['toolchain']}" for r in rows)


def _declarations(text: str, kernel: str) -> list[tuple[int, int]]:
    """The bytes and the alignment of each `.param` of the entry `kernel`, in order."""
    (params,) = (p for name, p in ENTRY.findall(text) if name == kernel)
    declared: list[tuple[int, int]] = []
    for declaration in params.split(","):
        matched = DECLARATION.search(declaration)
        assert matched is not None, declaration
        align, _, bits, _, count = matched.groups()
        width = int(bits) // 8
        declared.append((width * int(count or 1), int(align or width)))
    return declared


def _pack(values: tuple[int, ...], declared: list[tuple[int, int]]) -> tuple[bytes, list[int]]:
    """The parameter buffer that the declarations lay out, and each parameter's offset."""
    buffer, offsets = bytearray(), []
    for value, (nbytes, align) in zip(values, declared, strict=True):
        buffer += bytes(-len(buffer) % align)
        offsets.append(len(buffer))
        buffer += value.to_bytes(nbytes, "little")  # raises if the value does not fit
    return bytes(buffer), offsets


def _launch_inputs(key: str) -> tuple[str, Binding, list[int], tuple[int, ...], tuple[int, ...]]:
    """The defect's text and binding, and an address for each array, a value
    for each size symbol, and each scalar's largest value."""
    model, task, toolchain = key.split("/")
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    raws = read_raw(directory_of(model, task) / "signature.json")
    roles = read_roles(model, task)
    arrays = [r for r in raws if roles.arguments[r.name] not in (Role.SIZE, Role.SCALAR)]
    addresses = [BASE_ADDRESS + index * 0x10_0000 for index in range(len(arrays))]
    sizes = tuple(3 + 2 * index for index in range(len(binding.sizes)))
    scalars = tuple(p.upper for _, p in binding.parameters if isinstance(p, Bounded))
    return text, binding, addresses, sizes, scalars


@pytest.mark.parametrize("key", _defects())
def test_the_values_fill_the_entrys_declared_parameters(key: str) -> None:
    text, binding, addresses, sizes, scalars = _launch_inputs(key)
    values = baseline.parameters(binding, addresses, sizes, scalars)
    declared = _declarations(text, binding.kernel)
    assert len(values) == len(declared)
    packed, offsets = _pack(values, declared)
    for (_, parameter), offset, (nbytes, _) in zip(
        binding.parameters, offsets, declared, strict=True
    ):
        if isinstance(parameter, Base):
            assert nbytes == 8
            assert int.from_bytes(packed[offset : offset + 8], "little") in addresses


@pytest.mark.parametrize("key", [k for k in _defects() if k.endswith("/numba")])
def test_a_numba_array_reads_back_as_numbas_array_structure(key: str) -> None:
    model, task, _ = key.split("/")
    text, binding, addresses, sizes, scalars = _launch_inputs(key)
    values = baseline.parameters(binding, addresses, sizes, scalars)
    packed, offsets = _pack(values, _declarations(text, binding.kernel))

    def field(index: int) -> int:
        return int.from_bytes(packed[offsets[index] : offsets[index] + 8], "little")

    signature, elements = claim_of(model, task)
    index, buffers = 0, iter(addresses)
    for argument in signature.arguments:
        if isinstance(argument, sig.Scalar):
            index += 1
            continue
        rows = evaluate(argument.rows, sizes)
        width = elements[argument.name].width
        meminfo, parent, nitems, itemsize, data, shape, stride = (
            field(index + k) for k in range(7)
        )
        assert (meminfo, parent) == (0, 0)
        assert (nitems, itemsize, data) == (rows, width, next(buffers))
        assert (shape, stride) == (rows, width)
        index += 7
    assert index == len(values)


@pytest.mark.parametrize("key", [k for k in _defects() if k.endswith("/numba")])
def test_launch_passes_the_drivers_sizes_and_refuses_one_pointer_per_array(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    text, binding, addresses, sizes, scalars = _launch_inputs(key)
    values = baseline.parameters(binding, addresses, sizes, scalars)
    widths = tuple(nbytes for nbytes, _ in _declarations(text, binding.kernel))
    monkeypatch.setattr(baseline, "parameter_sizes", lambda function: widths)
    driver = Mock()
    driver.cuLaunchKernel = Mock(return_value=(0,))
    monkeypatch.setattr(baseline, "driver", driver)
    baseline.launch("function", (1, 1, 1), binding.block, values)
    passed, types = driver.cuLaunchKernel.call_args.args[9]
    assert passed == values
    assert types == tuple(baseline.UNSIGNED[width] for width in widths)
    # the earlier launch: one pointer per array argument, then each scalar
    with pytest.raises(ValueError, match="values for a function with"):
        baseline.launch("function", (1, 1, 1), binding.block, (*addresses, *sizes, *scalars))
