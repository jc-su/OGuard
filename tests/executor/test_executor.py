"""The executor runs only what the gate and the catalog accept.

A kernel is loaded once, and only after the gate accepts it; a refused
kernel raises `RefusedError` with the gate's reason. A step runs only after
`check_step` accepts it: `out[i] = 2 * values[i]` runs on the tenant's own
buffers and gives the right values, and a launch on another tenant's
buffer, on a buffer smaller than its extent, or with an index input the
catalog has not checked is refused and leaves the catalog as it was. A
guarded launch checks such an input when it runs, and runs its kernel only
if every element lies in the range. A region a tenant frees is cleared
before another tenant gets it, and kept when the same tenant gets it back.

Each parameter is marshaled at the size the driver gives for the loaded
function, also when the text's parameter list holds comments. A binding
that names the parameters in another order than the text, a text the
driver does not load, and a block or shared memory the device cannot
launch the function with are refused, and the executor goes on.

A captured graph performs an accepted step, fills as memset or fill nodes
and a guarded kernel in a conditional node; each replay is a step the
catalog checks, with effects of the graph's shape, and updates every node.
"""

import struct
from dataclasses import replace

import pytest

from opgate import signature as sig
from opgate.binding import ElementType
from opgate.catalog import (
    Checked,
    Fill,
    Grant,
    Launch,
    Raw,
    Step,
    TenantId,
    Write,
)
from opgate.executor import BufferId, Executor, Module, RefusedError
from opgate.frontends import Target
from opgate.frontends.nvcc import nvcc_operator
from opgate.signature import Affine, Direction

ALICE, BOB = TenantId(1), TenantId(2)
POOL = 1 << 24
DOUBLE = ElementType(8, False)
INDEX = ElementType(4, True)

SCALE_SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"),),
    arguments=(
        sig.Span("values", "values", Direction.READ, Affine.symbol(0)),
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
GATHER_SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"), sig.SizeSymbol("m")),
    arguments=(
        sig.Span("col", "col", Direction.READ, Affine.symbol(0)),
        sig.IndexIn("sel", "sel", Affine.symbol(1), "col"),
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(1)),
        sig.Scalar("m", sig.SizeOf(Affine.symbol(1))),
    ),
)


def scale(bound: str) -> str:
    """`out[i] = 2 * values[i]` for each thread's `i` with `i < bound`."""
    return f"""
extern "C" __global__ void scale(const double* values, double* out, long long n)
{{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < {bound})
        out[i] = 2 * values[i];
}}
"""


GATHER = """
extern "C" __global__ void gather(const double* col, const int* sel, double* out, long long m)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < m)
        out[i] = col[sel[i]];
}
"""


def doubles(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}d", *values)


def register_scale(executor: Executor) -> Module:
    operator = nvcc_operator(scale("n"), "scale", SCALE_SIGNATURE, (256, 1, 1), Target(8, 6))
    return executor.register(operator.text, operator.binding({"values": DOUBLE, "out": DOUBLE}))


@pytest.fixture
def executor() -> Executor:
    """An executor whose pool's two halves are ALICE's and BOB's reservations."""
    executor = Executor(0, POOL)
    for tenant in (ALICE, BOB):
        with executor.tenant(tenant):
            executor.reserve(POOL // 2)
    return executor


@pytest.mark.gpu
def test_an_accepted_kernel_is_loaded_once(executor: Executor) -> None:
    with executor:
        module = register_scale(executor)
        assert executor.register(module.text, module.binding) is module
        assert executor.catalog.binding(module.operator) == module.binding


@pytest.mark.gpu
def test_a_refused_kernel_raises_with_the_instruction(executor: Executor) -> None:
    operator = nvcc_operator(scale("n + 1"), "scale", SCALE_SIGNATURE, (256, 1, 1), Target(8, 6))
    with executor, pytest.raises(RefusedError) as refused:
        executor.register(operator.text, operator.binding({"values": DOUBLE, "out": DOUBLE}))
    assert refused.value.reason.startswith(
        "scale: the access is not proved inside its buffer, a shared variable, or a local "
        "variable: ld.global.f64 "
    )


@pytest.mark.gpu
def test_a_checked_launch_runs_on_the_tenants_buffers(executor: Executor) -> None:
    n = 1000
    with executor, executor.tenant(ALICE):
        module = register_scale(executor)
        values, out = executor.allocate(8 * n), executor.allocate(8 * n)
        launch = Launch(module.operator, (4, 1, 1), (values, out), (n,), (), False)
        executor.execute(
            Step(ALICE, (Write(values, doubles([float(i) for i in range(n)]), None), launch))
        )
        result = struct.unpack(f"<{n}d", executor.read(ALICE, out))
        assert result == tuple(2.0 * i for i in range(n))
        assert executor.catalog.buffer(out) is not None


@pytest.mark.gpu
def test_a_launch_the_catalog_refuses_leaves_no_trace(executor: Executor) -> None:
    n = 1000
    with executor:
        module = register_scale(executor)
        with executor.tenant(ALICE):
            values, small = executor.allocate(8 * n), executor.allocate(8 * (n - 1))
        with executor.tenant(BOB):
            theirs = executor.allocate(8 * n)
        before = executor.catalog
        refusals = {
            "the tenant may not read a buffer argument": (
                values,
                theirs,
            ),
            "a buffer is smaller than its extent": (values, small),
        }
        for reason, buffers in refusals.items():
            launch = Launch(module.operator, (4, 1, 1), buffers, (n,), (), False)
            with executor.tenant(ALICE), pytest.raises(RefusedError, match=reason):
                executor.execute(Step(ALICE, (Write(values, b"\0" * 8, None), launch)))
            assert executor.catalog == before
        with executor.tenant(BOB), pytest.raises(RefusedError, match="may not read"):
            executor.read(BOB, values)


@pytest.mark.gpu
def test_an_index_input_must_be_checked_inside_its_range(executor: Executor) -> None:
    n = 100
    operator = nvcc_operator(GATHER, "gather", GATHER_SIGNATURE, (256, 1, 1), Target(8, 6))
    binding = operator.binding({"col": DOUBLE, "sel": INDEX, "out": DOUBLE})
    with executor, executor.tenant(ALICE):
        module = executor.register(operator.text, binding)
        col, sel, out = (executor.allocate(size) for size in (8 * n, 4 * n, 8 * n))
        launch = Launch(module.operator, (1, 1, 1), (col, sel, out), (n, n), (), False)
        rows = struct.pack(f"<{n}i", *reversed(range(n)))
        column = doubles([float(i) for i in range(n)])
        with pytest.raises(RefusedError, match="not checked inside its declared range"):
            executor.execute(
                Step(ALICE, (Write(col, column, None), Write(sel, rows, None), launch))
            )
        past = struct.pack(f"<{n}i", *range(1, n + 1))
        with pytest.raises(RefusedError, match="not checked inside its declared range"):
            executor.execute(Step(ALICE, (Write(sel, past, INDEX), launch)))
        executor.execute(Step(ALICE, (Write(col, column, None), Write(sel, rows, INDEX), launch)))
        found = executor.catalog.buffer(sel)
        assert found is not None and found.status == Checked(INDEX, 0, n - 1)
        result = struct.unpack(f"<{n}d", executor.read(ALICE, out))
        assert result == tuple(float(n - 1 - i) for i in range(n))


def register_gather(executor: Executor) -> Module:
    operator = nvcc_operator(GATHER, "gather", GATHER_SIGNATURE, (256, 1, 1), Target(8, 6))
    return executor.register(
        operator.text, operator.binding({"col": DOUBLE, "sel": INDEX, "out": DOUBLE})
    )


def rows(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


@pytest.mark.gpu
def test_a_guarded_launch_checks_its_index_input_when_it_runs(executor: Executor) -> None:
    n = 100
    with executor, executor.tenant(ALICE):
        module = register_gather(executor)
        col, sel, out = (executor.allocate(size) for size in (8 * n, 4 * n, 8 * n))
        launch = Launch(module.operator, (1, 1, 1), (col, sel, out), (n, n), (), True)
        column = doubles([float(i) for i in range(n)])
        reverse = rows(list(reversed(range(n))))
        writes = (Write(col, column, None), Write(sel, reverse, None), Fill(out, DOUBLE, 0))
        assert executor.execute(Step(ALICE, (*writes, launch))) == (True,)
        result = struct.unpack(f"<{n}d", executor.read(ALICE, out))
        assert result == tuple(float(n - 1 - i) for i in range(n))
        # a check when the launch runs leaves the status as it was
        found = executor.catalog.buffer(sel)
        assert found is not None and found.status == Raw()
        past = rows(list(range(1, n + 1)))
        step = Step(ALICE, (Write(sel, past, None), Fill(out, DOUBLE, 0), launch))
        assert executor.execute(step) == (False,)
        assert executor.read(ALICE, out) == bytes(8 * n)


@pytest.mark.gpu
def test_a_graph_replays_checked_steps(executor: Executor) -> None:
    n = 1000
    with executor, executor.tenant(ALICE):
        module = register_scale(executor)
        a, b, out1, out2 = (executor.allocate(8 * n) for _ in range(4))
        first, second = [float(i) for i in range(n)], [1000.0 + i for i in range(n)]
        executor.execute(
            Step(ALICE, (Write(a, doubles(first), None), Write(b, doubles(second), None)))
        )

        def step(values: BufferId, out: BufferId) -> Step:
            launch = Launch(module.operator, (4, 1, 1), (values, out), (n,), (), False)
            return Step(ALICE, (Fill(out, DOUBLE, 5), launch))

        graph = executor.capture(step(a, out1))
        assert executor.ran(graph) == (True,)
        assert struct.unpack(f"<{n}d", executor.read(ALICE, out1)) == tuple(2 * v for v in first)
        executor.replay(graph, step(b, out2))
        assert executor.ran(graph) == (True,)
        assert struct.unpack(f"<{n}d", executor.read(ALICE, out2)) == tuple(2 * v for v in second)
        # every replay is a step the catalog checks, of the graph's shape
        with executor.tenant(BOB):
            theirs = executor.allocate(8 * n)
        before = executor.catalog
        with pytest.raises(RefusedError, match="may not read"):
            executor.replay(graph, step(a, theirs))
        launch = Launch(module.operator, (4, 1, 1), (a, out1), (n,), (), False)
        with pytest.raises(ValueError, match="shape"):
            executor.replay(graph, Step(ALICE, (launch,)))
        with pytest.raises(ValueError, match="upload"):
            executor.capture(Step(ALICE, (Write(a, doubles(first), None), launch)))
        assert executor.catalog == before
        executor.release(graph)


@pytest.mark.gpu
def test_a_graph_runs_a_guarded_kernel_only_if_its_checks_hold(executor: Executor) -> None:
    n = 100
    with executor, executor.tenant(ALICE):
        module = register_gather(executor)
        col, good, bad, out = (executor.allocate(size) for size in (8 * n, 4 * n, 4 * n, 8 * n))
        column = doubles([float(i) for i in range(n)])
        reverse, past = rows(list(reversed(range(n)))), rows(list(range(1, n + 1)))
        writes = (Write(col, column, None), Write(good, reverse, None), Write(bad, past, None))
        executor.execute(Step(ALICE, writes))

        def step(sel: BufferId) -> Step:
            launch = Launch(module.operator, (1, 1, 1), (col, sel, out), (n, n), (), True)
            return Step(ALICE, (Fill(out, DOUBLE, 0), launch))

        graph = executor.capture(step(good))
        assert executor.ran(graph) == (True,)
        expected = tuple(float(n - 1 - i) for i in range(n))
        assert struct.unpack(f"<{n}d", executor.read(ALICE, out)) == expected
        executor.replay(graph, step(bad))
        assert executor.ran(graph) == (False,)
        assert executor.read(ALICE, out) == bytes(8 * n)
        executor.replay(graph, step(good))
        assert executor.ran(graph) == (True,)
        assert struct.unpack(f"<{n}d", executor.read(ALICE, out)) == expected


@pytest.mark.gpu
def test_a_freed_region_is_cleared_for_another_tenant(executor: Executor) -> None:
    with executor:
        with executor.tenant(ALICE):
            secret = executor.allocate(256)
            executor.execute(Step(ALICE, (Fill(secret, DOUBLE, 7),)))
            executor.free(secret)
            again = executor.allocate(256)
            assert executor.read(ALICE, again) == struct.pack("<32q", *([7] * 32))
            executor.free(again)
            executor.unreserve()
        with executor.tenant(BOB):
            # BOB's new reservation is the region that ALICE's held
            executor.unreserve()
            executor.reserve(POOL // 2)
            theirs = executor.allocate(256)
            assert executor.read(BOB, theirs) == bytes(256)
            found = executor.catalog.buffer(theirs)
            assert found is not None and found.offset == 0 and found.status == Raw()
            executor.execute(Step(BOB, (Grant(theirs, ALICE),)))
            with executor.tenant(ALICE):
                assert executor.read(ALICE, theirs) == bytes(256)


SCALE32 = """
extern "C" __global__ void scale(const double* values, double* out, int n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = 2 * values[i];
}
"""
DECLARED = """scale(
\t.param .u64 scale_param_0,
\t.param .u64 scale_param_1,
\t.param .u32 scale_param_2
)"""
# the same parameters, with comments that hold commas, parentheses, and a
# declaration, and with runs of spaces, tabs, and line breaks
COMMENTED = """scale (  // the values ), .param .u64 ghost,
\t.param/* a comma, and ) */.u64 scale_param_0 ,
\t\t.param   .u64

  scale_param_1/* ) */,
  .param .u32 /* , .param .u64 ghost */ scale_param_2 // n, the rows
)"""


def run_scale(executor: Executor, module: Module, n: int) -> tuple[float, ...]:
    """`out` of the scale operator on `values[i] = i`, for the session's tenant."""
    tenant = executor.session_tenant
    values, out = executor.allocate(8 * n), executor.allocate(8 * n)
    launch = Launch(module.operator, (-(-n // 256), 1, 1), (values, out), (n,), (), False)
    column = doubles([float(i) for i in range(n)])
    executor.execute(Step(tenant, (Write(values, column, None), launch)))
    result = struct.unpack(f"<{n}d", executor.read(tenant, out))
    executor.free(values)
    executor.free(out)
    return result


@pytest.mark.gpu
def test_parameters_declared_with_comments_and_odd_whitespace_marshal_correctly(
    executor: Executor,
) -> None:
    operator = nvcc_operator(SCALE32, "scale", SCALE_SIGNATURE, (256, 1, 1), Target(8, 6))
    assert DECLARED in operator.text
    text = operator.text.replace(DECLARED, COMMENTED)
    n = 1000
    with executor, executor.tenant(ALICE):
        module = executor.register(text, operator.binding({"values": DOUBLE, "out": DOUBLE}))
        # the sizes the driver gives for the loaded function
        assert module.widths == (8, 8, 4)
        assert run_scale(executor, module, n) == tuple(2.0 * i for i in range(n))


@pytest.mark.gpu
def test_a_binding_must_name_the_parameters_in_the_texts_order(executor: Executor) -> None:
    operator = nvcc_operator(scale("n"), "scale", SCALE_SIGNATURE, (256, 1, 1), Target(8, 6))
    binding = operator.binding({"values": DOUBLE, "out": DOUBLE})
    values, out, n = binding.parameters
    # the gate finds each parameter by its name and accepts this binding; the
    # executor marshals each parameter by its position and refuses it
    swapped = replace(binding, parameters=(out, values, n))
    with executor:
        with pytest.raises(RefusedError, match="the binding names the parameters"):
            executor.register(operator.text, swapped)
        assert not executor.catalog.operators


@pytest.mark.gpu
def test_a_text_the_driver_does_not_load_is_refused_and_the_executor_goes_on(
    executor: Executor,
) -> None:
    operator = nvcc_operator(scale("n"), "scale", SCALE_SIGNATURE, (256, 1, 1), Target(8, 6))
    newer = operator.text.replace(".target sm_86", ".target sm_90")
    with executor, executor.tenant(ALICE):
        with pytest.raises(RefusedError, match="the driver does not load it"):
            executor.register(newer, operator.binding({"values": DOUBLE, "out": DOUBLE}))
        assert not executor.catalog.operators
        module = register_scale(executor)
        assert run_scale(executor, module, 100) == tuple(2.0 * i for i in range(100))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("block", "reason"),
    [((32, 32, 2), "a block of 2048 threads"), ((1, 1, 65), r"exceeds the device's \(1024")],
)
def test_a_block_the_device_cannot_launch_is_refused(
    executor: Executor, block: tuple[int, int, int], reason: str
) -> None:
    operator = nvcc_operator(scale("n"), "scale", SCALE_SIGNATURE, block, Target(8, 6))
    with executor, executor.tenant(ALICE):
        with pytest.raises(RefusedError, match=reason):
            executor.register(operator.text, operator.binding({"values": DOUBLE, "out": DOUBLE}))
        assert not executor.catalog.operators
        module = register_scale(executor)
        assert run_scale(executor, module, 100) == tuple(2.0 * i for i in range(100))


STAGED = """
extern "C" __global__ void staged(const double* values, double* out, long long n)
{
    __shared__ double stage[4096];
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n)
        stage[threadIdx.x] = values[i];
    __syncthreads();
    if (i < n)
        out[i] = 2 * stage[threadIdx.x];
}
"""


@pytest.mark.gpu
def test_shared_memory_past_the_devices_limit_for_a_block_is_refused(executor: Executor) -> None:
    # 32 KiB of static shared memory, and the device's 48 KiB for a block
    with executor:
        for shared, accepted in ((16384, True), (16385, False)):
            operator = nvcc_operator(
                STAGED, "staged", SCALE_SIGNATURE, (256, 1, 1), Target(8, 6), shared=Affine(shared)
            )
            binding = operator.binding({"values": DOUBLE, "out": DOUBLE})
            if accepted:
                executor.register(operator.text, binding)
            else:
                with pytest.raises(RefusedError, match="32768 bytes of static and up to 16385"):
                    executor.register(operator.text, binding)
        assert len(executor.catalog.operators) == 1
