"""The Numba front end: numba-cuda at NVVM optimization level 1.

numba-cuda's public API compiles at NVVM level 3 or 0 only, so
`numba_operator` runs the steps of `numba.cuda.compiler.compile` for a
kernel, `compile_cuda` and the kernel fixup, with `nvvm_options={"opt": 1}`
(design section 5.9). A Numba kernel takes each one-dimensional array as
seven parameters (`Convention.NUMBA`).

Numba lowers `range(start, stop, step)` as a trip count, a signed division,
that it counts down: no iteration compares the index with `stop`, so no
bound test guards the body's accesses. The front end installs its own
lowering of range iteration in numba-cuda's target, `RANGES` (design
decision 25). The iterator runs forward, negating the position for a negative
step. It uses an inclusive last position, one below the direction-normalized
stop, so a descending stop at the minimum integer has a representable bound.
It detects empty ranges before normalizing their endpoints, compares the
position with the bound before each iteration, and yields the original value.
The position advances unconditionally; the validity bit controls whether the
body runs, and exhaustion is sticky. A wrapped next position ends the range,
where Python's range has already ended. The supported types are signed 32-bit,
signed 64-bit, and unsigned 64-bit integers, with a nonzero step. The canary
tests in `tests/frontends` detect a change in the internal compile steps and
in the loop that the lowering gives.

Imports numba-cuda, `opgate.signature`, `opgate.binding`, and
`opgate.frontends`.
"""

from collections.abc import Callable

from llvmlite import ir
from numba.core import sigutils, types
from numba.core.compiler_lock import global_compiler_lock

# numba-cuda replaces numba's own `numba.cuda` when numba imports it; the type
# checker reads numba's own modules, which lack these names
from numba.cuda import cgutils  # ty: ignore[unresolved-import]
from numba.cuda.compiler import compile_cuda, kernel_fixup  # ty: ignore[unresolved-import]
from numba.cuda.core.imputils import (  # ty: ignore[unresolved-import]
    RefType,
    Registry,
    impl_ret_untracked,
    iternext_impl,
)
from numba.cuda.descriptor import cuda_target

from opgate.binding import GRID_LIMIT, NO_SHARED, Convention, validate_grid, validate_shared
from opgate.frontends import Operator, Target
from opgate.signature import Affine, Signature

__all__ = ["RANGES", "install_ranges", "numba_operator"]

RANGES = Registry("opgate_ranges")


def _forward(builder: ir.IRBuilder, up: ir.Value, value: ir.Value, signed: bool) -> ir.Value:
    """Normalize direction with modular negation; the bound handles MIN."""
    return builder.select(up, value, builder.neg(value)) if signed else value


def _install_range(state_type: types.RangeType, iter_type: types.RangeIteratorType) -> None:
    """Register the lowering of range iteration for one integer type."""
    int_type = iter_type.yield_type
    signed = int_type.signed
    zero = ir.Constant(ir.IntType(int_type.bitwidth), 0)
    state_struct = cgutils.create_struct_proxy(state_type)
    iter_struct = cgutils.create_struct_proxy(iter_type)

    def up_of(builder: ir.IRBuilder, step: ir.Value) -> ir.Value:
        return (
            builder.icmp_signed(">", step, zero)
            if signed
            else builder.icmp_unsigned("!=", step, zero)
        )

    def below(builder: ir.IRBuilder, a: ir.Value, b: ir.Value) -> ir.Value:
        return builder.icmp_signed("<", a, b) if signed else builder.icmp_unsigned("<", a, b)

    @RANGES.lower("getiter", state_type)
    def getiter(
        context: object, builder: ir.IRBuilder, sig: object, args: list[ir.Value]
    ) -> ir.Value:
        (value,) = args
        state = state_struct(context, builder, value)
        with cgutils.if_unlikely(builder, builder.icmp_unsigned("==", state.step, zero)):
            context.fndesc.call_conv.return_user_exc(  # ty: ignore[unresolved-attribute]
                builder, ValueError, ("range() arg 3 must not be zero",)
            )
        up = up_of(builder, state.step)
        position = cgutils.alloca_once(builder, state.start.type)
        builder.store(_forward(builder, up, state.start, signed), position)
        ended = cgutils.alloca_once(builder, state.start.type)
        empty = (
            builder.select(
                up,
                builder.icmp_signed(">=", state.start, state.stop),
                builder.icmp_signed("<=", state.start, state.stop),
            )
            if signed
            else builder.icmp_unsigned(">=", state.start, state.stop)
        )
        builder.store(builder.zext(empty, state.start.type), ended)
        iterator = iter_struct(context, builder)
        iterator.iter = position
        iterator.stop = builder.sub(
            _forward(builder, up, state.stop, signed), ir.Constant(state.stop.type, 1)
        )
        iterator.step = state.step
        iterator.count = ended
        return impl_ret_untracked(context, builder, iter_type, iterator._getvalue())

    @RANGES.lower("iternext", iter_type)
    @iternext_impl(RefType.BORROWED)
    def iternext(
        context: object, builder: ir.IRBuilder, sig: object, args: list[ir.Value], result: object
    ) -> None:
        (value,) = args
        iterator = iter_struct(context, builder, value)
        up = up_of(builder, iterator.step)
        # The magnitude is interpreted by modular addition; -MIN has its
        # unsigned magnitude, and the order test below detects a wrapped sum.
        step = (
            builder.select(up, iterator.step, builder.neg(iterator.step))
            if signed
            else iterator.step
        )
        position = builder.load(iterator.iter)
        live = builder.icmp_unsigned("==", builder.load(iterator.count), zero)
        within = (
            builder.icmp_signed("<=", position, iterator.stop)
            if signed
            else builder.icmp_unsigned("<=", position, iterator.stop)
        )
        valid = builder.and_(live, within)
        result.set_valid(valid)  # ty: ignore[unresolved-attribute]
        # Yield is observed only when valid. Advancing on every invocation keeps
        # a plain induction step in PTX; the sticky flag prevents resumption.
        result.yield_(_forward(builder, up, position, signed))  # ty: ignore[unresolved-attribute]
        # Preserve the position-based loop test: NVVM otherwise replaces some
        # fixed ranges by countdown loops. The opaque increment emits one plain
        # modular PTX add, which the gate parses and proves like any other add.
        register = "l" if int_type.bitwidth == 64 else "r"
        advance = ir.InlineAsm(
            ir.FunctionType(position.type, [position.type, position.type]),
            f"add.u{int_type.bitwidth} $0, $1, $2;",
            f"={register},{register},{register}",
            side_effect=False,
        )
        following = builder.call(advance, [position, step])
        builder.store(following, iterator.iter)
        wrapped = below(builder, following, position)
        ended = builder.or_(builder.not_(valid), wrapped)
        builder.store(builder.zext(ended, position.type), iterator.count)


for _state, _iter in (
    (types.range_state32_type, types.range_iter32_type),
    (types.range_state64_type, types.range_iter64_type),
    (types.unsigned_range_state64_type, types.unsigned_range_iter64_type),
):
    _install_range(_state, _iter)


def install_ranges() -> None:
    """Make `RANGES` the lowering of range iteration in numba-cuda's target.

    A refresh loads numba-cuda's own registries, which it loads at the first
    compile otherwise; `RANGES`, installed after them, replaces their range
    lowering, since the last implementation of a signature is the one chosen.
    """
    context = cuda_target.target_context
    context.refresh()
    context.install_registry(RANGES)


def numba_operator(
    function: Callable[..., None],
    argument_types: tuple[object, ...],
    signature: Signature,
    block: tuple[int, int, int],
    target: Target,
    *,
    grid: tuple[int, int, int] = GRID_LIMIT,
    shared: Affine = NO_SHARED,
) -> Operator:
    """The operator of the Python function `function`, compiled as a kernel for
    `argument_types`, the Numba types of its arguments, without debug code, launched
    in blocks of `block` threads and grids of at most `grid` blocks, any grid of the
    GPU unless the claim states less."""
    validate_grid(grid)
    validate_shared(signature.sizes, shared)
    cc = (target.major, target.minor)
    args, _ = sigutils.normalize_signature(argument_types)
    with global_compiler_lock:
        install_ranges()
        result = compile_cuda(
            function,
            types.void,
            args,
            debug=False,
            lineinfo=False,
            fastmath=False,
            nvvm_options={"fastmath": False, "opt": 1},
            cc=cc,
            abi="numba",
        )
        library = result.library
        kernel = result.fndesc.llvm_func_name
        kernel_fixup(library.get_function(kernel), False)
        texts = library.get_asm_strs(cc=cc)
    if len(texts) != 1:
        raise ValueError(f"{kernel}: numba-cuda compiled {len(texts)} PTX modules, not one")
    return Operator(texts[0], kernel, signature, Convention.NUMBA, block, grid, shared)
