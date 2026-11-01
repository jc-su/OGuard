"""Engines the executor is installed into.

`rapids.CatalogResource` is the allocation wrapper of the RAPIDS engines:
an RMM memory resource whose every region the catalog checks (design
section 6.2), and `rapids.BufferView` shows a catalog buffer to a trusted
engine operator. `cudf.install_cudf` routes the kernel that cuDF compiles
for `Series.apply` and `DataFrame.apply` through the gate and the execute
path (design section 8, phase 2). Sirius follows only after the paper's
milestones are done (design section 11).

`register` registers a text with the executor as written, or, if the gate
refuses an access that it does not prove in bounds or aligned, the
kernel's barrier participation, or a loop that it does not prove ends, the
text that the guard pass (`opgate.frontends.guard`) checks and limits; the
cuDF hook and the PyTorch engine
register through it, and the service applies the same pass on its
verification threads. `execute` and `launch` run a step or one launch of
registered modules: when the pass added checks, a module's last buffer
parameter is a status buffer, which they allocate for the session's tenant,
zero in the launch's step, read after the step completes, and free; a
failed check or a loop that reached its iteration limit raises
`CheckFailedError` naming the kernel.

Imports `opgate.binding`, `opgate.catalog`, `opgate.gate`, `opgate.frontends`, and
`opgate.executor`, and each module its engine.
"""

from dataclasses import dataclass, replace
from typing import Protocol, assert_never

from opgate.binding import Binding, ElementType
from opgate.catalog import Action, BufferId, Fill, Launch, OperatorId, Step
from opgate.executor import CheckFailedError, Executor, Module, RefusedError
from opgate.frontends.guard import (
    STATUS_BYTES,
    STATUS_DECLARED,
    Guarded,
    applies,
    declares_status,
    guard,
    reports,
)
from opgate.gate import Accepted, Refused, VerificationBudget, verify, zeroed

__all__ = ["Allocator", "CheckFailedError", "Registered", "execute", "launch", "register"]

_WORD = ElementType(STATUS_BYTES, False)  # the status buffer's one element


class Allocator(Protocol):
    """Buffers of the session's tenant: the executor's own (`Executor.allocate`),
    or those of a host's allocator (`rapids.CatalogResource.allocate`)."""

    def allocate(self, nbytes: int) -> BufferId: ...

    def free(self, buffer: BufferId) -> None: ...


@dataclass(frozen=True)
class Registered:
    """A text and binding as submitted to `register`, the guard pass's result when
    the gate refused them for a reason that the pass addresses, and the module
    loaded: of the pass's text and binding, or else of the text as submitted."""

    text: str
    binding: Binding
    guarded: Guarded | None
    module: Module


def register(executor: Executor, text: str, binding: Binding) -> Registered:
    """`executor.register` of `text` as written, or, if the gate refuses an access
    that it does not prove in bounds or aligned, the kernel's barrier
    participation, or a loop that it does not prove ends, of the guarded text
    and binding; a refusal of the guarded
    text is the gate's reason for it. All the gate's checks of one registration
    share one budget of `REGISTRATION_SECONDS`, as the service's do
    (`VerificationBudget`). A text that itself declares the guard pass's status
    parameter is refused, since only the pass adds it."""
    if declares_status(text):
        raise RefusedError(STATUS_DECLARED)
    try:
        loaded = zeroed(text, binding)
    except ValueError as error:
        raise RefusedError(
            f"{binding.kernel}: the zeroing prologue cannot be placed: {error}"
        ) from error
    check = VerificationBudget(checker=verify)
    match check(loaded, binding):
        case Accepted():
            return Registered(text, binding, None, executor.register(text, binding))
        case Refused(reason=reason) if not applies(reason):
            raise RefusedError(reason)
        case Refused():
            pass
        case _ as never:
            assert_never(never)
    guarded = guard(text, binding, check)
    if isinstance(guarded.decision, Refused):
        raise RefusedError(guarded.decision.reason)
    return Registered(text, binding, guarded, executor.register(guarded.text, guarded.binding))


def _reporting(executor: Executor, operator: OperatorId) -> Binding | None:
    """The operator's binding when it ends with the guard pass's status buffer
    (`opgate.frontends.guard.reports`), else None; an operator that the catalog
    does not record is the catalog's to refuse."""
    binding = executor.catalog.binding(operator)
    return binding if binding is not None and reports(binding) else None


def execute(executor: Executor, allocator: Allocator, step: Step) -> tuple[bool, ...]:
    """`Executor.execute` of `step`; whether each kernel ran, which a guarded
    launch's index checks decide.

    Each launch of the step names a buffer for each buffer parameter of its
    kernel as submitted. A launch of a module whose binding ends with the guard
    pass's status buffer takes a buffer of `STATUS_BYTES` that `allocator` gives
    the session's tenant as its last buffer, which a fill just before the launch
    sets to zeros. After the step completes, a nonzero status raises
    `CheckFailedError` naming the kernel of the first such launch. The status
    buffers are freed in either case."""
    statuses: list[tuple[BufferId, str]] = []
    actions: list[Action] = []
    try:
        for action in step.actions:
            binding = _reporting(executor, action.operator) if isinstance(action, Launch) else None
            if isinstance(action, Launch) and binding is not None:
                status = allocator.allocate(STATUS_BYTES)
                statuses.append((status, binding.kernel))
                actions.append(Fill(status, _WORD, 0))
                actions.append(replace(action, buffers=(*action.buffers, status)))
            else:
                actions.append(action)
        ran = executor.execute(Step(step.tenant, tuple(actions)))
        failed = [kernel for status, kernel in statuses if any(executor.read(step.tenant, status))]
    finally:
        for status, _ in statuses:
            allocator.free(status)
    if failed:
        raise CheckFailedError(failed[0])
    return ran


def launch(executor: Executor, allocator: Allocator, launch: Launch) -> bool:
    """Launch `launch` as a step of the session's tenant; whether the kernel ran,
    which a guarded launch's index checks decide. `launch` names a buffer for
    each buffer parameter of the kernel as submitted: a module with the guard
    pass's status buffer launches as `execute` launches it, and any other
    through `Executor.launch`, which keeps its prepared step."""
    if _reporting(executor, launch.operator) is None:
        return executor.launch(launch)
    (ran,) = execute(executor, allocator, Step(executor.session_tenant, (launch,)))
    return ran
