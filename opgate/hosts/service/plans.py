"""What a deployment supplies to the service: plan schemas and query nodes.

The operators and query operations belong to the deployment, which passes a
closed set of each when it starts the service. The shipped owner backend uses
RAPIDS (RMM and cuDF); another engine also needs allocation and launch
interception, beyond implementing these interfaces.

- A `PlanSchema` turns a registration's PTX and plan fields into a `Plan`:
  the binding under which the gate checks the PTX, which the server derives
  from the schema, never from the client, and how a `run_operator` request's
  fields become one launch.
- A `QueryNode` is a trusted engine operation that a request names by its
  `op`, such as a cuDF query over a session's buffers. It runs on the owner
  thread with a `NodeContext`, inside a boundary that releases cuDF query
  caches and fails the service if the node leaves a buffer live. The node
  releases the temporary allocations that it makes. A node launches a
  session's operator through `opgate.hosts.execute` with the context's
  resource, which gives a module with the guard pass's checks its status
  buffer and raises `opgate.hosts.CheckFailedError` when a check fails.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from opgate.binding import Binding
from opgate.catalog import BufferId, TenantId
from opgate.executor import Executor, Module
from opgate.hosts.service.protocol import Fields

if TYPE_CHECKING:
    from opgate.hosts.rapids import CatalogResource

__all__ = ["LaunchRequest", "NodeContext", "Plan", "PlanSchema", "QueryNode"]


@dataclass(frozen=True)
class LaunchRequest:
    """One launch of a registered operator, before its handles are resolved."""

    grid: tuple[int, int, int]
    buffers: tuple[str, ...]  # a buffer handle for each buffer parameter of the binding
    sizes: tuple[int, ...]  # a value for each size symbol of the binding
    scalars: tuple[int, ...]  # a value for each bounded parameter of the binding
    guarded: bool  # scan raw index inputs under the binding's proved index contract


class Plan(Protocol):
    """A registered operator's binding, and how a request runs it."""

    @property
    def kind(self) -> str: ...

    @property
    def binding(self) -> Binding: ...

    def launch(self, request: Fields) -> LaunchRequest:
        """The launch that a `run_operator` request's fields describe; a
        `RequestError` if they describe none."""
        ...


class PlanSchema(Protocol):
    """A closed kind of operator that clients may register."""

    @property
    def kind(self) -> str: ...

    def plan(self, text: str, fields: Fields) -> Plan:
        """The plan of `text` under this schema and the registration's plan
        fields; a `RequestError` if they do not fit."""
        ...


class NodeContext(Protocol):
    """What a query node may use: the session's buffers and operators, through
    checks, and the owner thread's executor and RMM resource."""

    @property
    def executor(self) -> Executor: ...

    @property
    def resource(self) -> "CatalogResource": ...

    @property
    def tenant(self) -> TenantId: ...

    def buffer(self, request: Fields, name: str) -> BufferId:
        """The buffer that the handle in field `name` names, which the session may read."""
        ...

    def operator(self, request: Fields, name: str, kind: str) -> Module:
        """The session's operator that the handle in field `name` names, registered
        under the schema `kind`: the module of the text as submitted, or of the
        guard pass's text, whose launches go through `opgate.hosts.execute`."""
        ...

    def launched(self, module: Module) -> None:
        """Count a launch of `module` in its registration's record."""
        ...


class QueryNode(Protocol):
    """A trusted engine operation that requests name by `op`."""

    @property
    def op(self) -> str: ...

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]: ...
