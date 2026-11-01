"""A multi-tenant service around the OpGate executor, for a RAPIDS host.

Clients connect over a Unix socket and exchange length-prefixed JSON
(`protocol`). A connection authenticates as a principal, which the server
maps to a tenant; requests name buffers and operators only by opaque handles
(`sessions`). One owner thread holds the executor, its RMM resource, and the
cuDF hook (`owner`); the server's threads parse requests and check
registrations (`server`). The deployment supplies the operators that clients
may register (plan schemas) and the query operations they may run (`plans`).
The shipped owner backend requires RAPIDS; these interfaces do not provide
another engine's allocation or launch interception. A text that the gate
refuses for an access that it does not prove, or for barrier participation,
is registered with the guard pass's checks (`opgate.frontends.guard`), and a
launch whose check fails answers "check_failed" to its tenant. Registration
records keep the submitted PTX, the derived binding, and the pass's text,
binding, and checks (`audit`).
"""

from opgate.hosts.service.audit import Registration, binding_record, parse_binding
from opgate.hosts.service.plans import LaunchRequest, NodeContext, Plan, PlanSchema, QueryNode
from opgate.hosts.service.protocol import Fields, RequestError
from opgate.hosts.service.schemas import OperatorSchema
from opgate.hosts.service.server import Client, Service
from opgate.hosts.service.sessions import Limits, Principal

__all__ = [
    "Client",
    "Fields",
    "LaunchRequest",
    "Limits",
    "NodeContext",
    "OperatorSchema",
    "Plan",
    "PlanSchema",
    "Principal",
    "QueryNode",
    "Registration",
    "RequestError",
    "Service",
    "binding_record",
    "parse_binding",
]
