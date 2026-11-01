"""RQ4: the cost of clearing a region that another tenant last owned.

    taskset -c <core> uv run python -m eval.timing eval.rq4_cost.clearing \\
        eval/records/clearing_<date>

An allocation step whose region another tenant last owned, which the
executor clears before the new owner can read it, against one whose region
the same tenant last owned, which it keeps, for regions of 1 MiB to 4 GiB.
A region reaches another tenant through the end of its owner's reservation,
which drops the owner's freed regions. The cleared measurement is a pair:
Bob's reservation and allocation of the region that Alice's reservation
held, and Alice's after his, each followed by its free and the end of its
reservation. The kept measurement is Alice's allocation and free in the
reservation whose region she freed last. The measurement protocol is
`eval.timing`'s.
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

from eval.timing import Environment, Timing, environment, host_time, pinned_core
from opgate.catalog import TenantId
from opgate.executor import Executor

MEBIBYTES = (1, 4, 16, 64, 256, 1024, 4096)
POOL = MEBIBYTES[-1] << 20  # the largest region, at offset 0
ALICE, BOB = TenantId(1), TenantId(2)


class Size(TypedDict):
    bytes: int
    cleared_pair: Timing
    kept: Timing


class Record(TypedDict):
    environment: Environment
    core: int
    sizes: list[Size]


def _size(executor: Executor, nbytes: int) -> Size:
    def allocate(tenant: TenantId, *, end: bool) -> None:
        """An allocation of the region at offset 0 and its free, in the tenant's
        reservation, which `end` ends afterwards."""
        with executor.tenant(tenant):
            if executor.catalog.reservation(tenant) is None:
                executor.reserve(nbytes)
            buffer = executor.allocate(nbytes)
            executor.synchronize()
            executor.free(buffer)
            if end:
                executor.unreserve()

    allocate(ALICE, end=True)
    cleared = host_time(lambda: (allocate(BOB, end=True), allocate(ALICE, end=True)))
    allocate(ALICE, end=False)
    kept = host_time(lambda: allocate(ALICE, end=False))
    with executor.tenant(ALICE):
        executor.unreserve()
    return {"bytes": nbytes, "cleared_pair": cleared, "kept": kept}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record: Record = {"environment": environment(), "core": pinned_core(), "sizes": []}
    with Executor(0, POOL) as executor:
        for mebibytes in MEBIBYTES:
            record["sizes"].append(_size(executor, mebibytes << 20))
            print(f"{mebibytes} MiB", flush=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
