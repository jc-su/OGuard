"""RQ4, cost: registration, the step check, clearing, kernel time, a query's overhead,
tenants on one GPU, and a query's cold registration."""

import heapq
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple, assert_never

from eval.checkpoint import snapshot_inputs
from eval.corpus.adapter import directory_of
from eval.export.corpus import REGISTRATION, Kernel, Key
from eval.export.records import (
    RECORDS,
    SCALE_NAMES,
    Numbers,
    across,
    field,
    items,
    load,
    number,
    obj,
    runs,
    table,
    text,
    values,
    write,
)
from eval.export.rq2 import GUARDED, guarded_rows
from eval.export.rq3 import TORCH_QUERIES
from eval.rq4_cost.consolidation import (
    ALIGNMENT,
    EXPORT_GRANULARITY,
    SHARED_COLUMNS,
    TABLE_BYTES_PER_ROW,
)
from eval.rq4_cost.consolidation import _source_texts as consolidation_sources
from eval.rq4_cost.fallback import BUILTINS
from eval.rq4_cost.generation import COMPILED
from eval.rq4_cost.generation import KIND as GENERATION_KIND
from eval.rq4_cost.generation import source_files as generation_sources
from eval.rq4_cost.guard_cost import source_files as guard_cost_sources
from eval.rq4_cost.mix import BY_OP, MIXES, host_of, runnable_templates, serves_refused
from eval.rq4_cost.mix import CONFIGURATIONS as MIXED_CONFIGURATIONS
from eval.rq4_cost.mix import Configuration as MixedConfiguration
from eval.rq4_cost.mix import kernels_of as mixed_kernels
from eval.rq4_cost.mixed import runs_of as mixed_runs_of
from eval.rq4_cost.mixed import sources as mixed_sources
from eval.rq4_cost.registration import inputs
from eval.service_plans import VERIFICATION_WORKERS
from eval.timing import REPEATS, RUNS, WARMUP_SECONDS
from eval.tpch.torch_engine import MODELS, QUERIES
from opgate.gate import REGISTRATION_SECONDS
from opgate.hosts.service import Limits

STEP_CHECK = RECORDS / "step_check_2026-10-24"  # rq4_cost/step_check.py
CLEARING = RECORDS / "clearing_2026-10-24"  # rq4_cost/clearing.py
KERNEL_TIME = RECORDS / "kernel_time_2026-10-24"  # rq4_cost/kernel_time.py
QUERY_OVERHEAD = RECORDS / "query_overhead_2026-10-24"  # rq4_cost/query_overhead.py
CONSOLIDATION = RECORDS / "consolidation_2026-10-28.json"  # rq4_cost/consolidation.py
MIXED = RECORDS / "mixed_2026-10-24b.json"  # rq4_cost/mixed.py
GUARD_COST = RECORDS / "guard_cost_2026-10-24b"  # rq4_cost/guard_cost.py
GUARD_COST_RUNS = 2  # the Makefile's GUARD_COST_RUNS
FAULT_ISOLATION = RECORDS / "fault_isolation_2026-10-24b.json"  # rq4_cost/fault_isolation.py
GENERATION = RECORDS / "generation_2026-10-24.json"  # rq4_cost/generation.py
COLD_QUERY = RECORDS / "cold_query_2026-10-24"  # rq4_cost/cold_query.py
# the cost experiment's texts of each kernel, against the kernel as written
CHECKED_VERSIONS = ("opgate", "every", "guardian")
# the revenue pipeline's phases in four groups: the catalog's allocation and free
# steps, the filter's step, the sum's step, and the host's transfers
PHASE_GROUPS = {
    "catalog": ("allocate", "allocate scratch", "free"),
    "filter": ("filter step",),
    "sum": ("sum step",),
    "host": ("read count", "write segments", "read total"),
}


def export_registration(kernels: list[Kernel]) -> Numbers:
    """The frozen checker's time for each corpus kernel, checked as written by
    the registration record's concurrent workers under the service's budget of
    one registration; a check past the budget is a refusal."""
    seconds = sorted(k.seconds for k in kernels)
    workers = field(load(REGISTRATION), "provenance", "workers")
    return {
        "valRegistrationWorkers": f"{int(number(workers, 'workers'))}",
        "valRegistrationKernels": f"{len(kernels):,}",
        "valRegistrationMedianSeconds": f"{statistics.median(seconds):.1f}",
        "valRegistrationUnderMinute": f"{sum(1 for s in seconds if s < 60):,}",
        "valRegistrationTimeouts": f"{sum(1 for k in kernels if k.final == 'timeout'):,}",
        "valRegistrationMaxSeconds": f"{max(seconds):.0f}",
        "valRegistrationAcceptedMaxSeconds": (
            f"{max(k.seconds for k in kernels if k.final == 'accepted'):.0f}"
        ),
    }


def export_step_check() -> Numbers:
    records = runs(STEP_CHECK)
    points = items(records[0]["points"], "points")
    rows = [
        (
            int(number(p["buffers"], "buffers")),
            *across(records, "points", i, "check_step").columns(),
        )
        for i, p in enumerate(points)
    ]
    write("step_check.csv", ("buffers", "check", "check_lo", "check_hi"), rows)
    fewest, most = (
        across(records, "points", 0, "check_step"),
        across(records, "points", len(points) - 1, "check_step"),
    )
    return {
        "valStepCheckFewBuffers": f"{rows[0][0]}",
        "valStepCheckFewMs": f"{fewest.median / 1e3:.2f}",
        "valStepCheckManyBuffers": f"{rows[-1][0]:,}",
        "valStepCheckManyMs": f"{most.median / 1e3:.2f}",
    }


def export_clearing() -> Numbers:
    records = runs(CLEARING)
    sizes = items(records[0]["sizes"], "sizes")
    rows = []
    for i, size in enumerate(sizes):
        cleared = across(records, "sizes", i, "cleared_pair")
        kept = across(records, "sizes", i, "kept")
        rows.append((number(size["bytes"], "bytes") / 2**20, *cleared.columns(), *kept.columns()))
    write(
        "clearing.csv",
        ("mebibytes", "cleared", "cleared_lo", "cleared_hi", "kept", "kept_lo", "kept_hi"),
        rows,
    )
    smallest = across(records, "sizes", 0, "cleared_pair").median
    largest = across(records, "sizes", len(sizes) - 1, "cleared_pair").median
    return {
        "valClearSmallUs": f"{smallest:.0f}",
        "valClearLargeMs": f"{largest / 1e3:.1f}",
    }


def export_kernel_time() -> Numbers:
    records = runs(KERNEL_TIME)
    operators = items(records[0]["operators"], "operators")
    rows, changes = [], []
    for i, operator in enumerate(operators):
        level1 = across(records, "operators", i, "level1")
        level3 = across(records, "operators", i, "level3")
        changes.append(100 * (level1.median / level3.median - 1))
        rows.append((text(operator["task"], "task"), *level1.columns(), *level3.columns()))
    write(
        "kernel_time.csv",
        ("task", "level1", "level1_lo", "level1_hi", "level3", "level3_lo", "level3_hi"),
        rows,
    )
    return {
        "valNvvmChangeLow": f"{min(changes):.1f}",
        "valNvvmChangeHigh": f"{max(changes):.1f}",
    }


def export_query_overhead() -> Numbers:
    """The queries at each scale against direct launches and cuDF, and the
    revenue pipeline's phases."""
    records = runs(QUERY_OVERHEAD)
    scales = items(records[0]["scales"], "scales")
    columns = {
        "q6_opgate": ("pipelines", "q6_opgate"),
        "q6_direct": ("pipelines", "q6_direct"),
        "q6_cudf": ("cudf", "q6"),
        "revenue_static": ("pipelines", "revenue_static"),
        "revenue_guarded": ("pipelines", "revenue_guarded"),
        "revenue_direct": ("pipelines", "revenue_direct"),
        "revenue_cudf": ("cudf", "revenue"),
    }
    rows, numbers = [], {}
    for s, scale in enumerate(scales):
        factor = number(scale["scale"], "scale")
        timings = {name: across(records, "scales", s, *path) for name, path in columns.items()}
        rows.append(
            (
                factor,
                int(number(scale["rows"], "rows")),
                *(c for t in timings.values() for c in t.columns()),
            )
        )
        name = SCALE_NAMES[factor]
        numbers[f"valQsixAdd{name}"] = (
            f"{(timings['q6_opgate'].median - timings['q6_direct'].median) / 1e3:.2f}"
        )
        numbers[f"valRevenueAdd{name}"] = (
            f"{(timings['revenue_static'].median - timings['revenue_direct'].median) / 1e3:.1f}"
        )
        numbers[f"valCudfQsix{name}"] = f"{timings['q6_cudf'].median / 1e3:.1f}"
        numbers[f"valCudfRevenue{name}"] = f"{timings['revenue_cudf'].median / 1e3:.1f}"
    write(
        "scaling.csv",
        ("scale", "rows", *(f"{c}{x}" for c in columns for x in ("", "_lo", "_hi"))),
        rows,
    )
    phase_rows = []
    for s, scale in enumerate(scales):
        factor = number(scale["scale"], "scale")
        for path in ("static", "guarded"):
            grouped = [
                sum(across(records, "scales", s, "phases", path, p).median for p in phases)
                for phases in PHASE_GROUPS.values()
            ]
            checks = [
                across(records, "scales", s, "phases", path, p).median
                for p in ("check filter step", "check sum step")
            ]
            phase_rows.append(
                (
                    f"SF {factor:g} {path}",
                    *(f"{v:.1f}" for v in grouped),
                    *(f"{v:.1f}" for v in checks),
                )
            )
    write("breakdown.csv", ("case", *PHASE_GROUPS, "check_filter", "check_sum"), phase_rows)
    cudf = [
        across(records, "scales", s, "cudf", query).median
        for s in range(len(scales))
        for query in ("q6", "revenue")
    ]
    numbers["valCudfLowMs"] = f"{min(cudf) / 1e3:.1f}"
    numbers["valCudfHighMs"] = f"{max(cudf) / 1e3:.1f}"
    factors = [number(scale["scale"], "scale") for scale in scales]
    if 10.0 not in factors:
        raise ValueError(f"{QUERY_OVERHEAD}: no scale factor 10")
    ten = factors.index(10.0)
    numbers["valSegmentUploadTenMs"] = (
        f"{across(records, 'scales', ten, 'phases', 'static', 'write segments').median / 1e3:.1f}"
    )
    return numbers


Cell = tuple[str, float, int]  # a configuration, a scale factor, and tenants
CONFIGURATIONS = (
    "service",
    "service_per_tenant",
    "process",
    "mps",
    "mps_shared",
    "mps_static",
    "guardian",
)
# the word that ends the name of a macro of each tenant count
TENANT_WORDS = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 12: "Twelve", 16: "Sixteen"}
MEMORY_TENANTS = (8, 16)  # the tenant counts of the memory macros


def _cells(record: dict[str, object]) -> dict[Cell, list[dict[str, object]]]:
    """The runs of the consolidation record by cell, each cell's repeats in order."""
    repeats = int(number(record["repeats"], "repeats"))
    cells: dict[Cell, list[dict[str, object]]] = {}
    for run in sorted(
        items(record["runs"], "runs"), key=lambda r: int(number(r["repeat"], "repeat"))
    ):
        key = (
            text(run["configuration"], "configuration"),
            number(run["scale"], "scale"),
            int(number(run["tenants"], "tenants")),
        )
        cells.setdefault(key, []).append(run)
    if any(len(runs) != repeats for runs in cells.values()):
        raise ValueError(f"{CONSOLIDATION}: a cell without {repeats} repeats")
    return cells


def _check_run(run: dict[str, object], duration: float, ptx: str, where: str) -> None:
    """A run's samples agree with its counts, and each service it ran served the
    workload's PTX, loaded it once, and shut down cleanly."""
    samples = values(run["latency_us_samples"], "latency_us_samples")
    queries = int(number(run["queries"], "queries"))
    if (
        queries != len(samples)
        or int(number(field(run, "latency_us", "count"), "count")) != queries
    ):
        raise ValueError(f"{where}: inconsistent samples")
    if abs(number(run["queries_per_second"], "qps") - queries / duration) > 1e-9:
        raise ValueError(f"{where}: inconsistent throughput")
    for service in items(run["services"], "services"):
        if text(service["ptx"], "service.ptx") != ptx:
            raise ValueError(f"{where}: the service's PTX differs")
        audit = items(service["audit"], "service.audit")
        loads = sum(int(number(row["module_load_count"], "module_load_count")) for row in audit)
        if loads != 1 or any(text(row["ptx"], "audit.ptx") != ptx for row in audit):
            raise ValueError(f"{where}: an invalid registration audit")
        final = obj(service["telemetry_after_cleanup"], "telemetry_after_cleanup")
        counters = (
            "live_buffers",
            "service_buffer_handles",
            "service_operator_handles",
            "input_errors",
            "fatal_errors",
        )
        if any(number(final[name], name) != 0 for name in counters):
            raise ValueError(f"{where}: an unclean service shutdown")


def _check_guardian(run: dict[str, object], tenants: int, where: str) -> None:
    """A guardian cell carries no service and one raw partition layout: a
    power-of-two partition per tenant, each base a multiple of it and one
    partition apart, inside a pool that holds them all, and a fenced kernel."""
    if values(run["services"], "services"):
        raise ValueError(f"{where}: a guardian cell recorded a service")
    layout = obj(run["guardian"], f"{where}.guardian")
    size = int(number(layout["partition_bytes"], "partition_bytes"))
    if size <= 0 or size & (size - 1):
        raise ValueError(f"{where}: the partition is not a positive power of two")
    pool = int(number(layout["pool_bytes"], "pool_bytes"))
    address = int(number(layout["pool_address"], "pool_address"))
    first = int(number(layout["first_partition"], "first_partition"))
    if pool != size * (tenants + 1) or first % size != 0 or not 0 <= first - address < size:
        raise ValueError(f"{where}: the pool does not hold {tenants} aligned partitions")
    rows = items(layout["tenants"], f"{where}.guardian.tenants")
    if len(rows) != tenants:
        raise ValueError(f"{where}: {len(rows)} partitions for {tenants} tenants")
    for index, tenant in enumerate(rows):
        base = int(number(tenant["base"], "base"))
        if base != first + index * size or int(number(tenant["mask"], "mask")) != size - 1:
            raise ValueError(f"{where}: partition {index} has the wrong base or mask")
    if int(number(layout["fenced_accesses"], "fenced_accesses")) <= 0:
        raise ValueError(f"{where}: the guardian kernel fences no access")
    if not text(layout["ptx"], f"{where}.guardian.ptx"):
        raise ValueError(f"{where}: the guardian cell kept no fenced PTX")


def _check_shared(run: dict[str, object], tenants: int, where: str) -> None:
    """A shared-table cell runs under default MPS, carries no service or
    guardian layout, and keeps its owner's raw evidence: the five exported
    columns hold the table's bytes, each in an allocation of whole 2 MiB units,
    and the owner compared all of them after the tenants ran; each tenant
    mapped the owner's handles at aligned addresses and ran the unfenced kernel."""
    if values(run["services"], "services") or run["guardian"] is not None:
        raise ValueError(f"{where}: a shared-table cell recorded a service or a guardian layout")
    if obj(run["mps"], f"{where}.mps").get("mode") != "default":
        raise ValueError(f"{where}: a shared-table cell outside default MPS")
    shared = obj(run["shared"], f"{where}.shared")
    columns = obj(shared["columns"], f"{where}.shared.columns")
    if set(columns) != set(SHARED_COLUMNS):
        raise ValueError(f"{where}: the owner exported {sorted(columns)}")
    owned = {name: obj(columns[name], f"{where}.shared.{name}") for name in SHARED_COLUMNS}
    table = 0
    for name, column in owned.items():
        nbytes = int(number(column["bytes"], f"{name}.bytes"))
        allocated = int(number(column["allocated"], f"{name}.allocated"))
        if allocated <= 0 or allocated % EXPORT_GRANULARITY or allocated < nbytes:
            raise ValueError(f"{where}: {name} is not exported in whole 2 MiB units")
        table += nbytes
    if table != int(number(run["rows"], "rows")) * TABLE_BYTES_PER_ROW:
        raise ValueError(f"{where}: the exported columns do not hold the table")
    if int(number(shared["compared_bytes"], "compared_bytes")) != table:
        raise ValueError(f"{where}: the owner did not compare the whole table")
    rows = items(run["per_tenant"], f"{where}.per_tenant")
    if len(rows) != tenants:
        raise ValueError(f"{where}: {len(rows)} tenants' mappings for {tenants} tenants")
    for tenant in rows:
        request = obj(tenant["request"], f"{where}.request")
        if request.get("fenced") is not False:
            raise ValueError(f"{where}: a shared-table tenant ran a fenced kernel")
        mapped = obj(request["mapped"], f"{where}.request.mapped")
        if set(mapped) != set(SHARED_COLUMNS):
            raise ValueError(f"{where}: a tenant mapped {sorted(mapped)}")
        for name in SHARED_COLUMNS:
            mapping = obj(mapped[name], f"{where}.mapped.{name}")
            handle = text(owned[name]["handle"], f"{name}.handle")
            if text(mapping["handle"], f"{name}.handle") != handle:
                raise ValueError(f"{where}: a tenant mapped another handle of {name}")
            if int(number(mapping["address"], f"{name}.address")) % ALIGNMENT:
                raise ValueError(f"{where}: a tenant mapped {name} at an unaligned address")


def _spread(runs: list[dict[str, object]], *path: str) -> tuple[float, float, float]:
    """The median, lowest, and highest of a value across a cell's repeats."""
    found = [number(field(run, *path), ".".join(path)) for run in runs]
    return statistics.median(found), min(found), max(found)


class Partition(NamedTuple):
    """Static MPS's SM partitions at one tenant count: the GPU's SMs, the SMs of
    one chunk, those of each tenant's partition, and those of no partition."""

    sms: int
    chunk: int
    each: int
    idle: int


def _partition(runs: list[dict[str, object]], tenants: int, where: str) -> Partition:
    """The partitions that every run of a static MPS cell recorded: one equal
    partition of whole chunks per tenant, with the chunks left over idle."""
    found = set()
    for run in runs:
        mps = obj(run["mps"], f"{where}.mps")
        if mps["mode"] != "static_sm_partitioning":
            raise ValueError(f"{where}: MPS without static partitions")
        chunks, sms, each, left = (
            int(number(mps[name], name))
            for name in ("total_chunks", "total_sms", "chunks_per_partition", "unassigned_chunks")
        )
        if sms % chunks or each != chunks // tenants or left != chunks - tenants * each:
            raise ValueError(f"{where}: partitions of other sizes than {tenants} equal shares")
        if int(number(mps["sms_per_partition"], "sms_per_partition")) != sms // chunks * each:
            raise ValueError(f"{where}: a partition's SMs differ from its chunks")
        found.add(Partition(sms, sms // chunks, sms // chunks * each, sms // chunks * left))
    if len(found) != 1:
        raise ValueError(f"{where}: the repeats recorded different partitions")
    return found.pop()


def export_consolidation() -> Numbers:
    """Tenants through the service against a service or a process per tenant,
    with and without MPS, under MPS with one table shared through CUDA IPC, and
    with Guardian's partitions: the median of the repeats, with their range.
    At the most tenants of each scale factor, the shared table must take less
    device memory than MPS with private tables. At scale factor 1 the macros
    give the table's MiB (`valConsTableMib`), the shared table's memory per
    added tenant (`valConsMpsSharedTenantMib`), and each
    configuration's device memory at 8 and 16 tenants (`valCons<Name>Mib<Count>`)
    and, for each configuration but the service, its memory divided by the
    service's (`valCons<Name>MibRatio<Count>`). At each scale factor (`Small`
    for 0.1) they give each configuration's throughput and p99 latency at the
    most tenants, its throughput and median latency at the fewest, and its
    highest throughput with its tenant count; and, at each tenant count, the
    SMs of each static MPS partition and those of none."""
    record = load(CONSOLIDATION)
    if record.get("complete") is not True:
        raise ValueError(f"{CONSOLIDATION}: not a completed record")
    if record["sources"] != consolidation_sources():
        raise ValueError(f"{CONSOLIDATION}: raw sources differ from the current implementation")
    configurations = tuple(
        text(v, "configurations") for v in values(record["configurations"], "configurations")
    )
    if configurations != CONFIGURATIONS:
        raise ValueError(f"{CONSOLIDATION}: unexpected configurations")
    scales = tuple(number(v, "scales") for v in values(record["scales"], "scales"))
    tenants = tuple(
        int(number(v, "tenants")) for v in values(record["tenant_counts"], "tenant_counts")
    )
    cells = _cells(record)
    if set(cells) != {(c, s, n) for c in configurations for s in scales for n in tenants}:
        raise ValueError(f"{CONSOLIDATION}: missing measurement cells")
    ptx = text(field(record, "workload", "ptx"), "workload.ptx")
    duration = number(record["duration_seconds"], "duration_seconds")
    for key, repeated in cells.items():
        match key[0]:
            case "service":
                services = 1
            case "service_per_tenant":
                services = key[2]
            case _:
                services = 0
        for run in repeated:
            if len(values(run["services"], "services")) != services:
                raise ValueError(f"{CONSOLIDATION}: {key} has not {services} services")
            _check_run(run, duration, ptx, f"{CONSOLIDATION} {key}")
            if key[0] == "guardian":
                _check_guardian(run, key[2], f"{CONSOLIDATION} {key}")
            if key[0] == "mps_shared":
                _check_shared(run, key[2], f"{CONSOLIDATION} {key}")
            elif run["shared"] is not None:
                raise ValueError(f"{CONSOLIDATION}: {key} recorded a shared table")
    most = max(tenants)
    numbers: Numbers = {
        "valConsTenants": f"{most}",
        "valConsRepeats": f"{int(number(record['repeats'], 'repeats'))}",
        "valConsSeconds": f"{duration:g}",
    }
    names = {
        "service": "Service",
        "service_per_tenant": "ServiceEach",
        "process": "Process",
        "mps": "Mps",
        "mps_shared": "MpsShared",
        "mps_static": "MpsStatic",
        "guardian": "Guardian",
    }
    fewest = min(tenants)
    if fewest not in TENANT_WORDS or any(n not in TENANT_WORDS for n in tenants):
        raise ValueError(f"{CONSOLIDATION}: a tenant count without a macro name")
    if fewest == most:
        raise ValueError(f"{CONSOLIDATION}: one tenant count gives no memory per tenant")
    for scale in scales:
        shared = _spread(cells[("mps_shared", scale, most)], "memory_mib")[0]
        private = _spread(cells[("mps", scale, most)], "memory_mib")[0]
        if shared >= private:
            raise ValueError(
                f"{CONSOLIDATION}: at scale factor {scale:g} and {most} tenants the shared "
                f"table takes {shared:g} MiB, not less than MPS's {private:g} MiB"
            )
    first, last = (_spread(cells[("mps_shared", 1.0, n)], "memory_mib")[0] for n in (fewest, most))
    table_rows = {
        int(number(run["rows"], "rows"))
        for (_, scale, _), repeated in cells.items()
        if scale == 1.0
        for run in repeated
    }
    if len(table_rows) != 1:
        raise ValueError(f"{CONSOLIDATION}: the runs at scale factor 1 have different tables")
    numbers |= {
        "valConsMpsSharedTenantMib": f"{(last - first) / (most - fewest):,.0f}",
        "valConsTableMib": f"{table_rows.pop() * TABLE_BYTES_PER_ROW / 2**20:.1f}",
    }
    for configuration, name in names.items():
        for scale, size in ((1.0, ""), (0.1, "Small")):
            qps = _spread(cells[(configuration, scale, most)], "queries_per_second")[0]
            tail = _spread(cells[(configuration, scale, most)], "latency_us", "p99")[0]
            alone = _spread(cells[(configuration, scale, fewest)], "queries_per_second")[0]
            middle = _spread(cells[(configuration, scale, fewest)], "latency_us", "p50")[0]
            peak = max(
                tenants,
                key=lambda n: _spread(cells[(configuration, scale, n)], "queries_per_second"),
            )
            highest = _spread(cells[(configuration, scale, peak)], "queries_per_second")[0]
            word = TENANT_WORDS[fewest]
            numbers |= {
                f"valCons{name}{size}Qps": f"{qps:,.0f}",
                f"valCons{name}{size}TailUs": f"{tail:,.0f}",
                f"valCons{name}{size}Qps{word}": f"{alone:,.0f}",
                f"valCons{name}{size}MedianUs{word}": f"{middle:,.0f}",
                f"valCons{name}{size}PeakQps": f"{highest:,.0f}",
                f"valCons{name}{size}PeakTenants": f"{peak}",
            }
    partitions = {
        (count, _partition(cells[("mps_static", scale, count)], count, f"{CONSOLIDATION} {count}"))
        for count in tenants
        for scale in scales
    }
    gpus = {(p.sms, p.chunk) for _, p in partitions}
    if len(partitions) != len(tenants) or len(gpus) != 1:
        raise ValueError(f"{CONSOLIDATION}: static MPS partitions differ between cells")
    ((sms, chunk),) = gpus
    numbers |= {"valConsSms": f"{sms}", "valConsChunkSms": f"{chunk}"}
    for count, partition in partitions:
        numbers[f"valConsStaticSms{TENANT_WORDS[count]}"] = f"{partition.each}"
        numbers[f"valConsStaticIdleSms{TENANT_WORDS[count]}"] = f"{partition.idle}"
    for count in MEMORY_TENANTS:
        word = TENANT_WORDS[count]
        if count not in tenants:
            raise ValueError(f"{CONSOLIDATION}: the record lacks cells of {count} tenants")
        memory = {
            configuration: _spread(cells[(configuration, 1.0, count)], "memory_mib")[0]
            for configuration in configurations
        }
        for configuration, name in names.items():
            numbers[f"valCons{name}Mib{word}"] = f"{memory[configuration]:,.0f}"
            if configuration != "service":
                ratio = memory[configuration] / memory["service"]
                numbers[f"valCons{name}MibRatio{word}"] = f"{ratio:.1f}"
    files = {0.1: "small", 1.0: "one"}
    for scale in scales:
        rows = []
        for n in tenants:
            row: list[object] = [n]
            for c in configurations:
                qps = _spread(cells[(c, scale, n)], "queries_per_second")
                p99 = _spread(cells[(c, scale, n)], "latency_us", "p99")
                memory = _spread(cells[(c, scale, n)], "memory_mib")
                row += [f"{v:.1f}" for v in (*qps, *p99)] + [f"{memory[0]:.0f}"]
            rows.append(row)
        write(
            f"consolidation_{files[scale]}.csv",
            (
                "tenants",
                *(
                    f"{m}_{c}{suffix}"
                    for c in configurations
                    for m, suffixes in (
                        ("qps", ("", "_lo", "_hi")),
                        ("p99_us", ("", "_lo", "_hi")),
                        ("memory_mib", ("",)),
                    )
                    for suffix in suffixes
                ),
            ),
            rows,
        )
    return numbers


# the word that names each configuration of the mixed record in a macro
MIXED_NAMES: dict[MixedConfiguration, str] = {
    "service": "Service",
    "service_refusals": "Refusals",
    "service_fallback": "Fallback",
    "mps": "Mps",
    "mps_shared": "MpsShared",
}
# the word that names each template in a macro
TEMPLATE_WORDS = {
    "q6": "Qsix",
    "prefix_sum": "PrefixSum",
    "like_sum": "LikeSum",
    "semi_join_sum": "SemiJoinSum",
    "join_count": "JoinCount",
    "group_by": "GroupBy",
    "string_equal": "StringEqual",
    "bloom_probe": "BloomProbe",
}


class MixedExpected(NamedTuple):
    """What the recorded decisions give a configuration of the mixed record: its
    refused kernels, the templates it skips, and the built-in operators that
    serve each template's steps."""

    refused: list[str]
    skipped: list[str]
    builtins: dict[str, list[str]]


# the fields of a mixed run that the macros and CSV files read
MIXED_SUMMARY = (
    "queries_per_second",
    "latency_us",
    "latency_us_by_template",
    "memory_mib",
    "builtin_share",
)


def _mixed_cells(
    record: Path,
    listed: list[dict[str, object]],
    expected: dict[MixedConfiguration, MixedExpected],
    duration: float,
) -> dict[tuple[MixedConfiguration, int], list[dict[str, object]]]:
    """Each run that the mixed record lists, read from its file and checked
    (`_check_mixed_run`), as the fields that `MIXED_SUMMARY` names, by
    configuration and tenants, each cell's repeats in order."""
    cells: dict[tuple[MixedConfiguration, int], list[dict[str, object]]] = {}
    for entry in sorted(listed, key=lambda r: int(number(r["repeat"], "repeat"))):
        name = text(entry["configuration"], "configuration")
        found = [c for c in MIXED_CONFIGURATIONS if c == name]
        if not found:
            raise ValueError(f"{record}: a run of the configuration {name!r}")
        (configuration,) = found
        tenants = int(number(entry["tenants"], "tenants"))
        path = mixed_runs_of(record) / text(entry["file"], "file")
        run = load(path)
        if any(run[key] != entry[key] for key in ("configuration", "tenants", "repeat")):
            raise ValueError(f"{path}: the run is not the cell that {record} lists")
        _check_mixed_run(run, configuration, expected[configuration], duration, str(path))
        summary = {key: run[key] for key in MIXED_SUMMARY}
        cells.setdefault((configuration, tenants), []).append(summary)
    return cells


def _mixed_expected(record: dict[str, object]) -> dict[MixedConfiguration, MixedExpected]:
    decisions = obj(record["recorded_decisions"], f"{MIXED}.recorded_decisions")
    refused = {
        kind
        for kind, decision in decisions.items()
        if obj(decision, f"{MIXED}.recorded_decisions.{kind}")["status"] == "refused"
    }
    expected: dict[MixedConfiguration, MixedExpected] = {}
    for configuration in MIXED_CONFIGURATIONS:
        runnable, skipped = runnable_templates(configuration, lambda k: k.kind not in refused)
        builtins = {
            t.name: [BUILTINS[k.task].op for k in t.kernels if k.kind in refused] for t in runnable
        }
        expected[configuration] = MixedExpected(
            [k.kind for k in mixed_kernels(configuration) if k.kind in refused],
            skipped,
            {name: ops for name, ops in builtins.items() if ops},
        )
    return expected


def _check_mixed_run(
    run: dict[str, object],
    configuration: MixedConfiguration,
    expected: MixedExpected,
    duration: float,
    where: str,
) -> None:
    """A run's counts agree with its tenants' samples; the refused kernels' templates
    are skipped or served by the built-in operators as the recorded decisions
    give, and the measured time of the built-in operators is the share the run
    gives; a service run kept the refusal reason of each refused kernel and
    shut down cleanly, and an mps_shared run's owner compared every shared
    input."""
    tenants = items(run["per_tenant"], f"{where}.per_tenant")
    if len(tenants) != int(number(run["tenants"], "tenants")):
        raise ValueError(f"{where}: {len(tenants)} tenants' samples")
    queries = int(number(run["queries"], "queries"))
    samples = [values(t["latency_us_samples"], f"{where}.latency") for t in tenants]
    if queries != sum(len(s) for s in samples):
        raise ValueError(f"{where}: the queries are not the tenants' samples")
    if int(number(field(run, "latency_us", "count"), "count")) != queries:
        raise ValueError(f"{where}: the latency counts another number of queries")
    if abs(number(run["queries_per_second"], "qps") - queries / duration) > 1e-9:
        raise ValueError(f"{where}: inconsistent throughput")
    refused = [text(v, "refused_kernels") for v in values(run["refused_kernels"], "refused")]
    if refused != expected.refused:
        raise ValueError(f"{where}: the refused kernels are {refused}")
    skipped = [text(v, "refused_templates") for v in values(run["refused_templates"], "skip")]
    served = obj(run["served"], f"{where}.served")
    builtins = {
        name: [text(v, f"{where}.served") for v in values(steps, name) if v in BY_OP]
        for name, steps in served.items()
    }
    if skipped != expected.skipped or {n: b for n, b in builtins.items() if b} != (
        expected.builtins
    ):
        raise ValueError(f"{where}: the run skipped {skipped} and served {builtins}")
    templates = [text(v, "templates") for v in values(run["templates"], "templates")]
    builtin = total = 0.0
    for tenant in tenants:
        indexes = [int(number(v, "template")) for v in values(tenant["template"], "template")]
        steps = values(tenant["step_us_samples"], f"{where}.step_us_samples")
        latency = values(tenant["latency_us_samples"], f"{where}.latency_us_samples")
        for index, step, seconds in zip(indexes, steps, latency, strict=True):
            names = values(served[templates[index]], "served")
            times = [number(v, "step") for v in values(step, "step")]
            builtin += sum(t for t, n in zip(times, names, strict=True) if n in BY_OP)
            total += number(seconds, "latency")
    if not math.isclose(builtin / total, number(run["builtin_share"], "builtin_share")):
        raise ValueError(f"{where}: the built-in share differs from the step samples")
    match host_of(configuration):
        case "service":
            service = obj(run["service"], f"{where}.service")
            decisions = obj(service["decisions"], f"{where}.decisions")
            for kind in refused:
                decision = obj(decisions[kind], f"{where}.decisions.{kind}")
                if decision["status"] != "refused" or not text(decision["reason"], "reason"):
                    raise ValueError(f"{where}: {kind} has no recorded refusal reason")
            final = obj(service["telemetry_after_cleanup"], f"{where}.telemetry")
            for counter in ("live_buffers", "service_buffer_handles", "input_errors"):
                if number(final[counter], counter) != 0:
                    raise ValueError(f"{where}: an unclean service shutdown")
        case "mps":
            mps = obj(run["mps"], f"{where}.mps")
            if obj(mps["mps"], f"{where}.mps.mps").get("mode") != "default":
                raise ValueError(f"{where}: an MPS run outside default MPS")
            if configuration == "mps_shared":
                owner = obj(mps["owner"], f"{where}.owner")
                shared = obj(owner["inputs"], f"{where}.owner.inputs")
                held = sum(int(number(obj(v, "input")["bytes"], "bytes")) for v in shared.values())
                if int(number(owner["compared_bytes"], "compared_bytes")) != held:
                    raise ValueError(f"{where}: the owner did not compare every shared input")
            elif mps["owner"] is not None:
                raise ValueError(f"{where}: an mps run recorded an owner")
        case _ as never:
            assert_never(never)


def export_mixed() -> Numbers:
    """Tenants running the mix of model-written operators: through the service
    with Sonnet's accepted kernels in place of Opus's refused ones (`Service`),
    with Opus's kernels as written and the refused kernels' templates skipped
    (`Refusals`) or served by the engine's built-in operators (`Fallback`), and
    the operators of `Fallback` under MPS with private (`Mps`) or shared
    (`MpsShared`) inputs. The median of the repeats: at the fewest and the most
    tenants, each configuration's throughput, median and 99th-percentile
    latency, and device memory, and the built-in operators' share of the
    queries' time; at the most tenants, the median and 99th-percentile latency
    of the two templates whose kernel the gate refuses, and the device memory
    of MPS divided by that of `Fallback`; and each configuration's highest
    throughput with its tenant count."""
    record = load(MIXED)
    if record.get("complete") is not True:
        raise ValueError(f"{MIXED}: not a completed record")
    if record["sources"] != mixed_sources():
        raise ValueError(f"{MIXED}: raw sources differ from the current implementation")
    configurations = tuple(
        text(v, "configurations") for v in values(record["configurations"], "configurations")
    )
    if configurations != MIXED_CONFIGURATIONS:
        raise ValueError(f"{MIXED}: unexpected configurations {configurations}")
    tenants = tuple(
        int(number(v, "tenants")) for v in values(record["tenant_counts"], "tenant_counts")
    )
    if any(n not in TENANT_WORDS for n in tenants):
        raise ValueError(f"{MIXED}: a tenant count without a macro name")
    duration = number(record["duration_seconds"], "duration_seconds")
    expected = _mixed_expected(record)
    repeats = int(number(record["repeats"], "repeats"))
    cells = _mixed_cells(MIXED, items(record["runs"], "runs"), expected, duration)
    if set(cells) != {(c, n) for c in MIXED_CONFIGURATIONS for n in tenants}:
        raise ValueError(f"{MIXED}: missing measurement cells")
    if any(len(repeated) != repeats for repeated in cells.values()):
        raise ValueError(f"{MIXED}: a cell without {repeats} repeats")
    fewest, most = min(tenants), max(tenants)
    # the templates whose kernel the gate refuses, which a built-in operator serves
    served = list(expected["service_fallback"].builtins)
    numbers: Numbers = {
        "valMixedTenants": f"{most}",
        "valMixedRepeats": f"{int(number(record['repeats'], 'repeats'))}",
        "valMixedSeconds": f"{duration:g}",
        "valMixedTemplates": f"{len(MIXES['service_fallback'])}",
        "valMixedRefusedKernels": f"{len(expected['service_fallback'].refused)}",
    }
    for configuration, name in MIXED_NAMES.items():
        for count in (fewest, most):
            word, repeated = TENANT_WORDS[count], cells[(configuration, count)]
            numbers |= {
                f"valMixed{name}Qps{word}": f"{_spread(repeated, 'queries_per_second')[0]:,.0f}",
                f"valMixed{name}MedianUs{word}": f"{_spread(repeated, 'latency_us', 'p50')[0]:,.0f}",
                f"valMixed{name}TailUs{word}": f"{_spread(repeated, 'latency_us', 'p99')[0]:,.0f}",
                f"valMixed{name}Mib{word}": f"{_spread(repeated, 'memory_mib')[0]:,.0f}",
            }
            if serves_refused(configuration):
                share = 100 * _spread(repeated, "builtin_share")[0]
                numbers[f"valMixed{name}BuiltinPercent{word}"] = f"{share:.1f}"
            for template in served:
                if template in expected[configuration].skipped or count != most:
                    continue
                path = ("latency_us_by_template", template)
                median = _spread(repeated, *path, "p50")[0]
                tail = _spread(repeated, *path, "p99")[0]
                label = TEMPLATE_WORDS[template]
                numbers[f"valMixed{name}{label}MedianUs{word}"] = f"{median:,.0f}"
                numbers[f"valMixed{name}{label}TailUs{word}"] = f"{tail:,.0f}"
        peak = max(tenants, key=lambda n: _spread(cells[(configuration, n)], "queries_per_second"))
        numbers[f"valMixed{name}PeakQps"] = (
            f"{_spread(cells[(configuration, peak)], 'queries_per_second')[0]:,.0f}"
        )
        numbers[f"valMixed{name}PeakTenants"] = f"{peak}"
    fallback = _spread(cells[("service_fallback", most)], "memory_mib")[0]
    for configuration in ("mps", "mps_shared"):
        memory = _spread(cells[(configuration, most)], "memory_mib")[0]
        ratio = f"{memory / fallback:.1f}"
        numbers[f"valMixed{MIXED_NAMES[configuration]}MibRatio{TENANT_WORDS[most]}"] = ratio
    rows = []
    for n in tenants:
        row: list[object] = [n]
        for c in MIXED_CONFIGURATIONS:
            repeated = cells[(c, n)]
            qps = _spread(repeated, "queries_per_second")
            p50 = _spread(repeated, "latency_us", "p50")
            p99 = _spread(repeated, "latency_us", "p99")
            memory = _spread(repeated, "memory_mib")[0]
            share = 100 * _spread(repeated, "builtin_share")[0]
            row += [f"{v:.1f}" for v in (*qps, p50[0], *p99)] + [f"{memory:.0f}", f"{share:.2f}"]
        rows.append(row)
    write(
        "mixed.csv",
        (
            "tenants",
            *(
                f"{m}_{c}{suffix}"
                for c in MIXED_CONFIGURATIONS
                for m, suffixes in (
                    ("qps", ("", "_lo", "_hi")),
                    ("p50_us", ("",)),
                    ("p99_us", ("", "_lo", "_hi")),
                    ("memory_mib", ("",)),
                    ("builtin_percent", ("",)),
                )
                for suffix in suffixes
            ),
        ),
        rows,
    )
    template_rows = []
    for c in MIXED_CONFIGURATIONS:
        for n in tenants:
            repeated = cells[(c, n)]
            for template in TEMPLATE_WORDS:
                if template in expected[c].skipped:
                    continue
                path = ("latency_us_by_template", template)
                template_rows.append(
                    (
                        c,
                        n,
                        template,
                        f"{_spread(repeated, *path, 'p50')[0]:.1f}",
                        f"{_spread(repeated, *path, 'p99')[0]:.1f}",
                    )
                )
    write(
        "mixed_templates.csv",
        ("configuration", "tenants", "template", "p50_us", "p99_us"),
        template_rows,
    )
    return numbers


def _untimed(kernel: dict[str, object]) -> set[str]:
    """The texts that the executor loads for an accepted kernel and that are not
    timed, because their bytes are those of the kernel as written: of the text
    that the service loads, which the record must hold, and of the text with
    every access checked, which it holds when the gate accepts that text; each
    other loaded text, and Guardian's, must be timed."""
    where = text(kernel["kernel"], "kernel")
    results = obj(kernel["results"], "results")
    loaded = {v: text(t, v) for v, t in obj(kernel["loaded"], "loaded").items()}
    expected = {
        "opgate",
        *(("every",) if kernel["every_decision"] == "accepted" else ()),
        *(("guardian",) if kernel["guardian_fences"] is not None else ()),
    }
    if set(loaded) != expected or "as_written" not in results:
        raise ValueError(f"{where}: the loaded texts are {sorted(loaded)}, not {sorted(expected)}")
    written = text(field(kernel, "original", "ptx"), "ptx")
    untimed = {v for v in ("opgate", "every") if v in loaded and loaded[v] == written}
    if set(results) != {"as_written", *loaded} - untimed:
        raise ValueError(
            f"{where}: the timed texts {sorted(results)} are not the loaded texts that differ "
            "from the kernel as written"
        )
    return untimed


def _cost_outcomes(kernel: dict[str, object]) -> dict[str, bool]:
    """Validate each repeat's actual outcomes before using any timing sample."""
    if kernel["decision"] != "accepted":
        if kernel["decision"] != "refused" or "results" in kernel or "loaded" in kernel:
            raise ValueError(f"{kernel['kernel']}: invalid verification or a refused kernel ran")
        return {}
    results = obj(kernel["results"], "results")
    _untimed(kernel)
    status = {
        "as_written": False,
        "opgate": bool(kernel["guarded_accesses"] or kernel["limited_loops"]),
        "every": True,
        "guardian": False,
    }
    outcomes = {}
    for version, raw in results.items():
        result = obj(raw, version)
        if version not in status or field(result, "check_failed") is not (
            False if status[version] else None
        ):
            raise ValueError(f"{kernel['kernel']} {version}: invalid runtime check outcome")
        outcomes[version] = field(result, "check") is None
    if outcomes["as_written"] and any(
        not correct for version, correct in outcomes.items() if version != "guardian"
    ):
        raise ValueError(f"{kernel['kernel']}: a transformed kernel fails its task reference")
    return outcomes


def export_guard_cost() -> Numbers:
    """Each safe kernel's device time as the service loads it, with every access
    checked, and with Guardian's fencing, against the kernel as written. The
    text that the service loads, with the pass's changes and the zeroing
    prologue, and the text with every access checked, loaded with its
    prologue, are timed whenever their bytes differ from the kernel as
    written (`_untimed`); a text whose recorded bytes equal the kernel as
    written takes the time of the kernel as written, so its ratio is 1. A
    kernel whose task check fails as written, as a filter that sets bits by
    addition, is timed but has no check to keep; for every other kernel the
    task's check must pass for every text but Guardian's, whose failures the
    CSV file states. A text with the pass's status buffer, which its checks
    and limits take, must report no failed check on the workload; a text
    without it reports none.

    The systems are compared on the common cohort: the kernels that pass the
    task reference as written and that all three systems run correctly, each
    with its three ratios against the same kernel as written. OpGate's
    overhead is also reported over the useful accepted kernels and over the
    kernels that the pass changed, with checks, a barrier rewrite, or
    iteration limits. The coverage of a system is the number of kernels of
    the full cost set with a ratio of that system; the experiment times the
    kernels that the gate accepts."""
    records = runs(GUARD_COST, GUARD_COST_RUNS)
    for index, record in enumerate(records, start=1):
        if record.get("complete") is not True:
            raise ValueError("the cost experiment has an incomplete run")
        snapshot_inputs(
            GUARD_COST / f"run{index}.json", "sources", guard_cost_sources(), create=False
        )
    names = [[text(k["kernel"], "kernel") for k in items(r["kernels"], "kernels")] for r in records]
    if any(n != names[0] for n in names):
        raise ValueError("the cost experiment's runs measured different kernels")
    rows = []
    covered = dict.fromkeys(CHECKED_VERSIONS, 0)
    opgate: list[float] = []  # OpGate's ratio on each useful accepted kernel
    # OpGate's ratio on each useful kernel that the pass changed, with whether
    # the pass limited a loop of it and whether it rewrote its participation
    changed: list[tuple[float, bool, bool]] = []
    common: list[tuple[str, dict[str, float]]] = []
    wrong: list[str] = []
    wrong_as_written: list[str] = []
    repetitions = [items(r["kernels"], "kernels") for r in records]
    for i, k in enumerate(repetitions[0]):
        outcomes = _cost_outcomes(k)
        for repeated in repetitions[1:]:
            other = repeated[i]
            fields = (
                "original",
                "loaded",
                "recorded",
                "decision",
                "every_decision",
                "guarded_accesses",
                "limited_loops",
                "restructured",
                "every_checks",
                "guardian_fences",
                "guardian_unsupported",
            )
            if any(other.get(f) != k.get(f) for f in fields) or _cost_outcomes(other) != outcomes:
                raise ValueError(
                    f"{k['kernel']}: verification or functional outcomes differ between repeats"
                )
        if k["decision"] != "accepted":
            if "results" in k:
                raise ValueError(f"{k['kernel']}: a kernel that the gate refuses ran")
            continue
        results = obj(k["results"], "results")
        checks = values(k["guarded_accesses"], "guarded_accesses")
        limits = values(k["limited_loops"], "limited_loops")
        if type(k["restructured"]) is not bool:
            raise ValueError(f"{k['kernel']}: restructured is not a boolean")
        untimed = _untimed(k)
        # whether each text takes the pass's status buffer
        status = {
            "as_written": False,
            "opgate": bool(checks or limits),
            "every": True,
            "guardian": False,
        }
        for version in results:
            if version not in status:
                raise ValueError(f"{k['kernel']}: a text {version} of no known kind")
            failed = field(results, version, "check_failed")
            if failed is not (False if status[version] else None):
                raise ValueError(f"{k['kernel']} {version}: the failed check is {failed}")
        written = across(records, "kernels", i, "results", "as_written", "time").median
        correct = field(results, "as_written", "check") is None
        if not correct:
            wrong_as_written.append(text(k["kernel"], "kernel"))
        # each text's time: of its run, or of the run of the kernel as written,
        # whose bytes it has
        times = dict.fromkeys(untimed, written)
        for version in CHECKED_VERSIONS:
            if version not in results:
                continue
            check = field(results, version, "check")
            if correct and check is not None and version != "guardian":
                raise ValueError(f"{k['kernel']} {version}: the task's check fails: {check}")
            if correct and check is not None:
                wrong.append(text(k["kernel"], "kernel"))
            times[version] = across(records, "kernels", i, "results", version, "time").median
        ratio = {v: times[v] / written for v in CHECKED_VERSIONS if v in times}
        for version in ratio:
            covered[version] += 1
        if correct and set(ratio) == set(CHECKED_VERSIONS) and k["kernel"] not in wrong:
            common.append((text(k["kernel"], "kernel"), ratio))
        if correct:
            opgate.append(ratio["opgate"])
            if checks or limits or k["restructured"]:
                changed.append((ratio["opgate"], bool(limits), k["restructured"] is True))
        rows.append(
            (
                k["kernel"],
                k["recorded"],
                f"{written:.1f}",
                *(f"{ratio[v]:.4f}" if v in ratio else "" for v in CHECKED_VERSIONS),
                len(checks),
                len(limits),
                k["restructured"],
                k["every_checks"],
                "" if k["guardian_fences"] is None else k["guardian_fences"],
                "" if k["guardian_unsupported"] is None else text(k["guardian_unsupported"], "why"),
                k["kernel"] in wrong,
                k["kernel"] in wrong_as_written,
            )
        )
    write(
        "guard_cost.csv",
        (
            "kernel",
            "recorded",
            "as_written_us",
            *(f"{v}_ratio" for v in CHECKED_VERSIONS),
            "checked_accesses",
            "limited_loops",
            "restructured",
            "every_checks",
            "guardian_fences",
            "guardian_unsupported",
            "guardian_wrong",
            "wrong_as_written",
        ),
        rows,
    )

    def geomean(values: list[float]) -> str:
        return f"{100 * (statistics.geometric_mean(values) - 1):.1f}"

    write(
        "guard_cost_common.csv",
        ("kernel", *CHECKED_VERSIONS),
        ((key, *(f"{ratio[v]:.4f}" for v in CHECKED_VERSIONS)) for key, ratio in common),
    )
    numbers: Numbers = {
        f"valCostCommon{v.capitalize()}Geomean": geomean([r[v] for _, r in common])
        for v in CHECKED_VERSIONS
    }
    for version in CHECKED_VERSIONS:
        numbers[f"valCost{version.capitalize()}Covered"] = f"{covered[version]}"
    doubled = [c for c in changed if c[0] > 2]
    # the text states that every changed kernel above twice its time has iteration limits
    if not all(limited for _, limited, _ in doubled):
        raise ValueError("a changed kernel takes more than twice as long without iteration limits")
    return numbers | {
        "valCostChangedMedian": f"{100 * (statistics.median(r for r, _, _ in changed) - 1):.1f}",
        "valCostChangedDoubled": f"{len(doubled)}",
        "valCostChangedDoubledRestructured": f"{sum(rewritten for _, _, rewritten in doubled)}",
        "valCostOpgateMaxRatio": f"{max(opgate):.2f}",
        "valCostSet": f"{len(repetitions[0])}",
        "valCostKernels": f"{len(rows)}",
        "valCostWrongAsWritten": f"{len(wrong_as_written)}",
        "valCostUsefulKernels": f"{len(rows) - len(wrong_as_written)}",
        "valCostGuardianWrong": f"{len(wrong)}",
        "valCostCommonKernels": f"{len(common)}",
        "valCostOpgateGeomean": geomean(opgate),
        "valCostOpgateMax": f"{100 * (max(opgate) - 1):.0f}",
        "valCostChangedKernels": f"{len(changed)}",
        "valCostChangedGeomean": geomean([r for r, _, _ in changed]),
    }


Isolation = Literal["shared_unchecked", "process", "mps", "service", "guardian"]
# the configurations of the fault isolation record, in its order and the figure's
ISOLATION_ORDER: tuple[Isolation, ...] = (
    "shared_unchecked",
    "process",
    "mps",
    "service",
    "guardian",
)
ISOLATION_NAMES: dict[Isolation, str] = {
    "process": "Process",
    "mps": "Mps",
    "service": "Service",
    "guardian": "Guardian",
}
ATTACKER = 9000  # the attacker's tenant
TENANT_FIELDS = frozenset(
    {
        "tenant",
        "role",
        "completions",
        "correct",
        "error",
        "ended",
        "note",
        "completed_after_fault",
        "max_gap_seconds",
    }
)
FAULT = "CUDA_ERROR_ILLEGAL_ADDRESS"  # the driver's error of the wild read
BIN = 0.05  # seconds of one point of the victims' throughput


def _contained(configuration: Isolation) -> bool:
    """Whether the victims continue after the attacker's fault."""
    match configuration:
        case "shared_unchecked":
            return False
        case "process" | "mps" | "service" | "guardian":
            return True
        case _ as unreachable:
            assert_never(unreachable)


def _times(tenant: dict[str, object], runtime: float, where: str) -> list[float]:
    """The completion times of a tenant's queries, in order. Only the last query
    may end after the window, since the victim loop starts none after it."""
    completions = items(tenant["completions"], f"{where}.completions")
    if any(set(c) != {"at", "revenue"} for c in completions):
        raise ValueError(f"{where}: a completion with other fields")
    times = [number(c["at"], f"{where}.at") for c in completions]
    if any(t < 0 for t in times) or times != sorted(times):
        raise ValueError(f"{where}: completion times out of order")
    if any(t >= runtime for t in times[:-1]):
        raise ValueError(f"{where}: a query completed after the window's end")
    return times


def _gap(times: list[float], runtime: float) -> float:
    """The longest time between the window's start, each completion, and its end."""
    bounds = [0.0, *times, runtime]
    return max(later - earlier for earlier, later in zip(bounds, bounds[1:], strict=False))


class Window(NamedTuple):
    """A fault isolation run's victims, window, and fault time, in seconds."""

    victims: int
    runtime: float
    fault: float


def _isolation_run(
    run: dict[str, object], configuration: Isolation, where: str
) -> tuple[Window, list[list[float]]]:
    """Validate one configuration's run against the outcome that the paper states
    for it: the victims continue with correct results where the fault is
    contained and end at it where it is not, and the attacker's wild read
    faults, except in the service, which answers its failed check. Return the
    run's window and each victim's completion times."""
    window = Window(
        int(number(run["victims"], "victims")),
        number(run["runtime_seconds"], "runtime_seconds"),
        number(run["fault_at_seconds"], "fault_at_seconds"),
    )
    if run["configuration"] != configuration or run["attacker"] is not True:
        raise ValueError(f"{where}: another configuration, or no attacker")
    tenants = items(run["tenants"], f"{where}.tenants")
    roles = [(t["tenant"], t["role"]) for t in tenants]
    if roles != [*((v, "victim") for v in range(1, window.victims + 1)), (ATTACKER, "attacker")]:
        raise ValueError(f"{where}: other tenants")
    contained = _contained(configuration)
    found: list[list[float]] = []
    for tenant in tenants:
        name = f"{where}.{tenant['tenant']}"
        if set(tenant) != TENANT_FIELDS:
            raise ValueError(f"{name}: the fields {sorted(tenant)}")
        times = _times(tenant, window.runtime, name)
        if (
            tenant["completed_after_fault"] != sum(1 for t in times if t >= window.fault)
            or tenant["max_gap_seconds"] != _gap(times, window.runtime)
            or tenant["correct"] is not True
        ):
            raise ValueError(f"{name}: a summary differs from its completions, or a wrong result")
        error, ended = tenant["error"], tenant["ended"]
        if tenant["role"] == "attacker":
            faulted = configuration not in ("service", "guardian")
            if ended is not faulted or faulted != (isinstance(error, str) and FAULT in error):
                raise ValueError(f"{name}: the attacker's fault is not as recorded")
        elif not times:
            raise ValueError(f"{name}: a victim completed no query")
        elif contained:
            if ended is not False or error is not None or times[-1] < window.runtime:
                raise ValueError(f"{name}: the victim did not continue to the window's end")
            found.append(times)
        else:
            if ended is not True or not isinstance(error, str) or times[-1] >= window.runtime:
                raise ValueError(f"{name}: the victim did not end at the fault")
            found.append(times)
    match configuration:
        case "service":
            telemetry = obj(field(run, "service", "telemetry"), f"{where}.telemetry")
            if (telemetry["failed_checks"], telemetry["fatal_errors"]) != (1, 0):
                raise ValueError(f"{where}: the service did not answer one failed check")
        case "mps":
            if field(run, "service", "mps", "mode") != "static_sm_partitioning":
                raise ValueError(f"{where}: MPS without static partitions")
        case "guardian":
            layout = obj(field(run, "service", "guardian"), f"{where}.guardian")
            size = int(number(layout["partition_bytes"], "partition_bytes"))
            base = int(number(layout["attacker_base"], "attacker_base"))
            unfenced = int(number(layout["unfenced_target"], "unfenced_target"))
            fenced = int(number(layout["fenced_target"], "fenced_target"))
            if int(number(layout["fenced_accesses"], "fenced_accesses")) <= 0:
                raise ValueError(f"{where}: the attacker's kernel fences zero accesses")
            if base <= unfenced < base + size:
                raise ValueError(f"{where}: the attacker's write did not aim outside its partition")
            if not base <= fenced < base + size:
                raise ValueError(f"{where}: the fence did not confine the write to the partition")
            if layout["sink_after"] != layout["write_value"]:
                raise ValueError(f"{where}: the fenced write did not land in the attacker's sink")
            if layout["victim_word_after"] != layout["victim_word_before"]:
                raise ValueError(f"{where}: a victim's buffer changed under the fenced write")
        case "shared_unchecked" | "process":
            if run["service"] is not None:
                raise ValueError(f"{where}: a service or MPS in a direct configuration")
        case _ as unreachable:
            assert_never(unreachable)
    return window, found


def export_fault_isolation() -> Numbers:
    """The victims' throughput around the attacker's fault in each configuration,
    and, where the fault is contained, the victims' longest stall in the window
    and the fewest queries that a victim completes after the fault. The record
    must have one run of each configuration, all in the same window."""
    record = load(FAULT_ISOLATION)
    if record["attacker"] is not True or values(record["configurations"], "configurations") != list(
        ISOLATION_ORDER
    ):
        raise ValueError(f"{FAULT_ISOLATION}: other configurations, or no attacker")
    found = items(record["runs"], "runs")
    if len(found) != len(ISOLATION_ORDER):
        raise ValueError(f"{FAULT_ISOLATION}: not one run for each configuration")
    windows: set[Window] = set()
    victims: dict[Isolation, list[list[float]]] = {}
    for run, configuration in zip(found, ISOLATION_ORDER, strict=True):
        window, victims[configuration] = _isolation_run(
            run, configuration, f"{FAULT_ISOLATION}: {configuration}"
        )
        windows.add(window)
    if len(windows) != 1:
        raise ValueError(f"{FAULT_ISOLATION}: the runs differ in window: {windows}")
    ((count, runtime, fault),) = windows
    bins = round(runtime / BIN)
    if not math.isclose(bins * BIN, runtime):
        raise ValueError(f"the window of {runtime} s is not a whole number of points")
    rates: dict[Isolation, list[int]] = {}
    for configuration, found in victims.items():
        counted = [0] * bins
        for t in (t for times in found for t in times if t < runtime):
            counted[int(t // BIN)] += 1
        rates[configuration] = counted
    write(
        "fault_isolation.csv",
        ("seconds", *ISOLATION_ORDER),
        (
            (f"{(i + 0.5) * BIN:.3f}", *(f"{rates[c][i] / BIN / 1e3:.2f}" for c in ISOLATION_ORDER))
            for i in range(bins)
        ),
    )
    numbers: Numbers = {
        "valIsoVictims": f"{count}",
        "valIsoIntervalMs": f"{BIN * 1e3:.0f}",
        "valIsoFaultSeconds": f"{fault:g}",
        "valIsoWindowSeconds": f"{runtime:g}",
    }
    for configuration, name in ISOLATION_NAMES.items():
        found = victims[configuration]
        stall = max(_gap(t, runtime) for t in found)
        after = min(sum(1 for x in t if x >= fault) for t in found)
        numbers |= {
            f"valIso{name}StallMs": f"{stall * 1e3:.0f}",
            f"valIso{name}AfterFault": f"{after:,}",
        }
    return numbers


class SizeGroup(NamedTuple):
    """The registrations of texts of `low` up to `high` PTX instructions, `high` excluded."""

    low: int
    high: int | None  # None: unbounded
    label: str  # in the table


SIZE_GROUPS = (
    SizeGroup(0, 100, "$<$100"),
    SizeGroup(100, 1_000, "100--999"),
    SizeGroup(1_000, 5_000, "1,000--4,999"),
    SizeGroup(5_000, None, "$\\geq$5,000"),
)
PERCENTILES = (90, 99)


def _percentile(values: list[float], percent: int) -> float:
    """The `percent`th percentile of `values`, interpolated between the order
    statistics (`statistics.quantiles`, inclusive method)."""
    return statistics.quantiles(values, n=100, method="inclusive")[percent - 1]


def _seconds(value: float) -> str:
    """Seconds in the text and the table: one decimal below 100 s, whole seconds from 100 s."""
    return f"{value:.1f}" if value < 100 else f"{value:,.0f}"


def _milliseconds(value: float) -> str:
    """Milliseconds in the text and the table: two decimals below 1 ms, one from 1 ms."""
    return f"{value:.2f}" if value < 1 else f"{value:.1f}"


def plans(kernels: list[Kernel]) -> list[list[Kernel]]:
    """The DataKernelBench plans: the kernels of one model whose task directories
    hold the same plan file, byte for byte, in the order of `kernels`."""
    found: dict[tuple[str, bytes], list[Kernel]] = {}
    for kernel in kernels:
        if kernel.collection != "DataKernelBench":
            continue
        model, task, _ = kernel.key
        plan = (directory_of(model, task) / "plan.py").read_bytes()
        found.setdefault((model, plan), []).append(kernel)
    return list(found.values())


def _compiles(kernels: list[Kernel]) -> dict[Key, float]:
    """Each CUDA C++ and Triton kernel's compile from the model's source to PTX,
    from the generation record, whose sources must be the current ones."""
    record = load(GENERATION)
    keys = sorted(k.key for k in kernels if k.key[2] in COMPILED)
    if record.get("complete") is not True or field(record, "provenance", "kind") != (
        GENERATION_KIND
    ):
        raise ValueError(f"{GENERATION}: not a complete record of the generation experiment")
    snapshot_inputs(GENERATION, "sources", generation_sources(keys), create=False)
    found: dict[Key, float] = {}
    for row in items(record["results"], "results"):
        model, task, toolchain = text(row["id"], "id").split("/")
        found[(model, task, toolchain)] = number(row["seconds"], "seconds")
    if sorted(found) != keys:
        raise ValueError(f"{GENERATION}: the compiled kernels are not the corpus's")
    return found


def schedule(seconds: list[float], workers: int) -> float:
    """The end of the last of the checks that take `seconds`, each started in
    its order on the first of `workers` threads that is free: the service's
    verification threads, which take the checks of a request in its order."""
    if workers < 1:
        raise ValueError("a schedule needs a thread")
    free = [0.0] * workers
    for check in seconds:
        heapq.heappush(free, heapq.heappop(free) + check)
    return max(free)


class FreshProcess(NamedTuple):
    """The PyTorch engine's cold start in one fresh process (`cold_query.py`),
    each the median of the runs: its kernels' compiles, the service's decisions,
    and the module loads, each summed, in seconds; the tables' copy into the
    tenant's session, in seconds; and each query's first run, in milliseconds."""

    runs: int
    compiled: float
    decided: float
    loaded: float
    delivered: float
    first: dict[str, float]


def _fresh_process() -> FreshProcess:
    """The cold-query record's runs, each at scale factor 1 with the engine's
    kernels under the service's budget: every compile's PTX equals the corpus's
    text, the gate accepts every kernel, and every query's result equals pandas'."""
    records = runs(COLD_QUERY)
    kernels = ["/".join((model, task, "triton")) for task, model in MODELS.items()]
    for record in records:
        if (
            number(record["scale"], "scale") != 1.0
            or record["models"] != MODELS
            or number(record["registration_seconds"], "registration_seconds")
            != REGISTRATION_SECONDS
        ):
            raise ValueError(f"{COLD_QUERY}: a run of another scale, engine, or budget")
        found = items(record["kernels"], "kernels")
        if [text(k["kernel"], "kernel") for k in found] != kernels:
            raise ValueError(f"{COLD_QUERY}: a run of other kernels than the engine's")
        if any(k["same_as_corpus"] is not True or k["refusal"] is not None for k in found):
            raise ValueError(f"{COLD_QUERY}: a compile differs from the corpus or a refusal")
        queries = items(record["queries"], "queries")
        if [text(q["query"], "query") for q in queries] != list(QUERIES):
            raise ValueError(f"{COLD_QUERY}: a run of other queries than the engine's")
        if any(q["mismatch"] is not None for q in queries):
            raise ValueError(f"{COLD_QUERY}: a result differs from pandas'")

    def summed(key: str) -> float:
        return statistics.median(
            sum(number(k[key], key) for k in items(r["kernels"], "kernels")) for r in records
        )

    return FreshProcess(
        len(records),
        summed("compile_seconds"),
        summed("decide_seconds"),
        summed("load_seconds"),
        statistics.median(number(r["tables_seconds"], "tables_seconds") for r in records),
        {
            name: statistics.median(
                number(field(r, "queries", q, "first_seconds"), "first_seconds") * 1e3
                for r in records
            )
            for q, name in enumerate(QUERIES)
        },
    )


def export_cold_query(kernels: list[Kernel], guarded: list[dict[str, object]]) -> Numbers:
    """The cold compile and registration of a query whose kernels are new to the
    service, beside the warm query on the registered kernels (`tab_cold.tex`).

    A kernel's compile is the generation record's front-end compile of its
    source. Its registration is the guarded record's: the check of the text as
    written and, when the guard pass applies, its rounds, as the service decides
    a registration (`opgate.hosts.service.server._check`), under one budget.
    The service checks the kernels of one register request at once on the
    `VERIFICATION_WORKERS` threads of the evaluation's deployments, each kernel
    in the request's order on the first thread that is free, so a query's wait
    for its registrations is the end of the last check of that schedule
    (`schedule`), with its kernels in the corpus's order (`plans`); the records
    ran as many checks at a time, and every plan fits in one request
    (`Limits.registrations_in_flight`). The sum over the kernels is the checks'
    time in all. A DataKernelBench plan counts when the service accepts each of
    its kernels. The PyTorch engine registers every kernel of
    `torch_engine.MODELS` before its queries, one after another
    (`GatedOperators.register`); its warm queries are the torch-queries
    record's executor times at scale factor 1, from the start of a query to its
    result on the host. The middle part is the engine's cold start in one fresh
    process (`_fresh_process`): its compiles, decisions, and module loads, the
    delivery of the tables to the tenant's session, and each query's first run,
    which includes its kernels' first launches. The size groups are over every
    corpus entry."""
    workers = {
        int(number(field(load(path), "provenance", "workers"), "workers"))
        for path in (REGISTRATION, GUARDED)
    }
    if len(workers) != 1:
        raise ValueError("the registration and guarded records ran different numbers of checks")
    (concurrent,) = workers
    if concurrent != VERIFICATION_WORKERS:
        raise ValueError(
            f"the records ran {concurrent} checks at a time, and the service runs "
            f"{VERIFICATION_WORKERS}"
        )
    places = Limits(verification_workers=VERIFICATION_WORKERS).registrations_in_flight
    compiled = _compiles(kernels)
    service: dict[Key, tuple[bool, float]] = {}
    for row in guarded:
        model, task, toolchain = text(row["id"], "id").split("/")
        service[(model, task, toolchain)] = (
            row["decision"] == "accepted",
            number(row["seconds"], "seconds"),
        )
    if set(service) != {k.key for k in kernels}:
        raise ValueError("the guarded rows are not the corpus's kernels")
    seconds = {key: s for key, (_, s) in service.items()}

    every = plans(kernels)
    complete = [[k.key for k in p] for p in every if all(service[k.key][0] for k in p)]
    if not complete:
        raise ValueError("the service accepts no DataKernelBench plan")
    if max(len(keys) for keys in every) > places:
        raise ValueError(f"a plan has more kernels than one request's {places}")
    engine: list[Key] = [(model, task, "triton") for task, model in MODELS.items()]
    if any(key not in service or not service[key][0] for key in engine):
        raise ValueError("the service refuses a kernel of the PyTorch engine")
    by_key = {k.key: k for k in kernels}

    def columns(keys: list[Key]) -> tuple[float, float, float, float, float]:
        """The kernels, their compiles, checks as written, and registrations,
        each summed, and the wait for the registrations in one request."""
        return (
            float(len(keys)),
            sum(compiled[k] for k in keys),
            sum(by_key[k].seconds for k in keys),
            sum(seconds[k] for k in keys),
            schedule([seconds[k] for k in keys], VERIFICATION_WORKERS),
        )

    found = [columns(keys) for keys in complete]
    (scale_one,) = (d for d in TORCH_QUERIES if number(runs(d)[0]["scale"], "scale") == 1.0)
    records = runs(scale_one)
    if any(r["models"] != MODELS for r in records):
        raise ValueError(f"{scale_one}: the runs ran other kernels than the engine's")
    queries = [text(q["query"], "query") for q in items(records[0]["queries"], "queries")]
    warm = [
        across(records, "queries", q, "measured", "opgate", "time").median / 1e3
        for q in range(len(queries))
    ]
    fresh = _fresh_process()
    if list(fresh.first) != queries:
        raise ValueError(f"{COLD_QUERY}: other queries than {scale_one}'s")

    def cells(values: tuple[float, ...]) -> str:
        count, *timed = values
        return " & ".join((f"{count:.0f}", *(_seconds(v) for v in timed)))

    def picked(label: str, pick: Callable[[list[float]], float]) -> str:
        values = tuple(pick([row[i] for row in found]) for i in range(5))
        return f"\\quad {label} & {cells(values)} & -- \\\\"

    torch_columns = columns(engine)
    rows = [
        f"PyTorch & {cells(torch_columns)} & {'/'.join(_milliseconds(t) for t in warm)} \\\\",
        f"\\multicolumn{{7}}{{@{{}}l}}{{DataKernelBench plans ({len(complete)})}} \\\\",
        picked("median", statistics.median),
        picked("90th", lambda v: _percentile(v, 90)),
        picked("max", max),
        "\\midrule",
        "\\emph{One process} & \\emph{Kernels} & \\emph{Compile} & "
        "\\emph{Registered} & \\emph{Load (ms)} & \\emph{Delivery} & "
        "\\emph{First (ms)} \\\\",
        f"PyTorch & {len(engine)} & {_seconds(fresh.compiled)} & {_seconds(fresh.decided)} & "
        f"{_milliseconds(fresh.loaded * 1e3)} & {_seconds(fresh.delivered)} & "
        f"{'/'.join(_milliseconds(t) for t in fresh.first.values())} \\\\",
        "\\midrule",
        "\\emph{Instructions} & \\emph{Entries} & \\emph{Written} & "
        "\\emph{Median} & \\emph{90th} & \\emph{99th} & \\emph{Max} \\\\",
        "\\midrule",
    ]
    for group in SIZE_GROUPS:
        members = [
            k
            for k in kernels
            if group.low <= k.instructions and (group.high is None or k.instructions < group.high)
        ]
        registrations = [seconds[k.key] for k in members]
        rows.append(
            f"{group.label} & {len(members):,} & "
            f"{_seconds(statistics.median(k.seconds for k in members))} & "
            f"{_seconds(statistics.median(registrations))} & "
            + " & ".join(_seconds(_percentile(registrations, p)) for p in PERCENTILES)
            + f" & {_seconds(max(registrations))} \\\\"
        )
    compile_workers = int(number(field(load(GENERATION), "provenance", "workers"), "workers"))
    table(
        "tab_cold",
        "@{}lrrrrrr@{}",
        "\\textbf{Query} & \\textbf{Kernels} & \\textbf{Compile} & \\textbf{Written} & "
        "\\textbf{Registered} & \\textbf{Service} & \\textbf{Warm (ms)}",
        rows,
        "Registration runs once per kernel; afterwards a query takes milliseconds. "
        "Seconds summed over a query's new kernels; Written: one check per "
        "text; Registered: the service's decision, with the guard pass's rounds; Service: a "
        f"request's wait. Middle: TPC-H {'/'.join(queries)} at scale factor 1 in one fresh "
        f"process, median of {fresh.runs}; Delivery: the tables' copy to the session. Lower: "
        f"entries by PTX instructions; a decision past {REGISTRATION_SECONDS:.0f}~s is a "
        f"refusal; {concurrent} checks and {compile_workers} compiles ran at once.",
    )
    distinct = {(item.ptx, item.binding) for item in inputs()}
    waits = [row[4] for row in found]
    ratios = [row[3] / row[1] for row in found]
    return {
        "valColdPlans": f"{len(complete)}",
        "valColdPlansAll": f"{len(every)}",
        "valColdPlanKernels": f"{statistics.median(row[0] for row in found):.0f}",
        "valColdPlanCompileSeconds": _seconds(statistics.median(row[1] for row in found)),
        "valColdPlanSeconds": _seconds(statistics.median(waits)),
        "valColdPlanMaxSeconds": _seconds(max(waits)),
        "valColdPlanCompileRatio": f"{statistics.median(ratios):.0f}",
        "valColdTorchKernels": f"{len(engine)}",
        "valColdTorchSeconds": _seconds(torch_columns[3]),
        "valColdTorchWarmQueries": f"{torch_columns[3] * 1e3 / max(warm):,.0f}",
        "valColdDistinct": f"{len(distinct):,}",
        "valColdWorkers": f"{VERIFICATION_WORKERS}",
    }


def export_protocol() -> Numbers:
    """The measurement protocol of the timed experiments (`eval.timing`): the
    runs of each record, the guard-cost experiment's runs, and each
    measurement's warm-up and samples."""
    return {
        "valTimingRuns": f"{RUNS}",
        "valTimingSamples": f"{REPEATS}",
        "valTimingWarmupSeconds": f"{WARMUP_SECONDS:g}",
        "valCostRuns": f"{GUARD_COST_RUNS}",
    }


def export(kernels: list[Kernel]) -> Numbers:
    guarded, _ = guarded_rows(GUARDED, kernels)
    return (
        export_protocol()
        | export_registration(kernels)
        | export_step_check()
        | export_clearing()
        | export_kernel_time()
        | export_query_overhead()
        | export_consolidation()
        | export_mixed()
        | export_guard_cost()
        | export_fault_isolation()
        | export_cold_query(kernels, guarded)
    )
