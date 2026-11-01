"""Pipelines: the static index path against the guarded one, the 22 TPC-H
queries on cuDF through the catalog, and TPC-H queries on a PyTorch engine."""

import statistics
from pathlib import Path

from eval.export.records import (
    RECORDS,
    SCALE_NAMES,
    Across,
    Numbers,
    across,
    field,
    items,
    number,
    obj,
    runs,
    text,
    write,
)

INDEX_PATH = RECORDS / "index_path_2026-10-24"  # rq3_pipelines/index_path.py
# rq3_pipelines/tpch_queries.py at scale factors 1 and 10: each one's name in the
# data files and in the numbers, and its record
TPCH_QUERIES = (
    ("sf1", "One", RECORDS / "tpch_queries_sf1_2026-10-24"),
    ("sf10", "Ten", RECORDS / "tpch_queries_sf10_2026-10-24"),
)
# rq3_pipelines/torch_queries.py at scale factors 1 and 10
TORCH_QUERIES = (
    RECORDS / "torch_queries_sf1_2026-10-24",
    RECORDS / "torch_queries_sf10_2026-10-24",
)


def _difference(
    records: list[dict[str, object]],
    minuend: tuple[str | int, ...],
    subtrahend: tuple[str | int, ...],
) -> Across:
    """`minuend - subtrahend` in each run, across the runs."""
    values = [
        number(field(r, *minuend, "median_us"), "minuend")
        - number(field(r, *subtrahend, "median_us"), "subtrahend")
        for r in records
    ]
    return Across(statistics.median(values), min(values), max(values))


def export_index_path() -> Numbers:
    """The sum step on each path for filters of every selectivity, and the time
    the guarded path adds, at each scale."""
    records = runs(INDEX_PATH)
    first = records[0]
    rows = []
    numbers: Numbers = {}
    for s, scale in enumerate(items(first["scales"], "scales")):
        factor = number(scale["scale"], "scale")
        lines = int(number(scale["rows"], "rows"))
        added = []
        for f, filter in enumerate(items(scale["filters"], "filters")):
            path = ("scales", s, "filters", f)
            selected = int(number(filter["selected"], "selected"))
            check = _difference(records, (*path, "eager_guarded"), (*path, "eager_static"))
            added.append(check.median)
            rows.append(
                (
                    factor,
                    f"{100 * selected / lines:.1f}",
                    *across(records, *path, "eager_static").columns(),
                    *across(records, *path, "eager_guarded").columns(),
                    *across(records, *path, "direct").columns(),
                    *check.columns(),
                )
            )
        name = SCALE_NAMES[factor]
        numbers[f"valGuardAdd{name}"] = f"{statistics.median(added) / 1e3:.2f}"
        numbers[f"valIndexElements{name}"] = f"{lines / 1e6:.1f}"
    write(
        "index_path.csv",
        (
            "scale",
            "percent",
            *(
                f"{c}{s}"
                for c in ("static", "guarded", "direct", "check")
                for s in ("", "_lo", "_hi")
            ),
        ),
        rows,
    )
    return numbers


def export_tpch_queries(scale: str, name: str, directory: Path) -> tuple[Numbers, list[int]]:
    """The 22 TPC-H queries on cuDF alone and with every allocation through the
    catalog at one scale factor: each query's time in both, the catalog's steps
    and clears, and the steps of each query. The engine allocates in its
    tenant's arena (decision 35), so a query takes a step only when the arena
    grows."""
    records = runs(directory)
    queries = items(records[0]["queries"], "queries")
    rows = []
    ratios, steps = [], []
    cudf_total = catalog_total = 0.0
    for q, query in enumerate(queries):
        counts = [field(r, "queries", q, "catalog_counts") for r in records]
        if any(c != counts[0] for c in counts):
            raise ValueError(f"{directory}: the runs counted other steps for {query['query']}")
        if query["mismatches"]:
            raise ValueError(f"{directory}: {query['query']} has a mismatch")
        cudf = across(records, "queries", q, "cudf")
        catalog = across(records, "queries", q, "catalog")
        count = int(number(field(query, "catalog_counts", "steps"), "steps"))
        ratios.append(catalog.median / cudf.median)
        steps.append(count)
        cudf_total += cudf.median
        catalog_total += catalog.median
        rows.append(
            (
                text(query["query"], "query").upper(),
                *cudf.columns(1e3, 2),
                *catalog.columns(1e3, 2),
                count,
                int(number(field(query, "catalog_counts", "clears"), "clears")),
                f"{number(field(query, 'catalog_counts', 'cleared_bytes'), 'bytes') / 2**20:.1f}",
            )
        )
    write(
        f"tpch_queries_{scale}.csv",
        (
            "query",
            *(f"{c}{s}" for c in ("cudf_ms", "catalog_ms") for s in ("", "_lo", "_hi")),
            "steps",
            "clears",
            "cleared_mib",
        ),
        rows,
    )
    numbers = {
        f"valTpch{name}RatioLow": f"{min(ratios):.2f}",
        f"valTpch{name}RatioMedian": f"{statistics.median(ratios):.2f}",
        f"valTpch{name}RatioHigh": f"{max(ratios):.2f}",
        f"valTpch{name}RatioTotal": f"{catalog_total / cudf_total:.2f}",
        f"valTpch{name}StepsLow": f"{min(steps):,}",
        f"valTpch{name}StepsHigh": f"{max(steps):,}",
    }
    return numbers, steps


TORCH_CONFIGURATIONS = ("torch", "direct", "opgate")


def export_torch_queries() -> Numbers:
    """TPC-H queries on the PyTorch engine with PyTorch's operators, the models'
    Triton kernels launched directly, and the same kernels' PTX verified by the
    gate and launched through the executor, at each scale factor: each
    configuration's time, and the time that the executor adds to direct
    launches. Every configuration's result must equal PyTorch's, and pandas'
    wherever the record compared it."""
    rows = []
    # each query's scale factor, name, and time through the executor and direct
    added: list[tuple[float, str, float, float]] = []
    for directory in TORCH_QUERIES:
        records = runs(directory)
        scales = {number(r["scale"], "scale") for r in records}
        if len(scales) != 1:
            raise ValueError(f"{directory}: the runs ran at different scale factors")
        (scale,) = scales
        for q, query in enumerate(items(records[0]["queries"], "queries")):
            name = text(query["query"], "query")
            for record in records:
                found = obj(field(record, "queries", q), name)
                if found["query"] != name or found["mismatches"]:
                    raise ValueError(f"{directory}: {name} has a mismatch")
                compared = [obj(found["equals_torch"], "equals_torch")]
                if found["equals_reference"] is not None:
                    compared.append(obj(found["equals_reference"], "equals_reference"))
                if any(
                    c[configuration] is not True
                    for c in compared
                    for configuration in TORCH_CONFIGURATIONS
                ):
                    raise ValueError(f"{directory}: {name} differs from its reference")
            timings = {
                c: across(records, "queries", q, "measured", c, "time")
                for c in TORCH_CONFIGURATIONS
            }
            added.append((scale, name, timings["opgate"].median, timings["direct"].median))
            rows.append(
                (f"{scale:g}", name, *(v for t in timings.values() for v in t.columns(1e3, 3)))
            )
    write(
        "torch_queries.csv",
        (
            "scale",
            "query",
            *(f"{c}_ms{s}" for c in TORCH_CONFIGURATIONS for s in ("", "_lo", "_hi")),
        ),
        rows,
    )
    milliseconds = [(opgate - direct) / 1e3 for _, _, opgate, direct in added]
    percent = [100 * (opgate / direct - 1) for _, _, opgate, direct in added]
    most_ms = max(range(len(added)), key=lambda i: milliseconds[i])
    most_percent = max(range(len(added)), key=lambda i: percent[i])
    return {
        "valTorchAddLowMs": f"{min(milliseconds):.2f}",
        "valTorchAddHighMs": f"{max(milliseconds):.2f}",
        "valTorchAddLowPercent": f"{min(percent):.1f}",
        "valTorchAddHighPercent": f"{max(percent):.0f}",
        # the query of the largest added time, and the query of the largest share
        "valTorchAddHighMsQuery": added[most_ms][1],
        "valTorchAddHighMsScale": f"{added[most_ms][0]:g}",
        "valTorchAddHighMsPercent": f"{percent[most_ms]:.1f}",
        "valTorchAddHighPercentQuery": added[most_percent][1],
        "valTorchAddHighPercentScale": f"{added[most_percent][0]:g}",
        "valTorchAddHighPercentMs": f"{milliseconds[most_percent]:.2f}",
        "valTorchAddHighPercentDirectMs": f"{added[most_percent][3] / 1e3:.2f}",
    }


def export() -> Numbers:
    numbers = export_index_path()
    steps: list[int] = []
    for scale, name, directory in TPCH_QUERIES:
        found, counted = export_tpch_queries(scale, name, directory)
        numbers |= found
        steps += counted
    return (
        numbers
        | export_torch_queries()
        | {
            "valTpchStepsLow": f"{min(steps):,}",
            "valTpchStepsHigh": f"{max(steps):,}",
        }
    )
