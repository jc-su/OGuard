"""E2: index-valued intermediates in independently published TPC-H plans.

The audit pins DataKernelBench and checks all 22 validated cuDF/Torch TPC-H
plans plus their physical-plan JSON.  Each source witness names a value that
one GPU operation produces and a later selection, gather, or scatter consumes.
The GPU replay then exercises pylibcudf's public index-producing APIs and feeds
their returned columns to ``gather`` with bounds checks disabled.

The plan audit establishes source and relational-plan prevalence.  It is not a
runtime trace of all 22 queries.  The small GPU replay establishes that the
index columns are concrete device buffers on current cuDF/libcudf.
"""

import argparse
import json
import platform
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    import pylibcudf as plc
    from pylibcudf.gpumemoryview import gpumemoryview

RESULT_PREFIX = "E2_INDEX_RESULT="
UPSTREAM_URL = "https://github.com/kerneldf/datakernelbench"
UPSTREAM_COMMIT = "43c361d4ed789d1feaffbcee750e5b256590571a"
QUERIES = tuple(f"q{number:02d}" for number in range(1, 23))
SOURCE_ROOT = Path("artifacts/torchplans")
PLAN_ROOT = Path("benchmarks/tpch/json_plans/sf10")

Carrier = Literal["group_ids", "selection_vector", "gather_map"]
Visibility = Literal["explicit", "expression"]


class Witness(TypedDict):
    """One reviewed dataflow witness: the index carrier and its source fragments."""

    carrier: Carrier
    visibility: Visibility
    producer: str
    consumer: str


# One reviewed dataflow witness per query.  The audit requires both fragments
# in the pinned source and records their exact line numbers.  ``explicit``
# means that the index value has a Python name between producer and consumer;
# ``expression`` means eager evaluation materializes the boolean column within
# one indexing expression.
WITNESSES: dict[str, Witness] = {
    "q01": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_keys, inverse = torch.unique(group_key, return_inverse=True)",
        "consumer": "out.scatter_add_(0, inverse, vals.to(torch.float64))",
    },
    "q02": {
        "carrier": "selection_vector",
        "visibility": "explicit",
        "producer": 'region_mask = (df_region["r_name"] == region_name).fillna(False)',
        "consumer": 'eu_region_keys = df_region["r_regionkey"][region_mask].to_arrow().to_pylist()',
    },
    "q03": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)",
        "consumer": "revenue_sum.scatter_add_(0, inverse, rev)",
    },
    "q04": {
        "carrier": "selection_vector",
        "visibility": "explicit",
        "producer": "order_mask = (",
        "consumer": "filt_orderkey = o_orderkey[order_mask]",
    },
    "q05": {
        "carrier": "gather_map",
        "visibility": "explicit",
        "producer": "order = torch.argsort(out, descending=True)",
        "consumer": "return nation_ids[order], out[order]",
    },
    "q06": {
        "carrier": "selection_vector",
        "visibility": "explicit",
        "producer": "mask = (",
        "consumer": "return torch.sum(l_extendedprice[mask] * l_discount[mask])",
    },
    "q07": {
        "carrier": "selection_vector",
        "visibility": "explicit",
        "producer": (
            "mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days <= shipdate_hi_days)"
        ),
        "consumer": "supp = supp_nation_code[mask]",
    },
    "q08": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_years, inv = torch.unique(o_year, return_inverse=True)",
        "consumer": "sum_brazil.scatter_add_(0, inv, brazil_volume)",
    },
    "q09": {
        "carrier": "selection_vector",
        "visibility": "expression",
        "producer": 'df_part["p_name"].str.contains(pat)',
        "consumer": 'parts_filtered = df_part[df_part["p_name"].str.contains(pat)]',
    },
    "q10": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_keys, inverse = torch.unique(c_custkey, return_inverse=True)",
        "consumer": "revenue.scatter_add_(0, inverse, revenue_per_row)",
    },
    "q11": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_keys, inverse = torch.unique(ps_partkey, return_inverse=True)",
        "consumer": "grouped_values.scatter_add_(0, inverse, value)",
    },
    "q12": {
        "carrier": "selection_vector",
        "visibility": "explicit",
        "producer": "mask = (",
        "consumer": "codes = l_shipmode_code[mask]",
    },
    "q13": {
        "carrier": "gather_map",
        "visibility": "explicit",
        "producer": "order1 = torch.argsort(unique_counts, descending=True, stable=True)",
        "consumer": "unique_counts = unique_counts[order1]",
    },
    "q14": {
        "carrier": "selection_vector",
        "visibility": "explicit",
        "producer": (
            "mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days < shipdate_hi_days)"
        ),
        "consumer": "revenue = revenue[mask]",
    },
    "q15": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "pos = torch.searchsorted(sorted_s, suppkey_f)",
        "consumer": "totals_sorted.scatter_add_(0, pos, revenue_f)",
    },
    "q16": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_pairs = torch.unique(pairs, dim=0)",
        "consumer": "counts.scatter_add_(0, unique_pairs[:, 0], ones)",
    },
    "q17": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "sel_idx = idx_clamped[in_part]",
        "consumer": "sum_qty.scatter_add_(0, sel_idx, sel_qty)",
    },
    "q18": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_keys, inverse = torch.unique(l_orderkey, return_inverse=True)",
        "consumer": "sums.scatter_add_(0, inverse, l_quantity)",
    },
    "q19": {
        "carrier": "selection_vector",
        "visibility": "explicit",
        "producer": "mask = cond1 | cond2 | cond3",
        "consumer": "return torch.sum(revenue[mask])",
    },
    "q20": {
        "carrier": "group_ids",
        "visibility": "explicit",
        "producer": "unique_pairs, inverse = torch.unique(all_pairs, dim=0, return_inverse=True)",
        "consumer": "sum_qty.scatter_add_(0, li_group, l_quantity)",
    },
    "q21": {
        "carrier": "gather_map",
        "visibility": "explicit",
        "producer": "idx = torch.searchsorted(lo_orderkeys, l1_orderkey)",
        "consumer": "distinct_supp = lo_distinct_supp[idx]",
    },
    "q22": {
        "carrier": "gather_map",
        "visibility": "explicit",
        "producer": "idx_clamped = torch.clamp(idx, max=sorted_okeys.numel() - 1)",
        "consumer": "sorted_okeys[idx_clamped] == c_custkey",
    },
}

INDEX_PLAN_OPERATORS = {
    "HASH_JOIN",
    "LEFT_DELIM_JOIN",
    "NESTED_LOOP_JOIN",
    "ORDER_BY",
    "RIGHT_DELIM_JOIN",
    "TOP_N",
}


class SourceWitness(TypedDict):
    carrier: Carrier
    visibility: Visibility
    producer: str
    producer_line: int
    consumer: str
    consumer_line: int


class QueryRecord(TypedDict):
    query: str
    source: str
    plan: str
    source_witness: SourceWitness
    physical_plan_index_operators: dict[str, int]


class AuditProvenance(TypedDict):
    url: str
    commit: str


class AuditCounts(TypedDict):
    queries: int
    queries_with_source_dataflow_witness: int
    queries_with_join_or_order_physical_operator: int
    source_carriers: dict[Carrier, int]
    source_visibility: dict[Visibility, int]
    physical_plan_operators: dict[str, int]
    physical_index_operator_instances: int


class Audit(TypedDict):
    experiment: str
    success: bool
    provenance: AuditProvenance
    counts: AuditCounts
    queries: list[QueryRecord]
    scope: str


class IndexRecord(TypedDict):
    values: list[object]
    rows: int
    dtype: str
    pointer: int
    bytes: int
    range: list[int]
    range_valid: bool


class SelectionRecord(TypedDict):
    values: list[object]
    pointer: int
    bytes: int


class GatherCase(TypedDict):
    case: str
    producer: str
    consumer: str
    consumer_bounds_policy: str
    source_pointer: int
    index: IndexRecord
    output: list[list[object]]
    success: bool


class JoinCase(TypedDict):
    case: str
    producer: str
    consumer: str
    consumer_bounds_policy: str
    source_pointers: list[int]
    left_index: IndexRecord
    right_index: IndexRecord
    left_output: list[object]
    right_output: list[object]
    success: bool


class MaskCase(TypedDict):
    case: str
    producer: str
    consumer: str
    source_pointer: int
    selection: SelectionRecord
    output: list[list[object]]
    success: bool


Case = GatherCase | JoinCase | MaskCase


class Environment(TypedDict):
    python: str
    cudf: str
    pylibcudf: str
    rmm: str
    gpu: str
    compute_capability: list[int]


class Replay(TypedDict):
    experiment: str
    success: bool
    environment: Environment
    separate_source_and_index_buffers: bool
    cases: list[Case]
    meaning: str


class AllCases(TypedDict):
    experiment: str
    success: bool
    cases: list[Audit | Replay]
    claim: str
    limitation: str


def _git_head(upstream: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _line_number(text: str, fragment: str) -> int:
    if text.count(fragment) != 1:
        raise RuntimeError(
            f"expected one occurrence of witness {fragment!r}, found {text.count(fragment)}"
        )
    return text[: text.index(fragment)].count("\n") + 1


def _walk_plan(node: dict[str, object], counts: Counter[str]) -> None:
    name = node.get("name")
    if not isinstance(name, str):
        raise RuntimeError(f"plan node has no string name: {node}")
    counts[name] += 1
    children = node.get("children", [])
    if not isinstance(children, list):
        raise RuntimeError(f"plan node children are not a list: {node}")
    for child in children:
        if not isinstance(child, dict):
            raise RuntimeError(f"plan child is not an object: {child}")
        _walk_plan(child, counts)


def audit(upstream: Path) -> Audit:
    upstream = upstream.resolve()
    if not upstream.is_dir():
        raise FileNotFoundError(f"DataKernelBench checkout does not exist: {upstream}")
    commit = _git_head(upstream)
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"expected DataKernelBench {UPSTREAM_COMMIT}, found {commit}")
    if set(WITNESSES) != set(QUERIES):
        raise RuntimeError("the source witness table does not cover exactly q01 through q22")

    query_records: list[QueryRecord] = []
    total_plan_nodes: Counter[str] = Counter()
    source_carriers: Counter[Carrier] = Counter()
    visibility: Counter[Visibility] = Counter()
    plan_queries_with_index_operator = 0
    for query in QUERIES:
        source_path = upstream / SOURCE_ROOT / f"{query}.py"
        plan_path = upstream / PLAN_ROOT / f"{query}.json"
        if not source_path.is_file() or not plan_path.is_file():
            raise FileNotFoundError(f"missing source or plan for {query}")
        text = source_path.read_text()
        witness = WITNESSES[query]
        producer_line = _line_number(text, witness["producer"])
        consumer_line = _line_number(text, witness["consumer"])
        if producer_line > consumer_line:
            raise RuntimeError(f"{query}: consumer precedes producer")

        loaded: object = json.loads(plan_path.read_text())
        if not isinstance(loaded, list) or not loaded:
            raise RuntimeError(f"{plan_path}: expected a nonempty list of roots")
        plan_counts: Counter[str] = Counter()
        for root in loaded:
            if not isinstance(root, dict):
                raise RuntimeError(f"{plan_path}: plan root is not an object")
            _walk_plan(root, plan_counts)
        total_plan_nodes.update(plan_counts)
        index_plan_counts = {
            name: count
            for name, count in sorted(plan_counts.items())
            if name in INDEX_PLAN_OPERATORS
        }
        if index_plan_counts:
            plan_queries_with_index_operator += 1
        source_carriers[witness["carrier"]] += 1
        visibility[witness["visibility"]] += 1
        query_records.append(
            {
                "query": query,
                "source": str(source_path.relative_to(upstream)),
                "plan": str(plan_path.relative_to(upstream)),
                "source_witness": {
                    "carrier": witness["carrier"],
                    "visibility": witness["visibility"],
                    "producer": witness["producer"],
                    "producer_line": producer_line,
                    "consumer": witness["consumer"],
                    "consumer_line": consumer_line,
                },
                "physical_plan_index_operators": index_plan_counts,
            }
        )

    return {
        "experiment": "E2a: index-valued intermediate audit of TPC-H plans",
        "success": (
            len(query_records) == 22
            and sum(source_carriers.values()) == 22
            and plan_queries_with_index_operator == 21
        ),
        "provenance": {"url": UPSTREAM_URL, "commit": commit},
        "counts": {
            "queries": len(query_records),
            "queries_with_source_dataflow_witness": sum(source_carriers.values()),
            "queries_with_join_or_order_physical_operator": plan_queries_with_index_operator,
            "source_carriers": dict(sorted(source_carriers.items())),
            "source_visibility": dict(sorted(visibility.items())),
            "physical_plan_operators": dict(sorted(total_plan_nodes.items())),
            "physical_index_operator_instances": sum(
                count for name, count in total_plan_nodes.items() if name in INDEX_PLAN_OPERATORS
            ),
        },
        "queries": query_records,
        "scope": (
            "The source witnesses are static dataflow evidence in validated cuDF/Torch "
            "plans. The JSON plans establish relational shape; they are DuckDB plans, not "
            "a trace of libcudf launches."
        ),
    }


def _column_values(column: "plc.Column") -> list[object]:
    return column.to_pylist()


def _sorted_ints(values: list[object]) -> list[int]:
    ints: list[int] = []
    for value in values:
        if not isinstance(value, int):
            raise TypeError(f"expected an integer index value, found {value!r}")
        ints.append(value)
    return sorted(ints)


def _buffer(column: "plc.Column") -> "gpumemoryview":
    from pylibcudf.gpumemoryview import gpumemoryview

    data = column.data()
    if not isinstance(data, gpumemoryview):
        raise RuntimeError(f"the column has no device buffer: {data!r}")
    return data


def _column_record(column: "plc.Column", source_rows: int) -> IndexRecord:
    values = _column_values(column)
    data = _buffer(column)
    return {
        "values": values,
        "rows": column.size(),
        "dtype": column.type().id().name,
        "pointer": int(data.ptr),
        "bytes": int(data.nbytes),
        "range": [0, source_rows - 1],
        "range_valid": all(isinstance(value, int) and 0 <= value < source_rows for value in values),
    }


def _table_values(table: "plc.Table") -> list[list[object]]:
    return [_column_values(column) for column in table.columns()]


def gpu_replay() -> Replay:
    import cudf
    import pylibcudf as plc
    import rmm
    from numba import cuda

    held: list[cudf.Series] = []

    def column(values: list[int]) -> "plc.Column":
        series = cudf.Series(values, dtype="int64")
        held.append(series)
        return series._column.to_pylibcudf()

    policy = plc.copying.OutOfBoundsPolicy.DONT_CHECK

    sort_source = column([3, 1, 2])
    sort_table = plc.Table([sort_source])
    order = plc.sorting.sorted_order(
        sort_table,
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.BEFORE],
    )
    sort_output = plc.copying.gather(sort_table, order, policy)
    sort_map = _column_record(order, 3)
    sort_success = sort_map["values"] == [1, 2, 0] and _table_values(sort_output) == [[1, 2, 3]]
    sort_case: GatherCase = {
        "case": "sorted_order_to_gather",
        "producer": "pylibcudf.sorting.sorted_order",
        "consumer": "pylibcudf.copying.gather",
        "consumer_bounds_policy": "DONT_CHECK",
        "source_pointer": int(_buffer(sort_source).ptr),
        "index": sort_map,
        "output": _table_values(sort_output),
        "success": sort_success and sort_map["range_valid"],
    }

    top_source = column([4, 1, 4, 2, 3])
    top_table = plc.Table([top_source])
    top_order = plc.sorting.top_k_order(top_source, 2, plc.types.Order.DESCENDING)
    top_output = plc.copying.gather(top_table, top_order, policy)
    top_map = _column_record(top_order, 5)
    top_success = _sorted_ints(top_map["values"]) == [0, 2] and _table_values(top_output) == [
        [4, 4]
    ]
    top_case: GatherCase = {
        "case": "top_k_order_to_gather",
        "producer": "pylibcudf.sorting.top_k_order",
        "consumer": "pylibcudf.copying.gather",
        "consumer_bounds_policy": "DONT_CHECK",
        "source_pointer": int(_buffer(top_source).ptr),
        "index": top_map,
        "output": _table_values(top_output),
        "success": top_success and top_map["range_valid"],
    }

    distinct_source = column([4, 1, 4, 2, 3])
    distinct_table = plc.Table([distinct_source])
    distinct_indices = plc.stream_compaction.distinct_indices(
        distinct_table,
        plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        plc.types.NullEquality.EQUAL,
        plc.types.NanEquality.ALL_EQUAL,
    )
    distinct_output = plc.copying.gather(distinct_table, distinct_indices, policy)
    distinct_map = _column_record(distinct_indices, 5)
    distinct_case: GatherCase = {
        "case": "distinct_indices_to_gather",
        "producer": "pylibcudf.stream_compaction.distinct_indices",
        "consumer": "pylibcudf.copying.gather",
        "consumer_bounds_policy": "DONT_CHECK",
        "source_pointer": int(_buffer(distinct_source).ptr),
        "index": distinct_map,
        "output": _table_values(distinct_output),
        "success": (
            distinct_map["values"] == [0, 1, 3, 4]
            and distinct_map["range_valid"]
            and _table_values(distinct_output) == [[4, 1, 2, 3]]
        ),
    }

    left_source = column([1, 2, 3])
    right_source = column([3, 1, 4])
    left_table = plc.Table([left_source])
    right_table = plc.Table([right_source])
    left_map_column, right_map_column = plc.join.inner_join(
        left_table,
        right_table,
        plc.types.NullEquality.EQUAL,
    )
    left_output = plc.copying.gather(left_table, left_map_column, policy)
    right_output = plc.copying.gather(right_table, right_map_column, policy)
    left_map = _column_record(left_map_column, 3)
    right_map = _column_record(right_map_column, 3)
    left_values = _table_values(left_output)[0]
    right_values = _table_values(right_output)[0]
    join_case: JoinCase = {
        "case": "inner_join_maps_to_two_gathers",
        "producer": "pylibcudf.join.inner_join",
        "consumer": "pylibcudf.copying.gather",
        "consumer_bounds_policy": "DONT_CHECK",
        "source_pointers": [int(_buffer(left_source).ptr), int(_buffer(right_source).ptr)],
        "left_index": left_map,
        "right_index": right_map,
        "left_output": left_values,
        "right_output": right_values,
        "success": (
            left_map["range_valid"]
            and right_map["range_valid"]
            and left_values == right_values
            and _sorted_ints(left_values) == [1, 3]
        ),
    }

    mask_source_series = cudf.Series([3, 1, 2], dtype="int64")
    mask_series = mask_source_series > 1
    held.extend((mask_source_series, mask_series))
    mask_source = mask_source_series._column.to_pylibcudf()
    mask_column = mask_series._column.to_pylibcudf()
    mask_output = plc.stream_compaction.apply_boolean_mask(
        plc.Table([mask_source]),
        mask_column,
    )
    mask_buffer = _buffer(mask_column)
    mask_case: MaskCase = {
        "case": "cudf_predicate_to_boolean_mask_consumer",
        "producer": "cudf.Series.__gt__",
        "consumer": "pylibcudf.stream_compaction.apply_boolean_mask",
        "source_pointer": int(_buffer(mask_source).ptr),
        "selection": {
            "values": _column_values(mask_column),
            "pointer": int(mask_buffer.ptr),
            "bytes": int(mask_buffer.nbytes),
        },
        "output": _table_values(mask_output),
        "success": (
            _column_values(mask_column) == [True, False, True]
            and _table_values(mask_output) == [[3, 2]]
        ),
    }

    cuda.synchronize()
    gather_cases = [sort_case, top_case, distinct_case]
    separate_simple_buffers = all(
        case["source_pointer"] != case["index"]["pointer"] for case in gather_cases
    )
    join_index_pointers = {
        join_case["left_index"]["pointer"],
        join_case["right_index"]["pointer"],
    }
    separate_join_buffers = len(join_index_pointers) == 2 and join_index_pointers.isdisjoint(
        join_case["source_pointers"]
    )
    separate_mask_buffer = mask_case["source_pointer"] != mask_case["selection"]["pointer"]
    separate_buffers = separate_simple_buffers and separate_join_buffers and separate_mask_buffer
    cases: list[Case] = [*gather_cases, join_case, mask_case]
    return {
        "experiment": "E2b: concrete libcudf index producer-to-consumer replay",
        "success": all(case["success"] for case in cases) and separate_buffers,
        "environment": {
            "python": platform.python_version(),
            "cudf": cudf.__version__,
            "pylibcudf": version("pylibcudf-cu13"),
            "rmm": rmm.__version__,
            "gpu": cuda.get_current_device().name,
            "compute_capability": list(cuda.get_current_device().compute_capability),
        },
        "separate_source_and_index_buffers": separate_buffers,
        "cases": cases,
        "meaning": (
            "Current libcudf APIs return device-resident row-index columns from sort, "
            "top-k, distinct, and join operations. Later gather operations consume those "
            "columns with DONT_CHECK, so the producer's value range is a cross-operation "
            "safety fact. A cuDF predicate likewise materializes a boolean selection column."
        ),
    }


def all_cases(upstream: Path) -> AllCases:
    audited = audit(upstream)
    replayed = gpu_replay()
    return {
        "experiment": "E2: index-valued intermediates in TPC-H and libcudf",
        "success": audited["success"] and replayed["success"],
        "cases": [audited, replayed],
        "claim": (
            "All 22 pinned TPC-H implementations contain a reviewed source dataflow in "
            "which a selection, gather, join, or group index is produced and consumed. "
            "The current libcudf replay confirms the corresponding index-column API path."
        ),
        "limitation": (
            "This is a source audit plus representative API replay, not a launch-level "
            "profile of every TPC-H query."
        ),
    }


def _emit(result: Audit | Replay | AllCases) -> None:
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("all", "audit", "gpu"), nargs="?", default="all")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.case in ("all", "audit") and args.upstream is None:
        parser.error("all and audit require --upstream at the pinned DataKernelBench checkout")
    result: Audit | Replay | AllCases
    if args.case == "audit":
        result = audit(args.upstream)
    elif args.case == "gpu":
        result = gpu_replay()
    else:
        result = all_cases(args.upstream)
    _emit(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
