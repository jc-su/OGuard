# TPC-H queries 1 to 22 on cuDF through the catalog

Status 2026-09-29: the generator, the queries, the runner, and the test are
done, and all 22 queries give the reference's results at scale factors 0.01
and 0.1 in both cuDF configurations. The timed record,
`records/tpch_queries_<date>.json`, is written later on an idle GPU by the
command below; until then this document holds no time.

RQ3 asks whether a query engine runs complete queries with OpGate installed,
and RQ4 what that costs. The pipelines (`README.md`, not included in this archive) are three derived
queries with model-written operators. This experiment runs the 22 TPC-H
queries on cuDF with every device allocation of each query, and of its
tables, made through `CatalogResource`, so that the catalog checks each one,
and compares each result with the same query on pandas and on cuDF alone.

## Data

[`tpch/database.py`](../tpch/database.py) generates the eight tables at a scale
factor with NumPy's PCG64 from a seed, with the value domains of the TPC-H
specification, clause 4.2.3: the table sizes; the sparse order keys; the four
suppliers of each part by the specification's formula, and a line's supplier
among its part's four; an order's customer uniform among the customers whose
key is not a multiple of 3, so a third of the customers have no order; the
numeric and date domains; the return flag, line status, and order status from
the dates; and the word lists of the part names, types, containers, brands,
manufacturers, segments, priorities, ship instructions, ship modes, nations,
and regions, with phone numbers whose country code is the nation key plus 10.
Dates are int32 days since 1970-01-01, as in [`tpch/data.py`](../tpch/data.py).

It is not dbgen, and its rows differ from dbgen's rows:

- Comments have the specification's lengths but are substrings of a pool of
  random words, not text of the specification's grammar. The pool holds none
  of the words that queries search for. `Customer ... Complaints` is written
  into the comments of 5 suppliers per unit of scale factor, rounded up so
  that every database has one, and `Customer ... Recommends` into as many
  others (Q16). `special ... requests` is written into 1.07 percent of the
  order comments, the rate of dbgen's data at scale factor 1, whose Q13
  answer counts 1,483,918 of the 1,500,000 orders (Q13).
- dbgen moves a customer key that is a multiple of 3 to a neighbor, which
  gives Q13's answer two modes; a uniform draw among the other keys, which
  the specification allows, gives one.

The runner's seed is 154. With it every query gives at least one row, Q8 a
positive share in both years, and Q6, Q14, Q17, and Q19 a positive value at
scale factors 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, and 1. At
the small scale factors few rows qualify for Q8, Q11, and Q17 to Q20, and
other seeds give some of them no row: Q11 keeps the parts whose value exceeds
0.0001 / SF of Germany's total, and with 17 German suppliers at scale factor
0.03, one seed gave no part above it.

## Queries

[`tpch/queries.py`](../tpch/queries.py) holds the 22 queries with the
validation parameters of clause 2.4; Q11's fraction is 0.0001 divided by the
scale factor. Each is written once as a function of the library module,
`pandas` or `cudf`, and the tables of that library, and returns the result
columns in the specification's order, sorted by its ORDER BY and cut by its
LIMIT. Each uses only operations that both libraries have with the same
semantics: column selection, masks, arithmetic, `isin`, `where`, the string
methods `startswith`, `endswith`, `contains`, and `slice`, `merge`, group-by
with named `sum`, `mean`, `min`, `count`, and `nunique`, `sort_values`, and
`head`. Where the two libraries differ:

- Row order. pandas keeps the left frame's order in a join and the order of
  first appearance in a group-by; cuDF gives rows in the order of its hash
  tables. No query depends on the order of rows before its final sort. Where
  the ORDER BY leaves rows with equal keys unordered, the query adds the
  remaining key column as the last sort key: `l_orderkey` in Q3, `n_name` in
  Q5, `c_custkey` in Q10, `ps_partkey` in Q11, and `o_orderkey` in Q18. This
  fixes the order of tied rows and the rows that a LIMIT keeps.
- Missing values. A left join gives a missing key as NaN in a float column in
  pandas and as a null in an integer column in cuDF. Q13 counts the
  non-missing keys, which `count` does in both.
- Scalars. A reduction of a cuDF column is a NumPy scalar on the host, as in
  pandas. Q6, Q14, Q17, and Q19 build their one-row result frame from it with
  the library's constructor; Q11, Q15, and Q22 compare a column with it.

Further choices, each stated in the code: Q6 compares the discount with the
decimal bounds 0.05 and 0.07, since the float sum 0.06 + 0.01 is below the
float 0.07; Q13's `LIKE '%special%requests%'` is the regular expression
`special.*requests`; Q10 groups by the customer key alone, since the other
grouped columns are the customer's; Q19 keeps the specification's
`'AIR REG'`, which matches no line, since the ship mode is `REG AIR`; and
Q18 names its `sum(l_quantity)` column `sum_quantity`.

## Configurations

[`tpch_queries.py`](tpch_queries.py) generates the database and runs each
query in three configurations:

- `reference`: pandas on the host, once. Above scale factor 1 it is skipped,
  and the record says so; the two cuDF configurations are then compared with
  each other only.
- `cudf`: cuDF with an RMM pool of the catalog configuration's size as RMM's
  current resource, which the tables are loaded into with `cudf.from_pandas`.
- `catalog`: cuDF inside `install_cudf`, in a session of one tenant, with
  `CatalogResource` as RMM's current resource, as `tests/hosts/test_rapids.py`
  sets it up. The tables are loaded in the same session, so they are catalog
  buffers of the tenant in the executor's pool. The pool is 3 times the
  tables' bytes plus 256 MiB, rounded up to 2 MiB; at scale factors 0.01 and
  0.1 the most live bytes of any query is 1.47 times the tables' bytes (Q1).
  After the last query the tables are freed, and the run fails if any catalog
  buffer is still live.

The queries have no UDF, so no untrusted operator runs and the gate decides
nothing here: what the `catalog` configuration adds to `cudf` is the
allocation wrapper, a catalog step for each allocation that takes no region
the tenant kept and one step that frees the kept regions after the query, and
the clearing of allocated regions; both configurations allocate from an RMM pool
of the same size.

## What is compared and recorded

A result equals another if it has the same columns in the same order and the
same number of rows, and each value in the same row and column is equal: a
floating-point value within a relative tolerance of 1e-9, which covers the
two libraries' different orders of summation, and any other value exactly.
The index and the column types are not compared. The results are not
compared with TPC-H's answer sets, which are for dbgen's data.

Per query the record holds the result's rows; whether each cuDF result equals
the reference (null above scale factor 1) and whether the two equal each
other, with the first difference if one does not; and the host time of each
cuDF configuration from the start of the query to its result in host memory,
the median, minimum, and maximum of 20 runs after a warm-up of at least one
second, as [`cost.py`](../rq4_cost/step_check.py) measures. From one more run in the `catalog`
configuration, after cuDF's query caches are released, it records the steps
that the catalog accepted, split into allocations and frees; the regions it
cleared and their bytes; and the most live buffers and bytes in the pool,
the tables' included. The catalog keeps a region's contents only if one freed
region of the tenant contains all of it, and it does not merge adjacent freed
regions, so an allocation that RMM's pool carves from two adjacent freed
blocks is cleared although the one tenant owned both. The record also holds
the pinned core, the scale factor, the seed, each table's rows, the tables'
bytes and their catalog buffers, and the pool's bytes. It is written before a
difference makes the command fail.

## Correctness so far

On 2026-09-29, with seed 154, all 22 queries gave the reference's result on
cuDF alone and through the catalog, at scale factor 0.01
(`tests/pipelines/test_tpch_queries.py` and a check run) and at 0.1 (a check
run and a smoke run of the command). The rows of each result:

| Scale factor | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Q12 | Q13 | Q14 | Q15 | Q16 | Q17 | Q18 | Q19 | Q20 | Q21 | Q22 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 4 | 2 | 10 | 5 | 5 | 1 | 4 | 2 | 174 | 20 | 2 | 2 | 25 | 1 | 1 | 268 | 1 | 2 | 1 | 2 | 4 | 7 |
| 0.1 | 4 | 58 | 10 | 5 | 5 | 1 | 4 | 2 | 175 | 20 | 208 | 2 | 28 | 1 | 1 | 2,863 | 1 | 8 | 1 | 11 | 48 | 7 |

## Times at scale factor 1

Recorded 2026-09-29 by `eval/rq3_pipelines/tpch_queries.py --scale 1`, pinned to core 6 of
the GPU's NUMA node, in
`tpch_queries_2026-09-29.json` (`../records/tpch_queries_2026-09-29.json`, not included in this archive).
Every result equals pandas' result. Median host times, and the catalog's
steps and clears of one run:

| Query | cuDF | Through the catalog | Steps | Clears | Cleared |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 21.0 ms | 31.3 ms | 299 | 48 | 786.6 MiB |
| Q2 | 20.5 ms | 81.0 ms | 1,053 | 134 | 13.0 MiB |
| Q3 | 11.3 ms | 26.8 ms | 332 | 55 | 130.6 MiB |
| Q4 | 7.3 ms | 15.8 ms | 186 | 38 | 140.7 MiB |
| Q5 | 14.9 ms | 45.8 ms | 560 | 99 | 17.8 MiB |
| Q6 | 4.9 ms | 10.7 ms | 122 | 23 | 44.2 MiB |
| Q7 | 20.1 ms | 69.4 ms | 790 | 126 | 73.6 MiB |
| Q8 | 21.6 ms | 70.3 ms | 822 | 149 | 17.3 MiB |
| Q9 | 20.1 ms | 64.4 ms | 742 | 123 | 71.7 MiB |
| Q10 | 16.7 ms | 49.0 ms | 600 | 86 | 132.3 MiB |
| Q11 | 8.1 ms | 27.7 ms | 308 | 53 | 1.9 MiB |
| Q12 | 9.5 ms | 28.1 ms | 286 | 56 | 42.6 MiB |
| Q13 | 10.5 ms | 24.1 ms | 258 | 53 | 174.5 MiB |
| Q14 | 6.8 ms | 20.2 ms | 250 | 40 | 37.8 MiB |
| Q15 | 9.1 ms | 26.6 ms | 322 | 57 | 14.6 MiB |
| Q16 | 17.6 ms | 38.4 ms | 358 | 60 | 28.1 MiB |
| Q17 | 10.1 ms | 30.1 ms | 358 | 57 | 1.3 MiB |
| Q18 | 13.0 ms | 29.4 ms | 336 | 63 | 126.1 MiB |
| Q19 | 18.8 ms | 47.4 ms | 572 | 74 | 43.9 MiB |
| Q20 | 15.4 ms | 44.1 ms | 480 | 86 | 73.3 MiB |
| Q21 | 40.5 ms | 82.4 ms | 846 | 160 | 577.3 MiB |
| Q22 | 7.8 ms | 26.6 ms | 322 | 53 | 16.7 MiB |

The 22 queries take 325 ms on cuDF alone and 890 ms through the catalog: 1.5
to 3.9 times as long, a median of 2.9. The difference divided by the steps is
34 to 65 us per step, a median of 55 us: each allocation and each free that
cuDF makes is a step that the Python catalog checks and records, and a query
makes 122 to 1,053 of them. Part of the rest is clearing: with one tenant, an
allocation is cleared when no single freed region of the tenant contains it,
because its bytes were never allocated or because it spans adjacent freed
regions (Q1 clears 787 MiB per run). A native catalog and freed regions that
merge would reduce both; neither changes what the catalog checks.

## Running

```sh
taskset -c <core> uv run python -m eval.rq3_pipelines.tpch_queries --scale S \
  --output eval/records/tpch_queries_<date>.json
uv run pytest tests/pipelines/test_tpch_queries.py -m 'gpu and slow'
```

The command runs on an otherwise idle GPU, pinned to one core of the GPU's
NUMA node, as the other timing scripts do (README (`../README.md`, not included in this archive)). Generating
the tables takes about 14 seconds at scale factor 1.

## What this does not show

- No untrusted operator runs; a query whose operators the gate decides is a
  pipeline of RQ3 (`README.md`, not included in this archive) or the typed Q6 of [The service-boundary experiment](../rq1_safety/service_boundary.md).
- One tenant runs; isolation between tenants in one pool is shown by
  `tests/hosts/test_rapids.py` and the service-boundary experiment.
- The difference between `cudf` and `catalog` is the cost of mediating a
  whole engine's allocations with a Python catalog, one step per allocation
  and per free; a query of our pipelines, whose buffers the planner
  allocates, has a few steps instead.
- pandas is the reference of the results, not a baseline of the times.
