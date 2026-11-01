"""The exporter: the evaluation's records to the CSV files, tables, and numbers that the paper reads.

    uv run python -m eval.export

reads the records that each module names (`rq1`, `rq2`, `rq3`, `rq4`, and
`implementation`), writes one CSV file per figure panel to
`paper/figures/data/`, each generated table to `paper/figures/tab_*.tex`,
and every number that the paper's text cites to `paper/figures/numbers.tex`,
as a LaTeX macro. The figures are pgfplots sources in `paper/figures/` that
read the CSV files. The exporter fails when a record is missing, has another
shape, or has another number of runs than its experiment states.
"""
