"""A sample of the corpus keeps the acceptance its tables record and refuses every defect.

The sample is, for each set of kernels and each toolchain, the first kernel
that its table records as accepted and the first defect (`eval.corpus.run.sample`).
`make rq1-corpus` decides the whole corpus, which takes hours. The tables
keep the decisions of the gate at the time each kernel entered them, so a
later rule may accept more safe kernels.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from eval.corpus.run import WORKERS, decide, read_table, sample
from opgate.gate import Accepted


@pytest.mark.slow
def test_a_sample_of_the_corpus_keeps_its_acceptance_and_refuses_defects() -> None:
    table = read_table()
    chosen = sample(table)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        decisions = list(pool.map(lambda key: decide(*key), chosen))
    for key, decision in zip(chosen, decisions, strict=True):
        row = table[key]
        if row.expected == "accepted":
            assert isinstance(decision, Accepted), (key, decision)
        if row.verdict != "safe":
            assert not isinstance(decision, Accepted), key
