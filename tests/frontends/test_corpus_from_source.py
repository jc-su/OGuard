"""A sample of the corpus, compiled from source, keeps its acceptance and refuses its defects.

Each kernel of the sample (`eval.corpus.run.sample`) is compiled from its
source by its front end, numba-cuda at NVVM level 1, nvcc with `-Xcicc -O1`,
or Triton, and registered with the executor, which verifies it and loads it
when the gate accepts it. `make rq1-corpus-from-source` does this for the
whole corpus, which takes hours.
"""

import pytest

from eval.corpus.run import compiled, read_table, sample
from opgate.gate import Accepted


@pytest.mark.slow
@pytest.mark.gpu
def test_a_sample_compiled_from_source_keeps_its_acceptance_and_refuses_defects() -> None:
    table = read_table()
    chosen = sample(table)
    for key, decision in zip(chosen, compiled(chosen), strict=True):
        row = table[key]
        if row.expected == "accepted":
            assert isinstance(decision, Accepted), (key, decision)
        if row.verdict != "safe":
            assert not isinstance(decision, Accepted), key
