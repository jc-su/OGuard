"""Ablations replay measured inputs and never count timeout as rule necessity."""

import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from eval import ROOT
from eval.checkpoint import Checkpoint
from eval.corpus.repair import proof_files
from eval.rq2_coverage import rule_ablation as experiment
from eval.rq2_coverage.rule_ablation import BASELINE, GROUPS, RULES, Entry, Group, Outcome, Run

SOURCES = ROOT / "formal"


@pytest.mark.parametrize(
    ("stdout", "expected"), [(" = true\n : bool\n", "accepted"), (" = false\n : bool\n", "refused")]
)
def test_query_requires_a_conclusive_boolean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, expected: Outcome
) -> None:
    monkeypatch.setattr(
        experiment.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout)
    )
    assert experiment._accepts(tmp_path, "terms", 1) == expected


def test_missing_boolean_is_an_experiment_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="nothing"),
    )
    with pytest.raises(RuntimeError, match="no unique boolean"):
        experiment._accepts(tmp_path, "terms", 1)


def test_timeout_is_distinct_from_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("coqc", 1)

    monkeypatch.setattr(experiment.subprocess, "run", timeout)
    assert experiment._accepts(tmp_path, "terms", 1) == "timeout"
    library = tmp_path / "library"
    library.mkdir()
    (library / "Makefile").write_text("FILES := GateParse.v\n\n")
    (library / "GateParse.v").write_text("unchanged")

    def build(group: Group, directory: Path, library: Path) -> None:
        (directory / "GateParse.v").write_text(f"changed by {group.name}")

    outcomes: dict[str, Outcome] = {"safe": "accepted", "slow": "timeout", "lost": "refused"}
    monkeypatch.setattr(experiment, "_build", build)
    monkeypatch.setattr(experiment, "_accepts", lambda directory, terms, limit: outcomes[terms])
    entries = [Entry(f"model/{t}/numba", "development", "safe", True, t) for t in outcomes]
    output = tmp_path / "ablation.json"
    group = GROUPS[0]
    with Checkpoint(output, {"kind": experiment.KIND}) as record:
        experiment.ablate([group], entries, record, Run(library, 1, 2))
        row = record.get(group.name)
    assert row is not None
    assert row["decisions"] == {
        "model/lost/numba": "refused",
        "model/safe/numba": "accepted",
        "model/slow/numba": "timeout",
    }
    assert set(row["seconds"]) == set(row["decisions"])
    variant = output.with_suffix(".json.inputs") / "variants" / experiment.slug(group)
    assert (variant / "GateParse.v").read_text() == f"changed by {group.name}"


def test_every_rule_is_in_exactly_one_group() -> None:
    members = Counter(name for g in GROUPS for name in (*g.rules, *g.subsumed))
    assert set(members) == {r.name for r in RULES}
    assert set(members.values()) == {1}
    assert 12 <= len(GROUPS) <= 16


def test_each_group_makes_its_changes_in_order() -> None:
    experiment.validate_rules(RULES, SOURCES)
    experiment.validate_groups(GROUPS, SOURCES)
    for group in GROUPS:
        edited = experiment.variant_sources(group, SOURCES)
        for name in (*group.rules, *group.subsumed):
            rule = experiment.RULE_NAMES[name]
            assert rule.old not in edited[rule.file]


def test_a_subsumed_rule_must_be_replaced_by_a_change_of_its_group() -> None:
    group = Group("partial", "a rule that no change replaces", ("R4",), ("R7",))
    with pytest.raises(ValueError, match="no change of the group replaces R7"):
        experiment.variant_sources(group, SOURCES)
    with pytest.raises(ValueError, match="occurs 0 times"):
        experiment.variant_sources(Group("twice", "", ("R4", "R4")), SOURCES)
    with pytest.raises(ValueError, match="distinct"):
        experiment.validate_groups([GROUPS[0], GROUPS[0]], SOURCES)
    with pytest.raises(ValueError, match="distinct"):
        experiment.validate_groups([Group(BASELINE.name, "", ("R4",))], SOURCES)


def _entries() -> list[Entry]:
    entries = [Entry(f"m{i}/safe{i}/triton", "development", "safe", True, "") for i in range(5)]
    entries.append(Entry("m0/refused/triton", "development", "safe", False, ""))
    for tasks, toolchain, count in (
        ("development", "numba", 30),
        ("development", "triton", 10),
        ("held-out-5", "nvcc", 20),
    ):
        entries += [
            Entry(f"m{i % 3}/{tasks}{i}/{toolchain}", tasks, "E", False, "") for i in range(count)
        ]
    return entries


def test_defect_sample_is_seeded_and_stratified_by_set_and_toolchain() -> None:
    entries = _entries()
    safe, defects = experiment.select(entries, 12, 7)
    assert safe == sorted(f"m{i}/safe{i}/triton" for i in range(5))
    assert experiment.select(entries, 12, 7) == (safe, defects)
    assert experiment.select(entries, 12, 8)[1] != defects
    strata = Counter(
        (next(e.tasks for e in entries if e.id == d), d.split("/")[2]) for d in defects
    )
    assert strata == {
        ("development", "numba"): 6,
        ("development", "triton"): 2,
        ("held-out-5", "nvcc"): 4,
    }
    with pytest.raises(ValueError, match="sample of 1 to 60"):
        experiment.select(entries, 61, 7)
    accepted = [*entries, Entry("m0/wrong/nvcc", "development", "C", True, "")]
    with pytest.raises(ValueError, match="accepts a defect"):
        experiment.select(accepted, 12, 7)


def test_baseline_must_accept_every_safe_entry_and_no_defect() -> None:
    row: dict[str, object] = {"decisions": {"safe": "accepted", "wrong": "refused"}}
    experiment.validate_baseline(row, ["safe"], ["wrong"])
    with pytest.raises(RuntimeError, match="does not accept"):
        experiment.validate_baseline(
            {"decisions": {"safe": "timeout", "wrong": "refused"}}, ["safe"], ["wrong"]
        )
    with pytest.raises(RuntimeError, match="accepts a defect"):
        experiment.validate_baseline(
            {"decisions": {"safe": "accepted", "wrong": "accepted"}}, ["safe"], ["wrong"]
        )


def test_registration_must_have_retained_the_library_sources(tmp_path: Path) -> None:
    registration = tmp_path / "registration.json"
    retained = tmp_path / "registration.json.inputs" / "sources" / "formal"
    retained.mkdir(parents=True)
    (retained / "GateParse.v").write_text("registered")
    library = tmp_path / "library"
    library.mkdir()
    (library / "GateParse.v").write_text("registered")
    experiment.same_library(registration, library)
    (library / "GateParse.v").write_text("changed")
    with pytest.raises(ValueError, match="sources differ"):
        experiment.same_library(registration, library)


@pytest.mark.slow
def test_every_ablation_variant_builds_without_changing_the_trusted_library() -> None:
    library = experiment.formal_dir().resolve()
    experiment.validate_rules(RULES, library)
    experiment.validate_groups(GROUPS, library)
    before = {name: p.read_bytes() for name, p in proof_files(library).items()}

    def build(group: Group) -> None:
        with TemporaryDirectory() as tmp:
            experiment._build(group, Path(tmp), library)
            assert (Path(tmp) / "GateParse.vos").is_file()
            assert not (Path(tmp) / "GateParse.vo").exists()

    singles = [Group(r.name, r.description, (r.name,)) for r in RULES]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(build, [BASELINE, *singles, *GROUPS]))
    assert {name: p.read_bytes() for name, p in proof_files(library).items()} == before
