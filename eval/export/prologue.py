"""The launch model's recognition of the zeroing prologue on the accepted corpus
entries (design decision 50): how many accepted entries have shared variables,
that the recognizer accepts every one, and how many declare local variables,
the entries whose launch rule takes zeroed local variables as its premise; and
the check that the verified prologue is in the SASS that the driver's JIT
compiles from each text that the service deploys with a prologue: those
entries, accepted as written, and the entries accepted with checks."""

import re
from pathlib import Path

from eval.checkpoint import read_artifact, snapshot_inputs
from eval.export.corpus import REGISTRATION
from eval.export.records import Numbers, items, load, number, obj, text
from eval.export.rq2 import GUARDED
from eval.rq1_safety.sass_prologue import JIT_LIBRARY
from eval.rq1_safety.sass_prologue import KIND as SASS_KIND
from eval.rq1_safety.sass_prologue import _sources as sass_sources
from eval.rq2_coverage.prologue_recognizer import KIND
from opgate.gate import formal_dir, zeroed
from opgate.hosts.service.audit import parse_binding

PROLOGUE = REGISTRATION.parent / "prologue_recognizer_2026-10-24c.json"
SASS = REGISTRATION.parent / "sass_prologue_2026-10-24d.json"  # rq1_safety/sass_prologue.py
# the driver release in the name of the JIT compiler library
JIT_RELEASE = re.compile(rf"/{re.escape(JIT_LIBRARY)}\.(\d+(?:\.\d+)+)$")


def export_recognizer() -> tuple[Numbers, list[str]]:
    """The counts of the recognizer record, which must be complete, measured on
    the export's registration record, and evaluated with the current library;
    and the entries that it lists with shared or local variables."""
    record = load(PROLOGUE)
    provenance = obj(record["provenance"], "provenance")
    if text(provenance["kind"], "kind") != KIND:
        raise ValueError(f"{PROLOGUE}: a record of another experiment")
    if record["complete"] is not True:
        raise ValueError(f"{PROLOGUE}: the record is incomplete")
    measured = Path(text(provenance["registration"], "registration")).name
    if measured != REGISTRATION.name:
        raise ValueError(
            f"{PROLOGUE}: measured on {measured}, and the export reads {REGISTRATION.name}"
        )
    if text(provenance["library"], "library") != str(formal_dir()):
        raise ValueError(f"{PROLOGUE}: evaluated with another library than the current one")
    summary = obj(record["summary"], "summary")
    kernels = int(number(summary["kernels"], "kernels"))
    shared = int(number(summary["with_shared_variables"], "with_shared_variables"))
    recognized = int(number(summary["recognized"], "recognized"))
    local = int(number(summary["with_local_variables"], "with_local_variables"))
    if recognized != shared:
        raise ValueError(f"{PROLOGUE}: {shared - recognized} prologues are unrecognized")
    listed = sorted(
        text(row["id"], "id")
        for row in items(record["results"], "results")
        if row["shared"] is True or row["local"] is True
    )
    return {
        "valPrologueKernels": f"{kernels:,}",
        "valPrologueShared": f"{shared:,}",
        "valPrologueLocal": f"{local:,}",
    }, listed


def _with_checks() -> list[str]:
    """The guarded record's entries that the gate accepts with checks and to whose
    final text `zeroed` gives a prologue."""
    listed = []
    for row in items(load(GUARDED)["results"], "results"):
        if row["first"] == "refused" and row["decision"] == "accepted":
            key = text(row["id"], "id")
            final = read_artifact(GUARDED, key + "/final", "ptx")
            if zeroed(final, parse_binding(row["binding"])) != final:
                listed.append(key)
    return sorted(listed)


def export_sass(listed: list[str]) -> Numbers:
    """The SASS check of every text that the service deploys with a zeroing
    prologue: the record must be complete, of the export's registration and
    guarded records, with the current sources; its entries accepted as written
    must be those that the recognizer record lists with shared or local
    variables, and its entries accepted with checks those of the guarded record
    to whose final text `zeroed` gives a prologue. Its summary's counts must be
    those of its rows, in which every entry passes, and one JIT compiler
    library, whose name holds the driver release, must have compiled every
    text."""
    record = load(SASS)
    provenance = obj(record["provenance"], "provenance")
    if text(provenance["kind"], "kind") != SASS_KIND or record["complete"] is not True:
        raise ValueError(f"{SASS}: not a complete record of the SASS prologue check")
    for key, expected in (("registration", REGISTRATION), ("guarded", GUARDED)):
        measured = Path(text(provenance[key], key)).name
        if measured != expected.name:
            raise ValueError(f"{SASS}: measured on {measured}, and the export reads {expected}")
    snapshot_inputs(SASS, "sources", sass_sources(), create=False)
    rows = items(record["results"], "results")
    groups = {
        group: sorted(text(row["id"], "id") for row in rows if row["group"] == group)
        for group in ("as_written", "with_checks")
    }
    if sum(len(ids) for ids in groups.values()) != len(rows):
        raise ValueError(f"{SASS}: a row of neither group")
    if groups["as_written"] != listed:
        raise ValueError(f"{SASS}: the entries accepted as written are not the recognizer's")
    if groups["with_checks"] != _with_checks():
        raise ValueError(f"{SASS}: the entries accepted with checks are not the guarded record's")
    failed = sorted(text(row["id"], "id") for row in rows if row["passed"] is not True)
    if failed:
        raise ValueError(f"{SASS}: the prologue fails in the SASS of {failed}")
    found = obj(record["summary"], "summary")
    if (
        int(number(found["kernels"], "kernels")) != len(rows)
        or int(number(found["as_written"], "as_written")) != len(groups["as_written"])
        or int(number(found["with_checks"], "with_checks")) != len(groups["with_checks"])
        or int(number(found["passed"], "passed")) != len(rows)
        or obj(found["failed"], "failed")
    ):
        raise ValueError(f"{SASS}: the summary's counts are not those of the rows")
    compilers = sorted({text(row["compiler"], "compiler") for row in rows})
    release = JIT_RELEASE.search(compilers[0]) if len(compilers) == 1 else None
    if release is None:
        raise ValueError(f"{SASS}: the texts were compiled by {compilers}")
    return {
        "valSassKernels": f"{len(rows):,}",
        "valSassWritten": f"{len(groups['as_written']):,}",
        "valSassChecked": f"{len(groups['with_checks']):,}",
        "valSassPassed": f"{len(rows) - len(failed):,}",
        "valSassDriverVersion": release.group(1),
    }


def export() -> Numbers:
    numbers, listed = export_recognizer()
    return numbers | export_sass(listed)
