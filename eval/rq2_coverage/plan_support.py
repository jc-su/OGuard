"""Aggregate extracted configuration decisions into represented query plans.

This reports static acceptance of extracted kernels, not complete-query
execution. A plan with every extracted configuration accepted can still have
omitted kernels, data-dependent variants, and trusted host construction. The
denominator includes only plans represented by at least one corpus PTX entry.
The current corpus, contracts, reviews and verifier sources must match the
complete registration record; stale records fail rather than mix generations.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.corpus.adapter import DATAKERNELBENCH, directory_of
from eval.export.corpus import Kernel, corpus

ROOT = Path(__file__).resolve().parents[2]
PLAN = re.compile(r"^(q\d{2})_(core_)?(cuda|triton)_.+$")


def summarize(kernels: list[Kernel]) -> list[dict]:
    """Keep plan, unique kernel, and extracted configuration denominators distinct."""
    groups: dict[tuple[str, str, str, str], list[Kernel]] = defaultdict(list)
    for kernel in kernels:
        if kernel.key[0] not in DATAKERNELBENCH.models:
            continue
        model, task, toolchain = kernel.key
        matched = PLAN.fullmatch(task)
        if matched is None:
            raise ValueError(f"unrecognized DataKernelBench task {task}")
        query, core, compiler = matched.groups()
        expected = "nvcc" if compiler == "cuda" else "triton"
        if toolchain != expected:
            raise ValueError(f"{kernel.key}: query plan/toolchain mismatch")
        groups[(model, "core" if core else "full", query, toolchain)].append(kernel)
    plans = []
    for (model, level, query, toolchain), entries in sorted(groups.items()):
        sources: dict[str, list[Kernel]] = defaultdict(list)
        for kernel in entries:
            sources[kernel.source].append(kernel)
        plans.append(
            {
                "model": model,
                "level": level,
                "query": query,
                "toolchain": toolchain,
                "configuration_entries": len(entries),
                "unique_extracted_kernels": len(sources),
                "all_configurations_accepted": all(k.final == "accepted" for k in entries),
                "all_entries_reviewed_safe": all(k.cls == "safe" for k in entries),
                "accepted_configurations": sum(k.final == "accepted" for k in entries),
                "refused_configurations": sum(k.final == "refused" for k in entries),
                "timeout_configurations": sum(k.final == "timeout" for k in entries),
                "kernels": [
                    {
                        "source": source,
                        "all_configurations_accepted": all(k.final == "accepted" for k in variants),
                        "configurations": [
                            {
                                "key": list(k.key),
                                "class": k.cls,
                                "decision": k.final,
                                "reason": k.check,
                                "seconds": k.seconds,
                            }
                            for k in variants
                        ],
                    }
                    for source, variants in sorted(sources.items())
                ],
            }
        )
    return plans


def totals(plans: list[dict]) -> dict:
    return {
        level: {
            "represented_query_plans": len(selected := [p for p in plans if p["level"] == level]),
            "all_extracted_configurations_accepted": sum(
                p["all_configurations_accepted"] for p in selected
            ),
            "all_entries_reviewed_safe": sum(p["all_entries_reviewed_safe"] for p in selected),
            "all_safe_plans_with_all_configurations_accepted": sum(
                p["all_entries_reviewed_safe"] and p["all_configurations_accepted"]
                for p in selected
            ),
            "unique_extracted_kernels": sum(p["unique_extracted_kernels"] for p in selected),
            "configuration_entries": sum(p["configuration_entries"] for p in selected),
            "distinct_query_numbers": sorted({p["query"] for p in selected}),
        }
        for level in ("full", "core")
    }


def report(registration: Path, output: Path) -> None:
    kernels = corpus(registration)  # Validates raw PTX, query, binding, review and source bytes.
    plans = summarize(kernels)
    files = {"registration.json": registration, "reporter.py": Path(__file__)}
    for name in (
        "eval/export/corpus.py",
        "eval/export/records.py",
        "eval/checkpoint.py",
        "eval/corpus/adapter.py",
        "eval/corpus/run.py",
        "eval/rq4_cost/registration.py",
        "tests/eval/test_plan_support.py",
    ):
        files[name] = ROOT / name
    registration_inputs = registration.with_suffix(registration.suffix + ".inputs")
    registration_artifacts = registration.with_suffix(registration.suffix + ".artifacts")
    for prefix, directory in (
        ("registration_inputs", registration_inputs),
        ("registration_artifacts", registration_artifacts),
    ):
        for path in directory.rglob("*"):
            if path.is_file():
                files[f"{prefix}/{path.relative_to(directory)}"] = path
    for plan in plans:
        first = None
        for kernel in plan["kernels"]:
            for row in kernel["configurations"]:
                model, task, _ = row["key"]
                path = directory_of(model, task) / "plan.py"
                raw = path.read_bytes()
                if first is not None and raw != first:
                    raise ValueError(
                        f"{model}/{plan['level']}/{plan['query']}: inconsistent plan.py copies"
                    )
                first = raw
                files[str(path.relative_to(ROOT))] = path
    for model in DATAKERNELBENCH.models:
        notes = DATAKERNELBENCH.root / model / "NOTES.md"
        files[str(notes.relative_to(ROOT))] = notes
    readme = DATAKERNELBENCH.root / "README.md"
    files[str(readme.relative_to(ROOT))] = readme
    with Checkpoint(
        output,
        {
            "scope": "current-source static acceptance of represented extracted query plans",
            "registration": str(registration),
            "limitation": "not end-to-end query acceptance; omitted and zero-kernel plans excluded; retained model notes describe extraction omissions",
        },
    ) as record:
        snapshot_inputs(output, "sources", files, create=True)
        for index, plan in enumerate(plans):
            record.add({"id": f"plan-{index}", **plan})
        # Recheck after capture so concurrent verifier/corpus changes cannot be mixed.
        if corpus(registration) != kernels:
            raise ValueError("corpus changed while summarizing represented plans")
        snapshot_inputs(output, "sources", files, create=False)
        record.finish(totals=totals(plans))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.registration, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
