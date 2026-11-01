"""Inputs and NumPy references for the corpus tasks, run through the executor.

For supported tasks of TASKS.md, HELDOUT.md, and HELDOUT2.md, `workload_of` builds inputs of a
number of rows that meet the task's contract, and `check` compares a
kernel's outputs with a NumPy reference. Each model hashes keys its own way, so the table that a
model's `hash_probe` probes comes from the model's own Numba `hash_build`,
and the reference is the join with the table as that kernel left it.
`prepare` allocates a buffer for each array argument of a loaded kernel,
writes each input (an index buffer with the range of its values, which the
catalog records as its status), and builds its launch with a value for each
size symbol and scalar, the launch of the kernel as written; `run` launches
it through the executor with `opgate.hosts.execute`, which gives a text with
the guard pass's checks its status buffer and raises
`opgate.hosts.CheckFailedError` if a check failed, and reads back each
output. The grid has a warp for each row, and for Triton a
program for each row; the gate's guarantee holds for every grid.
`corpus_kernel` loads a corpus kernel from its recorded PTX.
"""

import struct
from dataclasses import dataclass

import numpy as np

from eval.corpus import heldout_workloads
from eval.corpus.adapter import (
    ELEMENTS,
    Pointer,
    Points,
    Role,
    binding_of,
    directory_of,
    read_raw,
    read_roles,
)
from eval.corpus.run import TOOLCHAINS
from opgate.catalog import GRID_LIMIT, BufferId, Launch, Step, TenantId, Write
from opgate.executor import Executor, Module
from opgate.hosts import execute

ALPHABET = ("a", "b", "z", "é", "€", "😀")  # code points of 1 to 4 bytes in UTF-8
PREFIX = "ab"
PATTERN = "bz"
TILE = 1024  # the rows of a tile of tile_reverse and block_sort; the last tile can be shorter
BOUNDS = 64  # the bounds of bucket_histogram
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1
EPOCH_DAYS = (-719162, 2932896)  # the days of 0001-01-01 and 9999-12-31 from 1970-01-01


@dataclass(frozen=True)
class Workload:
    """The value of each argument and each size symbol, by name."""

    arrays: dict[str, np.ndarray]
    scalars: dict[str, int | float]
    sizes: dict[str, int]


def _power_of_two(at_least: int) -> int:
    return 1 << max(1, (at_least - 1).bit_length())


def _strings(n: int, rng: np.random.Generator) -> list[bytes]:
    texts = []
    for _ in range(n):
        body = "".join(rng.choice(ALPHABET, rng.integers(0, 11)))
        texts.append(((PREFIX if rng.random() < 0.3 else "") + body).encode("utf-8"))
    return texts


def _string_column(texts: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    lengths = [len(t) for t in texts]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    chars = np.frombuffer(b"".join(texts), dtype=np.uint8).copy()
    return offsets, chars


def _bloom_positions(keys: np.ndarray, words: int) -> tuple[np.ndarray, np.ndarray]:
    """HELDOUT2's two hashes, with unsigned 64-bit wrapping before the shift."""
    if not 1 <= words <= 2**26 or words & (words - 1):
        raise ValueError("Bloom words must be a power of two in [1, 2**26]")
    unsigned = keys.astype(np.uint64)
    return tuple(
        ((unsigned * np.uint64(multiplier)) >> np.uint64(32)) % np.uint64(32 * words)
        for multiplier in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F)
    )


def _bloom_bits(keys: np.ndarray, words: int) -> np.ndarray:
    """Build the specified filter on the CPU, including repeated bit positions."""
    positions = _bloom_positions(keys, words)
    bits = np.zeros(words, dtype=np.uint32)
    for position in positions:
        masks = np.uint32(1) << (position % np.uint64(32)).astype(np.uint32)
        np.bitwise_or.at(bits, position // np.uint64(32), masks)
    return bits


def workload(task: str, n: int, rng: np.random.Generator) -> Workload:
    """Inputs of `n` rows that meet the contract of `task`, with each output's
    initial contents, for a task whose inputs no kernel of the model gives."""
    if task in heldout_workloads.TASKS:
        return Workload(*heldout_workloads.inputs(task, n, rng))
    match task:
        case "filter_select":
            return Workload(
                {
                    "quantity": rng.uniform(1, 50, n),
                    "sel": np.zeros(n, dtype=np.int32),
                    "count": np.zeros(1, dtype=np.int32),
                },
                {"threshold": 24.0},
                {"n": n},
            )
        case "scatter_by_offsets":
            # the first row is flagged, so the output has an element
            flags = (rng.random(n) < 0.4).astype(np.int8)
            flags[0] = 1
            total = int(flags.sum())
            return Workload(
                {
                    "flags": flags,
                    "offsets": (np.cumsum(flags) - flags).astype(np.int32),
                    "out": np.zeros(total, dtype=np.int32),
                },
                {},
                {"n": n, "total": total},
            )
        case "gather":
            return Workload(
                {
                    "col": rng.uniform(0, 1, n),
                    "sel": rng.integers(0, n, n).astype(np.int32),
                    "out": np.zeros(n),
                },
                {},
                {"n": n, "m": n},
            )
        case "projection":
            return Workload(
                {
                    "price": rng.uniform(900, 105000, n),
                    "discount": rng.uniform(0, 0.1, n),
                    "out": np.zeros(n),
                },
                {},
                {"n": n},
            )
        case "q6_sum":
            return Workload(
                {
                    "shipdate": rng.integers(8000, 10000, n).astype(np.int32),
                    "discount": rng.uniform(0, 0.1, n),
                    "quantity": rng.uniform(1, 50, n),
                    "price": rng.uniform(900, 105000, n),
                    "result": np.zeros(1),
                },
                {
                    "date_lo": 8766,
                    "date_hi": 9131,
                    "disc_lo": 0.05,
                    "disc_hi": 0.07,
                    "qty_max": 24.0,
                },
                {"n": n},
            )
        case "hash_build":
            capacity = _power_of_two(2 * n)
            return Workload(
                {
                    "keys": rng.permutation(10 * n)[:n].astype(np.int64) * 7 + 3,
                    "table_keys": np.full(capacity, -1, dtype=np.int64),
                    "table_rows": np.zeros(capacity, dtype=np.int32),
                },
                {},
                {"n": n, "capacity": capacity},
            )
        case "bloom_build" | "bloom_probe":
            words = min(2**26, _power_of_two(max(1, n // 4)))
            keys = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
            edges = np.array([0, -1, -(2**63), 2**63 - 1, 1, -2, 0, -1], dtype=np.int64)
            keys[: min(n, len(edges))] = edges[:n]
            bits = (
                np.zeros(words, dtype=np.uint32)
                if task == "bloom_build"
                else _bloom_bits(keys[::2], words)
            )
            arrays = {"keys": keys, "bits": bits}
            if task == "bloom_probe":
                arrays["match"] = np.zeros(n, dtype=np.int8)
            return Workload(arrays, {}, {"n": n, "words": words})
        case "groupby_sum":
            groups = max(1, n // 10)
            capacity = _power_of_two(2 * groups)
            return Workload(
                {
                    "group": rng.integers(0, groups, n).astype(np.int32),
                    "value": rng.uniform(0, 1, n),
                    "table_keys": np.full(capacity, -1, dtype=np.int32),
                    "table_sums": np.zeros(capacity),
                },
                {},
                {"n": n, "capacity": capacity},
            )
        case "histogram":
            return Workload(
                {
                    "keys": rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32),
                    "counts": np.zeros(256, dtype=np.int32),
                },
                {"shift": 8},
                {"n": n},
            )
        case "segmented_sum":
            lengths = rng.integers(0, 9, n)
            offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
            values = int(offsets[-1])
            return Workload(
                {"offsets": offsets, "values": rng.uniform(0, 1, values), "sums": np.zeros(n)},
                {},
                {"g": n, "values": values},
            )
        case "string_equal":
            left = _strings(n, rng)
            right = [
                text
                if i % 3 == 0
                else text + b"x"
                if i % 3 == 1
                else text.decode("utf-8")[::-1].encode("utf-8")
                for i, text in enumerate(left)
            ]
            # Include empty, unequal-length, equal-length mismatch, and UTF-8
            # pairs independently of the random sample's length distribution.
            for i, (a, b) in enumerate(
                [(b"", b""), (b"", b"x"), (b"abc", b"abd"), ("é😀".encode(), "é😀".encode())]
            ):
                if i < n:
                    left[i], right[i] = a, b
            a_offsets, a_chars = _string_column(left)
            b_offsets, b_chars = _string_column(right)
            return Workload(
                {
                    "a_offsets": a_offsets,
                    "a_chars": a_chars,
                    "b_offsets": b_offsets,
                    "b_chars": b_chars,
                    "match": np.zeros(n, dtype=np.int8),
                },
                {},
                {"n": n, "a_chars": len(a_chars), "b_chars": len(b_chars)},
            )
        case "starts_with" | "string_length":
            offsets, chars = _string_column(_strings(n, rng))
            prefix = np.frombuffer(PREFIX.encode("utf-8"), dtype=np.uint8).copy()
            return Workload(
                {
                    "offsets": offsets,
                    "chars": chars,
                    "prefix": prefix,
                    "match": np.zeros(n, dtype=np.int8),
                    "lengths": np.zeros(n, dtype=np.int32),
                },
                {},
                {"n": n, "chars": len(chars), "k": len(prefix)},
            )
        case "valid_rows":
            valid = np.packbits(rng.random(n) < 0.6, bitorder="little")
            return Workload(
                {
                    "valid": valid,
                    "rows": np.zeros(n, dtype=np.int32),
                    "count": np.zeros(1, dtype=np.int32),
                },
                {},
                {"n": n, "valid": len(valid)},
            )
        case "dictionary_decode":
            d = max(1, n // 10)
            return Workload(
                {
                    "codes": rng.integers(0, d, n).astype(np.int32),
                    "dictionary": rng.uniform(0, 1, d),
                    "out": np.zeros(n),
                },
                {},
                {"n": n, "d": d},
            )
        case "run_end_decode":
            lengths = rng.integers(1, 9, n)
            ends = np.cumsum(lengths)
            ends = ends[ends < n]
            run_ends = np.concatenate((ends, [n])).astype(np.int32)
            return Workload(
                {
                    "run_values": rng.uniform(0, 1, len(run_ends)),
                    "run_ends": run_ends,
                    "out": np.zeros(n),
                },
                {},
                {"r": len(run_ends), "n": n},
            )
        case "string_contains":
            offsets, chars = _string_column(_strings(n, rng))
            pattern = np.frombuffer(PATTERN.encode("utf-8"), dtype=np.uint8).copy()
            return Workload(
                {
                    "offsets": offsets,
                    "chars": chars,
                    "pattern": pattern,
                    "match": np.zeros(n, dtype=np.int8),
                },
                {},
                {"n": n, "chars": len(chars), "k": len(pattern)},
            )
        case "min_max":
            return Workload(
                {"values": rng.uniform(-1e6, 1e6, n), "result": np.array([np.inf, -np.inf])},
                {},
                {"n": n},
            )
        case "rolling_sum":
            return Workload(
                {"values": rng.uniform(0, 1, n), "out": np.zeros(n)}, {"window": 7}, {"n": n}
            )
        case "radix_scatter":
            keys = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
            counts = np.bincount((keys >> np.uint32(8)) & np.uint32(255), minlength=256)
            return Workload(
                {
                    "keys": keys,
                    "cursors": (np.cumsum(counts) - counts).astype(np.int32),
                    "out": np.zeros(n, dtype=np.int32),
                },
                {"shift": 8},
                {"n": n},
            )
        case "days_to_year":
            return Workload(
                {
                    "days": rng.integers(EPOCH_DAYS[0], EPOCH_DAYS[1] + 1, n).astype(np.int32),
                    "year": np.zeros(n, dtype=np.int32),
                },
                {},
                {"n": n},
            )
        case "merge_sorted":
            # values from a small range, so that a and b share some
            return Workload(
                {
                    "a": np.sort(rng.integers(0, n, n)).astype(np.int64),
                    "b": np.sort(rng.integers(0, n, n)).astype(np.int64),
                    "out": np.zeros(2 * n, dtype=np.int64),
                },
                {},
                {"m": n, "p": n},
            )
        case "sorted_contains":
            members = np.unique(rng.integers(0, 4 * n, n)).astype(np.int64)
            keys = rng.integers(0, 4 * n, n).astype(np.int64)
            return Workload(
                {"keys": keys, "members": members, "match": np.zeros(n, dtype=np.int8)},
                {},
                {"n": n, "s": len(members)},
            )
        case "tile_reverse" | "block_sort":
            # the rows form tiles of TILE rows; n need not be a multiple of TILE
            name = "values" if task == "tile_reverse" else "keys"
            return Workload(
                {
                    name: rng.integers(-1000, 1000, n).astype(np.int32),
                    "out": np.zeros(n, dtype=np.int32),
                },
                {},
                {"n": n},
            )
        case "bucket_histogram":
            # bounds sorted ascending with repeats; integer values land on bounds
            bounds = np.sort(rng.integers(0, 1000, BOUNDS)).astype(np.float64)
            values = rng.integers(-10, 1011, n) + 0.5 * (rng.random(n) < 0.5)
            return Workload(
                {
                    "values": values,
                    "bounds": bounds,
                    "counts": np.zeros(BOUNDS + 1, dtype=np.int32),
                },
                {},
                {"n": n, "b": BOUNDS},
            )
        case "string_reverse":
            offsets, chars = _string_column(_strings(n, rng))
            # haiku's signature names the length of chars and out `variable`
            return Workload(
                {"offsets": offsets, "chars": chars, "out": np.zeros(len(chars), dtype=np.uint8)},
                {},
                {"n": n, "chars": len(chars), "variable": len(chars)},
            )
        case "lower_bound":
            # sorted ascending with repeats; keys fall below, on, between, and above its elements
            s = max(1, n)
            return Workload(
                {
                    "keys": rng.integers(-10, s + 10, n).astype(np.int64),
                    "sorted": np.sort(rng.integers(0, s, s)).astype(np.int64),
                    "out": np.zeros(n, dtype=np.int32),
                },
                {},
                {"n": n, "s": s},
            )
        case "masked_sum":
            return Workload(
                {
                    "values": rng.uniform(0, 1, n),
                    "mask": (rng.random(n) < 0.5).astype(np.int8),
                    "result": np.zeros(1),
                },
                {},
                {"n": n},
            )
        case "group_max":
            # the top quarter of the groups has no row, so best stays at its initial value
            if n == 0:
                raise ValueError("group_max needs a row: an empty group buffer cannot be written")
            capacity = max(4, n // 10)
            return Workload(
                {
                    "group": rng.integers(0, capacity - capacity // 4, n).astype(np.int32),
                    "value": rng.integers(_I64_MIN, _I64_MAX, n, endpoint=True),
                    "best": np.full(capacity, _I64_MIN, dtype=np.int64),
                },
                {},
                {"n": n, "capacity": capacity},
            )
        case "run_starts":
            # runs differ in the high 32 bits only
            return Workload(
                {
                    "values": np.cumsum(rng.random(n) < 0.3).astype(np.int64) << 32,
                    "flags": np.zeros(n, dtype=np.int8),
                },
                {},
                {"n": n},
            )
        case _:
            raise ValueError(f"no workload for {task} alone")


def workload_of(
    executor: Executor, tenant: TenantId, model: str, task: str, n: int, rng: np.random.Generator
) -> Workload:
    """Inputs of `n` rows that meet the contract of `task` for `model`'s kernels:
    for `hash_probe`, keys of which half are in a table of `n` keys that the
    model's Numba `hash_build` built."""
    if task != "hash_probe":
        return workload(task, n, rng)
    build = workload("hash_build", n, rng)
    table = run(executor, tenant, corpus_kernel(executor, model, "hash_build", "numba"), build)
    present = rng.choice(build.arrays["keys"], n // 2)
    # the build keys are 3 modulo 7, so these are in no table
    absent = rng.permutation(10 * n)[: n - len(present)].astype(np.int64) * 7 + 4
    return Workload(
        {
            "probe_keys": rng.permutation(np.concatenate([present, absent])),
            "table_keys": table["table_keys"],
            "table_rows": table["table_rows"],
            "out_build": np.zeros(n, dtype=np.int32),
            "out_probe": np.zeros(n, dtype=np.int32),
            "out_count": np.zeros(1, dtype=np.int32),
        },
        {},
        {"m": n, "capacity": len(table["table_keys"]), "out_cap": n},
    )


def _strings_of(w: Workload) -> list[bytes]:
    offsets, chars = w.arrays["offsets"], w.arrays["chars"].tobytes()
    return [chars[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def check(task: str, w: Workload, out: dict[str, np.ndarray]) -> str | None:
    """Why the outputs `out` differ from the NumPy reference, or None if they agree."""
    a = w.arrays
    if task in heldout_workloads.TASKS:
        return heldout_workloads.check(task, a, w.scalars, w.sizes, out)
    match task:
        case "filter_select":
            rows = np.flatnonzero(a["quantity"] < w.scalars["threshold"])
            count = int(out["count"][0])
            same = count == len(rows) and np.array_equal(np.sort(out["sel"][:count]), rows)
        case "scatter_by_offsets":
            same = np.array_equal(out["out"], np.flatnonzero(a["flags"]))
        case "hash_probe":
            slots = a["table_keys"] != -1
            keys, rows = a["table_keys"][slots].tolist(), a["table_rows"][slots].tolist()
            table = dict(zip(keys, rows, strict=True))
            expected = {(table[k], j) for j, k in enumerate(a["probe_keys"].tolist()) if k in table}
            count = int(out["out_count"][0])
            found = out["out_build"][:count].tolist(), out["out_probe"][:count].tolist()
            pairs = set(zip(*found, strict=True))
            same = count == len(expected) and pairs == expected
        case "gather":
            same = np.array_equal(out["out"], a["col"][a["sel"]])
        case "projection":
            same = np.allclose(out["out"], a["price"] * (1 - a["discount"]), rtol=1e-12, atol=0)
        case "q6_sum":
            s = w.scalars
            rows = (
                (s["date_lo"] <= a["shipdate"])
                & (a["shipdate"] < s["date_hi"])
                & (s["disc_lo"] <= a["discount"])
                & (a["discount"] <= s["disc_hi"])
                & (a["quantity"] < s["qty_max"])
            )
            expected = float(np.sum(a["price"][rows] * a["discount"][rows]))
            same = bool(np.isclose(out["result"][0], expected, rtol=1e-9, atol=1e-9))
        case "hash_build":
            slots = out["table_keys"] != -1
            keys, rows = out["table_keys"][slots].tolist(), out["table_rows"][slots].tolist()
            expected = {(int(k), i) for i, k in enumerate(a["keys"])}
            same = len(keys) == len(a["keys"]) and set(zip(keys, rows, strict=True)) == expected
        case "bloom_build":
            same = np.array_equal(out["bits"], _bloom_bits(a["keys"], w.sizes["words"]))
        case "bloom_probe":
            expected = np.ones(w.sizes["n"], dtype=np.int8)
            for position in _bloom_positions(a["keys"], w.sizes["words"]):
                masks = np.uint32(1) << (position % np.uint64(32)).astype(np.uint32)
                expected &= (a["bits"][position // np.uint64(32)] & masks) != 0
            same = np.array_equal(out["match"], expected)
        case "groupby_sum":
            slots = out["table_keys"] != -1
            keys = out["table_keys"][slots]
            expected = np.bincount(a["group"], weights=a["value"])
            same = (
                len(set(keys.tolist())) == len(keys)
                and set(keys.tolist()) == set(np.unique(a["group"]).tolist())
                and np.allclose(out["table_sums"][slots], expected[keys], rtol=1e-9, atol=1e-9)
            )
        case "histogram":
            bins = (a["keys"] >> np.uint32(w.scalars["shift"])) & np.uint32(255)
            same = np.array_equal(out["counts"], np.bincount(bins, minlength=256))
        case "segmented_sum":
            offsets, values = a["offsets"], a["values"]
            expected = np.array(
                [values[offsets[j] : offsets[j + 1]].sum() for j in range(len(offsets) - 1)]
            )
            same = np.allclose(out["sums"], expected, rtol=1e-9, atol=1e-9)
        case "string_equal":
            left, right = a["a_chars"].tobytes(), a["b_chars"].tobytes()
            ao, bo = a["a_offsets"], a["b_offsets"]
            expected = np.array(
                [left[ao[i] : ao[i + 1]] == right[bo[i] : bo[i + 1]] for i in range(w.sizes["n"])],
                dtype=np.int8,
            )
            same = np.array_equal(out["match"], expected)
        case "starts_with":
            prefix = a["prefix"].tobytes()
            expected = np.array([t.startswith(prefix) for t in _strings_of(w)], dtype=np.int8)
            same = np.array_equal(out["match"] != 0, expected != 0)
        case "string_length":
            expected = np.array([len(t.decode("utf-8")) for t in _strings_of(w)], dtype=np.int32)
            same = np.array_equal(out["lengths"], expected)
        case "valid_rows":
            rows = np.flatnonzero(np.unpackbits(a["valid"], bitorder="little")[: w.sizes["n"]])
            count = int(out["count"][0])
            same = count == len(rows) and np.array_equal(np.sort(out["rows"][:count]), rows)
        case "dictionary_decode":
            same = np.array_equal(out["out"], a["dictionary"][a["codes"]])
        case "run_end_decode":
            lengths = np.diff(np.concatenate(([0], a["run_ends"])))
            same = np.array_equal(out["out"], np.repeat(a["run_values"], lengths))
        case "string_contains":
            pattern = a["pattern"].tobytes()
            expected = np.array([pattern in t for t in _strings_of(w)], dtype=np.int8)
            same = np.array_equal(out["match"] != 0, expected != 0)
        case "min_max":
            values = a["values"]
            same = np.array_equal(out["result"], [values.min(), values.max()])
        case "rolling_sum":
            window, values = int(w.scalars["window"]), a["values"]
            expected = np.array(
                [values[max(0, i - window + 1) : i + 1].sum() for i in range(len(values))]
            )
            same = np.allclose(out["out"], expected, rtol=1e-9, atol=1e-9)
        case "radix_scatter":
            bins = (a["keys"] >> np.uint32(w.scalars["shift"])) & np.uint32(255)
            rows = out["out"]
            same = np.array_equal(np.sort(rows), np.arange(len(rows))) and np.array_equal(
                bins[rows], np.sort(bins)
            )
        case "days_to_year":
            dates = np.datetime64("1970-01-01") + a["days"].astype("timedelta64[D]")
            expected = dates.astype("datetime64[Y]").astype(np.int64) + 1970
            same = np.array_equal(out["year"], expected)
        case "merge_sorted":
            same = np.array_equal(out["out"], np.sort(np.concatenate((a["a"], a["b"]))))
        case "sorted_contains":
            same = np.array_equal(out["match"] != 0, np.isin(a["keys"], a["members"]))
        case "tile_reverse":
            expected = a["values"].copy()
            for i in range(0, w.sizes["n"], TILE):
                expected[i : i + TILE] = a["values"][i : i + TILE][::-1]
            same = np.array_equal(out["out"], expected)
        case "block_sort":
            expected = a["keys"].copy()
            for i in range(0, w.sizes["n"], TILE):
                expected[i : i + TILE] = np.sort(a["keys"][i : i + TILE])
            same = np.array_equal(out["out"], expected)
        case "bucket_histogram":
            bins = np.searchsorted(a["bounds"], a["values"], side="right")
            same = np.array_equal(out["counts"], np.bincount(bins, minlength=BOUNDS + 1))
        case "string_reverse":
            expected = b"".join(t[::-1] for t in _strings_of(w))
            same = out["out"].tobytes() == expected
        case "lower_bound":
            same = np.array_equal(out["out"], np.searchsorted(a["sorted"], a["keys"], side="left"))
        case "masked_sum":
            expected = float(np.sum(a["values"][a["mask"] != 0]))
            same = bool(np.isclose(out["result"][0], expected, rtol=1e-9, atol=1e-9))
        case "group_max":
            expected = a["best"].copy()
            np.maximum.at(expected, a["group"], a["value"])
            same = np.array_equal(out["best"], expected)
        case "run_starts":
            expected = np.ones(w.sizes["n"], dtype=np.int8)
            expected[1:] = a["values"][1:] != a["values"][:-1]
            same = np.array_equal(out["flags"], expected)
        case _:
            raise ValueError(f"no reference for {task}")
    return None if same else f"{task}: the outputs differ from the reference"


def encode(dtype: str, value: int | float) -> int:
    """A scalar as the unsigned integer that its bits spell, as the binding bounds it."""
    if dtype in ("float64", "float32"):
        fmt, unsigned = ("<d", "<Q") if dtype == "float64" else ("<f", "<I")
        (bits,) = struct.unpack(unsigned, struct.pack(fmt, value))
        return bits
    return int(value) % 2 ** (8 * ELEMENTS[dtype].width)


@dataclass(frozen=True)
class Prepared:
    """A corpus kernel's launch on a workload: its inputs written, each buffer, and
    each output's buffer, element type, and elements. The launch names a buffer
    for each array argument of the kernel as written; a text with the guard
    pass's checks also takes a status buffer (`opgate.hosts.execute`)."""

    module: Module
    launch: Launch
    buffers: tuple[BufferId, ...]
    outputs: dict[str, tuple[BufferId, np.dtype, int]]


@dataclass(frozen=True)
class Loaded:
    """A kernel of a model, a task, and a toolchain, verified and loaded."""

    model: str
    task: str
    toolchain: str
    module: Module


def corpus_kernel(executor: Executor, model: str, task: str, toolchain: str) -> Loaded:
    """The corpus kernel of `model`, `task`, and `toolchain` from its recorded
    PTX, verified and loaded."""
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    module = executor.register(text, binding_of(model, task, toolchain, text))
    return Loaded(model, task, toolchain, module)


def _grid(toolchain: str, sizes: tuple[int, ...]) -> tuple[int, int, int]:
    """Enough blocks of 256 threads for a warp per row of the largest size, and
    for Triton a program per row, since a Triton kernel may give a row to each
    program; the gate's guarantee holds for every grid."""
    rows = max(sizes)
    match toolchain:
        case "numba" | "nvcc":
            return (max(1, min(GRID_LIMIT[1], -(-32 * rows // 256))), 1, 1)
        case "triton":
            return (max(1, rows), 1, 1)
        case _:
            raise ValueError(f"no toolchain {toolchain}")


def prepare(executor: Executor, tenant: TenantId, kernel: Loaded, w: Workload) -> Prepared:
    """Allocate a buffer for each array argument of `kernel`, and write each
    input (an index buffer with the range of its values, which the catalog
    records as its status); the launch has a value for each size symbol and
    scalar."""
    module = kernel.module
    raws = read_raw(directory_of(kernel.model, kernel.task) / "signature.json")
    roles = read_roles(kernel.model, kernel.task)
    buffers: list[BufferId] = []
    writes: list[Write] = []
    outputs: dict[str, tuple[BufferId, np.dtype, int]] = {}
    with executor.tenant(tenant):
        for raw in raws:
            role = roles.arguments[raw.name]
            if role in (Role.SIZE, Role.SCALAR):
                continue
            array = np.ascontiguousarray(w.arrays[raw.name], dtype=np.dtype(raw.dtype))
            buffer = executor.allocate(max(array.nbytes, 1))
            buffers.append(buffer)
            if isinstance(role, Points):
                writes.append(Write(buffer, array.tobytes(), ELEMENTS[raw.dtype]))
            elif role is not Role.WRITE:
                writes.append(Write(buffer, array.tobytes(), None))
            if role in (Role.WRITE, Role.READ_WRITE) or (
                isinstance(role, Points) and role.pointer is Pointer.INDEX_OUT
            ):
                outputs[raw.name] = (buffer, array.dtype, len(array))
        executor.execute(Step(tenant, tuple(writes)))
    sizes = tuple(w.sizes[name] for name in roles.symbols)
    scalars = tuple(
        encode(raw.dtype, w.scalars[raw.name])
        for raw in raws
        if roles.arguments[raw.name] is Role.SCALAR
    )
    grid = _grid(kernel.toolchain, sizes)
    launch = Launch(module.operator, grid, tuple(buffers), sizes, scalars, False)
    return Prepared(module, launch, tuple(buffers), outputs)


def run(executor: Executor, tenant: TenantId, kernel: Loaded, w: Workload) -> dict[str, np.ndarray]:
    """Launch `kernel` on `w` through the executor (`opgate.hosts.execute`, which
    raises `CheckFailedError` if a check of the guard pass failed), and read
    back each output."""
    with executor.tenant(tenant):
        prepared = prepare(executor, tenant, kernel, w)
        try:
            execute(executor, executor, Step(tenant, (prepared.launch,)))
            return {
                name: np.frombuffer(executor.read(tenant, buffer), dtype=dtype, count=count)
                for name, (buffer, dtype, count) in prepared.outputs.items()
            }
        finally:
            for buffer in prepared.buffers:
                executor.free(buffer)
