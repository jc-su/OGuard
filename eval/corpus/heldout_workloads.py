"""Inputs and independent CPU references for the fourth and fifth task lists.

These are execution workloads for already published task contracts. Adding them
extends measured execution coverage; it does not create a new blind test set.
"""

import calendar
import datetime
import re

import numpy as np

TASKS = frozenset(
    {
        "stream_compact",
        "string_prefix",
        "join_count",
        "null_count",
        "prefix_offsets",
        "transpose",
        "hash_partition",
        "month_start",
        "window_max",
        "pair_sum",
        "between_bitmap",
        "gather_strings",
        "group_sum_valid",
        "asof_join",
        "decimal_rescale",
        "string_hash",
        "bitmap_slice",
        "join_materialize",
        "add_months",
        "argmin",
        "run_end_encode",
        "segmented_cummax",
        "string_concat",
        "hash_distinct",
        "range_window_sum",
        "bit_unpack",
        "group_argmax",
        "fill_forward",
        "timestamp_floor",
        "string_to_int",
    }
)
_EPOCH = datetime.date(1970, 1, 1)
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1


def _column(strings):
    offsets = np.asarray([0, *np.cumsum([len(s) for s in strings])], dtype=np.int32)
    return offsets, np.frombuffer(b"".join(strings), dtype=np.uint8).copy()


def _bits(array, n):
    return np.unpackbits(array, bitorder="little")[:n].astype(bool)


def _strings(arrays, prefix=""):
    offsets, chars = arrays[prefix + "offsets"], arrays[prefix + "chars"].tobytes()
    return [chars[int(lo) : int(hi)] for lo, hi in zip(offsets[:-1], offsets[1:], strict=True)]


def inputs(task: str, n: int, rng: np.random.Generator) -> tuple[dict, dict, dict]:
    if task not in TASKS or n < 0:
        raise ValueError(f"unsupported task or size: {task}, {n}")
    a: dict[str, np.ndarray] = {}
    s: dict[str, int] = {}
    sizes: dict[str, int] = {"n": n}
    values = rng.integers(-100, 101, n, dtype=np.int64)
    valid = np.packbits(rng.random(n) < 0.6, bitorder="little")
    g = min(1024, max(1, n // 4))
    group = rng.integers(0, g, n, dtype=np.int32)
    match task:
        case "stream_compact":
            a = dict(
                values=values,
                keep=(rng.random(n) < 0.6).astype(np.int8),
                out=np.zeros(n, np.int64),
                count=np.zeros(1, np.int32),
            )
        case "null_count":
            a = dict(valid=valid, count=np.zeros(1, np.int64))
        case "between_bitmap":
            a = dict(values=values, valid=valid, out=np.zeros(len(valid), np.uint8))
            s = dict(lo=-25, hi=25)
        case "group_sum_valid":
            a = dict(
                group=group,
                value=values,
                valid=valid,
                counts=np.zeros(g, np.int64),
                sums=np.zeros(g, np.int64),
            )
            sizes["g"] = g
        case "fill_forward":
            a = dict(values=values, valid=valid, out=np.zeros(n, np.int64))
        case "bitmap_slice":
            s = dict(offset=5)
            a = dict(
                source=np.packbits(rng.random(n + 5) < 0.6, bitorder="little"),
                out=np.zeros((n + 7) // 8, np.uint8),
            )
        case "prefix_offsets":
            a = dict(
                lengths=rng.integers(0, 8, n, dtype=np.int32), offsets=np.zeros(n + 1, np.int32)
            )
        case "transpose":
            cols, rows = 7, (n + 6) // 7
            a = dict(
                source=rng.integers(-100, 101, rows * cols, dtype=np.int32),
                out=np.zeros(rows * cols, np.int32),
            )
            sizes |= dict(rows=rows, cols=cols)
        case "hash_partition":
            a = dict(keys=values, part=np.zeros(n, np.int32), counts=np.zeros(g, np.int32))
            s["p"] = g
            sizes["p"] = g
        case "pair_sum":
            length = max(1, n)
            a = dict(
                left_idx=rng.integers(0, length, n, dtype=np.int32),
                right_idx=rng.integers(0, length, n, dtype=np.int32),
                left=rng.normal(size=length),
                right=rng.normal(size=length),
                out=np.zeros(n, np.float64),
            )
            sizes |= dict(m=n, l=length, r=length)
        case "window_max":
            a = dict(values=values, out=np.zeros(n, np.int64))
            s["w"] = 7
        case "argmin":
            a = dict(values=values, row=np.full(1, -1, np.int32))
        case "group_argmax":
            a = dict(group=group, value=values, best=np.full(g, -1, np.int32))
            sizes["g"] = g
        case "segmented_cummax":
            first = np.maximum.accumulate(np.where(rng.random(n) < 0.2, np.arange(n), 0)).astype(
                np.int32
            )
            a = dict(values=values, first=first, out=np.zeros(n, np.int64))
        case "run_end_encode":
            a = dict(
                values=np.repeat(values[::4], 4)[:n],
                run_ends=np.full(n, -7, np.int32),
                run_values=np.full(n, -11, np.int64),
                count=np.zeros(1, np.int32),
            )
        case "join_count" | "join_materialize":
            domain = max(201, 2 * n)
            build = np.sort(rng.integers(0, domain, n, dtype=np.int64))
            probe = rng.integers(0, domain, n, dtype=np.int64)
            low, high = (
                np.searchsorted(build, probe, side="left"),
                np.searchsorted(build, probe, side="right"),
            )
            sizes |= dict(b=n, p=n)
            a = dict(build=build, probe=probe)
            if task == "join_count":
                a["counts"] = np.zeros(n, np.int32)
            else:
                offsets = np.asarray([0, *np.cumsum(high - low)], dtype=np.int32)
                a |= dict(
                    offsets=offsets,
                    out_build=np.zeros(int(offsets[-1]), np.int32),
                    out_probe=np.zeros(int(offsets[-1]), np.int32),
                )
        case "asof_join":
            a = dict(left=values, right=np.sort(values[::2]), out=np.zeros(n, np.int32))
            sizes["r"] = len(a["right"])
            s["tolerance"] = 3
        case "decimal_rescale":
            edge = [_I64_MIN, _I64_MAX, -15, -5, 5, 15]
            values[: min(n, len(edge))] = edge[:n]
            a = dict(values=values, out=np.zeros(n, np.int64), overflow=np.zeros(n, np.int8))
            s["shift"] = 3
        case "timestamp_floor":
            a = dict(ts=values * 1_000_001, out=np.zeros(n, np.int64))
            s["unit"] = 1_000_000
        case "range_window_sum":
            a = dict(
                ts=np.cumsum(rng.integers(0, 3, n, dtype=np.int64)),
                values=values,
                out=np.zeros(n, np.int64),
            )
            s["range"] = 8
        case "month_start" | "add_months":
            a = dict(days=rng.integers(-10000, 20000, n, dtype=np.int32), out=np.zeros(n, np.int32))
            if task == "add_months":
                a["months"] = rng.integers(-24, 25, n, dtype=np.int32)
        case "bit_unpack":
            bits = 7
            words = (n * bits + 31) // 32
            a = dict(
                packed=rng.integers(0, 2**32, words, dtype=np.uint32), out=np.zeros(n, np.uint32)
            )
            sizes["w"] = words
            s["bits"] = bits
        case "hash_distinct":
            capacity = 1 << max(0, (max(1, 2 * n) - 1).bit_length())
            keys = np.where(values == -1, 0, values)
            a = dict(
                keys=keys,
                table_keys=np.full(capacity, -1, np.int64),
                table_rows=np.full(capacity, 2**31 - 1, np.int32),
            )
            sizes["capacity"] = capacity
        case "string_prefix" | "string_hash" | "string_to_int" | "gather_strings" | "string_concat":
            samples = [b"", b"a", b"ab", "é😀".encode(), b"12345678"]
            if task == "string_to_int":
                samples = [
                    b"",
                    b"0",
                    b"-0",
                    b"+0012",
                    b"-9223372036854775808",
                    b"9223372036854775807",
                    b"9223372036854775808",
                    b"-9223372036854775809",
                    b"12x",
                    b"+",
                    b" 1",
                    b"-0005",
                ]
            strings = [samples[int(i)] for i in rng.integers(0, len(samples), n)]
            offsets, chars = _column(strings)
            a = dict(offsets=offsets, chars=chars)
            if task == "string_prefix":
                s["k"] = sizes["k"] = 5
                a["out"] = np.zeros(n * 5, np.uint8)
            elif task == "string_hash":
                a["out"] = np.zeros(n, np.uint32)
            elif task == "string_to_int":
                a |= dict(out=np.zeros(n, np.int64), valid=np.zeros((n + 7) // 8, np.uint8))
            elif task == "gather_strings":
                indices = rng.integers(0, max(1, n), n, dtype=np.int32)
                out_offsets, out_chars = _column([strings[int(i)] for i in indices])
                a |= dict(
                    idx=indices,
                    out_offsets=out_offsets,
                    out_chars=np.zeros(len(out_chars), np.uint8),
                )
                sizes["m"] = n
            else:
                other = list(reversed(strings))
                b_offsets, b_chars = _column(other)
                out_offsets = offsets + b_offsets
                a = dict(
                    a_offsets=offsets,
                    a_chars=chars,
                    b_offsets=b_offsets,
                    b_chars=b_chars,
                    out_offsets=out_offsets,
                    out_chars=np.zeros(len(chars) + len(b_chars), np.uint8),
                )
        case _:
            raise AssertionError(task)
    # Independent extent symbols in the reviewed contracts are buffer lengths.
    sizes |= {name: len(array) for name, array in a.items() if name not in sizes}
    return a, s, sizes


def reference(task: str, a: dict, s: dict, sizes: dict) -> dict[str, np.ndarray]:
    n = sizes["n"]
    match task:
        case "stream_compact":
            selected = a["values"][a["keep"] != 0]
            return dict(out=selected, count=np.asarray([len(selected)], np.int32))
        case "null_count":
            return dict(count=np.asarray([n - np.count_nonzero(_bits(a["valid"], n))], np.int64))
        case "between_bitmap":
            selected = _bits(a["valid"], n) & (a["values"] >= s["lo"]) & (a["values"] <= s["hi"])
            return dict(out=np.packbits(selected, bitorder="little"))
        case "group_sum_valid":
            live = _bits(a["valid"], n)
            counts, sums = np.zeros(sizes["g"], np.int64), np.zeros(sizes["g"], np.int64)
            np.add.at(counts, a["group"][live], 1)
            np.add.at(sums, a["group"][live], a["value"][live])
            return dict(counts=counts, sums=sums)
        case "fill_forward":
            indices = np.maximum.accumulate(np.where(_bits(a["valid"], n), np.arange(n) + 1, 0))
            return dict(out=np.concatenate((np.zeros(1, np.int64), a["values"]))[indices])
        case "bitmap_slice":
            return dict(
                out=np.packbits(
                    _bits(a["source"], s["offset"] + n)[s["offset"] :], bitorder="little"
                )
            )
        case "prefix_offsets":
            return dict(offsets=np.asarray([0, *np.cumsum(a["lengths"], dtype=np.int32)], np.int32))
        case "transpose":
            return dict(out=a["source"].reshape(sizes["rows"], sizes["cols"]).T.copy().ravel())
        case "hash_partition":
            hashed = a["keys"].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
            part = ((hashed >> np.uint64(32)) % np.uint64(sizes["p"])).astype(np.int32)
            return dict(part=part, counts=np.bincount(part, minlength=sizes["p"]).astype(np.int32))
        case "pair_sum":
            return dict(out=a["left"][a["left_idx"]] + a["right"][a["right_idx"]])
        case "window_max":
            return dict(
                out=np.asarray(
                    [max(a["values"][max(0, i - s["w"] + 1) : i + 1]) for i in range(n)], np.int64
                )
            )
        case "argmin":
            return dict(row=np.asarray([int(np.argmin(a["values"])) if n else -1], np.int32))
        case "group_argmax":
            best = np.full(sizes["g"], -1, np.int32)
            for i, (group, value) in enumerate(zip(a["group"], a["value"], strict=True)):
                if best[group] == -1 or value > a["value"][best[group]]:
                    best[group] = i
            return dict(best=best)
        case "segmented_cummax":
            out = np.empty(n, np.int64)
            for i in range(n):
                out[i] = a["values"][i] if a["first"][i] == i else max(out[i - 1], a["values"][i])
            return dict(out=out)
        case "run_end_encode":
            ends = (
                np.flatnonzero(np.r_[a["values"][:-1] != a["values"][1:], True]) + 1
                if n
                else np.empty(0, np.int32)
            )
            run_ends, run_values = a["run_ends"].copy(), a["run_values"].copy()
            run_ends[: len(ends)], run_values[: len(ends)] = ends, a["values"][ends - 1]
            return dict(
                run_ends=run_ends, run_values=run_values, count=np.asarray([len(ends)], np.int32)
            )
        case "join_count" | "join_materialize":
            low, high = (
                np.searchsorted(a["build"], a["probe"], side="left"),
                np.searchsorted(a["build"], a["probe"], side="right"),
            )
            if task == "join_count":
                return dict(counts=(high - low).astype(np.int32))
            build = (
                np.concatenate(
                    [np.arange(lo, hi, dtype=np.int32) for lo, hi in zip(low, high, strict=True)]
                )
                if len(low)
                else np.empty(0, np.int32)
            )
            return dict(
                out_build=build,
                out_probe=np.repeat(np.arange(len(low), dtype=np.int32), high - low),
            )
        case "asof_join":
            rows = np.searchsorted(a["right"], a["left"], side="right") - 1
            if len(a["right"]):
                rows = np.where(
                    (rows >= 0) & (a["left"] - a["right"][np.maximum(rows, 0)] <= s["tolerance"]),
                    rows,
                    -1,
                )
            return dict(out=rows.astype(np.int32))
        case "decimal_rescale":
            values, overflow = [], []
            shift = s["shift"]
            for value in a["values"].tolist():
                if shift >= 0:
                    result = value * 10**shift
                else:
                    divisor = 10 ** (-shift)
                    result = (abs(value) + divisor // 2) // divisor * (-1 if value < 0 else 1)
                failed = not _I64_MIN <= result <= _I64_MAX
                values.append(0 if failed else result)
                overflow.append(failed)
            return dict(out=np.asarray(values, np.int64), overflow=np.asarray(overflow, np.int8))
        case "timestamp_floor":
            return dict(out=(a["ts"] // s["unit"]) * s["unit"])
        case "range_window_sum":
            lo = np.searchsorted(a["ts"], a["ts"] - s["range"], side="left")
            hi = np.searchsorted(a["ts"], a["ts"], side="right")
            prefix = np.r_[np.int64(0), np.cumsum(a["values"], dtype=np.int64)]
            return dict(out=prefix[hi] - prefix[lo])
        case "month_start" | "add_months":
            out = []
            for i, days in enumerate(a["days"].tolist()):
                date = _EPOCH + datetime.timedelta(days=days)
                if task == "month_start":
                    result = date.replace(day=1)
                else:
                    month = date.year * 12 + date.month - 1 + int(a["months"][i])
                    year, month = divmod(month, 12)
                    result = datetime.date(
                        year, month + 1, min(date.day, calendar.monthrange(year, month + 1)[1])
                    )
                out.append((result - _EPOCH).days)
            return dict(out=np.asarray(out, np.int32))
        case "bit_unpack":
            bits = np.unpackbits(a["packed"].astype("<u4").view(np.uint8), bitorder="little")[
                : n * s["bits"]
            ]
            weights = np.left_shift(np.uint64(1), np.arange(s["bits"], dtype=np.uint64))
            return dict(out=(bits.reshape(n, s["bits"]) @ weights).astype(np.uint32))
        case "string_prefix":
            return dict(
                out=np.frombuffer(
                    b"".join(text[: s["k"]].ljust(s["k"], b"\0") for text in _strings(a)), np.uint8
                )
            )
        case "string_hash":
            values = []
            for text in _strings(a):
                value = 2166136261
                for byte in text:
                    value = ((value ^ byte) * 16777619) % 2**32
                values.append(value)
            return dict(out=np.asarray(values, np.uint32))
        case "string_to_int":
            values, live = [], []
            for text in _strings(a):
                digits = text.lstrip(b"+-").lstrip(b"0") or b"0"
                parsed = (
                    int((b"-" if text.startswith(b"-") else b"") + digits)
                    if len(digits) <= 19 and re.fullmatch(rb"[+-]?[0-9]+", text)
                    else None
                )
                ok = parsed is not None and _I64_MIN <= parsed <= _I64_MAX
                values.append(parsed if ok else 0)
                live.append(ok)
            return dict(
                out=np.asarray(values, np.int64), valid=np.packbits(live, bitorder="little")
            )
        case "gather_strings":
            strings = _strings(a)
            return dict(
                out_chars=np.frombuffer(b"".join(strings[int(i)] for i in a["idx"]), np.uint8)
            )
        case "string_concat":
            return dict(
                out_chars=np.frombuffer(
                    b"".join(
                        left + right
                        for left, right in zip(_strings(a, "a_"), _strings(a, "b_"), strict=True)
                    ),
                    np.uint8,
                )
            )
        case _:
            raise ValueError(f"{task}: no deterministic reference layout")


def check(task: str, arrays: dict, scalars: dict, sizes: dict, outputs: dict) -> str | None:
    if task == "hash_distinct":
        keys, rows = outputs["table_keys"], outputs["table_rows"]
        expected = {}
        for i, key in enumerate(arrays["keys"].tolist()):
            expected.setdefault(key, i)
        occupied = keys != -1
        if (
            len(keys) != sizes["capacity"]
            or len(rows) != len(keys)
            or np.count_nonzero(occupied) != len(expected)
        ):
            return "hash table extent or distinct count differs"
        actual = dict(zip(keys[occupied].tolist(), rows[occupied].tolist(), strict=True))
        if actual != expected or np.any(rows[~occupied] != 2**31 - 1):
            return "hash table keys, first rows or empty slots differ"
        for slot in np.flatnonzero(occupied):
            start = (((int(keys[slot]) % 2**64) * 0x9E3779B97F4A7C15 % 2**64) >> 32) & (
                len(keys) - 1
            )
            while start != slot:
                if keys[start] == -1:
                    return "hash table probe chain crosses an empty slot"
                start = (start + 1) & (len(keys) - 1)
        return None
    expected = reference(task, arrays, scalars, sizes)
    if task == "stream_compact":
        count = int(outputs["count"][0])
        same = count == len(expected["out"]) and np.array_equal(
            np.sort(outputs["out"][:count]), np.sort(expected["out"])
        )
        return None if same else "compaction values or count differ"
    for name, values in expected.items():
        if (
            name not in outputs
            or outputs[name].dtype != values.dtype
            or not np.array_equal(outputs[name], values)
        ):
            return f"{task}.{name}: differs from CPU reference"
    return None
